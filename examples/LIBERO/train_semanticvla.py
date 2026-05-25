"""
SemanticVLA training entry-point.

Identical to train.py except _train_step also logs sub-losses
(pure_action_loss, lm_trace_loss) when the model returns them.

Convention used throughout SemanticVLA: the primary backprop loss is
from `output_dict["action_loss"]` (in SemanticVLA this is the weighted
total L_action + λ·L_lm_trace). Any other 0-dim tensor in the returned
dict is automatically picked up by the metrics logger and surfaced to
wandb / tqdm.
"""

import argparse
import json
import os
import re
import warnings
from pathlib import Path
from typing import Tuple
from torch.utils.data import DataLoader
import numpy as np
import time

import torch
import torch.distributed as dist
import wandb
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import GradientAccumulationPlugin, set_seed
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# SemanticVLA imports ─ paths are resolved relative to this file
import sys, pathlib
_here = pathlib.Path(__file__).resolve().parent
_repo_root = _here.parent.parent.parent   # semanticvla/
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root.parent))  # home root (for deployment etc.)

from semanticvla.training.trainer_utils.trainer_tools import normalize_dotlist_args
from semanticvla.model.framework import build_framework
from semanticvla.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups
from semanticvla.dataloader import build_dataloader

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = get_logger(__name__)


def _get_cfg_value(cfg, dotted_key, default=None):
    """Read a nested OmegaConf value without requiring the path to exist."""
    try:
        value = OmegaConf.select(cfg, dotted_key)
    except Exception:
        return default
    return default if value is None else value


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes", "on"}
    return bool(value)


def _infer_step_from_checkpoint(path: str) -> int:
    match = re.search(r"steps_(\d+)", str(path))
    if match:
        return int(match.group(1))
    return 0


def build_accelerator(cfg):
    grad_acc_steps = int(cfg.trainer.gradient_accumulation_steps)
    deepspeed_plugin = DeepSpeedPlugin(gradient_accumulation_steps=grad_acc_steps)
    grad_acc_plugin = GradientAccumulationPlugin(
        num_steps=grad_acc_steps,
        sync_each_batch=True,
    )
    accelerator = Accelerator(
        deepspeed_plugin=deepspeed_plugin,
        gradient_accumulation_plugin=grad_acc_plugin,
    )
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", accelerator.local_process_index))
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
    accelerator.print(
        f"[build_accelerator] honoring merged config gradient_accumulation_steps={grad_acc_steps}"
    )
    accelerator.print(accelerator.state)
    return accelerator


def synchronize_processes(accelerator: Accelerator | None = None) -> None:
    """Synchronize all ranks without relying on NCCL's implicit device inference."""
    if accelerator is not None:
        if dist.is_available() and dist.is_initialized() and torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", accelerator.local_process_index))
            torch.cuda.set_device(local_rank % torch.cuda.device_count())
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            accelerator.wait_for_everyone()
        return
    if dist.is_available() and dist.is_initialized():
        barrier_kwargs = {}
        if torch.cuda.is_available():
            barrier_kwargs["device_ids"] = [torch.cuda.current_device()]
        dist.barrier(**barrier_kwargs)


# ---------------------------------------------------------------------------
# Helpers (inlined to avoid importing train at module level, which
# would trigger a second Accelerator creation and raise NotImplementedError)
# ---------------------------------------------------------------------------

def setup_directories(cfg) -> Path:
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)
    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)
        OmegaConf.save(cfg, output_dir / "config.yaml")
        with open(output_dir / "config.yaml", "r") as f_yaml, \
             open(output_dir / "config.json", "w") as f_json:
            import yaml as _yaml
            json.dump(_yaml.safe_load(f_yaml), f_json, indent=2)
    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    accelerator.dataloader_config.dispatch_batches = False
    synchronize_processes(accelerator)
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple:
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )
    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )
    return optimizer, lr_scheduler


# ---------------------------------------------------------------------------
# SemanticVLATrainer
# ---------------------------------------------------------------------------

class SemanticVLATrainer(TrainerUtils):
    """
    Thin wrapper around the standard training loop.
    The only meaningful difference: _train_step logs sub-losses
    (_action_dit_loss, _progress_loss) when the model provides them.
    """

    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.completed_steps = 0
        self.total_batch_size = (
            cfg.datasets.vla_data.per_device_batch_size
            * accelerator.num_processes
            * accelerator.gradient_accumulation_steps
        )

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        resume_checkpoint = _get_cfg_value(
            self.config,
            "trainer.resume_from_checkpoint",
            _get_cfg_value(self.config, "resume_from_checkpoint", None),
        )
        resume_step = _get_cfg_value(self.config, "trainer.resume_step", None)
        if resume_checkpoint:
            reload_strict = _as_bool(getattr(self.config.trainer, "resume_reload_strict", True))
            self.model = self.load_pretrained_backbones(
                self.model,
                resume_checkpoint,
                reload_modules=None,
                strict=reload_strict,
            )
            if resume_step is None or str(resume_step) == "":
                resume_step = _infer_step_from_checkpoint(resume_checkpoint)
            self.completed_steps = int(resume_step)
            logger.info(
                f"Model-only resume from {resume_checkpoint}; "
                f"continuing at completed_steps={self.completed_steps}"
            )
        elif hasattr(self.config.trainer, "pretrained_checkpoint") and self.config.trainer.pretrained_checkpoint:
            reload_strict = _as_bool(getattr(self.config.trainer, "reload_strict", True))
            self.model = self.load_pretrained_backbones(
                self.model,
                self.config.trainer.pretrained_checkpoint,
                reload_modules=getattr(self.config.trainer, "reload_modules", None),
                strict=reload_strict,
            )
        if hasattr(self.model, "ensure_semantic_output_tokens"):
            self.model.ensure_semantic_output_tokens()

        freeze_modules = getattr(self.config.trainer, "freeze_modules", None)
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)
        # Rebuild optimizer/scheduler after checkpoint loading and any module freezes
        # so DeepSpeed only sees the final trainable parameter set.
        self.optimizer, self.lr_scheduler = setup_optimizer_and_scheduler(
            model=self.model,
            cfg=self.config,
        )
        if self.completed_steps > 0:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`",
                )
                for _ in range(self.completed_steps):
                    self.lr_scheduler.step()
            logger.info(f"Advanced LR scheduler by {self.completed_steps} resume steps.")

        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator, self.model, self.optimizer, self.vla_train_dataloader,
        )

        self._init_wandb()
        self._init_checkpointing()

    def _init_wandb(self):
        if self.accelerator.is_main_process:
            try:
                wandb_group = getattr(self.config, "wandb_group", "vla-train")
                wandb.init(
                    name=self.config.run_id,
                    dir=os.path.join(self.config.output_dir, "wandb"),
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    group=wandb_group,
                )
            except Exception as e:
                logger.warning(f"WandB init failed ({e}); continuing without WandB.")

    def _init_checkpointing(self):
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _save_checkpoint(self):
        if self.accelerator.is_main_process:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps({"steps": self.completed_steps}) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")
        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if dist.get_rank() == 0:
                metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
                metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
                if wandb.run is not None:
                    wandb.log(metrics, step=self.completed_steps)
                logger.info(f"Step {self.completed_steps}, Loss: {metrics}")

    def _create_data_iterators(self):
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        try:
            return next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            return next(self.vla_iter)

    def _train_step(self, batch_vla):
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(
                    batch_vla,
                    global_step=self.completed_steps,
                    max_train_steps=self.config.trainer.max_train_steps,
                )
                action_loss = output_dict["action_loss"]
                total_loss  = action_loss

            self.accelerator.backward(total_loss)

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.config.trainer.gradient_clipping
                )

            self.optimizer.step()
            self.lr_scheduler.step()

        metrics = {"action_dit_loss": total_loss.item()}
        for key, value in output_dict.items():
            if key == "action_loss":
                continue
            if isinstance(value, torch.Tensor) and value.ndim == 0:
                metrics[key.lstrip("_")] = value.item()
            elif isinstance(value, (float, int)):
                metrics[key.lstrip("_")] = float(value)

        return metrics

    def train(self):
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimisation steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size    = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation    = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size         = {self.total_batch_size}")

        self._create_data_iterators()

        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps),
            initial=self.completed_steps,
            total=self.config.trainer.max_train_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t0_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t1_data = time.perf_counter()

            t0_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t1_model = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix({
                    "data_t": f"{t1_data - t0_data:.3f}",
                    "model_t": f"{t1_model - t0_model:.3f}",
                })

            step_metrics["data_time"]  = t1_data  - t0_data
            step_metrics["model_time"] = t1_model - t0_model
            self._log_metrics(step_metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        # Final save
        skip_final_save = _as_bool(getattr(self.config.trainer, "skip_final_save", False))
        if self.accelerator.is_main_process and not skip_final_save:
            final_dir = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_dir, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, os.path.join(final_dir, "pytorch_model.pt"))
            logger.info(f"Training complete. Final model saved at {final_dir}")
        elif self.accelerator.is_main_process:
            logger.info("Training complete. Final model save skipped by trainer.skip_final_save.")
        if self.accelerator.is_main_process:
            if wandb.run is not None:
                wandb.finish()

        self.accelerator.wait_for_everyone()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg):
    logger.info("SemanticVLA Training :: Warming Up")
    accelerator = build_accelerator(cfg)

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)

    trainer = SemanticVLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=None,
        lr_scheduler=None,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    synchronize_processes(accelerator)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str,
                        default="semanticvla/config/training/semanticvla_libero.yaml")
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
