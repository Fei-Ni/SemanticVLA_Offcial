#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))

from examples.LIBERO.semanticvla.train_lam import (
    action_probe_metrics,
    code_distribution_metrics,
    load_cfg,
    save_checkpoint,
    set_seed,
    to_device,
    token_switch_rate,
)
from examples.SemanticVLA_OXE.oxe_lam_dataset import OxeLAMDataset, oxe_lam_collate_fn
from semanticvla.model.modules.latent_action_model import TraceLatentActionModel


def init_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed OXE LAM training requires CUDA.")
        cuda_count = torch.cuda.device_count()
        if local_rank >= cuda_count:
            raise RuntimeError(
                "Distributed OXE LAM local rank is outside visible CUDA devices: "
                f"LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}, "
                f"torch.cuda.device_count()={cuda_count}, "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}."
            )
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return distributed, rank, local_rank, world_size


def barrier(distributed: bool) -> None:
    if distributed:
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def save_oxe_checkpoint(model, optimizer, scheduler, step: int, output_dir: Path) -> Path:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"steps_{step}_pytorch_model.pt"
    model_to_save = unwrap_model(model)
    payload = {
        "step": step,
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)
    return path


def align_cosine_scheduler_after_resume(
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR | None,
    optimizer: torch.optim.Optimizer,
    *,
    completed_step: int,
    target_max_steps: int,
) -> dict[str, Any]:
    if scheduler is None or not isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
        return {}
    old_t_max = int(getattr(scheduler, "T_max", target_max_steps))
    if old_t_max == target_max_steps:
        return {
            "resume_scheduler_t_max": old_t_max,
            "resume_scheduler_t_max_adjusted": False,
        }

    scheduler.T_max = target_max_steps
    scheduler.last_epoch = completed_step
    clamped_step = min(max(completed_step, 0), target_max_steps)
    lrs = []
    for base_lr, group in zip(scheduler.base_lrs, optimizer.param_groups):
        lr = scheduler.eta_min + (base_lr - scheduler.eta_min) * (
            1.0 + math.cos(math.pi * clamped_step / target_max_steps)
        ) / 2.0
        group["lr"] = lr
        lrs.append(lr)
    scheduler._last_lr = lrs
    return {
        "resume_scheduler_t_max": old_t_max,
        "resume_scheduler_target_t_max": target_max_steps,
        "resume_scheduler_t_max_adjusted": True,
        "resume_scheduler_lr_after_adjust": lrs,
    }


@torch.no_grad()
def evaluate_oxe(
    model: TraceLatentActionModel,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    all_indices = []
    all_actions = []
    meta = []
    for bidx, batch in enumerate(loader):
        if bidx >= max_batches:
            break
        batch = to_device(batch, device)
        outputs = model(batch)
        _, metrics = model.compute_loss(outputs)
        bs = int(batch["videos"].shape[0])
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value.item()) * bs
        count += bs
        all_indices.append(outputs["indices"][:, 0].detach().cpu())
        all_actions.append(batch["future_actions"].detach().cpu().reshape(bs, -1).float())
        for name, ep, step in zip(batch["dataset_name"], batch["episode_index"].cpu().tolist(), batch["step_index"].cpu().tolist()):
            meta.append((name, int(ep), int(step)))

    if count == 0:
        return {}

    result = {f"eval/{key}": value / count for key, value in totals.items()}
    indices_t = torch.cat(all_indices, dim=0)
    actions_t = torch.cat(all_actions, dim=0)
    result.update(code_distribution_metrics(indices_t, model.num_latents, prefix="eval"))
    result.update(action_probe_metrics(indices_t, actions_t, model.num_latents, prefix="eval"))
    result["eval/token_switch_rate"] = token_switch_rate(indices_t, meta)

    per_dataset_r2 = []
    per_dataset_mse = []
    names = sorted({m[0] for m in meta})
    for name in names:
        mask = torch.tensor([m[0] == name for m in meta], dtype=torch.bool)
        if int(mask.sum()) < 16:
            continue
        ds_indices = indices_t[mask]
        ds_actions = actions_t[mask]
        mean = ds_actions.mean(dim=0, keepdim=True)
        std = ds_actions.std(dim=0, keepdim=True).clamp_min(1e-6)
        ds_actions_z = (ds_actions - mean) / std
        ds_metrics = action_probe_metrics(ds_indices, ds_actions_z, model.num_latents, prefix=f"eval/{name}")
        result.update(ds_metrics)
        mse = ds_metrics.get(f"eval/{name}/action_probe_mse", float("nan"))
        r2 = ds_metrics.get(f"eval/{name}/action_probe_r2", float("nan"))
        if math.isfinite(mse):
            per_dataset_mse.append(mse)
        if math.isfinite(r2):
            per_dataset_r2.append(r2)
    result["eval/action_probe_mse_macro"] = float(sum(per_dataset_mse) / len(per_dataset_mse)) if per_dataset_mse else float("nan")
    result["eval/action_probe_r2_macro"] = float(sum(per_dataset_r2) / len(per_dataset_r2)) if per_dataset_r2 else float("nan")
    model.train()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="examples/SemanticVLA_OXE/configs/oxe_lam_core.yaml")
    parser.add_argument(
        "--variant",
        required=True,
        choices=["paper_strict"],
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--max-samples-per-dataset", type=int, default=None)
    parser.add_argument("--eval-max-samples-per-dataset", type=int, default=None)
    parser.add_argument("--max-episodes-per-dataset", type=int, default=None)
    parser.add_argument("--sample-stride", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--lr-scheduler", choices=["none", "cosine"], default=None)
    parser.add_argument("--min-learning-rate", type=float, default=None)
    parser.add_argument("--mock-dino", action="store_true")
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()

    distributed, rank, local_rank, world_size = init_distributed()
    is_main = rank == 0

    cfg = load_cfg(args.config)
    if args.max_steps is not None:
        cfg["trainer"]["max_steps"] = args.max_steps
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.datasets is not None:
        cfg["data"]["datasets"] = list(args.datasets)
    if args.max_samples_per_dataset is not None:
        cfg["data"]["max_samples_per_dataset"] = args.max_samples_per_dataset
    if args.eval_max_samples_per_dataset is not None:
        cfg["data"]["eval_max_samples_per_dataset"] = args.eval_max_samples_per_dataset
    if args.max_episodes_per_dataset is not None:
        cfg["data"]["max_episodes_per_dataset"] = args.max_episodes_per_dataset
    if args.sample_stride is not None:
        cfg["data"]["sample_stride"] = args.sample_stride
    if args.eval_every is not None:
        cfg["trainer"]["eval_every"] = args.eval_every
    if args.eval_batches is not None:
        cfg["trainer"]["eval_batches"] = args.eval_batches
    if args.save_every is not None:
        cfg["trainer"]["save_every"] = args.save_every
    if args.grad_accum_steps is not None:
        cfg["trainer"]["grad_accum_steps"] = args.grad_accum_steps
    if args.lr_scheduler is not None:
        cfg["trainer"]["lr_scheduler"] = args.lr_scheduler
    if args.min_learning_rate is not None:
        cfg["trainer"]["min_learning_rate"] = args.min_learning_rate
    if args.mock_dino:
        cfg["model"]["mock_dino"] = True

    run_id = args.run_id or f"OXE_lam_{args.variant}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"
    output_root = Path(args.output_root or cfg["trainer"]["output_root"])
    output_dir = output_root / run_id
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.yaml", "w") as fp:
            yaml.safe_dump(cfg, fp, sort_keys=False)
        with open(output_dir / "variant.txt", "w") as fp:
            fp.write(f"{args.variant}\n")
    barrier(distributed)

    set_seed(int(cfg["trainer"].get("seed", 42)))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    data_cfg: dict[str, Any] = dict(cfg["data"])
    batch_size = int(data_cfg.pop("batch_size"))
    num_workers = int(data_cfg.pop("num_workers"))
    eval_split = str(data_cfg.pop("eval_split", "eval"))
    eval_max_samples = data_cfg.pop("eval_max_samples_per_dataset", data_cfg.get("max_samples_per_dataset"))

    train_data_cfg = dict(data_cfg)
    eval_data_cfg = dict(data_cfg)
    eval_data_cfg["max_samples_per_dataset"] = eval_max_samples
    train_dataset = OxeLAMDataset(split="train", **train_data_cfg)
    eval_dataset = OxeLAMDataset(split=eval_split, **eval_data_cfg)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    ) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=oxe_lam_collate_fn,
        drop_last=True,
    )
    eval_num_workers = 0 if num_workers == 0 else max(1, min(num_workers, 2))
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=eval_num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=eval_num_workers > 0,
        collate_fn=oxe_lam_collate_fn,
        drop_last=False,
    )

    lam_model = TraceLatentActionModel.from_config(cfg["model"], args.variant).to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        lam_model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    ) if distributed else lam_model
    optimizer = torch.optim.AdamW(
        [p for p in lam_model.parameters() if p.requires_grad],
        lr=float(cfg["trainer"]["learning_rate"]),
        weight_decay=float(cfg["trainer"]["weight_decay"]),
    )

    max_steps = int(cfg["trainer"]["max_steps"])
    log_every = int(cfg["trainer"]["log_every"])
    eval_every = int(cfg["trainer"]["eval_every"])
    save_every = int(cfg["trainer"]["save_every"])
    grad_clip = float(cfg["trainer"].get("grad_clip", 1.0))
    grad_accum_steps = max(1, int(cfg["trainer"].get("grad_accum_steps", 1)))
    amp_dtype = torch.bfloat16 if str(cfg["trainer"].get("precision", "bf16")) == "bf16" else torch.float16
    use_amp = torch.cuda.is_available()

    scheduler = None
    lr_scheduler = str(cfg["trainer"].get("lr_scheduler", "none"))
    if lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_steps,
            eta_min=float(cfg["trainer"].get("min_learning_rate", 0.0)),
        )
    elif lr_scheduler != "none":
        raise ValueError(f"Unsupported lr_scheduler: {lr_scheduler}")

    start_step = 1
    resume_info: dict[str, Any] = {}
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu")
        lam_model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        completed_step = int(ckpt.get("step", 0))
        resume_info = align_cosine_scheduler_after_resume(
            scheduler,
            optimizer,
            completed_step=completed_step,
            target_max_steps=max_steps,
        )
        resume_info["resume_from"] = str(args.resume_from)
        resume_info["resume_completed_step"] = completed_step
        start_step = completed_step + 1

    scaler = torch.cuda.amp.GradScaler(enabled=False)
    metrics_path = output_dir / "metrics.jsonl"

    if is_main:
        print(json.dumps({
            "run_id": run_id,
            "variant": args.variant,
            "output_dir": str(output_dir),
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "max_steps": max_steps,
            "start_step": start_step,
            "batch_size_per_rank": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "world_size": world_size,
            "effective_batch_size": batch_size * grad_accum_steps * world_size,
            "lr_scheduler": lr_scheduler,
            "device": str(device),
            "datasets": list(cfg["data"].get("datasets", [])),
            **resume_info,
        }, indent=2))

    train_iter = iter(train_loader)
    model.train()
    sampler_epoch = 0
    pbar = tqdm(range(start_step, max_steps + 1), desc=run_id, disable=not is_main)
    for step in pbar:
        optimizer.zero_grad(set_to_none=True)
        metric_sums: dict[str, float] = {}
        for _ in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                sampler_epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(sampler_epoch)
                train_iter = iter(train_loader)
                batch = next(train_iter)
            batch = to_device(batch, device)
            if hasattr(lam_model, "set_train_step"):
                lam_model.set_train_step(step)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(batch)
                loss, metrics = lam_model.compute_loss(outputs)
                scaled_loss = loss / grad_accum_steps
            scaler.scale(scaled_loss).backward()
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value.item())
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        metrics_avg = {key: value / grad_accum_steps for key, value in metric_sums.items()}

        if is_main and (step % log_every == 0 or step == start_step):
            row = {"step": step, "phase": "train"}
            row.update({f"train/{k}": float(v) for k, v in metrics_avg.items()})
            row["train/lr"] = float(optimizer.param_groups[0]["lr"])
            with open(metrics_path, "a") as fp:
                fp.write(json.dumps(row) + "\n")
            pbar.set_postfix({k: f"{float(v):.4f}" for k, v in metrics_avg.items() if k in {"loss", "recon_loss", "code_usage"}})

        if eval_every > 0 and step % eval_every == 0:
            barrier(distributed)
            if is_main:
                eval_metrics = evaluate_oxe(lam_model, eval_loader, device=device, max_batches=int(cfg["trainer"]["eval_batches"]))
                row = {"step": step, "phase": "eval"}
                row.update(eval_metrics)
                with open(metrics_path, "a") as fp:
                    fp.write(json.dumps(row) + "\n")
            barrier(distributed)
            model.train()

        if is_main and save_every > 0 and step % save_every == 0:
            save_oxe_checkpoint(model, optimizer, scheduler, step, output_dir)
        barrier(distributed)

    if is_main:
        save_oxe_checkpoint(model, optimizer, scheduler, max_steps, output_dir)
        final_metrics = evaluate_oxe(lam_model, eval_loader, device=device, max_batches=int(cfg["trainer"]["eval_batches"]))
        summary = {"run_id": run_id, "variant": args.variant, "step": max_steps, **final_metrics}
        with open(output_dir / "summary.json", "w") as fp:
            json.dump(summary, fp, indent=2)
        print(json.dumps(summary, indent=2))
    barrier(distributed)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
