#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))

from examples.LIBERO.semanticvla.lam_libero_dataset import LiberoLAMDataset, lam_collate_fn
from semanticvla.model.modules.latent_action_model import TraceLatentActionModel


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_cfg(path: str) -> dict[str, Any]:
    with open(path) as fp:
        return yaml.safe_load(fp)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


@torch.no_grad()
def evaluate(
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
        with torch.no_grad():
            outputs = model(batch)
            _, metrics = model.compute_loss(outputs)
        bs = int(batch["videos"].shape[0])
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value.item()) * bs
        count += bs
        indices = outputs["indices"][:, 0].detach().cpu()
        all_indices.append(indices)
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
    model.train()
    return result


def code_distribution_metrics(indices: torch.Tensor, num_latents: int, prefix: str) -> dict[str, float]:
    flat = indices.reshape(-1)
    counts = torch.bincount(flat, minlength=num_latents).float()
    used = counts > 0
    probs = counts / counts.sum().clamp_min(1.0)
    entropy = -(probs[used] * torch.log(probs[used].clamp_min(1e-12))).sum()
    return {
        f"{prefix}/global_code_usage": float(used.float().mean().item()),
        f"{prefix}/global_code_entropy": float(entropy.item()),
        f"{prefix}/global_code_perplexity": float(torch.exp(entropy).item()),
    }


def action_probe_metrics(indices: torch.Tensor, actions: torch.Tensor, num_latents: int, prefix: str) -> dict[str, float]:
    n = indices.shape[0]
    if n < 16:
        return {f"{prefix}/action_probe_mse": float("nan"), f"{prefix}/action_probe_r2": float("nan")}
    x = F.one_hot(indices.long(), num_classes=num_latents).float().reshape(n, -1)
    x = torch.cat([x, torch.ones(n, 1)], dim=1)
    y = actions.float()
    split = max(8, int(n * 0.8))
    if split >= n:
        split = n - 1
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]
    reg = 1e-3 * torch.eye(x_train.shape[1])
    reg[-1, -1] = 0.0
    try:
        w = torch.linalg.solve(x_train.T @ x_train + reg, x_train.T @ y_train)
        pred = x_val @ w
        mse = F.mse_loss(pred, y_val).item()
        base = F.mse_loss(y_train.mean(dim=0, keepdim=True).expand_as(y_val), y_val).item()
        r2 = 1.0 - mse / max(base, 1e-12)
    except RuntimeError:
        mse, r2 = float("nan"), float("nan")
    return {f"{prefix}/action_probe_mse": float(mse), f"{prefix}/action_probe_r2": float(r2)}


def token_switch_rate(indices: torch.Tensor, meta: list[tuple[str, int, int]]) -> float:
    switches = 0.0
    total = 0
    for i in range(1, len(meta)):
        prev = meta[i - 1]
        cur = meta[i]
        if prev[0] == cur[0] and prev[1] == cur[1] and cur[2] == prev[2] + 1:
            switches += float((indices[i] != indices[i - 1]).float().sum().item())
            total += int(indices.shape[1])
    return float(switches / total) if total else float("nan")


def save_checkpoint(model, optimizer, step: int, output_dir: Path) -> Path:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"steps_{step}_pytorch_model.pt"
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="examples/LIBERO/semanticvla/configs/m8_lam_core.yaml")
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
    parser.add_argument("--max-samples-per-suite", type=int, default=None)
    parser.add_argument("--eval-max-samples-per-suite", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--mock-dino", action="store_true")
    parser.add_argument("--init-from", default=None, help="Initialize model weights from a LAM checkpoint; optimizer is fresh.")
    parser.add_argument(
        "--start-step",
        type=int,
        default=None,
        help=(
            "Schedule step used for the first optimizer update. This controls "
            "two-stage warmup/freeze logic but checkpoint names remain local finetune steps."
        ),
    )
    parser.add_argument(
        "--finetune-stage2-only",
        action="store_true",
        help="For two-stage LAMs, start the schedule after stage1_warmup_steps so only stage2 loss is active.",
    )
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    if args.max_steps is not None:
        cfg["trainer"]["max_steps"] = args.max_steps
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.max_samples_per_suite is not None:
        cfg["data"]["max_samples_per_suite"] = args.max_samples_per_suite
    if args.eval_max_samples_per_suite is not None:
        cfg["data"]["eval_max_samples_per_suite"] = args.eval_max_samples_per_suite
    if args.learning_rate is not None:
        cfg["trainer"]["learning_rate"] = args.learning_rate
    if args.eval_every is not None:
        cfg["trainer"]["eval_every"] = args.eval_every
    if args.eval_batches is not None:
        cfg["trainer"]["eval_batches"] = args.eval_batches
    if args.save_every is not None:
        cfg["trainer"]["save_every"] = args.save_every
    if args.mock_dino:
        cfg["model"]["mock_dino"] = True

    run_id = args.run_id or f"lam_{args.variant}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"
    output_root = Path(args.output_root or cfg["trainer"]["output_root"])
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w") as fp:
        yaml.safe_dump(cfg, fp, sort_keys=False)
    with open(output_dir / "variant.txt", "w") as fp:
        fp.write(f"{args.variant}\n")

    set_seed(int(cfg["trainer"].get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = dict(cfg["data"])
    batch_size = int(data_cfg.pop("batch_size"))
    num_workers = int(data_cfg.pop("num_workers"))
    eval_split = str(data_cfg.pop("eval_split", "eval"))
    eval_max_samples_per_suite = data_cfg.pop(
        "eval_max_samples_per_suite",
        data_cfg.get("max_samples_per_suite"),
    )
    train_data_cfg = dict(data_cfg)
    eval_data_cfg = dict(data_cfg)
    eval_data_cfg["max_samples_per_suite"] = eval_max_samples_per_suite
    train_dataset = LiberoLAMDataset(split="train", **train_data_cfg)
    eval_dataset = LiberoLAMDataset(split=eval_split, **eval_data_cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=lam_collate_fn,
        drop_last=True,
    )
    eval_num_workers = 0 if num_workers == 0 else max(1, min(num_workers, 4))
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=eval_num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=eval_num_workers > 0,
        collate_fn=lam_collate_fn,
        drop_last=False,
    )

    model = TraceLatentActionModel.from_config(cfg["model"], args.variant)
    init_info: dict[str, Any] = {}
    if args.init_from:
        init_path = Path(args.init_from)
        ckpt = torch.load(init_path, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=True)
        init_info = {
            "init_from": str(init_path),
            "init_missing_keys": len(missing),
            "init_unexpected_keys": len(unexpected),
        }
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["trainer"]["learning_rate"]),
        weight_decay=float(cfg["trainer"]["weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    metrics_path = output_dir / "metrics.jsonl"
    max_steps = int(cfg["trainer"]["max_steps"])
    log_every = int(cfg["trainer"]["log_every"])
    eval_every = int(cfg["trainer"]["eval_every"])
    save_every = int(cfg["trainer"]["save_every"])
    grad_clip = float(cfg["trainer"].get("grad_clip", 1.0))
    amp_dtype = torch.bfloat16 if str(cfg["trainer"].get("precision", "bf16")) == "bf16" else torch.float16
    use_amp = torch.cuda.is_available()
    schedule_start_step = int(args.start_step) if args.start_step is not None else 1
    if args.finetune_stage2_only:
        warmup_steps = int(getattr(model, "stage1_warmup_steps", cfg["model"].get("stage1_warmup_steps", 0)))
        schedule_start_step = max(schedule_start_step, warmup_steps + 1)

    print(json.dumps({
        "run_id": run_id,
        "variant": args.variant,
        "output_dir": str(output_dir),
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "max_steps": max_steps,
        "schedule_start_step": schedule_start_step,
        "finetune_stage2_only": bool(args.finetune_stage2_only),
        "batch_size": batch_size,
        "device": str(device),
        **init_info,
    }, indent=2))

    train_iter = iter(train_loader)
    model.train()
    pbar = tqdm(range(1, max_steps + 1), desc=run_id)
    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        schedule_step = schedule_start_step + step - 1
        if hasattr(model, "set_train_step"):
            model.set_train_step(schedule_step)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            outputs = model(batch)
            loss, metrics = model.compute_loss(outputs)
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % log_every == 0 or step == 1:
            row = {"step": step, "phase": "train"}
            row.update({f"train/{k}": float(v.item()) for k, v in metrics.items()})
            with open(metrics_path, "a") as fp:
                fp.write(json.dumps(row) + "\n")
            pbar.set_postfix({k: f"{float(v.item()):.4f}" for k, v in metrics.items() if k in {"loss", "recon_loss", "code_usage"}})

        if eval_every > 0 and step % eval_every == 0:
            eval_metrics = evaluate(model, eval_loader, device=device, max_batches=int(cfg["trainer"]["eval_batches"]))
            row = {"step": step, "phase": "eval"}
            row.update(eval_metrics)
            with open(metrics_path, "a") as fp:
                fp.write(json.dumps(row) + "\n")

        if save_every > 0 and step % save_every == 0:
            save_checkpoint(model, optimizer, step, output_dir)

    save_checkpoint(model, optimizer, max_steps, output_dir)
    final_metrics = evaluate(model, eval_loader, device=device, max_batches=int(cfg["trainer"]["eval_batches"]))
    summary = {"run_id": run_id, "variant": args.variant, "step": max_steps, **final_metrics}
    with open(output_dir / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
