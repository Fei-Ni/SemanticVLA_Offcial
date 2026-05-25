#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from examples.LIBERO.semanticvla.lam_libero_dataset import (
    LIBERO_DATASETS,
    LiberoLAMDataset,
    lam_collate_fn,
)
from semanticvla.model.modules.latent_action_model import TraceLatentActionModel


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="examples/LIBERO/semanticvla/configs/m8_lam_core.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--variant",
        required=True,
        choices=["paper_strict"],
    )
    parser.add_argument("--output-root", default="${WORK_ROOT}/lam_labels")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples-per-suite", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LiberoLAMDataset(
        data_root=data_cfg["data_root"],
        trace_root=data_cfg["trace_root"],
        suites=data_cfg.get("suites", ["spatial", "object", "goal", "10"]),
        split="all",
        eval_stride=int(data_cfg.get("eval_stride", 10)),
        window_size=int(data_cfg.get("window_size", 12)),
        action_horizon=int(data_cfg.get("action_horizon", 8)),
        image_resolution=int(data_cfg.get("image_resolution", 224)),
        video_key=data_cfg.get("video_key", "observation.images.image"),
        video_backend=data_cfg.get("video_backend", "torchvision_av"),
        max_samples_per_suite=args.max_samples_per_suite,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lam_collate_fn,
    )

    model = TraceLatentActionModel.from_config(model_cfg, variant=args.variant)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    out_dir = Path(args.output_root) / Path(args.checkpoint).parent.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)

    name_to_suite = {v: k for k, v in LIBERO_DATASETS.items()}
    fps = {suite: open(out_dir / f"libero_{suite}.jsonl", "w") for suite in name_to_suite.values()}
    counts = Counter()
    token_counts = Counter()
    total_tokens = 0

    try:
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"precompute {args.variant}", total=len(loader)):
                batch_dev = to_device(batch, device)
                outputs = model.vq_encode(batch_dev)
                indices = outputs["indices"][:, 0].detach().cpu().long()
                for i, idx in enumerate(indices):
                    dataset_name = batch["dataset_name"][i]
                    suite = name_to_suite[dataset_name]
                    idx_list = [int(x) for x in idx.tolist()]
                    row = {
                        "dataset_name": dataset_name,
                        "suite": suite,
                        "episode_index": int(batch["episode_index"][i].item()),
                        "step_index": int(batch["step_index"][i].item()),
                        "indices": idx_list,
                    }
                    fps[suite].write(json.dumps(row, separators=(",", ":")) + "\n")
                    counts[suite] += 1
                    token_counts.update(idx_list)
                    total_tokens += len(idx_list)
    finally:
        for fp in fps.values():
            fp.close()

    used = len(token_counts)
    probs = torch.tensor([c / max(total_tokens, 1) for c in token_counts.values()], dtype=torch.float64)
    entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum().item()) if len(probs) else 0.0
    manifest = {
        "variant": args.variant,
        "checkpoint": args.checkpoint,
        "output_dir": str(out_dir),
        "num_labels": int(sum(counts.values())),
        "counts_by_suite": dict(counts),
        "num_action_tokens": int(model.num_action_tokens),
        "num_latents": int(model.num_latents),
        "global_code_usage": float(used / model.num_latents),
        "global_code_entropy": entropy,
        "global_code_perplexity": float(torch.exp(torch.tensor(entropy)).item()),
    }
    with open(out_dir / "manifest.json", "w") as fp:
        json.dump(manifest, fp, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
