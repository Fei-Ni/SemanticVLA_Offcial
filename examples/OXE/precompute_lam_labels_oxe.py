#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from examples.SemanticVLA_OXE.oxe_lam_dataset import OxeLAMDataset, oxe_lam_collate_fn
from semanticvla.model.modules.latent_action_model import TraceLatentActionModel


OXE_TO_LEROBOT_DATASET = {
    "bridge": "bridge_orig_1.0.0_lerobot",
    "fractal": "fractal20220817_data_0.1.0_lerobot",
    "bcz": "bcz_0.1.0_lerobot",
}


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute OXE LAM labels for SemanticVLA LeRobot training.")
    parser.add_argument("--config", default="examples/SemanticVLA_OXE/configs/oxe_lam_core.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", required=True, choices=["v0", "v1", "v2", "v3", "v4", "v5", "two_stage", "paper_strict"])
    parser.add_argument("--output-root", default="${WORK_ROOT}/oxe_lam_labels")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--eval-stride", type=int, default=0)
    parser.add_argument("--max-samples-per-dataset", type=int, default=None)
    parser.add_argument("--max-episodes-per-dataset", type=int, default=None)
    parser.add_argument("--image-resolution", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    data_cfg = dict(cfg["data"])
    model_cfg = cfg["model"]
    datasets = args.datasets or data_cfg.get("datasets", ["bridge"])
    data_cfg["datasets"] = datasets
    data_cfg["split"] = "all"
    data_cfg["sample_stride"] = int(args.sample_stride)
    data_cfg["eval_stride"] = int(args.eval_stride)
    data_cfg["max_samples_per_dataset"] = args.max_samples_per_dataset
    data_cfg["max_episodes_per_dataset"] = args.max_episodes_per_dataset
    if args.image_resolution is not None:
        data_cfg["image_resolution"] = int(args.image_resolution)

    for key in ("batch_size", "num_workers", "eval_split", "eval_max_samples_per_dataset"):
        data_cfg.pop(key, None)

    dataset = OxeLAMDataset(**data_cfg)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        collate_fn=oxe_lam_collate_fn,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TraceLatentActionModel.from_config(model_cfg, args.variant)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    out_dir = Path(args.output_root) / Path(args.checkpoint).parent.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = {
        dataset_name: open(out_dir / f"{OXE_TO_LEROBOT_DATASET[dataset_name]}.jsonl", "w")
        for dataset_name in datasets
    }
    counts = Counter()
    token_counts = Counter()
    total_tokens = 0

    try:
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"precompute OXE {args.variant}", total=len(loader)):
                batch_dev = to_device(batch, device)
                outputs = model.vq_encode(batch_dev)
                indices = outputs["indices"][:, 0].detach().cpu().long()
                for i, idx in enumerate(indices):
                    source_name = str(batch["dataset_name"][i])
                    lerobot_name = OXE_TO_LEROBOT_DATASET[source_name]
                    idx_list = [int(x) for x in idx.tolist()]
                    row = {
                        "dataset_name": lerobot_name,
                        "source_dataset": source_name,
                        "episode_index": int(batch["episode_index"][i].item()),
                        "step_index": int(batch["step_index"][i].item()),
                        "indices": idx_list,
                    }
                    fps[source_name].write(json.dumps(row, separators=(",", ":")) + "\n")
                    counts[lerobot_name] += 1
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
        "datasets": list(datasets),
        "num_labels": int(sum(counts.values())),
        "counts_by_dataset": dict(counts),
        "sample_stride": int(args.sample_stride),
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
