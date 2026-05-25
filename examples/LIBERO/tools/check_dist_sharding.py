#!/usr/bin/env python3
"""Smoke-only distributed dataloader sharding check.

This script intentionally stays outside the training path. It builds the same
SemanticVLA dataloader config, lets Accelerate prepare it, then verifies that
the prepared batch sampler assigns disjoint map-style dataset indices to ranks.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
from accelerate import Accelerator
from omegaconf import OmegaConf

from semanticvla.dataloader import build_dataloader
from semanticvla.training.trainer_utils.trainer_tools import normalize_dotlist_args


def _flatten_batch_indices(batch) -> list[int]:
    if isinstance(batch, torch.Tensor):
        return [int(x) for x in batch.flatten().tolist()]
    if isinstance(batch, (list, tuple)):
        out: list[int] = []
        for item in batch:
            out.extend(_flatten_batch_indices(item))
        return out
    return [int(batch)]


def _take_indices(batch_sampler: Iterable, num_batches: int) -> list[int]:
    indices: list[int] = []
    for batch_idx, batch in enumerate(batch_sampler):
        if batch_idx >= num_batches:
            break
        indices.extend(_flatten_batch_indices(batch))
    return indices


def _sample_triplets(dataloader, max_batches: int = 1, max_items: int = 8) -> list[dict]:
    triplets: list[dict] = []
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        for item in batch[:max_items]:
            if not isinstance(item, dict):
                continue
            triplets.append(
                {
                    "dataset_name": item.get("dataset_name"),
                    "trajectory_id": item.get("trajectory_id"),
                    "step": item.get("step"),
                }
            )
    return triplets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_batches", type=int, default=100)
    parser.add_argument("--sample_triplets", action="store_true")
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    cli_cfg = OmegaConf.from_dotlist(normalize_dotlist_args(clipargs))
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg.output_dir = args.output_dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator()
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", accelerator.local_process_index))
        torch.cuda.set_device(local_rank % torch.cuda.device_count())

    dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    prepared = accelerator.prepare(dataloader)

    batch_sampler = getattr(prepared, "batch_sampler", None)
    if batch_sampler is None:
        raise RuntimeError("Prepared dataloader has no batch_sampler; cannot verify map-style sharding.")

    indices = _take_indices(batch_sampler, args.num_batches)
    triplets = _sample_triplets(prepared) if args.sample_triplets else []

    rank = accelerator.process_index
    world = accelerator.num_processes
    out_path = Path(args.output_dir) / f"rank_{rank:02d}.json"
    out_path.write_text(
        json.dumps(
            {
                "rank": rank,
                "world": world,
                "num_indices": len(indices),
                "indices": indices,
                "sample_triplets": triplets,
            },
            indent=2,
            sort_keys=True,
        )
    )

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        rank_files = sorted(Path(args.output_dir).glob("rank_*.json"))
        records = [json.loads(path.read_text()) for path in rank_files]
        if len(records) != world:
            raise RuntimeError(f"Expected {world} rank records, found {len(records)} in {args.output_dir}")

        intersections: list[dict] = []
        for i, left in enumerate(records):
            left_set = set(left["indices"])
            for right in records[i + 1 :]:
                overlap = sorted(left_set.intersection(right["indices"]))
                if overlap:
                    intersections.append(
                        {
                            "rank_a": left["rank"],
                            "rank_b": right["rank"],
                            "count": len(overlap),
                            "first": overlap[:20],
                        }
                    )

        summary = {
            "world": world,
            "num_batches": args.num_batches,
            "per_rank_num_indices": {str(r["rank"]): r["num_indices"] for r in records},
            "intersection_count": len(intersections),
            "intersections": intersections[:20],
            "status": "pass" if not intersections else "fail",
        }
        summary_path = Path(args.output_dir) / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        if intersections:
            raise SystemExit(2)

    accelerator.wait_for_everyone()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
