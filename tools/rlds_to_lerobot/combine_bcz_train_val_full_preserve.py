#!/usr/bin/env python3
"""Combine BC-Z train and val full-preserve TraceX components.

The BC-Z release component is train + val with a continuous episode id space:
train uses 0..39349 and val uses 39350..43263. This script creates a lightweight
combined v2-style source tree for the v3 repacker by symlinking episode parquet
and video files and merging metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


DEFAULT_TRAIN = Path("${WORK_ROOT}/tracex/bcz_train_0.1.0_lerobot")
DEFAULT_VAL = Path("${WORK_ROOT}/tracex/bcz_val_0.1.0_lerobot")
DEFAULT_OUT = Path("${WORK_ROOT}/tracex/bcz_train_val_0.1.0_lerobot")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _symlink_tree_files(src_root: Path, dst_root: Path, pattern: str) -> int:
    count = 0
    for src in sorted(src_root.glob(pattern)):
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            raise FileExistsError(dst)
        os.symlink(src, dst)
        count += 1
    return count


def _merge_stat_payload(a: dict[str, Any], b: dict[str, Any], n_a: int, n_b: int) -> dict[str, Any]:
    if not all(key in a and key in b for key in ["mean", "std", "max", "min"]):
        return a
    mean_a = a["mean"]
    mean_b = b["mean"]
    std_a = a["std"]
    std_b = b["std"]
    if not isinstance(mean_a, list) or not isinstance(mean_b, list):
        return a
    if len(mean_a) != len(mean_b) or len(std_a) != len(std_b):
        return a
    if any(isinstance(x, list) for x in mean_a + mean_b + std_a + std_b):
        return a
    total = float(n_a + n_b)
    mean = [(float(x) * n_a + float(y) * n_b) / total for x, y in zip(mean_a, mean_b, strict=True)]
    var = []
    for idx, merged_mean in enumerate(mean):
        va = float(std_a[idx]) ** 2
        vb = float(std_b[idx]) ** 2
        ma = float(mean_a[idx])
        mb = float(mean_b[idx])
        var.append((n_a * (va + (ma - merged_mean) ** 2) + n_b * (vb + (mb - merged_mean) ** 2)) / total)
    return {
        "mean": mean,
        "std": [x**0.5 for x in var],
        "max": [max(float(x), float(y)) for x, y in zip(a["max"], b["max"], strict=True)],
        "min": [min(float(x), float(y)) for x, y in zip(a["min"], b["min"], strict=True)],
    }


def _merge_stats(train_root: Path, val_root: Path, train_frames: int, val_frames: int) -> dict[str, Any]:
    train_stats = _read_json(train_root / "meta/stats.json")
    val_stats = _read_json(val_root / "meta/stats.json")
    out: dict[str, Any] = {}
    for key in sorted(set(train_stats) | set(val_stats)):
        if key in train_stats and key in val_stats:
            out[key] = _merge_stat_payload(train_stats[key], val_stats[key], train_frames, val_frames)
        elif key in train_stats:
            out[key] = train_stats[key]
        else:
            out[key] = val_stats[key]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-root", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.out_root.exists() and args.overwrite:
        shutil.rmtree(args.out_root)
    if args.out_root.exists() and any(args.out_root.iterdir()):
        raise FileExistsError(f"{args.out_root} already exists; pass --overwrite to rebuild")
    for root in [args.train_root, args.val_root]:
        for rel in ["meta/info.json", "meta/episodes.jsonl", "meta/tasks.jsonl", "meta/stats.json"]:
            if not (root / rel).exists():
                raise FileNotFoundError(root / rel)

    train_info = _read_json(args.train_root / "meta/info.json")
    val_info = _read_json(args.val_root / "meta/info.json")
    if train_info["features"] != val_info["features"]:
        raise RuntimeError("BC-Z train/val feature schemas differ")

    args.out_root.mkdir(parents=True, exist_ok=True)
    data_files = _symlink_tree_files(args.train_root, args.out_root, "data/chunk-*/*.parquet")
    data_files += _symlink_tree_files(args.val_root, args.out_root, "data/chunk-*/*.parquet")
    video_files = _symlink_tree_files(args.train_root, args.out_root, "videos/chunk-*/*/*.mp4")
    video_files += _symlink_tree_files(args.val_root, args.out_root, "videos/chunk-*/*/*.mp4")

    episodes = _read_jsonl(args.train_root / "meta/episodes.jsonl") + _read_jsonl(args.val_root / "meta/episodes.jsonl")
    tasks = _read_jsonl(args.train_root / "meta/tasks.jsonl") + _read_jsonl(args.val_root / "meta/tasks.jsonl")
    episodes.sort(key=lambda row: int(row["episode_index"]))
    tasks.sort(key=lambda row: int(row["task_index"]))
    episode_ids = [int(row["episode_index"]) for row in episodes]
    if episode_ids != list(range(min(episode_ids), max(episode_ids) + 1)):
        raise RuntimeError("combined BC-Z episode ids are not contiguous")

    total_frames = int(train_info["total_frames"]) + int(val_info["total_frames"])
    info = dict(train_info)
    info.update(
        {
            "total_episodes": len(episodes),
            "total_frames": total_frames,
            "total_tasks": len(tasks),
            "total_videos": int(train_info["total_videos"]) + int(val_info["total_videos"]),
            "total_chunks": max(1, max(episode_ids) // int(train_info["chunks_size"]) + 1),
            "splits": {"train": f"{min(episode_ids)}:{max(episode_ids) + 1}"},
            "component_note": "BC-Z train + val combined for SemanticVLA TraceX release.",
        }
    )
    _write_jsonl(args.out_root / "meta/episodes.jsonl", episodes)
    _write_jsonl(args.out_root / "meta/tasks.jsonl", tasks)
    _write_json(args.out_root / "meta/info.json", info)
    _write_json(args.out_root / "meta/stats.json", _merge_stats(args.train_root, args.val_root, int(train_info["total_frames"]), int(val_info["total_frames"])))
    for name in ["modality.json", "rlds_features.json"]:
        src = args.train_root / "meta" / name
        if src.exists():
            shutil.copy2(src, args.out_root / "meta" / name)
    _write_json(
        args.out_root / "meta/conversion_summary.json",
        {
            "component": "bcz_train_val",
            "train_root": str(args.train_root),
            "val_root": str(args.val_root),
            "out_root": str(args.out_root),
            "episodes": len(episodes),
            "frames": total_frames,
            "data_files": data_files,
            "video_files": video_files,
            "symlinked": True,
        },
    )
    print(json.dumps({"out_root": str(args.out_root), "episodes": len(episodes), "frames": total_frames, "data_files": data_files, "video_files": video_files}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
