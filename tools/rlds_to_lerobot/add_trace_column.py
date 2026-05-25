#!/usr/bin/env python3
"""Copy a LeRobot dataset while adding one per-frame trace column.

The release format stores dense trace supervision directly in each episode
parquet next to `observation.state` and `action`.

Default field:
  observation.trace.xy: fixed_size_list<float32>[2]

Coordinates are copied from a mmap NPY trace index and remain in normalized
image-space [0, 100]. Downstream loaders can divide by 100 when they need [0, 1].
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--dst-root", type=Path, required=True)
    parser.add_argument("--trace-index-root", type=Path, required=True)
    parser.add_argument("--trace-key", required=True, help="Prefix for *_coords.npy and *_offsets.npy")
    parser.add_argument("--trace-field", default="observation.trace.xy")
    parser.add_argument(
        "--trace-episode-offset",
        type=int,
        default=0,
        help="Global episode id that maps to trace slot 0. Use 39350 for BC-Z val-only indexes.",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--video-mode",
        choices=("symlink", "copy", "none"),
        default="symlink",
        help="How to populate videos in dst-root. symlink is best for local staging.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
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


def _fixed_size_trace_array(coords: np.ndarray) -> pa.FixedSizeListArray:
    arr = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
    flat = pa.array(arr.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, 2)


def _relative_episode_path(pattern: str, episode_index: int, chunk_size: int) -> Path:
    episode_chunk = int(episode_index) // int(chunk_size)
    return Path(
        pattern.format(
            episode_index=int(episode_index),
            episode_chunk=episode_chunk,
            chunk_index=episode_chunk,
            file_index=int(episode_index),
        )
    )


def _copy_or_link(src: Path, dst: Path, mode: str) -> None:
    if mode == "none" or not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    rel_src = os.path.relpath(src, start=dst.parent)
    os.symlink(rel_src, dst)


def _episode_video_paths(info: dict[str, Any], episode_index: int, chunk_size: int) -> list[tuple[Path, Path]]:
    video_path_pattern = info.get("video_path")
    if not video_path_pattern:
        return []
    features = info.get("features", {})
    video_keys = [key for key, spec in features.items() if spec.get("dtype") == "video"]
    out: list[tuple[Path, Path]] = []
    episode_chunk = int(episode_index) // int(chunk_size)
    for video_key in video_keys:
        rel = Path(
            video_path_pattern.format(
                episode_index=int(episode_index),
                episode_chunk=episode_chunk,
                chunk_index=episode_chunk,
                file_index=int(episode_index),
                video_key=video_key,
            )
        )
        out.append((rel, rel))
    return out


def _process_one(
    *,
    src_root: Path,
    dst_root: Path,
    data_pattern: str,
    chunk_size: int,
    trace_field: str,
    trace_episode_offset: int,
    coords_mmap: np.ndarray,
    offsets: np.ndarray,
    episode: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    episode_index = int(episode["episode_index"])
    slot = episode_index - int(trace_episode_offset)
    if slot < 0 or slot + 1 >= len(offsets):
        raise IndexError(f"episode {episode_index} maps to missing trace slot {slot}")

    src_rel = _relative_episode_path(data_pattern, episode_index, chunk_size)
    src_path = src_root / src_rel
    dst_path = dst_root / src_rel
    if dst_path.exists() and not overwrite:
        table = pq.read_table(dst_path, columns=[trace_field])
        return {"episode_index": episode_index, "rows": table.num_rows, "skipped": True}

    table = pq.read_table(src_path)
    start, end = int(offsets[slot]), int(offsets[slot + 1])
    coords = coords_mmap[start:end]
    if table.num_rows != int(coords.shape[0]):
        raise ValueError(
            f"episode {episode_index}: parquet rows {table.num_rows} != trace rows {coords.shape[0]}"
        )

    if trace_field in table.column_names:
        col_idx = table.column_names.index(trace_field)
        table = table.set_column(col_idx, trace_field, _fixed_size_trace_array(coords))
    else:
        table = table.append_column(trace_field, _fixed_size_trace_array(coords))

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
    pq.write_table(table, tmp_path)
    os.replace(tmp_path, dst_path)
    return {
        "episode_index": episode_index,
        "rows": table.num_rows,
        "skipped": False,
        "trace_min": np.asarray(coords).min(axis=0).astype(float).tolist(),
        "trace_max": np.asarray(coords).max(axis=0).astype(float).tolist(),
        "trace_sum": np.asarray(coords, dtype=np.float64).sum(axis=0).tolist(),
        "trace_sumsq": np.square(np.asarray(coords, dtype=np.float64)).sum(axis=0).tolist(),
    }


def _trace_stats(results: list[dict[str, Any]]) -> dict[str, list[float]]:
    count = sum(int(r["rows"]) for r in results if not r.get("skipped"))
    if count <= 0:
        return {"mean": [0.0, 0.0], "std": [0.0, 0.0], "min": [0.0, 0.0], "max": [0.0, 0.0]}
    total = np.zeros(2, dtype=np.float64)
    total_sq = np.zeros(2, dtype=np.float64)
    min_v = np.array([np.inf, np.inf], dtype=np.float64)
    max_v = np.array([-np.inf, -np.inf], dtype=np.float64)
    for row in results:
        if row.get("skipped"):
            continue
        total += np.asarray(row["trace_sum"], dtype=np.float64)
        total_sq += np.asarray(row["trace_sumsq"], dtype=np.float64)
        min_v = np.minimum(min_v, np.asarray(row["trace_min"], dtype=np.float64))
        max_v = np.maximum(max_v, np.asarray(row["trace_max"], dtype=np.float64))
    mean = total / count
    var = np.maximum(total_sq / count - np.square(mean), 0.0)
    return {
        "mean": mean.astype(float).tolist(),
        "std": np.sqrt(var).astype(float).tolist(),
        "min": min_v.astype(float).tolist(),
        "max": max_v.astype(float).tolist(),
    }


def main() -> int:
    args = parse_args()
    src_meta = args.src_root / "meta"
    dst_meta = args.dst_root / "meta"
    info = _load_json(src_meta / "info.json")
    episodes = _read_jsonl(src_meta / "episodes.jsonl")
    if args.max_episodes is not None:
        episodes = episodes[: int(args.max_episodes)]

    chunk_size = int(info.get("chunks_size", 1000))
    data_pattern = info["data_path"]
    coords_path = args.trace_index_root / f"{args.trace_key}_coords.npy"
    offsets_path = args.trace_index_root / f"{args.trace_key}_offsets.npy"
    if not coords_path.exists() or not offsets_path.exists():
        raise FileNotFoundError(f"missing trace index files for key {args.trace_key!r} under {args.trace_index_root}")
    offsets = np.load(offsets_path)
    coords_mmap = np.load(coords_path, mmap_mode="r")

    args.dst_root.mkdir(parents=True, exist_ok=True)
    dst_meta.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [
            pool.submit(
                _process_one,
                src_root=args.src_root,
                dst_root=args.dst_root,
                data_pattern=data_pattern,
                chunk_size=chunk_size,
                trace_field=args.trace_field,
                trace_episode_offset=args.trace_episode_offset,
                coords_mmap=coords_mmap,
                offsets=offsets,
                episode=episode,
                overwrite=bool(args.overwrite),
            )
            for episode in episodes
        ]
        for future in as_completed(futures):
            results.append(future.result())

    selected_episode_ids = {int(ep["episode_index"]) for ep in episodes}
    if args.video_mode != "none":
        for episode_index in sorted(selected_episode_ids):
            for src_rel, dst_rel in _episode_video_paths(info, episode_index, chunk_size):
                _copy_or_link(args.src_root / src_rel, args.dst_root / dst_rel, args.video_mode)

    # Copy or filter metadata.
    selected_task_ids = {int(ep["tasks"][0]) for ep in episodes if ep.get("tasks") and str(ep["tasks"][0]).isdigit()}
    tasks_path = src_meta / "tasks.jsonl"
    if tasks_path.exists():
        task_rows = _read_jsonl(tasks_path)
        if selected_task_ids:
            task_rows = [row for row in task_rows if int(row.get("task_index", -1)) in selected_task_ids]
        _write_jsonl(dst_meta / "tasks.jsonl", task_rows)
    _write_jsonl(dst_meta / "episodes.jsonl", episodes)

    episode_indices = sorted(int(ep["episode_index"]) for ep in episodes)
    min_episode = episode_indices[0] if episode_indices else 0
    max_episode = episode_indices[-1] if episode_indices else -1
    updated_info = dict(info)
    updated_info["total_episodes"] = len(episodes)
    updated_info["total_frames"] = int(sum(int(ep["length"]) for ep in episodes))
    updated_info["total_videos"] = len(episodes)
    updated_info["total_chunks"] = max(1, max_episode // chunk_size + 1)
    updated_info["splits"] = {"train": f"{min_episode}:{max_episode + 1}"}
    updated_info.setdefault("features", {})[args.trace_field] = {
        "dtype": "float32",
        "shape": [2],
        "names": ["x", "y"],
        "info": {"coordinate_space": "normalized_image_xy_0_100"},
    }
    _write_json(dst_meta / "info.json", updated_info)

    stats = _load_json(src_meta / "stats.json") if (src_meta / "stats.json").exists() else {}
    stats[args.trace_field] = _trace_stats(results)
    _write_json(dst_meta / "stats.json", stats)

    modality_path = src_meta / "modality.json"
    if modality_path.exists():
        modality = _load_json(modality_path)
        modality.setdefault("trace", {})["xy"] = {
            "start": 0,
            "end": 2,
            "original_key": args.trace_field,
            "dtype": "float32",
            "coordinate_space": "normalized_image_xy_0_100",
        }
        _write_json(dst_meta / "modality.json", modality)

    manifest = {
        "src_root": str(args.src_root),
        "dst_root": str(args.dst_root),
        "trace_index_root": str(args.trace_index_root),
        "trace_key": args.trace_key,
        "trace_field": args.trace_field,
        "trace_episode_offset": int(args.trace_episode_offset),
        "episodes": len(episodes),
        "frames": updated_info["total_frames"],
        "video_mode": args.video_mode,
    }
    _write_json(args.dst_root / "trace_integration_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
