#!/usr/bin/env python3
"""Repack a full-preserve TraceX LeRobot component into LeRobot v3.0 layout.

This is the release-path v3 repacker. It starts from one of our full-preserve
v2-style TraceX component packages, groups many episodes per parquet/video file,
and renames the trace representation from `observation.trace.xy` to scalar
`trace.x` and `trace.y` columns. The source v2 package is left untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


TRACE_V2_FIELD = "observation.trace.xy"
TRACE_X_FIELD = "trace.x"
TRACE_Y_FIELD = "trace.y"

DEFAULT_SOURCE = Path("${WORK_ROOT}/tracex/bridge_train_1.0.0_lerobot")
DEFAULT_OUTPUT = Path("${WORK_ROOT}/tracex_v30/bridge_train_1.0.0_lerobot_v30")
DEFAULT_FFMPEG = Path("${REPO_ROOT}/tools/miniforge3/envs/oxe-convert/bin/ffmpeg")

DATA_PATH_TEMPLATE = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH_TEMPLATE = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PATH_TEMPLATE = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _update_chunk_file_indices(chunk_idx: int, file_idx: int, chunk_size: int) -> tuple[int, int]:
    if file_idx == chunk_size - 1:
        return chunk_idx + 1, 0
    return chunk_idx, file_idx + 1


def _file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024**2)


def _parquet_uncompressed_size_mb(path: Path) -> float:
    meta = pq.read_metadata(path)
    total = 0
    for row_group in range(meta.num_row_groups):
        rg = meta.row_group(row_group)
        for col_idx in range(rg.num_columns):
            total += rg.column(col_idx).total_uncompressed_size
    return total / (1024**2)


def _replace_trace_column(table: pa.Table) -> pa.Table:
    if TRACE_V2_FIELD not in table.column_names:
        raise KeyError(f"{TRACE_V2_FIELD} missing from parquet schema")
    trace = table[TRACE_V2_FIELD]
    trace_x = pc.cast(pc.list_element(trace, 0), pa.float32())
    trace_y = pc.cast(pc.list_element(trace, 1), pa.float32())
    table = table.drop([TRACE_V2_FIELD])
    table = table.append_column(TRACE_X_FIELD, trace_x)
    table = table.append_column(TRACE_Y_FIELD, trace_y)
    return table


def _write_parquet_atomic(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, compression="snappy", use_dictionary=True)
    os.replace(tmp, path)


def _flush_data_file(paths: list[Path], out_root: Path, chunk_idx: int, file_idx: int) -> None:
    tables = [_replace_trace_column(pq.read_table(path)) for path in paths]
    table = pa.concat_tables(tables, promote_options="default")
    _write_parquet_atomic(out_root / DATA_PATH_TEMPLATE.format(chunk_index=chunk_idx, file_index=file_idx), table)


def convert_data_files(
    *,
    source_root: Path,
    out_root: Path,
    episodes: list[dict[str, Any]],
    data_file_size_mb: int,
    chunk_size: int,
    max_episodes: int | None,
) -> list[dict[str, Any]]:
    ep_paths = sorted((source_root / "data").glob("chunk-*/*.parquet"))
    if max_episodes is not None:
        ep_paths = ep_paths[: int(max_episodes)]
    if len(ep_paths) != len(episodes):
        raise RuntimeError(f"data parquet count {len(ep_paths)} does not match episode metadata count {len(episodes)}")

    rows: list[dict[str, Any]] = []
    paths_to_cat: list[Path] = []
    size_mb = 0.0
    chunk_idx = 0
    file_idx = 0
    dataset_index = 0

    for row, ep_path in zip(episodes, ep_paths, strict=True):
        ep_size_mb = _parquet_uncompressed_size_mb(ep_path)
        ep_frames = pq.read_metadata(ep_path).num_rows
        episode_index = int(row["episode_index"])
        rows.append(
            {
                "episode_index": episode_index,
                "data/chunk_index": chunk_idx,
                "data/file_index": file_idx,
                "dataset_from_index": dataset_index,
                "dataset_to_index": dataset_index + int(ep_frames),
            }
        )
        dataset_index += int(ep_frames)
        size_mb += ep_size_mb
        paths_to_cat.append(ep_path)
        if size_mb >= float(data_file_size_mb):
            _flush_data_file(paths_to_cat, out_root, chunk_idx, file_idx)
            print(
                f"[data] wrote chunk={chunk_idx:03d} file={file_idx:03d} episodes={len(paths_to_cat)} frames_to={dataset_index}",
                flush=True,
            )
            paths_to_cat = []
            size_mb = 0.0
            chunk_idx, file_idx = _update_chunk_file_indices(chunk_idx, file_idx, chunk_size)

    if paths_to_cat:
        _flush_data_file(paths_to_cat, out_root, chunk_idx, file_idx)
        print(
            f"[data] wrote chunk={chunk_idx:03d} file={file_idx:03d} episodes={len(paths_to_cat)} frames_to={dataset_index}",
            flush=True,
        )
    return rows


def _concat_list_line(path: Path) -> str:
    escaped = str(path.resolve()).replace("'", r"'\''")
    return f"file '{escaped}'\n"


def _concat_videos(paths: list[Path], out_path: Path, ffmpeg_bin: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_suffix(out_path.suffix + ".tmp.mp4")
    if tmp_out.exists():
        tmp_out.unlink()
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".ffconcat", delete=False) as fh:
        list_path = Path(fh.name)
        for path in paths:
            fh.write(_concat_list_line(path))
    cmd = [
        str(ffmpeg_bin),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
        str(tmp_out),
    ]
    try:
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed with code {proc.returncode}: {out_path}")
        os.replace(tmp_out, out_path)
    finally:
        list_path.unlink(missing_ok=True)
        tmp_out.unlink(missing_ok=True)


def convert_one_video_key(payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]], int]:
    source_root = Path(payload["source_root"])
    out_root = Path(payload["out_root"])
    video_key = str(payload["video_key"])
    episodes = payload["episodes"]
    video_file_size_mb = int(payload["video_file_size_mb"])
    chunk_size = int(payload["chunk_size"])
    fps = float(payload["fps"])
    ffmpeg_bin = Path(payload["ffmpeg_bin"])

    ep_paths = sorted((source_root / "videos").glob(f"chunk-*/{video_key}/episode_*.mp4"))
    ep_paths = ep_paths[: len(episodes)]
    if len(ep_paths) != len(episodes):
        raise RuntimeError(f"{video_key}: video count {len(ep_paths)} does not match episode count {len(episodes)}")

    rows: list[dict[str, Any]] = []
    paths_to_cat: list[Path] = []
    pending_row_indices: list[int] = []
    size_mb = 0.0
    duration_s = 0.0
    chunk_idx = 0
    file_idx = 0
    files_written = 0

    def flush() -> None:
        nonlocal paths_to_cat, pending_row_indices, size_mb, duration_s, chunk_idx, file_idx, files_written
        if not paths_to_cat:
            return
        out_path = out_root / VIDEO_PATH_TEMPLATE.format(video_key=video_key, chunk_index=chunk_idx, file_index=file_idx)
        _concat_videos(paths_to_cat, out_path, ffmpeg_bin)
        for idx in pending_row_indices:
            rows[idx][f"videos/{video_key}/chunk_index"] = chunk_idx
            rows[idx][f"videos/{video_key}/file_index"] = file_idx
        files_written += 1
        print(
            f"[video:{video_key}] wrote chunk={chunk_idx:03d} file={file_idx:03d} episodes={len(paths_to_cat)} size_mb={size_mb:.1f}",
            flush=True,
        )
        paths_to_cat = []
        pending_row_indices = []
        size_mb = 0.0
        duration_s = 0.0
        chunk_idx, file_idx = _update_chunk_file_indices(chunk_idx, file_idx, chunk_size)

    for ep_path, episode in zip(ep_paths, episodes, strict=True):
        ep_size_mb = _file_size_mb(ep_path)
        if paths_to_cat and size_mb + ep_size_mb >= float(video_file_size_mb):
            flush()

        ep_duration = float(episode["length"]) / fps
        rows.append(
            {
                "episode_index": int(episode["episode_index"]),
                f"videos/{video_key}/from_timestamp": duration_s,
                f"videos/{video_key}/to_timestamp": duration_s + ep_duration,
            }
        )
        pending_row_indices.append(len(rows) - 1)
        paths_to_cat.append(ep_path)
        size_mb += ep_size_mb
        duration_s += ep_duration

    flush()
    return video_key, rows, files_written


def convert_videos(
    *,
    source_root: Path,
    out_root: Path,
    video_keys: list[str],
    episodes: list[dict[str, Any]],
    video_file_size_mb: int,
    chunk_size: int,
    fps: int,
    ffmpeg_bin: Path,
    workers: int,
) -> tuple[list[dict[str, Any]], int]:
    if not video_keys:
        return [], 0
    payloads = [
        {
            "source_root": str(source_root),
            "out_root": str(out_root),
            "video_key": video_key,
            "episodes": episodes,
            "video_file_size_mb": int(video_file_size_mb),
            "chunk_size": int(chunk_size),
            "fps": int(fps),
            "ffmpeg_bin": str(ffmpeg_bin),
        }
        for video_key in video_keys
    ]
    merged: dict[int, dict[str, Any]] = {int(row["episode_index"]): {"episode_index": int(row["episode_index"])} for row in episodes}
    total_files = 0
    with ProcessPoolExecutor(max_workers=max(1, min(int(workers), len(payloads)))) as pool:
        futures = [pool.submit(convert_one_video_key, payload) for payload in payloads]
        for future in as_completed(futures):
            video_key, rows, files_written = future.result()
            total_files += int(files_written)
            for row in rows:
                ep_idx = int(row["episode_index"])
                merged[ep_idx].update(row)
            print(f"[video:{video_key}] done files={files_written}", flush=True)
    return [merged[int(row["episode_index"])] for row in episodes], total_files


def _convert_features(features: dict[str, dict[str, Any]], fps: int) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, spec in features.items():
        if key == TRACE_V2_FIELD:
            continue
        copied = dict(spec)
        if copied.get("dtype") != "video":
            copied.setdefault("fps", int(fps))
        out[key] = copied
    out[TRACE_X_FIELD] = {"dtype": "float32", "shape": [1], "names": ["x"], "fps": int(fps)}
    out[TRACE_Y_FIELD] = {"dtype": "float32", "shape": [1], "names": ["y"], "fps": int(fps)}
    return out


def _split_trace_stats(stats: dict[str, Any]) -> dict[str, Any]:
    stats = dict(stats)
    trace = stats.pop(TRACE_V2_FIELD, None)
    if trace is None:
        return stats
    for dim, name in enumerate([TRACE_X_FIELD, TRACE_Y_FIELD]):
        stats[name] = {}
        for key, values in trace.items():
            if isinstance(values, list) and values and isinstance(values[0], list):
                stats[name][key] = [values[dim]]
            elif isinstance(values, list):
                stats[name][key] = [values[dim]]
            else:
                stats[name][key] = values
    return stats


def _write_tasks(source_root: Path, out_root: Path, max_episodes: int | None) -> int:
    rows = _load_jsonl(source_root / "meta/tasks.jsonl")
    if max_episodes is not None:
        rows = rows[: int(max_episodes)]
    df = pd.DataFrame({"task_index": [int(row["task_index"]) for row in rows]}, index=pd.Index([row["task"] for row in rows], name="task"))
    path = out_root / "meta/tasks.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return len(rows)


def _write_episodes(out_root: Path, rows: list[dict[str, Any]]) -> None:
    path = out_root / EPISODES_PATH_TEMPLATE.format(chunk_index=0, file_index=0)
    _write_parquet_atomic(path, pa.Table.from_pylist(rows))


def _yaml_tag(text: str) -> str:
    return (
        text.strip()
        .lower()
        .replace("_", "-")
        .replace("/", "-")
        .replace(" ", "-")
    )


def _write_readme(
    out_root: Path,
    info: dict[str, Any],
    *,
    repo_id: str,
    pretty_name: str,
    component_name: str,
    source_name: str,
    coverage_note: str | None,
) -> None:
    video_keys = sorted(key for key, spec in info["features"].items() if spec.get("dtype") == "video")
    tags = [
        "robotics",
        "lerobot",
        _yaml_tag(component_name),
        _yaml_tag(str(info.get("robot_type") or "")),
        "semanticvla",
        "tracex",
        "lerobot-v3",
    ]
    tags = [tag for idx, tag in enumerate(tags) if tag and tag not in tags[:idx]]
    coverage_section = ""
    if coverage_note:
        coverage_section = f"""
## Coverage

{coverage_note}
"""
    text = f"""---
pretty_name: {pretty_name}
tags:
{chr(10).join(f"- {tag}" for tag in tags)}
license: other
configs:
- config_name: default
  data_files:
  - split: train
    path: data/**/*.parquet
---

# {pretty_name}

LeRobot v3.0 repack of the SemanticVLA TraceX {component_name} component.  This
package preserves the component episode order and raw observation/action fields
from the full-preserve v2-style release package, while reducing the file count
by grouping multiple episodes per parquet/video file.

Trace columns in this v3 package:

- `trace.x`: `float32[1]`, normalized image-space x coordinate on a 0-100 scale.
- `trace.y`: `float32[1]`, normalized image-space y coordinate on a 0-100 scale.

The v2-style `observation.trace.xy` column is intentionally not present in this
v3 experiment.

## Contents

- Repository: `{repo_id}`
- Format: LeRobot v3.0
- Episodes: {info["total_episodes"]:,}
- Frames: {info["total_frames"]:,}
- FPS: {info["fps"]}
- Robot type: `{info.get("robot_type")}`
- Source dataset: {source_name}
- Data file target size: {info["data_files_size_in_mb"]} MB
- Video file target size: {info["video_files_size_in_mb"]} MB

## Video Streams

{chr(10).join(f"- `{key}`" for key in video_keys)}
{coverage_section}

## Notes

The `configs` block in this card is intentional: it makes Hugging Face Data
Studio read `data/**/*.parquet` as the train split, so the parquet columns such
as `action`, `observation.*`, `trace.x`, and `trace.y` are visible instead of
auto-detecting the repository as a video-only dataset.

Use of this dataset is subject to the original source dataset terms, plus the
terms of the added SemanticVLA TraceX annotations.
"""
    (out_root / "README.md").write_text(text, encoding="utf-8")


def _copy_optional_metadata(source_root: Path, out_root: Path) -> None:
    for name in ["rlds_features.json", "modality.json", "conversion_summary.json"]:
        src = source_root / "meta" / name
        if src.exists():
            dst = out_root / "meta" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo-id", default="spikefly/SemanticVLA-Bridge-TraceX-v3")
    parser.add_argument("--pretty-name", default=None)
    parser.add_argument("--component-name", default=None)
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--coverage-note", default=None)
    parser.add_argument("--data-file-size-mb", type=int, default=100)
    parser.add_argument("--video-file-size-mb", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ffmpeg-bin", type=Path, default=DEFAULT_FFMPEG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root
    out_root = args.out_root
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    if not args.ffmpeg_bin.exists():
        raise FileNotFoundError(args.ffmpeg_bin)
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"{out_root} already exists; pass --overwrite to rebuild")
    out_root.mkdir(parents=True, exist_ok=True)

    info_v2 = _read_json(source_root / "meta/info.json")
    component_name = args.component_name or source_root.name.removesuffix("_lerobot").replace("_", " ")
    pretty_name = args.pretty_name or f"SemanticVLA {component_name.title()} TraceX v3"
    source_name = args.source_name or str(source_root)
    episodes = _load_jsonl(source_root / "meta/episodes.jsonl")
    episodes.sort(key=lambda row: int(row["episode_index"]))
    if args.max_episodes is not None:
        episodes = episodes[: int(args.max_episodes)]

    fps = int(info_v2["fps"])
    video_keys = sorted(key for key, spec in info_v2["features"].items() if spec.get("dtype") == "video")
    print(
        json.dumps(
            {
                "source_root": str(source_root),
                "out_root": str(out_root),
                "episodes": len(episodes),
                "video_keys": video_keys,
                "data_file_size_mb": args.data_file_size_mb,
                "video_file_size_mb": args.video_file_size_mb,
            },
            indent=2,
        ),
        flush=True,
    )

    tasks_count = _write_tasks(source_root, out_root, args.max_episodes)
    data_rows = convert_data_files(
        source_root=source_root,
        out_root=out_root,
        episodes=episodes,
        data_file_size_mb=int(args.data_file_size_mb),
        chunk_size=int(args.chunk_size),
        max_episodes=args.max_episodes,
    )
    video_rows, total_video_files = convert_videos(
        source_root=source_root,
        out_root=out_root,
        video_keys=video_keys,
        episodes=episodes,
        video_file_size_mb=int(args.video_file_size_mb),
        chunk_size=int(args.chunk_size),
        fps=fps,
        ffmpeg_bin=args.ffmpeg_bin,
        workers=int(args.workers),
    )

    merged_rows: list[dict[str, Any]] = []
    by_data = {int(row["episode_index"]): row for row in data_rows}
    by_video = {int(row["episode_index"]): row for row in video_rows}
    for episode in episodes:
        ep_idx = int(episode["episode_index"])
        merged = dict(episode)
        merged.update(by_data[ep_idx])
        merged.update(by_video.get(ep_idx, {}))
        merged["meta/episodes/chunk_index"] = 0
        merged["meta/episodes/file_index"] = 0
        merged_rows.append(merged)
    _write_episodes(out_root, merged_rows)

    total_frames = int(sum(int(row["length"]) for row in episodes))
    features = _convert_features(info_v2["features"], fps)
    info_v3 = {
        "codebase_version": "v3.0",
        "robot_type": info_v2.get("robot_type"),
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": tasks_count,
        "chunks_size": int(args.chunk_size),
        "data_files_size_in_mb": int(args.data_file_size_mb),
        "video_files_size_in_mb": int(args.video_file_size_mb),
        "fps": fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": DATA_PATH_TEMPLATE,
        "video_path": VIDEO_PATH_TEMPLATE if video_keys else None,
        "features": features,
        "trace_alignment": {
            **info_v2.get("trace_alignment", {}),
            "original_trace_field": TRACE_V2_FIELD,
            "trace_fields": [TRACE_X_FIELD, TRACE_Y_FIELD],
            "format_note": "Trace is split into scalar x/y fields only in this LeRobot v3.0 experiment.",
        },
    }
    _write_json(out_root / "meta/info.json", info_v3)
    _write_json(out_root / "meta/stats.json", _split_trace_stats(_read_json(source_root / "meta/stats.json")))
    _copy_optional_metadata(source_root, out_root)
    _write_json(
        out_root / "meta/v30_conversion_summary.json",
        {
            "source_root": str(source_root),
            "output_root": str(out_root),
            "source_codebase_version": info_v2.get("codebase_version"),
            "output_codebase_version": "v3.0",
            "episodes": len(episodes),
            "frames": total_frames,
            "video_files": total_video_files,
            "data_files": len(list((out_root / "data").glob("chunk-*/*.parquet"))),
            "trace_fields": [TRACE_X_FIELD, TRACE_Y_FIELD],
        },
    )
    _write_readme(
        out_root,
        info_v3,
        repo_id=args.repo_id,
        pretty_name=pretty_name,
        component_name=component_name,
        source_name=source_name,
        coverage_note=args.coverage_note,
    )
    print(f"[DONE] {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
