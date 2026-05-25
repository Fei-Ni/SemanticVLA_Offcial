#!/usr/bin/env python3
"""Convert a selected DROID trace bundle to LeRobot v2 layout.

DROID trace bundles are compact camera-view selections.  Each selected episode
has a compact `episode_idx`, a `raw_episode_idx`, and a camera field in
`episode_metadata.json`.  This converter reads the matching raw RLDS record and
camera stream, then writes LeRobot episode parquet/video files with episode
indices aligned to the trace bundle.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))

from convert import RunningStats, _fixed_size_list, _write_json, _write_jsonl, _write_video  # noqa: E402
from examples.SemanticVLA_OXE.rlds_tfrecord import (  # noqa: E402
    bytes_feature,
    float_feature,
    iter_records,
    parse_example,
)


STATE_KEYS = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
VIDEO_KEY = "observation.images.image"


def _load_shard_lengths(root: Path) -> list[int]:
    with (root / "dataset_info.json").open("r", encoding="utf-8") as fp:
        info = json.load(fp)
    splits = info["splits"].values() if isinstance(info["splits"], dict) else info["splits"]
    for split in splits:
        if split["name"] == "train":
            return [int(x) for x in split["shardLengths"]]
    raise RuntimeError(f"no train split in {root / 'dataset_info.json'}")


def _shard_path(root: Path, shard_idx: int, total_shards: int) -> Path:
    return root / f"droid_101-train.tfrecord-{shard_idx:05d}-of-{total_shards:05d}"


def _episode_paths(out_root: Path, episode_index: int, chunk_size: int) -> tuple[Path, Path]:
    chunk = int(episode_index) // int(chunk_size)
    parquet = out_root / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
    video = out_root / f"videos/chunk-{chunk:03d}/{VIDEO_KEY}/episode_{episode_index:06d}.mp4"
    return parquet, video


def _instruction(example: Any, fallback: str | None) -> str:
    for key in (
        "steps/language_instruction",
        "steps/language_instruction_2",
        "steps/language_instruction_3",
    ):
        values = bytes_feature(example, key)
        for value in values:
            text = value.decode("utf-8", errors="replace")
            if text:
                return text
    return fallback or ""


def _droid_arrays(example: Any, camera: str, fallback_instruction: str | None) -> tuple[list[bytes], np.ndarray, np.ndarray, str]:
    images = bytes_feature(example, f"steps/observation/{camera}")
    actions = np.asarray(float_feature(example, "steps/action"), dtype=np.float32).reshape(-1, 7)
    cartesian = np.asarray(
        float_feature(example, "steps/observation/cartesian_position"),
        dtype=np.float32,
    ).reshape(-1, 6)
    gripper = np.asarray(
        float_feature(example, "steps/observation/gripper_position"),
        dtype=np.float32,
    ).reshape(-1, 1)
    states = np.concatenate([cartesian, gripper], axis=1).astype(np.float32, copy=False)
    return images, actions, states, _instruction(example, fallback_instruction)


def _write_parquet(
    path: Path,
    *,
    states: np.ndarray,
    actions: np.ndarray,
    timestamps: np.ndarray,
    episode_index: int,
    task_index: int,
    global_start_index: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    length = int(actions.shape[0])
    table = pa.Table.from_arrays(
        [
            _fixed_size_list(states, 7),
            _fixed_size_list(actions, 7),
            pa.array(timestamps.astype(np.float32), type=pa.float32()),
            pa.array(np.arange(length, dtype=np.int64), type=pa.int64()),
            pa.array(np.full(length, int(episode_index), dtype=np.int64), type=pa.int64()),
            pa.array(np.arange(global_start_index, global_start_index + length, dtype=np.int64), type=pa.int64()),
            pa.array(np.full(length, int(task_index), dtype=np.int64), type=pa.int64()),
        ],
        names=["observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index"],
    )
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp_path)
    os.replace(tmp_path, path)


def _process_job(payload: dict[str, Any]) -> dict[str, Any]:
    out_root = Path(payload["out_root"])
    chunk_size = int(payload["chunk_size"])
    fps = int(payload["fps"])
    image_size = tuple(payload["image_size"])
    overwrite = bool(payload["overwrite"])
    ffmpeg_bin = str(payload["ffmpeg_bin"])
    trace_offsets = np.load(payload["trace_offsets"]) if payload.get("trace_offsets") else None

    state_stats = RunningStats(7)
    action_stats = RunningStats(7)
    timestamp_stats = RunningStats(1)
    frame_index_stats = RunningStats(1)
    episode_index_stats = RunningStats(1)
    index_stats = RunningStats(1)
    task_index_stats = RunningStats(1)
    episodes: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    converted = 0
    skipped = 0

    for shard_payload in payload["shards"]:
        shard_path = Path(shard_payload["path"])
        targets_by_local: dict[int, list[dict[str, Any]]] = defaultdict(list)
        max_local = -1
        for item in shard_payload["items"]:
            local_idx = int(item["local_record_idx"])
            targets_by_local[local_idx].append(item)
            max_local = max(max_local, local_idx)

        for local_idx, serialized in enumerate(iter_records(shard_path)):
            if local_idx > max_local:
                break
            targets = targets_by_local.get(local_idx)
            if not targets:
                continue
            example = parse_example(serialized)
            for item in targets:
                episode_index = int(item["episode_idx"])
                camera = str(item["camera"])
                images, actions, states, instruction = _droid_arrays(
                    example,
                    camera,
                    item.get("language_instruction"),
                )
                length = min(len(images), int(actions.shape[0]), int(states.shape[0]))
                expected_steps = int(item.get("num_steps", length))
                if length != expected_steps:
                    length = min(length, expected_steps)
                if length <= 0:
                    continue
                images = images[:length]
                actions = actions[:length]
                states = states[:length]
                instruction = instruction or f"droid episode {episode_index}"
                parquet_path, video_path = _episode_paths(out_root, episode_index, chunk_size)
                if trace_offsets is not None:
                    global_start_index = int(trace_offsets[episode_index])
                else:
                    global_start_index = int(episode_index * 100000)
                timestamps = np.arange(length, dtype=np.float32) / float(fps)

                if overwrite or not (parquet_path.exists() and video_path.exists()):
                    _write_video(
                        video_path,
                        images=images,
                        image_size=image_size,
                        fps=fps,
                        ffmpeg_bin=ffmpeg_bin,
                    )
                    _write_parquet(
                        parquet_path,
                        states=states,
                        actions=actions,
                        timestamps=timestamps,
                        episode_index=episode_index,
                        task_index=episode_index,
                        global_start_index=global_start_index,
                    )
                    converted += 1
                else:
                    skipped += 1

                frame_index = np.arange(length, dtype=np.float64).reshape(-1, 1)
                global_index = np.arange(global_start_index, global_start_index + length, dtype=np.float64).reshape(-1, 1)
                state_stats.update(states)
                action_stats.update(actions)
                timestamp_stats.update(timestamps.reshape(-1, 1))
                frame_index_stats.update(frame_index)
                episode_index_stats.update(np.full((length, 1), float(episode_index)))
                index_stats.update(global_index)
                task_index_stats.update(np.full((length, 1), float(episode_index)))
                episodes.append({"episode_index": episode_index, "tasks": [instruction], "length": length})
                tasks.append({"task_index": episode_index, "task": instruction})
                selection_rows.append(item)

    return {
        "converted": converted,
        "skipped": skipped,
        "episodes": episodes,
        "tasks": tasks,
        "selection_rows": selection_rows,
        "stats": {
            "observation.state": state_stats.pack(),
            "action": action_stats.pack(),
            "timestamp": timestamp_stats.pack(),
            "frame_index": frame_index_stats.pack(),
            "episode_index": episode_index_stats.pack(),
            "index": index_stats.pack(),
            "task_index": task_index_stats.pack(),
        },
    }


def _build_jobs(
    *,
    dataset_root: Path,
    metadata: list[dict[str, Any]],
    shard_lengths: list[int],
    workers: int,
    out_root: Path,
    chunk_size: int,
    fps: int,
    image_size: tuple[int, int],
    overwrite: bool,
    ffmpeg_bin: str,
    trace_offsets: Path | None,
) -> list[dict[str, Any]]:
    starts = np.cumsum(np.asarray([0, *shard_lengths], dtype=np.int64))
    total_shards = len(shard_lengths)
    by_shard: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in metadata:
        raw_idx = int(row["raw_episode_idx"])
        shard_idx = int(np.searchsorted(starts, raw_idx, side="right") - 1)
        if shard_idx < 0 or shard_idx >= total_shards:
            raise IndexError(f"raw_episode_idx {raw_idx} outside DROID shard range")
        item = dict(row)
        item["local_record_idx"] = raw_idx - int(starts[shard_idx])
        by_shard[shard_idx].append(item)

    shard_indices = sorted(by_shard)
    group_size = max(1, math.ceil(len(shard_indices) / max(1, int(workers))))
    jobs: list[dict[str, Any]] = []
    for start in range(0, len(shard_indices), group_size):
        group = shard_indices[start : start + group_size]
        jobs.append(
            {
                "out_root": str(out_root),
                "chunk_size": int(chunk_size),
                "fps": int(fps),
                "image_size": list(image_size),
                "overwrite": overwrite,
                "ffmpeg_bin": ffmpeg_bin,
                "trace_offsets": str(trace_offsets) if trace_offsets else None,
                "shards": [
                    {
                        "path": str(_shard_path(dataset_root, shard_idx, total_shards)),
                        "items": sorted(by_shard[shard_idx], key=lambda item: int(item["local_record_idx"])),
                    }
                    for shard_idx in group
                ],
            }
        )
    return jobs


def _build_info(*, episodes: list[dict[str, Any]], chunk_size: int, fps: int) -> dict[str, Any]:
    total_episodes = len(episodes)
    total_frames = int(sum(int(ep["length"]) for ep in episodes))
    episode_indices = sorted(int(ep["episode_index"]) for ep in episodes)
    min_episode = episode_indices[0] if episode_indices else 0
    max_episode = episode_indices[-1] if episode_indices else -1
    height, width = 180, 320
    return {
        "codebase_version": "v2.0",
        "robot_type": "franka",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_episodes,
        "total_videos": total_episodes,
        "total_chunks": max(1, max_episode // int(chunk_size) + 1),
        "chunks_size": int(chunk_size),
        "fps": int(fps),
        "splits": {"train": f"{min_episode}:{max_episode + 1}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            VIDEO_KEY: {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.fps": float(fps),
                    "video.height": height,
                    "video.width": width,
                    "video.channels": 3,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [7],
                "names": {"motors": STATE_KEYS},
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": {"motors": ACTION_KEYS},
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }


def _build_modality() -> dict[str, Any]:
    return {
        "state": {
            key: {
                "start": idx,
                "end": idx + 1,
                "original_key": "observation.state",
                "dtype": "float32",
                "absolute": True,
            }
            for idx, key in enumerate(STATE_KEYS)
        },
        "action": {
            key: {
                "start": idx,
                "end": idx + 1,
                "original_key": "action",
                "dtype": "float32",
                "absolute": False,
            }
            for idx, key in enumerate(ACTION_KEYS)
        },
        "video": {"image": {"original_key": VIDEO_KEY}},
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
    }


def _build_stats(merged: dict[str, RunningStats]) -> dict[str, Any]:
    out = {name: stats.as_json() for name, stats in merged.items()}
    out[VIDEO_KEY] = {
        "mean": [[[0.5]], [[0.5]], [[0.5]]],
        "std": [[[0.5]], [[0.5]], [[0.5]]],
        "max": [[[1.0]], [[1.0]], [[1.0]]],
        "min": [[[0.0]], [[0.0]], [[0.0]]],
    }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("${DATA_ROOT}/datasets/droid/1.0.1"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("${DATA_ROOT}/droid_projection_runs/droid_56k_projection_trace_clip_dense_v1/episode_metadata.json"),
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--trace-index-root", type=Path, default=None)
    parser.add_argument("--trace-key", default="droid_56k")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    metadata = sorted(metadata, key=lambda row: int(row["episode_idx"]))
    if args.max_episodes is not None:
        metadata = metadata[: int(args.max_episodes)]
    if not metadata:
        raise ValueError(f"no selected DROID episodes in {args.metadata}")

    shard_lengths = _load_shard_lengths(args.dataset_root)
    trace_offsets = None
    if args.trace_index_root is not None:
        candidate = args.trace_index_root / f"{args.trace_key}_offsets.npy"
        if candidate.exists():
            trace_offsets = candidate

    args.out_root.mkdir(parents=True, exist_ok=True)
    jobs = _build_jobs(
        dataset_root=args.dataset_root,
        metadata=metadata,
        shard_lengths=shard_lengths,
        workers=int(args.workers),
        out_root=args.out_root,
        chunk_size=int(args.chunk_size),
        fps=int(args.fps),
        image_size=(320, 180),
        overwrite=bool(args.overwrite),
        ffmpeg_bin=str(args.ffmpeg_bin),
        trace_offsets=trace_offsets,
    )
    print(
        json.dumps(
            {
                "metadata": str(args.metadata),
                "out_root": str(args.out_root),
                "selected_episodes": len(metadata),
                "jobs": len(jobs),
                "workers": int(args.workers),
                "trace_offsets": str(trace_offsets) if trace_offsets else None,
            },
            indent=2,
        )
    )

    all_episodes: list[dict[str, Any]] = []
    all_tasks: list[dict[str, Any]] = []
    all_selection_rows: list[dict[str, Any]] = []
    merged_stats = {
        "observation.state": RunningStats(7),
        "action": RunningStats(7),
        "timestamp": RunningStats(1),
        "frame_index": RunningStats(1),
        "episode_index": RunningStats(1),
        "index": RunningStats(1),
        "task_index": RunningStats(1),
    }
    converted = 0
    skipped = 0
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(_process_job, job) for job in jobs]
        for idx, fut in enumerate(as_completed(futures), start=1):
            result = fut.result()
            converted += int(result["converted"])
            skipped += int(result["skipped"])
            all_episodes.extend(result["episodes"])
            all_tasks.extend(result["tasks"])
            all_selection_rows.extend(result["selection_rows"])
            for name, payload in result["stats"].items():
                merged_stats[name].merge(RunningStats.unpack(payload))
            print(
                f"progress jobs {idx}/{len(futures)} converted={converted} skipped={skipped} "
                f"episodes={len(all_episodes)}",
                flush=True,
            )

    all_episodes.sort(key=lambda row: int(row["episode_index"]))
    all_tasks.sort(key=lambda row: int(row["task_index"]))
    all_selection_rows.sort(key=lambda row: int(row["episode_idx"]))
    if len(all_episodes) != len(metadata):
        raise RuntimeError(f"wrote {len(all_episodes)} episodes, expected {len(metadata)}")

    meta = args.out_root / "meta"
    _write_jsonl(meta / "episodes.jsonl", all_episodes)
    _write_jsonl(meta / "tasks.jsonl", all_tasks)
    _write_jsonl(meta / "droid_trace_selection.jsonl", all_selection_rows)
    _write_json(meta / "info.json", _build_info(episodes=all_episodes, chunk_size=int(args.chunk_size), fps=int(args.fps)))
    _write_json(meta / "modality.json", _build_modality())
    _write_json(meta / "stats.json", _build_stats(merged_stats))
    _write_json(
        meta / "conversion_summary.json",
        {
            "dataset": "droid",
            "source_root": str(args.dataset_root),
            "metadata": str(args.metadata),
            "output_root": str(args.out_root),
            "episodes": len(all_episodes),
            "tasks": len(all_tasks),
            "converted": converted,
            "skipped": skipped,
            "max_episodes": args.max_episodes,
        },
    )
    print(f"[DONE] {args.out_root} episodes={len(all_episodes)} converted={converted} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
