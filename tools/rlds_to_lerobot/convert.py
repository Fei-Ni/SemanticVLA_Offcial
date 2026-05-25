#!/usr/bin/env python3
"""Convert selected OXE RLDS/TFRecord datasets into SemanticVLA LeRobot v2 layout.

The first production target is BC-Z. Episode indices follow TFDS train split
order exactly, so they remain aligned with the trace handoff files.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))

from examples.SemanticVLA_OXE.rlds_tfrecord import (
    bytes_feature,
    float_feature,
    int64_feature,
    iter_records,
    parse_example,
)


@dataclass(frozen=True)
class ConvertSpec:
    name: str
    output_name: str
    root: Path
    file_prefix: str
    split_name: str
    image_key: str
    robot_type: str
    image_size: tuple[int, int]
    fps: int


SPECS: dict[str, ConvertSpec] = {
    "bcz": ConvertSpec(
        name="bcz",
        output_name="bcz_0.1.0_lerobot",
        root=Path("${DATA_ROOT}/bc_z/0.1.0"),
        file_prefix="bc_z-train.tfrecord",
        split_name="train",
        image_key="steps/observation/image",
        robot_type="google_robot",
        image_size=(224, 224),
        fps=5,
    ),
    "bridge": ConvertSpec(
        name="bridge",
        output_name="bridge_train_1.0.0_lerobot",
        root=Path("${DATA_ROOT}/bridge_orig/1.0.0"),
        file_prefix="bridge_dataset-train.tfrecord",
        split_name="train",
        image_key="steps/observation/image_0",
        robot_type="widowx",
        image_size=(256, 256),
        fps=5,
    ),
    "fractal": ConvertSpec(
        name="fractal",
        output_name="fractal_train_present_0.1.0_lerobot",
        root=Path("${DATA_ROOT}/fractal20220817_data/0.1.0"),
        file_prefix="fractal20220817_data-train.tfrecord",
        split_name="train",
        image_key="steps/observation/image",
        robot_type="google_robot",
        image_size=(320, 256),
        fps=5,
    ),
}

STATE_KEYS = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


class RunningStats:
    def __init__(self, dim: int):
        self.dim = int(dim)
        self.count = 0
        self.sum = np.zeros(self.dim, dtype=np.float64)
        self.sumsq = np.zeros(self.dim, dtype=np.float64)
        self.min = np.full(self.dim, np.inf, dtype=np.float64)
        self.max = np.full(self.dim, -np.inf, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1, self.dim)
        if arr.size == 0:
            return
        self.count += int(arr.shape[0])
        self.sum += arr.sum(axis=0)
        self.sumsq += np.square(arr).sum(axis=0)
        self.min = np.minimum(self.min, arr.min(axis=0))
        self.max = np.maximum(self.max, arr.max(axis=0))

    def merge(self, other: "RunningStats") -> None:
        if other.count == 0:
            return
        self.count += other.count
        self.sum += other.sum
        self.sumsq += other.sumsq
        self.min = np.minimum(self.min, other.min)
        self.max = np.maximum(self.max, other.max)

    def as_json(self) -> dict[str, list[float]]:
        if self.count == 0:
            mean = np.zeros(self.dim, dtype=np.float64)
            std = np.zeros(self.dim, dtype=np.float64)
            min_v = np.zeros(self.dim, dtype=np.float64)
            max_v = np.zeros(self.dim, dtype=np.float64)
        else:
            mean = self.sum / self.count
            var = np.maximum(self.sumsq / self.count - np.square(mean), 0.0)
            std = np.sqrt(var)
            min_v = self.min
            max_v = self.max
        return {
            "mean": mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
            "max": max_v.astype(float).tolist(),
            "min": min_v.astype(float).tolist(),
        }

    def pack(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "count": self.count,
            "sum": self.sum.tolist(),
            "sumsq": self.sumsq.tolist(),
            "min": self.min.tolist(),
            "max": self.max.tolist(),
        }

    @classmethod
    def unpack(cls, payload: dict[str, Any]) -> "RunningStats":
        out = cls(int(payload["dim"]))
        out.count = int(payload["count"])
        out.sum = np.asarray(payload["sum"], dtype=np.float64)
        out.sumsq = np.asarray(payload["sumsq"], dtype=np.float64)
        out.min = np.asarray(payload["min"], dtype=np.float64)
        out.max = np.asarray(payload["max"], dtype=np.float64)
        return out


def _load_shard_lengths(root: Path, split_name: str) -> list[int]:
    with (root / "dataset_info.json").open("r", encoding="utf-8") as fp:
        info = json.load(fp)
    splits = info["splits"].values() if isinstance(info["splits"], dict) else info["splits"]
    for split in splits:
        if split["name"] == split_name:
            return [int(x) for x in split["shardLengths"]]
    raise RuntimeError(f"no {split_name} split in {root / 'dataset_info.json'}")


def _list_shards(spec: ConvertSpec, shard_lengths: list[int]) -> list[Path]:
    files = sorted(path for path in spec.root.iterdir() if path.name.startswith(spec.file_prefix))
    if len(files) != len(shard_lengths):
        raise RuntimeError(
            f"{spec.name}: file count {len(files)} != train shard count {len(shard_lengths)} "
            f"for prefix {spec.file_prefix!r}"
        )
    return files


def _list_contiguous_partial_shards(spec: ConvertSpec, shard_lengths: list[int]) -> tuple[list[Path], list[int]]:
    total = len(shard_lengths)
    shards: list[Path] = []
    for idx in range(total):
        path = spec.root / f"{spec.file_prefix}-{idx:05d}-of-{total:05d}"
        if not path.exists():
            break
        shards.append(path)
    if not shards:
        raise FileNotFoundError(f"{spec.name}: no contiguous shard 0 found under {spec.root}")
    return shards, shard_lengths[: len(shards)]


def _bcz_arrays(example: Any) -> tuple[list[bytes], np.ndarray, np.ndarray, str]:
    images = bytes_feature(example, "steps/observation/image")
    xyz = np.asarray(float_feature(example, "steps/action/future/xyz_residual"), dtype=np.float32).reshape(-1, 30)[:, :3]
    rot = np.asarray(float_feature(example, "steps/action/future/axis_angle_residual"), dtype=np.float32).reshape(-1, 30)[:, :3]
    grip = np.asarray(int64_feature(example, "steps/action/future/target_close"), dtype=np.float32).reshape(-1, 10)[:, :1]
    actions = np.concatenate([xyz, rot, grip], axis=1).astype(np.float32, copy=False)

    present_xyz = np.asarray(float_feature(example, "steps/observation/present/xyz"), dtype=np.float32).reshape(-1, 3)
    present_rot = np.asarray(float_feature(example, "steps/observation/present/axis_angle"), dtype=np.float32).reshape(-1, 3)
    present_grip = np.asarray(float_feature(example, "steps/observation/present/sensed_close"), dtype=np.float32).reshape(-1, 1)
    states = np.concatenate([present_xyz, present_rot, present_grip], axis=1).astype(np.float32, copy=False)

    instr_values = bytes_feature(example, "steps/observation/natural_language_instruction")
    if not instr_values:
        instr_values = bytes_feature(example, "steps/language_instruction")
    instruction = ""
    for value in instr_values:
        if value:
            instruction = value.decode("utf-8", errors="replace")
            break
    return images, actions, states, instruction


def _bridge_arrays(example: Any, image_key: str) -> tuple[list[bytes], np.ndarray, np.ndarray, str]:
    images = bytes_feature(example, image_key)
    actions = np.asarray(float_feature(example, "steps/action"), dtype=np.float32).reshape(-1, 7)
    states = np.asarray(float_feature(example, "steps/observation/state"), dtype=np.float32).reshape(-1, 7)
    instr_values = bytes_feature(example, "steps/language_instruction")
    instruction = ""
    for value in instr_values:
        if value:
            instruction = value.decode("utf-8", errors="replace")
            break
    return images, actions, states, instruction


def _fractal_arrays(example: Any, image_key: str) -> tuple[list[bytes], np.ndarray, np.ndarray, str]:
    images = bytes_feature(example, image_key)
    world = np.asarray(float_feature(example, "steps/action/world_vector"), dtype=np.float32).reshape(-1, 3)
    rot = np.asarray(float_feature(example, "steps/action/rotation_delta"), dtype=np.float32).reshape(-1, 3)
    grip = np.asarray(
        float_feature(example, "steps/action/gripper_closedness_action"),
        dtype=np.float32,
    ).reshape(-1, 1)
    actions = np.concatenate([world, rot, grip], axis=1).astype(np.float32, copy=False)
    states = np.asarray(
        float_feature(example, "steps/observation/base_pose_tool_reached"),
        dtype=np.float32,
    ).reshape(-1, 7)
    instr_values = bytes_feature(example, "steps/observation/natural_language_instruction")
    instruction = ""
    for value in instr_values:
        if value:
            instruction = value.decode("utf-8", errors="replace")
            break
    return images, actions, states, instruction


def _dataset_arrays(spec: ConvertSpec, example: Any) -> tuple[list[bytes], np.ndarray, np.ndarray, str]:
    if spec.name == "bcz":
        return _bcz_arrays(example)
    if spec.name == "bridge":
        return _bridge_arrays(example, spec.image_key)
    if spec.name == "fractal":
        return _fractal_arrays(example, spec.image_key)
    raise ValueError(f"unsupported dataset: {spec.name}")


def _fixed_size_list(values: np.ndarray, dim: int) -> pa.FixedSizeListArray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1, dim)
    flat = pa.array(arr.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, dim)


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


def _write_video(
    path: Path,
    *,
    images: list[bytes],
    image_size: tuple[int, int],
    fps: int,
    ffmpeg_bin: str,
) -> None:
    resolved_ffmpeg = shutil.which(ffmpeg_bin) or (ffmpeg_bin if Path(ffmpeg_bin).exists() else None)
    if resolved_ffmpeg is None:
        _write_video_pyav(
            path,
            images=images,
            image_size=image_size,
            fps=fps,
        )
        return

    width, height = image_size
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp.mp4")
    if tmp_path.exists():
        tmp_path.unlink()
    cmd = [
        resolved_ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        str(tmp_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for payload in images:
            img = Image.open(io.BytesIO(payload)).convert("RGB")
            if img.size != image_size:
                img = img.resize(image_size, Image.Resampling.BICUBIC)
            proc.stdin.write(np.asarray(img, dtype=np.uint8).tobytes())
        proc.stdin.close()
        rc = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {rc} for {path}")
    os.replace(tmp_path, path)


def _write_video_pyav(
    path: Path,
    *,
    images: list[bytes],
    image_size: tuple[int, int],
    fps: int,
) -> None:
    import av

    width, height = image_size
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp.mp4")
    if tmp_path.exists():
        tmp_path.unlink()
    container = av.open(str(tmp_path), mode="w")
    try:
        stream = container.add_stream("h264", rate=int(fps))
        stream.width = int(width)
        stream.height = int(height)
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "veryfast", "crf": "23"}
        for payload in images:
            img = Image.open(io.BytesIO(payload)).convert("RGB")
            if img.size != image_size:
                img = img.resize(image_size, Image.Resampling.BICUBIC)
            frame = av.VideoFrame.from_ndarray(np.asarray(img, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    os.replace(tmp_path, path)


def _episode_paths(out_root: Path, episode_index: int, chunk_size: int, video_key: str) -> tuple[Path, Path]:
    chunk = int(episode_index) // int(chunk_size)
    parquet = out_root / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
    video = out_root / f"videos/chunk-{chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    return parquet, video


def _process_shard_range(payload: dict[str, Any]) -> dict[str, Any]:
    spec = ConvertSpec(**payload["spec"])
    out_root = Path(payload["out_root"])
    shard_paths = [Path(p) for p in payload["shard_paths"]]
    shard_lengths = [int(x) for x in payload["shard_lengths"]]
    shard_start = int(payload["shard_start"])
    global_episode_start = int(payload["global_episode_start"])
    episode_index_offset = int(payload["episode_index_offset"])
    trace_episode_offset = int(payload["trace_episode_offset"])
    chunk_size = int(payload["chunk_size"])
    max_episodes = payload["max_episodes"]
    overwrite = bool(payload["overwrite"])
    ffmpeg_bin = str(payload["ffmpeg_bin"])
    trace_offsets = np.load(payload["trace_offsets"]) if payload.get("trace_offsets") else None
    trace_present = np.load(payload["trace_present"]) if payload.get("trace_present") else None

    state_stats = RunningStats(7)
    action_stats = RunningStats(7)
    timestamp_stats = RunningStats(1)
    frame_index_stats = RunningStats(1)
    episode_index_stats = RunningStats(1)
    index_stats = RunningStats(1)
    task_index_stats = RunningStats(1)
    episodes: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    converted = 0
    skipped = 0

    for local_shard_idx, shard_path in enumerate(shard_paths):
        shard_idx = shard_start + local_shard_idx
        shard_episode_base = global_episode_start + sum(shard_lengths[:local_shard_idx])
        for offset, serialized in enumerate(iter_records(shard_path)):
            local_episode_index = shard_episode_base + offset
            episode_index = episode_index_offset + local_episode_index
            trace_slot = episode_index - trace_episode_offset
            if max_episodes is not None and local_episode_index >= int(max_episodes):
                return {
                    "converted": converted,
                    "skipped": skipped,
                    "episodes": episodes,
                    "tasks": tasks,
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
            if trace_present is not None:
                if trace_slot < 0 or trace_slot >= len(trace_present):
                    continue
                if not bool(trace_present[trace_slot]):
                    continue
            example = parse_example(serialized)
            images, actions, states, instruction = _dataset_arrays(spec, example)
            length = min(len(images), int(actions.shape[0]), int(states.shape[0]))
            if length <= 0:
                continue
            images = images[:length]
            actions = actions[:length]
            states = states[:length]
            instruction = instruction or f"bcz episode {episode_index}"
            parquet_path, video_path = _episode_paths(out_root, episode_index, chunk_size, f"observation.images.{spec.image_key.rsplit('/', 1)[-1]}")
            if overwrite or not (parquet_path.exists() and video_path.exists()):
                timestamps = np.arange(length, dtype=np.float32) / float(spec.fps)
                if trace_offsets is not None:
                    global_start_index = int(trace_offsets[trace_slot])
                else:
                    global_start_index = int(episode_index * 100000)
                _write_video(video_path, images=images, image_size=spec.image_size, fps=spec.fps, ffmpeg_bin=ffmpeg_bin)
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
                timestamps = np.arange(length, dtype=np.float32) / float(spec.fps)
                if trace_offsets is not None:
                    global_start_index = int(trace_offsets[trace_slot])
                else:
                    global_start_index = int(episode_index * 100000)

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

        expected = shard_lengths[local_shard_idx]
        if offset + 1 != expected:
            raise RuntimeError(f"{shard_path.name}: read {offset + 1} records, expected {expected}")

    return {
        "converted": converted,
        "skipped": skipped,
        "episodes": episodes,
        "tasks": tasks,
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _build_info(spec: ConvertSpec, *, episodes: list[dict[str, Any]], chunk_size: int, total_tasks: int) -> dict[str, Any]:
    total_episodes = len(episodes)
    total_frames = int(sum(int(ep["length"]) for ep in episodes))
    episode_indices = sorted(int(ep["episode_index"]) for ep in episodes)
    min_episode = episode_indices[0] if episode_indices else 0
    max_episode = episode_indices[-1] if episode_indices else -1
    height, width = spec.image_size[1], spec.image_size[0]
    video_key = f"observation.images.{spec.image_key.rsplit('/', 1)[-1]}"
    return {
        "codebase_version": "v2.0",
        "robot_type": spec.robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes,
        "total_chunks": max(1, max_episode // int(chunk_size) + 1),
        "chunks_size": int(chunk_size),
        "fps": int(spec.fps),
        "splits": {"train": f"{min_episode}:{max_episode + 1}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            video_key: {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.fps": float(spec.fps),
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


def _build_modality(spec: ConvertSpec) -> dict[str, Any]:
    video_name = spec.image_key.rsplit("/", 1)[-1]
    video_key = f"observation.images.{video_name}"
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
        "video": {video_name: {"original_key": video_key}},
        "annotation": {
            "human.action.task_description": {
                "original_key": "task_index",
            }
        },
    }


def _build_stats(merged: dict[str, RunningStats], spec: ConvertSpec) -> dict[str, Any]:
    video_key = f"observation.images.{spec.image_key.rsplit('/', 1)[-1]}"
    out = {name: stats.as_json() for name, stats in merged.items()}
    out[video_key] = {
        "mean": [[[0.5]], [[0.5]], [[0.5]]],
        "std": [[[0.5]], [[0.5]], [[0.5]]],
        "max": [[[1.0]], [[1.0]], [[1.0]]],
        "min": [[[0.0]], [[0.0]], [[0.0]]],
    }
    return out


def _make_jobs(
    *,
    spec: ConvertSpec,
    out_root: Path,
    shards: list[Path],
    shard_lengths: list[int],
    workers: int,
    chunk_size: int,
    max_episodes: int | None,
    overwrite: bool,
    ffmpeg_bin: str,
    trace_offsets: Path | None,
    trace_present: Path | None,
    payload_episode_index_offset: int,
    payload_trace_episode_offset: int,
) -> list[dict[str, Any]]:
    shard_starts = np.cumsum(np.asarray([0, *shard_lengths], dtype=np.int64)).tolist()
    if max_episodes is not None:
        needed_shards = int(np.searchsorted(shard_starts, int(max_episodes), side="left"))
        shards = shards[:needed_shards]
        shard_lengths = shard_lengths[:needed_shards]
    total_shards = len(shards)
    group_size = max(1, math.ceil(total_shards / max(1, int(workers))))
    jobs: list[dict[str, Any]] = []
    for start in range(0, total_shards, group_size):
        end = min(total_shards, start + group_size)
        jobs.append(
            {
                "spec": {
                    "name": spec.name,
                    "output_name": spec.output_name,
                    "root": spec.root,
                    "file_prefix": spec.file_prefix,
                    "split_name": spec.split_name,
                    "image_key": spec.image_key,
                    "robot_type": spec.robot_type,
                    "image_size": spec.image_size,
                    "fps": spec.fps,
                },
                "out_root": str(out_root),
                "shard_paths": [str(path) for path in shards[start:end]],
                "shard_lengths": shard_lengths[start:end],
                "shard_start": start,
                "global_episode_start": int(sum(shard_lengths[:start])),
                "episode_index_offset": int(payload_episode_index_offset),
                "trace_episode_offset": int(payload_trace_episode_offset),
                "chunk_size": chunk_size,
                "max_episodes": max_episodes,
                "overwrite": overwrite,
                "ffmpeg_bin": ffmpeg_bin,
                "trace_offsets": str(trace_offsets) if trace_offsets else None,
                "trace_present": str(trace_present) if trace_present else None,
            }
        )
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(SPECS), required=True)
    parser.add_argument("--rlds-root", type=Path, default=None)
    parser.add_argument("--split-name", choices=("train", "val"), default=None)
    parser.add_argument("--episode-index-offset", type=int, default=0)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("${DATA_ROOT}/datasets/OXE_LEROBOT_SELFBUILT"))
    parser.add_argument("--trace-index-root", type=Path, default=Path("${WORK_ROOT}/trace_npy_index"))
    parser.add_argument("--trace-episode-offset", type=int, default=None)
    parser.add_argument(
        "--require-trace-present",
        action="store_true",
        help="Skip raw episodes that are false in <dataset>_present.npy.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--allow-partial-shards", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = SPECS[args.dataset]
    split_name = args.split_name or spec.split_name
    output_name = args.output_name
    if output_name is None and split_name != spec.split_name:
        output_name = f"{spec.name}_{split_name}_0.1.0_lerobot"
    if output_name is None:
        output_name = spec.output_name
    if args.rlds_root is not None or split_name != spec.split_name or output_name != spec.output_name:
        spec = ConvertSpec(
            name=spec.name,
            output_name=output_name,
            root=args.rlds_root or spec.root,
            file_prefix=f"bc_z-{split_name}.tfrecord" if spec.name == "bcz" else spec.file_prefix,
            split_name=split_name,
            image_key=spec.image_key,
            robot_type=spec.robot_type,
            image_size=spec.image_size,
            fps=spec.fps,
        )
    shard_lengths = _load_shard_lengths(spec.root, spec.split_name)
    if args.allow_partial_shards:
        shards, shard_lengths = _list_contiguous_partial_shards(spec, shard_lengths)
    else:
        shards = _list_shards(spec, shard_lengths)
    out_root = args.out_dir / spec.output_name
    out_root.mkdir(parents=True, exist_ok=True)
    trace_offsets = args.trace_index_root / f"{spec.name}_offsets.npy"
    if not trace_offsets.exists():
        trace_offsets = None
    trace_present = None
    if args.require_trace_present:
        trace_present = args.trace_index_root / f"{spec.name}_present.npy"
        if not trace_present.exists():
            raise FileNotFoundError(trace_present)
    trace_episode_offset = args.trace_episode_offset
    if trace_episode_offset is None:
        trace_episode_offset = int(args.episode_index_offset)

    jobs = _make_jobs(
        spec=spec,
        out_root=out_root,
        shards=shards,
        shard_lengths=shard_lengths,
        workers=int(args.workers),
        chunk_size=int(args.chunk_size),
        max_episodes=args.max_episodes,
        overwrite=bool(args.overwrite),
        ffmpeg_bin=str(args.ffmpeg_bin),
        trace_offsets=trace_offsets,
        trace_present=trace_present,
        payload_episode_index_offset=int(args.episode_index_offset),
        payload_trace_episode_offset=int(trace_episode_offset),
    )
    print(json.dumps({
        "dataset": spec.name,
        "out_root": str(out_root),
        "num_shards": len(shards),
        "jobs": len(jobs),
        "workers": int(args.workers),
        "max_episodes": args.max_episodes,
    }, indent=2))

    all_episodes: list[dict[str, Any]] = []
    all_tasks: list[dict[str, Any]] = []
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
        futures = [pool.submit(_process_shard_range, job) for job in jobs]
        for idx, fut in enumerate(as_completed(futures), start=1):
            result = fut.result()
            converted += int(result["converted"])
            skipped += int(result["skipped"])
            all_episodes.extend(result["episodes"])
            all_tasks.extend(result["tasks"])
            for name, payload in result["stats"].items():
                merged_stats[name].merge(RunningStats.unpack(payload))
            print(
                f"progress jobs {idx}/{len(futures)} converted={converted} skipped={skipped} "
                f"episodes={len(all_episodes)}",
                flush=True,
            )

    all_episodes.sort(key=lambda row: int(row["episode_index"]))
    all_tasks.sort(key=lambda row: int(row["task_index"]))
    if args.max_episodes is not None:
        all_episodes = [
            row
            for row in all_episodes
            if int(row["episode_index"]) - int(args.episode_index_offset) < int(args.max_episodes)
        ]
        all_tasks = [
            row
            for row in all_tasks
            if int(row["task_index"]) - int(args.episode_index_offset) < int(args.max_episodes)
        ]

    _write_jsonl(out_root / "meta/episodes.jsonl", all_episodes)
    _write_jsonl(out_root / "meta/tasks.jsonl", all_tasks)
    _write_json(out_root / "meta/info.json", _build_info(spec, episodes=all_episodes, chunk_size=int(args.chunk_size), total_tasks=len(all_tasks)))
    _write_json(out_root / "meta/modality.json", _build_modality(spec))
    _write_json(out_root / "meta/stats.json", _build_stats(merged_stats, spec))
    _write_json(
        out_root / "meta/conversion_summary.json",
        {
            "dataset": spec.name,
            "source_root": str(spec.root),
            "output_root": str(out_root),
            "episodes": len(all_episodes),
            "tasks": len(all_tasks),
            "converted": converted,
            "skipped": skipped,
            "max_episodes": args.max_episodes,
        },
    )
    print(f"[DONE] {out_root} episodes={len(all_episodes)} converted={converted} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
