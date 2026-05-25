#!/usr/bin/env python3
"""Convert RLDS traces to full-preserve LeRobot components with trace.

This converter is for release packaging, not training.  It keeps the selected
episode/frame order used by our trace indexes, preserves all raw RLDS step
leaves that can be represented as LeRobot parquet columns or videos, and adds
one per-frame trace column:

  observation.trace.xy

The Hugging Face Bridge LeRobot release is used only as a package-shape
reference (`data/`, `videos/`, `meta/`).  Episode order and trace alignment are
always derived from our local raw shards, compact DROID metadata, and trace
indexes.
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
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent))

from convert import RunningStats, _write_json, _write_jsonl  # noqa: E402
from examples.SemanticVLA_OXE.rlds_tfrecord import (  # noqa: E402
    bytes_feature,
    float_feature,
    int64_feature,
    iter_records,
    parse_example,
)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: Path
    split_name: str
    file_prefix: str
    output_name: str
    robot_type: str
    fps: int
    trace_key: str
    trace_episode_offset: int = 0


@dataclass(frozen=True)
class Leaf:
    raw_key: str
    column: str
    kind: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def width(self) -> int:
        out = 1
        for dim in self.shape:
            out *= int(dim)
        return int(out)


CONFIGS: dict[str, DatasetConfig] = {
    "bcz": DatasetConfig(
        name="bcz",
        root=Path("${DATA_ROOT}/bc_z/0.1.0"),
        split_name="train",
        file_prefix="bc_z-train.tfrecord",
        output_name="bcz_train_0.1.0_lerobot",
        robot_type="google_robot",
        fps=5,
        trace_key="bcz",
    ),
    "bridge": DatasetConfig(
        name="bridge",
        root=Path("${DATA_ROOT}/bridge_orig/1.0.0"),
        split_name="train",
        file_prefix="bridge_dataset-train.tfrecord",
        output_name="bridge_train_1.0.0_lerobot",
        robot_type="widowx",
        fps=5,
        trace_key="bridge",
    ),
    "fractal": DatasetConfig(
        name="fractal",
        root=Path("${DATA_ROOT}/fractal20220817_data/0.1.0"),
        split_name="train",
        file_prefix="fractal20220817_data-train.tfrecord",
        output_name="fractal_train_0.1.0_lerobot",
        robot_type="google_robot",
        fps=5,
        trace_key="fractal",
    ),
    "droid_56k": DatasetConfig(
        name="droid_56k",
        root=Path("${DATA_ROOT}/datasets/droid/1.0.1"),
        split_name="train",
        file_prefix="droid_101-train.tfrecord",
        output_name="droid_56k_1.0.1_lerobot",
        robot_type="franka",
        fps=15,
        trace_key="droid_56k",
    ),
}

TRACE_FIELD = "observation.trace.xy"
STANDARD_COLUMNS = {"timestamp", "frame_index", "episode_index", "index", "task_index", TRACE_FIELD}


def _dims(shape_obj: dict[str, Any] | None) -> tuple[int, ...]:
    dims = (shape_obj or {}).get("dimensions", [])
    return tuple(int(dim) for dim in dims)


def _leaf_from_node(raw_key: str, node: dict[str, Any]) -> Leaf:
    if "image" in node:
        image = node["image"]
        return Leaf(
            raw_key=raw_key,
            column=_column_name(raw_key, image=True),
            kind="image",
            dtype=str(image.get("dtype", "uint8")),
            shape=_dims(image.get("shape")),
        )
    if "text" in node:
        return Leaf(raw_key=raw_key, column=_column_name(raw_key), kind="string", dtype="string", shape=())
    tensor = node.get("tensor", {})
    dtype = str(tensor.get("dtype", "unknown"))
    kind = "string" if dtype == "string" else "tensor"
    return Leaf(
        raw_key=raw_key,
        column=_column_name(raw_key),
        kind=kind,
        dtype=dtype,
        shape=_dims(tensor.get("shape")),
    )


def _iter_feature_leaves(node: dict[str, Any], prefix: str = "") -> list[Leaf]:
    if "featuresDict" in node:
        leaves: list[Leaf] = []
        for key, child in node["featuresDict"].get("features", {}).items():
            child_prefix = f"{prefix}/{key}" if prefix else key
            leaves.extend(_iter_feature_leaves(child, child_prefix))
        return leaves
    if "sequence" in node:
        return _iter_feature_leaves(node["sequence"]["feature"], prefix)
    return [_leaf_from_node(prefix, node)]


def _column_name(raw_key: str, *, image: bool = False) -> str:
    if raw_key.startswith("steps/observation/"):
        suffix = raw_key.removeprefix("steps/observation/").replace("/", ".")
        return f"observation.images.{suffix}" if image else f"observation.{suffix}"
    if raw_key == "steps/action":
        return "action"
    if raw_key.startswith("steps/action/"):
        return "action." + raw_key.removeprefix("steps/action/").replace("/", ".")
    if raw_key.startswith("steps/action_dict/"):
        return "action_dict." + raw_key.removeprefix("steps/action_dict/").replace("/", ".")
    if raw_key.startswith("steps/"):
        return raw_key.removeprefix("steps/").replace("/", ".")
    return raw_key.replace("/", ".")


def _load_features(root: Path) -> tuple[list[Leaf], list[Leaf], dict[str, Any]]:
    payload = json.loads((root / "features.json").read_text(encoding="utf-8"))
    leaves = _iter_feature_leaves(payload)
    step_leaves = [leaf for leaf in leaves if leaf.raw_key.startswith("steps/")]
    episode_leaves = [leaf for leaf in leaves if leaf.raw_key.startswith("episode_metadata/")]
    return step_leaves, episode_leaves, payload


def _load_shard_lengths(root: Path, split_name: str) -> list[int]:
    info = json.loads((root / "dataset_info.json").read_text(encoding="utf-8"))
    splits = info["splits"].values() if isinstance(info["splits"], dict) else info["splits"]
    for split in splits:
        if split["name"] == split_name:
            return [int(x) for x in split["shardLengths"]]
    raise RuntimeError(f"no {split_name} split in {root / 'dataset_info.json'}")


def _list_shards(cfg: DatasetConfig, shard_lengths: list[int]) -> list[Path]:
    files = sorted(path for path in cfg.root.iterdir() if path.name.startswith(cfg.file_prefix))
    if len(files) != len(shard_lengths):
        raise RuntimeError(f"{cfg.name}: found {len(files)} shards, expected {len(shard_lengths)}")
    return files


def _has_feature(example: Any, key: str) -> bool:
    return key in example.features.feature


def _raw_values(example: Any, leaf: Leaf) -> list[Any]:
    if not _has_feature(example, leaf.raw_key):
        return []
    if leaf.kind in {"image", "string"}:
        return bytes_feature(example, leaf.raw_key)
    if leaf.dtype.startswith("float"):
        return float_feature(example, leaf.raw_key)
    return int64_feature(example, leaf.raw_key)


def _leaf_length(example: Any, leaf: Leaf) -> int:
    values = _raw_values(example, leaf)
    if leaf.kind in {"image", "string"}:
        return len(values)
    width = max(1, leaf.width)
    return len(values) // width


def _episode_length(example: Any, leaves: list[Leaf]) -> int:
    counts = [_leaf_length(example, leaf) for leaf in leaves if leaf.raw_key.startswith("steps/")]
    counts = [count for count in counts if count > 0]
    if not counts:
        return 0
    return max(counts)


def _pa_type(dtype: str) -> pa.DataType:
    if dtype == "float64":
        return pa.float64()
    if dtype == "float32" or dtype.startswith("float"):
        return pa.float32()
    if dtype == "int32":
        return pa.int32()
    if dtype == "bool":
        return pa.bool_()
    return pa.int64()


def _np_dtype(dtype: str) -> Any:
    if dtype == "float64":
        return np.float64
    if dtype == "float32" or dtype.startswith("float"):
        return np.float32
    if dtype == "int32":
        return np.int32
    if dtype == "bool":
        return np.bool_
    return np.int64


def _null_numeric(leaf: Leaf, length: int) -> pa.Array:
    if leaf.width == 1:
        return pa.nulls(length, type=_pa_type(leaf.dtype))
    value_type = _pa_type(leaf.dtype)
    flat = pa.nulls(length * leaf.width, type=value_type)
    return pa.FixedSizeListArray.from_arrays(flat, leaf.width)


def _tensor_array(example: Any, leaf: Leaf, length: int) -> tuple[pa.Array, np.ndarray | None]:
    values = _raw_values(example, leaf)
    width = max(1, leaf.width)
    if not values:
        return _null_numeric(leaf, length), None
    arr = np.asarray(values, dtype=_np_dtype(leaf.dtype)).reshape(-1, width)
    if arr.shape[0] == 1 and length > 1:
        arr = np.repeat(arr, length, axis=0)
    if arr.shape[0] < length:
        pad = np.zeros((length - arr.shape[0], width), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=0)
    arr = arr[:length]
    if leaf.width == 1:
        return pa.array(arr[:, 0], type=_pa_type(leaf.dtype)), arr
    flat = pa.array(arr.reshape(-1), type=_pa_type(leaf.dtype))
    return pa.FixedSizeListArray.from_arrays(flat, leaf.width), arr


def _string_array(example: Any, leaf: Leaf, length: int) -> pa.Array:
    values = _raw_values(example, leaf)
    strings = [value.decode("utf-8", errors="replace") for value in values[:length]]
    if len(strings) == 1 and length > 1:
        strings = strings * length
    if len(strings) < length:
        strings.extend([""] * (length - len(strings)))
    return pa.array(strings, type=pa.string())


def _fixed_trace(coords: np.ndarray) -> pa.FixedSizeListArray:
    arr = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
    flat = pa.array(arr.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, 2)


def _image_size(leaf: Leaf) -> tuple[int, int]:
    if len(leaf.shape) < 2:
        raise ValueError(f"{leaf.raw_key} has no image shape")
    height, width = int(leaf.shape[0]), int(leaf.shape[1])
    return width, height


def _write_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def _write_video_preserve(
    path: Path,
    *,
    images: list[bytes],
    image_size: tuple[int, int],
    fps: int,
    ffmpeg_bin: str,
) -> tuple[int, int]:
    """Write a LeRobot video without training-style resizing.

    H.264 with yuv420p requires even dimensions.  When a raw dataset has odd
    dimensions (BC-Z is 213x171), pad only the encoded frame boundary by one
    row/column while keeping all original pixels unchanged in the top-left
    region.  The returned size is the encoded video size.
    """

    width, height = image_size
    encoded_width = int(width + (width % 2))
    encoded_height = int(height + (height % 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp.mp4")
    if tmp_path.exists():
        tmp_path.unlink()
    resolved = shutil.which(ffmpeg_bin) or (ffmpeg_bin if Path(ffmpeg_bin).exists() else None)
    if resolved is None:
        raise FileNotFoundError(f"missing ffmpeg: {ffmpeg_bin}")
    cmd = [
        resolved,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{encoded_width}x{encoded_height}",
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
                raise ValueError(f"expected image size {image_size}, got {img.size} for {path}")
            arr = np.asarray(img, dtype=np.uint8)
            if encoded_width != width or encoded_height != height:
                padded = np.zeros((encoded_height, encoded_width, 3), dtype=np.uint8)
                padded[:height, :width] = arr
                arr = padded
            proc.stdin.write(arr.tobytes())
        proc.stdin.close()
        rc = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {rc} for {path}")
    os.replace(tmp_path, path)
    return encoded_width, encoded_height


def _episode_paths(out_root: Path, episode_index: int, chunk_size: int, video_key: str) -> tuple[Path, Path]:
    chunk = int(episode_index) // int(chunk_size)
    parquet = out_root / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
    video = out_root / f"videos/chunk-{chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    return parquet, video


def _metadata_value(example: Any, leaf: Leaf) -> Any:
    values = _raw_values(example, leaf)
    if not values:
        return None
    if leaf.kind == "string":
        return values[0].decode("utf-8", errors="replace")
    if leaf.kind == "tensor":
        width = max(1, leaf.width)
        arr = np.asarray(values, dtype=_np_dtype(leaf.dtype)).reshape(-1, width)
        if arr.shape[0] == 1 and width == 1:
            return arr[0, 0].item()
        return arr.tolist()
    return None


def _instruction(example: Any, leaves: list[Leaf]) -> str:
    preferred = [
        "steps/language_instruction",
        "steps/language_instruction_2",
        "steps/language_instruction_3",
        "steps/observation/natural_language_instruction",
    ]
    by_key = {leaf.raw_key: leaf for leaf in leaves}
    for key in preferred:
        leaf = by_key.get(key)
        if leaf is None:
            continue
        values = _raw_values(example, leaf)
        for value in values:
            text = value.decode("utf-8", errors="replace")
            if text:
                return text
    return ""


def _maybe_derived_columns(
    *,
    dataset: str,
    example: Any,
    length: int,
    existing: set[str],
) -> list[tuple[str, pa.Array, np.ndarray | None, dict[str, Any]]]:
    out: list[tuple[str, pa.Array, np.ndarray | None, dict[str, Any]]] = []

    def add(name: str, arr: np.ndarray, names: list[str], absolute: bool) -> None:
        if name in existing:
            return
        arr = np.asarray(arr, dtype=np.float32).reshape(length, -1)
        flat = pa.array(arr.reshape(-1), type=pa.float32())
        out.append(
            (
                name,
                pa.FixedSizeListArray.from_arrays(flat, arr.shape[1]),
                arr,
                {
                    "dtype": "float32",
                    "shape": [int(arr.shape[1])],
                    "names": {"motors": names},
                    "derived": True,
                    "absolute": bool(absolute),
                },
            )
        )

    try:
        if dataset == "bcz":
            xyz = np.asarray(float_feature(example, "steps/action/future/xyz_residual"), dtype=np.float32).reshape(-1, 30)[:length, :3]
            rot = np.asarray(float_feature(example, "steps/action/future/axis_angle_residual"), dtype=np.float32).reshape(-1, 30)[:length, :3]
            grip = np.asarray(int64_feature(example, "steps/action/future/target_close"), dtype=np.float32).reshape(-1, 10)[:length, :1]
            add("action", np.concatenate([xyz, rot, grip], axis=1), ["x", "y", "z", "rx", "ry", "rz", "gripper"], False)
            px = np.asarray(float_feature(example, "steps/observation/present/xyz"), dtype=np.float32).reshape(-1, 3)[:length]
            pr = np.asarray(float_feature(example, "steps/observation/present/axis_angle"), dtype=np.float32).reshape(-1, 3)[:length]
            pg = np.asarray(float_feature(example, "steps/observation/present/sensed_close"), dtype=np.float32).reshape(-1, 1)[:length]
            add("observation.state", np.concatenate([px, pr, pg], axis=1), ["x", "y", "z", "rx", "ry", "rz", "gripper"], True)
        elif dataset == "fractal":
            world = np.asarray(float_feature(example, "steps/action/world_vector"), dtype=np.float32).reshape(-1, 3)[:length]
            rot = np.asarray(float_feature(example, "steps/action/rotation_delta"), dtype=np.float32).reshape(-1, 3)[:length]
            grip = np.asarray(float_feature(example, "steps/action/gripper_closedness_action"), dtype=np.float32).reshape(-1, 1)[:length]
            add("action", np.concatenate([world, rot, grip], axis=1), ["x", "y", "z", "rx", "ry", "rz", "gripper"], False)
            state = np.asarray(float_feature(example, "steps/observation/base_pose_tool_reached"), dtype=np.float32).reshape(-1, 7)[:length]
            add("observation.state", state, ["x", "y", "z", "rx", "ry", "rz", "gripper"], True)
        elif dataset == "droid_56k":
            cart = np.asarray(float_feature(example, "steps/observation/cartesian_position"), dtype=np.float32).reshape(-1, 6)[:length]
            grip = np.asarray(float_feature(example, "steps/observation/gripper_position"), dtype=np.float32).reshape(-1, 1)[:length]
            add("observation.state", np.concatenate([cart, grip], axis=1), ["x", "y", "z", "rx", "ry", "rz", "gripper"], True)
    except Exception:
        # Derived columns are convenience only; raw leaves are still preserved.
        return out
    return out


def _feature_spec_for_leaf(leaf: Leaf) -> dict[str, Any]:
    if leaf.kind == "image":
        height, width = int(leaf.shape[0]), int(leaf.shape[1])
        return {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "rgb"],
            "original_key": leaf.raw_key,
            "info": {
                "video.fps": None,
                "video.height": height,
                "video.width": width,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    if leaf.kind == "string":
        return {"dtype": "string", "shape": [1], "names": None, "original_key": leaf.raw_key}
    shape = list(leaf.shape) if leaf.shape else [1]
    return {"dtype": leaf.dtype, "shape": shape, "names": None, "original_key": leaf.raw_key}


def _process_records(payload: dict[str, Any]) -> dict[str, Any]:
    cfg = DatasetConfig(**payload["cfg"])
    out_root = Path(payload["out_root"])
    step_leaves = [Leaf(**leaf) for leaf in payload["step_leaves"]]
    episode_leaves = [Leaf(**leaf) for leaf in payload["episode_leaves"]]
    image_leaves = [leaf for leaf in step_leaves if leaf.kind == "image"]
    parquet_leaves = [leaf for leaf in step_leaves if leaf.kind != "image"]
    trace_offsets = np.load(payload["trace_offsets"])
    trace_coords = np.load(payload["trace_coords"], mmap_mode="r")
    trace_present = np.load(payload["trace_present"]) if payload.get("trace_present") else None
    chunk_size = int(payload["chunk_size"])
    overwrite = bool(payload["overwrite"])
    ffmpeg_bin = str(payload["ffmpeg_bin"])
    max_episodes = payload.get("max_episodes")

    stats: dict[str, RunningStats] = {}
    features: dict[str, dict[str, Any]] = {}
    episodes: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    converted = 0
    skipped = 0

    def update_stats(name: str, arr: np.ndarray | None) -> None:
        if arr is None:
            return
        values = np.asarray(arr)
        if values.dtype.kind not in {"f", "i", "u"}:
            return
        flat = values.reshape(values.shape[0], -1).astype(np.float64, copy=False)
        if not np.isfinite(flat).all():
            flat = flat[np.isfinite(flat).all(axis=1)]
            if flat.shape[0] == 0:
                return
        stats.setdefault(name, RunningStats(flat.shape[1])).update(flat)

    by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in payload["items"]:
        if "serialized" in item:
            continue
        by_shard[str(item["shard_path"])].append(item)

    def iter_materialized_items():
        for direct in payload["items"]:
            if "serialized" in direct:
                yield direct
        for shard_path, shard_items in sorted(by_shard.items()):
            targets = defaultdict(list)
            max_local = -1
            for shard_item in shard_items:
                local_idx = int(shard_item["local_record_idx"])
                targets[local_idx].append(shard_item)
                max_local = max(max_local, local_idx)
            for local_idx, serialized in enumerate(iter_records(shard_path)):
                if local_idx > max_local:
                    break
                for shard_item in targets.get(local_idx, []):
                    materialized = dict(shard_item)
                    materialized["serialized"] = serialized
                    yield materialized

    for item in iter_materialized_items():
        episode_index = int(item["episode_index"])
        if max_episodes is not None and converted + skipped >= int(max_episodes):
            break
        trace_slot = episode_index - int(cfg.trace_episode_offset)
        if trace_present is not None:
            if trace_slot < 0 or trace_slot >= len(trace_present) or not bool(trace_present[trace_slot]):
                continue
        serialized = item["serialized"]
        example = parse_example(serialized)
        length = _episode_length(example, step_leaves)
        if length <= 0:
            continue
        trace_start, trace_end = int(trace_offsets[trace_slot]), int(trace_offsets[trace_slot + 1])
        coords = np.asarray(trace_coords[trace_start:trace_end], dtype=np.float32)
        if coords.shape[0] != length:
            raise ValueError(f"episode {episode_index}: raw length {length} != trace rows {coords.shape[0]}")

        arrays: list[pa.Array] = []
        names: list[str] = []
        existing: set[str] = set()
        for leaf in parquet_leaves:
            if leaf.kind == "string":
                arr = _string_array(example, leaf, length)
                values = None
            else:
                arr, values = _tensor_array(example, leaf, length)
            arrays.append(arr)
            names.append(leaf.column)
            existing.add(leaf.column)
            features.setdefault(leaf.column, _feature_spec_for_leaf(leaf))
            update_stats(leaf.column, values)

        for name, arr, values, spec in _maybe_derived_columns(dataset=cfg.name, example=example, length=length, existing=existing):
            arrays.append(arr)
            names.append(name)
            existing.add(name)
            features.setdefault(name, spec)
            update_stats(name, values)

        timestamps = np.arange(length, dtype=np.float32) / float(cfg.fps)
        frame_index = np.arange(length, dtype=np.int64)
        global_index = np.arange(trace_start, trace_end, dtype=np.int64)
        standards = [
            ("timestamp", pa.array(timestamps, type=pa.float32()), timestamps.reshape(-1, 1), {"dtype": "float32", "shape": [1], "names": None}),
            ("frame_index", pa.array(frame_index, type=pa.int64()), frame_index.reshape(-1, 1), {"dtype": "int64", "shape": [1], "names": None}),
            ("episode_index", pa.array(np.full(length, episode_index, dtype=np.int64), type=pa.int64()), np.full((length, 1), episode_index, dtype=np.int64), {"dtype": "int64", "shape": [1], "names": None}),
            ("index", pa.array(global_index, type=pa.int64()), global_index.reshape(-1, 1), {"dtype": "int64", "shape": [1], "names": None}),
            ("task_index", pa.array(np.full(length, episode_index, dtype=np.int64), type=pa.int64()), np.full((length, 1), episode_index, dtype=np.int64), {"dtype": "int64", "shape": [1], "names": None}),
            (TRACE_FIELD, _fixed_trace(coords), coords, {"dtype": "float32", "shape": [2], "names": ["x", "y"]}),
        ]
        for name, arr, values, spec in standards:
            arrays.append(arr)
            names.append(name)
            features.setdefault(name, spec)
            update_stats(name, values)

        parquet_path, _ = _episode_paths(out_root, episode_index, chunk_size, image_leaves[0].column if image_leaves else "image")
        missing_video = False
        if image_leaves:
            for leaf in image_leaves:
                _, video_path = _episode_paths(out_root, episode_index, chunk_size, leaf.column)
                if not video_path.exists():
                    missing_video = True
                    break
        if overwrite or not parquet_path.exists() or missing_video:
            _write_parquet(parquet_path, pa.Table.from_arrays(arrays, names=names))
            for leaf in image_leaves:
                values = _raw_values(example, leaf)
                video_payloads = list(values[:length])
                if len(video_payloads) < length:
                    video_payloads.extend([b""] * (length - len(video_payloads)))
                _, video_path = _episode_paths(out_root, episode_index, chunk_size, leaf.column)
                encoded_width, encoded_height = _write_video_preserve(
                    video_path,
                    images=video_payloads,
                    image_size=_image_size(leaf),
                    fps=int(cfg.fps),
                    ffmpeg_bin=ffmpeg_bin,
                )
                features.setdefault(leaf.column, _feature_spec_for_leaf(leaf))
                features[leaf.column]["info"]["video.fps"] = float(cfg.fps)
                features[leaf.column]["info"]["video.encoded_width"] = encoded_width
                features[leaf.column]["info"]["video.encoded_height"] = encoded_height
            converted += 1
        else:
            skipped += 1

        metadata = {leaf.raw_key.removeprefix("episode_metadata/").replace("/", "."): _metadata_value(example, leaf) for leaf in episode_leaves}
        task = _instruction(example, step_leaves) or f"{cfg.name} episode {episode_index}"
        episode_row = {"episode_index": episode_index, "tasks": [task], "length": length}
        if metadata:
            episode_row["episode_metadata"] = metadata
        if "source" in item:
            episode_row["source"] = item["source"]
        episodes.append(episode_row)
        tasks.append({"task_index": episode_index, "task": task})

    return {
        "converted": converted,
        "skipped": skipped,
        "episodes": episodes,
        "tasks": tasks,
        "features": features,
        "stats": {name: stat.pack() for name, stat in stats.items()},
    }


def _raw_items_for_shards(
    *,
    shards: list[Path],
    shard_lengths: list[int],
    episode_offset: int,
    max_episodes: int | None,
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    episode_base = 0
    emitted = 0
    for shard_path, shard_len in zip(shards, shard_lengths, strict=True):
        group: list[dict[str, Any]] = []
        for local_idx in range(int(shard_len)):
            if max_episodes is not None and emitted >= int(max_episodes):
                break
            episode_index = int(episode_offset + episode_base + local_idx)
            group.append(
                {
                    "episode_index": episode_index,
                    "shard_path": str(shard_path),
                    "local_record_idx": int(local_idx),
                    "source": {"shard": shard_path.name, "local_record_idx": local_idx},
                }
            )
            emitted += 1
        if len(group) > 0:
            groups.append(group)
        episode_base += int(shard_len)
        if max_episodes is not None and emitted >= int(max_episodes):
            break
    return groups


def _droid_items(
    *,
    cfg: DatasetConfig,
    metadata_path: Path,
    shard_lengths: list[int],
    max_episodes: int | None,
) -> list[list[dict[str, Any]]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = sorted(metadata, key=lambda row: int(row["episode_idx"]))
    if max_episodes is not None:
        metadata = metadata[: int(max_episodes)]
    starts = np.cumsum(np.asarray([0, *shard_lengths], dtype=np.int64))
    by_shard: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in metadata:
        raw_idx = int(row["raw_episode_idx"])
        shard_idx = int(np.searchsorted(starts, raw_idx, side="right") - 1)
        item = dict(row)
        item["local_record_idx"] = raw_idx - int(starts[shard_idx])
        by_shard[shard_idx].append(item)

    groups: list[list[dict[str, Any]]] = []
    total_shards = len(shard_lengths)
    for shard_idx in sorted(by_shard):
        shard_path = cfg.root / f"{cfg.file_prefix}-{shard_idx:05d}-of-{total_shards:05d}"
        group: list[dict[str, Any]] = []
        for row in sorted(by_shard[shard_idx], key=lambda item: int(item["local_record_idx"])):
            local_idx = int(row["local_record_idx"])
            group.append(
                {
                    "episode_index": int(row["episode_idx"]),
                    "shard_path": str(shard_path),
                    "local_record_idx": local_idx,
                    "source": {
                        "raw_episode_idx": int(row["raw_episode_idx"]),
                        "camera_for_trace": row.get("camera"),
                        "shard": shard_path.name,
                        "local_record_idx": local_idx,
                    },
                }
            )
        if group:
            groups.append(group)
    return groups


def _merge_stats(packed: dict[str, Any], merged: dict[str, RunningStats]) -> None:
    for name, payload in packed.items():
        stat = RunningStats.unpack(payload)
        merged.setdefault(name, RunningStats(stat.dim)).merge(stat)


def _stats_json(stats: dict[str, RunningStats], features: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {name: stat.as_json() for name, stat in stats.items()}
    for name, spec in features.items():
        if spec.get("dtype") == "video":
            out[name] = {
                "mean": [[[0.5]], [[0.5]], [[0.5]]],
                "std": [[[0.5]], [[0.5]], [[0.5]]],
                "max": [[[1.0]], [[1.0]], [[1.0]]],
                "min": [[[0.0]], [[0.0]], [[0.0]]],
            }
    return out


def _build_info(cfg: DatasetConfig, *, episodes: list[dict[str, Any]], features: dict[str, dict[str, Any]], chunk_size: int) -> dict[str, Any]:
    episode_indices = sorted(int(row["episode_index"]) for row in episodes)
    total_frames = int(sum(int(row["length"]) for row in episodes))
    min_episode = episode_indices[0] if episode_indices else 0
    max_episode = episode_indices[-1] if episode_indices else -1
    video_features = [name for name, spec in features.items() if spec.get("dtype") == "video"]
    return {
        "codebase_version": "v2.0",
        "robot_type": cfg.robot_type,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(episodes),
        "total_videos": len(episodes) * len(video_features),
        "total_chunks": max(1, max_episode // int(chunk_size) + 1),
        "chunks_size": int(chunk_size),
        "fps": int(cfg.fps),
        "splits": {"train": f"{min_episode}:{max_episode + 1}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "trace_alignment": {
            "trace_field": TRACE_FIELD,
            "trace_key": cfg.trace_key,
            "trace_episode_offset": int(cfg.trace_episode_offset),
            "order_source": "local raw shard order" if cfg.name != "droid_56k" else "compact DROID episode_metadata order",
        },
    }


def _build_modality(features: dict[str, dict[str, Any]]) -> dict[str, Any]:
    videos = {name.removeprefix("observation.images."): {"original_key": name} for name, spec in features.items() if spec.get("dtype") == "video"}
    modality: dict[str, Any] = {"video": videos, "annotation": {"human.action.task_description": {"original_key": "task_index"}}}
    if "action" in features:
        modality["action"] = {"action": {"start": 0, "end": int(features["action"]["shape"][0]), "original_key": "action", "dtype": features["action"]["dtype"]}}
    if "observation.state" in features:
        modality["state"] = {"state": {"start": 0, "end": int(features["observation.state"]["shape"][0]), "original_key": "observation.state", "dtype": features["observation.state"]["dtype"]}}
    return modality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--rlds-root", type=Path, default=None)
    parser.add_argument("--split-name", choices=("train", "val"), default=None)
    parser.add_argument("--episode-index-offset", type=int, default=0)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("${WORK_ROOT}/tracex"))
    parser.add_argument("--trace-index-root", type=Path, default=Path("${WORK_ROOT}/trace_npy_index"))
    parser.add_argument("--droid-metadata", type=Path, default=Path("${DATA_ROOT}/droid_projection_runs/droid_56k_projection_trace_clip_dense_v1/episode_metadata.json"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--require-trace-present", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ffmpeg-bin", default="${REPO_ROOT}/tools/miniforge3/envs/oxe-convert/bin/ffmpeg")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = CONFIGS[args.dataset]
    split_name = args.split_name or cfg.split_name
    if cfg.name == "bcz" and split_name != cfg.split_name:
        cfg = DatasetConfig(
            **{
                **asdict(cfg),
                "split_name": split_name,
                "file_prefix": f"bc_z-{split_name}.tfrecord",
                "output_name": args.output_name or f"bcz_{split_name}_0.1.0_lerobot",
                "trace_key": "bcz_val" if split_name == "val" else cfg.trace_key,
                "trace_episode_offset": int(args.episode_index_offset),
            }
        )
    if args.rlds_root is not None or args.output_name is not None:
        cfg = DatasetConfig(**{**asdict(cfg), "root": args.rlds_root or cfg.root, "output_name": args.output_name or cfg.output_name})

    step_leaves, episode_leaves, raw_features = _load_features(cfg.root)
    shard_lengths = _load_shard_lengths(cfg.root, cfg.split_name)
    out_root = args.out_dir / cfg.output_name
    trace_offsets = args.trace_index_root / f"{cfg.trace_key}_offsets.npy"
    trace_coords = args.trace_index_root / f"{cfg.trace_key}_coords.npy"
    if not trace_offsets.exists() or not trace_coords.exists():
        raise FileNotFoundError(f"missing trace index for {cfg.trace_key} under {args.trace_index_root}")
    trace_present = None
    if args.require_trace_present:
        trace_present = args.trace_index_root / f"{cfg.trace_key}_present.npy"
        if not trace_present.exists():
            raise FileNotFoundError(trace_present)

    if cfg.name == "droid_56k":
        item_groups = _droid_items(cfg=cfg, metadata_path=args.droid_metadata, shard_lengths=shard_lengths, max_episodes=args.max_episodes)
    else:
        shards = _list_shards(cfg, shard_lengths)
        item_groups = _raw_items_for_shards(
            shards=shards,
            shard_lengths=shard_lengths,
            episode_offset=int(args.episode_index_offset),
            max_episodes=args.max_episodes,
        )
    group_size = max(1, math.ceil(len(item_groups) / max(1, int(args.workers))))
    jobs = [
        {
            "cfg": asdict(cfg),
            "out_root": str(out_root),
            "step_leaves": [asdict(leaf) for leaf in step_leaves],
            "episode_leaves": [asdict(leaf) for leaf in episode_leaves],
            "trace_offsets": str(trace_offsets),
            "trace_coords": str(trace_coords),
            "trace_present": str(trace_present) if trace_present else None,
            "chunk_size": int(args.chunk_size),
            "overwrite": bool(args.overwrite),
            "ffmpeg_bin": str(args.ffmpeg_bin),
            "max_episodes": args.max_episodes,
            "items": [item for group in item_groups[i : i + group_size] for item in group],
        }
        for i in range(0, len(item_groups), group_size)
    ]

    out_root.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"dataset": cfg.name, "out_root": str(out_root), "jobs": len(jobs), "workers": args.workers, "max_episodes": args.max_episodes}, indent=2))

    episodes: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    features: dict[str, dict[str, Any]] = {}
    stats: dict[str, RunningStats] = {}
    converted = 0
    skipped = 0
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(_process_records, job) for job in jobs]
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            converted += int(result["converted"])
            skipped += int(result["skipped"])
            episodes.extend(result["episodes"])
            tasks.extend(result["tasks"])
            features.update(result["features"])
            _merge_stats(result["stats"], stats)
            print(f"progress jobs {idx}/{len(jobs)} converted={converted} skipped={skipped} episodes={len(episodes)}", flush=True)

    episodes.sort(key=lambda row: int(row["episode_index"]))
    tasks.sort(key=lambda row: int(row["task_index"]))
    _write_jsonl(out_root / "meta/episodes.jsonl", episodes)
    _write_jsonl(out_root / "meta/tasks.jsonl", tasks)
    _write_json(out_root / "meta/info.json", _build_info(cfg, episodes=episodes, features=features, chunk_size=int(args.chunk_size)))
    _write_json(out_root / "meta/modality.json", _build_modality(features))
    _write_json(out_root / "meta/stats.json", _stats_json(stats, features))
    _write_json(out_root / "meta/rlds_features.json", raw_features)
    _write_json(
        out_root / "meta/conversion_summary.json",
        {
            "dataset": cfg.name,
            "source_root": str(cfg.root),
            "output_root": str(out_root),
            "episodes": len(episodes),
            "converted": converted,
            "skipped": skipped,
            "max_episodes": args.max_episodes,
            "full_preserve": True,
        },
    )
    print(f"[DONE] {out_root} episodes={len(episodes)} converted={converted} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
