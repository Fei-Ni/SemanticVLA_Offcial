"""Fractal-specific data loading helpers for the Stage2 dense trace pipeline."""

from __future__ import annotations

import bisect
import io
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import tensorflow as tf
from PIL import Image

logger = logging.getLogger(__name__)

_DATASET_INDEX_CACHE: Dict[str, Tuple[List[str], Optional[List[int]], Optional[List[int]]]] = {}


@dataclass
class EpisodeData:
    """Container for per-episode frames and metadata."""

    frames: List[Image.Image]
    keyframes: List[Tuple[int, float, float]]
    episode_idx: int


def _disable_tf_gpu() -> None:
    """Prevent TensorFlow from reserving GPU memory during TFRecord reads."""
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:  # pragma: no cover - depends on runtime env
        pass


def _parse_tfrecord_example(example: tf.Tensor) -> Dict[str, tf.Tensor]:
    """Parse a raw TFRecord example produced by the Fractal dataset."""
    feature_description = {
        "steps/observation/image": tf.io.VarLenFeature(tf.string),
        "steps/language_instruction": tf.io.VarLenFeature(tf.string),
    }
    return tf.io.parse_single_example(example, feature_description)


def load_keyframe_json(path: str) -> List[Dict]:
    """Load Stage1 sparse keyframe annotations."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    logger.info("loaded keyframe entries %d", len(data))
    return data


def extract_episode_keyframes(
    episode_items: Sequence[Dict],
    max_keyframes: int,
) -> List[Tuple[int, float, float]]:
    """Extract sorted keyframes for one episode from Stage1 annotations."""
    keyframes: List[Tuple[int, float, float]] = []
    seen = set()
    for item in episode_items:
        if not item.get("is_keyframe", False):
            continue
        coords = item.get("molmo_coords") or item.get("coordinate") or [0.0, 0.0]
        step_idx = int(item["step_idx"])
        if len(coords) < 2 or step_idx in seen:
            continue
        keyframes.append((step_idx, float(coords[0]), float(coords[1])))
        seen.add(step_idx)
        if len(keyframes) >= max_keyframes:
            break
    keyframes.sort(key=lambda entry: entry[0])
    return keyframes


def load_shard_lengths_from_dataset_info(
    dataset_path: str,
    expected_num_shards: int,
) -> Optional[List[int]]:
    """Read per-shard episode counts from Fractal dataset_info.json."""
    info_path = os.path.join(dataset_path, "dataset_info.json")
    if not os.path.exists(info_path):
        return None

    try:
        with open(info_path, "r", encoding="utf-8") as handle:
            info = json.load(handle)
    except Exception as exc:
        logger.warning("failed to read dataset_info.json: %s", exc)
        return None

    splits = info.get("splits", [])
    if not isinstance(splits, list):
        return None

    train_split = None
    for split in splits:
        if split.get("name") == "train":
            train_split = split
            break
    if train_split is None and len(splits) == 1:
        train_split = splits[0]
    if train_split is None:
        return None

    raw_lengths = train_split.get("shardLengths")
    if not isinstance(raw_lengths, list):
        return None
    if len(raw_lengths) != expected_num_shards:
        logger.warning(
            "dataset_info shardLengths length (%d) does not match tfrecord count (%d); falling back to sequential scan",
            len(raw_lengths),
            expected_num_shards,
        )
        return None

    try:
        return [int(x) for x in raw_lengths]
    except Exception as exc:
        logger.warning("failed to parse shardLengths: %s", exc)
        return None


def _build_dataset_index(
    dataset_path: str,
) -> Tuple[List[str], Optional[List[int]], Optional[List[int]]]:
    tfrecord_files = sorted(
        [
            name
            for name in os.listdir(dataset_path)
            if name.endswith(".tfrecord") or ".tfrecord-" in name
        ]
    )
    if not tfrecord_files:
        raise FileNotFoundError(f"no TFRecord files found under {dataset_path}")

    shard_lengths = load_shard_lengths_from_dataset_info(dataset_path, len(tfrecord_files))
    shard_prefix: Optional[List[int]] = None
    if shard_lengths is not None:
        shard_prefix = [0]
        for length in shard_lengths:
            shard_prefix.append(shard_prefix[-1] + length)
        logger.info("building Fractal index from dataset_info.shardLengths")
    else:
        logger.warning("no usable dataset_info shardLengths; falling back to a slower sequential scan")

    return tfrecord_files, shard_lengths, shard_prefix


def _get_dataset_index(
    dataset_path: str,
) -> Tuple[List[str], Optional[List[int]], Optional[List[int]]]:
    cached = _DATASET_INDEX_CACHE.get(dataset_path)
    if cached is None:
        cached = _build_dataset_index(dataset_path)
        _DATASET_INDEX_CACHE[dataset_path] = cached
    return cached


def _resolve_episode_location(
    episode_idx: int,
    dataset_path: str,
) -> Tuple[str, int, int]:
    tfrecord_files, _shard_lengths, shard_prefix = _get_dataset_index(dataset_path)

    if shard_prefix is not None:
        file_idx = bisect.bisect_right(shard_prefix, episode_idx) - 1
        if (
            0 <= file_idx < len(tfrecord_files)
            and shard_prefix[file_idx] <= episode_idx < shard_prefix[file_idx + 1]
        ):
            local_episode_idx = episode_idx - shard_prefix[file_idx]
            return tfrecord_files[file_idx], file_idx, local_episode_idx
        raise ValueError(f"Episode {episode_idx} is out of the range defined in dataset_info")

    current_episode = 0
    for file_idx, tfrecord_name in enumerate(tfrecord_files):
        file_path = os.path.join(dataset_path, tfrecord_name)
        raw_dataset = tf.data.TFRecordDataset(file_path)
        file_episodes = sum(1 for _ in raw_dataset)
        if current_episode + file_episodes > episode_idx:
            local_episode_idx = episode_idx - current_episode
            return tfrecord_name, file_idx, local_episode_idx
        current_episode += file_episodes

    raise ValueError(f"Episode {episode_idx} not found in {dataset_path}")


def load_episode_frames(episode_idx: int, dataset_path: str) -> List[Image.Image]:
    """Load all RGB frames for a Fractal episode, directly locating the target shard when possible."""
    _disable_tf_gpu()

    tfrecord_name, file_idx, local_episode_idx = _resolve_episode_location(episode_idx, dataset_path)
    file_path = os.path.join(dataset_path, tfrecord_name)
    logger.info(
        "📂 Episode %d → tfrecord=%d local_episode=%d file=%s",
        episode_idx,
        file_idx,
        local_episode_idx,
        tfrecord_name,
    )

    dataset = tf.data.TFRecordDataset(file_path)
    dataset = dataset.map(_parse_tfrecord_example)

    for idx, episode in enumerate(dataset):
        if idx != local_episode_idx:
            continue

        image_values = episode["steps/observation/image"].values.numpy()
        frames: List[Image.Image] = []
        for img_bytes in image_values:
            try:
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                frames.append(image)
            except Exception as exc:
                logger.warning("failed to decode images for episode %d: %s", episode_idx, exc)

        logger.info("✅ loaded %d frames for episode %d from %s", episode_idx, len(frames), tfrecord_name)
        return frames

    raise ValueError(
        f"Episode {episode_idx} not found at offset {local_episode_idx} of {tfrecord_name}"
    )
