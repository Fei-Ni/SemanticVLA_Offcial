"""Load Fractal trace annotation and match it to the original Fractal dataset.

This module is intentionally self-contained: it only needs TensorFlow (for
reading the raw TFRecords) and a copy of the Open X-Embodiment / RT-1 Fractal
dataset (`fractal20220817_data/0.1.0`).

Two annotation files are produced by the pipeline (both shipped in
`../annotations/`):

  * `fractal_stage1_keyframes.json` — sparse Molmo-72B keyframe labels.
    Schema: list[dict] with keys
        episode_idx, step_idx, generated_text, molmo_coords, is_keyframe
    (~10 keyframes per episode, ~858k rows total, 87k episodes)

  * `fractal_stage2_dense_trace.json` — dense CoTracker trace, every step
    of every episode. Schema: list[dict] with keys
        episode_idx, step_idx, coordinate, molmo_coords, is_keyframe,
        is_interpolated, stage1_coordinate
    (~3.79M rows total, 87,182 episodes).

`coordinate` is in normalized image-space units (0–100), so multiply by
W/100 and H/100 to get pixel coords on the raw 320x256 Fractal RGB frame.

Usage:
    from load_annotation import (
        load_stage2, group_by_episode,
        resolve_episode, load_episode_frames,
    )

    rows = load_stage2("../annotations/fractal_stage2_dense_trace.json")
    by_ep = group_by_episode(rows)         # dict[int, list[dict]]
    fn, offset = resolve_episode(0, dataset_root)
    frames, instr = load_episode_frames(0, dataset_root)
"""
from __future__ import annotations

import io
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ---------- annotation IO ----------

def load_stage1(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def load_stage2(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def group_by_episode(rows: List[Dict]) -> Dict[int, List[Dict]]:
    """Group a flat row list by episode_idx, sorted by step_idx within."""
    out: Dict[int, List[Dict]] = defaultdict(list)
    for r in rows:
        out[int(r["episode_idx"])].append(r)
    for ep in out.values():
        ep.sort(key=lambda r: int(r["step_idx"]))
    return dict(out)


# ---------- episode -> tfrecord resolver ----------
#
# Fractal `dataset_info.json` lists `shardLengths` for the train split:
# a 2048-long array where shardLengths[i] = number of episodes in
# tfrecord file i. To find the file containing episode E, walk the
# cumulative sum until you cross E.

_SHARD_LENS_CACHE: Optional[List[int]] = None


def _shard_lengths(dataset_root: str) -> List[int]:
    global _SHARD_LENS_CACHE
    if _SHARD_LENS_CACHE is not None:
        return _SHARD_LENS_CACHE
    info_path = os.path.join(dataset_root, "dataset_info.json")
    with open(info_path) as f:
        info = json.load(f)
    for s in info["splits"]:
        if s["name"] == "train":
            _SHARD_LENS_CACHE = [int(x) for x in s["shardLengths"]]
            return _SHARD_LENS_CACHE
    raise RuntimeError("no train split in dataset_info.json")


def _train_tfrecord_files(dataset_root: str) -> List[str]:
    """Return sorted list of train tfrecord filenames (no path).

    Fractal has only train shards in this dir so all .tfrecord files match,
    but we follow the same conservative filter we use for BC-Z (which mixes
    train and val in one directory).
    """
    names = []
    for fn in os.listdir(dataset_root):
        if not (fn.endswith(".tfrecord") or ".tfrecord-" in fn):
            continue
        if any(tok in fn for tok in ("-val.", "-eval.", "-test.")):
            continue
        names.append(fn)
    return sorted(names)


def resolve_episode(episode_idx: int, dataset_root: str) -> Tuple[str, int]:
    """Return (tfrecord_filename, offset_within_file) for a given episode.

    Args:
        episode_idx: global episode index, 0..N-1 where N=sum(shardLengths)
        dataset_root: path that contains fractal20220817_data tfrecords
                      and dataset_info.json
    """
    shard_lens = _shard_lengths(dataset_root)
    files = _train_tfrecord_files(dataset_root)
    if len(files) != len(shard_lens):
        raise RuntimeError(
            f"file count {len(files)} != shardLengths len {len(shard_lens)}; "
            "did you point at the right dataset version?"
        )
    cum = 0
    for i, L in enumerate(shard_lens):
        if episode_idx < cum + L:
            return files[i], episode_idx - cum
        cum += L
    raise ValueError(f"episode {episode_idx} out of range (total {cum})")


# ---------- read raw frames from one tfrecord episode ----------

def load_episode_frames(
    episode_idx: int,
    dataset_root: str,
) -> Tuple[List["PIL.Image.Image"], Optional[str]]:
    """Decode one episode's RGB frames + natural language instruction.

    Returns (frames, instruction). Frames are PIL.Image, RGB, 320x256.
    """
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    from PIL import Image

    fn, offset = resolve_episode(episode_idx, dataset_root)
    path = os.path.join(dataset_root, fn)
    feature_spec = {
        "steps/observation/image": tf.io.VarLenFeature(tf.string),
        "steps/observation/natural_language_instruction":
            tf.io.VarLenFeature(tf.string),
    }
    ds = tf.data.TFRecordDataset(path).map(
        lambda x: tf.io.parse_single_example(x, feature_spec)
    )
    for idx, ep in enumerate(ds):
        if idx != offset:
            continue
        imgs = ep["steps/observation/image"].values.numpy()
        instrs = ep["steps/observation/natural_language_instruction"].values.numpy()
        instr = instrs[0].decode("utf-8") if len(instrs) else None
        frames = [Image.open(io.BytesIO(b)).convert("RGB") for b in imgs]
        return frames, instr
    raise ValueError(f"ep {episode_idx} offset {offset} not in {fn}")


def coord_to_pixel(coord_xy, width: int, height: int) -> Tuple[int, int]:
    """Convert a normalized (0..100) coordinate to integer pixel (x, y)."""
    x, y = coord_xy
    px = int(round(x / 100.0 * (width - 1)))
    py = int(round(y / 100.0 * (height - 1)))
    return px, py


if __name__ == "__main__":
    # Smoke check: print first episode's first 3 dense-trace rows.
    import sys
    if len(sys.argv) < 3:
        print(
            "usage: python load_annotation.py "
            "<dataset_root> <fractal_stage2_dense_trace.json> [episode_idx]"
        )
        sys.exit(2)
    root = sys.argv[1]
    ann = sys.argv[2]
    ep = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    fn, off = resolve_episode(ep, root)
    print(f"ep {ep} -> file '{fn}' offset {off}")

    rows = load_stage2(ann)
    by_ep = group_by_episode(rows)
    if ep not in by_ep:
        print(f"!! ep {ep} not in annotation (was it dropped due to <2 keyframes?)")
        sys.exit(1)
    print(f"ep {ep}: {len(by_ep[ep])} rows")
    for r in by_ep[ep][:3]:
        print(" ", r)
