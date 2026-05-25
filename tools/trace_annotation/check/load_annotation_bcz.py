"""Load BC-Z trace annotation and match it to the original BC-Z dataset."""
from __future__ import annotations

import io
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def load_stage1(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def load_stage2(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def group_by_episode(rows: List[Dict]) -> Dict[int, List[Dict]]:
    out: Dict[int, List[Dict]] = defaultdict(list)
    for r in rows:
        out[int(r["episode_idx"])].append(r)
    for ep_rows in out.values():
        ep_rows.sort(key=lambda r: int(r["step_idx"]))
    return dict(out)


_SHARD_LENS_CACHE: Optional[List[int]] = None


def _shard_lengths(dataset_root: str) -> List[int]:
    global _SHARD_LENS_CACHE
    if _SHARD_LENS_CACHE is not None:
        return _SHARD_LENS_CACHE
    with open(os.path.join(dataset_root, "dataset_info.json")) as f:
        info = json.load(f)
    for split in info["splits"]:
        if split["name"] == "train":
            _SHARD_LENS_CACHE = [int(x) for x in split["shardLengths"]]
            return _SHARD_LENS_CACHE
    raise RuntimeError("no train split in dataset_info.json")


def _train_tfrecord_files(dataset_root: str) -> List[str]:
    names = []
    for fn in os.listdir(dataset_root):
        if not (fn.endswith(".tfrecord") or ".tfrecord-" in fn):
            continue
        if "bc_z-train.tfrecord" in fn or "-train." in fn or "-train-" in fn:
            names.append(fn)
    return sorted(names)


def resolve_episode(episode_idx: int, dataset_root: str) -> Tuple[str, int]:
    shard_lens = _shard_lengths(dataset_root)
    files = _train_tfrecord_files(dataset_root)
    if len(files) != len(shard_lens):
        raise RuntimeError(
            f"file count {len(files)} != shardLengths len {len(shard_lens)}; "
            "check that val shards were not included"
        )
    cum = 0
    for i, length in enumerate(shard_lens):
        if episode_idx < cum + length:
            return files[i], episode_idx - cum
        cum += length
    raise ValueError(f"episode {episode_idx} out of range (total {cum})")


def load_episode_frames(episode_idx: int, dataset_root: str):
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    from PIL import Image

    fn, offset = resolve_episode(episode_idx, dataset_root)
    feature_spec = {
        "steps/observation/image": tf.io.VarLenFeature(tf.string),
        "steps/observation/natural_language_instruction": tf.io.VarLenFeature(tf.string),
        "steps/language_instruction": tf.io.VarLenFeature(tf.string),
    }
    ds = tf.data.TFRecordDataset(os.path.join(dataset_root, fn)).map(
        lambda x: tf.io.parse_single_example(x, feature_spec)
    )
    for idx, ep in enumerate(ds):
        if idx != offset:
            continue
        imgs = ep["steps/observation/image"].values.numpy()
        instr_values = ep["steps/observation/natural_language_instruction"].values.numpy()
        if len(instr_values) == 0:
            instr_values = ep["steps/language_instruction"].values.numpy()
        instr = instr_values[0].decode("utf-8") if len(instr_values) else None
        frames = [Image.open(io.BytesIO(b)).convert("RGB") for b in imgs]
        return frames, instr
    raise ValueError(f"ep {episode_idx} offset {offset} not in {fn}")


def coord_to_pixel(coord_xy, width: int, height: int) -> Tuple[int, int]:
    x, y = coord_xy
    return int(round(x / 100.0 * (width - 1))), int(round(y / 100.0 * (height - 1)))


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: python load_annotation.py <dataset_root> <bcz_stage2_dense_trace.json> [episode_idx]")
        sys.exit(2)
    root = sys.argv[1]
    ann = sys.argv[2]
    ep = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    fn, off = resolve_episode(ep, root)
    print(f"ep {ep} -> file '{fn}' offset {off}")
    rows = load_stage2(ann)
    by_ep = group_by_episode(rows)
    if ep not in by_ep:
        print(f"!! ep {ep} not in annotation")
        sys.exit(1)
    print(f"ep {ep}: {len(by_ep[ep])} rows")
    for r in by_ep[ep][:3]:
        print(" ", r)
