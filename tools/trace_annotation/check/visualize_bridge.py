"""Visualize Bridge Stage2 dense-trace JSON on top of raw TFRecord frames.

Self-contained: only depends on tensorflow / pillow / numpy / matplotlib (for GIF).
Designed to run on a cluster that has the Bridge dataset TFRecords.

Usage:
    python visualize_bridge_dense_trace.py \
        --dense_json /path/to/bridge_merge_train_results.json \
        --dataset_path /path/to/bridge_dataset/1.0.0/train \
        --episode_mapping_json /path/to/episode_mapping.json \
        --output_dir bridge_viz_samples \
        --num_samples 20

Picks `--num_samples` episodes evenly spaced across the JSON, loads each
episode's frames from TFRecord, overlays the dense trajectory + stage1
keyframe anchors, and writes a GIF + trajectory.png per episode.

Schema of dense_json (from Stage2 output):
    [{"episode_idx": int, "step_idx": int,
      "coordinate": [x, y],              # 0-100 normalized
      "is_keyframe": bool,
      "stage1_coordinate": [x, y]?       # only on keyframe rows
     }, ...]

The coordinates are in the [0, 100] normalized space; we scale them back
to the actual frame H/W when drawing.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger("bridge_viz")


# ---------- TFRecord loading (Bridge schema) ----------

def _parse_example(example):
    import tensorflow as tf
    feature_description = {
        "steps/observation/image_0": tf.io.VarLenFeature(tf.string),
        "steps/language_instruction": tf.io.VarLenFeature(tf.string),
        "episode_metadata/episode_id": tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(example, feature_description)


def _build_episode_mapping(mapping_json_path: str) -> Dict[int, Tuple[str, int]]:
    with open(mapping_json_path, "r") as f:
        mapping_data = json.load(f)
    file_mapping = mapping_data.get("file_mapping", {})
    out: Dict[int, Tuple[str, int]] = {}
    for tfrecord_file, (start_ep, end_ep) in file_mapping.items():
        for ep in range(start_ep, end_ep + 1):
            out[ep] = (tfrecord_file, ep - start_ep)
    logger.info("built episode mapping: %d episodes", len(out))
    return out


_MAPPING_CACHE: Optional[Dict[int, Tuple[str, int]]] = None


def load_episode_frames(
    episode_idx: int,
    dataset_path: str,
    mapping_json_path: Optional[str],
) -> Tuple[List[Image.Image], Optional[str]]:
    """Return (frames, language_instruction_or_None) for episode_idx."""
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    global _MAPPING_CACHE
    if _MAPPING_CACHE is None and mapping_json_path and os.path.exists(mapping_json_path):
        _MAPPING_CACHE = _build_episode_mapping(mapping_json_path)
    elif _MAPPING_CACHE is None:
        _MAPPING_CACHE = {}

    if episode_idx in _MAPPING_CACHE:
        tfrecord_filename, offset = _MAPPING_CACHE[episode_idx]
        file_path = os.path.join(dataset_path, tfrecord_filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"TFRecord missing: {file_path}")
        dataset = tf.data.TFRecordDataset(file_path).map(_parse_example)
        for idx, ep in enumerate(dataset):
            if idx == offset:
                imgs = ep["steps/observation/image_0"].values.numpy()
                instr_arr = ep["steps/language_instruction"].values.numpy()
                instr = instr_arr[0].decode("utf-8") if len(instr_arr) else None
                frames = [Image.open(io.BytesIO(b)).convert("RGB") for b in imgs]
                return frames, instr
        raise ValueError(f"offset {offset} not found in {tfrecord_filename}")

    # Legacy scan path (slow). Only used if no mapping JSON is available.
    logger.warning("no mapping for episode %d, falling back to scan", episode_idx)
    files = sorted(
        f for f in os.listdir(dataset_path)
        if f.endswith(".tfrecord") or ".tfrecord-" in f
    )
    cur = 0
    for fn in files:
        ds = tf.data.TFRecordDataset(os.path.join(dataset_path, fn)).map(_parse_example)
        for ep in ds:
            if cur == episode_idx:
                imgs = ep["steps/observation/image_0"].values.numpy()
                instr_arr = ep["steps/language_instruction"].values.numpy()
                instr = instr_arr[0].decode("utf-8") if len(instr_arr) else None
                frames = [Image.open(io.BytesIO(b)).convert("RGB") for b in imgs]
                return frames, instr
            cur += 1
    raise ValueError(f"episode {episode_idx} not found under {dataset_path}")


# ---------- Dense trace JSON parsing ----------

def index_dense_json(path: str) -> Dict[int, List[Dict]]:
    """Return {episode_idx: [row, ...]} sorted by step_idx."""
    logger.info("loading dense json %s ...", path)
    with open(path, "r") as f:
        rows = json.load(f)
    by_ep: Dict[int, List[Dict]] = defaultdict(list)
    for r in rows:
        by_ep[int(r["episode_idx"])].append(r)
    for ep in by_ep.values():
        ep.sort(key=lambda r: int(r["step_idx"]))
    logger.info("dense json: %d episodes, %d rows", len(by_ep), len(rows))
    return by_ep


# ---------- Drawing ----------

_TRAIL_LEN = 8  # how many past steps to overlay as trail


def _coord_to_pixel(coord_xy: Tuple[float, float], W: int, H: int) -> Tuple[int, int]:
    """Map [0, 100] normalized (x, y) -> (px, py) in image of size WxH.

    The Stage2 pipeline writes coords in the same normalized [0,100] space as
    CoTracker output, which uses image pixel order: x is column, y is row.
    """
    x, y = coord_xy
    px = int(round(x / 100.0 * (W - 1)))
    py = int(round(y / 100.0 * (H - 1)))
    return px, py


def render_episode_gif(
    frames: List[Image.Image],
    rows: List[Dict],
    out_gif: Path,
    out_png: Path,
    instr: Optional[str] = None,
    fps: int = 8,
) -> Dict:
    """Render trajectory overlay GIF and a single summary PNG.

    Returns small stats dict (n_frames, n_keyframes, n_rows).
    """
    if not frames:
        raise ValueError("no frames")
    W, H = frames[0].size

    step_to_row = {int(r["step_idx"]): r for r in rows}
    n_kf = sum(1 for r in rows if r.get("is_keyframe"))

    overlaid = []
    trail: List[Tuple[int, int]] = []
    for step_idx, frame in enumerate(frames):
        img = frame.copy()
        draw = ImageDraw.Draw(img, "RGBA")

        row = step_to_row.get(step_idx)
        if row is not None:
            px, py = _coord_to_pixel(row["coordinate"], W, H)
            trail.append((px, py))
            if len(trail) > _TRAIL_LEN:
                trail = trail[-_TRAIL_LEN:]

            # Trail (fading green)
            for i, (tx, ty) in enumerate(trail[:-1]):
                a = int(60 + 140 * (i + 1) / max(1, len(trail) - 1))
                draw.ellipse((tx - 2, ty - 2, tx + 2, ty + 2), fill=(0, 255, 0, a))

            # Current point (green filled)
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(0, 255, 0, 230),
                         outline=(0, 0, 0, 255))

            # Stage1 keyframe anchor (red cross) if available on this row
            if row.get("is_keyframe") and "stage1_coordinate" in row:
                sx, sy = _coord_to_pixel(row["stage1_coordinate"], W, H)
                draw.line((sx - 6, sy, sx + 6, sy), fill=(255, 40, 40, 255), width=2)
                draw.line((sx, sy - 6, sx, sy + 6), fill=(255, 40, 40, 255), width=2)

        # HUD
        kf_mark = "KF" if (row and row.get("is_keyframe")) else ""
        hud = f"step {step_idx}/{len(frames)-1}  {kf_mark}"
        draw.text((4, 4), hud, fill=(255, 255, 255, 255))
        if instr:
            draw.text((4, H - 16), instr[:80], fill=(255, 255, 0, 255))

        overlaid.append(img)

    # GIF
    duration_ms = max(40, int(1000 / fps))
    overlaid[0].save(
        out_gif,
        save_all=True,
        append_images=overlaid[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )

    # Summary PNG: trajectory on the final frame
    final = frames[-1].copy()
    draw = ImageDraw.Draw(final, "RGBA")
    pts = []
    for r in rows:
        if int(r["step_idx"]) < len(frames):
            pts.append(_coord_to_pixel(r["coordinate"], W, H))
    if len(pts) >= 2:
        draw.line(pts, fill=(0, 255, 0, 200), width=2)
    for r in rows:
        if r.get("is_keyframe") and "stage1_coordinate" in r:
            sx, sy = _coord_to_pixel(r["stage1_coordinate"], W, H)
            draw.line((sx - 6, sy, sx + 6, sy), fill=(255, 40, 40, 255), width=2)
            draw.line((sx, sy - 6, sx, sy + 6), fill=(255, 40, 40, 255), width=2)
    final.save(out_png)

    return {
        "n_frames": len(frames),
        "n_rows": len(rows),
        "n_keyframes": n_kf,
    }


# ---------- Sample selection ----------

def pick_sample_episodes(by_ep: Dict[int, List[Dict]], num_samples: int, seed: int) -> List[int]:
    """Mix of strategies: 70% uniform stride across ep range, 30% length-diverse."""
    all_eps = sorted(by_ep.keys())
    n = len(all_eps)
    if n == 0:
        return []
    if num_samples >= n:
        return all_eps

    rng = np.random.default_rng(seed)
    n_uniform = int(num_samples * 0.7)
    n_diverse = num_samples - n_uniform

    # Uniform stride
    uniform_idx = np.linspace(0, n - 1, n_uniform, dtype=int)
    uniform = [all_eps[i] for i in uniform_idx]

    # Length diversity (short / medium / long)
    lengths = [(ep, len(by_ep[ep])) for ep in all_eps]
    lengths.sort(key=lambda x: x[1])
    buckets = [
        lengths[: n // 10],            # shortest 10%
        lengths[n // 2 - n // 20 : n // 2 + n // 20],  # median
        lengths[-(n // 10) :],         # longest 10%
    ]
    diverse = []
    for b in buckets:
        if not b:
            continue
        choose = rng.choice(len(b), size=min(len(b), max(1, n_diverse // 3)), replace=False)
        for c in choose:
            diverse.append(b[c][0])

    picks = sorted(set(uniform + diverse))[:num_samples]
    # Top up if we lost duplicates
    if len(picks) < num_samples:
        remaining = [e for e in all_eps if e not in picks]
        rng.shuffle(remaining)
        picks = sorted(set(picks + remaining[: num_samples - len(picks)]))
    return picks


# ---------- Main ----------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense_json", required=True,
                    help="Bridge Stage2 dense trace JSON (e.g. bridge_merge_train_results.json)")
    ap.add_argument("--dataset_path", required=True,
                    help="Bridge TFRecord dir (e.g. .../bridge_dataset/1.0.0/train)")
    ap.add_argument("--episode_mapping_json", default=None,
                    help="Optional episode_mapping.json (HUGELY speeds up loading)")
    ap.add_argument("--output_dir", default="bridge_viz_samples")
    ap.add_argument("--num_samples", type=int, default=20)
    ap.add_argument("--episodes", type=str, default=None,
                    help="Comma-separated explicit episode ids (overrides num_samples sampling)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=int, default=8)
    return ap.parse_args()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    by_ep = index_dense_json(args.dense_json)

    if args.episodes:
        episodes = [int(x) for x in args.episodes.split(",") if x.strip()]
        episodes = [e for e in episodes if e in by_ep]
    else:
        episodes = pick_sample_episodes(by_ep, args.num_samples, args.seed)

    logger.info("will render %d episodes: %s", len(episodes), episodes)

    summary = []
    for ep in episodes:
        ep_dir = out / f"episode_{ep:06d}"
        ep_dir.mkdir(exist_ok=True)
        try:
            frames, instr = load_episode_frames(ep, args.dataset_path, args.episode_mapping_json)
        except Exception as exc:
            logger.error("episode %d frame load failed: %s", ep, exc)
            continue
        rows = by_ep[ep]
        # Truncate rows to actual frame count (defensive)
        rows = [r for r in rows if int(r["step_idx"]) < len(frames)]

        try:
            stats = render_episode_gif(
                frames=frames,
                rows=rows,
                out_gif=ep_dir / "trajectory.gif",
                out_png=ep_dir / "trajectory.png",
                instr=instr,
                fps=args.fps,
            )
        except Exception as exc:
            logger.error("episode %d render failed: %s", ep, exc)
            continue
        stats["episode_idx"] = ep
        stats["instruction"] = instr
        summary.append(stats)
        logger.info("done ep=%d  frames=%d  keyframes=%d  -> %s",
                    ep, stats["n_frames"], stats["n_keyframes"], ep_dir)

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("ALL DONE. summary at %s", out / "summary.json")


if __name__ == "__main__":
    main()
