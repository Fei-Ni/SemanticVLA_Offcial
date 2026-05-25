"""Sanity-check Fractal trace annotation against raw frames.

For each requested episode, this script:
  1. Loads raw RGB frames from the Fractal TFRecord.
  2. Loads either Stage1 keyframe coords or Stage2 dense-trace coords.
  3. Overlays a marker on every frame:
       - red cross for keyframe (and a small filled circle)
       - thin yellow dot for interpolated step
  4. Saves one GIF per episode + per-keyframe PNG snapshots.

Usage:
    python visualize_check.py \
        --dataset_root /path/to/fractal20220817_data/0.1.0 \
        --annotation ../annotations/fractal_stage2_dense_trace.json \
        --output_dir viz_out \
        --episodes 0,100,500,5000,87000

You can pass a Stage1 keyframe JSON to `--annotation` as well — the script
detects which schema you gave it.

Requires: tensorflow (CPU is fine), Pillow.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageDraw

# Allow direct execution from this directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_annotation import (
    coord_to_pixel,
    group_by_episode,
    load_episode_frames,
    load_stage2,
)

logger = logging.getLogger("fractal_viz_check")


def _row_coord(row: Dict):
    """Pick the best coordinate field: prefer dense `coordinate`, else `molmo_coords`."""
    if "coordinate" in row and row["coordinate"] is not None:
        return row["coordinate"]
    return row.get("molmo_coords")


def render_episode(
    frames: List[Image.Image],
    rows: List[Dict],
    out_dir: Path,
    instr: Optional[str],
    fps: int,
) -> Dict:
    if not frames:
        raise ValueError("no frames")
    W, H = frames[0].size
    by_step = {int(r["step_idx"]): r for r in rows}

    out_frames = []
    for step_idx, frame in enumerate(frames):
        img = frame.copy()
        draw = ImageDraw.Draw(img, "RGBA")
        r = by_step.get(step_idx)
        if r is not None:
            coord = _row_coord(r)
            if coord is not None:
                px, py = coord_to_pixel(coord, W, H)
                is_kf = bool(r.get("is_keyframe", False))
                if is_kf:
                    L = 4
                    draw.line((px - L, py, px + L, py), fill=(255, 40, 40, 255), width=2)
                    draw.line((px, py - L, px, py + L), fill=(255, 40, 40, 255), width=2)
                    draw.ellipse((px - 2, py - 2, px + 2, py + 2),
                                 outline=(255, 40, 40, 255), width=1)
                else:
                    draw.ellipse((px - 2, py - 2, px + 2, py + 2),
                                 fill=(255, 230, 0, 200))
        label = f"step {step_idx}/{len(frames)-1}"
        if step_idx in by_step and by_step[step_idx].get("is_keyframe"):
            label += " KF"
        draw.text((4, 4), label, fill=(255, 255, 255, 255))
        if instr:
            draw.text((4, H - 16), instr[:80], fill=(255, 255, 0, 255))
        out_frames.append(img)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_frames[0].save(
        out_dir / "trajectory.gif",
        save_all=True,
        append_images=out_frames[1:],
        duration=max(40, int(1000 / fps)),
        loop=0,
    )
    for step_idx, r in by_step.items():
        if r.get("is_keyframe") and step_idx < len(out_frames):
            out_frames[step_idx].save(out_dir / f"keyframe_{step_idx:04d}.png")

    n_kf = sum(1 for r in rows if r.get("is_keyframe"))
    return {"n_frames": len(frames), "n_rows": len(rows), "n_keyframes": n_kf}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True,
                    help="Path to fractal20220817_data/0.1.0 (with dataset_info.json)")
    ap.add_argument("--annotation", required=True,
                    help="Either fractal_stage1_keyframes.json or "
                         "fractal_stage2_dense_trace.json")
    ap.add_argument("--output_dir", default="fractal_viz_check")
    ap.add_argument("--episodes", default="0,100,500,5000,87000",
                    help="Comma-separated episode ids")
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    logger.info("loading annotation: %s", args.annotation)
    rows = load_stage2(args.annotation)
    by_ep = group_by_episode(rows)
    logger.info("annotation: %d rows across %d episodes", len(rows), len(by_ep))

    eps = [int(x) for x in args.episodes.split(",") if x.strip()]
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for ep in eps:
        if ep not in by_ep:
            logger.warning("episode %d not in annotation, skipping", ep)
            continue
        try:
            frames, instr = load_episode_frames(ep, args.dataset_root)
        except Exception as exc:
            logger.error("could not load frames for ep %d: %s", ep, exc)
            continue
        ep_out = out_root / f"episode_{ep:06d}"
        info = render_episode(frames, by_ep[ep], ep_out, instr, args.fps)
        info["episode_idx"] = ep
        info["instruction"] = instr
        summary.append(info)
        logger.info("done ep=%d frames=%d rows=%d kfs=%d -> %s",
                    ep, info["n_frames"], info["n_rows"],
                    info["n_keyframes"], ep_out)

    import json
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("ALL DONE -> %s", out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
