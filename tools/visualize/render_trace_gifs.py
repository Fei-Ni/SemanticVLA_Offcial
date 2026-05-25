"""Render trace-overlay GIFs from TraceX v3 LeRobot datasets.

Style is borrowed from /home/u6gs/spikefly.u6gs/render_trace_viz.py (orange
gradient trail + small dots + bright current point + compact top-left label
box). Data path stays on our v3 LeRobot conversions.

Output: /projects/u6gs/spikefly.u6gs/trace_viz_gifs/<dataset>/episode_<idx>.gif
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import imageio
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

ROOT_BASE = Path("/projects/u6gs/spikefly.u6gs/datasets/TRACE240K_LEROBOT_V30")
OUT_BASE = Path("/projects/u6gs/spikefly.u6gs/trace_viz_gifs")

# One consistent style, regardless of dataset.
ORANGE = (255, 145, 28)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 230, 80)
TRAIL_LEN = 14  # number of past steps to overlay as gradient trail

DATASETS = {
    "bridge":  {"root": ROOT_BASE / "bridge_train_1.0.0_lerobot_v30",  "video_key": "observation.images.image_0",                  "fps": 5},
    "fractal": {"root": ROOT_BASE / "fractal_train_0.1.0_lerobot_v30", "video_key": "observation.images.image",                    "fps": 5},
    "bcz":     {"root": ROOT_BASE / "bcz_train_val_0.1.0_lerobot_v30", "video_key": "observation.images.image",                    "fps": 5},
    "droid":   {"root": ROOT_BASE / "droid_56k_1.0.1_lerobot_v30",     "video_key": "observation.images.exterior_image_1_left",    "fps": 15},
}

EPISODES_PER_DATASET = 10


@dataclass
class EpisodeMeta:
    episode_index: int
    length: int
    data_chunk: int
    data_file: int
    video_chunk: int
    video_file: int
    video_from_ts: float
    video_to_ts: float
    task: str
    video_key: str  # DROID overrides this per-episode (camera_for_trace)


def load_episodes_meta(root: Path, video_key: str) -> List[EpisodeMeta]:
    """Stream meta/episodes/*/* and collect per-episode info.

    For datasets with per-episode `source.camera_for_trace` (DROID), override
    `video_key` per episode so the trace overlays on the right camera.
    """
    ep_dir = root / "meta" / "episodes"
    out: List[EpisodeMeta] = []
    for chunk in sorted(p for p in ep_dir.iterdir() if p.is_dir()):
        for f in sorted(chunk.glob("*.parquet")):
            df = pq.read_table(f).to_pandas()
            for _, row in df.iterrows():
                tasks = row.get("tasks", [])
                if hasattr(tasks, "tolist"):
                    tasks = tasks.tolist()
                task = tasks[0] if tasks else ""

                src = row.get("source", None)
                cam = src.get("camera_for_trace") if isinstance(src, dict) else None
                ep_video_key = f"observation.images.{cam}" if cam else video_key
                if f"videos/{ep_video_key}/chunk_index" not in row:
                    ep_video_key = video_key

                out.append(EpisodeMeta(
                    episode_index=int(row["episode_index"]),
                    length=int(row["length"]),
                    data_chunk=int(row["data/chunk_index"]),
                    data_file=int(row["data/file_index"]),
                    video_chunk=int(row[f"videos/{ep_video_key}/chunk_index"]),
                    video_file=int(row[f"videos/{ep_video_key}/file_index"]),
                    video_from_ts=float(row[f"videos/{ep_video_key}/from_timestamp"]),
                    video_to_ts=float(row[f"videos/{ep_video_key}/to_timestamp"]),
                    task=task,
                    video_key=ep_video_key,
                ))
    out.sort(key=lambda e: e.episode_index)
    return out


def pick_episodes(metas: List[EpisodeMeta], n: int) -> List[EpisodeMeta]:
    """Pick n episodes spread across the index range, biased to lengths >= 20."""
    candidates = [m for m in metas if 20 <= m.length <= 250]
    if len(candidates) < n:
        candidates = [m for m in metas if 10 <= m.length <= 400]
    if not candidates:
        candidates = metas
    if len(candidates) <= n:
        return candidates
    step = len(candidates) / n
    return [candidates[int(i * step)] for i in range(n)]


def read_trace_for_episode(root: Path, meta: EpisodeMeta) -> Tuple[np.ndarray, np.ndarray]:
    pf = root / "data" / f"chunk-{meta.data_chunk:03d}" / f"file-{meta.data_file:03d}.parquet"
    tbl = pq.read_table(pf, columns=["episode_index", "frame_index", "trace.x", "trace.y"]).to_pandas()
    sub = tbl[tbl["episode_index"] == meta.episode_index].sort_values("frame_index")
    return sub["trace.x"].to_numpy(dtype=np.float32), sub["trace.y"].to_numpy(dtype=np.float32)


def read_video_frames(root: Path, meta: EpisodeMeta) -> List[np.ndarray]:
    vp = root / "videos" / meta.video_key / f"chunk-{meta.video_chunk:03d}" / f"file-{meta.video_file:03d}.mp4"
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {vp}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    start_frame = max(0, int(round(meta.video_from_ts * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(meta.length):
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def scaled_radius(width: int, height: int, base: float) -> int:
    scale = max(0.9, min(1.25, min(width, height) / 224.0))
    return max(2, int(round(base * scale)))


def draw_label(draw: ImageDraw.ImageDraw, width: int, height: int, lines: List[str]) -> None:
    margin = 4
    padding = 3
    line_h = 11
    text_w = min(width - margin * 2, max(60, max(len(line) for line in lines) * 6))
    box_h = padding * 2 + line_h * len(lines)
    draw.rectangle(
        (margin, margin, margin + text_w + padding * 2, margin + box_h),
        fill=(0, 0, 0, 145),
    )
    y = margin + padding
    for idx, line in enumerate(lines):
        is_last_and_long = idx == len(lines) - 1 and len(lines) > 2
        color = YELLOW + (255,) if is_last_and_long else WHITE + (255,)
        draw.text((margin + padding, y), line[:90], fill=color)
        y += line_h


def render_trace_overlay(
    frames: Sequence[np.ndarray],
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    out_size: int = 320,
    task: str = "",
    episode_idx: int = 0,
    trail_len: int = TRAIL_LEN,
) -> List[Image.Image]:
    """Draw a Bridge-style orange gradient trail + current point on every frame."""
    n = min(len(frames), len(xs), len(ys))
    if n == 0:
        return []

    h0, w0 = frames[0].shape[:2]
    scale = out_size / max(h0, w0)
    new_w = int(round(w0 * scale))
    new_h = int(round(h0 * scale))
    point_radius = scaled_radius(new_w, new_h, 6.0)
    trail_radius = max(2, point_radius - 3)

    out: List[Image.Image] = []
    trail: List[Tuple[int, int]] = []
    for i in range(n):
        img = cv2.resize(frames[i], (new_w, new_h), interpolation=cv2.INTER_AREA)
        pil = Image.fromarray(img).convert("RGB")
        draw = ImageDraw.Draw(pil, "RGBA")

        cx = max(0.0, min(100.0, float(xs[i]))) / 100.0 * (new_w - 1)
        cy = max(0.0, min(100.0, float(ys[i]))) / 100.0 * (new_h - 1)
        trail.append((int(round(cx)), int(round(cy))))
        trail = trail[-trail_len:]

        # gradient line trail (alpha grows along the trail)
        if len(trail) >= 2:
            for k in range(1, len(trail)):
                alpha = int(45 + 165 * k / max(1, len(trail) - 1))
                draw.line((trail[k - 1], trail[k]), fill=ORANGE + (alpha,), width=3)

        # gradient dot trail (skip the last point, the current marker covers it)
        for k, point in enumerate(trail[:-1]):
            alpha = int(35 + 115 * (k + 1) / max(1, len(trail) - 1))
            x, y = point
            draw.ellipse(
                (x - trail_radius, y - trail_radius, x + trail_radius, y + trail_radius),
                fill=ORANGE + (alpha,),
            )

        # current point: solid orange disc with thin black outline
        x, y = trail[-1]
        draw.ellipse(
            (x - point_radius, y - point_radius, x + point_radius, y + point_radius),
            fill=ORANGE + (235,),
            outline=BLACK + (230,),
            width=2,
        )

        # compact top-left label box
        labels = [f"ep {episode_idx}", f"step {i}/{n - 1}"]
        if task:
            labels.append(task)
        draw_label(draw, new_w, new_h, labels)

        out.append(pil)
    return out


def render_dataset(name: str, cfg: dict, out_dir: Path, n_episodes: int) -> List[dict]:
    print(f"\n=== {name} ===")
    metas = load_episodes_meta(cfg["root"], cfg["video_key"])
    print(f"  total episodes: {len(metas)}")
    chosen = pick_episodes(metas, n_episodes)
    print(f"  rendering {len(chosen)} episodes: " + ", ".join(str(m.episode_index) for m in chosen))

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for m in chosen:
        t0 = time.time()
        try:
            xs, ys = read_trace_for_episode(cfg["root"], m)
            frames = read_video_frames(cfg["root"], m)
            pil_frames = render_trace_overlay(
                frames, xs, ys,
                task=m.task,
                episode_idx=m.episode_index,
            )
            if not pil_frames:
                print(f"  ep{m.episode_index:06d}: no frames"); continue

            out_path = out_dir / f"episode_{m.episode_index:06d}.gif"
            imageio.mimsave(
                out_path,
                [np.array(f) for f in pil_frames],
                duration=1.0 / max(cfg["fps"], 1),
                loop=0,
            )
            dt = time.time() - t0
            print(f"  ✓ ep{m.episode_index:06d} ({len(pil_frames)} frames, {dt:.1f}s) — {m.task[:40]}")
            manifest.append({
                "dataset": name,
                "episode_index": m.episode_index,
                "task": m.task,
                "n_frames": len(pil_frames),
                "camera": m.video_key.split(".")[-1],
                "gif": str(out_path),
            })
        except Exception as e:
            print(f"  ✗ ep{m.episode_index:06d} failed: {type(e).__name__}: {e}")
    return manifest


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys())
    all_manifest = []
    for name in only:
        if name not in DATASETS:
            print(f"unknown dataset {name}, skipping"); continue
        all_manifest.extend(render_dataset(name, DATASETS[name], OUT_BASE / name, EPISODES_PER_DATASET))

    manifest_path = OUT_BASE / "manifest.json"
    manifest_path.write_text(json.dumps(all_manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"\nManifest -> {manifest_path}")
    print(f"Total GIFs: {len(all_manifest)}")


if __name__ == "__main__":
    main()
