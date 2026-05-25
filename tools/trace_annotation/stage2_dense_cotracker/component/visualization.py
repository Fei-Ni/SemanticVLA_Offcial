"""Visualization utilities for the refactored CoTracker pipeline."""

from __future__ import annotations

import logging
import os
from typing import Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from .tracking import BidirectionalTrack

logger = logging.getLogger(__name__)


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _norm_to_pixel(coords: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    clipped = np.clip(coords, 0.0, 100.0)
    xs = clipped[:, 0] * (max(width - 1, 1) / 100.0)
    ys = clipped[:, 1] * (max(height - 1, 1) / 100.0)
    return xs, ys


def save_keyframe_scatter(
    keyframes: List[Tuple[int, float, float]],
    output_path: str,
) -> None:
    """Plot all keyframe coordinates on a scatter chart."""
    if not keyframes:
        logger.warning("no keyframes available for visualization: %s", output_path)
        return
    xs = [kf[1] for kf in keyframes]
    ys = [kf[2] for kf in keyframes]
    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, c="red", s=60)
    for frame, x, y in keyframes:
        plt.text(x + 0.5, y + 0.5, str(frame), fontsize=8)
    ax = plt.gca()
    ax.set_xlim(0, 100)
    ax.set_ylim(100, 0)
    ax.set_aspect("equal", adjustable="box")
    plt.title("Keyframe Scatter")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    _ensure_dir(output_path)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_candidate_paths(
    candidates: Iterable["BidirectionalTrack"],
    keyframes: List[Tuple[int, float, float]],
    output_dir: str,
) -> List[str]:
    """Plot forward/backward trajectories for each keyframe separately."""
    os.makedirs(output_dir, exist_ok=True)
    saved_paths: List[str] = []
    for idx, (track, (frame_idx, kx, ky)) in enumerate(zip(candidates, keyframes)):
        plt.figure(figsize=(6, 6))
        plt.plot(track.forward[:, 0], track.forward[:, 1],
                 color="blue", linewidth=2, label="forward")
        plt.plot(track.backward[:, 0], track.backward[:, 1],
                 color="orange", linestyle="--", linewidth=2, label="backward")
        plt.scatter([kx], [ky], c="red", s=60, label="keyframe")
        ax = plt.gca()
        ax.set_xlim(0, 100)
        ax.set_ylim(100, 0)
        ax.set_aspect("equal", adjustable="box")
        plt.title(f"Keyframe {frame_idx}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right")
        file_path = os.path.join(output_dir, f"keyframe_{frame_idx:04d}.png")
        _ensure_dir(file_path)
        plt.savefig(file_path, dpi=200, bbox_inches="tight")
        plt.close()
        saved_paths.append(file_path)
    return saved_paths


def save_fused_trajectory(
    fused: np.ndarray,
    keyframes: List[Tuple[int, float, float]],
    output_path: str,
) -> None:
    """Plot final fused trajectory with keyframe markers."""
    plt.figure(figsize=(6, 6))
    plt.plot(fused[:, 0], fused[:, 1], color="dodgerblue",
             linewidth=2, label="Fused")
    if keyframes:
        plt.scatter([kf[1] for kf in keyframes], [kf[2]
                    for kf in keyframes], c="red", s=50, label="Keyframes")
    ax = plt.gca()
    ax.set_xlim(0, 100)
    ax.set_ylim(100, 0)
    ax.set_aspect("equal", adjustable="box")
    plt.title("Fused Trajectory")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _ensure_dir(output_path)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_candidate_frame_scatters(
    frames: Sequence[Image.Image],
    candidates: Sequence["BidirectionalTrack"],
    keyframes: List[Tuple[int, float, float]],
    output_dir: str,
    kept_keyframes: Optional[Set[int]] = None,
) -> List[str]:
    """Overlay candidate positions at each keyframe on the corresponding frame image."""
    os.makedirs(output_dir, exist_ok=True)
    if not keyframes:
        logger.warning("no keyframes available for scatter rendering: %s", output_dir)
        return []
    cmap = plt.cm.get_cmap("tab10", max(len(candidates), 1))
    saved_paths: List[str] = []

    for key_idx, (frame_idx, kx, ky) in enumerate(keyframes):
        if frame_idx >= len(frames):
            logger.warning("keyframe %s exceeds frame list length %s; skipping scatter plot", frame_idx, len(frames))
            continue

        frame_img = frames[frame_idx]
        if not isinstance(frame_img, Image.Image):
            frame_img = Image.fromarray(np.asarray(frame_img))
        frame_rgb = frame_img.convert("RGB")
        width, height = frame_rgb.size
        frame_array = np.asarray(frame_rgb)

        plt.figure(figsize=(6, 6))
        ax = plt.gca()
        ax.imshow(frame_array)

        for cand_idx, track in enumerate(candidates):
            if kept_keyframes is not None and track.start_frame not in kept_keyframes:
                continue
            if frame_idx >= track.full.shape[0]:
                logger.debug("candidate trajectory %s on frame %s is missing the full trajectory length",
                             track.start_frame, frame_idx)
                continue
            coord = track.full[frame_idx: frame_idx + 1]
            xs, ys = _norm_to_pixel(coord, width, height)
            ax.scatter(
                xs,
                ys,
                s=20,
                color=cmap(cand_idx),
                alpha=0.8,
                label=f"Cand {track.start_frame}",
            )

        ax.set_title(f"Keyframe {frame_idx}")
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis("off")

        handles, labels = ax.get_legend_handles_labels()
        if labels:
            unique = {}
            for handle, label in zip(handles, labels):
                if label not in unique:
                    unique[label] = handle
            ax.legend(unique.values(), unique.keys(), loc="upper right")

        scatter_path = os.path.join(
            output_dir, f"keyframe_{frame_idx:04d}.png")
        _ensure_dir(scatter_path)
        plt.savefig(scatter_path, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close()
        saved_paths.append(scatter_path)

    if saved_paths:
        logger.info("keyframe scatter plot written: %s", output_dir)
    else:
        logger.warning("no keyframe scatter plot generated: %s", output_dir)
    return saved_paths
