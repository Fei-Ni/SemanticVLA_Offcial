"""Animation helpers for rendering fused trajectories to GIF."""

from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from PIL import Image, ImageDraw, ImageFont

_DEFAULT_SIZE = (256, 256)


def _molmo_to_pixel(value: float) -> int:
    value = max(0.0, min(100.0, value))
    return int(round(value * (_DEFAULT_SIZE[0] - 1) / 100.0))


def _ensure_parent(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def save_episode_gif(
    frames: Sequence[Image.Image],
    fused_coords: Sequence[Tuple[float, float]],
    keyframes: Iterable[int],  # TODO possibly removable
    output_path: str,
    duration: int = 100,  # in milliseconds; larger values play slower (typical: 200)
    include_label: bool = True,
    stage1_coords: Optional[Dict[int, Tuple[float, float]]] = None,
    episode_idx: Optional[int] = None,
) -> str:
    """Overlay trajectory, keyframe markers, and text labels on every frame; save the result as an animated GIF."""
    total_frames = min(len(frames), len(fused_coords))
    if total_frames == 0:
        raise ValueError("no valid frames available for GIF generation")

    trajectory_pixels: List[Tuple[int, int]] = []
    stage1_norm_map: Dict[int, Tuple[float, float]] = stage1_coords or {}
    stage1_pixel_map: Dict[int, Tuple[int, int]] = {
        frame_idx: (_molmo_to_pixel(x_val), _molmo_to_pixel(y_val))
        for frame_idx, (x_val, y_val) in stage1_norm_map.items()
    }
    stage1_order = sorted(stage1_pixel_map.keys())
    matched_stage1: Set[int] = set()
    visible_stage1: Set[int] = set()
    animation_frames: List[Image.Image] = []

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
    except Exception:  # pragma: no cover - optional dependency
        font = ImageFont.load_default()
    # iterate frame-by-frame: draw trajectory, keyframes, coordinates, labels
    for idx in range(total_frames):
        frame = frames[idx]
        # normalize all frames to the same size
        if frame.size != _DEFAULT_SIZE:
            frame = frame.resize(_DEFAULT_SIZE, Image.Resampling.LANCZOS)
        else:
            frame = frame.copy()

        draw = ImageDraw.Draw(frame)  # create a drawing object on top of this frame
        x_norm, y_norm = fused_coords[idx]  # fetch the point in coordinate space
        # convert to image pixel coordinates
        x_pix = _molmo_to_pixel(x_norm)
        y_pix = _molmo_to_pixel(y_norm)
        trajectory_pixels.append((x_pix, y_pix))  # append current point to trajectory list

        if len(trajectory_pixels) > 1:  # draw the trajectory each frame; previous points must all be drawn to make it visible
            draw.line(trajectory_pixels, fill=(255, 0, 0), width=3)

        # draw the current frame's trajectory point
        radius = 6
        draw.ellipse(
            [x_pix - radius, y_pix - radius, x_pix + radius, y_pix + radius],
            fill=(255, 255, 255),
            outline=(255, 0, 0),
            width=2,
        )

        # checkalignment between the current frame point and the Stage1 keyframe point
        if idx in stage1_norm_map:
            sx_norm, sy_norm = stage1_norm_map[idx]
            if math.hypot(sx_norm - x_norm, sy_norm - y_norm) <= 1.5:
                matched_stage1.add(idx)

        # draw keyframe markers based on alignment
        if stage1_order:
            for frame_idx in stage1_order:  # iterate Stage1 keyframes
                if frame_idx <= idx:  # any keyframe whose id is <= the current frame must appear on this frame
                    visible_stage1.add(frame_idx)
            default_color = (0, 191, 255)
            matched_color = (255, 215, 0)
            stage_radius = radius + 3
            for frame_idx in sorted(visible_stage1):  # iterate keyframes to render, colour-coded
                sx, sy = stage1_pixel_map[frame_idx]
                color = matched_color if frame_idx in matched_stage1 else default_color
                draw.ellipse(
                    [sx - stage_radius, sy - stage_radius,
                        sx + stage_radius, sy + stage_radius],
                    outline=color,
                    width=3,
                )

        # add an informational label at the top-left of each frame
        if include_label:
            padding = 3
            line_spacing = 1
            margin = 4

            ep_label = f"Episode: {episode_idx}" if episode_idx is not None else "Episode"
            labels = [ep_label, f"Frame: {idx}",
                      f"({x_norm:.1f}, {y_norm:.1f})"]
            max_width = 0
            total_height = -line_spacing
            for text in labels:
                bbox = draw.textbbox((0, 0), text, font=font)
                line_height = bbox[3] - bbox[1]
                line_width = bbox[2] - bbox[0]
                max_width = max(max_width, line_width)
                total_height += line_height + line_spacing
            box_width = max_width + padding * 2
            box_height = total_height + padding * 2
            draw.rectangle([margin, margin, margin + box_width,
                           margin + box_height], fill=(0, 0, 0, 160))
            cursor_y = margin + padding
            for idx_line, text in enumerate(labels):
                if idx_line == 2:
                    color = (255, 255, 0)
                else:
                    color = (255, 255, 255)
                draw.text((margin + padding, cursor_y),
                          text, fill=color, font=font)
                bbox = draw.textbbox((0, 0), text, font=font)
                cursor_y += (bbox[3] - bbox[1]) + line_spacing

        animation_frames.append(frame)  # append each rendered frame to the animation sequence

    _ensure_parent(output_path)
    animation_frames[0].save(
        output_path,
        save_all=True,
        append_images=animation_frames[1:],
        duration=duration,
        loop=0,
    )
    # save all frames as a single GIF
    return output_path
