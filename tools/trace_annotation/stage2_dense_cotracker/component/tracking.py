"""Tracking helpers for the refactored CoTracker pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, List

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class BidirectionalTrack:
    """Container storing per-keyframe tracking results (normalized to [0, 100])."""

    full: np.ndarray  # (T,2) T=F+B
    visibility: np.ndarray  # (T,)
    forward_frames: np.ndarray  # indices for forward segment
    forward: np.ndarray  # (F,2)
    forward_visibility: np.ndarray  # (F,)
    backward_frames: np.ndarray  # indices for backward segment
    backward: np.ndarray  # (B,2)
    backward_visibility: np.ndarray  # (B,)
    start_frame: int


def _to_pixel(value: float, size: int) -> float:
    """Convert normalized [0, 100] coordinate to pixel space given dimension size."""
    size = max(int(size), 1)
    clamped = max(0.0, min(100.0, value))
    if size == 1:
        return 0.0
    return clamped * ((size - 1) / 100.0)


def _tensor_to_normalized(coords: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Convert pixel-space coordinates to normalized [0, 100] range."""
    if coords.ndim < 2 or coords.shape[-1] != 2:
        raise ValueError("coords tensor must have shape (..., 2)")
    width = max(int(width), 1)
    height = max(int(height), 1)
    max_x = max(width - 1, 1)
    max_y = max(height - 1, 1)
    scale_x = 100.0 / max_x
    scale_y = 100.0 / max_y
    clamped = coords.clone()
    clamped[..., 0] = torch.clamp(clamped[..., 0], 0.0, float(max_x))
    clamped[..., 1] = torch.clamp(clamped[..., 1], 0.0, float(max_y))
    clamped[..., 0] = clamped[..., 0] * scale_x
    clamped[..., 1] = clamped[..., 1] * scale_y
    return clamped


def track_from_path(path: np.ndarray) -> BidirectionalTrack:
    """Wrap a precomputed trajectory into a BidirectionalTrack container."""
    path_array = np.asarray(path, dtype=np.float32)
    if path_array.ndim != 2 or path_array.shape[1] != 2:
        raise ValueError("path must have shape (T, 2)")
    length = path_array.shape[0]
    visibility = np.ones(length, dtype=np.float32)
    frames = np.arange(length, dtype=int)
    return BidirectionalTrack(
        full=path_array.copy(),
        visibility=visibility.copy(),
        forward_frames=frames.copy(),
        forward=path_array.copy(),
        forward_visibility=visibility.copy(),
        backward_frames=frames.copy(),
        backward=path_array.copy(),
        backward_visibility=visibility.copy(),
        start_frame=0,
    )


def interpolate_keyframe_path(
    total_frames: int,
    keyframes: Sequence[Tuple[int, float, float]],
) -> np.ndarray:
    """Linearly interpolate Stage1 keyframes into a dense trajectory."""
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    keyframe_list: List[Tuple[int, float, float]] = list(keyframes)
    if not keyframe_list:
        raise ValueError("keyframes are empty; cannot interpolate")
    sorted_kf = sorted(keyframe_list, key=lambda entry: entry[0])
    path = np.zeros((total_frames, 2), dtype=np.float32)

    first_frame, first_x, first_y = sorted_kf[0]
    first_frame = max(0, min(total_frames - 1, int(first_frame)))
    first_coord = np.array([first_x, first_y], dtype=np.float32)
    path[: first_frame + 1] = first_coord

    for (start_frame, start_x, start_y), (end_frame, end_x, end_y) in zip(
        sorted_kf[:-1], sorted_kf[1:]
    ):
        start_frame = max(0, min(total_frames - 1, int(start_frame)))
        end_frame = max(0, min(total_frames - 1, int(end_frame)))
        if end_frame <= start_frame:
            continue
        segment_len = end_frame - start_frame
        xs = np.linspace(start_x, end_x, segment_len + 1, dtype=np.float32)
        ys = np.linspace(start_y, end_y, segment_len + 1, dtype=np.float32)
        path[start_frame: end_frame + 1, 0] = xs
        path[start_frame: end_frame + 1, 1] = ys

    last_frame, last_x, last_y = sorted_kf[-1]
    last_frame = max(0, min(total_frames - 1, int(last_frame)))
    last_coord = np.array([last_x, last_y], dtype=np.float32)
    path[last_frame:] = last_coord
    return path


def track_keyframe_bidirectional(
    cotracker_model: torch.nn.Module,
    video_tensor: torch.Tensor,
    keyframe: Tuple[int, float, float],
    device: Optional[torch.device] = None,
    use_forward_terminal_for_backward: bool = False,
    nocut_for_backward: bool = False,
) -> BidirectionalTrack:
    """Run bidirectional tracking for a single keyframe and return CPU arrays."""
    start_frame, start_x, start_y = keyframe
    if device is None:
        device = video_tensor.device

    total_frames = video_tensor.shape[1]
    height = int(video_tensor.shape[-2])
    width = int(video_tensor.shape[-1])
    logger.debug("Start bidirectional tracking at frame %s", start_frame)

    x_img = _to_pixel(start_x, width)
    y_img = _to_pixel(start_y, height)
    query_forward = torch.tensor(
        [[start_frame, x_img, y_img]], dtype=torch.float32, device=device)

    with torch.no_grad():
        forward_tracks_raw, forward_vis_raw = cotracker_model(
            video_tensor, query_forward.unsqueeze(0))

    forward_tracks = forward_tracks_raw.squeeze(0).squeeze(1)
    forward_vis = forward_vis_raw.squeeze(0).squeeze(1)

    backward_tracks = None
    backward_vis = None
    if use_forward_terminal_for_backward and forward_tracks.shape[0] > 0:
        terminal_idx = min(total_frames - 1, forward_tracks.shape[0] - 1)
        terminal_xy = forward_tracks[terminal_idx]
        terminal_query = torch.tensor(
            [[0, terminal_xy[0].item(), terminal_xy[1].item()]],
            dtype=torch.float32,
            device=device,
        )
        video_reversed = torch.flip(video_tensor, dims=[1])
        with torch.no_grad():
            reversed_tracks_raw, reversed_vis_raw = cotracker_model(
                video_reversed, terminal_query.unsqueeze(0))
        reversed_tracks = reversed_tracks_raw.squeeze(0).squeeze(1)
        reversed_vis = reversed_vis_raw.squeeze(0).squeeze(1)
        backward_tracks = torch.flip(reversed_tracks, dims=[0])
        backward_vis = torch.flip(reversed_vis, dims=[0])
    elif start_frame > 0 or nocut_for_backward:
        if nocut_for_backward:
            video_reversed = torch.flip(video_tensor, dims=[1])
            reverse_start_idx = max(total_frames - start_frame - 1, 0)
        else:
            video_to_reverse = video_tensor[:, : start_frame + 1, :, :, :]
            video_reversed = torch.flip(video_to_reverse, dims=[1])
            reverse_start_idx = 0
        query_backward = torch.tensor(
            [[reverse_start_idx, x_img, y_img]], dtype=torch.float32, device=device)
        with torch.no_grad():
            reversed_tracks_raw, reversed_vis_raw = cotracker_model(
                video_reversed, query_backward.unsqueeze(0))
        reversed_tracks = reversed_tracks_raw.squeeze(0).squeeze(1)
        reversed_vis = reversed_vis_raw.squeeze(0).squeeze(1)
        backward_tracks = torch.flip(reversed_tracks, dims=[0])
        backward_vis = torch.flip(reversed_vis, dims=[0])

    final_tracks = torch.zeros(
        (total_frames, 2), device=device, dtype=torch.float32)
    final_visibility = torch.zeros(
        total_frames, device=device, dtype=torch.float32)

    forward_len = 0
    if start_frame < total_frames:
        forward_len = min(
            forward_tracks.shape[0] - start_frame, total_frames - start_frame)
        final_tracks[start_frame: start_frame +
                     forward_len] = forward_tracks[start_frame: start_frame + forward_len]
        final_visibility[start_frame: start_frame +
                         forward_len] = forward_vis[start_frame: start_frame + forward_len]

    backward_len = 0
    if backward_tracks is not None:
        backward_len = min(backward_tracks.shape[0], start_frame + 1)
        final_tracks[:backward_len] = backward_tracks[:backward_len]
        final_visibility[:backward_len] = backward_vis[:backward_len]
    else:
        backward_tracks = forward_tracks
        backward_vis = forward_vis
        backward_len = start_frame + 1
        final_tracks[:backward_len] = backward_tracks[:backward_len]
        final_visibility[:backward_len] = backward_vis[:backward_len]

    # Convert to normalised coordinates and move to CPU.
    full_norm = _tensor_to_normalized(
        final_tracks, width=width, height=height).cpu().numpy()
    visibility_cpu = final_visibility.cpu().numpy().astype(np.float32, copy=False)

    forward_frames = np.arange(
        start_frame, start_frame + forward_len, dtype=int)
    forward_norm = _tensor_to_normalized(
        forward_tracks[start_frame: start_frame + forward_len],
        width=width,
        height=height,
    ).cpu().numpy()
    if forward_len > 0:
        forward_vis_cpu = (
            forward_vis[start_frame: start_frame + forward_len]
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
    else:
        forward_vis_cpu = np.array([], dtype=np.float32)

    backward_frames = np.arange(0, backward_len, dtype=int)
    backward_norm = _tensor_to_normalized(
        backward_tracks[:backward_len],
        width=width,
        height=height,
    ).cpu().numpy()
    backward_vis_cpu = (
        backward_vis[:backward_len]
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )

    return BidirectionalTrack(
        full=full_norm,
        visibility=visibility_cpu,
        forward_frames=forward_frames,
        forward=forward_norm,
        forward_visibility=forward_vis_cpu,
        backward_frames=backward_frames,
        backward=backward_norm,
        backward_visibility=backward_vis_cpu,
        start_frame=start_frame,
    )
