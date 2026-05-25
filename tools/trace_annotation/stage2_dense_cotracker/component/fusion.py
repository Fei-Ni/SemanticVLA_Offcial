"""Fusion logic for combining multiple candidate trajectories."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .tracking import BidirectionalTrack

logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    """Parameters and thresholds for the fusion strategy"""

    use_visibility_weight: bool = True
    use_median: bool = True
    use_mean: bool = True
    use_consistency: bool = True
    smoothing_window: int = 5  # moving average window (odd number recommended)
    min_avg_visibility: float = 0.2
    max_jump: float = 50.0
    max_avg_step: float = 10.0
    min_span_length: float = 2
    min_path_length: float = 5
    max_path_tracks: Optional[int] = None


def _apply_smoothing(positions: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or window > len(positions):
        return positions
    half = window // 2
    smoothed = np.empty_like(positions)
    for idx in range(len(positions)):
        start = max(0, idx - half)
        end = min(len(positions), idx + half + 1)
        smoothed[idx] = positions[start:end].mean(axis=0)
    return smoothed


def _filter_candidates(
    tracks: Sequence[BidirectionalTrack],
    cfg: FusionConfig,
) -> Tuple[List[BidirectionalTrack], List[Dict]]:
    if len(tracks) <= 2:
        stats = []
        for cand in tracks:
            diffs = np.diff(cand.full, axis=0)
            step_norm = np.linalg.norm(
                diffs, axis=1) if diffs.size else np.array([0.0])
            span = float(np.linalg.norm(
                cand.full[-1] - cand.full[0])) if len(cand.full) > 1 else 0.0
            path_length = float(step_norm.sum()) if step_norm.size else 0.0
            keep = not (span < cfg.min_path_length and path_length <
                        cfg.min_path_length)
            stats.append(
                {
                    "keyframe": cand.start_frame,
                    "avg_visibility": float(cand.visibility.mean()),
                    "max_jump": float(step_norm.max()) if step_norm.size else 0.0,
                    "avg_step": float(step_norm.mean()) if step_norm.size else 0.0,
                    "span": span,
                    "path_length": path_length,
                    "kept": keep,
                }
            )
        return list(tracks), stats

    filtered: List[BidirectionalTrack] = []
    stats: List[Dict] = []
    # Filter and reject the bidirectional trajectories from each keyframe anchor.
    for candidate in tracks:
        avg_vis = float(candidate.visibility.mean())
        diffs = np.diff(candidate.full, axis=0)
        step_norm = np.linalg.norm(
            diffs, axis=1) if diffs.size else np.array([0.0])
        avg_step = float(step_norm.mean()) if step_norm.size else 0.0
        max_step = float(step_norm.max()) if step_norm.size else 0.0
        span = float(np.linalg.norm(
            candidate.full[-1] - candidate.full[0])) if len(candidate.full) > 1 else 0.0
        path_length = float(step_norm.sum()) if step_norm.size else 0.0
        keep = (
            avg_vis >= cfg.min_avg_visibility
            and max_step < cfg.max_jump
            and avg_step < cfg.max_avg_step
            and not (span < cfg.min_path_length and path_length < cfg.min_path_length)
        )
        if keep:
            filtered.append(candidate)
        else:
            logger.debug(
                "rejecting candidate trajectory: keyframe=%s avg_vis=%.3f max_jump=%.3f avg_step=%.3f span=%.3f",
                candidate.start_frame,
                avg_vis,
                max_step,
                avg_step,
                span,
            )
        stats.append(
            {
                "keyframe": candidate.start_frame,
                "avg_visibility": avg_vis,
                "max_jump": max_step,
                "avg_step": avg_step,
                "span": span,
                "path_length": path_length,
                "kept": keep,
            }
        )
    if not filtered:
        return list(tracks), stats
    return filtered, stats


def fuse_candidate_tracks(
    tracks: Sequence[BidirectionalTrack],
    cfg: FusionConfig,
    max_tracks_by_path: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """Fuse multiple candidate trajectories into a single path."""
    if not tracks:
        raise ValueError("candidate trajectories are empty; cannot fuse")

    candidates, stats = _filter_candidates(tracks, cfg)
    effective_limit = max_tracks_by_path
    if effective_limit is None:
        effective_limit = cfg.max_path_tracks

    if effective_limit is not None and effective_limit > 0:
        kept_stats = [stat for stat in stats if stat["kept"]]
        if len(kept_stats) > effective_limit:
            kept_stats.sort(key=lambda entry: entry.get(
                "path_length", 0.0), reverse=True)
            allowed_keys = {entry["keyframe"]
                            for entry in kept_stats[:effective_limit]}
            if allowed_keys:
                candidates = [
                    cand for cand in candidates if cand.start_frame in allowed_keys]
                for stat in stats:
                    if stat["kept"]:
                        stat["kept"] = stat["keyframe"] in allowed_keys
            else:
                logger.warning("path-length filter empty; keeping original candidate trajectories")
    total_frames = candidates[0].full.shape[0]

    # fuse multiple trajectories per-frame along the time axis
    fused_points: List[np.ndarray] = []
    for frame_idx in range(total_frames):
        xs = np.array([cand.full[frame_idx, 0] for cand in candidates])
        ys = np.array([cand.full[frame_idx, 1] for cand in candidates])
        vis = np.array([cand.visibility[frame_idx] for cand in candidates])

        components: List[np.ndarray] = []
        weights: List[float] = []

        if cfg.use_visibility_weight:
            if vis.sum() > 1e-6:
                weighted = np.array([
                    (xs * vis).sum() / vis.sum(),
                    (ys * vis).sum() / vis.sum(),
                ])
            else:
                weighted = np.array([xs.mean(), ys.mean()])
            components.append(weighted)
            weights.append(1.0)

        if cfg.use_median:
            components.append(np.array([np.median(xs), np.median(ys)]))
            weights.append(1.0)

        if cfg.use_mean:
            components.append(np.array([xs.mean(), ys.mean()]))
            weights.append(1.0)

        if cfg.use_consistency:
            consistency = 1.0 / (1.0 + np.std(xs) + np.std(ys))
            base = np.array([xs.mean(), ys.mean()])
            components.append(base)
            weights.append(consistency)

        if not components:
            point = np.array([xs.mean(), ys.mean()])
        else:
            stacked = np.stack(components, axis=0)
            weight_array = np.array(weights).reshape(-1, 1)
            point = (stacked * weight_array).sum(axis=0) / weight_array.sum()
        fused_points.append(point)

    fused_array = np.stack(fused_points, axis=0)  # stack the point list into a single numpy array
    fused_array = _apply_smoothing(fused_array, cfg.smoothing_window)

    analysis = {
        "num_candidates": len(tracks),
        "num_used": len(candidates),
        "path_length_limit": effective_limit,
        "fusion_config": {
            "use_visibility_weight": cfg.use_visibility_weight,
            "use_median": cfg.use_median,
            "use_mean": cfg.use_mean,
            "use_consistency": cfg.use_consistency,
            "smoothing_window": cfg.smoothing_window,
            "min_path_length": cfg.min_path_length,
            "max_path_tracks": cfg.max_path_tracks,
        },
        "candidate_stats": stats,
    }
    return fused_array, analysis
