"""Modular pipeline for CoTracker v4.2 refactor."""

from .model import load_cotracker_model
from .data import load_keyframe_json, extract_episode_keyframes, load_episode_frames
from .tracking import (
    BidirectionalTrack,
    track_keyframe_bidirectional,
    track_from_path,
    interpolate_keyframe_path,
)
from .fusion import FusionConfig, fuse_candidate_tracks
from .animation import save_episode_gif
from .visualization import (
    save_keyframe_scatter,
    save_candidate_paths,
    save_fused_trajectory,
    save_candidate_frame_scatters,
)

__all__ = [
    "load_cotracker_model",
    "load_keyframe_json",
    "extract_episode_keyframes",
    "load_episode_frames",
    "BidirectionalTrack",
    "track_keyframe_bidirectional",
    "track_from_path",
    "interpolate_keyframe_path",
    "FusionConfig",
    "fuse_candidate_tracks",
    "save_episode_gif",
    "save_keyframe_scatter",
    "save_candidate_paths",
    "save_fused_trajectory",
    "save_candidate_frame_scatters",
]
