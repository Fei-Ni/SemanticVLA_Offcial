#!/usr/bin/env python3
"""Refactored CoTracker v4.2 pipeline entry point."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from component import (
    BidirectionalTrack,
    FusionConfig,
    extract_episode_keyframes,
    fuse_candidate_tracks,
    interpolate_keyframe_path,
    load_cotracker_model,
    load_episode_frames,
    load_keyframe_json,
    save_episode_gif,
    save_candidate_frame_scatters,
    save_candidate_paths,
    save_fused_trajectory,
    save_keyframe_scatter,
    track_from_path,
    track_keyframe_bidirectional,
)

HANDOFF_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_KEYFRAME_JSON = str(HANDOFF_ROOT / "annotations" / "bcz_stage1_keyframes.json")
DEFAULT_DATASET_PATH = os.environ.get("BCZ_DATASET_PATH", "/home/n84416302/dataset/bc_z/0.1.0")
DEFAULT_OUTPUT_JSON_NAME = "complete_trajectories_bcz_dense_trace.json"

logger = logging.getLogger("run_bcz_dense_trace")
# shared module-level logger (avoid the root logger because tqdm / torch may attach their own handlers)


def _prepare_video_tensor(frames: List[np.ndarray], device: torch.device) -> torch.Tensor:
    """
    Convert a frame list into the tensor format expected by CoTracker.
    """
    processed = []
    for frame in frames:
        arr = np.array(frame)  # convert each frame to numpy
        arr = cv2.resize(arr, (256, 256))  # normalise each frame size
        processed.append(arr)  # append the processed frame to the buffer
    # stack frames along a new axis to get (T, H, W, C)
    stacked = np.stack(processed, axis=0)
    tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).float() / 255.0
    # convert numpy to torch tensor with shape (T, C, H, W), float32, normalised
    tensor = tensor.unsqueeze(0)
    # add an outer batch dim so the tensor becomes (1, T, C, H, W); the outermost dim represents the full video
    return tensor.to(device)  # move the tensor to the target device


def _build_episode_entries(
    episode_idx: int,
    fused: np.ndarray,
    keyframes: List[Tuple[int, float, float]],
    fusion_analysis: Dict,
) -> List[Dict]:
    """
    Assemble the trajectory sequence into a structured JSON list.
    """
    keyframe_map = {frame: (x, y) for frame, x, y in keyframes}
    # build a dict from keyframes to their coordinates
    entries: List[Dict] = []
    # store the structured value of every frame
    for step_idx, (x, y) in enumerate(fused):  # iterate the fused trajectory: frame_id from the enumerate, (x, y) from fused
        is_keyframe = step_idx in keyframe_map
        entry = {
            "episode_idx": episode_idx,
            "step_idx": step_idx,
            "coordinate": [float(x), float(y)],
            "molmo_coords": [float(x), float(y)],
            "is_keyframe": is_keyframe,
            "is_interpolated": not is_keyframe,
        }
        if is_keyframe:  # if this is a keyframe, add an extra key
            base_x, base_y = keyframe_map[step_idx]
            entry["stage1_coordinate"] = [base_x, base_y]
        entries.append(entry)  # append the structured entry to the list
    return entries


def _apply_keyframe_anchors(
    fused: np.ndarray,
    keyframes: List[Tuple[int, float, float]],
) -> np.ndarray:
    """Force keyframe steps to exactly match Stage1 anchor coordinates."""
    anchored = np.asarray(fused, dtype=np.float32).copy()
    total_frames = anchored.shape[0]
    for frame_idx, x, y in keyframes:
        if 0 <= frame_idx < total_frames:
            anchored[frame_idx, 0] = float(x)
            anchored[frame_idx, 1] = float(y)
    return anchored


def _write_partial_results(entries: List[Dict], output_path: str) -> None:
    """Safe JSON write using a temp-file swap to avoid corrupting the output."""
    temp_path = output_path + ".tmp"  # build a temp file path
    with open(temp_path, "w", encoding="utf-8") as handle:
        # open the temp file (with-block auto-handles enter/exit)
        json.dump(entries, handle, ensure_ascii=False, indent=2)
        # dump entries with ensure_ascii=False and indent=2
    os.replace(temp_path, output_path)
    # atomic rename: even if this fails, the old file is not corrupted


def _append_to_json(new_entries: List[Dict], output_path: str) -> int:
    """
    Append entries to a JSON file incrementally to avoid memory growth.
    
    Efficient append strategy:
    - if the file is missing or small, read-append-write directly
    - if the file is large, append to the array tail via a file pointer
    
    Args:
        new_entries: new entries to append
        output_path: JSON file path
        
    Returns:
        total entry count after appending
    """
    # if the file does not exist, create it
    if not os.path.exists(output_path):
        temp_path = output_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(new_entries, handle, ensure_ascii=False, indent=2)
        os.replace(temp_path, output_path)
        return len(new_entries)
    
    # read the existing data (kept temporarily so the merge is correct)
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            existing_entries = json.load(f)
    except (json.JSONDecodeError, IOError):
        # if the file is corrupted, start fresh
        existing_entries = []
    
    existing_entries.extend(new_entries)
    total_count = len(existing_entries)
    
    temp_path = output_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(existing_entries, handle, ensure_ascii=False, indent=2)
    os.replace(temp_path, output_path)
    
    del existing_entries
    
    return total_count


def _prepare_output_dir(path: str) -> str:
    """Create the output directory (no auto-suffix; multi-process safe)."""
    norm_path = path.rstrip("/\\")
    if not norm_path:
        norm_path = "results"
    os.makedirs(norm_path, exist_ok=True)
    return norm_path


def _configure_logging(output_dir: str, log_name: str = "run.log") -> None:
    """Logging setup — configure the root logger so child modules can emit."""
    log_path = os.path.join(output_dir, log_name)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)
    
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False


def _save_candidate_jsons(
    episode_idx: int,
    keyframes: List[tuple],
    candidates: List[BidirectionalTrack],
    episode_dir: str,
) -> None:
    """Save the forward and backward keyframe-anchored trajectories as separate JSON files."""
    forward_dir = os.path.join(episode_dir, "forward_tracks")
    backward_dir = os.path.join(episode_dir, "backward_tracks")
    os.makedirs(forward_dir, exist_ok=True)
    os.makedirs(backward_dir, exist_ok=True)
    for track, (frame, _x, _y) in zip(candidates, keyframes):
        base_name = f"keyframe_{frame:04d}.json"
        forward_path = os.path.join(forward_dir, base_name)
        backward_path = os.path.join(backward_dir, base_name)
        with open(forward_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "episode_idx": episode_idx,
                    "keyframe_idx": frame,
                    "frames": track.forward_frames.tolist(),
                    "trajectory": track.forward.tolist(),
                    "visibility": track.forward_visibility.tolist(),
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )
        with open(backward_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "episode_idx": episode_idx,
                    "keyframe_idx": frame,
                    "frames": track.backward_frames.tolist(),
                    "trajectory": track.backward.tolist(),
                    "visibility": track.backward_visibility.tolist(),
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )


def _build_math_trajectory(
    total_frames: int,
    keyframes: List[tuple],
    fusion_cfg: FusionConfig,
) -> tuple[np.ndarray, Dict]:
    """Construct a trajectory purely from Stage1 keyframes."""
    interpolated = interpolate_keyframe_path(total_frames, keyframes)
    baseline_track = track_from_path(interpolated)
    fused, analysis = fuse_candidate_tracks(
        [baseline_track],
        fusion_cfg,
        max_tracks_by_path=1,
    )
    analysis["mode"] = "puremath"
    return fused, analysis


def _build_single_keyframe_trajectory(
    total_frames: int,
    keyframe: tuple,
) -> tuple[np.ndarray, Dict]:
    """Build a dense constant trajectory when only one Stage1 anchor exists."""
    frame_idx, x, y = keyframe
    fused = np.tile(np.array([[float(x), float(y)]], dtype=np.float32), (total_frames, 1))
    analysis = {
        "mode": "single_keyframe_constant",
        "num_candidates": 0,
        "num_used": 0,
        "candidate_stats": [],
        "single_keyframe": int(frame_idx),
    }
    return fused, analysis


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Define model-runtime CLI arguments."""
    parser = argparse.ArgumentParser(
        description="CoTracker4Trace pipeline")
    parser.add_argument(
        "--keyframe_json",
        type=str,
        default=DEFAULT_KEYFRAME_JSON,
        help="Path to the BC-Z Stage1 sparse keyframe JSON",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="BC-Z RLDS dataset root (also overridable via BCZ_DATASET_PATH)",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="train",
        choices=("train", "val"),
        help="BC-Z split name; 'val' is used to append validation traces",
    )
    parser.add_argument(
        "--episode_index_offset",
        type=int,
        default=0,
        help="Output episode_idx offset (recommend 39350 for BC-Z val so it follows train)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory",
    )
    parser.add_argument(
        "--output_json_name",
        type=str,
        default=DEFAULT_OUTPUT_JSON_NAME,
        help="Output JSON filename (avoids collisions during multi-GPU parallel runs)",
    )
    parser.add_argument(
        "--log_name",
        type=str,
        default="run.log",
        help="Log filename (avoids collisions during multi-GPU parallel runs)",
    )
    parser.add_argument(
        "--start_episode",
        type=int,
        default=0,
        help="First episode to process (inclusive)",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Number of episodes to process sequentially; empty = run to end of data",
    )
    parser.add_argument(
        "--sample_episodes",
        type=int,
        default=None,
        help="Number of episodes to sample randomly (overrides start/max)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random sampling seed",
    )
    parser.add_argument(
        "--max_keyframes",
        type=int,
        default=20,
        help="Maximum keyframes allowed per episode",
    )
    parser.add_argument(
        "--max_path_tracks",
        type=int,
        default=10,
        help="Sort by path length and keep only the top N trajectories for fusion",
    )
    parser.add_argument(
        "--last4backward",
        action="store_true",
        help="Run forward tracking first, then start backward tracking from the forward endpoint",
    )
    parser.add_argument(
        "--nocut4backward",
        action="store_true",
        help="Use the full video for backward tracking; do not truncate at the keyframe start",
    )
    parser.add_argument(
        "--last4cotracker",
        action="store_true",
        help="After fusion, run a full-video backward CoTracker pass and fuse again",
    )
    parser.add_argument(
        "--export_gif",
        action="store_true",
        help="Also produce a GIF animation per episode",
    )
    parser.add_argument(
        "--boostrun",
        action="store_true",
        help="Compact output: keep only the final JSON (and the GIF if enabled)",
    )
    parser.add_argument(
        "--puremath",
        action="store_true",
        help="Skip CoTracker; produce trajectories purely from the Stage1 keyframes",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device, e.g. cuda:0 / cuda:1 / cpu; 'auto' picks CUDA if available",
    )
    return parser.parse_args(argv)


def run_pipeline(args: argparse.Namespace, minimal_outputs: bool = False) -> None:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)  # cuda:0,cuda:1
    args.output_dir = _prepare_output_dir(args.output_dir)
    _configure_logging(args.output_dir, args.log_name)
    logger.info("output directory: %s", args.output_dir)
    logger.info("device: %s", device)

    keyframe_data = load_keyframe_json(args.keyframe_json)
    episodes_map: Dict[int, List[Dict]] = {}
    for item in keyframe_data:
        # episodes_map.setdefault(item["episode_idx"], []).append(item)
        key = item["episode_idx"]
        episodes_map.setdefault(key, [])
        episodes_map[key].append(item)
    episode_indices = sorted(episodes_map.keys())
    if args.sample_episodes is not None:
        random.seed(args.random_seed)
        episode_indices = sorted(random.sample(
            episode_indices, min(args.sample_episodes, len(episode_indices))))
    else:
        range_end = None if args.max_episodes is None else args.start_episode + args.max_episodes
        episode_indices = [
            episode_idx
            for episode_idx in episode_indices
            if episode_idx >= args.start_episode
            and (range_end is None or episode_idx < range_end)
        ]
        if args.max_episodes is not None and len(episode_indices) < args.max_episodes:
            logger.info(
                "requested absolute episode range [%s, %s), but only %d episodes are available in the Stage1 sparse JSON",
                args.start_episode,
                range_end,
                len(episode_indices),
            )

    logger.info("will process episodes: %s", episode_indices)

    fusion_cfg = FusionConfig()
    if args.max_path_tracks is not None:
        fusion_cfg.max_path_tracks = args.max_path_tracks

    cotracker_model = None
    if not args.puremath:
        cotracker_model = load_cotracker_model(device)
        if cotracker_model is None:
            raise RuntimeError("CoTracker model load failed")

    output_json = os.path.join(
        args.output_dir, args.output_json_name)
    

    for episode_idx in tqdm(episode_indices, desc="Episodes"):
        episode_items = episodes_map[episode_idx]
        keyframes = extract_episode_keyframes(
            episode_items, args.max_keyframes)
        logger.info("Episode %s keyframe indices: %s", episode_idx,
                    [kf[0] for kf in keyframes])
        if len(keyframes) < 1:
            logger.warning("Episode %s has no usable keyframes; skipping", episode_idx)
            continue

        frames = load_episode_frames(
            episode_idx,
            args.dataset_path,
            split_name=args.split_name,
            episode_index_offset=args.episode_index_offset,
        )
        fused: np.ndarray
        analysis: Dict
        candidates: List[BidirectionalTrack] = []
        used_cotracker = False

        if len(keyframes) == 1:
            logger.warning(
                "Episode %s has only 1 usable keyframe; using constant dense fallback",
                episode_idx,
            )
            fused, analysis = _build_single_keyframe_trajectory(
                len(frames),
                keyframes[0],
            )
        elif args.puremath:
            fused, analysis = _build_math_trajectory(
                len(frames),
                keyframes,
                fusion_cfg,
            )
        else:
            video_tensor = _prepare_video_tensor(frames, device)
            used_cotracker = True

            candidates = []
            for keyframe in keyframes:
                track = track_keyframe_bidirectional(
                    cotracker_model,
                    video_tensor,
                    keyframe,
                    device=device,
                    use_forward_terminal_for_backward=args.last4backward,
                    nocut_for_backward=args.nocut4backward,
                )
                candidates.append(track)
                forward_diff = np.linalg.norm(
                    np.diff(track.forward, axis=0), axis=1)
                backward_diff = np.linalg.norm(
                    np.diff(track.backward, axis=0), axis=1)
                forward_total = float(
                    forward_diff.sum()) if forward_diff.size else 0.0
                backward_total = float(
                    backward_diff.sum()) if backward_diff.size else 0.0
                logger.info(
                    "Episode %s keyframe %s: avg_vis=%.3f forward_span=%.3f backward_span=%.3f",
                    episode_idx,
                    keyframe[0],
                    float(track.visibility.mean()),
                    forward_total,
                    backward_total,
                )
                if forward_total < 5.0:
                    logger.warning(
                        "Episode %s keyframe %s: forward trajectory has near-zero displacement (%.3f)",
                        episode_idx,
                        keyframe[0],
                        forward_total,
                    )
                if backward_total < 5.0:
                    logger.warning(
                        "Episode %s keyframe %s: backward trajectory has near-zero displacement (%.3f)",
                        episode_idx,
                        keyframe[0],
                        backward_total,
                    )
            fused, analysis = fuse_candidate_tracks(
                candidates,
                fusion_cfg,
                max_tracks_by_path=args.max_path_tracks,
            )
        for stat in analysis.get("candidate_stats", []):
            logger.info(
                "Episode %s keyframe %s summary: avg_vis=%.3f max_jump=%.3f avg_step=%.3f span=%.3f path_len=%.3f kept=%s",
                episode_idx,
                stat["keyframe"],
                stat["avg_visibility"],
                stat["max_jump"],
                stat["avg_step"],
                stat["span"],
                stat.get("path_length", stat["span"]),
                stat["kept"],
            )
        logger.info(
            "Episode %s fusion: candidates %d -> %d, using vis=%s median=%s mean=%s consistency=%s",
            episode_idx,
            analysis["num_candidates"],
            analysis["num_used"],
            fusion_cfg.use_visibility_weight,
            fusion_cfg.use_median,
            fusion_cfg.use_mean,
            fusion_cfg.use_consistency,
        )
        if analysis.get("path_length_limit"):
            kept_ids = [
                stat["keyframe"]
                for stat in analysis.get("candidate_stats", [])
                if stat.get("kept")
            ]
            logger.info(
                "Episode %s path-length filter enabled: cap=%s; retained keyframe trajectories=%s",
                episode_idx,
                analysis["path_length_limit"],
                kept_ids,
            )
        if args.last4cotracker and used_cotracker:
            last_frame_idx = fused.shape[0] - 1
            terminal_point = fused[-1]
            refinement_keyframe = (
                last_frame_idx,
                float(terminal_point[0]),
                float(terminal_point[1]),
            )
            logger.info(
                "Episode %s enabled last4cotracker: endpoint frame=%s coord=(%.3f, %.3f)",
                episode_idx,
                last_frame_idx,
                refinement_keyframe[1],
                refinement_keyframe[2],
            )
            baseline_track = track_from_path(fused)
            refinement_track = track_keyframe_bidirectional(
                cotracker_model,
                video_tensor,
                refinement_keyframe,
                device=device,
                use_forward_terminal_for_backward=False,
                nocut_for_backward=True,
            )
            refined_fused, refined_analysis = fuse_candidate_tracks(
                [baseline_track, refinement_track],
                fusion_cfg,
                max_tracks_by_path=2,
            )
            fused = refined_fused
            analysis["refinement"] = refined_analysis
            for stat in refined_analysis.get("candidate_stats", []):
                label = (
                    "refine_baseline" if stat["keyframe"] == 0 else "refine_backward"
                )
                logger.info(
                    "Episode %s Refinement %s: avg_vis=%.3f max_jump=%.3f avg_step=%.3f span=%.3f path_len=%.3f kept=%s",
                    episode_idx,
                    label,
                    stat["avg_visibility"],
                    stat["max_jump"],
                    stat["avg_step"],
                    stat["span"],
                    stat.get("path_length", stat["span"]),
                    stat["kept"],
                )
            logger.info(
                "Episode %s refinement fusion: candidates %d -> %d",
                episode_idx,
                refined_analysis["num_candidates"],
                refined_analysis["num_used"],
            )

        fused = _apply_keyframe_anchors(fused, keyframes)

        need_episode_dir = args.export_gif or not minimal_outputs
        if need_episode_dir:
            episode_dir = os.path.join(
                args.output_dir, f"episode_{episode_idx:04d}")
            os.makedirs(episode_dir, exist_ok=True)
        
        if not minimal_outputs:
            save_keyframe_scatter(
                keyframes,
                os.path.join(episode_dir, "keyframes.png"),
            )
        if used_cotracker and not minimal_outputs:
            candidate_plot_dir = os.path.join(
                episode_dir, "frame_cotrack_bidirectional"
            )
            candidate_files = save_candidate_paths(
                candidates,
                keyframes,
                candidate_plot_dir,
            )
            kept_keyframes = {
                stat["keyframe"]
                for stat in analysis.get("candidate_stats", [])
                if stat.get("kept")
            }
            scatter_dir = os.path.join(episode_dir, "frame_cotrack_scatter")
            scatter_files = save_candidate_frame_scatters(
                frames,
                candidates,
                keyframes,
                scatter_dir,
                kept_keyframes=kept_keyframes,
            )
            logger.debug(
                "Episode %s candidate trajectory viz: %s", episode_idx, candidate_files
            )
            logger.debug(
                "Episode %s candidate scatter: %s", episode_idx, scatter_files
            )
        if not minimal_outputs:
            fused_plot_path = os.path.join(episode_dir, "fused.png")
            save_fused_trajectory(fused, keyframes, fused_plot_path)

        if args.export_gif:
            gif_path = os.path.join(episode_dir, "trajectory.gif")
            fused_seq = [tuple(map(float, point))
                         for point in fused.tolist()]
            stage1_map = {frame: (float(x), float(y))
                          for frame, x, y in keyframes}
            save_episode_gif(
                frames,
                fused_seq,
                [kf[0] for kf in keyframes],
                gif_path,
                stage1_coords=stage1_map,
                episode_idx=episode_idx,
            )
            logger.info("Episode %s GIF saved: %s", episode_idx, gif_path)

        entries = _build_episode_entries(
            episode_idx, fused, keyframes, analysis)
        total_entries = _append_to_json(entries, output_json)
        logger.info(
            "Episode %s result written to %s (running total %d rows)",
            episode_idx,
            output_json,
            total_entries,
        )
        if used_cotracker and not minimal_outputs:
            _save_candidate_jsons(episode_idx, keyframes,
                                  candidates, episode_dir)
        if used_cotracker:
            del video_tensor
            if device.type == "cuda":
                torch.cuda.empty_cache()

    logger.info("done; results saved to %s", output_json)


def boostrun(**overrides: object) -> None:
    """
    Quick pipeline run: outputs only the GIF and the final JSON.

    Keyword arguments override the default CLI values, e.g.:
        boostrun(dataset_path="/path/to/dataset", start_episode=10, max_episodes=5)
    """
    args = parse_args([])
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise AttributeError(f"unknown argument: {key}")
        setattr(args, key, value)
    args.boostrun = True
    run_pipeline(args, minimal_outputs=True)


def main() -> None:
    args = parse_args()
    run_pipeline(args, minimal_outputs=args.boostrun)


if __name__ == "__main__":
    main()
