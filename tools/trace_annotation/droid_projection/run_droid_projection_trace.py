#!/usr/bin/env python3
"""Generate unified dense traces for DROID by projecting 3D TCP positions.

DROID differs from Molmo/CoTracker datasets: the trace is produced directly by
projecting `steps/observation/cartesian_position[:3]` into a calibrated camera.
The output coordinate is normalized image space `[0, 100]`, matching the other
trace annotation bundles.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from Droid2Trace import (  # noqa: E402
    AnnotationStore,
    CAMERA_CONFIGS,
    CalibrationError,
    decode_rgb_image,
    derive_relative_path,
    iter_droid_examples,
)


logger = logging.getLogger("run_droid_projection_trace")


@dataclass
class ProjectionEpisodeData:
    raw_index: int
    relative_path: Optional[str]
    episode_id: str
    tcp_world_positions: np.ndarray
    intrinsic: np.ndarray
    extrinsic_world_from_camera: np.ndarray
    image_shape: tuple[int, int]
    language_instruction: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--annotations_dir", type=Path, default=SCRIPT_DIR / "droid")
    parser.add_argument(
        "--camera",
        choices=sorted(CAMERA_CONFIGS),
        default="exterior_image_1_left",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument(
        "--end_episode",
        type=int,
        default=None,
        help="Inclusive raw RLDS episode index. Defaults to the dataset end.",
    )
    parser.add_argument(
        "--episode_index_mode",
        choices=("raw", "sequential"),
        default="raw",
        help=(
            "raw keeps episode_idx equal to the RLDS episode index. sequential "
            "assigns compact camera-view ids and writes a metadata map."
        ),
    )
    parser.add_argument(
        "--episode_index_offset",
        type=int,
        default=0,
        help="Offset applied only in sequential mode.",
    )
    parser.add_argument(
        "--max_accepted_episodes",
        type=int,
        default=None,
        help="Stop after this many accepted trace episodes for the current camera.",
    )
    parser.add_argument("--shard_episodes", type=int, default=1000)
    parser.add_argument("--skip_empty_projection", action="store_true", default=True)
    parser.add_argument("--keep_empty_projection", dest="skip_empty_projection", action="store_false")
    parser.add_argument("--include_pixel_coordinate", action="store_true")
    parser.add_argument(
        "--out_of_frame_policy",
        choices=("mask", "clip", "keep_raw"),
        default="mask",
        help=(
            "How to handle projected pixels outside the image. mask writes null "
            "coordinates for out-of-frame steps; clip clamps them to image bounds; "
            "keep_raw writes normalized coordinates that can fall outside [0, 100]."
        ),
    )
    parser.add_argument("--log_level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def build_projection_episode_from_example(
    index: int,
    example: Dict,
    camera_config,
    annotations: AnnotationStore,
) -> ProjectionEpisodeData:
    """Build only the data needed for projection trace generation.

    The original visualizer decodes every frame. Dense trace generation only
    needs image shape, TCP xyz, calibration, and language metadata, so decoding
    the first frame is enough and keeps full-DROID projection practical.
    """
    recording_folder = example["episode_metadata/recording_folderpath"].numpy().decode("utf-8")
    file_path = example["episode_metadata/file_path"].numpy().decode("utf-8")
    relative_path = derive_relative_path(recording_folder, file_path)
    episode_id = annotations.find_episode_id(relative_path)
    if episode_id is None:
        raise CalibrationError("missing_calibration_entry", f"no calibration entry for {relative_path}")

    serial = annotations.resolve_camera_serial(episode_id, camera_config)
    if serial is None:
        raise CalibrationError(
            "missing_camera_serial",
            f"{episode_id} missing camera serial number ({camera_config.rlds_field})",
        )

    intrinsic_info = annotations.get_intrinsic_matrix(episode_id, serial)
    if intrinsic_info is None:
        raise CalibrationError("missing_intrinsic", f"{episode_id} missing intrinsics ({serial})")

    extrinsic = annotations.get_extrinsic_matrix(episode_id, serial)
    if extrinsic is None:
        raise CalibrationError("missing_extrinsic", f"{episode_id} missing extrinsics ({serial})")

    images_raw = example[f"steps/observation/{camera_config.rlds_field}"].values.numpy()
    if len(images_raw) == 0:
        raise ValueError("this episode has no image frames")
    first_frame = decode_rgb_image(images_raw[0])
    image_shape = first_frame.shape[:2]

    cartesian_flat = example["steps/observation/cartesian_position"].values.numpy().astype(np.float64)
    if cartesian_flat.size == 0:
        raise ValueError("this episode is missing cartesian_position data")
    num_steps = len(images_raw)
    if cartesian_flat.size % num_steps != 0:
        raise ValueError("cartesian_position length does not match the image frame count")
    state_dim = cartesian_flat.size // num_steps
    if state_dim < 3:
        raise ValueError("cartesian_position is too short; xyz is required at minimum")
    tcp_world_positions = cartesian_flat.reshape(num_steps, state_dim)[:, :3]

    language_instruction = None
    for key in (
        "steps/language_instruction",
        "steps/language_instruction_2",
        "steps/language_instruction_3",
    ):
        values = example.get(key)
        if values is None:
            continue
        arr = values.values.numpy()
        if len(arr):
            language_instruction = arr[0].decode("utf-8")
            if language_instruction:
                break
    if not language_instruction:
        language_instruction = annotations.get_language_instruction(episode_id)

    frame_height, frame_width = image_shape
    source_width = intrinsic_info.width or frame_width
    source_height = intrinsic_info.height or frame_height
    intrinsic_matrix = intrinsic_info.matrix.copy()
    if source_width and source_height and (
        source_width != frame_width or source_height != frame_height
    ):
        scale_x = frame_width / float(source_width)
        scale_y = frame_height / float(source_height)
        intrinsic_matrix[0, 0] *= scale_x
        intrinsic_matrix[1, 1] *= scale_y
        intrinsic_matrix[0, 2] *= scale_x
        intrinsic_matrix[1, 2] *= scale_y

    return ProjectionEpisodeData(
        raw_index=index,
        relative_path=relative_path,
        episode_id=episode_id,
        tcp_world_positions=tcp_world_positions,
        intrinsic=intrinsic_matrix,
        extrinsic_world_from_camera=extrinsic,
        image_shape=image_shape,
        language_instruction=language_instruction,
    )


def project_world_to_pixels(
    points_world: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic_world_from_camera: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project world xyz points without clipping to the image rectangle."""
    if points_world.shape[1] < 3:
        raise ValueError("points_world must have at least xyz coordinates")

    num_points = points_world.shape[0]
    homo_world = np.concatenate(
        [points_world[:, :3], np.ones((num_points, 1), dtype=np.float64)],
        axis=1,
    )

    cam_t_world = np.linalg.inv(extrinsic_world_from_camera)
    cam_coords = (cam_t_world @ homo_world.T).T[:, :3]
    depths = cam_coords[:, 2]
    depth_positive_mask = np.isfinite(depths) & (depths > 1e-6)
    pixels = np.full((num_points, 2), np.nan, dtype=np.float64)

    if np.any(depth_positive_mask):
        x_cam = cam_coords[depth_positive_mask, 0]
        y_cam = cam_coords[depth_positive_mask, 1]
        z_cam = depths[depth_positive_mask]
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
        pixels[depth_positive_mask, 0] = fx * (x_cam / z_cam) + cx
        pixels[depth_positive_mask, 1] = fy * (y_cam / z_cam) + cy

    height, width = image_shape
    finite_mask = np.isfinite(pixels).all(axis=1)
    in_frame_mask = (
        finite_mask
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] <= float(width - 1))
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] <= float(height - 1))
    )
    return pixels, depth_positive_mask, finite_mask, in_frame_mask


def apply_out_of_frame_policy(
    pixels: np.ndarray,
    in_frame_mask: np.ndarray,
    image_shape: tuple[int, int],
    policy: str,
) -> np.ndarray:
    output = pixels.copy()
    if policy == "mask":
        output[~in_frame_mask] = np.nan
        return output
    if policy == "clip":
        finite_mask = np.isfinite(output).all(axis=1)
        height, width = image_shape
        output[finite_mask, 0] = np.clip(output[finite_mask, 0], 0.0, width - 1.0)
        output[finite_mask, 1] = np.clip(output[finite_mask, 1], 0.0, height - 1.0)
        return output
    if policy == "keep_raw":
        return output
    raise ValueError(f"unknown out_of_frame_policy: {policy}")


def pixel_to_normalized(
    pixels: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[List[Optional[List[float]]], List[Optional[List[float]]]]:
    height, width = image_shape
    valid_mask = np.isfinite(pixels).all(axis=1)
    norm_coords: List[Optional[List[float]]] = []
    pixel_coords: List[Optional[List[float]]] = []
    for idx, coord in enumerate(pixels):
        if not valid_mask[idx]:
            norm_coords.append(None)
            pixel_coords.append(None)
            continue
        x = float(coord[0])
        y = float(coord[1])
        denom_x = max(1.0, float(width - 1))
        denom_y = max(1.0, float(height - 1))
        norm_coords.append([x / denom_x * 100.0, y / denom_y * 100.0])
        pixel_coords.append([x, y])
    return norm_coords, pixel_coords


def write_json_atomic(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def flush_trace_shard(
    output_dir: Path,
    shard_index: int,
    entries: List[Dict],
    start_episode: Optional[int],
    end_episode: Optional[int],
) -> Optional[Path]:
    if not entries or start_episode is None or end_episode is None:
        return None
    path = output_dir / "annotations" / (
        f"droid_stage2_dense_trace_shard_{shard_index:05d}_"
        f"ep{start_episode:06d}_{end_episode:06d}.json"
    )
    write_json_atomic(path, entries)
    logger.info("wrote %s rows=%d", path, len(entries))
    return path


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    dataset_path = args.dataset_path.resolve()
    annotations_dir = args.annotations_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_config = CAMERA_CONFIGS[args.camera]
    annotations = AnnotationStore(annotations_dir)

    trace_entries: List[Dict] = []
    episode_metadata: List[Dict] = []
    manifest_shards: List[Dict] = []
    skip_stats: Counter = Counter()
    coordinate_source_counts: Counter = Counter()
    processed = 0
    accepted = 0
    trace_episode_idx = args.episode_index_offset
    shard_index = 0
    shard_start: Optional[int] = None
    shard_end: Optional[int] = None

    for raw_idx, example in iter_droid_examples(dataset_path, camera_config):
        if raw_idx < args.start_episode:
            continue
        if args.end_episode is not None and raw_idx > args.end_episode:
            break
        processed += 1
        try:
            episode = build_projection_episode_from_example(raw_idx, example, camera_config, annotations)
        except CalibrationError as exc:
            skip_stats[exc.reason] += 1
            logger.debug("skip raw_episode=%s reason=%s %s", raw_idx, exc.reason, exc)
            continue
        except Exception as exc:
            skip_stats["other"] += 1
            logger.warning("skip raw_episode=%s reason=other %s", raw_idx, exc)
            logger.debug("raw_episode=%s traceback", raw_idx, exc_info=True)
            continue

        raw_pixels, depth_positive_mask, finite_mask, in_frame_mask = project_world_to_pixels(
            episode.tcp_world_positions,
            episode.intrinsic,
            episode.extrinsic_world_from_camera,
            image_shape=episode.image_shape,
        )
        positive_depth_count = int(np.count_nonzero(depth_positive_mask))
        finite_count = int(np.count_nonzero(finite_mask))
        in_frame_count = int(np.count_nonzero(in_frame_mask))
        if args.skip_empty_projection and in_frame_count == 0:
            skip_stats["empty_projection"] += 1
            continue

        pixels = apply_out_of_frame_policy(
            raw_pixels,
            in_frame_mask,
            episode.image_shape,
            args.out_of_frame_policy,
        )

        if args.episode_index_mode == "raw":
            out_episode_idx = raw_idx
        else:
            out_episode_idx = trace_episode_idx
            trace_episode_idx += 1

        norm_coords, pixel_coords = pixel_to_normalized(pixels, episode.image_shape)
        episode_source_counts: Counter = Counter()
        for step_idx, coord in enumerate(norm_coords):
            if coord is None:
                coordinate_source = "projection_invalid"
            elif bool(in_frame_mask[step_idx]):
                coordinate_source = "projection_in_frame"
            elif args.out_of_frame_policy == "clip" and bool(finite_mask[step_idx]):
                coordinate_source = "projection_clipped"
            elif args.out_of_frame_policy == "keep_raw" and bool(finite_mask[step_idx]):
                coordinate_source = "projection_raw_out_of_frame"
            else:
                coordinate_source = "projection_valid_other"
            episode_source_counts[coordinate_source] += 1
            coordinate_source_counts[coordinate_source] += 1
            entry = {
                "episode_idx": int(out_episode_idx),
                "step_idx": int(step_idx),
                "coordinate": coord,
                "is_keyframe": False,
                "is_interpolated": False,
                "trace_source": "droid_projection",
                "coordinate_source": coordinate_source,
            }
            if args.include_pixel_coordinate:
                entry["pixel_coordinate"] = pixel_coords[step_idx]
            trace_entries.append(entry)
        valid_coordinate_count = sum(1 for coord in norm_coords if coord is not None)

        episode_metadata.append(
            {
                "episode_idx": int(out_episode_idx),
                "raw_episode_idx": int(raw_idx),
                "episode_id": episode.episode_id,
                "camera": args.camera,
                "relative_path": episode.relative_path,
                "num_steps": int(len(norm_coords)),
                "positive_depth_steps": positive_depth_count,
                "finite_projection_steps": finite_count,
                "in_frame_projection_steps": in_frame_count,
                "valid_projection_steps": valid_coordinate_count,
                "valid_coordinate_steps": valid_coordinate_count,
                "null_coordinate_steps": int(len(norm_coords) - valid_coordinate_count),
                "clipped_projection_steps": int(episode_source_counts.get("projection_clipped", 0)),
                "coordinate_source_counts": dict(episode_source_counts),
                "out_of_frame_policy": args.out_of_frame_policy,
                "image_height": int(episode.image_shape[0]),
                "image_width": int(episode.image_shape[1]),
                "language_instruction": episode.language_instruction,
            }
        )
        accepted += 1
        if shard_start is None:
            shard_start = int(out_episode_idx)
        shard_end = int(out_episode_idx)

        if accepted % args.shard_episodes == 0:
            shard_path = flush_trace_shard(output_dir, shard_index, trace_entries, shard_start, shard_end)
            if shard_path:
                manifest_shards.append(
                    {
                        "file": str(shard_path),
                        "rows": len(trace_entries),
                        "start_episode": shard_start,
                        "end_episode": shard_end,
                    }
                )
                shard_index += 1
            trace_entries = []
            shard_start = None
            shard_end = None
            write_json_atomic(output_dir / "episode_metadata.json", episode_metadata)

        if accepted <= 5 or accepted % 100 == 0:
            logger.info(
                "accepted=%d raw_episode=%d out_episode=%d steps=%d valid=%d in_frame=%d",
                accepted,
                raw_idx,
                out_episode_idx,
                len(norm_coords),
                valid_coordinate_count,
                in_frame_count,
            )
        if args.max_accepted_episodes is not None and accepted >= args.max_accepted_episodes:
            logger.info(
                "reached max_accepted_episodes=%d at raw_episode=%d",
                args.max_accepted_episodes,
                raw_idx,
            )
            break

    shard_path = flush_trace_shard(output_dir, shard_index, trace_entries, shard_start, shard_end)
    if shard_path:
        manifest_shards.append(
            {
                "file": str(shard_path),
                "rows": len(trace_entries),
                "start_episode": shard_start,
                "end_episode": shard_end,
            }
        )

    write_json_atomic(output_dir / "episode_metadata.json", episode_metadata)
    write_json_atomic(
        output_dir / "manifest.json",
        {
            "dataset_path": str(dataset_path),
            "annotations_dir": str(annotations_dir),
            "camera": args.camera,
            "out_of_frame_policy": args.out_of_frame_policy,
            "episode_index_mode": args.episode_index_mode,
            "episode_index_offset": args.episode_index_offset,
            "processed_raw_episodes": processed,
            "accepted_trace_episodes": accepted,
            "skip_stats": dict(skip_stats),
            "coordinate_source_counts": dict(coordinate_source_counts),
            "shards": manifest_shards,
        },
    )
    logger.info(
        "done processed_raw=%d accepted=%d skip_stats=%s",
        processed,
        accepted,
        dict(skip_stats),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
