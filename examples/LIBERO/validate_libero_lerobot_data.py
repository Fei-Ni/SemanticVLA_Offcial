#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_DATASETS = (
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot",
)

REQUIRED_FILES = (
    "meta/modality.json",
    "meta/episodes.jsonl",
    "meta/info.json",
)


def _episode_stems(episodes_path: Path) -> list[str]:
    stems: list[str] = []
    for line in episodes_path.read_text().splitlines():
        if not line.strip():
            continue
        episode = json.loads(line)
        stems.append(f"episode_{int(episode['episode_index']):06d}")
    return stems


def _gather_video_stems(dataset_root: Path, original_key: str) -> set[str]:
    stems: set[str] = set()
    for chunk_dir in sorted((dataset_root / "videos").glob("*")):
        video_dir = chunk_dir / original_key
        if not video_dir.is_dir():
            continue
        stems.update(path.stem for path in video_dir.glob("*.mp4"))
    return stems


def _preview_missing(names: Iterable[str], limit: int = 5) -> str:
    missing = sorted(names)
    if len(missing) <= limit:
        return ", ".join(missing)
    return ", ".join(missing[:limit]) + f", ... ({len(missing)} missing)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SemanticVLA LIBERO LeRobot datasets.")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory that should contain the four LIBERO LeRobot dataset folders.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset directory names to validate under --data-root.",
    )
    return parser.parse_args()


def validate_dataset(dataset_root: Path) -> list[str]:
    errors: list[str] = []

    if not dataset_root.is_dir():
        return [f"missing dataset directory: {dataset_root}"]

    for rel_path in REQUIRED_FILES:
        if not (dataset_root / rel_path).is_file():
            errors.append(f"missing required file: {dataset_root / rel_path}")

    episodes_path = dataset_root / "meta/episodes.jsonl"
    episode_stems = _episode_stems(episodes_path) if episodes_path.is_file() else []
    expected_episode_set = set(episode_stems)

    parquet_files = sorted(dataset_root.glob("data/*/*.parquet"))
    if not parquet_files:
        errors.append(f"missing parquet shards under: {dataset_root / 'data'}")
    elif expected_episode_set:
        parquet_stems = {path.stem for path in parquet_files}
        missing_parquet = expected_episode_set - parquet_stems
        if missing_parquet:
            errors.append(
                f"missing parquet shards for episodes under {dataset_root / 'data'}: "
                f"{_preview_missing(missing_parquet)}"
            )

    video_dirs = sorted((dataset_root / "videos").glob("*")) if (dataset_root / "videos").is_dir() else []
    if not video_dirs:
        errors.append(f"missing video directory tree under: {dataset_root / 'videos'}")

    modality_path = dataset_root / "meta/modality.json"
    if modality_path.is_file():
        try:
            modality = json.loads(modality_path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"invalid JSON in {modality_path}: {exc}")
        else:
            if "video" not in modality:
                errors.append(f"modality metadata missing 'video' section: {modality_path}")
            if "action" not in modality:
                errors.append(f"modality metadata missing 'action' section: {modality_path}")
            if "state" not in modality:
                errors.append(f"modality metadata missing 'state' section: {modality_path}")
            if "video" in modality and expected_episode_set:
                for video_name, video_meta in modality["video"].items():
                    original_key = video_meta.get("original_key")
                    if not original_key:
                        errors.append(f"video modality entry missing original_key: {modality_path}::{video_name}")
                        continue
                    video_stems = _gather_video_stems(dataset_root, original_key)
                    if not video_stems:
                        errors.append(
                            f"missing video files for {original_key} under: {dataset_root / 'videos'}"
                        )
                        continue
                    missing_videos = expected_episode_set - video_stems
                    if missing_videos:
                        errors.append(
                            f"missing video files for {original_key} under {dataset_root / 'videos'}: "
                            f"{_preview_missing(missing_videos)}"
                        )

    if not errors:
        print(
            f"[OK] {dataset_root.name}: "
            f"{len(parquet_files)} parquet shards, "
            f"{len(video_dirs)} top-level video chunks"
        )
    return errors


def main() -> int:
    args = parse_args()
    all_errors: list[str] = []

    print(f"Validating LIBERO LeRobot datasets under: {args.data_root}")
    for dataset_name in args.datasets:
        dataset_root = args.data_root / dataset_name
        all_errors.extend(validate_dataset(dataset_root))

    if all_errors:
        print("\nValidation failed:", file=sys.stderr)
        for error in all_errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("All requested LIBERO LeRobot datasets passed validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
