#!/usr/bin/env python3
"""Merge multiple BC-Z dense trace worker outputs into one JSON file."""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

DEFAULT_RESULT_NAME = "complete_trajectories_bcz_dense_trace.json"


def _load_result_file(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def merge_bcz_results(
    result_dir_pattern: str,
    output_file: str,
    result_name: str = DEFAULT_RESULT_NAME,
) -> None:
    print("=" * 70)
    print("Merge BC-Z Dense Trace Results")
    print("=" * 70)

    result_files = sorted(glob.glob(os.path.join(result_dir_pattern, result_name)))
    if not result_files:
        raise FileNotFoundError(
            f"no result files found at {os.path.join(result_dir_pattern, result_name)}"
        )

    print(f"found {len(result_files)} result files")
    for path in result_files:
        print(f"  - {path}")

    merged: List[Dict] = []
    episode_presence: Counter[int] = Counter()
    seen_keys: set[Tuple[int, int]] = set()
    duplicate_rows = 0

    for file_path in result_files:
        data = _load_result_file(file_path)
        episodes = {int(item["episode_idx"]) for item in data}
        episode_presence.update(episodes)
        print(
            f"loaded {file_path}: {len(data):,} rows, {len(episodes):,} episodes"
        )
        for item in data:
            key = (int(item["episode_idx"]), int(item["step_idx"]))
            if key in seen_keys:
                duplicate_rows += 1
                continue
            seen_keys.add(key)
            merged.append(item)

    merged.sort(key=lambda item: (int(item["episode_idx"]), int(item["step_idx"])))

    duplicate_episodes = {
        episode_idx: count for episode_idx, count in episode_presence.items() if count > 1
    }
    total_episodes = len({int(item["episode_idx"]) for item in merged})
    total_rows = len(merged)
    keyframes = sum(1 for item in merged if item.get("is_keyframe", False))
    avg_frames = total_rows / total_episodes if total_episodes else 0.0

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, ensure_ascii=False, indent=2)

    print()
    print("summary:")
    print(f"  total episodes: {total_episodes:,}")
    print(f"  total trajectory points: {total_rows:,}")
    print(f"  keyframes: {keyframes:,}")
    print(f"  average frames per episode: {avg_frames:.1f}")
    print(f"  duplicate step rows dropped: {duplicate_rows:,}")
    if duplicate_episodes:
        print(f"  warning: {len(duplicate_episodes)} episodes appear in multiple worker outputs")
    print(f"  output file: {output_path}")
    print("done")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge BC-Z dense trace worker outputs.")
    parser.add_argument(
        "--input_pattern",
        type=str,
        default="./worker_*",
        help="glob for worker output directories",
    )
    parser.add_argument(
        "--result_name",
        type=str,
        default=DEFAULT_RESULT_NAME,
        help="JSON filename inside each worker directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./merged_bcz_dense_trace.json",
        help="merged output JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_bcz_results(args.input_pattern, args.output, args.result_name)


if __name__ == "__main__":
    main()
