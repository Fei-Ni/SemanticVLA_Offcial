#!/usr/bin/env python3
"""Fill DROID null coordinates while preserving episode/step alignment.

The masked DROID bundle keeps `coordinate: null` for out-of-frame projections.
Training code that expects dense traces cannot consume null targets, but rows
must not be dropped because `step_idx` has to stay aligned with raw DROID frames.

This script writes a no-null dense bundle by linearly interpolating missing
coordinates within each episode and using the nearest valid coordinate for
leading/trailing null spans. The input bundle is left untouched for audit.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_INPUT = Path(
    "/path/to/droid_projection_trace"
)
DEFAULT_OUTPUT = Path(
    "/path/to/droid_projection_trace_dense"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--shard_prefix",
        default="droid_48k_dense_trace_shard",
        help="Output annotation shard filename prefix.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    annotations_dir = output_dir / "annotations"
    existing = []
    if annotations_dir.exists():
        existing.extend(annotations_dir.glob("*.json"))
    existing.extend(output_dir.glob("*.json"))
    if existing and not overwrite:
        raise FileExistsError(f"{output_dir} already has files; pass --overwrite")
    for path in existing:
        path.unlink()
    annotations_dir.mkdir(parents=True, exist_ok=True)


def interpolate_episode(rows: List[Dict], metadata: Dict) -> Tuple[List[Dict], int, Counter]:
    rows = sorted(rows, key=lambda row: int(row["step_idx"]))
    valid_points: List[Tuple[int, float, float]] = []
    for row in rows:
        coord = row.get("coordinate")
        if coord is not None:
            valid_points.append((int(row["step_idx"]), float(coord[0]), float(coord[1])))

    if not valid_points:
        raise ValueError(f"episode {rows[0].get('episode_idx')} has no valid coordinates")

    valid_steps = [item[0] for item in valid_points]
    valid_xs = [item[1] for item in valid_points]
    valid_ys = [item[2] for item in valid_points]

    filled_count = 0
    source_counts: Counter = Counter()
    out_rows: List[Dict] = []
    next_valid_index = 0
    last_valid_index = 0
    width = int(metadata.get("image_width", 0) or 0)
    height = int(metadata.get("image_height", 0) or 0)

    for row in rows:
        out = dict(row)
        if out.get("coordinate") is None:
            step = int(out["step_idx"])
            while next_valid_index < len(valid_steps) and valid_steps[next_valid_index] < step:
                last_valid_index = next_valid_index
                next_valid_index += 1

            if step <= valid_steps[0]:
                x, y = valid_xs[0], valid_ys[0]
                method = "temporal_backfill"
            elif step >= valid_steps[-1]:
                x, y = valid_xs[-1], valid_ys[-1]
                method = "temporal_forward_fill"
            else:
                left = max(0, last_valid_index)
                right = max(next_valid_index, left + 1)
                if valid_steps[right] <= step:
                    right = min(right + 1, len(valid_steps) - 1)
                span = float(valid_steps[right] - valid_steps[left])
                alpha = 0.0 if span == 0.0 else (step - valid_steps[left]) / span
                x = valid_xs[left] + alpha * (valid_xs[right] - valid_xs[left])
                y = valid_ys[left] + alpha * (valid_ys[right] - valid_ys[left])
                method = "temporal_interpolation"

            x = min(100.0, max(0.0, float(x)))
            y = min(100.0, max(0.0, float(y)))
            out["coordinate"] = [x, y]
            out["coordinate_fill_method"] = method
            out["coordinate_was_null"] = True
            out["coordinate_source"] = method
            if width > 1 and height > 1:
                out["pixel_coordinate"] = [
                    x / 100.0 * float(width - 1),
                    y / 100.0 * float(height - 1),
                ]
            filled_count += 1
        else:
            out.setdefault("coordinate_source", "projection_in_frame")
        source_counts[str(out.get("coordinate_source"))] += 1
        out_rows.append(out)

    return out_rows, filled_count, source_counts


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    input_manifest_path = input_dir / "manifest.json"
    input_metadata_path = input_dir / "episode_metadata.json"
    manifest = load_json(input_manifest_path)
    metadata = load_json(input_metadata_path)
    metadata_by_episode = {int(item["episode_idx"]): item for item in metadata}

    prepare_output(output_dir, args.overwrite)

    total_rows = 0
    filled_rows = 0
    output_shards = []
    per_episode_filled = defaultdict(int)
    per_episode_source_counts: Dict[int, Counter] = defaultdict(Counter)
    coordinate_source_counts: Counter = Counter()

    for shard_index, shard in enumerate(manifest["shards"]):
        source_path = Path(shard["file"])
        rows = load_json(source_path)
        by_episode: Dict[int, List[Dict]] = defaultdict(list)
        for row in rows:
            by_episode[int(row["episode_idx"])].append(row)

        out_rows: List[Dict] = []
        for episode_idx in sorted(by_episode):
            if episode_idx not in metadata_by_episode:
                raise KeyError(f"missing metadata for episode {episode_idx}")
            episode_rows, episode_filled, episode_source_counts = interpolate_episode(
                by_episode[episode_idx],
                metadata_by_episode[episode_idx],
            )
            out_rows.extend(episode_rows)
            per_episode_filled[episode_idx] += episode_filled
            per_episode_source_counts[episode_idx].update(episode_source_counts)
            coordinate_source_counts.update(episode_source_counts)
            filled_rows += episode_filled

        if any(row.get("coordinate") is None for row in out_rows):
            raise RuntimeError(f"null coordinate remains in {source_path}")

        output_path = output_dir / "annotations" / (
            f"{args.shard_prefix}_{shard_index:05d}_"
            f"ep{int(shard['start_episode']):06d}_{int(shard['end_episode']):06d}.json"
        )
        write_json_atomic(output_path, out_rows)
        total_rows += len(out_rows)
        output_shards.append(
            {
                "file": str(output_path.resolve()),
                "rows": len(out_rows),
                "start_episode": int(shard["start_episode"]),
                "end_episode": int(shard["end_episode"]),
            }
        )

    output_metadata = []
    for item in metadata:
        out_item = dict(item)
        out_item["filled_null_coordinate_steps"] = int(per_episode_filled[int(item["episode_idx"])])
        out_item["dense_coordinate_source_counts"] = dict(
            per_episode_source_counts[int(item["episode_idx"])]
        )
        output_metadata.append(out_item)

    if total_rows != int(manifest["total_rows"]):
        raise RuntimeError(f"row count changed: {total_rows} vs {manifest['total_rows']}")
    if filled_rows != int(manifest.get("null_coordinate_rows", 0)):
        raise RuntimeError(
            f"filled rows {filled_rows} != input null rows {manifest.get('null_coordinate_rows')}"
        )

    write_json_atomic(output_dir / "episode_metadata.json", output_metadata)
    write_json_atomic(
        output_dir / "manifest.json",
        {
            "trace_bundle": output_dir.name,
            "source_trace_bundle": str(input_dir),
            "source_manifest": str(input_manifest_path),
            "target_episodes": manifest.get("target_episodes"),
            "selected_episodes": manifest.get("selected_episodes"),
            "shard_episodes": manifest.get("shard_episodes"),
            "total_rows": total_rows,
            "valid_coordinate_rows": total_rows,
            "null_coordinate_rows": 0,
            "source_null_coordinate_rows": int(manifest.get("null_coordinate_rows", 0)),
            "filled_coordinate_rows": filled_rows,
            "allows_null_coordinate": False,
            "episode_index_mode": manifest.get("episode_index_mode", "compact_reindexed"),
            "selection_policy": manifest.get("selection_policy"),
            "fill_policy": (
                "clip-first dense trace: source bundle should be generated with "
                "--out_of_frame_policy clip; residual null coordinates use "
                "linear interpolation between valid coordinates; leading null "
                "spans use temporal_backfill; trailing null spans use "
                "temporal_forward_fill; coordinates clipped to [0, 100]"
            ),
            "coordinate_source_counts": dict(coordinate_source_counts),
            "shards": output_shards,
        },
    )
    print(
        f"wrote dense DROID bundle to {output_dir} rows={total_rows} filled={filled_rows}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
