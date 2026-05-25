#!/usr/bin/env python3
"""Build the paper-count DROID 48K trace bundle from projected camera traces.

The projection runner writes one directory per camera view. This script merges
those directories in the order supplied, selects a deterministic prefix of
camera-view episodes, reindexes them to a compact 0..N-1 range, and rewrites the
dense trace shards plus metadata.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SourceKey = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        action="append",
        type=Path,
        required=True,
        help="Projected DROID trace directory. Pass ext1 first, then ext2.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--target_episodes", type=int, default=48000)
    parser.add_argument("--shard_episodes", type=int, default=1000)
    parser.add_argument(
        "--shard_prefix",
        default="droid_48k_stage2_dense_trace_shard",
        help="Output annotation shard filename prefix.",
    )
    parser.add_argument(
        "--allow_fewer",
        action="store_true",
        help="Write a smaller bundle if inputs have fewer than target_episodes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove this script's previous output files before writing.",
    )
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


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    annotations_dir = output_dir / "annotations"
    existing = []
    if annotations_dir.exists():
        existing.extend(annotations_dir.glob("*.json"))
    existing.extend(output_dir.glob("*.json"))
    if existing and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains output files; pass --overwrite to replace them"
        )
    for path in existing:
        path.unlink()
    annotations_dir.mkdir(parents=True, exist_ok=True)


def iter_source_metadata(input_dirs: List[Path]) -> Iterable[Tuple[int, Dict]]:
    for source_index, input_dir in enumerate(input_dirs):
        metadata_path = input_dir / "episode_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"missing metadata: {metadata_path}")
        metadata = load_json(metadata_path)
        if not isinstance(metadata, list):
            raise ValueError(f"metadata must be a list: {metadata_path}")
        for item in sorted(metadata, key=lambda row: int(row["episode_idx"])):
            yield source_index, item


def select_episodes(
    input_dirs: List[Path],
    target_episodes: int,
    allow_fewer: bool,
) -> Tuple[Dict[SourceKey, int], List[Dict]]:
    selected: Dict[SourceKey, int] = {}
    selected_metadata: List[Dict] = []
    for source_index, item in iter_source_metadata(input_dirs):
        if len(selected_metadata) >= target_episodes:
            break
        source_episode_idx = int(item["episode_idx"])
        key = (source_index, source_episode_idx)
        if key in selected:
            raise ValueError(f"duplicate source episode key: {key}")
        output_episode_idx = len(selected_metadata)
        selected[key] = output_episode_idx

        out_item = dict(item)
        out_item["episode_idx"] = output_episode_idx
        out_item["source_episode_idx"] = source_episode_idx
        out_item["source_projection_dir"] = str(input_dirs[source_index].resolve())
        selected_metadata.append(out_item)

    if len(selected_metadata) < target_episodes and not allow_fewer:
        raise RuntimeError(
            f"only found {len(selected_metadata)} episodes, need {target_episodes}; "
            "pass --allow_fewer for smoke tests"
        )
    return selected, selected_metadata


def source_shards(input_dir: Path) -> List[Path]:
    annotations_dir = input_dir / "annotations"
    if not annotations_dir.exists():
        raise FileNotFoundError(f"missing annotations dir: {annotations_dir}")
    shards = sorted(annotations_dir.glob("*.json"))
    if not shards:
        raise FileNotFoundError(f"no annotation shards in {annotations_dir}")
    return shards


def flush_group(
    output_dir: Path,
    shard_prefix: str,
    shard_index: int,
    rows: List[Dict],
    start_episode: int,
    end_episode: int,
) -> Dict:
    output_path = output_dir / "annotations" / (
        f"{shard_prefix}_{shard_index:05d}_"
        f"ep{start_episode:06d}_{end_episode:06d}.json"
    )
    write_json_atomic(output_path, rows)
    return {
        "file": str(output_path.resolve()),
        "rows": len(rows),
        "start_episode": start_episode,
        "end_episode": end_episode,
    }


def rewrite_selected_rows(
    input_dirs: List[Path],
    output_dir: Path,
    selected: Dict[SourceKey, int],
    shard_episodes: int,
    shard_prefix: str,
) -> Tuple[List[Dict], int, int]:
    manifest_shards: List[Dict] = []
    rows_buffer: List[Dict] = []
    current_group = None
    current_start = None
    current_end = None
    total_rows = 0
    total_valid_rows = 0

    for source_index, input_dir in enumerate(input_dirs):
        for shard_path in source_shards(input_dir):
            rows = load_json(shard_path)
            if not isinstance(rows, list):
                raise ValueError(f"annotation shard must be a list: {shard_path}")
            for row in rows:
                source_episode_idx = int(row["episode_idx"])
                key = (source_index, source_episode_idx)
                if key not in selected:
                    continue
                output_episode_idx = selected[key]
                group = output_episode_idx // shard_episodes
                if current_group is None:
                    current_group = group
                    current_start = output_episode_idx
                elif group != current_group:
                    if current_start is None or current_end is None:
                        raise RuntimeError("internal shard bounds error")
                    manifest_shards.append(
                        flush_group(
                            output_dir,
                            shard_prefix,
                            int(current_group),
                            rows_buffer,
                            int(current_start),
                            int(current_end),
                        )
                    )
                    rows_buffer = []
                    current_group = group
                    current_start = output_episode_idx

                out_row = dict(row)
                out_row["episode_idx"] = output_episode_idx
                rows_buffer.append(out_row)
                current_end = output_episode_idx
                total_rows += 1
                if out_row.get("coordinate") is not None:
                    total_valid_rows += 1

    if rows_buffer:
        if current_group is None or current_start is None or current_end is None:
            raise RuntimeError("internal final shard bounds error")
        manifest_shards.append(
            flush_group(
                output_dir,
                shard_prefix,
                int(current_group),
                rows_buffer,
                int(current_start),
                int(current_end),
            )
        )

    return manifest_shards, total_rows, total_valid_rows


def main() -> int:
    args = parse_args()
    input_dirs = [path.resolve() for path in args.input_dir]
    output_dir = args.output_dir.resolve()
    prepare_output_dir(output_dir, args.overwrite)

    selected, selected_metadata = select_episodes(
        input_dirs=input_dirs,
        target_episodes=args.target_episodes,
        allow_fewer=args.allow_fewer,
    )
    manifest_shards, total_rows, total_valid_rows = rewrite_selected_rows(
        input_dirs=input_dirs,
        output_dir=output_dir,
        selected=selected,
        shard_episodes=args.shard_episodes,
        shard_prefix=args.shard_prefix,
    )

    episodes_with_rows = set()
    for shard in manifest_shards:
        # The output IDs are compact and every selected episode has at least one
        # source row, so shard bounds are sufficient for this sanity count.
        episodes_with_rows.update(range(int(shard["start_episode"]), int(shard["end_episode"]) + 1))
    if len(episodes_with_rows) != len(selected_metadata):
        raise RuntimeError(
            f"selected {len(selected_metadata)} episodes but wrote rows for "
            f"{len(episodes_with_rows)} episodes"
        )

    write_json_atomic(output_dir / "episode_metadata.json", selected_metadata)
    write_json_atomic(
        output_dir / "manifest.json",
        {
            "trace_bundle": output_dir.name,
            "input_dirs": [str(path) for path in input_dirs],
            "target_episodes": args.target_episodes,
            "selected_episodes": len(selected_metadata),
            "shard_episodes": args.shard_episodes,
            "total_rows": total_rows,
            "valid_coordinate_rows": total_valid_rows,
            "null_coordinate_rows": total_rows - total_valid_rows,
            "episode_index_mode": "compact_reindexed",
            "selection_policy": "input directory order, then source episode_idx ascending",
            "shards": manifest_shards,
        },
    )
    print(
        f"wrote {len(selected_metadata)} episodes, {total_rows} rows "
        f"({total_valid_rows} valid coordinates) to {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
