#!/usr/bin/env python3
"""Verify dense trace annotation bundles.

The verifier is intentionally dependency-free and streams large JSON arrays so
BC-Z/Fractal/Bridge can be checked without loading every row at once.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional


# Set TRACE_ROOT / DEFAULT_DROID_DIR via --trace_root / --droid_dir on the CLI.
TRACE_ROOT = Path("/path/to/trace_annotations")
DEFAULT_DROID_DIR = Path("/path/to/droid_projection_trace")


@dataclass
class EpisodeStats:
    rows: int = 0
    min_step: Optional[int] = None
    max_step: Optional[int] = None
    step_sum: int = 0
    step_sumsq: int = 0
    keyframes: int = 0
    interpolated: int = 0
    valid_coords: int = 0
    null_coords: int = 0

    def add(self, step_idx: int, is_keyframe: bool, is_interpolated: bool, has_coord: bool) -> None:
        self.rows += 1
        self.min_step = step_idx if self.min_step is None else min(self.min_step, step_idx)
        self.max_step = step_idx if self.max_step is None else max(self.max_step, step_idx)
        self.step_sum += step_idx
        self.step_sumsq += step_idx * step_idx
        self.keyframes += int(is_keyframe)
        self.interpolated += int(is_interpolated)
        self.valid_coords += int(has_coord)
        self.null_coords += int(not has_coord)

    def contiguous_zero_based(self) -> bool:
        if self.rows == 0 or self.min_step != 0 or self.max_step is None:
            return False
        if self.max_step != self.rows - 1:
            return False
        n = self.rows
        expected_sum = n * (n - 1) // 2
        expected_sumsq = n * (n - 1) * (2 * n - 1) // 6
        return self.step_sum == expected_sum and self.step_sumsq == expected_sumsq


@dataclass
class TraceSummary:
    name: str
    paths: List[str]
    status: str = "ok"
    rows: int = 0
    episodes: int = 0
    min_episode: Optional[int] = None
    max_episode: Optional[int] = None
    keyframes: int = 0
    interpolated: int = 0
    valid_coords: int = 0
    null_coords: int = 0
    coord_min_x: Optional[float] = None
    coord_max_x: Optional[float] = None
    coord_min_y: Optional[float] = None
    coord_max_y: Optional[float] = None
    bad_rows: List[str] = field(default_factory=list)
    noncontiguous_episodes: List[int] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def fail(self, message: str) -> None:
        self.status = "failed"
        if len(self.bad_rows) < 25:
            self.bad_rows.append(message)


def iter_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterator[Dict]:
    """Yield top-level objects from a JSON array without materializing it.

    Trace shards are arrays of row objects. Scanning for balanced top-level
    braces is less fragile at chunk boundaries than repeatedly invoking
    JSONDecoder.raw_decode on a moving text buffer.
    """
    with path.open("r", encoding="utf-8") as handle:
        item_chars: List[str] = []
        depth = 0
        in_string = False
        escape = False
        saw_array = False

        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            for ch in chunk:
                if not saw_array:
                    if ch.isspace():
                        continue
                    if ch != "[":
                        raise ValueError(f"{path} is not a JSON array")
                    saw_array = True
                    continue

                if depth == 0:
                    if ch.isspace() or ch == ",":
                        continue
                    if ch == "]":
                        return
                    if ch != "{":
                        raise ValueError(f"{path} expected object start, got {ch!r}")
                    item_chars = [ch]
                    depth = 1
                    in_string = False
                    escape = False
                    continue

                item_chars.append(ch)
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue

                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        yield json.loads("".join(item_chars))
                        item_chars = []

        if depth != 0:
            raise ValueError(f"{path} ended inside an object")
        if not saw_array:
            raise ValueError(f"{path} is empty")


def update_coord_bounds(summary: TraceSummary, coord: object, allow_null: bool) -> bool:
    if coord is None:
        if not allow_null:
            summary.fail("coordinate is null but null coordinates are not allowed")
        return False
    if not (
        isinstance(coord, list)
        and len(coord) == 2
        and all(isinstance(value, (int, float)) for value in coord)
    ):
        summary.fail(f"invalid coordinate shape: {coord!r}")
        return False

    x = float(coord[0])
    y = float(coord[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        summary.fail(f"non-finite coordinate: {coord!r}")
        return False
    if not (0.0 <= x <= 100.0 and 0.0 <= y <= 100.0):
        summary.fail(f"coordinate outside [0, 100]: {coord!r}")
        return False

    summary.coord_min_x = x if summary.coord_min_x is None else min(summary.coord_min_x, x)
    summary.coord_max_x = x if summary.coord_max_x is None else max(summary.coord_max_x, x)
    summary.coord_min_y = y if summary.coord_min_y is None else min(summary.coord_min_y, y)
    summary.coord_max_y = y if summary.coord_max_y is None else max(summary.coord_max_y, y)
    return True


def summarize_rows(
    name: str,
    paths: List[Path],
    allow_null_coords: bool,
    expected_episodes: Optional[int] = None,
    expected_min_episode: Optional[int] = None,
    expected_max_episode: Optional[int] = None,
    expected_rows: Optional[int] = None,
) -> TraceSummary:
    summary = TraceSummary(name=name, paths=[str(path) for path in paths])
    episode_stats: Dict[int, EpisodeStats] = {}

    for path in paths:
        if not path.exists():
            summary.fail(f"missing file: {path}")
            continue
        for row_idx, row in enumerate(iter_json_array(path), start=1):
            try:
                episode_idx = int(row["episode_idx"])
                step_idx = int(row["step_idx"])
            except Exception as exc:
                summary.fail(f"{path}:{row_idx} missing integer episode_idx/step_idx: {exc}")
                continue

            coord_ok = update_coord_bounds(summary, row.get("coordinate"), allow_null_coords)
            stats = episode_stats.setdefault(episode_idx, EpisodeStats())
            stats.add(
                step_idx=step_idx,
                is_keyframe=bool(row.get("is_keyframe", False)),
                is_interpolated=bool(row.get("is_interpolated", False)),
                has_coord=coord_ok,
            )
            summary.rows += 1

    if episode_stats:
        episodes = sorted(episode_stats)
        summary.episodes = len(episodes)
        summary.min_episode = episodes[0]
        summary.max_episode = episodes[-1]
        for episode_idx, stats in episode_stats.items():
            summary.keyframes += stats.keyframes
            summary.interpolated += stats.interpolated
            summary.valid_coords += stats.valid_coords
            summary.null_coords += stats.null_coords
            if not stats.contiguous_zero_based() and len(summary.noncontiguous_episodes) < 25:
                summary.noncontiguous_episodes.append(episode_idx)
        if summary.noncontiguous_episodes:
            summary.fail("one or more episodes have non-contiguous step_idx values")

    if expected_rows is not None and summary.rows != expected_rows:
        summary.fail(f"expected {expected_rows} rows, found {summary.rows}")
    if expected_episodes is not None and summary.episodes != expected_episodes:
        summary.fail(f"expected {expected_episodes} episodes, found {summary.episodes}")
    if expected_min_episode is not None and summary.min_episode != expected_min_episode:
        summary.fail(f"expected min_episode {expected_min_episode}, found {summary.min_episode}")
    if expected_max_episode is not None and summary.max_episode != expected_max_episode:
        summary.fail(f"expected max_episode {expected_max_episode}, found {summary.max_episode}")

    return summary


def verify_droid(droid_dir: Path, skip_missing: bool) -> TraceSummary:
    manifest_path = droid_dir / "manifest.json"
    metadata_path = droid_dir / "episode_metadata.json"
    if not manifest_path.exists():
        summary = TraceSummary(name="droid_48k", paths=[str(manifest_path)])
        if skip_missing:
            summary.status = "missing"
        else:
            summary.fail(f"missing DROID manifest: {manifest_path}")
        return summary

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    shard_paths = [Path(item["file"]) for item in manifest.get("shards", [])]
    allow_null_coords = bool(manifest.get("allows_null_coordinate", True))
    summary = summarize_rows(
        name="droid_48k",
        paths=shard_paths,
        allow_null_coords=allow_null_coords,
        expected_episodes=48000,
        expected_min_episode=0,
        expected_max_episode=47999,
        expected_rows=manifest.get("total_rows"),
    )
    summary.metadata["manifest_path"] = str(manifest_path)
    summary.metadata["manifest_selected_episodes"] = manifest.get("selected_episodes")
    summary.metadata["manifest_total_rows"] = manifest.get("total_rows")
    summary.metadata["manifest_valid_coordinate_rows"] = manifest.get("valid_coordinate_rows")
    summary.metadata["manifest_null_coordinate_rows"] = manifest.get("null_coordinate_rows")
    summary.metadata["manifest_allows_null_coordinate"] = allow_null_coords
    summary.metadata["manifest_coordinate_source_counts"] = manifest.get("coordinate_source_counts")
    summary.metadata["manifest_source_null_coordinate_rows"] = manifest.get(
        "source_null_coordinate_rows"
    )
    summary.metadata["manifest_filled_coordinate_rows"] = manifest.get("filled_coordinate_rows")

    if manifest.get("selected_episodes") != 48000:
        summary.fail(f"manifest selected_episodes is not 48000: {manifest.get('selected_episodes')}")
    if manifest.get("valid_coordinate_rows") is not None and summary.valid_coords != manifest.get(
        "valid_coordinate_rows"
    ):
        summary.fail(
            "manifest valid_coordinate_rows mismatch: "
            f"{manifest.get('valid_coordinate_rows')} vs scanned {summary.valid_coords}"
        )
    if manifest.get("null_coordinate_rows") is not None and summary.null_coords != manifest.get(
        "null_coordinate_rows"
    ):
        summary.fail(
            "manifest null_coordinate_rows mismatch: "
            f"{manifest.get('null_coordinate_rows')} vs scanned {summary.null_coords}"
        )
    if not allow_null_coords and summary.null_coords:
        summary.fail(f"DROID dense bundle still has {summary.null_coords} null coordinates")
    if metadata_path.exists():
        metadata = json.load(metadata_path.open("r", encoding="utf-8"))
        summary.metadata["metadata_rows"] = len(metadata)
        if len(metadata) != 48000:
            summary.fail(f"metadata has {len(metadata)} rows, expected 48000")
    else:
        summary.fail(f"missing DROID metadata: {metadata_path}")
    return summary


def as_dict(summary: TraceSummary) -> Dict:
    return {
        "name": summary.name,
        "status": summary.status,
        "paths": summary.paths,
        "rows": summary.rows,
        "episodes": summary.episodes,
        "min_episode": summary.min_episode,
        "max_episode": summary.max_episode,
        "keyframes": summary.keyframes,
        "interpolated": summary.interpolated,
        "valid_coords": summary.valid_coords,
        "null_coords": summary.null_coords,
        "coord_bounds": {
            "min_x": summary.coord_min_x,
            "max_x": summary.coord_max_x,
            "min_y": summary.coord_min_y,
            "max_y": summary.coord_max_y,
        },
        "noncontiguous_episodes_sample": summary.noncontiguous_episodes,
        "bad_rows_sample": summary.bad_rows,
        "metadata": summary.metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace_root", type=Path, default=TRACE_ROOT)
    parser.add_argument("--droid_dir", type=Path, default=DEFAULT_DROID_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["bcz", "fractal", "bridge", "droid"],
        choices=["bcz", "fractal", "bridge", "droid", "droid_smoke"],
    )
    parser.add_argument(
        "--droid_smoke_dir",
        type=Path,
        default=Path("/path/to/droid_smoke_dir"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries: List[TraceSummary] = []
    root = args.trace_root

    if "bcz" in args.datasets:
        summaries.append(
            summarize_rows(
                name="bcz_train",
                paths=[root / "bcz_annotation_handoff/annotations/bcz_stage2_dense_trace.json"],
                allow_null_coords=False,
                expected_rows=5471693,
                expected_episodes=39350,
                expected_min_episode=0,
                expected_max_episode=39349,
            )
        )
    if "fractal" in args.datasets:
        summaries.append(
            summarize_rows(
                name="fractal_train",
                paths=[root / "fractal_annotation_handoff/annotations/fractal_stage2_dense_trace.json"],
                allow_null_coords=False,
                expected_rows=3786274,
                expected_episodes=87182,
                expected_min_episode=0,
                expected_max_episode=87211,
            )
        )
    if "bridge" in args.datasets:
        summaries.append(
            summarize_rows(
                name="bridge_train",
                paths=[root / "bridge_dense_trace_handoff/annotations/bridge_stage2_dense_trace.json"],
                allow_null_coords=False,
                expected_rows=1999410,
                expected_episodes=53192,
                expected_min_episode=0,
                expected_max_episode=53191,
            )
        )
    if "droid" in args.datasets:
        summaries.append(verify_droid(args.droid_dir, skip_missing=args.skip_missing))
    if "droid_smoke" in args.datasets:
        smoke_manifest = args.droid_smoke_dir / "manifest.json"
        if smoke_manifest.exists():
            with smoke_manifest.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            summaries.append(
                summarize_rows(
                    name="droid_3k_smoke",
                    paths=[Path(item["file"]) for item in manifest.get("shards", [])],
                    allow_null_coords=True,
                    expected_rows=manifest.get("total_rows"),
                    expected_episodes=manifest.get("selected_episodes"),
                    expected_min_episode=0,
                    expected_max_episode=manifest.get("selected_episodes", 1) - 1,
                )
            )
        else:
            missing = TraceSummary(name="droid_3k_smoke", paths=[str(smoke_manifest)])
            missing.status = "missing" if args.skip_missing else "failed"
            missing.bad_rows.append(f"missing smoke manifest: {smoke_manifest}")
            summaries.append(missing)

    result = {
        "status": "ok" if all(item.status in {"ok", "missing"} for item in summaries) else "failed",
        "summaries": [as_dict(item) for item in summaries],
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp.write_text(text + "\n", encoding="utf-8")
        os.replace(tmp, args.output)
    print(text)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
