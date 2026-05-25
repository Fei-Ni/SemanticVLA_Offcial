#!/usr/bin/env python3
"""Build mmap-friendly NPY indexes from dense trace JSON files.

The trace handoff files are large flat JSON arrays. Loading them in every
DataLoader worker would be wasteful, so this script converts each dataset to:

  - <dataset>_coords.npy: float32 array with shape [num_rows, 2]
  - <dataset>_offsets.npy: int64 prefix-sum offsets by episode id
  - meta.json: paths, counts, and validation stats

The coords arrays are written with numpy's .npy format so downstream code can
use np.load(path, mmap_mode="r").
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np


TRACE_ROOT = Path("${TRACE_ANNOTATIONS_ROOT}")
DEFAULT_OUTPUT = TRACE_ROOT / "_npy_index"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    episode_start: int
    episode_end: int
    allows_null_coordinate: bool = False
    trace_file: Path | None = None
    trace_glob: str | None = None

    @property
    def episode_slots(self) -> int:
        return self.episode_end - self.episode_start + 1

    def trace_files(self) -> list[Path]:
        if self.trace_file is not None:
            return [self.trace_file]
        if self.trace_glob is not None:
            return [Path(path) for path in sorted(glob.glob(self.trace_glob))]
        raise ValueError(f"{self.name}: trace_file or trace_glob is required")


DATASETS: dict[str, DatasetSpec] = {
    "bcz": DatasetSpec(
        name="bcz",
        episode_start=0,
        episode_end=39349,
        trace_file=TRACE_ROOT / "bcz_annotation_handoff/annotations/bcz_stage2_dense_trace.json",
    ),
    "bcz_val": DatasetSpec(
        name="bcz_val",
        episode_start=39350,
        episode_end=43263,
        trace_file=Path(
            "${DATA_ROOT}/bcz_val_trace_runs/bcz_val_stage2_dense_trace.json"
        ),
    ),
    "bridge": DatasetSpec(
        name="bridge",
        episode_start=0,
        episode_end=53191,
        trace_file=TRACE_ROOT / "bridge_dense_trace_handoff/annotations/bridge_stage2_dense_trace.json",
    ),
    "fractal": DatasetSpec(
        name="fractal",
        episode_start=0,
        episode_end=87211,
        trace_file=TRACE_ROOT / "fractal_annotation_handoff/annotations/fractal_stage2_dense_trace.json",
    ),
    "droid_48k": DatasetSpec(
        name="droid_48k",
        episode_start=0,
        episode_end=47999,
        trace_glob=(
            "${DATA_ROOT}/droid_projection_runs/"
            "droid_48k_projection_trace_clip_dense_v2/annotations/*.json"
        ),
    ),
    "droid_56k": DatasetSpec(
        name="droid_56k",
        episode_start=0,
        episode_end=56361,
        trace_glob=(
            "${DATA_ROOT}/droid_projection_runs/"
            "droid_56k_projection_trace_clip_dense_v1/annotations/*.json"
        ),
    ),
}


def _read_more(handle, buffer: str, chunk_size: int) -> tuple[str, bool]:
    chunk = handle.read(chunk_size)
    if chunk:
        return buffer + chunk, False
    return buffer, True


def iter_json_array(path: Path, *, chunk_size: int = 1 << 20) -> Iterator[dict[str, Any]]:
    """Yield objects from a top-level JSON array without retaining the array."""

    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        pos = 0
        eof = False
        started = False

        while True:
            if pos > chunk_size:
                buffer = buffer[pos:]
                pos = 0

            while True:
                if pos >= len(buffer) and not eof:
                    buffer, eof = _read_more(handle, buffer, chunk_size)
                    continue
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if not started:
                    if pos >= len(buffer):
                        if eof:
                            raise ValueError(f"{path} is empty")
                        continue
                    if buffer[pos] != "[":
                        raise ValueError(f"{path} is not a top-level JSON array")
                    started = True
                    pos += 1
                    continue
                if pos >= len(buffer):
                    if eof:
                        raise ValueError(f"{path} ended before closing array")
                    continue
                if buffer[pos] == ",":
                    pos += 1
                    continue
                if buffer[pos] == "]":
                    return
                break

            while True:
                try:
                    obj, end = decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    buffer, eof = _read_more(handle, buffer, chunk_size)
                    continue
                pos = end
                if not isinstance(obj, dict):
                    raise ValueError(f"Expected object row in {path}, got {type(obj).__name__}")
                yield obj
                break


def _episode_to_slot(spec: DatasetSpec, episode_idx: int) -> int:
    if episode_idx < spec.episode_start or episode_idx > spec.episode_end:
        raise ValueError(
            f"{spec.name}: episode_idx {episode_idx} outside "
            f"[{spec.episode_start}, {spec.episode_end}]"
        )
    return episode_idx - spec.episode_start


def _coordinate(row: dict[str, Any], spec: DatasetSpec) -> tuple[float, float] | None:
    coord = row.get("coordinate")
    if coord is None:
        if spec.allows_null_coordinate:
            return None
        raise ValueError(f"{spec.name}: null coordinate at episode={row.get('episode_idx')} step={row.get('step_idx')}")
    if not isinstance(coord, list | tuple) or len(coord) != 2:
        raise ValueError(f"{spec.name}: invalid coordinate {coord!r}")
    return float(coord[0]), float(coord[1])


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(path.stem + ".tmp.npy")
    with tmp.open("wb") as handle:
        np.save(handle, array)
    os.replace(tmp, path)


def build_one(
    spec: DatasetSpec,
    *,
    output_dir: Path,
    overwrite: bool,
    max_rows: int | None,
    chunk_size: int,
) -> dict[str, Any]:
    trace_files = spec.trace_files()
    if not trace_files:
        raise FileNotFoundError(
            f"{spec.name}: no trace files matched {spec.trace_file or spec.trace_glob}"
        )
    for trace_file in trace_files:
        if not trace_file.exists():
            raise FileNotFoundError(trace_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    coords_path = output_dir / f"{spec.name}_coords.npy"
    offsets_path = output_dir / f"{spec.name}_offsets.npy"
    present_path = output_dir / f"{spec.name}_present.npy"
    for path in (coords_path, offsets_path, present_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace")

    counts = np.zeros(spec.episode_slots, dtype=np.int64)
    row_count = 0
    min_coord = np.array([np.inf, np.inf], dtype=np.float64)
    max_coord = np.array([-np.inf, -np.inf], dtype=np.float64)

    for trace_file in trace_files:
        for row in iter_json_array(trace_file, chunk_size=chunk_size):
            if max_rows is not None and row_count >= max_rows:
                break
            ep = int(row["episode_idx"])
            slot = _episode_to_slot(spec, ep)
            coord = _coordinate(row, spec)
            if coord is None:
                continue
            counts[slot] += 1
            min_coord = np.minimum(min_coord, np.asarray(coord, dtype=np.float64))
            max_coord = np.maximum(max_coord, np.asarray(coord, dtype=np.float64))
            row_count += 1
        if max_rows is not None and row_count >= max_rows:
            break

    offsets = np.zeros(spec.episode_slots + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    present = counts > 0

    tmp_coords_path = coords_path.with_name(coords_path.stem + ".tmp.npy")
    if tmp_coords_path.exists():
        tmp_coords_path.unlink()
    coords = np.lib.format.open_memmap(
        tmp_coords_path,
        mode="w+",
        dtype=np.float32,
        shape=(int(row_count), 2),
    )
    fill_counts = np.zeros(spec.episode_slots, dtype=np.int64)
    written = 0
    step_mismatches: list[dict[str, int]] = []

    for trace_file in trace_files:
        for row in iter_json_array(trace_file, chunk_size=chunk_size):
            if max_rows is not None and written >= row_count:
                break
            ep = int(row["episode_idx"])
            slot = _episode_to_slot(spec, ep)
            coord = _coordinate(row, spec)
            if coord is None:
                continue
            expected_step = int(fill_counts[slot])
            step_idx = int(row.get("step_idx", expected_step))
            if step_idx != expected_step and len(step_mismatches) < 20:
                step_mismatches.append(
                    {"episode_idx": ep, "expected_step": expected_step, "actual_step": step_idx}
                )
            write_idx = int(offsets[slot] + fill_counts[slot])
            coords[write_idx] = coord
            fill_counts[slot] += 1
            written += 1
        if max_rows is not None and written >= row_count:
            break

    coords.flush()
    del coords
    if written != row_count:
        raise RuntimeError(f"{spec.name}: wrote {written} rows, expected {row_count}")
    if not np.array_equal(fill_counts, counts):
        raise RuntimeError(f"{spec.name}: second pass counts differ from first pass")
    os.replace(tmp_coords_path, coords_path)
    _save_npy_atomic(offsets_path, offsets)
    _save_npy_atomic(present_path, present)

    missing = np.flatnonzero(~present).astype(int) + spec.episode_start
    nonzero_counts = counts[present]
    if len(nonzero_counts):
        count_stats = {
            "min": int(nonzero_counts.min()),
            "median": float(np.median(nonzero_counts)),
            "max": int(nonzero_counts.max()),
        }
    else:
        count_stats = {"min": 0, "median": 0.0, "max": 0}

    if not np.isfinite(min_coord).all():
        min_coord = np.array([np.nan, np.nan])
        max_coord = np.array([np.nan, np.nan])

    return {
        "name": spec.name,
        "source_json_files": [str(path) for path in trace_files],
        "source_json": str(trace_files[0]) if len(trace_files) == 1 else None,
        "source_glob": spec.trace_glob,
        "coords_path": coords_path.name,
        "offsets_path": offsets_path.name,
        "present_path": present_path.name,
        "coords_dtype": "float32",
        "coords_shape": [int(row_count), 2],
        "offsets_shape": [int(offsets.shape[0])],
        "episode_index_range": [spec.episode_start, spec.episode_end],
        "episode_slots": spec.episode_slots,
        "present_episodes": int(present.sum()),
        "missing_episode_count": int((~present).sum()),
        "missing_episodes_preview": missing[:50].tolist(),
        "row_count": int(row_count),
        "rows_per_present_episode": count_stats,
        "coordinate_min": [float(min_coord[0]), float(min_coord[1])],
        "coordinate_max": [float(max_coord[0]), float(max_coord[1])],
        "step_order_mismatch_count_previewed": len(step_mismatches),
        "step_order_mismatches_preview": step_mismatches,
        "truncated_by_max_rows": max_rows is not None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["bridge", "fractal", "bcz"],
        choices=sorted(DATASETS),
        help="Datasets to index.",
    )
    parser.add_argument(
        "--trace-file",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Override a dataset's flat trace JSON path.",
    )
    parser.add_argument(
        "--trace-glob",
        action="append",
        default=[],
        metavar="DATASET=GLOB",
        help="Override a dataset's sharded trace JSON glob.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-rows-per-dataset",
        type=int,
        default=None,
        help="Optional smoke limit. Writes a truncated index for the first N rows per dataset.",
    )
    parser.add_argument("--chunk-size", type=int, default=1 << 20)
    return parser.parse_args()


def _parse_overrides(items: list[str], kind: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{kind} override must be DATASET=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        if name not in DATASETS:
            raise ValueError(f"unknown dataset in {kind} override: {name}")
        out[name] = value
    return out


def _apply_overrides(
    spec: DatasetSpec,
    *,
    trace_files: dict[str, str],
    trace_globs: dict[str, str],
) -> DatasetSpec:
    if spec.name in trace_files and spec.name in trace_globs:
        raise ValueError(f"{spec.name}: use only one of --trace-file or --trace-glob")
    if spec.name in trace_files:
        return DatasetSpec(
            name=spec.name,
            episode_start=spec.episode_start,
            episode_end=spec.episode_end,
            allows_null_coordinate=spec.allows_null_coordinate,
            trace_file=Path(trace_files[spec.name]),
        )
    if spec.name in trace_globs:
        return DatasetSpec(
            name=spec.name,
            episode_start=spec.episode_start,
            episode_end=spec.episode_end,
            allows_null_coordinate=spec.allows_null_coordinate,
            trace_glob=trace_globs[spec.name],
        )
    return spec


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_file_overrides = _parse_overrides(args.trace_file, "--trace-file")
    trace_glob_overrides = _parse_overrides(args.trace_glob, "--trace-glob")

    meta_path = output_dir / "meta.json"
    if meta_path.exists() and not args.overwrite:
        raise FileExistsError(f"{meta_path} exists; pass --overwrite to replace")

    meta: dict[str, Any] = {
        "schema_version": 1,
        "format": "per-array-npy",
        "coordinate_space": "normalized_image_xy_0_100",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "datasets": {},
    }

    for dataset in args.datasets:
        spec = _apply_overrides(
            DATASETS[dataset],
            trace_files=trace_file_overrides,
            trace_globs=trace_glob_overrides,
        )
        print(
            f"[build_trace_index] {dataset}: {spec.trace_file or spec.trace_glob}",
            flush=True,
        )
        meta["datasets"][dataset] = build_one(
            spec,
            output_dir=output_dir,
            overwrite=args.overwrite,
            max_rows=args.max_rows_per_dataset,
            chunk_size=args.chunk_size,
        )
        ds_meta = meta["datasets"][dataset]
        print(
            f"[build_trace_index] {dataset}: rows={ds_meta['row_count']} "
            f"episodes={ds_meta['present_episodes']} missing={ds_meta['missing_episode_count']}",
            flush=True,
        )

    tmp_meta = meta_path.with_suffix(".tmp.json")
    tmp_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_meta, meta_path)
    print(f"[build_trace_index] wrote {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
