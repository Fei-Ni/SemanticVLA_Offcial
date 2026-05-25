#!/usr/bin/env python3
"""Verify LeRobot datasets that should contain per-frame trace columns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


DEFAULT_EXPECTED = {
    "bcz_train": {
        "root": "${WORK_ROOT}/tracex/bcz_train_0.1.0_lerobot",
        "episodes": 39350,
        "frames": 5471693,
    },
    "bridge": {
        "root": "${WORK_ROOT}/tracex/bridge_train_1.0.0_lerobot",
        "episodes": 53192,
        "frames": 1999410,
    },
    "fractal": {
        "root": "${WORK_ROOT}/tracex/fractal_train_0.1.0_lerobot",
        "episodes": 87182,
        "frames": 3786274,
    },
    "bcz_val": {
        "root": "${WORK_ROOT}/tracex/bcz_val_0.1.0_lerobot",
        "episodes": 3914,
    },
    "droid_56k": {
        "root": "${WORK_ROOT}/tracex/droid_56k_1.0.1_lerobot",
        "episodes": 56362,
    },
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _episode_parquet(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunk_size = int(info["chunks_size"])
    return root / info["data_path"].format(
        episode_chunk=episode_index // chunk_size,
        chunk_index=episode_index // chunk_size,
        episode_index=episode_index,
        file_index=episode_index,
    )


def verify_one(name: str, root: Path, expected: dict[str, Any], trace_field: str) -> dict[str, Any]:
    info_path = root / "meta/info.json"
    episodes_path = root / "meta/episodes.jsonl"
    if not info_path.exists():
        raise FileNotFoundError(info_path)
    if not episodes_path.exists():
        raise FileNotFoundError(episodes_path)

    info = json.loads(info_path.read_text(encoding="utf-8"))
    episodes = _read_jsonl(episodes_path)
    if int(info["total_episodes"]) != len(episodes):
        raise RuntimeError(f"{name}: info episodes {info['total_episodes']} != jsonl {len(episodes)}")
    if "episodes" in expected and int(info["total_episodes"]) != int(expected["episodes"]):
        raise RuntimeError(f"{name}: episodes {info['total_episodes']} != expected {expected['episodes']}")
    if "frames" in expected and int(info["total_frames"]) != int(expected["frames"]):
        raise RuntimeError(f"{name}: frames {info['total_frames']} != expected {expected['frames']}")
    if trace_field not in info.get("features", {}):
        raise RuntimeError(f"{name}: missing {trace_field} in meta/info.json features")

    sample_eps = [episodes[0], episodes[-1]]
    sample_rows = []
    for episode in sample_eps:
        episode_index = int(episode["episode_index"])
        parquet_path = _episode_parquet(root, info, episode_index)
        table = pq.read_table(parquet_path, columns=[trace_field, "episode_index", "frame_index"])
        if table.num_rows != int(episode["length"]):
            raise RuntimeError(
                f"{name}: {parquet_path} rows {table.num_rows} != episode length {episode['length']}"
            )
        sample_rows.append(
            {
                "episode_index": episode_index,
                "parquet": str(parquet_path),
                "rows": table.num_rows,
                "first_trace": table[trace_field][0].as_py(),
                "last_trace": table[trace_field][-1].as_py(),
            }
        )

    return {
        "name": name,
        "root": str(root),
        "episodes": int(info["total_episodes"]),
        "frames": int(info["total_frames"]),
        "trace_field": trace_field,
        "samples": sample_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-field", default="observation.trace.xy")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="NAME=ROOT",
        help="Override or add dataset root. Expected counts are used only for known names.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = {name: dict(spec) for name, spec in DEFAULT_EXPECTED.items()}
    for item in args.dataset:
        if "=" not in item:
            raise ValueError(f"--dataset must be NAME=ROOT, got {item!r}")
        name, root = item.split("=", 1)
        specs.setdefault(name, {})["root"] = root

    results = []
    for name, spec in specs.items():
        results.append(verify_one(name, Path(spec["root"]), spec, args.trace_field))

    total_episodes = sum(int(item["episodes"]) for item in results)
    if total_episodes != 240000:
        raise RuntimeError(f"total episodes {total_episodes} != expected Trace-240K count 240000")
    payload = {
        "trace_field": args.trace_field,
        "total_episodes": total_episodes,
        "expected_total_episodes": 240000,
        "datasets": results,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
