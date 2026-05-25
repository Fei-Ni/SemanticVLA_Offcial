#!/usr/bin/env python3
"""Normalize BC-Z val dense trace handoff episode ids.

External Stage2 handoffs may use either:
  - val-local episode ids: 0..3913
  - global Trace-240K episode ids: 39350..43263

The release pipeline expects global ids because BC-Z train and val share one
continuous episode id space. This script validates the handoff and writes a
canonical global-id JSON array.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterator


VAL_LOCAL_START = 0
VAL_LOCAL_END = 3913
VAL_GLOBAL_START = 39350
VAL_GLOBAL_END = 43263
VAL_EPISODES = VAL_LOCAL_END - VAL_LOCAL_START + 1


def _read_more(handle, buffer: str, chunk_size: int) -> tuple[str, bool]:
    chunk = handle.read(chunk_size)
    if chunk:
        return buffer + chunk, False
    return buffer, True


def iter_json_array(path: Path, *, chunk_size: int = 1 << 20) -> Iterator[dict[str, Any]]:
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


def _scan(path: Path) -> dict[str, Any]:
    row_count = 0
    episodes: set[int] = set()
    min_episode: int | None = None
    max_episode: int | None = None
    missing_coordinate = 0
    bad_coordinate = 0

    for row in iter_json_array(path):
        ep = int(row["episode_idx"])
        row_count += 1
        episodes.add(ep)
        min_episode = ep if min_episode is None else min(min_episode, ep)
        max_episode = ep if max_episode is None else max(max_episode, ep)
        coord = row.get("coordinate")
        if coord is None:
            missing_coordinate += 1
        elif not isinstance(coord, list | tuple) or len(coord) != 2:
            bad_coordinate += 1

    return {
        "row_count": row_count,
        "episodes": len(episodes),
        "min_episode": min_episode,
        "max_episode": max_episode,
        "missing_coordinate": missing_coordinate,
        "bad_coordinate": bad_coordinate,
    }


def _infer_offset(stats: dict[str, Any], *, allow_partial: bool = False) -> int:
    min_ep = stats["min_episode"]
    max_ep = stats["max_episode"]
    episodes = stats["episodes"]
    if episodes != VAL_EPISODES and not allow_partial:
        raise ValueError(f"expected {VAL_EPISODES} val episodes, got {episodes}")
    if min_ep == VAL_GLOBAL_START and max_ep == VAL_GLOBAL_END:
        return 0
    if min_ep == VAL_LOCAL_START and max_ep == VAL_LOCAL_END:
        return VAL_GLOBAL_START
    if allow_partial:
        if VAL_GLOBAL_START <= min_ep <= max_ep <= VAL_GLOBAL_END:
            return 0
        if VAL_LOCAL_START <= min_ep <= max_ep <= VAL_LOCAL_END:
            return VAL_GLOBAL_START
    raise ValueError(
        "BC-Z val episode range must be either "
        f"{VAL_LOCAL_START}..{VAL_LOCAL_END} or {VAL_GLOBAL_START}..{VAL_GLOBAL_END}; "
        f"got {min_ep}..{max_ep}"
    )


def _write_normalized(in_path: Path, out_path: Path, *, offset: int, overwrite: bool) -> None:
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists; pass --overwrite")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write("[")
        first = True
        for row in iter_json_array(in_path):
            if offset:
                row = dict(row)
                row["episode_idx"] = int(row["episode_idx"]) + offset
            if not first:
                handle.write(",")
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            first = False
        handle.write("]\n")
    os.replace(tmp_path, out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--allow-partial", action="store_true", help="Only for smoke tests; production checks require all val episodes.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats = _scan(args.input)
    if stats["row_count"] <= 0:
        raise ValueError(f"{args.input} has no rows")
    if stats["missing_coordinate"] or stats["bad_coordinate"]:
        raise ValueError(
            f"invalid coordinate rows: missing={stats['missing_coordinate']} "
            f"bad={stats['bad_coordinate']}"
        )
    offset = _infer_offset(stats, allow_partial=args.allow_partial)
    print({"input": str(args.input), **stats, "episode_offset_to_apply": offset}, flush=True)

    if args.check_only:
        if offset != 0:
            raise ValueError("check-only input is val-local; normalize it before packaging")
        return 0
    if args.output is None:
        raise ValueError("--output is required unless --check-only is set")
    _write_normalized(args.input, args.output, offset=offset, overwrite=args.overwrite)

    out_stats = _scan(args.output)
    out_offset = _infer_offset(out_stats, allow_partial=args.allow_partial)
    if out_offset != 0:
        raise RuntimeError(f"normalization did not produce global ids: {out_stats}")
    print({"output": str(args.output), **out_stats}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
