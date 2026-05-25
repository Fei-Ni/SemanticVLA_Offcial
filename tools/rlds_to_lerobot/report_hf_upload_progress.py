#!/usr/bin/env python3
"""Summarize upload_large_folder progress reports from Slurm logs."""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path


REPORT_RE = re.compile(
    r"-{10} (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?"
    r"Files:\s+hashed (?P<hashed>\d+)/(?:\d+) \((?P<hashed_size>[^)]*)\) \| "
    r"pre-uploaded: (?P<pre>\d+)/(?:\d+) \((?P<pre_size>[^)]*)\) \| "
    r"committed: (?P<committed>\d+)/(?:\d+) \((?P<commit_size>[^)]*)\)",
    re.DOTALL,
)


def parse_size(text: str) -> float:
    first = text.split("/", 1)[0].strip()
    match = re.match(r"([0-9.]+)([kMGT]?B?)", first)
    if not match:
        return 0.0
    value = float(match.group(1))
    unit = match.group(2)
    scale = {
        "": 1.0 / (1024**3),
        "B": 1.0 / (1024**3),
        "kB": 1.0 / (1024**2),
        "KB": 1.0 / (1024**2),
        "M": 1.0 / 1024,
        "MB": 1.0 / 1024,
        "G": 1.0,
        "GB": 1.0,
        "T": 1024.0,
        "TB": 1024.0,
    }[unit]
    return value * scale


def reports(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    out = []
    for match in REPORT_RE.finditer(text):
        out.append(
            {
                "ts": datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S"),
                "hashed": int(match.group("hashed")),
                "pre": int(match.group("pre")),
                "committed": int(match.group("committed")),
                "pre_gb": parse_size(match.group("pre_size")),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.logs:
        rows = reports(path)
        print(f"== {path} ==")
        if not rows:
            print("no upload_large_folder reports yet")
            continue
        last = rows[-1]
        print(
            f"latest {last['ts']} pre={last['pre']} pre_gb={last['pre_gb']:.2f} "
            f"committed={last['committed']} hashed={last['hashed']}"
        )
        if len(rows) >= 2:
            prev = rows[-2]
            hours = (last["ts"] - prev["ts"]).total_seconds() / 3600
            gbph = (float(last["pre_gb"]) - float(prev["pre_gb"])) / hours if hours > 0 else 0.0
            filesph = (int(last["pre"]) - int(prev["pre"])) / hours if hours > 0 else 0.0
            print(f"last_interval speed={gbph:.2f} GB/h files={filesph:.0f}/h")
        if len(rows) >= 4:
            prev = rows[-4]
            hours = (last["ts"] - prev["ts"]).total_seconds() / 3600
            gbph = (float(last["pre_gb"]) - float(prev["pre_gb"])) / hours if hours > 0 else 0.0
            filesph = (int(last["pre"]) - int(prev["pre"])) / hours if hours > 0 else 0.0
            print(f"recent_window speed={gbph:.2f} GB/h files={filesph:.0f}/h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
