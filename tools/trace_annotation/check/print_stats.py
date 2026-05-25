"""Print summary statistics of a BC-Z annotation file."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter


def main(path: str) -> int:
    print(f"file: {path}")
    print(f"size: {os.path.getsize(path) / (1024**2):.1f} MB")
    with open(path) as f:
        rows = json.load(f)
    print(f"rows: {len(rows):,}")
    eps = Counter(int(r["episode_idx"]) for r in rows)
    print(f"unique episodes: {len(eps):,}")
    print(f"  min ep_idx: {min(eps)}")
    print(f"  max ep_idx: {max(eps)}")
    kf = sum(1 for r in rows if r.get("is_keyframe"))
    interp = sum(1 for r in rows if r.get("is_interpolated"))
    print(f"keyframes: {kf:,}")
    if interp:
        print(f"interpolated steps: {interp:,}")
    lens = sorted(eps.values())
    print(f"rows per episode: min={lens[0]} median={lens[len(lens)//2]} max={lens[-1]}")
    missing = sorted(set(range(39350)) - set(eps))
    print(f"missing episodes vs train[0,39350): {len(missing)}")
    if missing:
        print(f"  first missing: {missing[:20]}")
    print("\nschema (keys present in row 0):")
    for k, v in rows[0].items():
        print(f"  {k}: {type(v).__name__} = {v}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
