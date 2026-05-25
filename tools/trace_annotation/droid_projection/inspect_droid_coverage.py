#!/usr/bin/env python3
"""Summarize DROID calibration coverage for projection-based trace generation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Set


CAMERA_CONFIGS = {
    "exterior_image_1_left": ("ext1_cam_serial", "exterior_image_1_left"),
    "exterior_image_2_left": ("ext2_cam_serial", "exterior_image_2_left"),
    "wrist_image_left": ("wrist_cam_serial", "wrist_image_left"),
}


def load_json(root: Path, name: str) -> Dict:
    with (root / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def numeric_serials(payload: Optional[Dict]) -> Set[str]:
    if not isinstance(payload, dict):
        return set()
    out: Set[str] = set()
    for key, value in payload.items():
        if str(key).isdigit() and isinstance(value, list) and len(value) == 6:
            out.add(str(key))
    return out


def get_serial(serials: Dict, episode_id: str, keys: Iterable[str]) -> Optional[str]:
    serial_map = serials.get(episode_id) or {}
    for key in keys:
        value = serial_map.get(key)
        if value:
            return str(value)
    values = [
        str(value)
        for value in serial_map.values()
        if isinstance(value, (int, str)) and str(value).strip().isdigit()
    ]
    if len(values) == 1:
        return values[0]
    return None


def has_language(language: Dict, episode_id: str) -> bool:
    entry = language.get(episode_id)
    return isinstance(entry, dict) and any(
        entry.get(key)
        for key in (
            "language_instruction1",
            "language_instruction2",
            "language_instruction3",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations_dir", type=Path, default=Path("droid"))
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args()

    root = args.annotations_dir.resolve()
    episode_id_to_path = load_json(root, "episode_id_to_path.json")
    language = load_json(root, "droid_language_annotations.json")
    camera_serials = load_json(root, "camera_serials.json")
    intrinsics = load_json(root, "intrinsics.json")
    cam2base = load_json(root, "cam2base_extrinsics.json")
    superset = load_json(root, "cam2base_extrinsic_superset.json")
    keep_ranges = load_json(root, "keep_ranges_1_0_1.json")

    universe = set(episode_id_to_path)
    summary = {
        "annotation_counts": {
            "episode_id_to_path": len(episode_id_to_path),
            "language": len(language),
            "keep_ranges": len(keep_ranges),
            "camera_serials": len(camera_serials),
            "intrinsics": len(intrinsics),
            "cam2base_extrinsics": len(cam2base),
            "cam2base_extrinsic_superset": len(superset),
        },
        "cameras": {},
    }

    camera_good: Dict[str, Set[str]] = {}
    for camera, serial_keys in CAMERA_CONFIGS.items():
        counts = Counter()
        good: Set[str] = set()
        for episode_id in universe:
            counts["episodes"] += 1
            if has_language(language, episode_id):
                counts["has_language"] += 1
            if episode_id in camera_serials:
                counts["has_camera_serials"] += 1
            serial = get_serial(camera_serials, episode_id, serial_keys)
            if serial:
                counts["resolved_serial"] += 1
            if serial and isinstance(intrinsics.get(episode_id), dict) and serial in intrinsics[episode_id]:
                counts["has_intrinsic_for_serial"] += 1
            payload = superset.get(episode_id) or cam2base.get(episode_id)
            if payload:
                counts["has_selected_extrinsic_payload"] += 1
            if serial and payload and serial in numeric_serials(payload):
                counts["has_extrinsic_for_serial"] += 1
            if (
                serial
                and isinstance(intrinsics.get(episode_id), dict)
                and serial in intrinsics[episode_id]
                and payload
                and serial in numeric_serials(payload)
            ):
                counts["projectable"] += 1
                good.add(episode_id)
        camera_good[camera] = good
        summary["cameras"][camera] = dict(counts)

    external = camera_good["exterior_image_1_left"] | camera_good["exterior_image_2_left"]
    summary["external_camera_combined"] = {
        "unique_episodes_ext1_or_ext2": len(external),
        "camera_view_trajectories_ext1_plus_ext2": (
            len(camera_good["exterior_image_1_left"])
            + len(camera_good["exterior_image_2_left"])
        ),
        "episodes_with_both_ext1_and_ext2": len(
            camera_good["exterior_image_1_left"] & camera_good["exterior_image_2_left"]
        ),
    }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
