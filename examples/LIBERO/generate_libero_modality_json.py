#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_DATASETS = (
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot",
)

STATE_KEYS = [
    ("x", 0, 1),
    ("y", 1, 2),
    ("z", 2, 3),
    ("roll", 3, 4),
    ("pitch", 4, 5),
    ("yaw", 5, 6),
    ("pad", 6, 7),
    ("gripper", 7, 8),
]

ACTION_KEYS = [
    ("x", 0, 1),
    ("y", 1, 2),
    ("z", 2, 3),
    ("roll", 3, 4),
    ("pitch", 4, 5),
    ("yaw", 5, 6),
    ("gripper", 6, 7),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SemanticVLA-compatible modality.json files for LIBERO LeRobot datasets.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    return parser.parse_args()


def build_modality_payload(info: dict) -> dict:
    features = info["features"]

    if "observation.images.image" not in features or "observation.images.wrist_image" not in features:
        raise KeyError("Expected both observation.images.image and observation.images.wrist_image in info.json")
    if features["observation.state"]["shape"] != [8]:
        raise ValueError(f"Expected observation.state shape [8], got {features['observation.state']['shape']}")
    if features["action"]["shape"] != [7]:
        raise ValueError(f"Expected action shape [7], got {features['action']['shape']}")

    state = {}
    for key, start, end in STATE_KEYS:
        state[key] = {
            "start": start,
            "end": end,
            "original_key": "observation.state",
            "dtype": "float32",
            "absolute": True,
        }

    action = {}
    for key, start, end in ACTION_KEYS:
        action[key] = {
            "start": start,
            "end": end,
            "original_key": "action",
            "dtype": "float32",
            "absolute": False,
        }

    return {
        "state": state,
        "action": action,
        "video": {
            "primary_image": {
                "original_key": "observation.images.image",
            },
            "wrist_image": {
                "original_key": "observation.images.wrist_image",
            },
        },
        "annotation": {
            "human.action.task_description": {
                "original_key": "task_index",
            }
        },
    }


def main() -> int:
    args = parse_args()

    for dataset_name in args.datasets:
        dataset_root = args.data_root / dataset_name
        info_path = dataset_root / "meta" / "info.json"
        modality_path = dataset_root / "meta" / "modality.json"

        if not info_path.is_file():
            raise FileNotFoundError(f"Missing info.json: {info_path}")

        info = json.loads(info_path.read_text())
        modality = build_modality_payload(info)
        modality_path.write_text(json.dumps(modality, indent=2) + "\n")
        print(f"Wrote {modality_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
