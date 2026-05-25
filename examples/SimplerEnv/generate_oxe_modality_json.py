#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASET_SPECS = {
    "bridge_orig_1.0.0_lerobot": {
        "robot_type": "widowx",
        "video_key": "image_0",
        "video_original_key": "observation.images.image_0",
        "state_keys": [
            ("x", 0, 1),
            ("y", 1, 2),
            ("z", 2, 3),
            ("roll", 3, 4),
            ("pitch", 4, 5),
            ("yaw", 5, 6),
            ("pad", 6, 7),
            ("gripper", 7, 8),
        ],
        "action_keys": [
            ("x", 0, 1),
            ("y", 1, 2),
            ("z", 2, 3),
            ("roll", 3, 4),
            ("pitch", 4, 5),
            ("yaw", 5, 6),
            ("gripper", 6, 7),
        ],
    },
    "fractal20220817_data_0.1.0_lerobot": {
        "robot_type": "google_robot",
        "video_key": "image",
        "video_original_key": "observation.images.image",
        "state_keys": [
            ("x", 0, 1),
            ("y", 1, 2),
            ("z", 2, 3),
            ("rx", 3, 4),
            ("ry", 4, 5),
            ("rz", 5, 6),
            ("rw", 6, 7),
            ("gripper", 7, 8),
        ],
        "action_keys": [
            ("x", 0, 1),
            ("y", 1, 2),
            ("z", 2, 3),
            ("roll", 3, 4),
            ("pitch", 4, 5),
            ("yaw", 5, 6),
            ("gripper", 6, 7),
        ],
    },
    "bcz_0.1.0_lerobot": {
        "robot_type": "google_robot",
        "video_key": "image",
        "video_original_key": "observation.images.image",
        "state_keys": [
            ("x", 0, 1),
            ("y", 1, 2),
            ("z", 2, 3),
            ("rx", 3, 4),
            ("ry", 4, 5),
            ("rz", 5, 6),
            ("gripper", 6, 7),
        ],
        "action_keys": [
            ("x", 0, 1),
            ("y", 1, 2),
            ("z", 2, 3),
            ("roll", 3, 4),
            ("pitch", 4, 5),
            ("yaw", 5, 6),
            ("gripper", 6, 7),
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SemanticVLA-compatible modality.json files for OXE Bridge/RT1 LeRobot datasets."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_SPECS))
    return parser.parse_args()


def _build_state_or_action(keys: list[tuple[str, int, int]], *, original_key: str, absolute: bool) -> dict:
    payload = {}
    for key, start, end in keys:
        payload[key] = {
            "start": start,
            "end": end,
            "original_key": original_key,
            "dtype": "float32",
            "absolute": absolute,
        }
    return payload


def build_modality_payload(info: dict, dataset_name: str) -> dict:
    if dataset_name not in DATASET_SPECS:
        raise KeyError(f"Unsupported dataset for modality generation: {dataset_name}")

    spec = DATASET_SPECS[dataset_name]
    features = info["features"]

    if info.get("robot_type") != spec["robot_type"]:
        raise ValueError(
            f"{dataset_name}: expected robot_type={spec['robot_type']}, got {info.get('robot_type')}"
        )
    if spec["video_original_key"] not in features:
        raise KeyError(
            f"{dataset_name}: expected video feature {spec['video_original_key']} in info.json"
        )
    if "observation.state" not in features:
        raise KeyError(f"{dataset_name}: missing observation.state in info.json")
    if "action" not in features:
        raise KeyError(f"{dataset_name}: missing action in info.json")
    if "task_index" not in features:
        raise KeyError(f"{dataset_name}: missing task_index in info.json")

    expected_state_dim = spec["state_keys"][-1][2]
    expected_action_dim = spec["action_keys"][-1][2]
    if features["observation.state"]["shape"] != [expected_state_dim]:
        raise ValueError(
            f"{dataset_name}: expected observation.state shape {[expected_state_dim]}, "
            f"got {features['observation.state']['shape']}"
        )
    if features["action"]["shape"] != [expected_action_dim]:
        raise ValueError(
            f"{dataset_name}: expected action shape {[expected_action_dim]}, got {features['action']['shape']}"
        )

    return {
        "state": _build_state_or_action(
            spec["state_keys"], original_key="observation.state", absolute=True
        ),
        "action": _build_state_or_action(
            spec["action_keys"], original_key="action", absolute=False
        ),
        "video": {
            spec["video_key"]: {
                "original_key": spec["video_original_key"],
            }
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
        modality = build_modality_payload(info, dataset_name)
        modality_path.write_text(json.dumps(modality, indent=2) + "\n")
        print(f"Wrote {modality_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
