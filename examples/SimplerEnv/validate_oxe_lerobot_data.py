#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


DATASET_SPECS = {
    "bridge_orig_1.0.0_lerobot": {
        "robot_type": "widowx",
        "video_key": "image_0",
        "video_original_key": "observation.images.image_0",
        "state_keys": ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"],
        "action_keys": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
    },
    "fractal20220817_data_0.1.0_lerobot": {
        "robot_type": "google_robot",
        "video_key": "image",
        "video_original_key": "observation.images.image",
        "state_keys": ["x", "y", "z", "rx", "ry", "rz", "rw", "gripper"],
        "action_keys": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
    },
    "bcz_0.1.0_lerobot": {
        "robot_type": "google_robot",
        "video_key": "image",
        "video_original_key": "observation.images.image",
        "state_keys": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
        "action_keys": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
    },
}

REQUIRED_FILES = (
    "meta/modality.json",
    "meta/episodes.jsonl",
    "meta/tasks.jsonl",
    "meta/info.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate downloaded OXE Bridge/RT1 LeRobot datasets before SemanticVLA training."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_SPECS))
    parser.add_argument(
        "--sample-episodes",
        type=int,
        default=8,
        help="How many episodes per dataset to inspect deeply.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _sample_episode_indices(episodes: list[dict], sample_count: int) -> list[int]:
    if not episodes:
        return []
    capped = min(max(sample_count, 1), len(episodes))
    if capped == len(episodes):
        return [int(ep["episode_index"]) for ep in episodes]
    if capped == 1:
        return [int(episodes[0]["episode_index"])]

    picks: list[int] = []
    seen: set[int] = set()
    last_idx = len(episodes) - 1
    for i in range(capped):
        episode_pos = round(i * last_idx / (capped - 1))
        episode_idx = int(episodes[episode_pos]["episode_index"])
        if episode_idx not in seen:
            picks.append(episode_idx)
            seen.add(episode_idx)
    return picks


def _format_episode_relative_path(pattern: str, episode_index: int, chunk_size: int, *, video_key: str | None = None) -> Path:
    if "episode_index" not in pattern:
        raise ValueError(
            "Current SemanticVLA OXE validation only supports per-episode LeRobot layouts. "
            f"Unsupported pattern: {pattern}"
        )
    episode_chunk = episode_index // chunk_size
    kwargs = {
        "episode_index": episode_index,
        "episode_chunk": episode_chunk,
        "chunk_index": episode_chunk,
        "file_index": episode_index,
    }
    if video_key is not None:
        kwargs["video_key"] = video_key
    try:
        return Path(pattern.format(**kwargs))
    except KeyError as exc:
        raise ValueError(f"Unsupported path format token {exc} in pattern: {pattern}") from exc


def _validate_modality(modality: dict, dataset_name: str) -> None:
    spec = DATASET_SPECS[dataset_name]

    if "state" not in modality or "action" not in modality or "video" not in modality:
        raise ValueError(f"{dataset_name}: modality.json missing one of state/action/video sections")

    if list(modality["state"].keys()) != spec["state_keys"]:
        raise ValueError(
            f"{dataset_name}: unexpected state keys {list(modality['state'].keys())}, expected {spec['state_keys']}"
        )
    if list(modality["action"].keys()) != spec["action_keys"]:
        raise ValueError(
            f"{dataset_name}: unexpected action keys {list(modality['action'].keys())}, expected {spec['action_keys']}"
        )
    if list(modality["video"].keys()) != [spec["video_key"]]:
        raise ValueError(
            f"{dataset_name}: unexpected video keys {list(modality['video'].keys())}, expected {[spec['video_key']]}"
        )

    for expected_idx, key in enumerate(spec["state_keys"]):
        entry = modality["state"][key]
        if entry.get("original_key") != "observation.state":
            raise ValueError(f"{dataset_name}: state.{key} has wrong original_key={entry.get('original_key')}")
        if int(entry.get("start", -1)) != expected_idx or int(entry.get("end", -1)) != expected_idx + 1:
            raise ValueError(f"{dataset_name}: state.{key} has wrong span {entry.get('start')}:{entry.get('end')}")
        if bool(entry.get("absolute")) is not True:
            raise ValueError(f"{dataset_name}: state.{key} should be absolute")

    for expected_idx, key in enumerate(spec["action_keys"]):
        entry = modality["action"][key]
        if entry.get("original_key") != "action":
            raise ValueError(f"{dataset_name}: action.{key} has wrong original_key={entry.get('original_key')}")
        if int(entry.get("start", -1)) != expected_idx or int(entry.get("end", -1)) != expected_idx + 1:
            raise ValueError(f"{dataset_name}: action.{key} has wrong span {entry.get('start')}:{entry.get('end')}")
        if bool(entry.get("absolute")) is not False:
            raise ValueError(f"{dataset_name}: action.{key} should be relative")

    video_entry = modality["video"][spec["video_key"]]
    if video_entry.get("original_key") != spec["video_original_key"]:
        raise ValueError(
            f"{dataset_name}: video.{spec['video_key']} has wrong original_key={video_entry.get('original_key')}"
        )

    annotation = modality.get("annotation", {}).get("human.action.task_description")
    if annotation is None:
        raise ValueError(f"{dataset_name}: annotation.human.action.task_description missing in modality.json")
    if annotation.get("original_key") != "task_index":
        raise ValueError(
            f"{dataset_name}: annotation.human.action.task_description should map from task_index"
        )


def validate_dataset(dataset_root: Path, sample_episodes: int) -> None:
    dataset_name = dataset_root.name
    if dataset_name not in DATASET_SPECS:
        raise KeyError(f"Unsupported dataset for OXE validation: {dataset_name}")

    spec = DATASET_SPECS[dataset_name]

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {dataset_root}")

    for rel_path in REQUIRED_FILES:
        path = dataset_root / rel_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    info = _load_json(dataset_root / "meta" / "info.json")
    modality = _load_json(dataset_root / "meta" / "modality.json")
    episodes = _load_jsonl(dataset_root / "meta" / "episodes.jsonl")
    tasks = _load_jsonl(dataset_root / "meta" / "tasks.jsonl")

    if info.get("robot_type") != spec["robot_type"]:
        raise ValueError(
            f"{dataset_name}: expected robot_type={spec['robot_type']}, got {info.get('robot_type')}"
        )
    if "features" not in info or "observation.state" not in info["features"] or "action" not in info["features"]:
        raise ValueError(f"{dataset_name}: malformed info.json features block")
    if info["features"]["observation.state"]["shape"] != [len(spec["state_keys"])]:
        raise ValueError(
            f"{dataset_name}: observation.state shape mismatch {info['features']['observation.state']['shape']}"
        )
    if info["features"]["action"]["shape"] != [len(spec["action_keys"])]:
        raise ValueError(f"{dataset_name}: action shape mismatch {info['features']['action']['shape']}")
    if spec["video_original_key"] not in info["features"]:
        raise ValueError(
            f"{dataset_name}: info.json missing required video feature {spec['video_original_key']}"
        )
    if not episodes:
        raise ValueError(f"{dataset_name}: episodes.jsonl is empty")
    if not tasks:
        raise ValueError(f"{dataset_name}: tasks.jsonl is empty")

    _validate_modality(modality, dataset_name)

    chunk_size = int(info["chunks_size"])
    data_pattern = info["data_path"]
    video_pattern = info["video_path"]
    sample_indices = _sample_episode_indices(episodes, sample_episodes)
    task_index_to_text = {int(record["task_index"]): record["task"] for record in tasks}

    checked_rows = 0
    for episode_index in sample_indices:
        parquet_rel = _format_episode_relative_path(data_pattern, episode_index, chunk_size)
        parquet_path = dataset_root / parquet_rel
        if not parquet_path.is_file():
            raise FileNotFoundError(f"{dataset_name}: missing parquet for episode {episode_index}: {parquet_path}")

        video_rel = _format_episode_relative_path(
            video_pattern, episode_index, chunk_size, video_key=spec["video_original_key"]
        )
        video_path = dataset_root / video_rel
        if not video_path.is_file():
            raise FileNotFoundError(f"{dataset_name}: missing video for episode {episode_index}: {video_path}")

        frame_df = pd.read_parquet(parquet_path)
        if frame_df.empty:
            raise ValueError(f"{dataset_name}: empty parquet shard {parquet_path}")
        for column in ("observation.state", "action", "task_index"):
            if column not in frame_df.columns:
                raise ValueError(f"{dataset_name}: parquet {parquet_path} missing column {column}")

        first_state = frame_df["observation.state"].iloc[0]
        first_action = frame_df["action"].iloc[0]
        if len(first_state) != len(spec["state_keys"]):
            raise ValueError(
                f"{dataset_name}: parquet {parquet_path} has state dim {len(first_state)}, "
                f"expected {len(spec['state_keys'])}"
            )
        if len(first_action) != len(spec["action_keys"]):
            raise ValueError(
                f"{dataset_name}: parquet {parquet_path} has action dim {len(first_action)}, "
                f"expected {len(spec['action_keys'])}"
            )

        task_index = int(frame_df["task_index"].iloc[0])
        task_text = task_index_to_text.get(task_index, "")
        if not task_text:
            raise ValueError(
                f"{dataset_name}: task_index={task_index} from {parquet_path} missing in tasks.jsonl"
            )
        checked_rows += len(frame_df)

    print(
        f"[OK] {dataset_name}: episodes={len(episodes)}, sampled={len(sample_indices)}, "
        f"checked_rows={checked_rows}"
    )


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()

    print(f"Validating OXE LeRobot datasets under: {data_root}")
    try:
        for dataset_name in args.datasets:
            validate_dataset(data_root / dataset_name, sample_episodes=int(args.sample_episodes))
    except Exception as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1

    print("All requested OXE datasets passed validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
