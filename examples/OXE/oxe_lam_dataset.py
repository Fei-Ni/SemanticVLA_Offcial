from __future__ import annotations

import io
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from examples.SemanticVLA_OXE.rlds_tfrecord import bytes_feature, float_feature, int64_feature, parse_example, read_record_at


@dataclass(frozen=True)
class OxeDatasetSpec:
    name: str
    root: Path
    file_prefix: str
    image_key: str
    state_key: str | None


DEFAULT_SPECS = {
    "bridge": OxeDatasetSpec(
        name="bridge",
        root=Path("${DATA_ROOT}/bridge_orig/1.0.0"),
        file_prefix="bridge_dataset-train.tfrecord",
        image_key="steps/observation/image_0",
        state_key="steps/observation/state",
    ),
    "fractal": OxeDatasetSpec(
        name="fractal",
        root=Path("${DATA_ROOT}/fractal20220817_data/0.1.0"),
        file_prefix="fractal20220817_data-train.tfrecord",
        image_key="steps/observation/image",
        state_key="steps/observation/base_pose_tool_reached",
    ),
    "bcz": OxeDatasetSpec(
        name="bcz",
        root=Path("${DATA_ROOT}/bc_z/0.1.0"),
        file_prefix="bc_z-train.tfrecord",
        image_key="steps/observation/image",
        state_key=None,
    ),
}


class OxeTraceManager:
    """Mmap-backed trace lookup using per-array NPY indices."""

    def __init__(self, trace_index_root: str | Path, datasets: list[str] | tuple[str, ...]):
        self.root = Path(trace_index_root)
        if not self.root.exists():
            raise FileNotFoundError(f"trace_index_root not found: {self.root}")
        self.coords: dict[str, np.ndarray] = {}
        self.offsets: dict[str, np.ndarray] = {}
        self.present: dict[str, np.ndarray] = {}
        for name in datasets:
            self.coords[name] = np.load(self.root / f"{name}_coords.npy", mmap_mode="r")
            self.offsets[name] = np.load(self.root / f"{name}_offsets.npy")
            present_path = self.root / f"{name}_present.npy"
            if present_path.exists():
                self.present[name] = np.load(present_path)
            else:
                self.present[name] = self.offsets[name][1:] > self.offsets[name][:-1]

    def num_episode_slots(self, dataset: str) -> int:
        return int(len(self.offsets[dataset]) - 1)

    def has_episode(self, dataset: str, episode_index: int) -> bool:
        ep = int(episode_index)
        return 0 <= ep < len(self.present[dataset]) and bool(self.present[dataset][ep])

    def episode_length(self, dataset: str, episode_index: int) -> int:
        ep = int(episode_index)
        offsets = self.offsets[dataset]
        return int(offsets[ep + 1] - offsets[ep])

    def get_window(self, dataset: str, episode_index: int, frame_index: int, window_size: int) -> np.ndarray:
        ep = int(episode_index)
        offsets = self.offsets[dataset]
        start_off, end_off = int(offsets[ep]), int(offsets[ep + 1])
        if end_off <= start_off:
            raise KeyError(f"missing trace for {dataset} episode={ep}")
        coords = np.asarray(self.coords[dataset][start_off:end_off], dtype=np.float32) / 100.0
        t = min(max(int(frame_index), 0), coords.shape[0] - 1)
        start = t - int(window_size) + 1
        if start < 0:
            pad = np.broadcast_to(coords[0:1], (-start, 2))
            window = np.concatenate([pad, coords[: t + 1]], axis=0)
        else:
            window = coords[start : t + 1]
        return window.astype(np.float32, copy=False)


class OxeLAMDataset(Dataset):
    """Raw RLDS TFRecord dataset for OXE LAM smoke/debug training.

    This intentionally avoids TensorFlow/TFDS in the SemanticVLA training env by
    parsing TFRecord + tf.train.Example directly with protobuf.
    """

    def __init__(
        self,
        *,
        trace_index_root: str,
        datasets: list[str] | tuple[str, ...] = ("bridge", "fractal"),
        split: str = "train",
        eval_stride: int = 10,
        sample_stride: int = 8,
        max_samples_per_dataset: int | None = None,
        max_episodes_per_dataset: int | None = None,
        window_size: int = 12,
        action_horizon: int = 8,
        image_resolution: int = 224,
        cache_episodes: int = 4,
    ):
        super().__init__()
        if split not in {"train", "eval", "all"}:
            raise ValueError(f"split must be train/eval/all, got {split!r}")
        self.datasets = tuple(datasets)
        self.split = split
        self.eval_stride = int(eval_stride)
        self.sample_stride = max(1, int(sample_stride))
        self.max_samples_per_dataset = max_samples_per_dataset
        self.max_episodes_per_dataset = max_episodes_per_dataset
        self.window_size = int(window_size)
        self.action_horizon = int(action_horizon)
        self.image_resolution = int(image_resolution)
        self.cache_episodes = int(cache_episodes)
        self.specs = {name: DEFAULT_SPECS[name] for name in self.datasets}
        self.trace_manager = OxeTraceManager(trace_index_root, self.datasets)
        self.shards = {name: self._load_shards(self.specs[name]) for name in self.datasets}
        self.samples = self._build_samples()
        self._episode_cache: OrderedDict[tuple[str, int], dict[str, Any]] = OrderedDict()

    def _load_shards(self, spec: OxeDatasetSpec) -> dict[str, Any]:
        if not spec.root.exists():
            raise FileNotFoundError(f"{spec.name} root not found: {spec.root}")
        with open(spec.root / "dataset_info.json") as fp:
            info = json.load(fp)
        splits = info["splits"].values() if isinstance(info["splits"], dict) else info["splits"]
        shard_lengths = None
        for split in splits:
            if split["name"] == "train":
                shard_lengths = [int(x) for x in split["shardLengths"]]
                break
        if shard_lengths is None:
            raise RuntimeError(f"no train split in {spec.root / 'dataset_info.json'}")
        files = sorted(p for p in spec.root.iterdir() if p.name.startswith(spec.file_prefix))
        if len(files) != len(shard_lengths):
            raise RuntimeError(
                f"{spec.name}: file count {len(files)} != shardLengths len {len(shard_lengths)} "
                f"for prefix {spec.file_prefix!r}"
            )
        cumulative = np.cumsum(np.asarray([0, *shard_lengths], dtype=np.int64))
        return {"files": files, "lengths": shard_lengths, "cumulative": cumulative}

    def _resolve_episode(self, dataset: str, episode_index: int) -> tuple[Path, int]:
        shards = self.shards[dataset]
        ep = int(episode_index)
        shard_idx = int(np.searchsorted(shards["cumulative"], ep, side="right") - 1)
        if shard_idx < 0 or shard_idx >= len(shards["files"]):
            raise IndexError(f"{dataset} episode {ep} out of range")
        offset = ep - int(shards["cumulative"][shard_idx])
        return shards["files"][shard_idx], offset

    def _build_samples(self) -> list[tuple[str, int, int]]:
        samples: list[tuple[str, int, int]] = []
        raw_pos = 0
        for dataset in self.datasets:
            added = 0
            slots = self.trace_manager.num_episode_slots(dataset)
            max_eps = slots if self.max_episodes_per_dataset is None else min(slots, int(self.max_episodes_per_dataset))
            for ep in range(max_eps):
                if not self.trace_manager.has_episode(dataset, ep):
                    continue
                ep_len = self.trace_manager.episode_length(dataset, ep)
                for step in range(0, ep_len, self.sample_stride):
                    is_eval = self.eval_stride > 0 and (raw_pos % self.eval_stride == 0)
                    raw_pos += 1
                    if self.split == "train" and is_eval:
                        continue
                    if self.split == "eval" and not is_eval:
                        continue
                    samples.append((dataset, ep, step))
                    added += 1
                    if self.max_samples_per_dataset is not None and added >= self.max_samples_per_dataset:
                        break
                if self.max_samples_per_dataset is not None and added >= self.max_samples_per_dataset:
                    break
        if not samples:
            raise ValueError("No OXE LAM samples built; check datasets/trace index/split limits.")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_episode(self, dataset: str, episode_index: int) -> dict[str, Any]:
        key = (dataset, int(episode_index))
        if key in self._episode_cache:
            self._episode_cache.move_to_end(key)
            return self._episode_cache[key]
        spec = self.specs[dataset]
        path, offset = self._resolve_episode(dataset, episode_index)
        example = parse_example(read_record_at(path, offset))
        images = bytes_feature(example, spec.image_key)
        if dataset == "bridge":
            actions = np.asarray(float_feature(example, "steps/action"), dtype=np.float32).reshape(-1, 7)
            state = np.asarray(float_feature(example, "steps/observation/state"), dtype=np.float32).reshape(-1, 7)
        elif dataset == "fractal":
            world = np.asarray(float_feature(example, "steps/action/world_vector"), dtype=np.float32).reshape(-1, 3)
            rot = np.asarray(float_feature(example, "steps/action/rotation_delta"), dtype=np.float32).reshape(-1, 3)
            grip = np.asarray(float_feature(example, "steps/action/gripper_closedness_action"), dtype=np.float32).reshape(-1, 1)
            actions = np.concatenate([world, rot, grip], axis=1)
            state = np.asarray(float_feature(example, "steps/observation/base_pose_tool_reached"), dtype=np.float32).reshape(-1, 7)
        elif dataset == "bcz":
            xyz = np.asarray(float_feature(example, "steps/action/future/xyz_residual"), dtype=np.float32).reshape(-1, 30)[:, :3]
            rot = np.asarray(float_feature(example, "steps/action/future/axis_angle_residual"), dtype=np.float32).reshape(-1, 30)[:, :3]
            grip = np.asarray(int64_feature(example, "steps/action/future/target_close"), dtype=np.float32).reshape(-1, 10)[:, :1]
            actions = np.concatenate([xyz, rot, grip], axis=1)
            present_xyz = np.asarray(float_feature(example, "steps/observation/present/xyz"), dtype=np.float32).reshape(-1, 3)
            present_rot = np.asarray(float_feature(example, "steps/observation/present/axis_angle"), dtype=np.float32).reshape(-1, 3)
            present_grip = np.asarray(float_feature(example, "steps/observation/present/sensed_close"), dtype=np.float32).reshape(-1, 1)
            state = np.concatenate([present_xyz, present_rot, present_grip], axis=1)
        else:
            raise ValueError(f"Unsupported OXE dataset: {dataset}")
        episode = {"images": images, "actions": actions, "state": state}
        self._episode_cache[key] = episode
        while len(self._episode_cache) > self.cache_episodes:
            self._episode_cache.popitem(last=False)
        return episode

    def _decode_frame(self, image_bytes: bytes) -> torch.Tensor:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((self.image_resolution, self.image_resolution), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _future_actions(self, actions: np.ndarray, step_index: int) -> torch.Tensor:
        ep_len = int(actions.shape[0])
        idx = np.minimum(np.arange(step_index, step_index + self.action_horizon), ep_len - 1)
        return torch.from_numpy(actions[idx].astype(np.float32, copy=False))

    def __getitem__(self, index: int) -> dict[str, Any]:
        dataset, episode_index, step_index = self.samples[index]
        episode = self._load_episode(dataset, episode_index)
        ep_len = min(len(episode["images"]), int(episode["actions"].shape[0]))
        end = min(int(step_index), ep_len - 1)
        start = max(0, end - self.window_size + 1)
        frame_indices = [start, end]
        videos = torch.stack([self._decode_frame(episode["images"][i]) for i in frame_indices], dim=0)
        traces = self.trace_manager.get_window(dataset, episode_index, end, self.window_size)
        traces_t = torch.from_numpy(traces.astype(np.float32, copy=False))
        return {
            "videos": videos,
            "traces": traces_t,
            "trace_mask": torch.ones(traces_t.shape[0], dtype=torch.bool),
            "future_actions": self._future_actions(episode["actions"], end),
            "dataset_name": dataset,
            "episode_index": int(episode_index),
            "step_index": int(end),
        }


def oxe_lam_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("videos", "traces", "trace_mask", "future_actions"):
        out[key] = torch.stack([sample[key] for sample in batch], dim=0)
    out["dataset_name"] = [sample["dataset_name"] for sample in batch]
    out["episode_index"] = torch.tensor([sample["episode_index"] for sample in batch], dtype=torch.long)
    out["step_index"] = torch.tensor([sample["step_index"] for sample in batch], dtype=torch.long)
    return out
