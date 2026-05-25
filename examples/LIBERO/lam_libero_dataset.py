from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from semanticvla.dataloader.gr00t_lerobot.trace_loader import LiberoTraceManager
from semanticvla.dataloader.gr00t_lerobot.video import get_frames_by_timestamps


LIBERO_DATASETS = {
    "spatial": "libero_spatial_no_noops_1.0.0_lerobot",
    "object": "libero_object_no_noops_1.0.0_lerobot",
    "goal": "libero_goal_no_noops_1.0.0_lerobot",
    "10": "libero_10_no_noops_1.0.0_lerobot",
}


class LiberoLAMDataset(Dataset):
    """SemanticVLA-native LIBERO dataset for LAM pretraining.

    Each sample uses a historical pair `(t - window_size + 1, t)` and the
    existing LeRobot-aligned trace window ending at `t`.
    """

    def __init__(
        self,
        *,
        data_root: str,
        trace_root: str,
        suites: list[str] | tuple[str, ...] = ("spatial", "object", "goal", "10"),
        split: str = "train",
        eval_stride: int = 10,
        window_size: int = 12,
        action_horizon: int = 8,
        image_resolution: int = 224,
        video_key: str = "observation.images.image",
        video_backend: str = "torchvision_av",
        max_samples_per_suite: int | None = None,
    ):
        super().__init__()
        if split not in {"train", "eval", "all"}:
            raise ValueError(f"split must be train/eval/all, got {split!r}")
        self.data_root = Path(data_root)
        self.trace_root = Path(trace_root)
        self.suites = tuple(suites)
        self.split = split
        self.eval_stride = int(eval_stride)
        self.window_size = int(window_size)
        self.action_horizon = int(action_horizon)
        self.image_resolution = int(image_resolution)
        self.video_key = video_key
        self.video_backend = video_backend
        self.max_samples_per_suite = max_samples_per_suite

        dataset_names = [LIBERO_DATASETS[s] for s in self.suites]
        self.trace_manager = LiberoTraceManager(
            trace_root=self.trace_root,
            dataset_names=dataset_names,
            window_size=self.window_size,
            normalize=True,
            anchor_indices=None,
        )
        self.dataset_info = {name: self._load_info(name) for name in dataset_names}
        self.samples = self._build_samples(dataset_names)
        self._traj_cache: OrderedDict[tuple[str, int], pd.DataFrame] = OrderedDict()

    def _load_info(self, dataset_name: str) -> dict[str, Any]:
        path = self.data_root / dataset_name / "meta/info.json"
        with open(path) as fp:
            return json.load(fp)

    def _build_samples(self, dataset_names: list[str]) -> list[tuple[str, int, int]]:
        samples: list[tuple[str, int, int]] = []
        raw_pos = 0
        for dataset_name in dataset_names:
            loader = self.trace_manager.loaders[dataset_name]
            suite_count = 0
            for episode_index, coords in sorted(loader.episodes.items()):
                for step_index in range(int(coords.shape[0])):
                    is_eval = self.eval_stride > 0 and (raw_pos % self.eval_stride == 0)
                    raw_pos += 1
                    if self.split == "train" and is_eval:
                        continue
                    if self.split == "eval" and not is_eval:
                        continue
                    samples.append((dataset_name, int(episode_index), int(step_index)))
                    suite_count += 1
                    if self.max_samples_per_suite is not None and suite_count >= self.max_samples_per_suite:
                        break
                if self.max_samples_per_suite is not None and suite_count >= self.max_samples_per_suite:
                    break
        if not samples:
            raise ValueError("No LAM samples built; check suites/split/eval_stride.")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _episode_chunk(self, dataset_name: str, episode_index: int) -> int:
        chunk_size = int(self.dataset_info[dataset_name].get("chunks_size", 1000))
        return int(episode_index) // chunk_size

    def _trajectory_data(self, dataset_name: str, episode_index: int) -> pd.DataFrame:
        key = (dataset_name, int(episode_index))
        if key in self._traj_cache:
            self._traj_cache.move_to_end(key)
            return self._traj_cache[key]
        info = self.dataset_info[dataset_name]
        rel = info["data_path"].format(
            episode_chunk=self._episode_chunk(dataset_name, episode_index),
            episode_index=episode_index,
        )
        df = pd.read_parquet(self.data_root / dataset_name / rel)
        self._traj_cache[key] = df
        if len(self._traj_cache) > 8:
            self._traj_cache.popitem(last=False)
        return df

    def _video_path(self, dataset_name: str, episode_index: int) -> Path:
        info = self.dataset_info[dataset_name]
        rel = info["video_path"].format(
            episode_chunk=self._episode_chunk(dataset_name, episode_index),
            episode_index=episode_index,
            video_key=self.video_key,
        )
        return self.data_root / dataset_name / rel

    def _load_video_pair(self, dataset_name: str, episode_index: int, frame_indices: np.ndarray, df: pd.DataFrame) -> torch.Tensor:
        timestamps = df["timestamp"].to_numpy()[frame_indices].astype(np.float64)
        frames = get_frames_by_timestamps(
            self._video_path(dataset_name, episode_index).as_posix(),
            timestamps,
            video_backend=self.video_backend,
        )
        if frames.shape[0] < len(frame_indices):
            pad = np.repeat(frames[-1:], len(frame_indices) - frames.shape[0], axis=0)
            frames = np.concatenate([frames, pad], axis=0)
        frames = frames[: len(frame_indices)]
        tensors = []
        for frame in frames:
            img = Image.fromarray(frame.astype(np.uint8)).resize(
                (self.image_resolution, self.image_resolution),
                Image.BICUBIC,
            )
            arr = np.asarray(img, dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
        return torch.stack(tensors, dim=0)

    def _future_actions(self, df: pd.DataFrame, step_index: int) -> torch.Tensor:
        ep_len = len(df)
        idx = np.minimum(np.arange(step_index, step_index + self.action_horizon), ep_len - 1)
        vals = [np.asarray(v, dtype=np.float32) for v in df["action"].iloc[idx].to_list()]
        return torch.from_numpy(np.stack(vals, axis=0))

    def __getitem__(self, index: int) -> dict[str, Any]:
        dataset_name, episode_index, step_index = self.samples[index]
        df = self._trajectory_data(dataset_name, episode_index)
        ep_len = len(df)
        end = min(int(step_index), ep_len - 1)
        start = max(0, end - self.window_size + 1)
        frame_indices = np.asarray([start, end], dtype=np.int64)

        videos = self._load_video_pair(dataset_name, episode_index, frame_indices, df)
        traces = self.trace_manager.get_window(dataset_name, episode_index, end)
        if traces is None:
            raise KeyError(f"No trace for {dataset_name} episode={episode_index} step={end}")
        traces_t = torch.from_numpy(traces.astype(np.float32))
        trace_mask = torch.ones(traces_t.shape[0], dtype=torch.bool)

        return {
            "videos": videos,
            "traces": traces_t,
            "trace_mask": trace_mask,
            "future_actions": self._future_actions(df, end),
            "dataset_name": dataset_name,
            "episode_index": int(episode_index),
            "step_index": int(end),
        }


def lam_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    tensor_keys = ["videos", "traces", "trace_mask", "future_actions"]
    for key in tensor_keys:
        out[key] = torch.stack([sample[key] for sample in batch], dim=0)
    out["dataset_name"] = [sample["dataset_name"] for sample in batch]
    out["episode_index"] = torch.tensor([sample["episode_index"] for sample in batch], dtype=torch.long)
    out["step_index"] = torch.tensor([sample["step_index"] for sample in batch], dtype=torch.long)
    return out
