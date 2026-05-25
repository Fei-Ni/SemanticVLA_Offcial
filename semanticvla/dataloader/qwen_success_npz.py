"""Qwen3VL SimplerEnv success-demo dataset for SemanticVLA finetuning.

This dataset reads the compact `.npz` trajectories collected from SimplerEnv
and pairs them with the dense Molmo Stage-1 trace index produced by
`tools/simpler/build_qwen_success_trace_index.py`.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from semanticvla.model.modules.action_model.trace_text_codec import get_default_anchor_indices


def collate_fn(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except Exception:
            pass
    return getattr(cfg, key, default)


def _as_list(value: Any) -> list[str] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        out = [part.strip() for part in value.split(",") if part.strip()]
        return out or None
    return [str(item) for item in value]


def _stable_u64(parts: tuple[Any, ...]) -> int:
    digest = hashlib.sha256(repr(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


class QwenSuccessNpzDataset(Dataset):
    """Direct SimplerEnv success trajectory dataset.

    The action target is `raw_actions` by default. That matches the current
    SimplerEnv inference contract, where the policy predicts raw Bridge-style
    deltas and the environment wrapper converts rotation/gripper before
    stepping the simulator.
    """

    def __init__(self, data_cfg: Any, mode: str = "train", **_: Any):
        super().__init__()
        self.data_cfg = data_cfg
        self.mode = mode

        self.data_root = Path(_cfg_get(data_cfg, "data_root_dir"))
        trace_cfg = _cfg_get(data_cfg, "trace", None)
        trace_root = _cfg_get(data_cfg, "trace_index_root", None)
        if trace_root is None:
            trace_root = _cfg_get(trace_cfg, "root", None)
        if trace_root is None:
            raise ValueError("qwen_success_npz requires datasets.vla_data.trace.root or trace_index_root")
        self.trace_root = Path(trace_root)

        self.action_key = str(_cfg_get(data_cfg, "action_key", "raw_actions"))
        self.fallback_action_keys = list(_as_list(_cfg_get(data_cfg, "fallback_action_keys", "raw_actions,actions")) or [])
        if self.action_key not in self.fallback_action_keys:
            self.fallback_action_keys.insert(0, self.action_key)
        self.image_key = str(_cfg_get(data_cfg, "image_key", "images"))
        self.dataset_name = str(_cfg_get(data_cfg, "dataset_name", "qwen_success_npz"))
        self.statistics_key = str(_cfg_get(data_cfg, "statistics_key", "oxe_bridge"))
        self.image_size = tuple(int(v) for v in _cfg_get(data_cfg, "image_size", [224, 224]))
        self.action_horizon = int(_cfg_get(data_cfg, "action_horizon", 16))
        self.window_size = int(_cfg_get(trace_cfg, "window_size", _cfg_get(data_cfg, "trace_window_size", 12)))
        self.normalize_trace = bool(_cfg_get(trace_cfg, "normalize", _cfg_get(data_cfg, "normalize_trace", True)))
        self.seed = int(_cfg_get(data_cfg, "seed", 42))
        self.randomize_samples = bool(_cfg_get(data_cfg, "randomize_samples", mode == "train"))
        self.epoch = 0
        self.cache_size = max(0, int(_cfg_get(data_cfg, "episode_cache_size", 2)))
        self._episode_cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

        latent_cfg = _cfg_get(data_cfg, "latent_action_labels", None)
        self.latent_label_manager = None
        self.latent_out_key = str(_cfg_get(latent_cfg, "out_key", "latent_action_idx"))
        self.latent_strict = bool(_cfg_get(latent_cfg, "strict", True))
        if latent_cfg is not None and bool(_cfg_get(latent_cfg, "enabled", False)):
            from semanticvla.dataloader.gr00t_lerobot.trace_loader import LatentActionLabelManager

            self.latent_label_manager = LatentActionLabelManager(
                label_root=_cfg_get(latent_cfg, "root"),
                dataset_names=[self.dataset_name],
                variant=_cfg_get(latent_cfg, "variant", None),
                missing_policy=str(_cfg_get(latent_cfg, "missing_policy", "clip")),
            )

        explicit_anchors = _cfg_get(trace_cfg, "anchor_indices", _cfg_get(data_cfg, "anchor_indices", None))
        if explicit_anchors is not None:
            self.anchor_indices = tuple(int(i) for i in explicit_anchors)
        else:
            num_anchor_points = int(_cfg_get(trace_cfg, "num_anchor_points", _cfg_get(data_cfg, "num_anchor_points", 4)) or 0)
            self.anchor_indices = get_default_anchor_indices(num_anchor_points) if num_anchor_points else None

        self.coords = np.load(self.trace_root / "qwen_success_coords.npy", mmap_mode="r")
        self.offsets = np.load(self.trace_root / "qwen_success_offsets.npy")
        present_path = self.trace_root / "qwen_success_present.npy"
        self.present = np.load(present_path) if present_path.exists() else self.offsets[1:] > self.offsets[:-1]
        self.episodes = _load_jsonl(self.trace_root / "episodes.jsonl")

        if len(self.episodes) + 1 != len(self.offsets):
            raise ValueError(
                f"trace index mismatch: {len(self.episodes)} episodes but {len(self.offsets)} offsets"
            )

        include_tags = set(_as_list(_cfg_get(data_cfg, "tags", None)) or [])
        include_tasks = set(_as_list(_cfg_get(data_cfg, "tasks", None)) or [])
        self.sample_index: list[tuple[int, int]] = []
        for ep_idx, episode in enumerate(self.episodes):
            if not bool(self.present[ep_idx]):
                continue
            if include_tags and str(episode.get("tag")) not in include_tags:
                continue
            if include_tasks and str(episode.get("task")) not in include_tasks:
                continue
            length = int(episode["length"])
            if length <= 0:
                continue
            npz_path = self._resolve_npz_path(episode)
            episode["resolved_npz"] = str(npz_path)
            self.sample_index.extend((ep_idx, step) for step in range(length))

        if not self.sample_index:
            raise ValueError(f"no Qwen success samples found under trace_root={self.trace_root}")

        epoch_length = int(_cfg_get(data_cfg, "epoch_length", 0) or 0)
        self.epoch_length = epoch_length if epoch_length > 0 else len(self.sample_index)

        self.action_stats = self._load_or_compute_action_stats()

    def __len__(self) -> int:
        return int(self.epoch_length)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ep_idx, step = self._map_index(index)
        episode = self.episodes[ep_idx]
        loaded = self._load_episode(ep_idx)

        images = loaded[self.image_key]
        raw_actions = loaded["actions_for_training"]
        step = min(max(int(step), 0), int(images.shape[0]) - 1)

        image = Image.fromarray(images[step]).resize(self.image_size)
        action = self._normalize_actions(self._action_chunk(raw_actions, step)).astype(np.float16, copy=False)
        trace = self._trace_window(ep_idx, step)

        instruction = str(episode.get("instruction") or "")
        sample = {
            "action": action,
            "image": [image],
            "lang": instruction,
            "task_text": instruction,
            "dataset_name": self.dataset_name,
            "trajectory_id": int(ep_idx),
            "step": int(step),
            "trace_coords_window": trace,
        }
        if self.latent_label_manager is not None:
            indices = self.latent_label_manager.get(self.dataset_name, ep_idx, step)
            if indices is not None:
                sample[self.latent_out_key] = indices
            elif self.latent_strict:
                raise KeyError(
                    f"No latent action label for {self.dataset_name} episode={ep_idx} step={step}"
                )
        return sample

    def save_dataset_statistics(self, save_path: Path | str, format: str = "json") -> None:
        if format.lower() != "json":
            raise ValueError(f"qwen_success_npz only supports json statistics, got {format!r}")
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            self.statistics_key: {
                "action": _to_jsonable(self.action_stats["action"]),
                "num_trajectories": int(len({ep for ep, _ in self.sample_index})),
                "num_transitions": int(len(self.sample_index)),
            }
        }
        with save_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def _map_index(self, index: int) -> tuple[int, int]:
        if not self.randomize_samples:
            return self.sample_index[int(index) % len(self.sample_index)]
        rng = np.random.default_rng(_stable_u64((self.seed, self.epoch, int(index))))
        return self.sample_index[int(rng.integers(0, len(self.sample_index)))]

    def _resolve_npz_path(self, episode: dict[str, Any]) -> Path:
        candidates: list[Path] = []
        source = episode.get("source_npz")
        if source:
            source_path = Path(str(source))
            candidates.append(source_path)
            parts = source_path.parts
            if len(parts) > 3 and parts[:3] == ("/", "lus", "lfs1aip2"):
                candidates.append(Path("/" + str(Path(*parts[3:]))))
        episode_name = str(episode["episode_name"])
        if not episode_name.endswith(".npz"):
            episode_name = f"{episode_name}.npz"
        candidates.append(
            self.data_root / str(episode["task"]) / str(episode["tag"]) / "episodes" / episode_name
        )
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            "could not resolve Qwen success npz; tried: "
            + ", ".join(str(path) for path in candidates)
        )

    def _load_episode(self, ep_idx: int) -> dict[str, np.ndarray]:
        if ep_idx in self._episode_cache:
            cached = self._episode_cache.pop(ep_idx)
            self._episode_cache[ep_idx] = cached
            return cached

        episode = self.episodes[ep_idx]
        path = Path(str(episode["resolved_npz"]))
        with np.load(path) as data:
            if self.image_key not in data:
                raise KeyError(f"{path} missing image key {self.image_key!r}; keys={data.files}")
            action_key = next((key for key in self.fallback_action_keys if key in data), None)
            if action_key is None:
                raise KeyError(f"{path} has none of action keys {self.fallback_action_keys}; keys={data.files}")
            loaded = {
                self.image_key: np.asarray(data[self.image_key]),
                "actions_for_training": np.asarray(data[action_key], dtype=np.float32),
            }

        if loaded[self.image_key].shape[0] != loaded["actions_for_training"].shape[0]:
            raise ValueError(
                f"{path} image/action length mismatch: "
                f"{loaded[self.image_key].shape[0]} vs {loaded['actions_for_training'].shape[0]}"
            )
        expected_len = int(episode["length"])
        if loaded[self.image_key].shape[0] != expected_len:
            raise ValueError(
                f"{path} length mismatch against trace index: "
                f"{loaded[self.image_key].shape[0]} vs {expected_len}"
            )

        if self.cache_size > 0:
            self._episode_cache[ep_idx] = loaded
            while len(self._episode_cache) > self.cache_size:
                self._episode_cache.popitem(last=False)
        return loaded

    def _action_chunk(self, raw_actions: np.ndarray, step: int) -> np.ndarray:
        out = np.zeros((self.action_horizon, raw_actions.shape[-1]), dtype=np.float32)
        end = min(int(raw_actions.shape[0]), int(step) + self.action_horizon)
        n = max(0, end - int(step))
        if n:
            out[:n] = raw_actions[int(step):end]
        return out

    def _trace_window(self, ep_idx: int, step: int) -> np.ndarray:
        start_off, end_off = int(self.offsets[ep_idx]), int(self.offsets[ep_idx + 1])
        coords = np.asarray(self.coords[start_off:end_off], dtype=np.float32)
        if coords.shape[0] <= 0:
            raise ValueError(f"empty trace for episode {ep_idx}")
        if self.normalize_trace:
            coords = coords / 100.0
        t = min(max(int(step), 0), coords.shape[0] - 1)
        start = t - self.window_size + 1
        if start < 0:
            pad = np.broadcast_to(coords[0:1], (-start, 2))
            window = np.concatenate([pad, coords[: t + 1]], axis=0)
        else:
            window = coords[start : t + 1]
        window = window.astype(np.float32, copy=False)
        if self.anchor_indices is not None:
            window = window[list(self.anchor_indices)]
        return window

    def _load_or_compute_action_stats(self) -> dict[str, Any]:
        stats_path = _cfg_get(self.data_cfg, "action_norm_stats_path", None)
        stats_key = str(_cfg_get(self.data_cfg, "action_norm_stats_key", self.statistics_key))
        if stats_path:
            with Path(stats_path).open("r", encoding="utf-8") as handle:
                stats_payload = json.load(handle)
            if stats_key not in stats_payload:
                if len(stats_payload) == 1:
                    stats_key = next(iter(stats_payload))
                else:
                    raise KeyError(
                        f"action_norm_stats_key={stats_key!r} not in {stats_path}; "
                        f"available={list(stats_payload)}"
                    )
            return {"action": stats_payload[stats_key]["action"]}

        rows: list[np.ndarray] = []
        for ep_idx in sorted({ep_idx for ep_idx, _ in self.sample_index}):
            loaded = self._load_episode(ep_idx)
            rows.append(np.asarray(loaded["actions_for_training"], dtype=np.float32))
        actions = np.concatenate(rows, axis=0)
        action_stats = {
            "mean": actions.mean(axis=0).tolist(),
            "std": actions.std(axis=0).tolist(),
            "max": actions.max(axis=0).tolist(),
            "min": actions.min(axis=0).tolist(),
            "q01": np.quantile(actions, 0.01, axis=0).tolist(),
            "q99": np.quantile(actions, 0.99, axis=0).tolist(),
            "mask": [True] * (actions.shape[1] - 1) + [False],
        }
        return {"action": action_stats}

    def _normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        stats = self.action_stats["action"]
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
        norm_mask = np.asarray(stats.get("mask", q01 != q99), dtype=bool)
        norm_mask = np.logical_and(norm_mask, q01 != q99)
        out = np.asarray(actions, dtype=np.float32).copy()
        denom = np.where(norm_mask, q99 - q01, 1.0).astype(np.float32)
        out[..., norm_mask] = 2.0 * ((out[..., norm_mask] - q01[norm_mask]) / denom[norm_mask]) - 1.0
        out[..., norm_mask] = np.clip(out[..., norm_mask], -1.0, 1.0)
        out[..., ~norm_mask] = (out[..., ~norm_mask] > 0.5).astype(np.float32)
        return out


def get_vla_dataset(
    data_cfg: Any,
    mode: str = "train",
    **kwargs: Any,
) -> QwenSuccessNpzDataset:
    return QwenSuccessNpzDataset(data_cfg=data_cfg, mode=mode, **kwargs)
