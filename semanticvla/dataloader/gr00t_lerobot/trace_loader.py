"""LIBERO trace loader for SemanticVLA on LeRobot.

Reads the LeRobot-aligned sim-ground-truth (x,y) trace JSONs produced by
`trace_annotations/libero/build_lerobot_aligned_traces.py` and exposes a
`(dataset_name, episode_index, frame_index) -> [W, 2]` window lookup.

Wraps a base LeRobotMixtureDataset as `TraceAugmentedDataset` so the trace
field is attached per sample without modifying the existing dataset code.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from torch.utils.data import Dataset


_LIBERO_DATASET_RE = re.compile(r"^libero_(spatial|object|goal|10)_no_noops_")
_OXE_TRACE_DATASET_MAP = {
    "bridge_orig_1.0.0_lerobot": "bridge",
    "fractal20220817_data_0.1.0_lerobot": "fractal",
    "bcz_0.1.0_lerobot": "bcz",
}


def parse_libero_suite(dataset_name: str) -> str | None:
    m = _LIBERO_DATASET_RE.match(dataset_name)
    return m.group(1) if m else None


class LiberoTraceLoader:
    """Per-suite (x,y) coord lookup with left-pad-repeat windowing."""

    def __init__(
        self,
        trace_json_path: str | Path,
        image_size: tuple[int, int] = (256, 256),
        normalize: bool = True,
    ):
        path = Path(trace_json_path)
        with open(path) as fp:
            d = json.load(fp)
        self.suite: str = d["suite"]
        self.image_size: tuple[int, int] = tuple(d["image_size"])
        self.view: str = d.get("view", "observation.images.image")
        self.episodes: dict[int, np.ndarray] = {
            int(k): np.asarray(v["coords"], dtype=np.float32)
            for k, v in d["episodes"].items()
        }
        self._norm = np.asarray(image_size, dtype=np.float32) if normalize else None

    def __len__(self) -> int:
        return len(self.episodes)

    def episode_length(self, episode_index: int) -> int:
        return int(self.episodes[int(episode_index)].shape[0])

    def get_window(
        self,
        episode_index: int,
        frame_index: int,
        window_size: int = 12,
        anchor_indices: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """Return `[N, 2]` (x, y) coords for the trace at `frame_index`.

        Window = coords[t - W + 1 : t + 1] (length `window_size`). If
        t < W-1, the missing leading slots are filled with coords[0]
        (left-pad-repeat — matches LIBERO gripper starting still at the
        home position).

        Args:
            anchor_indices: if `None`, return the full `window_size`-long
                window (v0 / backward-compat: shape `(window_size, 2)`).
                If a tuple of ints, sub-sample the window at those positions
                and return shape `(len(anchor_indices), 2)`. Indices must
                fall in `[0, window_size)`. Typical  usage:
                  `anchor_indices=(0, 4, 8, 11)` on `window_size=12` →
                  4 anchor points (start / 1/3 / 2/3 / end).
        """
        coords = self.episodes.get(int(episode_index))
        if coords is None:
            raise KeyError(
                f"episode_index {episode_index} not in suite '{self.suite}' "
                f"(have {len(self.episodes)} episodes)"
            )
        t = int(frame_index)
        ep_len = coords.shape[0]
        if t < 0 or t >= ep_len:
            raise IndexError(
                f"frame_index {t} out of range [0, {ep_len}) for "
                f"episode {episode_index} in suite '{self.suite}'"
            )
        start = t - window_size + 1
        if start < 0:
            pad = np.broadcast_to(coords[0:1], (-start, 2))
            window = np.concatenate([pad, coords[: t + 1]], axis=0)
        else:
            window = coords[start : t + 1]
        if self._norm is not None:
            window = window / self._norm
        window = window.astype(np.float32, copy=False)
        if anchor_indices is not None:
            # Validate indices are within window; reject early so configuration
            # errors surface before they propagate into model shape mismatches.
            for ai in anchor_indices:
                if ai < 0 or ai >= window_size:
                    raise IndexError(
                        f"anchor_index {ai} out of range [0, {window_size}) "
                        f"for window_size={window_size}"
                    )
            window = window[list(anchor_indices)]
        return window


class LiberoTraceManager:
    """Multi-suite trace dispatcher keyed by LeRobot dataset_name.

    Non-LIBERO datasets are no-op: `get_window(...)` returns `None`, callers
    skip attaching the trace field. This keeps mixtures with non-LIBERO
    siblings (bridge, fractal, etc.) safe.
    """

    def __init__(
        self,
        trace_root: str | Path,
        dataset_names: Iterable[str] | None = None,
        image_size: tuple[int, int] = (256, 256),
        normalize: bool = True,
        window_size: int = 12,
        anchor_indices: tuple[int, ...] | None = None,
    ):
        self.trace_root = Path(trace_root)
        if not self.trace_root.exists():
            raise FileNotFoundError(f"trace_root not found: {self.trace_root}")
        self.window_size = window_size
        self.anchor_indices = anchor_indices  # None = full 12-frame window (v0)
        self.loaders: dict[str, LiberoTraceLoader] = {}
        names = list(dataset_names) if dataset_names else self._discover_all_libero()
        for name in names:
            suite = parse_libero_suite(name)
            if not suite:
                continue
            path = self.trace_root / f"libero_{suite}.json"
            if path.exists():
                self.loaders[name] = LiberoTraceLoader(
                    path, image_size=image_size, normalize=normalize
                )

    def _discover_all_libero(self) -> list[str]:
        # Map e.g. libero_spatial.json -> libero_spatial_no_noops_1.0.0_lerobot.
        out = []
        for p in self.trace_root.glob("libero_*.json"):
            suite = p.stem.removeprefix("libero_")
            out.append(f"libero_{suite}_no_noops_1.0.0_lerobot")
        return out

    def has(self, dataset_name: str) -> bool:
        return dataset_name in self.loaders

    def get_window(
        self,
        dataset_name: str,
        episode_index: int,
        frame_index: int,
        window_size: int | None = None,
        anchor_indices: tuple[int, ...] | None = None,
    ) -> np.ndarray | None:
        loader = self.loaders.get(dataset_name)
        if loader is None:
            return None
        return loader.get_window(
            episode_index,
            frame_index,
            window_size=window_size or self.window_size,
            anchor_indices=anchor_indices if anchor_indices is not None else self.anchor_indices,
        )

    def __repr__(self) -> str:
        return (
            f"LiberoTraceManager(trace_root={self.trace_root}, "
            f"suites={[ld.suite for ld in self.loaders.values()]}, "
            f"window_size={self.window_size})"
        )


class OxeTraceIndexManager:
    """Mmap-backed OXE dense trace lookup keyed by LeRobot dataset_name.

    The OXE trace handoff is stored as per-array NPY files:
    `{bridge,fractal,bcz}_coords.npy`, `{name}_offsets.npy`, and optional
    `{name}_present.npy`. Coordinates are normalized image-space `[0, 100]`;
    by default this manager returns `[0, 1]` coordinates to match the LIBERO
    trace wrapper contract consumed by SemanticVLA text/trace encoders.
    """

    def __init__(
        self,
        trace_root: str | Path,
        dataset_names: Iterable[str] | None = None,
        normalize: bool = True,
        window_size: int = 12,
        anchor_indices: tuple[int, ...] | None = None,
    ):
        self.trace_root = Path(trace_root)
        if not self.trace_root.exists():
            raise FileNotFoundError(f"trace_root not found: {self.trace_root}")
        self.normalize = bool(normalize)
        self.window_size = int(window_size)
        self.anchor_indices = anchor_indices
        self.dataset_to_trace_key: dict[str, str] = {}
        self.coords: dict[str, np.ndarray] = {}
        self.offsets: dict[str, np.ndarray] = {}
        self.present: dict[str, np.ndarray] = {}

        names = list(dataset_names) if dataset_names else list(_OXE_TRACE_DATASET_MAP)
        for dataset_name in names:
            trace_key = _OXE_TRACE_DATASET_MAP.get(dataset_name)
            if trace_key is None:
                continue
            coords_path = self.trace_root / f"{trace_key}_coords.npy"
            offsets_path = self.trace_root / f"{trace_key}_offsets.npy"
            if not coords_path.exists() or not offsets_path.exists():
                continue
            self.dataset_to_trace_key[dataset_name] = trace_key
            if trace_key not in self.coords:
                self.coords[trace_key] = np.load(coords_path, mmap_mode="r")
                self.offsets[trace_key] = np.load(offsets_path)
                present_path = self.trace_root / f"{trace_key}_present.npy"
                if present_path.exists():
                    self.present[trace_key] = np.load(present_path)
                else:
                    offsets = self.offsets[trace_key]
                    self.present[trace_key] = offsets[1:] > offsets[:-1]

    def has(self, dataset_name: str) -> bool:
        return dataset_name in self.dataset_to_trace_key

    def get_window(
        self,
        dataset_name: str,
        episode_index: int,
        frame_index: int,
        window_size: int | None = None,
        anchor_indices: tuple[int, ...] | None = None,
    ) -> np.ndarray | None:
        trace_key = self.dataset_to_trace_key.get(dataset_name)
        if trace_key is None:
            return None
        ep = int(episode_index)
        present = self.present[trace_key]
        if ep < 0 or ep >= len(present) or not bool(present[ep]):
            return None
        offsets = self.offsets[trace_key]
        start_off, end_off = int(offsets[ep]), int(offsets[ep + 1])
        if end_off <= start_off:
            return None
        coords = np.asarray(self.coords[trace_key][start_off:end_off], dtype=np.float32)
        if self.normalize:
            coords = coords / 100.0
        t = min(max(int(frame_index), 0), coords.shape[0] - 1)
        w = int(window_size or self.window_size)
        start = t - w + 1
        if start < 0:
            pad = np.broadcast_to(coords[0:1], (-start, 2))
            window = np.concatenate([pad, coords[: t + 1]], axis=0)
        else:
            window = coords[start : t + 1]
        window = window.astype(np.float32, copy=False)
        anchors = anchor_indices if anchor_indices is not None else self.anchor_indices
        if anchors is not None:
            for ai in anchors:
                if ai < 0 or ai >= w:
                    raise IndexError(f"anchor_index {ai} out of range [0, {w})")
            window = window[list(anchors)]
        return window

    def __repr__(self) -> str:
        return (
            f"OxeTraceIndexManager(trace_root={self.trace_root}, "
            f"datasets={self.dataset_to_trace_key}, window_size={self.window_size})"
        )


class TraceAugmentedDataset(Dataset):
    """Wrap a base mixture/dataset and attach `trace_coords_window` per sample.

    Behavior:
      - `__getitem__(i)` calls `base[i]` then, if the sample is a dict and
        carries `dataset_name`, `trajectory_id`, `step`, asks the trace
        manager for a window. If the dataset is not a LIBERO one (manager
        returns `None`), the sample passes through unchanged.
      - Everything else (length, attribute access, statistics) delegates to
        the base dataset.

    This wrapper is intentionally zero-touch on the semanticvla codebase: the
    base `LeRobotMixtureDataset` already exposes `dataset_name`,
    `trajectory_id`, `step` in the sample dict (see datasets.py:1972-1974),
    so no upstream modification is required.
    """

    def __init__(
        self,
        base: Dataset,
        trace_manager: LiberoTraceManager | None,
        out_key: str = "trace_coords_window",
    ):
        super().__init__()
        self.base = base
        self.trace_manager = trace_manager
        self.out_key = out_key

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        sample = self.base[index]
        if self.trace_manager is None or not isinstance(sample, dict):
            return sample
        name = sample.get("dataset_name")
        ep = sample.get("trajectory_id")
        st = sample.get("step")
        if name is None or ep is None or st is None:
            return sample
        window = self.trace_manager.get_window(name, ep, st)
        if window is not None:
            sample[self.out_key] = window
        return sample

    def __getattr__(self, name: str):
        # Fallback for attributes not defined on the wrapper (datasets,
        # metadata, set_epoch, ...). Without this, len/repr from outside
        # would not see base properties.
        return getattr(self.base, name)


class LatentActionLabelManager:
    """Lookup precomputed LAM indices by `(dataset_name, episode, step)`."""

    def __init__(
        self,
        label_root: str | Path,
        dataset_names: Iterable[str] | None = None,
        variant: str | None = None,
        missing_policy: str = "error",
    ):
        self.label_root = Path(label_root)
        if variant:
            maybe_variant_dir = self.label_root / str(variant)
            if maybe_variant_dir.exists():
                self.label_root = maybe_variant_dir
        if not self.label_root.exists():
            raise FileNotFoundError(f"latent action label_root not found: {self.label_root}")
        if missing_policy not in {"error", "clip", "nearest"}:
            raise ValueError(
                "missing_policy must be one of {'error', 'clip', 'nearest'}, "
                f"got {missing_policy!r}"
            )
        self.missing_policy = missing_policy

        wanted = set(dataset_names) if dataset_names else None
        self.labels: dict[tuple[str, int, int], np.ndarray] = {}
        steps_by_episode: dict[tuple[str, int], list[int]] = {}
        self.num_tokens: int | None = None
        self.num_labels = 0

        for path in sorted(self.label_root.glob("*.jsonl")):
            with open(path) as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    name = d["dataset_name"]
                    if wanted is not None and name not in wanted:
                        continue
                    key = (name, int(d["episode_index"]), int(d["step_index"]))
                    indices = np.asarray(d["indices"], dtype=np.int64)
                    if self.num_tokens is None:
                        self.num_tokens = int(indices.shape[0])
                    self.labels[key] = indices
                    steps_by_episode.setdefault((name, int(d["episode_index"])), []).append(
                        int(d["step_index"])
                    )
                    self.num_labels += 1

        if not self.labels:
            raise ValueError(f"No latent action labels loaded from {self.label_root}")
        self.steps_by_episode: dict[tuple[str, int], np.ndarray] = {
            key: np.asarray(sorted(set(steps)), dtype=np.int64)
            for key, steps in steps_by_episode.items()
        }

    def get(self, dataset_name: str, episode_index: int, frame_index: int) -> np.ndarray | None:
        name = str(dataset_name)
        ep = int(episode_index)
        st = int(frame_index)
        value = self.labels.get((name, ep, st))
        if value is None and self.missing_policy != "error":
            steps = self.steps_by_episode.get((name, ep))
            if steps is not None and len(steps) > 0:
                pos = int(np.searchsorted(steps, st))
                if pos >= len(steps):
                    fallback_step = int(steps[-1])
                elif pos == 0:
                    fallback_step = int(steps[0])
                elif self.missing_policy == "nearest":
                    lo = int(steps[pos - 1])
                    hi = int(steps[pos])
                    fallback_step = lo if abs(st - lo) <= abs(hi - st) else hi
                else:
                    # For VLA samples beyond the dense trace annotation horizon,
                    # mirror trace-window behavior by clipping to the last label.
                    fallback_step = int(steps[pos - 1])
                value = self.labels.get((name, ep, fallback_step))
        return value.copy() if value is not None else None

    def __repr__(self) -> str:
        return (
            f"LatentActionLabelManager(label_root={self.label_root}, "
            f"num_labels={self.num_labels}, num_tokens={self.num_tokens}, "
            f"missing_policy={self.missing_policy!r})"
        )


class LatentActionAugmentedDataset(Dataset):
    """Attach precomputed `latent_action_idx` to each sample."""

    def __init__(
        self,
        base: Dataset,
        label_manager: LatentActionLabelManager | None,
        out_key: str = "latent_action_idx",
        strict: bool = True,
    ):
        super().__init__()
        self.base = base
        self.label_manager = label_manager
        self.out_key = out_key
        self.strict = bool(strict)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict:
        sample = self.base[index]
        if self.label_manager is None or not isinstance(sample, dict):
            return sample
        name = sample.get("dataset_name")
        ep = sample.get("trajectory_id")
        st = sample.get("step")
        if name is None or ep is None or st is None:
            return sample
        indices = self.label_manager.get(name, ep, st)
        if indices is not None:
            sample[self.out_key] = indices
        elif self.strict:
            raise KeyError(f"No latent action label for {name} episode={ep} step={st}")
        return sample

    def __getattr__(self, name: str):
        return getattr(self.base, name)
