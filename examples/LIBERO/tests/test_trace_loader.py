"""Unit tests for LiberoTraceLoader / LiberoTraceManager / TraceAugmentedDataset.

Run from repo root:
    cd ${REPO_ROOT}
    PYTHONPATH=. python -m pytest examples/LIBERO/semanticvla/tests/test_trace_loader.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from torch.utils.data import Dataset

from semanticvla.dataloader.gr00t_lerobot.trace_loader import (
    LiberoTraceLoader,
    LiberoTraceManager,
    TraceAugmentedDataset,
    parse_libero_suite,
)


TRACE_ROOT = Path("${TRACE_ANNOTATIONS_ROOT}/libero_lerobot_aligned")
SPATIAL_JSON = TRACE_ROOT / "libero_spatial.json"


# -------- parse_libero_suite --------


def test_parse_suite_known():
    assert parse_libero_suite("libero_spatial_no_noops_1.0.0_lerobot") == "spatial"
    assert parse_libero_suite("libero_object_no_noops_1.0.0_lerobot") == "object"
    assert parse_libero_suite("libero_goal_no_noops_1.0.0_lerobot") == "goal"
    assert parse_libero_suite("libero_10_no_noops_1.0.0_lerobot") == "10"


def test_parse_suite_non_libero():
    assert parse_libero_suite("bridge_orig_1.0.0_lerobot") is None
    assert parse_libero_suite("libero_90_no_noops_lerobot") is None  # not in our 4
    assert parse_libero_suite("random_name") is None


# -------- LiberoTraceLoader --------


@pytest.fixture(scope="module")
def loader_spatial():
    assert SPATIAL_JSON.exists(), f"missing trace file: {SPATIAL_JSON}"
    return LiberoTraceLoader(SPATIAL_JSON)


def test_loader_metadata(loader_spatial):
    assert loader_spatial.suite == "spatial"
    assert loader_spatial.image_size == (256, 256)
    assert loader_spatial.view == "observation.images.image"
    assert len(loader_spatial) == 432


def test_loader_episode_lengths_match_lerobot(loader_spatial):
    meta_path = (
        "${DATA_ROOT}/libero_lerobot/"
        "libero_spatial_no_noops_1.0.0_lerobot/meta/episodes.jsonl"
    )
    with open(meta_path) as fp:
        lengths = {
            json.loads(line)["episode_index"]: json.loads(line)["length"]
            for line in fp
        }
    # Verify every loaded episode matches LeRobot
    for ep_idx, expected in lengths.items():
        assert loader_spatial.episode_length(ep_idx) == expected


def test_window_shape(loader_spatial):
    w = loader_spatial.get_window(0, 50, window_size=12)
    assert w.shape == (12, 2)
    assert w.dtype == np.float32


def test_window_normalized(loader_spatial):
    # default normalize=True → coords / image_size → [0, 1]
    w = loader_spatial.get_window(0, 50, window_size=12)
    assert (w >= 0.0).all() and (w <= 1.0).all()


def test_window_left_padding_at_episode_start(loader_spatial):
    # at frame_index=0, the entire window must be coords[0] repeated
    w = loader_spatial.get_window(0, 0, window_size=12)
    assert w.shape == (12, 2)
    np.testing.assert_array_equal(w[0], w[-1])  # all rows equal
    for k in range(1, 12):
        np.testing.assert_array_equal(w[0], w[k])


def test_window_partial_padding(loader_spatial):
    # at frame_index=3, the first 8 rows should equal coords[0], last 4 = coords[0:4]
    w = loader_spatial.get_window(0, 3, window_size=12)
    assert w.shape == (12, 2)
    # rows 0..8 (indices 0..7) are the 8 left-pad copies of coords[0]
    for k in range(8):
        np.testing.assert_array_equal(w[k], w[0])
    # last 4 rows are coords[0:4]
    raw = LiberoTraceLoader(SPATIAL_JSON, normalize=False)
    coords_unnorm = raw.episodes[0]
    expected_tail = coords_unnorm[0:4] / np.asarray(raw.image_size, dtype=np.float32)
    np.testing.assert_allclose(w[8:12], expected_tail, atol=1e-6)


def test_window_normal_middle(loader_spatial):
    # at frame_index=20, window=[9..20] (12 frames, no padding)
    raw = LiberoTraceLoader(SPATIAL_JSON, normalize=False)
    expected = raw.episodes[0][9:21] / np.asarray(raw.image_size, dtype=np.float32)
    w = loader_spatial.get_window(0, 20, window_size=12)
    np.testing.assert_allclose(w, expected, atol=1e-6)


def test_window_end_of_episode(loader_spatial):
    # frame_index = ep_len - 1 must succeed
    ep_len = loader_spatial.episode_length(0)
    w = loader_spatial.get_window(0, ep_len - 1, window_size=12)
    assert w.shape == (12, 2)


def test_unnormalized_range(loader_spatial):
    raw = LiberoTraceLoader(SPATIAL_JSON, normalize=False)
    w = raw.get_window(0, 50, window_size=12)
    # raw pixel range [0, 256] roughly
    assert (w >= 0.0).all() and (w <= 256.0).all()


def test_bad_episode_raises(loader_spatial):
    with pytest.raises(KeyError):
        loader_spatial.get_window(99999, 0, window_size=12)


def test_bad_frame_raises(loader_spatial):
    with pytest.raises(IndexError):
        loader_spatial.get_window(0, 99999, window_size=12)


# -------- LiberoTraceManager --------


def test_manager_discovers_all_four_suites():
    mgr = LiberoTraceManager(TRACE_ROOT)
    expected_dataset_names = {
        "libero_spatial_no_noops_1.0.0_lerobot",
        "libero_object_no_noops_1.0.0_lerobot",
        "libero_goal_no_noops_1.0.0_lerobot",
        "libero_10_no_noops_1.0.0_lerobot",
    }
    assert set(mgr.loaders.keys()) == expected_dataset_names


def test_manager_explicit_names_subset():
    mgr = LiberoTraceManager(
        TRACE_ROOT,
        dataset_names=["libero_spatial_no_noops_1.0.0_lerobot", "bridge_orig_1.0.0_lerobot"],
    )
    assert "libero_spatial_no_noops_1.0.0_lerobot" in mgr.loaders
    assert "bridge_orig_1.0.0_lerobot" not in mgr.loaders  # non-LIBERO ignored


def test_manager_get_window_libero():
    mgr = LiberoTraceManager(TRACE_ROOT)
    w = mgr.get_window("libero_object_no_noops_1.0.0_lerobot", 0, 30)
    assert w is not None
    assert w.shape == (12, 2)


def test_manager_get_window_non_libero_returns_none():
    mgr = LiberoTraceManager(TRACE_ROOT)
    assert mgr.get_window("bridge_orig_1.0.0_lerobot", 0, 0) is None


# -------- TraceAugmentedDataset --------


class _FakeBase(Dataset):
    """Mimics LeRobotMixtureDataset sample dict shape from datasets.py:1967-1976."""

    def __init__(self, n=4):
        self._n = n
        self.dataset_name = "fake_mixture"

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return {
            "action": np.zeros((1, 7), dtype=np.float32),
            "image": [],
            "lang": "fake task",
            "task_text": "fake task",
            "dataset_name": "libero_spatial_no_noops_1.0.0_lerobot",
            "trajectory_id": idx % 5,
            "step": 30 + idx,
        }


def test_wrapper_attaches_trace_for_libero():
    mgr = LiberoTraceManager(TRACE_ROOT)
    wrap = TraceAugmentedDataset(_FakeBase(n=3), mgr)
    s = wrap[0]
    assert "trace_coords_window" in s
    assert s["trace_coords_window"].shape == (12, 2)
    assert s["trace_coords_window"].dtype == np.float32


def test_wrapper_passthrough_when_manager_none():
    wrap = TraceAugmentedDataset(_FakeBase(n=2), None)
    s = wrap[0]
    assert "trace_coords_window" not in s
    assert s["dataset_name"] == "libero_spatial_no_noops_1.0.0_lerobot"


def test_wrapper_passthrough_for_non_libero_dataset():
    class _NonLiberoBase(Dataset):
        def __len__(self):
            return 1
        def __getitem__(self, idx):
            return {
                "action": np.zeros((1, 7), dtype=np.float32),
                "image": [],
                "lang": "x",
                "task_text": "x",
                "dataset_name": "bridge_orig_1.0.0_lerobot",
                "trajectory_id": 0,
                "step": 0,
            }
    wrap = TraceAugmentedDataset(_NonLiberoBase(), LiberoTraceManager(TRACE_ROOT))
    s = wrap[0]
    assert "trace_coords_window" not in s


def test_wrapper_attr_passthrough():
    wrap = TraceAugmentedDataset(_FakeBase(n=3), None)
    # attribute fallback to base
    assert wrap.dataset_name == "fake_mixture"
    assert len(wrap) == 3


# -------- anchor_indices sub-sampling --------


def test_anchor_indices_none_returns_full_window(loader_spatial):
    """v0 / backward-compat: anchor_indices=None → full window_size."""
    w = loader_spatial.get_window(0, 50, window_size=12, anchor_indices=None)
    assert w.shape == (12, 2)


def test_anchor_indices_4_returns_4_points(loader_spatial):
    """the LM-trace: sub-sample window down to 4 points."""
    w = loader_spatial.get_window(0, 50, window_size=12, anchor_indices=(0, 4, 8, 11))
    assert w.shape == (4, 2)
    # Equivalent to slicing the full window
    full = loader_spatial.get_window(0, 50, window_size=12, anchor_indices=None)
    np.testing.assert_array_equal(w, full[[0, 4, 8, 11]])


def test_anchor_indices_6_returns_6_points(loader_spatial):
    w = loader_spatial.get_window(0, 50, window_size=12, anchor_indices=(0, 2, 5, 7, 9, 11))
    assert w.shape == (6, 2)


def test_anchor_indices_out_of_window_raises(loader_spatial):
    with pytest.raises(IndexError):
        loader_spatial.get_window(0, 50, window_size=12, anchor_indices=(0, 4, 12))


def test_manager_anchor_indices_passthrough():
    mgr = LiberoTraceManager(TRACE_ROOT, anchor_indices=(0, 4, 8, 11))
    w = mgr.get_window("libero_spatial_no_noops_1.0.0_lerobot", 0, 50)
    assert w is not None
    assert w.shape == (4, 2)


def test_manager_anchor_indices_per_call_override():
    mgr = LiberoTraceManager(TRACE_ROOT, anchor_indices=None)
    # Default → full 12 frames
    w12 = mgr.get_window("libero_spatial_no_noops_1.0.0_lerobot", 0, 50)
    assert w12.shape == (12, 2)
    # Per-call override → 4 frames
    w4 = mgr.get_window("libero_spatial_no_noops_1.0.0_lerobot", 0, 50, anchor_indices=(0, 4, 8, 11))
    assert w4.shape == (4, 2)
