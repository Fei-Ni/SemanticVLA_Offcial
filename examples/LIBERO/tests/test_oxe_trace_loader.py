from pathlib import Path

import numpy as np

from semanticvla.dataloader.gr00t_lerobot.trace_loader import (
    LatentActionLabelManager,
    OxeTraceIndexManager,
)


TRACE_INDEX_ROOT = Path("${WORK_ROOT}/trace_npy_index")


def test_oxe_trace_index_manager_loads_bridge_window():
    mgr = OxeTraceIndexManager(
        trace_root=TRACE_INDEX_ROOT,
        dataset_names=["bridge_orig_1.0.0_lerobot"],
        window_size=12,
        anchor_indices=(0, 4, 8, 11),
    )

    assert mgr.has("bridge_orig_1.0.0_lerobot")
    window = mgr.get_window("bridge_orig_1.0.0_lerobot", 0, 16)

    assert window is not None
    assert window.shape == (4, 2)
    assert window.dtype == np.float32
    assert np.all(window >= 0.0)
    assert np.all(window <= 1.0)


def test_oxe_trace_index_manager_ignores_non_oxe_dataset():
    mgr = OxeTraceIndexManager(
        trace_root=TRACE_INDEX_ROOT,
        dataset_names=["libero_spatial_no_noops_1.0.0_lerobot"],
    )

    assert not mgr.has("libero_spatial_no_noops_1.0.0_lerobot")
    assert mgr.get_window("libero_spatial_no_noops_1.0.0_lerobot", 0, 0) is None


def test_latent_action_label_manager_clip_policy(tmp_path):
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    (label_dir / "bridge_orig_1.0.0_lerobot.jsonl").write_text(
        "\n".join(
            [
                '{"dataset_name":"bridge_orig_1.0.0_lerobot","episode_index":7,"step_index":0,"indices":[1,2]}',
                '{"dataset_name":"bridge_orig_1.0.0_lerobot","episode_index":7,"step_index":3,"indices":[3,4]}',
            ]
        )
        + "\n"
    )

    strict_mgr = LatentActionLabelManager(label_dir, missing_policy="error")
    assert strict_mgr.get("bridge_orig_1.0.0_lerobot", 7, 4) is None

    clip_mgr = LatentActionLabelManager(label_dir, missing_policy="clip")
    np.testing.assert_array_equal(
        clip_mgr.get("bridge_orig_1.0.0_lerobot", 7, 4),
        np.asarray([3, 4], dtype=np.int64),
    )
    assert clip_mgr.get("bridge_orig_1.0.0_lerobot", 8, 0) is None
