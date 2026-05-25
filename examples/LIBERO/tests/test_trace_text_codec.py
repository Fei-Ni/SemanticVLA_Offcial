"""Unit tests for trace_text_codec — format/parse roundtrip and malformed input.

Run:
    cd ${REPO_ROOT}
    PYTHONPATH=. python -m pytest examples/LIBERO/semanticvla/tests/test_trace_text_codec.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from semanticvla.model.modules.action_model.trace_text_codec import (
    DEFAULT_ANCHOR_INDICES,
    PROMPT_STYLES,
    format_trace_batch,
    format_trace_to_text,
    get_default_anchor_indices,
    parse_trace_batch,
    parse_trace_text,
    prompt_template,
)


COORDS_4 = np.array(
    [[0.544, 0.372], [0.615, 0.420], [0.728, 0.512], [0.823, 0.612]],
    dtype=np.float32,
)
COORDS_6 = np.array(
    [[0.10, 0.20], [0.30, 0.40], [0.50, 0.60], [0.70, 0.80], [0.85, 0.90], [0.92, 0.95]],
    dtype=np.float32,
)


# ---------------- format ----------------


def test_format_plain():
    txt = format_trace_to_text(COORDS_4, style="plain")
    assert txt == "[[544, 372], [615, 420], [728, 512], [823, 612]]"


def test_format_cot_bbox_same_as_plain():
    a = format_trace_to_text(COORDS_4, style="plain")
    b = format_trace_to_text(COORDS_4, style="cot_bbox")
    assert a == b


def test_format_qwen_point_2d_includes_labels():
    txt = format_trace_to_text(COORDS_4, style="qwen_point_2d")
    assert '"point_2d": [544, 372]' in txt
    assert '"label": "start"' in txt
    assert '"label": "end"' in txt


def test_format_unknown_style_raises():
    with pytest.raises(ValueError):
        format_trace_to_text(COORDS_4, style="not_a_style")


def test_format_invalid_shape_raises():
    with pytest.raises(ValueError):
        format_trace_to_text(np.zeros((4, 3), dtype=np.float32), style="plain")


def test_format_clips_out_of_range():
    out_of_range = np.array([[-0.1, 0.5], [0.5, 1.2]], dtype=np.float32)
    txt = format_trace_to_text(out_of_range, style="plain")
    # Both clipped to [0, 1] before *1000
    assert "[0, 500]" in txt
    assert "[500, 1000]" in txt


# ---------------- prompt template ----------------


@pytest.mark.parametrize("style", PROMPT_STYLES)
def test_prompt_template_returns_string(style):
    prompt = prompt_template(style, 4)
    assert isinstance(prompt, str)
    assert "4" in prompt  # mentions the anchor count


# ---------------- round-trip (lossless modulo int rounding) ----------------


@pytest.mark.parametrize("style", PROMPT_STYLES)
def test_roundtrip_lossless_for_int_friendly_coords(style):
    """When coords map to exact integers under *1000, parse should return the
    original floats exactly."""
    coords = np.array([[0.544, 0.372], [0.615, 0.420]], dtype=np.float32)
    txt = format_trace_to_text(coords, style=style)
    parsed = parse_trace_text(txt, num_anchors=2, style=style)
    np.testing.assert_allclose(parsed, coords, atol=1e-6)


@pytest.mark.parametrize("style", PROMPT_STYLES)
def test_roundtrip_max_err_under_rounding_tolerance(style):
    """For arbitrary floats, rounding to int costs at most 1/1000 = 1e-3."""
    rng = np.random.default_rng(0)
    coords = rng.random((4, 2)).astype(np.float32)
    txt = format_trace_to_text(coords, style=style)
    parsed = parse_trace_text(txt, num_anchors=4, style=style)
    assert np.abs(parsed - coords).max() < 1.5e-3


# ---------------- robust parsing of malformed input ----------------


def test_parse_empty_returns_zeros_padded_to_num_anchors():
    out = parse_trace_text("", num_anchors=4, style="plain")
    assert out.shape == (4, 2)
    np.testing.assert_array_equal(out, np.zeros((4, 2), dtype=np.float32))


def test_parse_truncated_pads_with_last():
    txt = "[[544, 372], [615, 420"  # second pair never closed
    out = parse_trace_text(txt, num_anchors=4, style="plain")
    assert out.shape == (4, 2)
    # First parsed pair extracted via regex
    np.testing.assert_allclose(out[0], [0.544, 0.372], atol=1e-6)
    # Last 3 rows = repeat of last successfully parsed pair
    for i in range(1, 4):
        np.testing.assert_allclose(out[i], out[0], atol=1e-6)


def test_parse_too_many_truncates():
    txt = "[[100, 100], [200, 200], [300, 300], [400, 400], [500, 500], [600, 600]]"
    out = parse_trace_text(txt, num_anchors=4, style="plain")
    assert out.shape == (4, 2)
    np.testing.assert_allclose(out[0], [0.1, 0.1], atol=1e-6)
    np.testing.assert_allclose(out[-1], [0.4, 0.4], atol=1e-6)


def test_parse_qwen_point_2d_malformed_falls_back_to_regex():
    txt = '{"point_2d": [544, 372], "label": "start"'  # truncated JSON
    out = parse_trace_text(txt, num_anchors=4, style="qwen_point_2d")
    assert out.shape == (4, 2)
    np.testing.assert_allclose(out[0], [0.544, 0.372], atol=1e-6)


def test_parse_garbage_gives_zeros():
    out = parse_trace_text("no coordinates here at all", num_anchors=4, style="plain")
    np.testing.assert_array_equal(out, np.zeros((4, 2), dtype=np.float32))


# ---------------- batch helpers ----------------


def test_format_and_parse_batch():
    batch_coords = [COORDS_4, COORDS_4 * 0.5]
    texts = format_trace_batch(batch_coords, style="plain")
    assert len(texts) == 2
    parsed = parse_trace_batch(texts, num_anchors=4, style="plain")
    assert parsed.shape == (2, 4, 2)
    np.testing.assert_allclose(parsed[0], COORDS_4, atol=1.5e-3)


# ---------------- default anchor indices ----------------


def test_default_anchors_4():
    assert get_default_anchor_indices(4) == (0, 4, 8, 11)


def test_default_anchors_6():
    assert get_default_anchor_indices(6) == (0, 2, 5, 7, 9, 11)


def test_default_anchors_unknown_raises():
    with pytest.raises(ValueError):
        get_default_anchor_indices(7)


def test_default_anchors_registry_consistent():
    for n, idxs in DEFAULT_ANCHOR_INDICES.items():
        assert len(idxs) == n, f"{n}: {idxs} length mismatch"
        assert all(0 <= i < 12 for i in idxs), f"{n}: {idxs} out of window"
        assert idxs == tuple(sorted(idxs)), f"{n}: indices not monotone"
