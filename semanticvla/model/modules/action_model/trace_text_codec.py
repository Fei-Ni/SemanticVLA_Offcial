"""Trace text codec for SemanticVLA LM-head trace prediction.

Three prompt styles (yaml `framework.action_model.trace.prompt_style`):

- **P1 "plain"** — compact nested list, no labels, minimal token count.
    prompt suffix:  "Predict <N> trace anchors (start, 1/<k>, ..., end) in
                     [[x,y]] format with coords 0-1000."
    target:         "[[544, 372], [615, 420], [728, 512], [823, 612]]"

- **P2 "qwen_point_2d"** — official Qwen3-VL grounding output shape.
    prompt suffix:  "Predict <N> trace anchors as JSON list, each as
                     {\\"point_2d\\": [x, y], \\"label\\": \\"anchor_N\\"},
                     coords 0-1000."
    target:         "[{\\"point_2d\\": [544, 372], \\"label\\": \\"anchor_0\\"}, ...]"

- **P3 "cot_bbox"** — parallels semanticvla's existing `CoT_prompt` style.
    prompt suffix:  "To identify the gripper trace, locate <N> anchor points
                     in [[x,y]] format with coords 0-1000."
    target:         "[[544, 372], [615, 420], [728, 512], [823, 612]]"

Coordinate convention (REQUIRED by Qwen3-VL):
    integers in [0, 1000] relative to the image, NOT absolute pixel coords.
    The dataloader stores normalized [0, 1] floats; this codec multiplies
    by 1000 and rounds to int on the way out.

Parsing is best-effort robust: if the LM-generated text is malformed, we
fall back to regex extraction of integer pairs and pad/truncate to the
expected `num_anchors` count.
"""

from __future__ import annotations

import json
import re
from typing import Iterable

import numpy as np


PROMPT_STYLES = ("plain", "qwen_point_2d", "cot_bbox")


# -------------------------------------------------------------------------
# prompt templates (what we wrap the user instruction with)
# -------------------------------------------------------------------------


def _anchor_label(n: int) -> str:
    if n == 0:
        return "start"
    if n == 1:
        return "one_third"
    if n == 2:
        return "two_third"
    if n == 3:
        return "end"
    return f"anchor_{n}"


def prompt_template(style: str, num_anchors: int) -> str:
    """Return the suffix to append to the user instruction.

    Caller composes the full user prompt as `"<instruction>. <suffix>"`.
    Suffix is intentionally short to minimize token overhead.
    """
    style = style.lower()
    if style == "plain":
        return (
            f"Predict the gripper trace as {num_anchors} anchor points "
            f"(start, intermediate, end) in [[x,y]] format with coords 0-1000."
        )
    if style == "qwen_point_2d":
        # Inline an example so Qwen knows the exact JSON schema.
        return (
            f"Predict the gripper trace as a JSON list of {num_anchors} anchor "
            f'points, each formatted as {{"point_2d": [x, y], "label": "anchor_N"}}, '
            f"with coords 0-1000."
        )
    if style == "cot_bbox":
        return (
            f"To identify the gripper trace, locate {num_anchors} anchor "
            f"points in [[x,y]] format with coords 0-1000."
        )
    raise ValueError(f"unknown prompt_style {style!r}; expected one of {PROMPT_STYLES}")


# -------------------------------------------------------------------------
# encoding: coords → text (training target)
# -------------------------------------------------------------------------


def _coords_to_1000_int(coords: np.ndarray, coord_range: int = 1000) -> np.ndarray:
    """Convert normalized [0, 1] coords to integer [0, coord_range].

    Accepts shape (N, 2) — single sample.
    """
    arr = np.asarray(coords, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"coords must be (N, 2), got {arr.shape}")
    # clip in case of small numerical overflow at edges
    arr = np.clip(arr, 0.0, 1.0)
    return np.rint(arr * coord_range).astype(np.int32)


def format_trace_to_text(
    coords: np.ndarray,
    style: str = "plain",
    coord_range: int = 1000,
) -> str:
    """Encode a single sample's trace coords (N, 2) as a text string.

    `coords` are normalized floats in [0, 1] (as produced by
    `LiberoTraceLoader(..., normalize=True)`).
    Output ints are in [0, coord_range].
    """
    ints = _coords_to_1000_int(coords, coord_range=coord_range)
    style = style.lower()

    if style in ("plain", "cot_bbox"):
        body = ", ".join(f"[{int(x)}, {int(y)}]" for x, y in ints)
        return f"[{body}]"

    if style == "qwen_point_2d":
        items = [
            {"point_2d": [int(x), int(y)], "label": _anchor_label(i)}
            for i, (x, y) in enumerate(ints)
        ]
        return json.dumps(items, separators=(", ", ": "))

    raise ValueError(f"unknown prompt_style {style!r}")


def format_trace_batch(
    batch_coords: Iterable[np.ndarray],
    style: str = "plain",
    coord_range: int = 1000,
) -> list[str]:
    """Apply `format_trace_to_text` over a batch (List[np.ndarray])."""
    return [format_trace_to_text(c, style=style, coord_range=coord_range) for c in batch_coords]


# -------------------------------------------------------------------------
# decoding: text → coords (used at inference time after LM generate)
# -------------------------------------------------------------------------

_INT_PAIR_RE = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")


def parse_trace_text(
    text: str,
    num_anchors: int,
    style: str = "plain",
    coord_range: int = 1000,
) -> np.ndarray:
    """Parse generated text back to normalized coords (N, 2) in [0, 1].

    Tolerates malformed output by regex-extracting integer pairs and
    padding/truncating to `num_anchors`. Returned coords are float32 in
    [0, 1] (divided by `coord_range`, clipped).
    """
    style = style.lower()

    parsed_ints: list[tuple[int, int]] | None = None

    # First try structured parsing per style.
    try:
        if style == "qwen_point_2d":
            data = json.loads(text)
            if isinstance(data, list):
                parsed_ints = []
                for item in data:
                    if isinstance(item, dict) and "point_2d" in item:
                        x, y = item["point_2d"]
                        parsed_ints.append((int(x), int(y)))
        elif style in ("plain", "cot_bbox"):
            # try literal eval — strip surrounding whitespace
            try:
                import ast
                data = ast.literal_eval(text.strip())
                if isinstance(data, list):
                    parsed_ints = [(int(p[0]), int(p[1])) for p in data]
            except Exception:
                parsed_ints = None
    except Exception:
        parsed_ints = None

    # Fallback: regex over text — extract any `[int, int]` pairs.
    if parsed_ints is None or len(parsed_ints) == 0:
        parsed_ints = [(int(m.group(1)), int(m.group(2))) for m in _INT_PAIR_RE.finditer(text)]

    # Pad / truncate to num_anchors
    if len(parsed_ints) < num_anchors:
        if parsed_ints:
            # repeat last
            parsed_ints = parsed_ints + [parsed_ints[-1]] * (num_anchors - len(parsed_ints))
        else:
            # all-zero fallback when totally malformed
            parsed_ints = [(0, 0)] * num_anchors
    elif len(parsed_ints) > num_anchors:
        parsed_ints = parsed_ints[:num_anchors]

    arr = np.asarray(parsed_ints, dtype=np.float32)
    arr = np.clip(arr / float(coord_range), 0.0, 1.0)
    return arr


def parse_trace_batch(
    texts: list[str],
    num_anchors: int,
    style: str = "plain",
    coord_range: int = 1000,
) -> np.ndarray:
    """Parse a batch of generated trace texts → (B, N, 2) float32 in [0, 1]."""
    return np.stack(
        [parse_trace_text(t, num_anchors=num_anchors, style=style, coord_range=coord_range) for t in texts],
        axis=0,
    )


# -------------------------------------------------------------------------
# anchor index sampling — sub-sample W-frame window down to N anchors
# -------------------------------------------------------------------------

DEFAULT_ANCHOR_INDICES = {
    4: (0, 4, 8, 11),                  # start / 1/3 / 2/3 / end on W=12
    6: (0, 2, 5, 7, 9, 11),            # five-equal-ish + end on W=12
}


def get_default_anchor_indices(num_anchors: int) -> tuple[int, ...]:
    if num_anchors in DEFAULT_ANCHOR_INDICES:
        return DEFAULT_ANCHOR_INDICES[num_anchors]
    raise ValueError(
        f"no default anchor_indices for num_anchors={num_anchors}; "
        f"available: {sorted(DEFAULT_ANCHOR_INDICES.keys())}. "
        "Pass `anchor_indices` explicitly to override."
    )
