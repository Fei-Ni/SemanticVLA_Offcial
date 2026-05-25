"""Text codec for SemanticVLA trace and latent-action LM targets."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np

from semanticvla.model.modules.action_model.trace_text_codec import (
    format_trace_to_text,
    parse_trace_text,
    prompt_template as trace_prompt_template,
)


_LAM_RE_TEMPLATE = r"<LAM_(\d+)>"


def lam_tokens(num_tokens: int) -> list[str]:
    return [f"<LAM_{i}>" for i in range(int(num_tokens))]


def format_lam_tokens(indices: Iterable[int], prefix: str = "LAM") -> str:
    return "".join(f"<{prefix}_{int(idx)}>" for idx in indices)


def parse_lam_tokens(text: str, *, prefix: str = "LAM", max_tokens: int | None = None) -> list[int]:
    pattern = _LAM_RE_TEMPLATE if prefix == "LAM" else rf"<{re.escape(prefix)}_(\d+)>"
    out = [int(m.group(1)) for m in re.finditer(pattern, text)]
    return out[:max_tokens] if max_tokens is not None else out


def semantic_prompt_template(
    *,
    mode: str,
    prompt_style: str,
    num_anchors: int,
    latent_num_tokens: int,
    order: str = "trace_latent",
) -> str:
    mode = str(mode).lower()
    order = str(order).lower()
    if mode == "trace_only":
        return trace_prompt_template(prompt_style, num_anchors)
    if mode == "latent_only":
        return f"Predict {latent_num_tokens} latent action tokens."
    if mode in {"trace_latent", "latent_trace"}:
        trace_part = trace_prompt_template(prompt_style, num_anchors)
        if order == "latent_trace" or mode == "latent_trace":
            return f"Predict {latent_num_tokens} latent action tokens, then {trace_part}"
        return f"{trace_part} Then predict {latent_num_tokens} latent action tokens."
    raise ValueError(f"unknown semantic output mode: {mode!r}")


def format_semantic_target(
    *,
    mode: str,
    trace_coords: np.ndarray | None,
    latent_indices: Iterable[int] | None,
    prompt_style: str = "plain",
    coord_range: int = 1000,
    order: str = "trace_latent",
    latent_prefix: str = "LAM",
) -> str:
    mode = str(mode).lower()
    order = str(order).lower()
    trace_text = ""
    latent_text = ""
    if mode in {"trace_only", "trace_latent", "latent_trace"}:
        if trace_coords is None:
            raise ValueError(f"mode={mode} requires trace_coords")
        trace_text = format_trace_to_text(trace_coords, style=prompt_style, coord_range=coord_range)
    if mode in {"latent_only", "trace_latent", "latent_trace"}:
        if latent_indices is None:
            raise ValueError(f"mode={mode} requires latent_indices")
        latent_text = format_lam_tokens(latent_indices, prefix=latent_prefix)
    if mode == "trace_only":
        return trace_text
    if mode == "latent_only":
        return latent_text
    if order == "latent_trace" or mode == "latent_trace":
        return f"{latent_text}{trace_text}"
    return f"{trace_text}{latent_text}"


def format_semantic_batch(
    *,
    mode: str,
    batch_trace_coords: Iterable[np.ndarray] | None,
    batch_latent_indices: Iterable[Iterable[int]] | None,
    prompt_style: str = "plain",
    coord_range: int = 1000,
    order: str = "trace_latent",
    latent_prefix: str = "LAM",
) -> list[str]:
    traces = list(batch_trace_coords) if batch_trace_coords is not None else None
    latents = list(batch_latent_indices) if batch_latent_indices is not None else None
    if traces is None and latents is None:
        return []
    n = len(traces) if traces is not None else len(latents)
    out = []
    for i in range(n):
        out.append(
            format_semantic_target(
                mode=mode,
                trace_coords=traces[i] if traces is not None else None,
                latent_indices=latents[i] if latents is not None else None,
                prompt_style=prompt_style,
                coord_range=coord_range,
                order=order,
                latent_prefix=latent_prefix,
            )
        )
    return out


def parse_trace_from_semantic_text(
    text: str,
    *,
    num_anchors: int,
    style: str = "plain",
    coord_range: int = 1000,
) -> np.ndarray:
    return parse_trace_text(text, num_anchors=num_anchors, style=style, coord_range=coord_range)
