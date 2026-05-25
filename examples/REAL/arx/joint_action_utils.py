from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from semanticvla.model.framework.share_tools import read_mode_config


def resolve_unnorm_key(norm_stats: dict[str, Any], unnorm_key: str | None) -> str:
    available_keys = sorted(norm_stats.keys())
    if unnorm_key is None:
        if len(available_keys) == 1:
            return available_keys[0]
        raise ValueError(
            "`unnorm_key` must be provided when multiple normalization-stat entries exist. "
            f"Available keys: {available_keys}"
        )

    if unnorm_key not in norm_stats:
        raise KeyError(f"Unknown `unnorm_key`: {unnorm_key}. Available keys: {available_keys}")
    return unnorm_key


def get_action_chunk_size(policy_ckpt_path: str | Path) -> int:
    model_config, _ = read_mode_config(Path(policy_ckpt_path))
    return model_config["framework"]["action_model"]["future_action_window_size"] + 1


def get_action_stats(
    policy_ckpt_path: str | Path,
    unnorm_key: str | None = None,
    action_mode: str = "abs",
) -> tuple[str, dict[str, Any]]:
    _, norm_stats = read_mode_config(Path(policy_ckpt_path))
    resolved_key = resolve_unnorm_key(norm_stats, unnorm_key)
    stats = norm_stats[resolved_key]

    if action_mode in stats:
        mode_stats = stats[action_mode]
        return resolved_key, mode_stats.get("action", mode_stats)

    if "action" in stats:
        if action_mode != "abs":
            raise ValueError(
                f"Statistics for `{resolved_key}` only provide absolute actions, "
                f"but action_mode=`{action_mode}` was requested."
            )
        return resolved_key, stats["action"]

    raise ValueError(
        f"Invalid statistics format for `{resolved_key}`. Top-level keys: {sorted(stats.keys())}"
    )


def _get_bounds(action_stats: dict[str, Any], normalization_mode: str = "min_max") -> tuple[np.ndarray, np.ndarray]:
    if normalization_mode == "min_max":
        if "min" not in action_stats or "max" not in action_stats:
            raise KeyError("Expected `min`/`max` in action statistics for min_max unnormalization.")
        return np.asarray(action_stats["max"]), np.asarray(action_stats["min"])

    if normalization_mode == "q99":
        if "q01" not in action_stats or "q99" not in action_stats:
            raise KeyError("Expected `q01`/`q99` in action statistics for q99 unnormalization.")
        return np.asarray(action_stats["q99"]), np.asarray(action_stats["q01"])

    raise ValueError(f"Unsupported normalization_mode: {normalization_mode}")


def unnormalize_joint_actions(
    normalized_actions: np.ndarray,
    action_stats: dict[str, Any],
    normalization_mode: str = "min_max",
    clip_range: tuple[float, float] = (-1.0, 1.0),
    respect_mask: bool = True,
) -> np.ndarray:
    action_high, action_low = _get_bounds(action_stats, normalization_mode=normalization_mode)
    if respect_mask:
        mask = np.asarray(action_stats.get("mask", np.ones_like(action_low, dtype=bool))).astype(bool)
    else:
        mask = np.ones_like(action_low, dtype=bool)

    normalized = np.asarray(normalized_actions, dtype=np.float32)
    normalized = np.clip(normalized, clip_range[0], clip_range[1])

    raw_actions = np.where(
        mask,
        0.5 * (normalized + 1.0) * (action_high - action_low) + action_low,
        normalized,
    )
    return raw_actions.astype(np.float32)


def build_deployment_schedule(
    policy_ckpt_path: str | Path,
    dataset_fps: float = 20.0,
    execute_horizon: int = 4,
) -> dict[str, float | int]:
    action_chunk_size = get_action_chunk_size(policy_ckpt_path)
    if execute_horizon <= 0:
        raise ValueError("execute_horizon must be positive")
    if execute_horizon > action_chunk_size:
        raise ValueError(
            f"execute_horizon ({execute_horizon}) cannot exceed action_chunk_size ({action_chunk_size})."
        )

    control_hz = float(dataset_fps)
    query_hz = control_hz / execute_horizon
    return {
        "dataset_fps": control_hz,
        "action_chunk_size": action_chunk_size,
        "execute_horizon": execute_horizon,
        "query_hz": query_hz,
        "control_dt": 1.0 / control_hz,
    }
