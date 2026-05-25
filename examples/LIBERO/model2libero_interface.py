from __future__ import annotations
from collections import deque
from typing import List, Optional, Sequence
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

from examples.SimplerEnv.adaptive_ensemble import AdaptiveEnsembler
from typing import Dict

from semanticvla.model.tools import read_mode_config


class SemanticVLAInference:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble = True,
        action_ensemble_horizon: Optional[int] = 3, # different cross sim
        image_size: List[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha = 0.1,
        host="0.0.0.0",
        port=10095,
        max_steps: Optional[int] = None,
        probe_direction: Optional[np.ndarray] = None,
        progress_source: str = "",
        progress_query_cache_root: str = "",
        progress_semantic_variant_root: str = "",
        progress_delta_direction_key: str = "",
        record_progress_variants: bool = False,
    ) -> None:
        
        # build client to connect server policy
        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon #0
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.max_steps = max_steps
        self.latest_predicted_progress = None
        self.latest_progress_variants: Optional[Dict[str, float]] = None
        self.latest_selected_progress_source: Optional[str] = None

        # Probe-based progress estimation (Mode B for Y1)
        self.probe_direction = probe_direction  # unit vector [H] or None
        self._probe_proj_buffer: List[float] = []  # rolling projection history for min-max norm
        self._probe_progress_override: Optional[float] = None  # sent to server on next call
        self.progress_source = str(progress_source or "").strip()
        self.progress_query_cache_root = str(progress_query_cache_root or "").strip()
        self.progress_semantic_variant_root = str(progress_semantic_variant_root or "").strip()
        self.progress_delta_direction_key = str(progress_delta_direction_key or "").strip()
        self.record_progress_variants = bool(record_progress_variants)

        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        self.action_chunk_size = self.get_action_chunk_size(policy_ckpt_path=policy_ckpt_path)
        

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.latest_predicted_progress = None
        self.latest_progress_variants = None
        self.latest_selected_progress_source = None
        self._probe_proj_buffer = []
        self._probe_progress_override = None


    def step(
        self, 
        images, 
        task_description: Optional[str] = None,
        step: int = 0,
        **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Perform one step of inference
        :param image: Input image in the format (H, W, 3), type uint8
        :param task_description: Task description text
        :return: (raw action, processed action)
        """

        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        # image: Image.Image = Image.fromarray(image)

        images = [self._resize_image(image) for image in images]
        vla_input = {
            "batch_images": [images],
            "instructions": [self.task_description],
            "unnorm_key": self.unnorm_key,
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
            "step": step,
        }
        if self.max_steps is not None:
            vla_input["max_steps"] = self.max_steps
        if step == 0:
            vla_input["progress_reset_episode"] = True
        if self.progress_source:
            vla_input["progress_source"] = self.progress_source
        if self.progress_query_cache_root:
            vla_input["progress_query_cache_root"] = self.progress_query_cache_root
        if self.progress_semantic_variant_root:
            vla_input["progress_semantic_variant_root"] = self.progress_semantic_variant_root
        if self.progress_delta_direction_key:
            vla_input["progress_delta_direction_key"] = self.progress_delta_direction_key
        if self.record_progress_variants:
            vla_input["record_progress_variants"] = True



        
        action_chunk_size = self.action_chunk_size
        if step % action_chunk_size == 0:
            # Inject probe-based progress override for optional probe-direction (oracle mode)
            if self.probe_direction is not None and self._probe_progress_override is not None:
                vla_input["progress_override"] = self._probe_progress_override

            response = self.client.infer(vla_input)
            if not isinstance(response, dict):
                raise RuntimeError(f"Unexpected inference response type: {type(response)!r}")
            if response.get("ok") is False:
                raise RuntimeError(f"Policy server inference failed: {response.get('error', response)!r}")

            response_data = response.get("data", response)
            if "normalized_actions" not in response_data:
                raise RuntimeError(f"Malformed inference response without normalized_actions: {response!r}")

            normalized_actions = response_data["normalized_actions"]  # B, chunk, D
            normalized_actions = normalized_actions[0]
            self.raw_actions = self.unnormalize_actions(normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats)

            # Capture predicted progress from  (None for other model types)
            raw_progress = response_data.get("predicted_progress", None)
            if raw_progress is not None:
                self.latest_predicted_progress = float(np.asarray(raw_progress).flatten()[0])
            else:
                self.latest_predicted_progress = None

            raw_progress_variants = response_data.get("progress_variants", None)
            if raw_progress_variants is not None:
                parsed_variants: Dict[str, float] = {}
                for key, value in raw_progress_variants.items():
                    parsed_variants[str(key)] = float(np.asarray(value).flatten()[0])
                self.latest_progress_variants = parsed_variants
            else:
                self.latest_progress_variants = None

            raw_selected_source = response_data.get("selected_progress_source", None)
            self.latest_selected_progress_source = (
                str(raw_selected_source) if raw_selected_source is not None else None
            )

            # Update probe-based progress estimate for next call (optional probe-direction (oracle mode))
            l18 = response_data.get("l18_last_token", None)
            if l18 is not None and self.probe_direction is not None:
                h = np.asarray(l18).reshape(-1)  # [H]
                proj = float(h @ self.probe_direction)
                self._probe_proj_buffer.append(proj)
                lo = min(self._probe_proj_buffer)
                hi = max(self._probe_proj_buffer)
                self._probe_progress_override = float(np.clip((proj - lo) / (hi - lo + 1e-6), 0.0, 1.0))
                self.latest_predicted_progress = self._probe_progress_override

        raw_actions = self.raw_actions[step % action_chunk_size][None]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        result = {"raw_action": raw_action}
        if self.latest_predicted_progress is not None:
            result["predicted_progress"] = self.latest_predicted_progress
        if self.latest_progress_variants is not None:
            result["progress_variants"] = dict(self.latest_progress_variants)
        if self.latest_selected_progress_source is not None:
            result["selected_progress_source"] = self.latest_selected_progress_source
        return result

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1) 
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Duplicate stats accessor (retained for backward compatibility).
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)  # read config and norm_stats

        unnorm_key = SemanticVLAInference._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path):
        model_config, _ = read_mode_config(policy_ckpt_path)  # read config and norm_stats
        # import ipdb; ipdb.set_trace()
        return model_config['framework']['action_model']['future_action_window_size'] + 1


    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
    
    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Duplicate helper (retained for backward compatibility).
        See primary _check_unnorm_key above.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key
