from __future__ import annotations

import argparse
import logging
import socket
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.model_server.arx.joint_action_utils import (
    get_action_chunk_size,
    get_action_stats,
    unnormalize_joint_actions,
)
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from semanticvla.model.framework.base_framework import baseframework


def parse_camera_keys(raw_value: str) -> tuple[str, ...] | None:
    camera_keys = tuple(key.strip() for key in raw_value.split(",") if key.strip())
    return camera_keys or None


def coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "0", "false", "none", "no"}


def normalize_image_size(value) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return [int(value[0]), int(value[1])]
    return None


def resolve_include_state(dataset_cfg, action_cfg) -> bool:
    dataset_value = getattr(dataset_cfg, "include_state", None) if dataset_cfg is not None else None
    if dataset_value is not None:
        return coerce_bool(dataset_value)
    return int(getattr(action_cfg, "state_dim", 0) or 0) > 0


class JointActionPolicyWrapper:
    """Wrap a SemanticVLA checkpoint and return both normalized and raw joint actions."""

    def __init__(
        self,
        ckpt_path: str,
        unnorm_key: str | None = None,
        action_mode: str = "abs",
        normalization_mode: str = "min_max",
        respect_action_mask: bool = True,
    ) -> None:
        self.ckpt_path = ckpt_path
        self.policy = baseframework.from_pretrained(ckpt_path)
        self.unnorm_key, self.action_stats = get_action_stats(
            ckpt_path,
            unnorm_key=unnorm_key,
            action_mode=action_mode,
        )
        self.action_chunk_size = get_action_chunk_size(ckpt_path)
        self.normalization_mode = normalization_mode
        self.respect_action_mask = respect_action_mask

    def to(self, *args, **kwargs):
        self.policy = self.policy.to(*args, **kwargs)
        return self

    def eval(self):
        self.policy = self.policy.eval()
        return self

    def predict_action(
        self,
        examples=None,
        batch_images=None,
        instructions=None,
        state=None,
        **kwargs,
    ):
        if examples is not None:
            result = self.policy.predict_action(examples=examples, **kwargs)
        else:
            if batch_images is None or instructions is None:
                raise TypeError(
                    "predict_action requires either `examples` or both `batch_images` and `instructions`"
                )
            result = self.policy.predict_action(
                batch_images=batch_images,
                instructions=instructions,
                state=state,
                **kwargs,
            )
        normalized_actions = result["normalized_actions"]
        raw_actions = unnormalize_joint_actions(
            normalized_actions,
            self.action_stats,
            normalization_mode=self.normalization_mode,
            respect_mask=self.respect_action_mask,
        )
        result["raw_actions"] = raw_actions
        result["unnorm_key"] = self.unnorm_key
        result["action_chunk_size"] = self.action_chunk_size
        result["normalization_mode"] = self.normalization_mode
        return result


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout", type=int, default=1800)
    parser.add_argument("--unnorm_key", type=str, default=None)
    parser.add_argument("--normalization_mode", type=str, default="min_max")
    parser.add_argument(
        "--unnormalize_all_action_dims",
        action="store_true",
        help=(
            "Ignore the saved action mask during unnormalization. Use this for datasets where "
            "gripper dimensions were min-max normalized during training and must be converted "
            "back to raw robot units at deployment/eval time."
        ),
    )
    parser.add_argument(
        "--num_inference_timesteps_override",
        type=int,
        default=None,
        help="Override the checkpoint's flow-matching inference steps at deployment time.",
    )
    parser.add_argument(
        "--camera_keys",
        type=str,
        default="",
        help="Optional comma-separated camera names to expose in server metadata for client validation.",
    )
    return parser


def main(args) -> None:
    policy = JointActionPolicyWrapper(
        ckpt_path=args.ckpt_path,
        unnorm_key=args.unnorm_key,
        action_mode="abs",
        normalization_mode=args.normalization_mode,
        respect_action_mask=not args.unnormalize_all_action_dims,
    )

    if args.num_inference_timesteps_override is not None:
        override_steps = int(args.num_inference_timesteps_override)
        if override_steps <= 0:
            raise ValueError("num_inference_timesteps_override must be positive")
        policy.policy.action_model.num_inference_timesteps = override_steps
        policy.policy.config.framework.action_model.num_inference_timesteps = override_steps

    if args.use_bf16:
        policy = policy.to(torch.bfloat16)
    policy = policy.to("cuda").eval()

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating joint-action policy server at host=%s ip=%s", hostname, local_ip)
    logging.info("Resolved unnorm_key=%s", policy.unnorm_key)
    logging.info(
        "Using num_inference_timesteps=%s",
        policy.policy.action_model.num_inference_timesteps,
    )

    action_cfg = policy.policy.config.framework.action_model
    dataset_cfg = getattr(policy.policy.config.datasets, "vla_data", None)
    server_camera_keys = parse_camera_keys(args.camera_keys)
    include_state = resolve_include_state(dataset_cfg, action_cfg)
    image_size = normalize_image_size(getattr(dataset_cfg, "image_size", None))
    train_obs_keys = list(getattr(dataset_cfg, "obs", []) or []) if dataset_cfg is not None else []

    if args.idle_timeout != 1800:
        logging.info(
            "Ignoring idle_timeout=%s because the current WebsocketPolicyServer does not expose that option",
            args.idle_timeout,
        )

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata={
            "env": "gravity_single_joint",
            "unnorm_key": policy.unnorm_key,
            "normalization_mode": args.normalization_mode,
            "action_chunk_size": policy.action_chunk_size,
            "action_dim": int(getattr(action_cfg, "action_dim", 0) or 0),
            "state_dim": int(getattr(action_cfg, "state_dim", 0) or 0),
            "include_state": include_state,
            "image_size": image_size,
            "train_obs_keys": train_obs_keys,
            "camera_keys": list(server_camera_keys) if server_camera_keys is not None else None,
            "num_cameras": len(server_camera_keys) if server_camera_keys is not None else None,
            "num_inference_timesteps": policy.policy.action_model.num_inference_timesteps,
            "unnormalize_all_action_dims": bool(args.unnormalize_all_action_dims),
        },
    )
    logging.info("server running ...")
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    main(args)
