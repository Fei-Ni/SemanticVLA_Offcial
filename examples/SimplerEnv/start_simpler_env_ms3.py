from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval import *  # noqa: F401,F403
from mani_skill.utils import common

from examples.SimplerEnv.model2simpler_interface import SemanticVLAInference
from simpler_env.utils.env.observation_utils import get_image_from_maniskill3_obs_dict


TASK_TO_ENV_ID = {
    "widowx_spoon_on_towel": "PutSpoonOnTableClothInScene-v1",
    "widowx_carrot_on_plate": "PutCarrotOnPlateInScene-v1",
    "widowx_stack_cube": "StackGreenCubeOnYellowCubeBakedTexInScene-v1",
    "widowx_put_eggplant_in_basket": "PutEggplantInBasketScene-v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a SemanticVLA policy in SimplerEnv ManiSkill3.")
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--task", default="widowx_put_eggplant_in_basket", choices=sorted(TASK_TO_ENV_ID))
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shader", default="default")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--logging-dir", required=True)
    parser.add_argument("--save-tag", default="")
    parser.add_argument("--max-episode-steps", type=int, default=120)
    parser.add_argument("--policy-server-protocol", choices=("legacy", "policy_wrapper"), default="legacy")
    return parser.parse_args()


def to_numpy_image(obs_image: torch.Tensor) -> np.ndarray:
    image = obs_image
    if image.ndim == 4:
        image = image[0]
    image = image[..., :3].to(torch.uint8).cpu().numpy()
    return image


def main() -> None:
    args = parse_args()

    env_id = TASK_TO_ENV_ID[args.task]
    logging_dir = Path(args.logging_dir).expanduser().resolve()
    logging_dir.mkdir(parents=True, exist_ok=True)

    sensor_configs = {"shader_pack": args.shader}
    env = gym.make(
        env_id,
        obs_mode="rgb+segmentation",
        num_envs=1,
        sensor_configs=sensor_configs,
    )

    model = SemanticVLAInference(
        policy_ckpt_path=args.ckpt_path,
        policy_setup="widowx_bridge",
        port=args.port,
        host=args.host,
        action_scale=1.0,
        cfg_scale=1.5,
        server_protocol=args.policy_server_protocol,
    )

    success_values: list[float] = []
    episode_summaries: list[dict[str, object]] = []
    started_at = time.time()

    for episode_offset in range(args.num_episodes):
        episode_seed = args.seed + episode_offset
        reset_options = {"episode_id": torch.tensor([episode_seed])}
        obs, _ = env.reset(seed=episode_seed, options=reset_options)
        instruction = env.unwrapped.get_language_instruction()[0]
        model.reset(instruction)

        predicted_terminated = False
        truncated = False
        terminated = False
        step_count = 0
        success = 0.0

        while not (predicted_terminated or truncated) and step_count < args.max_episode_steps:
            image = to_numpy_image(get_image_from_maniskill3_obs_dict(env, obs))
            _, processed_action = model.step(image, instruction)
            action = np.concatenate(
                [
                    processed_action["world_vector"],
                    processed_action["rot_axangle"],
                    processed_action["gripper"],
                ],
                axis=0,
            ).astype(np.float32)
            predicted_terminated = bool(processed_action["terminate_episode"][0] > 0)

            obs, reward, terminated, truncated_tensor, info = env.step(torch.from_numpy(action)[None, :])
            info = common.to_numpy(info)
            truncated = bool(np.asarray(common.to_numpy(truncated_tensor)).any())
            terminated = bool(np.asarray(common.to_numpy(terminated)).any())
            success_arr = np.asarray(info.get("success", [0]), dtype=np.float32).reshape(-1)
            success = float(success_arr[0]) if success_arr.size else 0.0
            step_count += 1

            new_instruction = env.unwrapped.get_language_instruction()[0]
            if new_instruction != instruction:
                instruction = new_instruction

            if terminated:
                break

        max_steps_reached = bool(
            step_count >= args.max_episode_steps and not (predicted_terminated or truncated or terminated)
        )
        success_values.append(success)
        episode_summary = {
            "episode_id": episode_offset,
            "seed": episode_seed,
            "success": success,
            "steps": step_count,
            "predicted_terminated": predicted_terminated,
            "terminated": terminated,
            "truncated": truncated,
            "max_steps_reached": max_steps_reached,
        }
        episode_summaries.append(episode_summary)
        print(json.dumps({"episode_done": episode_summary}), flush=True)

    summary = {
        "task": args.task,
        "env_id": env_id,
        "num_episodes": args.num_episodes,
        "seed": args.seed,
        "save_tag": args.save_tag,
        "max_episode_steps": args.max_episode_steps,
        "ckpt_path": args.ckpt_path,
        "host": args.host,
        "port": args.port,
        "mean_success": float(np.mean(success_values)) if success_values else 0.0,
        "elapsed_sec": time.time() - started_at,
        "episodes": episode_summaries,
    }

    summary_path = logging_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    env.close()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
