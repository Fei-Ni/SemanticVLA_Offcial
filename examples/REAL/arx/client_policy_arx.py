from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.model_server.arx.client_utils import (
    build_control_payload,
    build_deployment_config_from_args,
    capture_live_observation,
    close_arx_env,
    configure_logging,
    connect_policy_client,
    create_arx_env,
    ensure_numpy_available,
    query_policy,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_host", type=str, default="127.0.0.1")
    parser.add_argument("--policy_port", type=int, default=10093)
    parser.add_argument("--control_dt", type=float, default=0.05)
    parser.add_argument("--execute_horizon", type=int, default=10)
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--task_prompt", type=str, required=True)
    parser.add_argument("--arm_side", type=str, required=True)
    parser.add_argument("--camera_keys", type=str, default="camera_h,camera_r")
    parser.add_argument("--image_size", type=str, default="640,480")
    parser.add_argument("--no_state", action="store_true")
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser


def run_live_policy(args: argparse.Namespace) -> None:
    cfg = build_deployment_config_from_args(args)
    client, metadata, action_chunk_size = connect_policy_client(cfg)
    arx = None
    try:
        arx = create_arx_env(cfg)
        arx.reset()

        step_idx = 0
        while step_idx < cfg.max_episode_steps:
            images, state = capture_live_observation(arx, cfg)

            query_start = time.perf_counter()
            action_chunk = query_policy(client, images, state, cfg.task_prompt, cfg=cfg, metadata=metadata)
            query_latency = time.perf_counter() - query_start

            execute_count = min(cfg.execute_horizon, len(action_chunk), cfg.max_episode_steps - step_idx)
            for local_idx in range(execute_count):
                action = np.asarray(action_chunk[local_idx], dtype=np.float32).reshape(-1)
                action_start = time.perf_counter()
                arx.step_raw_joint(build_control_payload(action, cfg.arm_side))
                step_idx += 1

                sleep_time = max(0.0, cfg.control_dt - (time.perf_counter() - action_start))
                if sleep_time > 0:
                    time.sleep(sleep_time)

            print(
                f"[live] step={step_idx} query_latency={query_latency:.3f}s "
                f"execute_count={execute_count} action_chunk_size={action_chunk_size}",
                flush=True,
            )
    finally:
        client.close()
        close_arx_env(arx)


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    ensure_numpy_available()
    configure_logging(args.log_level)
    run_live_policy(args)


if __name__ == "__main__":
    main()
