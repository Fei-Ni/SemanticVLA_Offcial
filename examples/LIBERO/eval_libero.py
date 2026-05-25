from __future__ import annotations
import dataclasses
import datetime as dt
import json
import logging
import math
import os
import pathlib
from pathlib import Path
import requests
import time

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.LIBERO.model2libero_interface import SemanticVLAInference


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    resize_size = [224,224]

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    task_id: int = -1  # Optional: run only a single task id within the suite
    episode_idx: int = -1  # Optional: run only a single init-state episode index
    episode_start_idx: int = 0  # Optional: first init-state episode index when episode_idx is unset
    episode_end_idx: int = -1  # Optional: exclusive end init-state episode index when episode_idx is unset

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    post_process_action: bool = True

    job_name: str = "test"

    # Optional: directory containing {suite}/exp_directions_L18.npz for optional probe-direction (oracle mode)
    probe_direction_dir: str = ""
    progress_source: str = ""
    progress_query_cache_root: str = ""
    progress_semantic_variant_root: str = ""
    progress_delta_direction_key: str = ""
    record_progress_variants: bool = False


def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # args.video_out_path = f"{date_base}+{args.job_name}"
    
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # Load probe direction for optional probe-direction (oracle mode) (d_C_suite from suite-specific npz)
    probe_direction = None
    if args.probe_direction_dir:
        probe_npz = pathlib.Path(args.probe_direction_dir) / args.task_suite_name / "exp_directions_L18.npz"
        if probe_npz.exists():
            data = np.load(str(probe_npz))
            probe_direction = data["d_C_suite"].astype(np.float32)
            logging.info(f"Loaded probe direction from {probe_npz}, shape={probe_direction.shape}")
        else:
            logging.warning(f"Probe direction file not found: {probe_npz}. Falling back to oracle mode.")

    model = SemanticVLAInference(
        policy_ckpt_path=args.pretrained_path,
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
        max_steps=max_steps,
        probe_direction=probe_direction,
        progress_source=args.progress_source,
        progress_query_cache_root=args.progress_query_cache_root,
        progress_semantic_variant_root=args.progress_semantic_variant_root,
        progress_delta_direction_key=args.progress_delta_direction_key,
        record_progress_variants=args.record_progress_variants,
    )


    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_ids = [args.task_id] if args.task_id >= 0 else list(range(num_tasks_in_suite))
    for task_id in tqdm.tqdm(task_ids):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Start episodes
        task_episodes, task_successes = 0, 0
        if args.episode_idx >= 0:
            episode_ids = [args.episode_idx]
        else:
            episode_start = max(0, int(args.episode_start_idx))
            episode_end = (
                int(args.episode_end_idx)
                if args.episode_end_idx >= 0
                else int(args.num_trials_per_task)
            )
            episode_end = min(episode_end, int(args.num_trials_per_task))
            if episode_end <= episode_start:
                raise ValueError(
                    f"Invalid episode range: start={episode_start}, end={episode_end}, "
                    f"num_trials_per_task={args.num_trials_per_task}"
                )
            episode_ids = list(range(episode_start, episode_end))

        for episode_idx in tqdm.tqdm(episode_ids):
            if episode_idx >= len(initial_states):
                raise IndexError(
                    f"episode_idx={episode_idx} out of range for task_id={task_id} "
                    f"(available init states={len(initial_states)})"
                )
            # Recreate env per episode to avoid EGL/render context accumulation crash.
            # Long-lived OffScreenRenderEnv accumulates MuJoCo/EGL state that causes
            # SIGABRT in robosuite.utils.binding_utils.read_pixels after many episodes.
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

            logging.info(f"\nTask: {task_description}")

            # Reset environment
            model.reset(task_description=task_description)  # Reset the client connection
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            full_actions = []
            progress_trace = []   # per-step predicted progress ((unused for SemanticVLA) empty for other models)
            progress_variant_traces = {}
            selected_progress_source = None

            logging.info(f"Starting episode {task_episodes + 1}...")
            step = 0
            
            # full_actions = np.load("./debug/action.npy")
            
            while t < max_steps + args.num_steps_wait:
                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(
                    obs["robot0_eye_in_hand_image"][::-1, ::-1]
                )

                # Save preprocessed image for replay video
                replay_images.append(img)

                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )

                observation = { # 
                    "observation.primary": np.expand_dims(
                        img, axis=0
                    ),  # (H, W, C), dtype=unit8, range(0-255)
                    "observation.wrist_image": np.expand_dims(
                        wrist_img, axis=0
                    ),  # (H, W, C)
                    "observation.state": np.expand_dims(state, axis=0),
                    "instruction": [str(task_description)],
                }

                # align key with model API
                obs_input = {
                    "images": [observation["observation.primary"][0], observation["observation.wrist_image"][0]],
                    "task_description": observation["instruction"][0],  
                    "step": step,
                }

                
                start_time = time.time()

                response = model.step(**obs_input)

                end_time = time.time()

                raw_action = response["raw_action"]
                if "predicted_progress" in response:
                    progress_trace.append(response["predicted_progress"])
                if "progress_variants" in response:
                    for key, value in response["progress_variants"].items():
                        progress_variant_traces.setdefault(str(key), []).append(float(value))
                if "selected_progress_source" in response:
                    selected_progress_source = str(response["selected_progress_source"])
                
                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper_open(open_gripper)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    logging.warning(f"Unexpected action sizes: "
                                    f"wv={world_vector_delta.shape}, rot={rotation_delta.shape}, grip={gripper.shape}. "
                                    f"Falling back to LIBERO_DUMMY_ACTION.")
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                    )
                else:
                    delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

                full_actions.append(delta_action)
                
                # __import__("ipdb").set_trace()
                # see ../robosuite/controllers/controller_factory.py
                obs, reward, done, info = env.step(delta_action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")

            # Save per-step progress trace ((unused for SemanticVLA) no-op for baseline/Y1)
            if progress_trace or progress_variant_traces:
                progress_path = (
                    pathlib.Path(args.video_out_path)
                    / f"progress_{task_segment}_episode{episode_idx}_{suffix}.json"
                )
                payload = {
                    "task": task_description,
                    "episode": episode_idx,
                    "success": bool(done),
                }
                action_chunk_size = int(getattr(model, "action_chunk_size", 1) or 1)
                payload["progress_update_stride"] = action_chunk_size
                payload["progress_update_mode"] = (
                    "action_chunk_repeat" if action_chunk_size > 1 else "per_step"
                )
                if progress_trace:
                    payload["progress"] = [float(p) for p in progress_trace]
                if progress_variant_traces:
                    payload["progress_variants"] = {
                        key: [float(v) for v in values]
                        for key, values in sorted(progress_variant_traces.items())
                    }
                if selected_progress_source is not None:
                    payload["selected_progress_source"] = selected_progress_source
                with open(progress_path, "w") as _f:
                    json.dump(payload, _f)

            imageio.mimwrite(
                pathlib.Path(args.video_out_path)
                / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            
            if full_actions:
                full_actions = np.stack(full_actions)
            # np.save(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.npy", full_actions)

            # print(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4")
            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
            )

            # Close env to release EGL/render context before next episode
            try:
                env.close()
            except Exception:
                pass

        # Log final results
        logging.info(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}"
        )
        logging.info(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}"
        )

    logging.info(
        f"Total success rate: {float(total_successes) / float(total_episodes)}"
    )
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(
        seed
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def start_debugpy_once():
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True

if __name__ == "__main__":
    if os.getenv("DEBUG", "").lower() in {"1", "true", "yes"}:
        start_debugpy_once()
    tyro.cli(eval_libero)
