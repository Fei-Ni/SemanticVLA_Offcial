from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LEROBOT_SRC_ROOT = REPO_ROOT / "lerobot" / "src"
if str(LEROBOT_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC_ROOT))
ROS2_ROOT = REPO_ROOT / "ARX_Realenv" / "ROS2"
if str(ROS2_ROOT) not in sys.path:
    sys.path.insert(0, str(ROS2_ROOT))


DEFAULT_CAMERA_KEYS = ("camera_h", "camera_r")
DEFAULT_IMAGE_SIZE = (640, 480)
HOME_MODE = 1


def coerce_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"", "0", "false", "none", "no"}


def parse_server_camera_keys(raw_value: Any) -> tuple[str, ...] | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        return parse_camera_keys(raw_value)
    if isinstance(raw_value, (list, tuple)):
        camera_keys = tuple(str(key).strip() for key in raw_value if str(key).strip())
        return camera_keys or None
    return None


def parse_server_image_size(raw_value: Any) -> tuple[int, int] | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, (list, tuple)) and len(raw_value) == 2:
        return (int(raw_value[0]), int(raw_value[1]))
    return None


def positive_metadata_int(metadata: dict[str, Any], key: str) -> int | None:
    raw_value = metadata.get(key)
    if raw_value is None:
        return None
    value = int(raw_value)
    return value if value > 0 else None


def selected_action_dim(arm_side: str) -> int:
    return 14 if arm_side == "both" else 7


def selected_state_dim(arm_side: str) -> int:
    return 14 if arm_side == "both" else 7


@dataclass
class DeploymentConfig:
    arm_side: str
    policy_host: str = "127.0.0.1"
    policy_port: int = 10093
    control_dt: float = 0.05
    execute_horizon: int = 10
    max_episode_steps: int = 200
    task_prompt: str = (
        "stack the two paper cups on top of the paper cup closest to the shelf "
        "one by one and place the stacked cups on the shelf"
    )
    camera_keys: tuple[str, ...] = DEFAULT_CAMERA_KEYS
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
    include_state: bool = True


def ensure_numpy_available() -> None:
    if np is None:
        raise ModuleNotFoundError(
            "numpy is required to run the ARX policy client. "
            "Use a Python environment with numpy installed."
        )


def resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def parse_camera_keys(raw_value: str) -> tuple[str, ...]:
    camera_keys = tuple(key.strip() for key in raw_value.split(",") if key.strip())
    if not camera_keys:
        raise ValueError("camera_keys cannot be empty")
    return camera_keys


def parse_image_size(raw_value: str) -> tuple[int, int]:
    parts = [part.strip() for part in raw_value.lower().split(",")]
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError("image_size must look like '640,480'")
    width, height = (int(parts[0]), int(parts[1]))
    if width <= 0 or height <= 0:
        raise ValueError("image_size must be positive")
    return (width, height)


def load_websocket_client_policy():
    from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

    return WebsocketClientPolicy


def load_arx_robot_env():
    from arx_ros2_env import ARXRobotEnv

    return ARXRobotEnv


def scalar_value(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return float(np.asarray(value).reshape(-1)[0])


def vector_value(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def bgr_to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), but got {array.shape}")
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)
    return np.ascontiguousarray(array[:, :, ::-1])


def dual_arm_robot_order_to_model_order(action: np.ndarray) -> np.ndarray:
    """Map [left6, left_gripper, right6, right_gripper] -> [left6, right6, left_gripper, right_gripper]."""
    array = np.asarray(action, dtype=np.float32).reshape(-1)
    if array.shape[0] != 14:
        return array.copy()
    return np.concatenate([array[:6], array[7:13], array[6:7], array[13:14]], axis=0).astype(np.float32)


def dual_arm_model_order_to_robot_order(action: np.ndarray) -> np.ndarray:
    """Map [left6, right6, left_gripper, right_gripper] -> [left6, left_gripper, right6, right_gripper]."""
    array = np.asarray(action, dtype=np.float32).reshape(-1)
    if array.shape[0] != 14:
        return array.copy()
    return np.concatenate([array[:6], array[12:13], array[6:12], array[13:14]], axis=0).astype(np.float32)


def build_state_from_status(status: dict[str, Any], arm_side: str) -> np.ndarray:
    def joint_state_for(side: str) -> np.ndarray:
        arm_status = status.get(side) if isinstance(status, dict) else None
        if arm_status is None:
            raise RuntimeError(f"{side} arm status is unavailable")

        joint_pos = getattr(arm_status, "joint_pos", None)
        if joint_pos is None:
            raise RuntimeError(f"{side} joint_pos is unavailable")

        state = np.asarray(joint_pos, dtype=np.float32).reshape(-1)
        if state.shape[0] < 7:
            raise RuntimeError(f"{side} joint_pos shape invalid: {state.shape}")
        return state[:7].copy()

    if arm_side == "both":
        left = joint_state_for("left")
        right = joint_state_for("right")
        return np.concatenate([left[:6], right[:6], left[6:7], right[6:7]], axis=0).astype(np.float32)
    return joint_state_for(arm_side)


def build_control_payload(action: np.ndarray, arm_side: str) -> dict[str, np.ndarray]:
    action = np.asarray(action, dtype=np.float32).reshape(-1)

    if arm_side == "both":
        if action.shape[0] < 14:
            raise RuntimeError(f"Expected 14D action for dual-arm control, but got {action.shape}")
        robot_order = dual_arm_model_order_to_robot_order(action)
        return {"left": robot_order[:7], "right": robot_order[7:14]}

    if action.shape[0] < 7:
        raise RuntimeError(f"Expected at least 7D action for single-arm control, but got {action.shape}")

    if action.shape[0] >= 14:
        start = 0 if arm_side == "left" else 7
        return {arm_side: action[start:start + 7]}
    return {arm_side: action[:7]}


def validate_execution_config(cfg: DeploymentConfig, action_chunk_size: int) -> None:
    if cfg.arm_side not in {"left", "right", "both"}:
        raise ValueError(f"Unsupported arm_side={cfg.arm_side!r}")
    if not cfg.camera_keys:
        raise ValueError("camera_keys cannot be empty")
    if action_chunk_size <= 0:
        raise ValueError(f"Invalid action_chunk_size={action_chunk_size}")
    if cfg.execute_horizon <= 0:
        raise ValueError("execute_horizon must be positive")
    if cfg.execute_horizon > action_chunk_size:
        raise ValueError(
            f"execute_horizon={cfg.execute_horizon} exceeds action_chunk_size={action_chunk_size}"
        )
    if cfg.control_dt <= 0:
        raise ValueError("control_dt must be positive")
    if cfg.max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")


def validate_server_compatibility(cfg: DeploymentConfig, metadata: dict[str, Any]) -> None:
    server_action_dim = positive_metadata_int(metadata, "action_dim")
    if server_action_dim is not None:
        expected_action_dim = selected_action_dim(cfg.arm_side)
        if server_action_dim != expected_action_dim:
            raise ValueError(
                f"arm_side={cfg.arm_side!r} expects action_dim={expected_action_dim}, "
                f"but server reports action_dim={server_action_dim}."
            )

    server_include_state = coerce_optional_bool(metadata.get("include_state"))
    if server_include_state is not None and server_include_state != cfg.include_state:
        raise ValueError(
            f"include_state={cfg.include_state} does not match server include_state={server_include_state}."
        )

    server_state_dim = positive_metadata_int(metadata, "state_dim")
    if cfg.include_state and server_state_dim is not None:
        expected_state_dim = selected_state_dim(cfg.arm_side)
        if server_state_dim != expected_state_dim:
            raise ValueError(
                f"arm_side={cfg.arm_side!r} with include_state=True expects state_dim={expected_state_dim}, "
                f"but server reports state_dim={server_state_dim}."
            )

    server_camera_keys = parse_server_camera_keys(metadata.get("camera_keys"))
    if server_camera_keys is not None and server_camera_keys != cfg.camera_keys:
        raise ValueError(
            f"camera_keys={list(cfg.camera_keys)} do not match server camera_keys={list(server_camera_keys)}."
        )

    server_num_cameras = positive_metadata_int(metadata, "num_cameras")
    if server_num_cameras is not None and len(cfg.camera_keys) != server_num_cameras:
        raise ValueError(
            f"camera_keys expects {len(cfg.camera_keys)} cameras, but server reports num_cameras={server_num_cameras}."
        )

    server_image_size = parse_server_image_size(metadata.get("image_size"))
    if server_image_size is not None and server_image_size != cfg.image_size:
        logging.warning(
            "Client image_size=%s differs from server training image_size=%s. "
            "This can still work if preprocessing handles resizing, but it is worth checking.",
            cfg.image_size,
            server_image_size,
        )


def validate_observation_payload(
    images: list[np.ndarray],
    state: np.ndarray | None,
    cfg: DeploymentConfig,
    metadata: dict[str, Any],
) -> np.ndarray | None:
    if len(images) != len(cfg.camera_keys):
        raise ValueError(f"Expected {len(cfg.camera_keys)} images from camera_keys, but got {len(images)}")

    server_num_cameras = positive_metadata_int(metadata, "num_cameras")
    if server_num_cameras is not None and len(images) != server_num_cameras:
        raise ValueError(f"Expected {server_num_cameras} images for the server, but got {len(images)}")

    if cfg.include_state:
        if state is None:
            raise ValueError("include_state=True but no state was provided")
        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        expected_state_dim = positive_metadata_int(metadata, "state_dim") or selected_state_dim(cfg.arm_side)
        if state_array.shape[0] != expected_state_dim:
            raise ValueError(
                f"Expected state_dim={expected_state_dim}, but received state with shape {state_array.shape}."
            )
        return state_array

    if state is not None:
        logging.warning("State was provided even though include_state=False; it will be omitted from the request.")
    return None


def connect_policy_client(cfg: DeploymentConfig) -> tuple[Any, dict[str, Any], int]:
    WebsocketClientPolicy = load_websocket_client_policy()
    client = WebsocketClientPolicy(host=cfg.policy_host, port=cfg.policy_port)
    try:
        metadata = client.get_server_metadata()
        action_chunk_size = int(metadata["action_chunk_size"])
    except Exception:
        client.close()
        raise

    validate_execution_config(cfg, action_chunk_size)
    validate_server_compatibility(cfg, metadata)
    logging.info("Connected to policy server at %s:%s", cfg.policy_host, cfg.policy_port)
    logging.info("Server metadata: %s", metadata)
    logging.info(
        "Client config: arm_side=%s camera_keys=%s include_state=%s action_chunk_size=%s "
        "execute_horizon=%s control_dt=%.4f query_hz=%.3f",
        cfg.arm_side,
        list(cfg.camera_keys),
        cfg.include_state,
        action_chunk_size,
        cfg.execute_horizon,
        cfg.control_dt,
        1.0 / (cfg.control_dt * float(cfg.execute_horizon)),
    )
    return client, metadata, action_chunk_size


def build_request(
    images: list[np.ndarray],
    state: np.ndarray | None,
    task_prompt: str,
) -> dict[str, Any]:
    if not images:
        raise ValueError("At least one image is required")

    validated_images: list[np.ndarray] = []
    for idx, image in enumerate(images):
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"Image {idx} must have shape (H, W, 3), but got {array.shape}")
        if array.dtype != np.uint8:
            array = array.astype(np.uint8)
        validated_images.append(array)

    request: dict[str, Any] = {
        "batch_images": [validated_images],
        "instructions": [task_prompt],
        "do_sample": False,
    }
    if state is not None:
        request["state"] = [np.asarray(state, dtype=np.float32).reshape(1, -1)]

    return request


def query_policy(
    client: Any,
    images: list[np.ndarray],
    state: np.ndarray | None,
    task_prompt: str,
    cfg: DeploymentConfig,
    metadata: dict[str, Any],
) -> np.ndarray:
    validated_state = validate_observation_payload(images=images, state=state, cfg=cfg, metadata=metadata)
    request = build_request(images=images, state=validated_state, task_prompt=task_prompt)
    if hasattr(client, "predict_action"):
        response = client.predict_action(request)
    elif hasattr(client, "infer"):
        response = client.infer(request)
    else:
        raise AttributeError("Websocket client is missing both `predict_action` and `infer` methods.")
    data = response["data"]
    if "raw_actions" not in data:
        raise KeyError(
            "Server response is missing 'raw_actions'. "
            "Use deployment/model_server/arx/server_policy_arx.py so unnormalization stays on the "
            "server side."
        )

    raw_actions = np.asarray(data["raw_actions"][0], dtype=np.float32)
    if raw_actions.ndim != 2:
        raise ValueError(f"Expected raw_actions with shape [T, D], but got {raw_actions.shape}")
    expected_action_dim = positive_metadata_int(metadata, "action_dim") or selected_action_dim(cfg.arm_side)
    if raw_actions.shape[-1] != expected_action_dim:
        raise ValueError(
            f"Expected raw_actions to have action_dim={expected_action_dim}, but got {raw_actions.shape}."
        )
    return raw_actions


def create_arx_env(cfg: DeploymentConfig):
    ARXRobotEnv = load_arx_robot_env()
    return ARXRobotEnv(
        duration_per_step=1.0 / 20.0,
        min_steps=20,
        max_v_xyz=0.25,
        max_a_xyz=0.20,
        max_v_rpy=0.3,
        max_a_rpy=1.00,
        camera_type="all",
        camera_view=cfg.camera_keys,
        img_size=cfg.image_size,
    )


def capture_live_observation(arx: Any, cfg: DeploymentConfig) -> tuple[list[np.ndarray], np.ndarray | None]:
    frames, status = arx.get_camera(
        save_dir=None,
        video=False,
        target_size=arx.img_size,
        return_status=True,
    )

    images: list[np.ndarray] = []
    for camera_key in cfg.camera_keys:
        frame_key = f"{camera_key}_color"
        if frame_key not in frames:
            raise RuntimeError(f"Missing camera frame: {frame_key}")
        images.append(bgr_to_rgb_uint8(frames[frame_key]))

    state = build_state_from_status(status, cfg.arm_side) if cfg.include_state else None
    return images, state


def close_arx_env(arx: Any | None) -> None:
    if arx is None:
        return

    errors: list[str] = []
    try:
        success, error_message = arx.set_special_mode(HOME_MODE, side="both")
        if not success:
            errors.append(f"failed to home both arms: {error_message}")
        time.sleep(3.0)
    except Exception as exc:
        errors.append(f"failed during home sequence: {exc}")
    finally:
        try:
            arx.close()
        except Exception as exc:
            errors.append(f"failed to close ARX robot env: {exc}")

    if errors:
        raise RuntimeError("; ".join(errors))


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        force=True,
    )


def build_deployment_config_from_args(args: Any) -> DeploymentConfig:
    return DeploymentConfig(
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        control_dt=args.control_dt,
        execute_horizon=args.execute_horizon,
        max_episode_steps=args.max_episode_steps,
        task_prompt=args.task_prompt,
        arm_side=args.arm_side,
        camera_keys=parse_camera_keys(args.camera_keys),
        image_size=parse_image_size(args.image_size),
        include_state=not args.no_state,
    )
