#!/usr/bin/env bash
set -euo pipefail

policy_host="${POLICY_HOST:-127.0.0.1}"
policy_port="${POLICY_PORT:-10093}"
control_dt="${CONTROL_DT:-0.05}"
execute_horizon="${EXECUTE_HORIZON:-10}"
max_episode_steps="${MAX_EPISODE_STEPS:-200}"
task_prompt="${TASK_PROMPT:-stack the two paper cups on top of the paper cup closest to the shelf one by one and place the stacked cups on the shelf}"
arm_side="${ARM_SIDE:-right}"
camera_keys="${CAMERA_KEYS:-camera_h,camera_r}"
image_size="${IMAGE_SIZE:-640,480}"
include_state="${INCLUDE_STATE:-1}"
smoke_test_dataset="${SMOKE_TEST_DATASET:-Collect/lerobot_v3/gravity_single4IL}"
python_bin="${PYTHON_BIN:-python3}"

export PYTHONUNBUFFERED=1

extra_args=("$@")
if [[ "${include_state}" != "1" ]]; then
  extra_args+=(--no_state)
fi

"${python_bin}" deployment/model_server/arx/client_policy_arx.py \
  --policy_host "${policy_host}" \
  --policy_port "${policy_port}" \
  --control_dt "${control_dt}" \
  --execute_horizon "${execute_horizon}" \
  --max_episode_steps "${max_episode_steps}" \
  --task_prompt "${task_prompt}" \
  --arm_side "${arm_side}" \
  --camera_keys "${camera_keys}" \
  --image_size "${image_size}" \
  --smoke_test "${smoke_test_dataset}" \
  "${extra_args[@]}"
