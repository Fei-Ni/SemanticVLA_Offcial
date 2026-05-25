#!/bin/bash

set -euo pipefail

SEMANTICVLA_ROOT="${SEMANTICVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODELS_ROOT="${MODELS_ROOT:-${HOME}/models}"
RESULTS_ROOT="${RESULTS_ROOT:-${HOME}/results}"
SIMPLER_RESULTS_ROOT="${SIMPLER_RESULTS_ROOT:-${RESULTS_ROOT}/simpler_env_eval}"
SIM_PYTHON="${SIM_PYTHON:-${HOME}/tools/miniforge3/envs/simpler_env/bin/python}"
SIMPLERENV_PATH="${SIMPLERENV_PATH:-${HOME}/SimplerEnv}"
MODEL_PATH="${1:-${MODEL_PATH:-${MODELS_ROOT}/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt}}"
PORT="${PORT:-5678}"
TASK_REPEATS="${TASK_REPEATS:-1}"

export PYTHONPATH="${SEMANTICVLA_ROOT}:${PYTHONPATH:-}"

if [ ! -x "${SIM_PYTHON}" ]; then
  echo "SimplerEnv python not found: ${SIM_PYTHON}" >&2
  exit 1
fi

if [ ! -d "${SIMPLERENV_PATH}" ]; then
  echo "SIMPLERENV_PATH not found: ${SIMPLERENV_PATH}" >&2
  exit 1
fi

if [ ! -f "${MODEL_PATH}" ]; then
  echo "Checkpoint not found: ${MODEL_PATH}" >&2
  echo "Set MODEL_PATH or pass the checkpoint path as the first argument." >&2
  exit 1
fi

ckpt_base="$(basename "${MODEL_PATH}")"
ckpt_name="${ckpt_base%.*}"
LOGGING_DIR="${LOGGING_DIR:-${SIMPLER_RESULTS_ROOT}/${ckpt_name}/maniskill_eval}"

mkdir -p "${LOGGING_DIR}"

cd "${SEMANTICVLA_ROOT}"

export DISPLAY=""
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export LD_LIBRARY_PATH="/usr/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

scene_name=bridge_table_1_v1
robot=widowx
rgb_overlay_path="${SIMPLERENV_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png"
robot_init_x=0.147
robot_init_y=0.028

declare -a ENV_NAMES=(
  StackGreenCubeOnYellowCubeBakedTexInScene-v0
  PutCarrotOnPlateInScene-v0
  PutSpoonOnTableClothInScene-v0
)

for env in "${ENV_NAMES[@]}"; do
  for ((run_idx=1; run_idx<=TASK_REPEATS; run_idx++)); do
    echo "▶️ Launching task [${env}] run#${run_idx}"
    save_tag="${ckpt_name}_${env}_run${run_idx}"
    declare -a eval_args=(
      --ckpt-path "${MODEL_PATH}"
      --port "${PORT}"
      --robot "${robot}"
      --policy-setup widowx_bridge
      --control-freq 5
      --sim-freq 500
      --max-episode-steps 120
      --env-name "${env}"
      --scene-name "${scene_name}"
      --rgb-overlay-path "${rgb_overlay_path}"
      --robot-init-x "${robot_init_x}" "${robot_init_x}" 1
      --robot-init-y "${robot_init_y}" "${robot_init_y}" 1
      --obj-variation-mode episode
      --obj-episode-range 0 24
      --robot-init-rot-quat-center 0 0 0 1
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1
      --logging-dir "${LOGGING_DIR}"
      --additional-env-save-tags "${save_tag}"
    )
    "${SIM_PYTHON}" examples/SimplerEnv/start_simpler_env.py "${eval_args[@]}"
  done
done

scene_name=bridge_table_1_v2
robot=widowx_sink_camera_setup
rgb_overlay_path="${SIMPLERENV_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png"
robot_init_x=0.127
robot_init_y=0.06

env="PutEggplantInBasketScene-v0"
for ((run_idx=1; run_idx<=TASK_REPEATS; run_idx++)); do
  echo "▶️ Launching V2 task [${env}] run#${run_idx}"
  save_tag="${ckpt_name}_${env}_run${run_idx}"
  declare -a eval_args=(
    --ckpt-path "${MODEL_PATH}"
    --port "${PORT}"
    --robot "${robot}"
    --policy-setup widowx_bridge
    --control-freq 5
    --sim-freq 500
    --max-episode-steps 120
    --env-name "${env}"
    --scene-name "${scene_name}"
    --rgb-overlay-path "${rgb_overlay_path}"
    --robot-init-x "${robot_init_x}" "${robot_init_x}" 1
    --robot-init-y "${robot_init_y}" "${robot_init_y}" 1
    --obj-variation-mode episode
    --obj-episode-range 0 24
    --robot-init-rot-quat-center 0 0 0 1
    --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1
    --logging-dir "${LOGGING_DIR}"
    --additional-env-save-tags "${save_tag}"
  )
  "${SIM_PYTHON}" examples/SimplerEnv/start_simpler_env.py "${eval_args[@]}"
done
