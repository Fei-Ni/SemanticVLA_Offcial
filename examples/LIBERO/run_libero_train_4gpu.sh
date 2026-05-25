#!/usr/bin/env bash
set -euo pipefail

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_2,mlx5_3}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1000}"
export WANDB_MODE="${WANDB_MODE:-offline}"

FRAMEWORK_NAME="${FRAMEWORK_NAME:-QwenGR00T}"
MODELS_ROOT="${MODELS_ROOT:-${HOME}/models}"
DATASETS_ROOT="${DATASETS_ROOT:-${HOME}/datasets}"
RESULTS_ROOT="${RESULTS_ROOT:-${HOME}/results}"
BASE_VLM="${BASE_VLM:-${MODELS_ROOT}/Qwen3-VL-4B-Instruct}"
CONFIG_YAML="${CONFIG_YAML:-./semanticvla/config/training/cotrain_libero.yaml}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-${DATASETS_ROOT}/LEROBOT_LIBERO_DATA}"
DATA_MIX="${DATA_MIX:-libero_all}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${RESULTS_ROOT}/vla_train}"
RUN_ID="${RUN_ID:-qwen3vl_gr00t_libero4in1_4gpu_repro_$(date -u +%Y%m%d_%H%M%S)}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-30000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-100}"
LEARNING_RATE_BASE="${LEARNING_RATE_BASE:-4e-5}"
FREEZE_MODULE_LIST="${FREEZE_MODULE_LIST:-}"
WANDB_PROJECT="${WANDB_PROJECT:-semanticvla}"
WANDB_ENTITY="${WANDB_ENTITY:-offline}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-semanticvla/config/deepseeds/deepspeed_zero2_4gpu.yaml}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"

if [[ ! -f "${CONFIG_YAML}" ]]; then
  echo "Missing config yaml: ${CONFIG_YAML}" >&2
  exit 1
fi

if [[ ! -f "${ACCELERATE_CONFIG}" ]]; then
  echo "Missing accelerate config: ${ACCELERATE_CONFIG}" >&2
  exit 1
fi

if [[ ! -d "${BASE_VLM}" ]]; then
  echo "Missing base model directory: ${BASE_VLM}" >&2
  exit 1
fi

if [[ ! -d "${LIBERO_DATA_ROOT}" ]]; then
  echo "Missing LIBERO data root: ${LIBERO_DATA_ROOT}" >&2
  exit 1
fi

if [[ -z "${MAIN_PROCESS_PORT}" || "${MAIN_PROCESS_PORT}" == "0" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    MAIN_PROCESS_PORT="$((20000 + (SLURM_JOB_ID % 20000)))"
  else
    MAIN_PROCESS_PORT="$(python - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)"
  fi
fi

OUTPUT_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
cp "$0" "${OUTPUT_DIR}/"

declare -a TRAIN_ARGS=(
  --config_yaml "${CONFIG_YAML}"
  --framework.name "${FRAMEWORK_NAME}"
  --framework.qwenvl.base_vlm "${BASE_VLM}"
  --datasets.vla_data.data_root_dir "${LIBERO_DATA_ROOT}"
  --datasets.vla_data.data_mix "${DATA_MIX}"
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}"
  --trainer.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}"
  --trainer.save_interval "${SAVE_INTERVAL}"
  --trainer.logging_frequency "${LOGGING_FREQUENCY}"
  --trainer.eval_interval "${EVAL_INTERVAL}"
  --trainer.learning_rate.base "${LEARNING_RATE_BASE}"
  --run_root_dir "${RUN_ROOT_DIR}"
  --run_id "${RUN_ID}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_entity "${WANDB_ENTITY}"
)

if [[ -n "${FREEZE_MODULE_LIST}" ]]; then
  TRAIN_ARGS+=(--trainer.freeze_modules "${FREEZE_MODULE_LIST}")
fi

echo "==== LIBERO 4-GPU Training Config ===="
printf 'FRAMEWORK_NAME=%s\n' "${FRAMEWORK_NAME}"
printf 'BASE_VLM=%s\n' "${BASE_VLM}"
printf 'LIBERO_DATA_ROOT=%s\n' "${LIBERO_DATA_ROOT}"
printf 'DATA_MIX=%s\n' "${DATA_MIX}"
printf 'RUN_ROOT_DIR=%s\n' "${RUN_ROOT_DIR}"
printf 'RUN_ID=%s\n' "${RUN_ID}"
printf 'PER_DEVICE_BATCH_SIZE=%s\n' "${PER_DEVICE_BATCH_SIZE}"
printf 'GRADIENT_ACCUMULATION_STEPS=%s\n' "${GRADIENT_ACCUMULATION_STEPS}"
printf 'MAX_TRAIN_STEPS=%s\n' "${MAX_TRAIN_STEPS}"
printf 'SAVE_INTERVAL=%s\n' "${SAVE_INTERVAL}"
printf 'EVAL_INTERVAL=%s\n' "${EVAL_INTERVAL}"
printf 'WANDB_MODE=%s\n' "${WANDB_MODE}"
printf 'MAIN_PROCESS_PORT=%s\n' "${MAIN_PROCESS_PORT}"

accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes 4 \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  semanticvla/training/train.py \
  "${TRAIN_ARGS[@]}"
