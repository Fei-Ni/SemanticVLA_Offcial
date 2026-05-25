#!/bin/bash

set -euo pipefail

SEMANTICVLA_ROOT="${SEMANTICVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-${HOME}/molmoact-modified/experiments/LIBERO}"
LIBERO_PYTHON="${LIBERO_PYTHON:-${HOME}/tools/miniforge3/envs/libero/bin/python}"
LIBERO_BENCHMARK_ROOT="${LIBERO_BENCHMARK_ROOT:-${LIBERO_HOME}/libero/libero}"
LIBERO_DATASETS_ROOT="${LIBERO_DATASETS_ROOT:-${LIBERO_HOME}/libero/datasets}"
RESULTS_ROOT="${RESULTS_ROOT:-${HOME}/results}"
TRAIN_RESULTS_ROOT="${TRAIN_RESULTS_ROOT:-${RESULTS_ROOT}/vla_train}"
EVAL_RESULTS_ROOT="${EVAL_RESULTS_ROOT:-${RESULTS_ROOT}/vla_eval}"
DEFAULT_MODEL_PATH="$(ls -dt "${TRAIN_RESULTS_ROOT}"/*/checkpoints/steps_30000_pytorch_model.pt 2>/dev/null | head -1 || true)"

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${HOME}/.libero}"
export PYTHONPATH="${LIBERO_HOME}:${SEMANTICVLA_ROOT}:${PYTHONPATH:-}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

MODEL_PATH="${1:-${MODEL_PATH:-${DEFAULT_MODEL_PATH}}}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_goal}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
POLICY_SERVER_HOST="${POLICY_SERVER_HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-10093}"
LOG_DIR="${LOG_DIR:-${EVAL_RESULTS_ROOT}/logs/$(date +"%Y%m%d_%H%M%S")}"
LIBERO_CONFIG_FILE="${LIBERO_CONFIG_PATH}/config.yaml"

if [ ! -d "${LIBERO_HOME}" ]; then
  echo "LIBERO_HOME not found: ${LIBERO_HOME}" >&2
  exit 1
fi

if [ ! -x "${LIBERO_PYTHON}" ]; then
  echo "LIBERO python not found: ${LIBERO_PYTHON}" >&2
  exit 1
fi

if [ ! -f "${MODEL_PATH}" ]; then
  echo "Checkpoint not found: ${MODEL_PATH}" >&2
  echo "Set MODEL_PATH or pass the checkpoint path as the first argument." >&2
  exit 1
fi

FOLDER_NAME="$(echo "${MODEL_PATH}" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')"
VIDEO_OUT_PATH="${VIDEO_OUT_PATH:-${EVAL_RESULTS_ROOT}/${TASK_SUITE_NAME}/${FOLDER_NAME}}"

mkdir -p "${VIDEO_OUT_PATH}" "${LOG_DIR}" "${LIBERO_CONFIG_PATH}"

# Avoid LIBERO's interactive first-run prompt in batch jobs by materializing the default config.
if [ ! -f "${LIBERO_CONFIG_FILE}" ]; then
  cat > "${LIBERO_CONFIG_FILE}" <<CFG
benchmark_root: ${LIBERO_BENCHMARK_ROOT}
bddl_files: ${LIBERO_BENCHMARK_ROOT}/bddl_files
init_states: ${LIBERO_BENCHMARK_ROOT}/init_files
datasets: ${LIBERO_DATASETS_ROOT}
assets: ${LIBERO_BENCHMARK_ROOT}/assets
CFG
fi

cd "${SEMANTICVLA_ROOT}"

PROBE_ARGS=()
if [[ -n "${PROBE_DIRECTION_DIR:-}" ]]; then
  PROBE_ARGS+=(--args.probe-direction-dir "${PROBE_DIRECTION_DIR}")
fi

"${LIBERO_PYTHON}" ./examples/LIBERO/eval_libero.py \
  --args.pretrained-path "${MODEL_PATH}" \
  --args.host "${POLICY_SERVER_HOST}" \
  --args.port "${BASE_PORT}" \
  --args.task-suite-name "${TASK_SUITE_NAME}" \
  --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
  --args.video-out-path "${VIDEO_OUT_PATH}" \
  "${PROBE_ARGS[@]}"
