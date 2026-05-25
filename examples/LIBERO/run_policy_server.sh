#!/bin/bash

set -euo pipefail

SEMANTICVLA_ROOT="${SEMANTICVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SEMANTICVLA_PYTHON="${SEMANTICVLA_PYTHON:-${HOME}/tools/miniforge3/envs/semanticvla/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-${HOME}/results}"
TRAIN_RESULTS_ROOT="${TRAIN_RESULTS_ROOT:-${RESULTS_ROOT}/vla_train}"
DEFAULT_MODEL_PATH="$(ls -dt "${TRAIN_RESULTS_ROOT}"/*/checkpoints/steps_30000_pytorch_model.pt 2>/dev/null | head -1 || true)"
MODEL_PATH="${1:-${MODEL_PATH:-${DEFAULT_MODEL_PATH}}}"
SERVER_PORT="${2:-${SERVER_PORT:-${PORT:-${BASE_PORT:-10093}}}}"

if [ ! -x "${SEMANTICVLA_PYTHON}" ]; then
  echo "SemanticVLA python not found: ${SEMANTICVLA_PYTHON}" >&2
  exit 1
fi

if [ ! -f "${MODEL_PATH}" ]; then
  echo "Checkpoint not found: ${MODEL_PATH}" >&2
  echo "Set MODEL_PATH or pass the checkpoint path as the first argument." >&2
  exit 1
fi

cd "${SEMANTICVLA_ROOT}"
export PYTHONPATH="${SEMANTICVLA_ROOT}:${PYTHONPATH:-}"

echo "Starting LIBERO policy server on port ${SERVER_PORT}"
exec "${SEMANTICVLA_PYTHON}" -m deployment.model_server.server_policy --ckpt_path "${MODEL_PATH}" --port "${SERVER_PORT}" --use_bf16
