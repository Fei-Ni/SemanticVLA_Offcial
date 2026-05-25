#!/bin/bash

set -euo pipefail

SEMANTICVLA_ROOT="${SEMANTICVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODELS_ROOT="${MODELS_ROOT:-${HOME}/models}"
SEMANTICVLA_PYTHON="${SEMANTICVLA_PYTHON:-${HOME}/tools/miniforge3/envs/semanticvla/bin/python}"
MODEL_PATH="${1:-${MODEL_PATH:-${MODELS_ROOT}/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt}}"
PORT="${PORT:-5678}"
GPU_ID="${GPU_ID:-0}"

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

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${SEMANTICVLA_PYTHON}" -m deployment.model_server.server_policy --ckpt_path "${MODEL_PATH}" --port "${PORT}" --use_bf16
