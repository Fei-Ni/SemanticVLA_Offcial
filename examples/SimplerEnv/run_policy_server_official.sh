#!/usr/bin/env bash
set -euo pipefail

SEMANTICVLA_OFFICIAL_ROOT="${SEMANTICVLA_OFFICIAL_ROOT:-${REPO_ROOT}/semanticvla.official}"
SEMANTICVLA_PYTHON="${SEMANTICVLA_PYTHON:-python}"

MODEL_PATH="${1:?Usage: $0 <checkpoint_path> [port]}"
SERVER_PORT="${2:-${SERVER_PORT:-${PORT:-${BASE_PORT:-10093}}}}"

if [ ! -d "${SEMANTICVLA_OFFICIAL_ROOT}" ]; then
  echo "SEMANTICVLA_OFFICIAL_ROOT not found: ${SEMANTICVLA_OFFICIAL_ROOT}" >&2
  exit 1
fi
if [ ! -x "${SEMANTICVLA_PYTHON}" ]; then
  echo "SEMANTICVLA_PYTHON not found: ${SEMANTICVLA_PYTHON}" >&2
  exit 1
fi
if [ ! -f "${MODEL_PATH}" ]; then
  echo "Checkpoint not found: ${MODEL_PATH}" >&2
  exit 1
fi

cd "${SEMANTICVLA_OFFICIAL_ROOT}"
export PYTHONPATH="${SEMANTICVLA_OFFICIAL_ROOT}:${PYTHONPATH:-}"
exec "${SEMANTICVLA_PYTHON}" deployment/model_server/server_policy.py \
  --ckpt_path "${MODEL_PATH}" \
  --port "${SERVER_PORT}" \
  --use_bf16
