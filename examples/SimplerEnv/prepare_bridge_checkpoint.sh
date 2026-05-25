#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEMANTICVLA_ROOT="${SEMANTICVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PROJECTS_ROOT="${PROJECTS_ROOT:-${DATA_ROOT}}"
MODELS_ROOT="${MODELS_ROOT:-${HOME}/models}"
DEFAULT_MODELS_ROOT="${PROJECTS_ROOT}/models"
if [[ ! -d "${PROJECTS_ROOT}/models" ]]; then
  DEFAULT_MODELS_ROOT="${MODELS_ROOT}"
fi
TARGET_DIR="${TARGET_DIR:-${DEFAULT_MODELS_ROOT}/Qwen3VL-GR00T-Bridge-RT-1}"
DEFAULT_HF_HOME="${PROJECTS_ROOT}/.cache/huggingface"
if [[ ! -d "${PROJECTS_ROOT}/.cache/huggingface" ]]; then
  DEFAULT_HF_HOME="${HOME}/.cache/huggingface"
fi
HF_HOME="${HF_HOME:-${DEFAULT_HF_HOME}}"
SEMANTICVLA_ENV_ROOT="${SEMANTICVLA_ENV_ROOT:-${HOME}/tools/miniforge3/envs/semanticvla}"
HF_CLI="${HF_CLI:-${SEMANTICVLA_ENV_ROOT}/bin/huggingface-cli}"
REPO_ID="${REPO_ID:-StarVLA/Qwen3VL-GR00T-Bridge-RT-1}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-5}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-120}"

if [[ ! -x "${HF_CLI}" ]]; then
  echo "Missing huggingface-cli at ${HF_CLI}" >&2
  exit 1
fi

mkdir -p "${TARGET_DIR}"
mkdir -p "${HF_HOME}"

export HF_HOME
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-0}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "HF token is not set. Public download may still work, but Hugging Face may rate-limit anonymous access." >&2
fi

echo "Preparing Bridge checkpoint under ${TARGET_DIR}"
echo "REPO_ID=${REPO_ID}"
echo "HF_HOME=${HF_HOME}"
echo "HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET}"
echo "DOWNLOAD_RETRIES=${DOWNLOAD_RETRIES}"

attempt=1
while true; do
  if "${HF_CLI}" download \
    --repo-type model \
    --resume-download \
    "${REPO_ID}" \
    --local-dir "${TARGET_DIR}"; then
    break
  fi

  if [[ "${attempt}" -ge "${DOWNLOAD_RETRIES}" ]]; then
    echo "Failed to download ${REPO_ID} after ${attempt} attempts." >&2
    exit 1
  fi

  echo "Download attempt ${attempt} for ${REPO_ID} failed; retrying in ${RETRY_SLEEP_SECONDS}s..." >&2
  attempt="$((attempt + 1))"
  sleep "${RETRY_SLEEP_SECONDS}"
done
