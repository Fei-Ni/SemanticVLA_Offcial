#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEMANTICVLA_ROOT="${SEMANTICVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DATASETS_ROOT="${DATASETS_ROOT:-${HOME}/datasets}"
DATA_ROOT="${DATA_ROOT:-${LIBERO_DATA_ROOT:-${DATASETS_ROOT}/LEROBOT_LIBERO_DATA}}"
HF_CLI="${HF_CLI:-$(command -v huggingface-cli || true)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
GENERATE_SCRIPT="${SEMANTICVLA_ROOT}/examples/LIBERO/generate_libero_modality_json.py"
VALIDATE_SCRIPT="${SEMANTICVLA_ROOT}/examples/LIBERO/validate_libero_lerobot_data.py"
MAX_WORKERS="${MAX_WORKERS:-1}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-20}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-60}"

DATASETS=(
  "libero_object_no_noops_1.0.0_lerobot"
  "libero_goal_no_noops_1.0.0_lerobot"
  "libero_spatial_no_noops_1.0.0_lerobot"
  "libero_10_no_noops_1.0.0_lerobot"
)

if [[ ! -x "${HF_CLI}" ]]; then
  echo "Missing huggingface-cli at ${HF_CLI}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing python interpreter at ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${GENERATE_SCRIPT}" ]]; then
  echo "Missing generate script: ${GENERATE_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${VALIDATE_SCRIPT}" ]]; then
  echo "Missing validate script: ${VALIDATE_SCRIPT}" >&2
  exit 1
fi

mkdir -p "${DATA_ROOT}"

export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

echo "Preparing LIBERO LeRobot datasets under ${DATA_ROOT}"
echo "HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET}"
echo "HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER}"
echo "MAX_WORKERS=${MAX_WORKERS}"
echo "DOWNLOAD_RETRIES=${DOWNLOAD_RETRIES}"

for dataset_name in "${DATASETS[@]}"; do
  repo_id="IPEC-COMMUNITY/${dataset_name}"
  local_dir="${DATA_ROOT}/${dataset_name}"

  echo
  echo "==== Downloading ${repo_id} ===="
  attempt=1
  while true; do
    if "${HF_CLI}" download \
      --repo-type dataset \
      --resume-download \
      --max-workers "${MAX_WORKERS}" \
      "${repo_id}" \
      --local-dir "${local_dir}"; then
      break
    fi

    if [[ "${attempt}" -ge "${DOWNLOAD_RETRIES}" ]]; then
      echo "Failed to download ${repo_id} after ${attempt} attempts." >&2
      exit 1
    fi

    echo "Download attempt ${attempt} for ${repo_id} failed; retrying in ${RETRY_SLEEP_SECONDS}s..." >&2
    attempt="$((attempt + 1))"
    sleep "${RETRY_SLEEP_SECONDS}"
  done
done

echo
echo "==== Generating SemanticVLA modality metadata ===="
"${PYTHON_BIN}" "${GENERATE_SCRIPT}" \
  --data-root "${DATA_ROOT}"

echo
echo "==== Validating downloaded datasets ===="
"${PYTHON_BIN}" "${VALIDATE_SCRIPT}" \
  --data-root "${DATA_ROOT}"
