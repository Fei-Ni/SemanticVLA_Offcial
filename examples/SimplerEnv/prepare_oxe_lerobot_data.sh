#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEMANTICVLA_ROOT="${SEMANTICVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PROJECTS_ROOT="${PROJECTS_ROOT:-${DATA_ROOT}}"
DATASETS_ROOT="${DATASETS_ROOT:-${HOME}/datasets}"
DEFAULT_DATA_ROOT="${PROJECTS_ROOT}/datasets/OXE_LEROBOT_DATASET"
if [[ ! -d "${PROJECTS_ROOT}/datasets" ]]; then
  DEFAULT_DATA_ROOT="${DATASETS_ROOT}/OXE_LEROBOT_DATASET"
fi
DATA_ROOT="${DATA_ROOT:-${VLA_DATA_ROOT:-${DEFAULT_DATA_ROOT}}}"
DEFAULT_HF_HOME="${PROJECTS_ROOT}/.cache/huggingface"
if [[ ! -d "${PROJECTS_ROOT}/.cache/huggingface" ]]; then
  DEFAULT_HF_HOME="${HOME}/.cache/huggingface"
fi
HF_HOME="${HF_HOME:-${DEFAULT_HF_HOME}}"
SEMANTICVLA_ENV_ROOT="${SEMANTICVLA_ENV_ROOT:-${HOME}/tools/miniforge3/envs/semanticvla}"
HF_CLI="${HF_CLI:-${SEMANTICVLA_ENV_ROOT}/bin/huggingface-cli}"
PYTHON_BIN="${PYTHON_BIN:-${SEMANTICVLA_ENV_ROOT}/bin/python}"
GENERATE_SCRIPT="${SEMANTICVLA_ROOT}/examples/SimplerEnv/generate_oxe_modality_json.py"
VALIDATE_SCRIPT="${SEMANTICVLA_ROOT}/examples/SimplerEnv/validate_oxe_lerobot_data.py"
DATA_MIX="${DATA_MIX:-bridge_rt_1}"
MAX_WORKERS="${MAX_WORKERS:-1}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-20}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-60}"
SAMPLE_EPISODES="${SAMPLE_EPISODES:-8}"

if [[ ! -x "${HF_CLI}" ]]; then
  echo "Missing huggingface-cli at ${HF_CLI}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing python interpreter at ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${GENERATE_SCRIPT}" ]]; then
  echo "Missing modality generation script: ${GENERATE_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${VALIDATE_SCRIPT}" ]]; then
  echo "Missing dataset validation script: ${VALIDATE_SCRIPT}" >&2
  exit 1
fi

mkdir -p "${DATA_ROOT}"
mkdir -p "${HF_HOME}"

export HF_HOME
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

declare -a DATASETS=()
case "${DATA_MIX}" in
  bridge)
    DATASETS=("bridge_orig_1.0.0_lerobot")
    ;;
  rt1|fractal)
    DATASETS=("fractal20220817_data_0.1.0_lerobot")
    ;;
  bridge_rt_1)
    DATASETS=("bridge_orig_1.0.0_lerobot" "fractal20220817_data_0.1.0_lerobot")
    ;;
  *)
    echo "Unsupported DATA_MIX=${DATA_MIX}" >&2
    echo "Use DATA_MIX=bridge, DATA_MIX=rt1, DATA_MIX=fractal, or DATA_MIX=bridge_rt_1." >&2
    exit 1
    ;;
esac

repo_id_for_dataset() {
  local dataset_name="$1"
  case "${dataset_name}" in
    bridge_orig_1.0.0_lerobot)
      printf 'IPEC-COMMUNITY/bridge_orig_lerobot\n'
      ;;
    fractal20220817_data_0.1.0_lerobot)
      printf 'IPEC-COMMUNITY/fractal20220817_data_lerobot\n'
      ;;
    *)
      echo "Unsupported dataset_name=${dataset_name}" >&2
      return 1
      ;;
  esac
}

echo "Preparing OXE LeRobot datasets under ${DATA_ROOT}"
echo "DATA_MIX=${DATA_MIX}"
echo "HF_HOME=${HF_HOME}"
echo "HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET}"
echo "HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER}"
echo "MAX_WORKERS=${MAX_WORKERS}"
echo "DOWNLOAD_RETRIES=${DOWNLOAD_RETRIES}"

for dataset_name in "${DATASETS[@]}"; do
  repo_id="$(repo_id_for_dataset "${dataset_name}")"
  local_dir="${DATA_ROOT}/${dataset_name}"

  echo
  echo "==== Downloading ${repo_id} -> ${local_dir} ===="
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
  --data-root "${DATA_ROOT}" \
  --datasets "${DATASETS[@]}"

echo
echo "==== Validating downloaded datasets ===="
"${PYTHON_BIN}" "${VALIDATE_SCRIPT}" \
  --data-root "${DATA_ROOT}" \
  --sample-episodes "${SAMPLE_EPISODES}" \
  --datasets "${DATASETS[@]}"
