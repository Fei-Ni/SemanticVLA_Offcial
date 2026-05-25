#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEMANTICVLA_ROOT="${SEMANTICVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

SEMANTIC_RUN_SCRIPT="${SEMANTIC_RUN_SCRIPT:-${SEMANTICVLA_ROOT}/examples/LIBERO/semanticvla/run_semanticvla_train.sh}"
PROJECTS_ROOT="${BRIDGE_PROJECTS_ROOT:-${DATA_ROOT}}"
RESULTS_ROOT="${BRIDGE_RESULTS_ROOT:-${PROJECTS_ROOT}/migration_validation_20260512}"
MODELS_ROOT="${MODELS_ROOT:-${PROJECTS_ROOT}/models}"

CONFIG_YAML="${CONFIG_YAML:-${SEMANTICVLA_ROOT}/semanticvla/config/training/semanticvla_oxe_bridge.yaml}"
BASE_VLM="${BASE_VLM:-${MODELS_ROOT}/Qwen/Qwen3-VL-4B-Instruct}"
VLA_DATA_ROOT="${VLA_DATA_ROOT:-${PROJECTS_ROOT}/datasets/OXE_LEROBOT_DATASET}"
TRACE_ROOT="${TRACE_ROOT:-${RESULTS_ROOT}/trace_npy_index}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${RESULTS_ROOT}/vla_train}"
DATA_MIX="${DATA_MIX:-bridge}"

LAM_VARIANT="${LAM_VARIANT:-v4}"
LATENT_LABELS_ROOT="${LATENT_LABELS_ROOT:-${RESULTS_ROOT}/oxe_lam_labels}"
LATENT_LABELS_VARIANT="${LATENT_LABELS_VARIANT:-}"
LATENT_LABELS_ENABLED="${LATENT_LABELS_ENABLED:-true}"
LATENT_LABELS_STRICT="${LATENT_LABELS_STRICT:-true}"
LATENT_LABELS_MISSING_POLICY="${LATENT_LABELS_MISSING_POLICY:-clip}"

MODE="${MODE:-smoke}"   # smoke | quick | full
case "${MODE}" in
  smoke)
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-500}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
    EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
    LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-10}"
    ;;
  quick)
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-5000}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-2500}"
    EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
    LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-50}"
    ;;
  full)
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-100000}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
    EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
    LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-10}"
    ;;
  *)
    echo "[ERROR] MODE must be smoke|quick|full, got ${MODE}" >&2
    exit 1
    ;;
esac

case "${DATA_MIX}" in
  bridge) ;;
  *)
    echo "[ERROR] Bridge SemanticVLA wrapper currently expects DATA_MIX=bridge, got ${DATA_MIX}" >&2
    echo "This avoids trace/latent-label holes for non-Bridge OXE datasets." >&2
    exit 1
    ;;
esac

case "${LAM_VARIANT}" in
  paper_strict)
    SEMANTIC_LATENT_NUM_TOKENS="${SEMANTIC_LATENT_NUM_TOKENS:-4}"
    SEMANTIC_LATENT_VOCAB_SIZE="${SEMANTIC_LATENT_VOCAB_SIZE:-32}"
    ;;
  *)
    echo "[ERROR] Unknown LAM_VARIANT='${LAM_VARIANT}'. Only 'paper_strict' is supported." >&2
    exit 1
    ;;
esac

INJECTION_MODE="${INJECTION_MODE:-none}"
case "${INJECTION_MODE}" in
  sa_embs|adaln|both|none) ;;
  *) echo "[ERROR] INJECTION_MODE must be sa_embs|adaln|both|none; got ${INJECTION_MODE}" >&2; exit 1 ;;
esac

if [[ -z "${SEMANTIC_PARSE_TRACE_FOR_DECODER:-}" ]]; then
  if [[ "${INJECTION_MODE}" == "none" ]]; then
    SEMANTIC_PARSE_TRACE_FOR_DECODER="false"
  else
    SEMANTIC_PARSE_TRACE_FOR_DECODER="true"
  fi
fi

SEMANTIC_OUTPUT_ENABLED="${SEMANTIC_OUTPUT_ENABLED:-true}"
SEMANTIC_OUTPUT_MODE="${SEMANTIC_OUTPUT_MODE:-trace_latent}"
SEMANTIC_OUTPUT_ORDER="${SEMANTIC_OUTPUT_ORDER:-trace_latent}"
SEMANTIC_LM_LOSS_WEIGHT="${SEMANTIC_LM_LOSS_WEIGHT:-0.1}"
LM_AUX_LOSS="${LM_AUX_LOSS:-false}"
PROMPT_STYLE="${PROMPT_STYLE:-plain}"
NUM_ANCHOR_POINTS="${NUM_ANCHOR_POINTS:-4}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE_BASE="${LEARNING_RATE_BASE:-4e-5}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
RUN_ID="${RUN_ID:-semanticvla_oxe_bridge_${LAM_VARIANT}_${INJECTION_MODE}_lw${SEMANTIC_LM_LOSS_WEIGHT}_${MODE}_$(date -u +%Y%m%d_%H%M%S)}"
WANDB_GROUP="${WANDB_GROUP:-semanticvla-oxe-bridge-${MODE}}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${PROJECTS_ROOT}/models/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt}"
RELOAD_STRICT="${RELOAD_STRICT:-false}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
RESUME_STEP="${RESUME_STEP:-}"
RESUME_RELOAD_STRICT="${RESUME_RELOAD_STRICT:-true}"
RESUME_LATEST_FROM_RUN_ID="${RESUME_LATEST_FROM_RUN_ID:-}"

SEMANTICVLA_ENV_ROOT="${SEMANTICVLA_ENV_ROOT:-${CONDA_ENV}}"
SEMANTICVLA_PYTHON="${SEMANTICVLA_PYTHON:-${SEMANTICVLA_ENV_ROOT}/bin/python}"

[[ -f "${SEMANTIC_RUN_SCRIPT}" ]] || { echo "[ERROR] missing run script: ${SEMANTIC_RUN_SCRIPT}" >&2; exit 1; }
[[ -f "${CONFIG_YAML}" ]] || { echo "[ERROR] missing config yaml: ${CONFIG_YAML}" >&2; exit 1; }
[[ -d "${BASE_VLM}" ]] || { echo "[ERROR] missing BASE_VLM: ${BASE_VLM}" >&2; exit 1; }
[[ -d "${VLA_DATA_ROOT}/bridge_orig_1.0.0_lerobot" ]] || { echo "[ERROR] missing Bridge LeRobot dataset under ${VLA_DATA_ROOT}" >&2; exit 1; }
[[ -d "${TRACE_ROOT}" ]] || { echo "[ERROR] missing TRACE_ROOT: ${TRACE_ROOT}" >&2; exit 1; }
[[ -f "${PRETRAINED_CHECKPOINT}" ]] || { echo "[ERROR] missing PRETRAINED_CHECKPOINT: ${PRETRAINED_CHECKPOINT}" >&2; exit 1; }
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  [[ -f "${RESUME_FROM_CHECKPOINT}" ]] || { echo "[ERROR] missing RESUME_FROM_CHECKPOINT: ${RESUME_FROM_CHECKPOINT}" >&2; exit 1; }
fi
[[ -x "${SEMANTICVLA_PYTHON}" ]] || { echo "[ERROR] missing SEMANTICVLA_PYTHON: ${SEMANTICVLA_PYTHON}" >&2; exit 1; }

if [[ "${LATENT_LABELS_ENABLED}" == "true" ]]; then
  [[ -d "${LATENT_LABELS_ROOT}" ]] || { echo "[ERROR] missing LATENT_LABELS_ROOT: ${LATENT_LABELS_ROOT}" >&2; exit 1; }
  if [[ -n "${LATENT_LABELS_VARIANT}" && ! -d "${LATENT_LABELS_ROOT}/${LATENT_LABELS_VARIANT}" ]]; then
    echo "[ERROR] missing latent label variant dir: ${LATENT_LABELS_ROOT}/${LATENT_LABELS_VARIANT}" >&2
    exit 1
  fi
fi

if [[ -n "${RESUME_LATEST_FROM_RUN_ID}" && -z "${RESUME_FROM_CHECKPOINT}" ]]; then
  RESUME_SOURCE_DIR="${RUN_ROOT_DIR}/${RESUME_LATEST_FROM_RUN_ID}/checkpoints"
  TARGET_CHECKPOINT="${RUN_ROOT_DIR}/${RUN_ID}/checkpoints/steps_${MAX_TRAIN_STEPS}_pytorch_model.pt"
  if [[ -f "${TARGET_CHECKPOINT}" ]]; then
    echo "[INFO] target checkpoint already exists; nothing to resume: ${TARGET_CHECKPOINT}"
    exit 0
  fi
  [[ -d "${RESUME_SOURCE_DIR}" ]] || { echo "[ERROR] missing resume source checkpoint dir: ${RESUME_SOURCE_DIR}" >&2; exit 1; }
  RESUME_FROM_CHECKPOINT="$("${SEMANTICVLA_PYTHON}" - "${RESUME_SOURCE_DIR}" <<'PY'
import pathlib
import re
import sys

checkpoint_dir = pathlib.Path(sys.argv[1])
best = None
for path in checkpoint_dir.glob("steps_*_pytorch_model.pt"):
    match = re.search(r"steps_(\d+)_pytorch_model\.pt$", path.name)
    if not match:
        continue
    step = int(match.group(1))
    if best is None or step > best[0]:
        best = (step, path)
if best is None:
    raise SystemExit(f"no steps_*_pytorch_model.pt found in {checkpoint_dir}")
print(best[1])
PY
)"
  RESUME_STEP="$(basename "${RESUME_FROM_CHECKPOINT}" | sed -E 's/^steps_([0-9]+)_pytorch_model\.pt$/\1/')"
fi

export SEMANTICVLA_ROOT
export PROJECTS_ROOT
export RESULTS_ROOT
export CONFIG_YAML
export BASE_VLM
export VLA_DATA_ROOT
export TRACE_ROOT
export RUN_ROOT_DIR
export DATA_MIX
export RUN_ID
export WANDB_GROUP
export INJECTION_MODE
export PROMPT_STYLE
export NUM_ANCHOR_POINTS
export LM_AUX_LOSS
export SEMANTIC_OUTPUT_ENABLED
export SEMANTIC_OUTPUT_MODE
export SEMANTIC_OUTPUT_ORDER
export SEMANTIC_LM_LOSS_WEIGHT
export SEMANTIC_LATENT_VOCAB_SIZE
export SEMANTIC_LATENT_NUM_TOKENS
export SEMANTIC_PARSE_TRACE_FOR_DECODER
export LATENT_LABELS_ENABLED
export LATENT_LABELS_ROOT
export LATENT_LABELS_VARIANT
export LATENT_LABELS_STRICT
export LATENT_LABELS_MISSING_POLICY
export PRETRAINED_CHECKPOINT
export RELOAD_STRICT
export RESUME_FROM_CHECKPOINT
export RESUME_STEP
export RESUME_RELOAD_STRICT
export RESUME_LATEST_FROM_RUN_ID
export MAX_TRAIN_STEPS
export SAVE_INTERVAL
export EVAL_INTERVAL
export LOGGING_FREQUENCY
export PER_DEVICE_BATCH_SIZE
export GRADIENT_ACCUMULATION_STEPS
export LEARNING_RATE_BASE
export NUM_PROCESSES
export SEMANTICVLA_ENV_ROOT
export SEMANTICVLA_PYTHON

echo "==== Bridge SemanticVLA wrapper ===="
echo "MODE=${MODE}"
echo "RUN_ID=${RUN_ID}"
echo "CONFIG_YAML=${CONFIG_YAML}"
echo "VLA_DATA_ROOT=${VLA_DATA_ROOT}"
echo "TRACE_ROOT=${TRACE_ROOT}"
echo "LATENT_LABELS_ROOT=${LATENT_LABELS_ROOT}"
echo "LATENT_LABELS_VARIANT=${LATENT_LABELS_VARIANT}"
echo "LATENT_LABELS_MISSING_POLICY=${LATENT_LABELS_MISSING_POLICY}"
echo "LAM_VARIANT=${LAM_VARIANT}"
echo "INJECTION_MODE=${INJECTION_MODE}"
echo "SEMANTIC_OUTPUT_MODE=${SEMANTIC_OUTPUT_MODE}"
echo "SEMANTIC_LM_LOSS_WEIGHT=${SEMANTIC_LM_LOSS_WEIGHT}"
echo "SEMANTIC_LATENT_NUM_TOKENS=${SEMANTIC_LATENT_NUM_TOKENS}"
echo "PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT}"
echo "RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT}"
echo "RESUME_STEP=${RESUME_STEP}"
echo "RESUME_LATEST_FROM_RUN_ID=${RESUME_LATEST_FROM_RUN_ID}"
echo "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
echo "===================================="

cd "${SEMANTICVLA_ROOT}"
bash "${SEMANTIC_RUN_SCRIPT}" "$@"
