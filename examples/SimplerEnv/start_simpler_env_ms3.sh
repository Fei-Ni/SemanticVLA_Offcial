#!/usr/bin/env bash
set -euo pipefail

SEMANTICVLA_ROOT="${SEMANTICVLA_ROOT:-${HOME}/semanticvla}"
SEMANTICVLA_PYTHON="${SEMANTICVLA_PYTHON:-${HOME}/tools/miniforge3/envs/semanticvla/bin/python}"
PROJECTS_ROOT="${PROJECTS_ROOT:-${DATA_ROOT}}"
MODELS_ROOT="${MODELS_ROOT:-${HOME}/models}"
BASE_VLM_DEFAULT="${PROJECTS_ROOT}/models/Qwen/Qwen3-VL-4B-Instruct"
if [ -d "${MODELS_ROOT}/Qwen/Qwen3-VL-4B-Instruct" ]; then
  BASE_VLM_DEFAULT="${MODELS_ROOT}/Qwen/Qwen3-VL-4B-Instruct"
fi
BASE_VLM="${BASE_VLM:-${BASE_VLM_DEFAULT}}"
SIMPLERENV_PATH="${SIMPLERENV_PATH:-${HOME}/SimplerEnv_maniskill3}"
MANISKILL_PATH="${MANISKILL_PATH:-${HOME}/ManiSkill_aarch64}"
SAPIEN_BUILD_ROOT_DEFAULT="${HOME}/SAPIEN_aarch64/docker_sapien_build"
SAPIEN_STAGE_BUILD="/projects/public/u6cu/migration_to_u6gs_20260512/home/SAPIEN_aarch64/docker_sapien_build"
if [ ! -d "${SAPIEN_BUILD_ROOT_DEFAULT}" ] && [ -d "${SAPIEN_STAGE_BUILD}" ]; then
  SAPIEN_BUILD_ROOT_DEFAULT="${SAPIEN_STAGE_BUILD}"
fi
SAPIEN_BUILD_ROOT="${SAPIEN_BUILD_ROOT:-${SAPIEN_BUILD_ROOT_DEFAULT}}"
SIMPLER_RUNTIME_ROOT="${SIMPLER_RUNTIME_ROOT:-${PROJECTS_ROOT}/migration_validation_20260423/simpler_env_runtime}"
SIMPLER_OVERLAY_DEFAULT="${HOME}/results/simpler_env_runtime/site-packages"
if [ -d "${SIMPLER_RUNTIME_ROOT}/site-packages_known_good" ]; then
  SIMPLER_OVERLAY_DEFAULT="${SIMPLER_RUNTIME_ROOT}/site-packages_known_good"
elif [ -d "${SIMPLER_RUNTIME_ROOT}/site-packages" ]; then
  SIMPLER_OVERLAY_DEFAULT="${SIMPLER_RUNTIME_ROOT}/site-packages"
fi
SIMPLER_OVERLAY="${SIMPLER_OVERLAY:-${SIMPLER_OVERLAY_DEFAULT}}"
SIM_PYTHON="${SIM_PYTHON:-${HOME}/tools/miniforge3/envs/semanticvla/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-${HOME}/results}"
SIMPLER_RESULTS_ROOT="${SIMPLER_RESULTS_ROOT:-${RESULTS_ROOT}/simpler_env_eval_ms3}"
MS_ASSET_DIR="${MS_ASSET_DIR:-${HOME}/datasets/mani_skill}"
PROBE_SERVER_SCRIPT="${PROBE_SERVER_SCRIPT:-${SEMANTICVLA_ROOT}/examples/LIBERO/probe/run_probe_server.sh}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-120}"
POLICY_SERVER_PROTOCOL="${POLICY_SERVER_PROTOCOL:-legacy}"

if [ $# -lt 1 ]; then
  echo "Usage: $0 <checkpoint_path> [task_name] [num_episodes] [port]" >&2
  exit 1
fi

ckpt_path="$1"
task_name="${2:-widowx_put_eggplant_in_basket}"
num_episodes="${3:-5}"
if [ $# -ge 4 ]; then
  port="$4"
else
  port="$("${SEMANTICVLA_PYTHON}" -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')"
fi

cleanup() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    unset SERVER_PID
  fi
}

trap cleanup EXIT

if [ ! -x "${SEMANTICVLA_PYTHON}" ]; then
  echo "SemanticVLA python not found: ${SEMANTICVLA_PYTHON}" >&2
  exit 1
fi

if [ ! -x "${SIM_PYTHON}" ]; then
  echo "Sim python not found: ${SIM_PYTHON}" >&2
  exit 1
fi

if [ ! -d "${SIMPLERENV_PATH}" ]; then
  echo "SIMPLERENV_PATH not found: ${SIMPLERENV_PATH}" >&2
  exit 1
fi

if [ ! -d "${MANISKILL_PATH}" ]; then
  echo "MANISKILL_PATH not found: ${MANISKILL_PATH}" >&2
  exit 1
fi

if [ ! -d "${SAPIEN_BUILD_ROOT}" ]; then
  echo "SAPIEN_BUILD_ROOT not found: ${SAPIEN_BUILD_ROOT}" >&2
  exit 1
fi

if [ ! -d "${SIMPLER_OVERLAY}" ]; then
  echo "SIMPLER_OVERLAY not found: ${SIMPLER_OVERLAY}" >&2
  exit 1
fi

if [ ! -d "${MS_ASSET_DIR}" ]; then
  echo "MS_ASSET_DIR not found: ${MS_ASSET_DIR}" >&2
  exit 1
fi

if [ ! -d "${BASE_VLM}" ]; then
  echo "BASE_VLM not found: ${BASE_VLM}" >&2
  exit 1
fi

if [ ! -f "${PROBE_SERVER_SCRIPT}" ]; then
  echo "PROBE_SERVER_SCRIPT not found: ${PROBE_SERVER_SCRIPT}" >&2
  exit 1
fi

ckpt_run="$(basename "$(dirname "$(dirname "${ckpt_path}")")")"
ckpt_name="$(basename "${ckpt_path}" .pt)"
timestamp="${SIMPLER_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
run_root="${SIMPLER_RESULTS_ROOT}/${ckpt_run}/${ckpt_name}/${task_name}/${timestamp}"
mkdir -p "${run_root}"
server_log="${run_root}/server.log"

export DISPLAY=""
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.aarch64.json}"
export MS_ASSET_DIR
export BASE_VLM
export PYTHONPATH="${SIMPLER_OVERLAY}:${SAPIEN_BUILD_ROOT}/lib.linux-aarch64-cpython-310:${MANISKILL_PATH}:${SIMPLERENV_PATH}:${SEMANTICVLA_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${SAPIEN_BUILD_ROOT}/_sapien_install/lib64:/usr/lib64:${LD_LIBRARY_PATH:-}"

cd "${SEMANTICVLA_ROOT}"
if [ "${SIMPLER_SKIP_SERVER:-0}" != "1" ]; then
  bash "${PROBE_SERVER_SCRIPT}" "${ckpt_path}" "${port}" >"${server_log}" 2>&1 &
  SERVER_PID=$!

  ready=0
  for _ in $(seq 1 72); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "Probe server exited before ready." >&2
      tail -n 80 "${server_log}" >&2 || true
      exit 1
    fi
    if grep -q "server running" "${server_log}" 2>/dev/null; then
      sleep 2
      ready=1
      break
    fi
    sleep 5
  done

  if [ "${ready}" != "1" ]; then
    echo "Probe server did not become ready." >&2
    tail -n 80 "${server_log}" >&2 || true
    exit 1
  fi
fi

"${SIM_PYTHON}" examples/SimplerEnv/start_simpler_env_ms3.py \
  --ckpt-path "${ckpt_path}" \
  --task "${task_name}" \
  --num-episodes "${num_episodes}" \
  --host "127.0.0.1" \
  --port "${port}" \
  --logging-dir "${run_root}" \
  --save-tag "${timestamp}" \
  --max-episode-steps "${MAX_EPISODE_STEPS}" \
  --policy-server-protocol "${POLICY_SERVER_PROTOCOL}"

echo "SUMMARY_JSON=${run_root}/summary.json"
