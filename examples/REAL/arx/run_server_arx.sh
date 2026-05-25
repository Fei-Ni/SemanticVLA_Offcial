#!/bin/bash

set -euo pipefail

###########################################################################################
# === Please modify the following paths according to your environment ===
ckpt_path=/data/yxz/semanticvla/outputs/gravity_single4il_qwen3gr00t_10k/checkpoints/steps_10000_pytorch_model.pt
gpu_id=5
port=10093
num_inference_timesteps_override=4
camera_keys="${CAMERA_KEYS:-}"
# === End of environment variable configuration ===
###########################################################################################

extra_args=()
if [[ -n "${camera_keys}" ]]; then
  extra_args+=(--camera_keys "${camera_keys}")
fi

CUDA_VISIBLE_DEVICES=${gpu_id} /data/yxz/conda/envs/semanticvla/bin/python \
  deployment/model_server/arx/server_policy_arx.py \
  --ckpt_path "${ckpt_path}" \
  --port "${port}" \
  --use_bf16 \
  --num_inference_timesteps_override "${num_inference_timesteps_override}" \
  "${extra_args[@]}"
