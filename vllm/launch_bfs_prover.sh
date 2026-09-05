#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/not_backed_up/yxu209/DSP-Plus}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/vllm/models/BFS-Prover-V1-7B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30001}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-bytedance-research/BFS-Prover}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Missing local model directory: $MODEL_PATH" >&2
  echo "Place or symlink ByteDance-Seed/BFS-Prover-V1-7B there before launching vLLM." >&2
  exit 1
fi

if [[ -z "$(find "$MODEL_PATH" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Model directory is empty: $MODEL_PATH" >&2
  echo "Place or symlink ByteDance-Seed/BFS-Prover-V1-7B there before launching vLLM." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec vllm serve "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --generation-config vllm

