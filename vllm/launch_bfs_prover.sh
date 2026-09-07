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

if [[ "${CONDA_DEFAULT_ENV:-}" == "dsp" ]]; then
  echo "Warning: active conda env is 'dsp'. Prefer a separate 'bfs-vllm' env for vLLM." >&2
fi

python - <<'PY'
from __future__ import annotations

import importlib.metadata as metadata
import re
import sys


def version_tuple(value: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", value.split("+", 1)[0])
    return tuple(int(part) for part in parts[:3])


try:
    vllm_version = metadata.version("vllm")
except metadata.PackageNotFoundError:
    print("vLLM is not installed in this environment.", file=sys.stderr)
    print("Install it with: pip install -r vllm/requirements.txt", file=sys.stderr)
    raise SystemExit(1)

try:
    transformers_version = metadata.version("transformers")
except metadata.PackageNotFoundError:
    transformers_version = ""

if vllm_version.startswith("0.9.") and transformers_version:
    if version_tuple(transformers_version) >= (4, 54, 0):
        print("Incompatible vLLM/Transformers versions detected:", file=sys.stderr)
        print(f"  vllm={vllm_version}", file=sys.stderr)
        print(f"  transformers={transformers_version}", file=sys.stderr)
        print("vLLM 0.9.x requires transformers < 4.54.0.", file=sys.stderr)
        print("Recommended fix:", file=sys.stderr)
        print("  conda create -n bfs-vllm python=3.10 -y", file=sys.stderr)
        print("  conda activate bfs-vllm", file=sys.stderr)
        print("  cd /data/not_backed_up/yxu209/DSP-Plus", file=sys.stderr)
        print("  pip install -r vllm/requirements.txt", file=sys.stderr)
        raise SystemExit(1)
PY

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

