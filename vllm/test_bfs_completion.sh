#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:30001/v1}"

curl -fsS "$BASE_URL/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bytedance-research/BFS-Prover",
    "prompt": "n : Nat\n⊢ Nat.gcd n n = n:::",
    "max_tokens": 64,
    "temperature": 1.1,
    "top_p": 1.0,
    "n": 2,
    "logprobs": 1,
    "use_beam_search": false
  }'
printf "\n"

