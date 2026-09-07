#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:30001/v1}"

curl -fsS "$BASE_URL/models"
printf "\n"

