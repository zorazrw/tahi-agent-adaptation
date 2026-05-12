#!/usr/bin/env bash
# Load TINKER_API_KEY (and any other vars) from .env next to this script.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$ROOT/.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
else
  echo "Missing $ENV_FILE — create it with TINKER_API_KEY=..." >&2
  exit 1
fi

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "TINKER_API_KEY is empty after sourcing .env" >&2
  exit 1
fi

python tinker_dpo.py \
  --train-path dpo.json \
  --model-name Qwen/Qwen3-4B-Instruct-2507 \
  --renderer-name qwen3_instruct \
  --log-path ./logs/tinker_dpo \
  --batch-size 1 \
  --learning-rate 1e-5 \
  --dpo-beta 0.1