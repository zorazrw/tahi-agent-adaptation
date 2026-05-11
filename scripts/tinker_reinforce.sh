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

python tinker_reinforce.py \
  --train-path reinforce_claude45_task1_step10.json \
  --model-name Qwen/Qwen3.5-4B \
  --renderer-name qwen3_5 \
  --log-path ./logs/reinforce_qwen3.5-4b_claude45-task1-step10 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --reward-alpha 0.05 \
  --epochs 100