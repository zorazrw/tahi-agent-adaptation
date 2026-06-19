#!/usr/bin/env bash
# Upload session JSON files in scripts/sessions/ to the local proxy server.
#
# Defaults:
#   - uploads every session in lexicographic order
#   - waits for a model update after each upload
#   - prints per-session wall-clock training time
#
# Environment:
#   AGENT_COWORK_PROXY_URL=http://localhost:8000
#   HOLDOUT_LAST=1          # hold back the last session as test data
#   WAIT_FOR_TRAINING=0     # enqueue all sessions without waiting/timing
#   TRAINING_TIMEOUT_SECONDS=7200
#   TRAINING_WINDOW_SESSIONS=2
#   TRAINING_WINDOW_MIN_SESSIONS=2  # first min_sessions-1 uploads are warmup only
set -euo pipefail

PROXY_URL="${AGENT_COWORK_PROXY_URL:-http://localhost:8000}"
HOLDOUT_LAST="${HOLDOUT_LAST:-0}"
WAIT_FOR_TRAINING="${WAIT_FOR_TRAINING:-1}"
TRAINING_TIMEOUT_SECONDS="${TRAINING_TIMEOUT_SECONDS:-7200}"
TRAINING_WINDOW_SESSIONS="${TRAINING_WINDOW_SESSIONS:-0}"
TRAINING_WINDOW_MIN_SESSIONS="${TRAINING_WINDOW_MIN_SESSIONS:-$TRAINING_WINDOW_SESSIONS}"

shopt -s nullglob
sessions=( scripts/sessions/*.json )
n=${#sessions[@]}
if [ "$n" -eq 0 ]; then
  echo "No session files found in scripts/sessions/"
  exit 0
fi
if [ "$n" -eq 1 ]; then
  echo "Only one session found; refusing to upload it (it would be the test session)."
  echo "Test session: ${sessions[0]}"
  exit 0
fi

upload_count="$n"
if [ "$HOLDOUT_LAST" = "1" ]; then
  upload_count=$((n - 1))
  test_session="${sessions[$upload_count]}"
  echo "Holding out test session: $test_session"
fi
echo "Uploading $upload_count training session(s)..."

current_model_state() {
  curl -fsS "${PROXY_URL%/}/v1/tinker/current" 2>/dev/null || true
}

session_idx=0
for task in "${sessions[@]:0:$upload_count}"; do
  session_idx=$((session_idx + 1))
  echo "Uploading $task"
  before="$(current_model_state)"
  start_ts="$(date +%s)"

  curl -sfS -X POST "${PROXY_URL%/}/session" \
    -H "Content-Type: application/json" \
    --data-binary @"$task"
  echo

  if [ "$WAIT_FOR_TRAINING" != "1" ]; then
    sleep 0.05
    continue
  fi

  if [ "$TRAINING_WINDOW_MIN_SESSIONS" -gt 1 ] && [ "$session_idx" -lt "$TRAINING_WINDOW_MIN_SESSIONS" ]; then
    echo "Warmup upload ${session_idx}/${upload_count}: sliding window has not reached min training sessions (${TRAINING_WINDOW_MIN_SESSIONS}); not waiting for model update."
    sleep 0.05
    continue
  fi

  echo "Waiting for training/model update..."
  while true; do
    after="$(current_model_state)"
    if [ "$after" != "$before" ] && echo "$after" | grep -q '"slug"'; then
      end_ts="$(date +%s)"
      echo "Updated in $((end_ts - start_ts))s: $after"
      break
    fi
    now_ts="$(date +%s)"
    if [ "$TRAINING_TIMEOUT_SECONDS" -gt 0 ] && [ "$((now_ts - start_ts))" -ge "$TRAINING_TIMEOUT_SECONDS" ]; then
      echo "Timed out waiting for model update after ${TRAINING_TIMEOUT_SECONDS}s: $task" >&2
      exit 1
    fi
    sleep 10
  done
done
