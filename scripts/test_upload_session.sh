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
#   TRAINING_CONFIG=scripts/config-dpo-window-adj.yaml
#   HOLDOUT_LAST=1          # hold back the last session as test data
#   WAIT_FOR_TRAINING=0     # enqueue all sessions without waiting/timing
#   TRAINING_TIMEOUT_SECONDS=7200
#   TRAINING_WINDOW_SESSIONS=2
#   TRAINING_WINDOW_MIN_SESSIONS=2  # first min_sessions-1 uploads are warmup only
# If TRAINING_WINDOW_MIN_SESSIONS is unset, the script will try to infer it
# from TRAINING_CONFIG. If TRAINING_CONFIG is also unset, it will scan
# scripts/config*.yaml for a matching proxy_port.
set -euo pipefail

PROXY_URL="${AGENT_COWORK_PROXY_URL:-http://localhost:8000}"
TRAINING_CONFIG="${TRAINING_CONFIG:-}"
HOLDOUT_LAST="${HOLDOUT_LAST:-0}"
WAIT_FOR_TRAINING="${WAIT_FOR_TRAINING:-1}"
TRAINING_TIMEOUT_SECONDS="${TRAINING_TIMEOUT_SECONDS:-7200}"
TRAINING_WINDOW_SESSIONS="${TRAINING_WINDOW_SESSIONS:-0}"
TRAINING_WINDOW_MIN_SESSIONS="${TRAINING_WINDOW_MIN_SESSIONS:-$TRAINING_WINDOW_SESSIONS}"

yaml_get_key() {
  local file="$1"
  local key="$2"
  awk -F':' -v want="$key" '
    $0 ~ "^[[:space:]]*" want "[[:space:]]*:" {
      val = substr($0, index($0, ":") + 1)
      sub(/[[:space:]]*#.*$/, "", val)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", val)
      print val
      exit
    }
  ' "$file"
}

infer_training_config_from_port() {
  local port="$1"
  local match=""
  local file
  for file in scripts/config*.yaml; do
    [[ -f "$file" ]] || continue
    local file_port
    file_port="$(yaml_get_key "$file" "proxy_port")"
    [[ "$file_port" == "$port" ]] || continue
    if [[ -n "$(yaml_get_key "$file" "training_window_min_sessions")" ]]; then
      echo "$file"
      return 0
    fi
    if [[ -z "$match" ]]; then
      match="$file"
    fi
  done
  [[ -n "$match" ]] && echo "$match"
}

if [[ "$TRAINING_WINDOW_MIN_SESSIONS" == "0" ]]; then
  inferred_port="${PROXY_URL##*:}"
  inferred_port="${inferred_port%%/*}"
  if [[ -z "$TRAINING_CONFIG" && -n "$inferred_port" ]]; then
    TRAINING_CONFIG="$(infer_training_config_from_port "$inferred_port" || true)"
  fi
  if [[ -n "$TRAINING_CONFIG" && -f "$TRAINING_CONFIG" ]]; then
    inferred_min="$(yaml_get_key "$TRAINING_CONFIG" "training_window_min_sessions")"
    if [[ -z "$inferred_min" ]]; then
      inferred_min="$(yaml_get_key "$TRAINING_CONFIG" "training_window_sessions")"
    fi
    if [[ -n "$inferred_min" && "$inferred_min" != "null" ]]; then
      TRAINING_WINDOW_MIN_SESSIONS="$inferred_min"
      echo "Inferred TRAINING_WINDOW_MIN_SESSIONS=${TRAINING_WINDOW_MIN_SESSIONS} from ${TRAINING_CONFIG}"
    fi
  fi
fi

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
