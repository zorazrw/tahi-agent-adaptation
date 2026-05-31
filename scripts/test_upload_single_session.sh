#!/usr/bin/env bash
# Upload session JSON to the local training proxy (default http://localhost:8000/session).
#
# Usage:
#   ./scripts/test_upload_session.sh
#     Upload the first session in scripts/sessions/ (lexicographic order), holding out
#     the second as the test session.
#
#   ./scripts/test_upload_session.sh 0018-task1-before-qwen3.5-35b-....json
#   ./scripts/test_upload_session.sh scripts/sessions/0018-task1-....json
#     Upload one or more specific session files (basename or path).

set -euo pipefail

PROXY_URL="${AGENT_COWORK_PROXY_URL:-http://localhost:8000}"
SESSIONS_DIR="scripts/sessions"

upload_file() {
  local file="$1"
  if [[ ! -f "$file" ]]; then
    echo "Not found: $file" >&2
    return 1
  fi
  echo "Uploading $file"
  curl -sfS -X POST "${PROXY_URL%/}/session" \
    -H "Content-Type: application/json" \
    --data-binary @"$file"
  echo
}

resolve_session_file() {
  local arg="$1"
  if [[ -f "$arg" ]]; then
    echo "$arg"
    return 0
  fi
  local under="${SESSIONS_DIR}/${arg}"
  if [[ -f "$under" ]]; then
    echo "$under"
    return 0
  fi
  echo "Unknown session file: $arg (tried ./${arg} and ${under})" >&2
  return 1
}

if [[ $# -gt 0 ]]; then
  for arg in "$@"; do
    upload_file "$(resolve_session_file "$arg")"
    sleep 0.05
  done
  exit 0
fi

# Default: upload first session, hold out second as test session.
shopt -s nullglob
sessions=( "${SESSIONS_DIR}"/*.json )
n=${#sessions[@]}
if [[ "$n" -eq 0 ]]; then
  echo "No session files found in ${SESSIONS_DIR}/"
  exit 0
fi
if [[ "$n" -eq 1 ]]; then
  echo "Only one session found; refusing to upload it (it would be the test session)."
  echo "Test session: ${sessions[0]}"
  exit 0
fi

last_idx=1
test_session="${sessions[$last_idx]}"
echo "Holding out test session: $test_session"
echo "Uploading $last_idx training session(s)..."

for task in "${sessions[@]:0:$last_idx}"; do
  upload_file "$task"
  sleep 0.05
done
