#!/usr/bin/env bash
# Upload every session JSON in scripts/sessions/ to the local proxy server,
# except the last one (lexicographic order), which is held back as a test
# session that the model should never see during training.
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

# upload first 12 sessions for training
last_idx=1
test_session="${sessions[$last_idx]}"
echo "Holding out test session: $test_session"
echo "Uploading $last_idx training session(s)..."

for task in "${sessions[@]:0:$last_idx}"; do
  curl -X POST http://localhost:8000/session \
    -H "Content-Type: application/json" \
    --data-binary @"$task"

  sleep 0.05
done
