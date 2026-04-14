jq '.[1]' scripts/out.json | curl -X POST http://localhost:8000/session \
    -H "Content-Type: application/json" \
    -d @-