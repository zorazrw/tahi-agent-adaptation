curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello, who are you?"}],
    "stream": false
  }' | python3 -m json.tool
