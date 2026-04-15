from __future__ import annotations

import argparse
import json
import sys

from .bridge import run_request_sync


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test Tinker tool-call parsing through the bridge")
    parser.add_argument("--model", required=True, help="Registered Tinker model id")
    parser.add_argument("--base-model", required=True, help="Base Hugging Face model id")
    parser.add_argument("--model-path", help="Optional Tinker checkpoint/model path")
    parser.add_argument("--base-url", help="Optional Tinker base URL")
    parser.add_argument("--api-key", help="Optional explicit Tinker API key")
    parser.add_argument("--renderer", help="Optional renderer override")
    parser.add_argument(
        "--prompt",
        default="What's the weather like in San Francisco right now?",
        help="Prompt that should trigger a tool call",
    )
    args = parser.parse_args()

    payload = {
        "provider": {
            "base_url": args.base_url or "",
            "api_key": args.api_key or "",
        },
        "model": {
            "id": args.model,
            "base_model": args.base_model,
            "model_path": args.model_path or "",
            "renderer_name": args.renderer or "",
        },
        "options": {
            "max_tokens": 512,
            "temperature": 0.0,
        },
        "context": {
            "system_prompt": "You are a helpful assistant.",
            "messages": [
                {
                    "role": "user",
                    "content": args.prompt,
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City name, e.g. 'San Francisco'",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                            "required": ["location"],
                        },
                    },
                }
            ],
        },
    }

    result = run_request_sync(payload)
    print(json.dumps(result, indent=2))

    tool_calls = ((result.get("message") or {}).get("tool_calls") or []) if result.get("ok") else []
    if not result.get("ok"):
        return 1
    if not tool_calls:
        print("No parsed tool calls were returned.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
