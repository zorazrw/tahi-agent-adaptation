from __future__ import annotations

import json
import sys

from .bridge import resolve_checkpoint_sync, run_request_sync


def main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        sys.stdout.write(json.dumps({"ok": False, "error": "Expected JSON payload on stdin"}))
        return 0

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        sys.stdout.write(json.dumps({"ok": False, "error": f"Invalid JSON input: {error}"}))
        return 0

    try:
        command = payload.get("command", "sample")
        if command == "resolve_checkpoint":
            result = resolve_checkpoint_sync(payload)
        else:
            result = run_request_sync(payload)
    except Exception as error:  # pragma: no cover - surfaced to TypeScript caller
        sys.stdout.write(json.dumps({"ok": False, "error": str(error)}))
        return 0

    sys.stdout.write(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
