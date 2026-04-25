from __future__ import annotations

import json
import sys
import argparse

from .bridge import resolve_checkpoint_sync, run_request_sync


def _run_payload(payload: dict) -> dict:
    command = payload.get("command", "sample")
    if command == "resolve_checkpoint":
        return resolve_checkpoint_sync(payload)
    if command == "ping":
        return {"ok": True, "pong": True}
    return run_request_sync(payload)


def _serve_stdio_loop() -> int:
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            result = _run_payload(payload)
        except Exception as error:  # pragma: no cover - surfaced to TypeScript caller
            result = {"ok": False, "error": str(error)}
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Tinker bridge entrypoint")
    parser.add_argument("--serve", action="store_true", help="Run persistent JSON-lines server mode")
    args = parser.parse_args()

    if args.serve:
        return _serve_stdio_loop()

    raw = sys.stdin.read()
    if not raw.strip():
        sys.stdout.write(json.dumps({"ok": False, "error": "Expected JSON payload on stdin"}))
        return 0

    try:
        payload = json.loads(raw)
        result = _run_payload(payload)
    except json.JSONDecodeError as error:
        result = {"ok": False, "error": f"Invalid JSON input: {error}"}
    except Exception as error:  # pragma: no cover - surfaced to TypeScript caller
        result = {"ok": False, "error": str(error)}

    sys.stdout.write(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
