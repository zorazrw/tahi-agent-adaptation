#!/usr/bin/env python3
"""Split a weight-format session export into one POST-ready file per session."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _sessions(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [s for s in payload if isinstance(s, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("sessions"), list):
        return [s for s in payload["sessions"] if isinstance(s, dict)]
    if isinstance(payload, dict):
        return [payload]
    raise ValueError("Input must be a session object, a list of sessions, or {'sessions': [...]}")


def _slug(text: str, fallback: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip()).strip("-._")
    return slug[:80] or fallback


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=here / "out_weight.json")
    parser.add_argument("-o", "--output-dir", type=Path, default=here / "sessions")
    args = parser.parse_args()

    sessions = _sessions(json.loads(args.input.read_text(encoding="utf-8")))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for i, session in enumerate(sessions, start=1):
        session_id = str(session.get("uuid") or i)
        name = str(session.get("name") or session_id)
        path = args.output_dir / f"{i:04d}-{_slug(name, session_id)}-{_slug(session_id, str(i))}.json"
        path.write_text(json.dumps([session], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(path)

    print(f"Wrote {len(sessions)} session file(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
