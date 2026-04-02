#!/usr/bin/env python3
"""OPD from export_task_sessions JSON: one learning unit per agent→human segment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

import session_export_common as s  # noqa: E402


def _opd_units(traj: list) -> tuple[dict | None, list[dict]]:
    initial, pairs = s.pair_segments(traj)
    if not pairs:
        return initial, []
    prefix = list(initial["steps"]) if initial else []
    units = []
    for j, (ag, hm) in enumerate(pairs):
        units.append(
            {
                "index": j,
                "user_messages": [s.user_text(x) for x in prefix],
                "agent_trajectory": list(ag),
                "human_trajectory": list(hm),
            }
        )
        prefix.extend(hm)
    return initial, units


def _session(blob: dict) -> dict:
    meta = {"uuid": blob.get("uuid"), "name": blob.get("name")}
    traj = blob.get("trajectory")
    if not isinstance(traj, list):
        out = {**meta, "initial_message": None, "learning_units": [], "error": "bad_trajectory"}
    else:
        initial, units = _opd_units(traj)
        out = {**meta, "initial_message": initial, "learning_units": units}
    return s.strip_actor(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Export trajectories as OPD learning units.")
    p.add_argument("input", nargs="?", default="-")
    p.add_argument("-o", "--output", default="-")
    args = p.parse_args()
    raw = json.load(sys.stdin) if args.input == "-" else json.loads(Path(args.input).read_text(encoding="utf-8"))
    results = [_session(b) for b in s.blobs(raw)]
    payload = results[0] if len(results) == 1 else {"sessions": results}
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output == "-":
        sys.stdout.write(text)
    else:
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
