#!/usr/bin/env python3
"""DPO rows from export_task_sessions JSON: pairwise agent segments, stripped ``actor`` fields."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _strip_actor(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _strip_actor(v) for k, v in x.items() if k != "actor"}
    if isinstance(x, list):
        return [_strip_actor(i) for i in x]
    return x


def _user_text(step: dict) -> str:
    a = step.get("action")
    if not isinstance(a, str):
        return ""
    if a.startswith("message(") and a.endswith(")"):
        inner = a[len("message(") : -1].strip()
        if len(inner) >= 2 and inner[0] == '"' == inner[-1]:
            try:
                return str(json.loads(inner))
            except json.JSONDecodeError:
                pass
    return a


def _blobs(data: Any) -> list[dict]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if not isinstance(data, dict):
        return []
    if isinstance(data.get("sessions"), list):
        return [x for x in data["sessions"] if isinstance(x, dict)]
    if isinstance(data.get("trajectory"), list):
        return [data]
    return []


def _dpo_from_trajectory(traj: list) -> tuple[dict | None, list[dict]]:
    chunks: list[tuple[str, list[dict]]] = []
    for step in traj:
        if not isinstance(step, dict) or step.get("actor") not in ("user", "agent"):
            continue
        who = step["actor"]
        if chunks and chunks[-1][0] == who:
            chunks[-1][1].append(step)
        else:
            chunks.append((who, [step]))

    initial = None
    if chunks and chunks[0][0] == "user":
        initial = {"actor": "user", "steps": chunks[0][1]}
        chunks = chunks[1:]

    pairs: list[tuple[list[dict], list[dict]]] = []
    i = 0
    while i + 1 < len(chunks):
        if chunks[i][0] == "agent" and chunks[i + 1][0] == "user":
            pairs.append((chunks[i][1], chunks[i + 1][1]))
            i += 2
        else:
            i += 1

    if len(pairs) < 2:
        return initial, []

    prefix = list(initial["steps"]) if initial else []
    units: list[dict] = []
    for j in range(len(pairs) - 1):
        ag, hm = pairs[j]
        ag1, _ = pairs[j + 1]
        units.append(
            {
                "index": j,
                "user_messages": [_user_text(s) for s in prefix],
                "rejected_trajectory": list(ag),
                "chosen_trajectory": list(ag1),
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
        initial, units = _dpo_from_trajectory(traj)
        out = {**meta, "initial_message": initial, "learning_units": units}
    return _strip_actor(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Export trajectories as DPO learning units.")
    p.add_argument("input", nargs="?", default="-")
    p.add_argument("-o", "--output", default="-")
    args = p.parse_args()
    raw = json.load(sys.stdin) if args.input == "-" else json.loads(Path(args.input).read_text(encoding="utf-8"))
    results = [_session(b) for b in _blobs(raw)]
    payload = results[0] if len(results) == 1 else {"sessions": results}
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output == "-":
        sys.stdout.write(text)
    else:
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
