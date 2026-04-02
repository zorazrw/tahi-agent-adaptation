#!/usr/bin/env python3
"""REINFORCE-style export: OPD units + reward {verifier success ratio, human step count}."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

import session_export_common as s  # noqa: E402


def _chunk_reward(agent_steps: list[dict], human_steps: list[dict]) -> dict[str, float | int]:
    env = None
    for step in reversed(agent_steps + human_steps):
        e = step.get("environment") if isinstance(step, dict) else None
        if isinstance(e, dict):
            env = e
            break
    statuses: list[str] = []
    wf = env.get("workflow") if env else None

    def walk(nodes: object) -> None:
        if not isinstance(nodes, list):
            return
        for n in nodes:
            if not isinstance(n, dict):
                continue
            for v in n.get("verifiers") or []:
                if isinstance(v, dict) and isinstance(v.get("status"), str):
                    statuses.append(v["status"])
            walk(n.get("children"))

    walk(wf)
    n = len(statuses)
    ver = 0.0 if not n else round(statuses.count("success") / n, 4)
    return {"verifier": ver, "human": len(human_steps)}


def _units(traj: list) -> tuple[dict | None, list[dict]]:
    initial, pairs = s.pair_segments(traj)
    if not pairs:
        return initial, []
    prefix = list(initial["steps"]) if initial else []
    out: list[dict] = []
    for j, (ag, hm) in enumerate(pairs):
        ag, hm = list(ag), list(hm)
        out.append(
            {
                "index": j,
                "user_messages": [s.user_text(x) for x in prefix],
                "agent_trajectory": ag,
                "human_trajectory": hm,
                "reward": _chunk_reward(ag, hm),
            }
        )
        prefix.extend(hm)
    return initial, out


def _session(blob: dict) -> dict:
    meta = {"uuid": blob.get("uuid"), "name": blob.get("name")}
    traj = blob.get("trajectory")
    if not isinstance(traj, list):
        return s.strip_actor({**meta, "initial_message": None, "learning_units": [], "error": "bad_trajectory"})
    initial, units = _units(traj)
    return s.strip_actor({**meta, "initial_message": initial, "learning_units": units})


def main() -> None:
    p = argparse.ArgumentParser(description="Export trajectories with REINFORCE-style rewards.")
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
