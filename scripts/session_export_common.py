"""Shared trajectory parsing for export_dpo_data / export_opd_data."""

from __future__ import annotations

import json
from typing import Any


def strip_actor(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: strip_actor(v) for k, v in x.items() if k != "actor"}
    if isinstance(x, list):
        return [strip_actor(i) for i in x]
    return x


def user_text(step: dict) -> str:
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


def blobs(data: Any) -> list[dict]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if not isinstance(data, dict):
        return []
    if isinstance(data.get("sessions"), list):
        return [x for x in data["sessions"] if isinstance(x, dict)]
    if isinstance(data.get("trajectory"), list):
        return [data]
    return []


def pair_segments(traj: list) -> tuple[dict | None, list[tuple[list[dict], list[dict]]]]:
    """(initial_user_chunk_or_none, [(agent_steps, human_steps), ...])."""
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
    return initial, pairs
