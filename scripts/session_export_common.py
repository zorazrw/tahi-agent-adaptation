"""Shared trajectory parsing for export_dpo_data, export_opd_data, export_reinforce_data."""

from __future__ import annotations

import json
from typing import Any, TypedDict


class _InitialUserChunk(TypedDict):
    actor: str
    steps: list[dict]


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
    if isinstance(data.get("task_units"), list):
        return [data]
    return []


def trajectory_from_blob(blob: dict) -> list | None:
    """Flatten session data to the legacy ``trajectory`` shape export scripts expect.

    - If ``trajectory`` is a list at top level, return it unchanged.
    - If ``task_units`` is present, expand each unit's inner steps and set ``actor``
      from the unit; copy unit-level ``environment`` onto steps that omit it so
      workflow/file snapshots remain visible to reward and plan injection logic.

    Row spans over the returned list match :func:`task_unit_row_bounds` for the same ``task_units``.
    """
    t = blob.get("trajectory")
    if isinstance(t, list):
        return t
    units = blob.get("task_units")
    if not isinstance(units, list):
        return None
    out: list[dict] = []
    for unit in units:
        if not isinstance(unit, dict):
            continue
        actor = unit.get("actor")
        if actor not in ("user", "agent"):
            continue
        inner = unit.get("trajectory")
        if not isinstance(inner, list):
            continue
        unit_env = unit.get("environment")
        unit_env_dict = unit_env if isinstance(unit_env, dict) else None
        for step in inner:
            if not isinstance(step, dict):
                continue
            row = dict(step)
            row["actor"] = actor
            if unit_env_dict is not None and "environment" not in row:
                row["environment"] = unit_env_dict
            out.append(row)
    return out


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


def user_task_unit_indices(task_units: list[Any]) -> list[int]:
    """Indices of ``task_units`` rows whose ``actor`` is ``'user'``."""
    return [i for i, u in enumerate(task_units) if isinstance(u, dict) and u.get("actor") == "user"]


def task_unit_row_bounds(task_units: list[Any], traj: list[Any]) -> dict[int, tuple[int, int]] | None:
    """Map each processed task-unit index to ``[start, end)`` row spans in ``traj``.

    Spans must match :func:`trajectory_from_blob` (same skip rules for invalid rows/steps).
    Returns ``None`` if the flattened length does not match ``len(traj)``.
    """
    row = 0
    bounds: dict[int, tuple[int, int]] = {}
    for ui, unit in enumerate(task_units):
        if not isinstance(unit, dict) or unit.get("actor") not in ("user", "agent"):
            continue
        inner = unit.get("trajectory")
        if not isinstance(inner, list):
            continue
        start = row
        for step in inner:
            if not isinstance(step, dict):
                continue
            if row >= len(traj):
                return None
            row += 1
        bounds[ui] = (start, row)
    if row != len(traj):
        return None
    return bounds


def pairs_from_task_units(
    blob: dict[str, Any], traj: list[Any]
) -> tuple[_InitialUserChunk, list[tuple[list[dict], list[dict]]], list[int]] | None:
    """Learning-unit boundaries at each ``actor: 'user'`` task unit (same pairing intent as :func:`pair_segments`).

    Each pair is ``(agent_steps, human_steps)``: agent steps after user unit ``k`` until the next
    user unit, and that next user unit's inner steps. The first user chunk becomes ``initial``.

    Returns ``None`` if ``blob`` has no usable ``task_units`` or spans do not align with ``traj``.
    """
    tu = blob.get("task_units")
    if not isinstance(tu, list) or not traj:
        return None
    bounds = task_unit_row_bounds(tu, traj)
    if bounds is None:
        return None
    u_idx = user_task_unit_indices(tu)
    if not u_idx:
        return None
    u0 = u_idx[0]
    if u0 not in bounds:
        return None
    s0, e0 = bounds[u0]
    initial: _InitialUserChunk = {"actor": "user", "steps": list(traj[s0:e0])}
    pairs: list[tuple[list[dict], list[dict]]] = []
    for k in range(len(u_idx) - 1):
        ua = u_idx[k]
        ub = u_idx[k + 1]
        if ua not in bounds or ub not in bounds:
            return None
        _, e_a = bounds[ua]
        s_b, e_b = bounds[ub]
        pairs.append((list(traj[e_a:s_b]), list(traj[s_b:e_b])))
    return initial, pairs, u_idx
