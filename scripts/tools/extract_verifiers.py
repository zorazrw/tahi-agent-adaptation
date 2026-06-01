#!/usr/bin/env python3
"""
Extract final verifier criteria from exported task sessions.

Reads weight-format exports (``task_units`` / ``workflow_tree_final``) or default-format
exports (last ``environment.workflow`` snapshot).

Output JSON is an array of objects::

  {
    "uuid": "<session id>",
    "instruction": "<initial user instruction>",
    "verifiers": ["criterion 1", "criterion 2", ...]
  }

Examples:
  python scripts/tools/extract_verifiers.py scripts/sessions
  python scripts/tools/extract_verifiers.py scripts/sessions -o verifiers.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator


def load_sessions_from_path(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        sessions: list[dict[str, Any]] = []
        for fp in sorted(path.glob("*.json")):
            sessions.extend(_load_sessions_file(fp))
        return sessions
    return _load_sessions_file(path)


def _load_sessions_file(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        if isinstance(raw.get("sessions"), list):
            return [x for x in raw["sessions"] if isinstance(x, dict)]
        return [raw]
    raise ValueError(f"{path.name}: JSON root must be a session object or array")


def session_instruction(session: dict[str, Any]) -> str:
    raw = session.get("initial_task_instruction")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()

    for task in session.get("task_units") or []:
        if not isinstance(task, dict):
            continue
        for traj in task.get("agent_trajectories") or []:
            if not isinstance(traj, dict):
                continue
            for m in traj.get("messages") or []:
                if isinstance(m, dict) and m.get("role") == "user":
                    content = m.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()

    traj = session.get("trajectory")
    if isinstance(traj, list):
        for step in traj:
            if not isinstance(step, dict) or step.get("actor") != "user":
                continue
            action = step.get("action")
            if isinstance(action, str) and action.startswith('message("'):
                # message("...") — best-effort unwrap
                inner = action[len('message("') :]
                if inner.endswith('")'):
                    inner = inner[:-2]
                if inner.strip():
                    return inner.strip()
    return ""


def _criteria_from_node(node: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for v in node.get("verifiers") or []:
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
        elif isinstance(v, dict):
            c = v.get("criterion")
            if isinstance(c, str) and c.strip():
                out.append(c.strip())
    return out


def _criteria_from_tree(nodes: list[Any]) -> list[str]:
    """Depth-first criteria from workflow tree nodes."""
    out: list[str] = []

    def walk(node: dict[str, Any]) -> None:
        out.extend(_criteria_from_node(node))
        for child in node.get("children") or []:
            if isinstance(child, dict):
                walk(child)

    for node in nodes:
        if isinstance(node, dict):
            walk(node)
    return out


def _criteria_from_unit_verifiers(verifiers: list[Any]) -> list[str]:
    out: list[str] = []
    for v in verifiers:
        if isinstance(v, dict):
            c = v.get("criterion")
            if isinstance(c, str) and c.strip():
                out.append(c.strip())
    return out


def extract_verifier_criteria(session: dict[str, Any]) -> list[str]:
    units = session.get("task_units")
    if isinstance(units, list) and units:
        planning = next(
            (u for u in units if isinstance(u, dict) and u.get("intent") == "planning"),
            units[0] if units and isinstance(units[0], dict) else None,
        )
        tree_final = None
        if isinstance(planning, dict):
            wf = planning.get("workflow_tree_final")
            if isinstance(wf, list) and wf:
                tree_final = wf

        if tree_final:
            criteria = _criteria_from_tree(tree_final)
            # Append any criteria recorded on execution units but missing from the tree.
            seen = set(criteria)
            for unit in units:
                if not isinstance(unit, dict) or unit.get("intent") == "planning":
                    continue
                vs = unit.get("verifiers")
                if not isinstance(vs, list):
                    continue
                for c in _criteria_from_unit_verifiers(vs):
                    if c not in seen:
                        criteria.append(c)
                        seen.add(c)
            return criteria

        # No workflow_tree_final: concatenate execution-unit verifiers in order.
        out: list[str] = []
        seen: set[str] = set()
        for unit in units:
            if not isinstance(unit, dict) or unit.get("intent") == "planning":
                continue
            vs = unit.get("verifiers")
            if not isinstance(vs, list):
                continue
            for c in _criteria_from_unit_verifiers(vs):
                if c not in seen:
                    out.append(c)
                    seen.add(c)
        if out:
            return out

    return _criteria_from_trajectory(session)


def _criteria_from_trajectory(session: dict[str, Any]) -> list[str]:
    last_wf: list[Any] | None = None
    for env in _iter_environments(session):
        wf = env.get("workflow")
        if isinstance(wf, list) and wf:
            last_wf = wf
    if not last_wf:
        return []
    return _criteria_from_tree(last_wf)


def _iter_environments(session: dict[str, Any]) -> Iterator[dict[str, Any]]:
    units = session.get("task_units")
    if isinstance(units, list):
        for unit in units:
            if isinstance(unit, dict):
                env = unit.get("environment")
                if isinstance(env, dict):
                    yield env
        return
    traj = session.get("trajectory")
    if isinstance(traj, list):
        for step in traj:
            if isinstance(step, dict):
                env = step.get("environment")
                if isinstance(env, dict):
                    yield env


def session_record(session: dict[str, Any]) -> dict[str, Any]:
    uid = session.get("uuid")
    return {
        "uuid": str(uid) if uid is not None else "",
        "instruction": session_instruction(session),
        "verifiers": extract_verifier_criteria(session),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("sessions"),
        help="Session JSON file or directory of *.json exports (default: sessions)",
    )
    parser.add_argument("-o", "--output", type=Path, help="Write JSON array here (default: stdout)")
    args = parser.parse_args(argv)

    input_path = args.input
    if not input_path.is_absolute():
        for c in (input_path, Path(__file__).resolve().parent.parent / input_path):
            if c.exists():
                input_path = c
                break

    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    sessions = load_sessions_from_path(input_path)
    if not sessions:
        print(f"No sessions in {input_path}", file=sys.stderr)
        return 1

    records = [session_record(s) for s in sessions]
    text = json.dumps(records, indent=2, ensure_ascii=False) + "\n"

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {len(records)} session(s) to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
