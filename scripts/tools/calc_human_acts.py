#!/usr/bin/env python3
"""
Count human iterations and user actions in exported session JSON.

Uses the same trajectory merge rules and user-action counting as
``rate_file_versions.py``: each contiguous block of steps with
``actor`` == ``user`` is one iteration; the block length is the user
action count for that iteration.

Supports ``trajectory``, ``task_units`` (per-step ``trajectory``), and
weight exports (``agent_trajectories`` + ``workflow_tree_final``).

Examples:
  python scripts/tools/calc_human_acts.py -j out.json
  python scripts/tools/calc_human_acts.py -j out.json -s <uuid>
  python scripts/tools/calc_human_acts.py -j out.json -o human_acts.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_tools = Path(__file__).resolve().parent
if str(_tools) not in sys.path:
    sys.path.insert(0, str(_tools))

from rate_file_versions import (  # noqa: E402
    load_sessions,
    resolve_session,
    session_meta,
    session_trajectory,
    session_user_action_stats,
)


def human_acts_for_session(session: dict[str, Any]) -> dict[str, Any]:
    per_iter = session_user_action_stats(session)
    traj = session_trajectory(session)
    meta = session_meta(session)
    return {
        **meta,
        "trajectory_length": len(traj),
        "iteration_count": len(per_iter),
        "user_actions_total": sum(per_iter),
        "user_actions_per_iteration": per_iter,
    }


def summarize_sessions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    iteration_counts = [int(r.get("iteration_count") or 0) for r in rows]
    action_totals = [int(r.get("user_actions_total") or 0) for r in rows]
    return {
        "session_count": len(rows),
        "total_iterations": sum(iteration_counts),
        "total_user_actions": sum(action_totals),
        "mean_iterations_per_session": (sum(iteration_counts) / len(rows)) if rows else None,
        "mean_user_actions_per_session": (sum(action_totals) / len(rows)) if rows else None,
    }


def print_session_summary(row: dict[str, Any], *, session_index: int | None = None, session_total: int = 1) -> None:
    label = row.get("name") or row.get("uuid") or "session"
    if session_total > 1 and session_index is not None:
        print(f"--- session {session_index + 1}/{session_total}: {label} ---", file=sys.stderr)
    else:
        print(f"--- {label} ---", file=sys.stderr)
    if row.get("uuid"):
        print(f"uuid: {row['uuid']}", file=sys.stderr)
    per_iter = row.get("user_actions_per_iteration") or []
    print(
        f"user actions: {row.get('iteration_count', len(per_iter))} iteration(s), "
        f"{row.get('user_actions_total', sum(per_iter))} total",
        file=sys.stderr,
    )
    for i, count in enumerate(per_iter, start=1):
        print(f"  iteration {i}: {count} user {'action' if count == 1 else 'actions'}", file=sys.stderr)


def print_aggregate_summary(summary: dict[str, Any]) -> None:
    print("--- aggregate ---", file=sys.stderr)
    print(f"sessions: {summary.get('session_count', 0)}", file=sys.stderr)
    print(
        f"iterations: {summary.get('total_iterations', 0)} total | "
        f"user actions: {summary.get('total_user_actions', 0)} total",
        file=sys.stderr,
    )
    mean_iter = summary.get("mean_iterations_per_session")
    mean_actions = summary.get("mean_user_actions_per_session")
    if isinstance(mean_iter, (int, float)):
        print(f"mean iterations/session: {mean_iter:.2f}", file=sys.stderr)
    if isinstance(mean_actions, (int, float)):
        print(f"mean user actions/session: {mean_actions:.2f}", file=sys.stderr)
    print("--- end summary ---", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-j", "--json", type=Path, required=True, help="Session export JSON (one object or array)")
    p.add_argument("-s", "--session", default=None, help="Session uuid (required when JSON has multiple sessions)")
    p.add_argument("-o", "--out", type=Path, default=None, help="Write report JSON here (default: stdout)")
    args = p.parse_args()

    if not args.json.is_file():
        raise SystemExit(f"Not a file: {args.json}")

    sessions = load_sessions(args.json)
    if args.session and str(args.session).strip():
        selected = [resolve_session(sessions, args.session)]
    elif len(sessions) == 1:
        selected = sessions
    else:
        selected = sessions

    rows = [human_acts_for_session(s) for s in selected]
    report: dict[str, Any] = {
        "sessions_file": str(args.json.resolve()),
        "sessions": rows,
        "summary": summarize_sessions(rows),
    }

    for i, row in enumerate(rows):
        print_session_summary(row, session_index=i, session_total=len(rows))
    if len(rows) > 1:
        print_aggregate_summary(report["summary"])

    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote report to {args.out.resolve()}", file=sys.stderr)
    else:
        sys.stdout.write(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
