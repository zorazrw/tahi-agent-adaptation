#!/usr/bin/env python3
"""
Compute summary metrics from an exported session JSON (e.g. export_task_sessions default format).

Input may be either:

- One session object with ``trajectory`` (and optional ``uuid``, ``name``), or
- A JSON **array** of such session objects (e.g. multi-session export).

Each session is analyzed independently; human mode prints a block per session; ``--json`` emits
``{"file", "session_count", "sessions": [...]}`` with one merged record per session.

Metrics:
  1. Verifier success: for each step with ``environment.workflow``, use only **top-level** nodes
     (the array’s immediate items); ``children`` subtrees are ignored. success = 1 when
     ``status`` == ``success``. Prints per-step pass rates in trajectory order, plus mean and pooled.
  2. User run lengths: contiguous blocks of steps with ``actor`` == ``user``.
  3. Actor switches: count of indices i > 0 where ``actor[i] != actor[i-1]``.

Usage:
  python analyze_trajectory_metrics.py path/to/session.json
  python analyze_trajectory_metrics.py path/to/session.json --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def collect_verifier_success_bits_top_level(wf: Any) -> List[bool]:
    """Only top-level ``workflow`` nodes; skip nested ``children``. True = verifier passed."""
    if not isinstance(wf, list):
        return []
    out: List[bool] = []
    for node in wf:
        if not isinstance(node, dict):
            continue
        for v in node.get("verifiers") or []:
            if isinstance(v, dict):
                out.append(str(v.get("status")) == "success")
    return out


def analyze_trajectory(traj: List[dict]) -> dict:
    per_step_rates: List[float] = []
    per_step_indices: List[int] = []
    per_step_counts: List[Tuple[int, int]] = []  # (successes, total) per contributing step
    all_bits: List[int] = []

    for idx, step in enumerate(traj):
        env = step.get("environment")
        if not isinstance(env, dict):
            continue
        wf = env.get("workflow")
        bits = collect_verifier_success_bits_top_level(wf)
        if not bits:
            continue
        s = sum(bits)
        t = len(bits)
        rate = s / t
        per_step_rates.append(rate)
        per_step_indices.append(idx)
        per_step_counts.append((s, t))
        all_bits.extend(1 if b else 0 for b in bits)

    avg_rate_over_steps: Optional[float] = (
        sum(per_step_rates) / len(per_step_rates) if per_step_rates else None
    )
    overall_rate: Optional[float] = sum(all_bits) / len(all_bits) if all_bits else None

    # Continuous user runs
    user_runs: List[int] = []
    i = 0
    n = len(traj)
    while i < n:
        if traj[i].get("actor") == "user":
            j = i
            while j < n and traj[j].get("actor") == "user":
                j += 1
            user_runs.append(j - i)
            i = j
        else:
            i += 1

    actors = [step.get("actor") for step in traj]
    actor_switches = sum(1 for k in range(1, len(actors)) if actors[k] != actors[k - 1])

    return {
        "trajectory_length": len(traj),
        "verifier_steps_with_data": len(per_step_rates),
        "verifier_step_indices": per_step_indices,
        "per_step_verifier_pass_rates": per_step_rates,
        "per_step_verifier_counts": [{"successes": a, "total": b} for a, b in per_step_counts],
        "verifier_checks_total": len(all_bits),
        "verifier_successes_total": sum(all_bits),
        "mean_per_step_verifier_pass_rate": avg_rate_over_steps,
        "pooled_verifier_pass_rate": overall_rate,
        "user_action_runs_count": len(user_runs),
        "user_action_run_lengths": user_runs,
        "user_action_steps_total": sum(user_runs),
        "actor_switch_count": actor_switches,
    }


def parse_session_records(raw: Any) -> List[dict]:
    """Return a list of session dicts, each with a ``trajectory`` list."""
    if isinstance(raw, dict) and isinstance(raw.get("trajectory"), list):
        return [raw]
    if isinstance(raw, list):
        out: List[dict] = []
        for item in raw:
            if isinstance(item, dict) and isinstance(item.get("trajectory"), list):
                out.append(item)
        return out
    return []


def print_session_metrics_text(
    path: Path,
    session_index: int,
    session_total: int,
    session_blob: dict,
    metrics: dict,
) -> None:
    name = session_blob.get("name") or session_blob.get("uuid") or path.stem
    sid = session_blob.get("uuid", "")
    if session_total > 1:
        print(f"--- Session {session_index + 1} / {session_total} ---")
        if sid:
            print(f"uuid: {sid}")
        print(f"name: {name}")
    else:
        print(f"File: {path.name}")
        print(f"Session: {name}")
        if sid:
            print(f"uuid: {sid}")
    print(f"Trajectory length: {metrics['trajectory_length']}")
    print()
    print("=== 1. Verifier success (environment.workflow, top-level nodes only) ===")
    print(f"Steps with ≥1 verifier: {metrics['verifier_steps_with_data']}")
    rates = metrics["per_step_verifier_pass_rates"]
    counts = metrics["per_step_verifier_counts"]
    if rates:
        print("Per-step pass rates (chronological trajectory order; step_index: rate [successes/total]):")
        print(f"{[round(r, 3) for r in rates]}")
    else:
        print("Per-step pass rates: (none)")
    mps = metrics["mean_per_step_verifier_pass_rate"]
    pool = metrics["pooled_verifier_pass_rate"]
    print(
        "Mean of per-step success rates:",
        f"{mps:.3f}" if mps is not None else "n/a",
    )
    print(
        "Overall (all verifiers pooled):",
        f"{pool:.3f}" if pool is not None else "n/a",
    )
    print(
        f"Total verifier checks: {metrics['verifier_checks_total']} "
        f"| successes: {metrics['verifier_successes_total']}",
    )
    print()
    print("=== 2. Continuous user action sequences ===")
    print(f"Number of user runs: {metrics['user_action_runs_count']}")
    print(f"Lengths: {metrics['user_action_run_lengths']}")
    lengths = metrics["user_action_run_lengths"]
    if lengths:
        print(f"Min / max / sum: {min(lengths)} / {max(lengths)} / {sum(lengths)}")
    print()
    print("=== 3. Actor switches ===")
    print(f"Count (i>0 where actor[i] != actor[i-1]): {metrics['actor_switch_count']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Trajectory metrics for exported session JSON.")
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to JSON file with a 'trajectory' array (default export format).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print results as JSON only (no prose).",
    )
    args = parser.parse_args()
    path: Path = args.json_path
    if not path.is_file():
        print(f"Not a file: {path}", file=sys.stderr)
        return 1

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as e:
        print(f"Could not read {path}: {e}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}", file=sys.stderr)
        return 1

    sessions = parse_session_records(raw)
    if not sessions:
        print(
            "Expected a session object with 'trajectory', or a JSON array of such objects.",
            file=sys.stderr,
        )
        return 1

    n = len(sessions)
    traj_items = [[s for s in (sess.get("trajectory") or []) if isinstance(s, dict)] for sess in sessions]
    all_metrics: List[dict] = [analyze_trajectory(ti) for ti in traj_items]

    if args.json:
        out_sessions: List[Dict[str, Any]] = []
        for sess, m in zip(sessions, all_metrics):
            row: Dict[str, Any] = dict(m)
            if "uuid" in sess:
                row["uuid"] = sess["uuid"]
            if "name" in sess:
                row["name"] = sess["name"]
            out_sessions.append(row)
        print(
            json.dumps(
                {
                    "file": path.name,
                    "session_count": n,
                    "sessions": out_sessions,
                },
                indent=2,
            )
        )
        return 0

    print(f"File: {path.name}")
    print(f"Sessions: {n}")
    print()
    for idx, (sess_blob, metrics) in enumerate(zip(sessions, all_metrics)):
        print_session_metrics_text(path, idx, n, sess_blob, metrics)
        if idx < n - 1:
            print()
            print("=" * 72)
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
