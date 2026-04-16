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
  python analyze_trajectory_metrics.py path/to/session.json --plot-step-curves
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import OrderedDict
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


def workflow_step_name(node: dict, wf_idx: int) -> str:
    """Stable, readable name for a workflow step."""
    return (
        str(node.get("description") or "").strip()
        or str(node.get("id") or "").strip()
        or f"workflow_step_{wf_idx + 1}"
    )


def count_successes(verifiers: Any) -> Tuple[int, int]:
    """Return (successes, total) for verifier dict list."""
    if not isinstance(verifiers, list):
        return (0, 0)
    total = 0
    successes = 0
    for v in verifiers:
        if not isinstance(v, dict):
            continue
        total += 1
        if str(v.get("status")) == "success":
            successes += 1
    return (successes, total)


def make_dense_series(
    per_step_indices: List[int],
    action_indices: List[int],
    success_rates: List[float],
) -> List[Optional[float]]:
    """Align per-step rates to the shared verifier-action timeline with None as missing."""
    action_index_to_pos = {a_idx: pos for pos, a_idx in enumerate(per_step_indices)}
    dense_rates: List[Optional[float]] = [None] * len(per_step_indices)
    for a_idx, rate in zip(action_indices, success_rates):
        pos = action_index_to_pos.get(a_idx)
        if pos is not None:
            dense_rates[pos] = rate
    return dense_rates


def mean_non_none(values: List[Optional[float]]) -> Optional[float]:
    """Average of numeric entries only; None values are ignored."""
    nums = [float(v) for v in values if isinstance(v, (float, int))]
    return (sum(nums) / len(nums)) if nums else None


def sanitize_filename(name: str) -> str:
    """Create a filesystem-safe filename stem."""
    text = re.sub(r"[^\w\-\. ]+", "_", name.strip())
    text = re.sub(r"\s+", "_", text).strip("._")
    return text or "session"


def plot_session_step_curves(session_blob: dict, metrics: dict, out_dir: Path) -> Optional[Path]:
    """Plot workflow-step dots in stacked subplots; return saved path."""
    # In sandboxed/non-interactive runs, direct matplotlib caches into workspace paths.
    out_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir = out_dir / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for --plot-step-curves. Install with: pip install matplotlib"
        ) from exc

    step_series = metrics.get("workflow_step_success_rate_series") or []
    step_series = [
        row
        for row in step_series
        if (row.get("success_rates") or []) and (row.get("success_rates") or [])[-1] is not None
    ]
    if not step_series:
        return None

    x = list(range(1, len(metrics.get("verifier_step_indices") or []) + 1))
    n_steps = len(step_series)
    fig_height = max(2.4 * n_steps, 5.5)
    fig, axes = plt.subplots(n_steps, 1, figsize=(12, fig_height), sharex=True)
    if n_steps == 1:
        axes = [axes]

    for ax, row in zip(axes, step_series):
        y_raw = row.get("success_rates") or []
        if len(y_raw) != len(x):
            continue
        points = [(xi, yi) for xi, yi in zip(x, y_raw) if isinstance(yi, (float, int))]
        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.scatter(xs, ys, s=18, alpha=0.9)
        label = str(row.get("step_name") or row.get("step_label") or "workflow_step")
        ax.set_title(label, loc="left", fontsize=10, pad=6)
        ax.set_ylabel("Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.grid(False)

    session_name = str(session_blob.get("name") or session_blob.get("uuid") or "session")
    fig.suptitle(f"Workflow Step Success Rates Throughout Actions\n{session_name}", y=0.995)
    axes[-1].set_xlabel("Action index among actions with verifier data")
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = out_dir / f"{sanitize_filename(session_name)}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def analyze_trajectory(traj: List[dict]) -> dict:
    per_step_rates: List[float] = []
    per_step_indices: List[int] = []
    per_step_counts: List[Tuple[int, int]] = []  # (successes, total) per contributing step
    all_bits: List[int] = []
    workflow_steps: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()

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

        if isinstance(wf, list):
            for wf_idx, node in enumerate(wf):
                if not isinstance(node, dict):
                    continue
                node_successes, node_total = count_successes(node.get("verifiers") or [])
                if node_total == 0:
                    continue
                step_name = workflow_step_name(node, wf_idx)
                entry = workflow_steps.get(wf_idx)
                if entry is None:
                    entry = {
                        "workflow_step_index": wf_idx,
                        "step_label": f"step {wf_idx + 1}",
                        "step_name": step_name,
                        "occurrences": 0,
                        "verifier_successes_total": 0,
                        "verifier_checks_total": 0,
                        "action_indices": [],
                        "success_rates": [],
                    }
                    workflow_steps[wf_idx] = entry
                entry["occurrences"] += 1
                entry["verifier_successes_total"] += node_successes
                entry["verifier_checks_total"] += node_total
                entry["action_indices"].append(idx)
                entry["success_rates"].append(node_successes / node_total)

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

    workflow_step_series_out: List[Dict[str, Any]] = []
    workflow_step_average_success_rates: List[Dict[str, Any]] = []
    for row in workflow_steps.values():
        dense_rates = make_dense_series(
            per_step_indices=per_step_indices,
            action_indices=row["action_indices"],
            success_rates=row["success_rates"],
        )
        workflow_step_series_out.append({**row, "success_rates": dense_rates})
        checks_total = row["verifier_checks_total"]
        workflow_step_average_success_rates.append(
            {
                "workflow_step_index": row["workflow_step_index"],
                "step_label": row["step_label"],
                "step_name": row["step_name"],
                "occurrences": row["occurrences"],
                "verifier_successes_total": row["verifier_successes_total"],
                "verifier_checks_total": checks_total,
                "average_success_rate": (
                    row["verifier_successes_total"] / checks_total if checks_total > 0 else None
                ),
            }
        )

    all_step_series_values: List[Optional[float]] = []
    for row in workflow_step_series_out:
        all_step_series_values.extend(row.get("success_rates") or [])
    overall_workflow_step_success_rate = mean_non_none(all_step_series_values)

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
        "workflow_step_average_success_rates": workflow_step_average_success_rates,
        "workflow_step_success_rate_series": workflow_step_series_out,
        "overall_workflow_step_success_rate": overall_workflow_step_success_rate,
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
    if rates:
        print("Per-action success rates:")
        print(f"{[round(r, 3) for r in rates]}")
    else:
        print("Per-action pass rates: (none)")
    mps = metrics["mean_per_step_verifier_pass_rate"]
    print(
        "Mean of per-step success rates:",
        f"{mps:.3f}" if mps is not None else "n/a",
    )
    print()
    print("Per-workflow-step, per-action success rates:")
    wf_series = metrics.get("workflow_step_success_rate_series") or []
    if wf_series:
        for i, row in enumerate(wf_series):
            values = [
                round(v, 3) if isinstance(v, (float, int)) else None
                for v in (row.get("success_rates") or [])
            ]
            step_title = row.get("step_name") or row.get("step_label")
            checks_total = row.get("verifier_checks_total") or 0
            successes_total = row.get("verifier_successes_total") or 0
            avg = (successes_total / checks_total) if checks_total else None
            avg_text = f"{avg:.3f}" if isinstance(avg, (float, int)) else "n/a"
            step_avg = f"{avg_text} across {row.get('occurrences')} updates"
            print(f"[step {i + 1}] {step_title}: {values} --> {step_avg}")
    else:
        print("(none)")
    overall_step_rate = metrics.get("overall_workflow_step_success_rate")
    print(
        "Overall workflow-step success rate (mean of all non-None per-action step rates):",
        f"{overall_step_rate:.3f}" if isinstance(overall_step_rate, (float, int)) else "n/a",
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
    parser.add_argument(
        "--plot-step-curves",
        action="store_true",
        help="Save one PNG per session with workflow-step success-rate curves over actions.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("scripts/plots"),
        help="Directory for --plot-step-curves output PNG files.",
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

    if args.plot_step_curves:
        try:
            for sess, metrics in zip(sessions, all_metrics):
                out_path = plot_session_step_curves(sess, metrics, args.plot_dir)
                if out_path is not None and not args.json:
                    print(f"Saved plot: {out_path}")
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            return 1

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
