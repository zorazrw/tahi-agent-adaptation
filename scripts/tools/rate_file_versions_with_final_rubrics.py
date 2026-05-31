#!/usr/bin/env python3
"""
For each workflow step (task) in the **final** trajectory workflow snapshot, take that step’s
last rubric text (verifier criteria in order), collect **unique** output-file snapshots from
**agent** trajectory steps in chronological order, then call an LM once per snapshot to label
pass/fail per criterion — matching the prompt contract in ``src/electron/libs/verifier-labeler.ts``.

Identical **(rubrics + file contents)** across tasks shares **one** LLM evaluation (no duplicate API calls).
Each version records ``trajectory_step_index`` (agent step where that snapshot first appears in the task
timeline), ``first_seen_trajectory_step_index`` (step where that eval key was first scored), and
``unique_eval_sequence`` (1-based id for each distinct eval key in the session).

Per-version **average success rate** (passing criteria / total × 100) is printed to **stderr** as ``xx.x%``
(so stdout JSON stays valid when redirected). Use ``--plot path.png`` to save a scatter plot:
x = trajectory step index, y = average success rate (``scatter_plot_data`` in the JSON mirrors the points).

Input JSON: a session object or an array of sessions (``export_task_sessions`` shape).
Supports legacy top-level ``trajectory`` and current ``task_units`` exports (``scripts/out.json``):
each unit’s ``environment`` is the end-of-turn state and is attached only to that unit’s last step.

Environment / CLI — same stack as ``scripts/induce.py``:
  ``python-dotenv`` loads ``--env-file`` or ``scripts/.env`` then ``./.env`` (use ``--dotenv-override`` to beat stale shell exports).
  Credentials via ``induce.resolve_anthropic_config()``: app ``api-config.json``, ``~/.claude/settings.json``, then ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN`` (+ optional base URL / model in env).
  ``--no-api-config`` / ``--no-claude-settings`` skip those resolution steps. LLM calls use the **Anthropic Python SDK** (``induce.anthropic_user_text`` → ``client.messages.create``), not raw HTTP.

A **401 invalid x-api-key** almost always means: wrong key type (not Anthropic), expired key, extra whitespace,
or **base URL does not match** the key (e.g. official key but proxy URL, or vice versa).

Examples:
  python scripts/rate_file_versions_with_final_rubrics.py \\
    --json scripts/out.json --session <uuid> --out ratings.json

  python scripts/rate_file_versions_with_final_rubrics.py \\
    -j scripts/out.json -s <uuid> --node-id 890a9d5a-b527-4c2e-b887-eaab2298dfcd --dry-run

  # Only leaf steps; auto-pick uuid when the file has a single session:
  python scripts/rate_file_versions_with_final_rubrics.py -j one_session.json --leaves-only --dry-run

  # JSON to ratings.json and a scatter plot (step index vs avg success %%):
  python scripts/rate_file_versions_with_final_rubrics.py -j out.json -s <uuid> -o ratings.json --plot

  # If ``-o ratings.json`` already exists, skip LLM and load that file (summary + ``--plot`` only); ``--force`` re-runs API.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

_scripts_dir = Path(__file__).resolve().parent.parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import induce  # noqa: E402
from dotenv import load_dotenv  # noqa: E402


def normalize_secret(s: str) -> str:
    """Strip whitespace and accidental CR/LF from pasted keys."""
    return s.strip().replace("\r", "").replace("\n", "")


def mask_secret(s: str, *, prefix: int = 7, suffix: int = 4) -> str:
    if len(s) <= prefix + suffix:
        return f"(len={len(s)})"
    return f"{s[:prefix]}…{s[-suffix:]} (len={len(s)})"


def merge_resolved_config(
    cfg: induce.ResolvedAnthropicConfig,
    *,
    cli_api_key: str | None,
    cli_base_url: str | None,
    cli_model: str | None,
) -> induce.ResolvedAnthropicConfig:
    """Apply CLI overrides on top of ``resolve_anthropic_config`` result."""
    key = (
        normalize_secret(cli_api_key)
        if (cli_api_key is not None and str(cli_api_key).strip())
        else normalize_secret(cfg.api_key)
    )
    if cli_base_url is not None and str(cli_base_url).strip():
        base = str(cli_base_url).strip().rstrip("/") or None
    else:
        base = cfg.base_url
    model = (str(cli_model).strip() if cli_model else "") or cfg.model
    return induce.ResolvedAnthropicConfig(api_key=key, base_url=base, model=model)


def load_sessions(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return [raw]
    raise ValueError("JSON root must be an array of sessions or a single session object")


def session_trajectory(session: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Flat per-step trajectory with ``actor`` and ``environment``.

    Legacy exports use top-level ``trajectory``. Current exports use ``task_units``: each unit
    has unit-level ``environment`` (end-of-turn) and step rows without per-step environment.
    The unit environment is attached only to that unit's last step so file snapshots match
    post-turn state and trajectory indices align with unit boundaries.
    """
    raw = session.get("trajectory")
    if isinstance(raw, list) and raw:
        return [x for x in raw if isinstance(x, dict)]

    units = session.get("task_units")
    if not isinstance(units, list):
        return []

    merged: list[dict[str, Any]] = []
    env_carry: dict[str, Any] | None = None
    for unit in units:
        if not isinstance(unit, dict):
            continue
        actor = str(unit.get("actor") or "user")
        unit_env = unit.get("environment")
        if isinstance(unit_env, dict):
            env_carry = unit_env
        traj = unit.get("trajectory")
        if not isinstance(traj, list):
            continue
        for step_idx, step in enumerate(traj):
            if not isinstance(step, dict):
                continue
            row = dict(step)
            row["actor"] = actor
            step_env = step.get("environment")
            if isinstance(step_env, dict):
                row["environment"] = step_env
                env_carry = step_env
            elif step_idx == len(traj) - 1 and env_carry is not None:
                row["environment"] = env_carry
            merged.append(row)
    return merged


def find_session(sessions: list[dict[str, Any]], session_id: str | None) -> dict[str, Any]:
    if session_id is None or session_id == "":
        if len(sessions) == 1:
            return sessions[0]
        raise SystemExit("Pass --session <uuid> when the JSON contains multiple sessions")
    for s in sessions:
        if s.get("uuid") == session_id:
            return s
    raise SystemExit(f"No session with uuid {session_id!r}")


def get_file_snapshot(file_field: Any, filename: str) -> str | None:
    """Return file text for ``filename`` from ``environment.file``, or None."""
    if file_field is None:
        return None
    if isinstance(file_field, dict):
        val = file_field.get(filename)
        return val if isinstance(val, str) else None
    if isinstance(file_field, list):
        for item in file_field:
            if not isinstance(item, dict):
                continue
            if item.get("path") == filename:
                c = item.get("content")
                return c if isinstance(c, str) else None
    return None


def messages_api_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return f"{base}/messages" if base.endswith("/v1") else f"{base}/v1/messages"


def parse_json_from_model_text(text: str) -> dict[str, Any]:
    """Extract first JSON object from model text (fenced or bare), matching verifier-labeler.ts."""
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    raw = (fence.group(1) if fence else text).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object in model response")
    return json.loads(raw[start : end + 1])


def verifier_strings(verifiers: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(verifiers, list):
        return out
    for v in verifiers:
        if isinstance(v, dict) and v.get("criterion") is not None:
            out.append(str(v["criterion"]))
        elif isinstance(v, str):
            out.append(v)
    return out


def flatten_workflow_nodes(tree: list[Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for node in tree:
        if not isinstance(node, dict):
            continue
        nodes.append(node)
        ch = node.get("children")
        if isinstance(ch, list):
            nodes.extend(flatten_workflow_nodes(ch))
    return nodes


def is_leaf_workflow_node(node: dict[str, Any]) -> bool:
    ch = node.get("children")
    return not isinstance(ch, list) or len(ch) == 0


def get_node_path(tree: list[dict[str, Any]], node_id: str) -> str:
    path: list[str] = []

    def find(nodes: list[dict[str, Any]]) -> bool:
        for node in nodes:
            path.append(str(node.get("description") or ""))
            if str(node.get("id") or "") == node_id:
                return True
            ch = node.get("children")
            if isinstance(ch, list) and find(ch):
                return True
            path.pop()
        return False

    find(tree)
    return " > ".join(path)


def last_workflow_in_trajectory(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    last: list[dict[str, Any]] | None = None
    for step in trajectory:
        if not isinstance(step, dict):
            continue
        env = step.get("environment")
        if not isinstance(env, dict):
            continue
        wf = env.get("workflow")
        if isinstance(wf, list) and wf:
            last = wf  # type: ignore[assignment]
    return last


def truncate_file_text(text: str, max_len: int = 14_000) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n... [truncated]"


def collect_unique_snapshots_for_node(
    trajectory: list[dict[str, Any]],
    output_files: list[str],
    *,
    agent_only: bool = True,
) -> list[dict[str, Any]]:
    """
    Chronological unique snapshots: each item is
    ``{ "trajectory_step_index", "file_blocks": [(rel, text), ...] }``
    keyed by tuple of contents for ``output_files`` order.
    """
    seen: set[tuple[str, ...]] = set()
    snapshots: list[dict[str, Any]] = []

    for step_idx, step in enumerate(trajectory):
        if not isinstance(step, dict):
            continue
        if agent_only and step.get("actor") != "agent":
            continue
        env = step.get("environment")
        if not isinstance(env, dict):
            continue
        file_field = env.get("file")
        parts: list[str] = []
        blocks: list[tuple[str, str]] = []
        skip_step = True
        for rel in output_files:
            content = get_file_snapshot(file_field, rel)
            if content is None:
                parts.append("")
                blocks.append((rel, ""))
            else:
                skip_step = False
                parts.append(content)
                blocks.append((rel, content))
        if skip_step:
            continue
        key = tuple(parts)
        if key in seen:
            continue
        seen.add(key)
        snapshots.append({"trajectory_step_index": step_idx, "file_blocks": blocks})
    return snapshots


def build_labeler_user_message(
    *,
    step_path: str,
    step_description: str,
    criteria: list[str],
    file_blocks: list[tuple[str, str]],
) -> str:
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(criteria))
    rendered_blocks: list[str] = []
    for rel, text in file_blocks:
        body = truncate_file_text(text) if text else "(file missing or empty at this snapshot)"
        rendered_blocks.append(f"### {rel}\n\n{body}")
    files_joined = "\n\n---\n\n".join(rendered_blocks) if rendered_blocks else "(no output files listed)"

    return "\n".join(
        [
            "You are an automated checker for a completed workflow step.",
            "Given verifier criteria and the current output files (below), decide whether each criterion is satisfied.",
            'Reply with ONLY a JSON object of this exact shape: {"results":[{"pass":true},{"pass":false},...]}',
            "The results array must have exactly one object per verifier line, in the same order (indices 0 .. n-1).",
            "pass: true means the criterion is satisfied; false means it is not.",
            "",
            f"Step path: {step_path}",
            f"Step task: {step_description}",
            "",
            "Verifier criteria (in order):",
            numbered,
            "",
            "Output files and contents:",
            files_joined,
        ]
    )


def call_verifier_llm(client: Any, model: str, user_text: str, *, max_tokens: int = 1024) -> str:
    """Anthropic Messages API via SDK — same pattern as ``induce._llm_text`` / ``induce.anthropic_user_text``."""
    try:
        return induce.anthropic_user_text(
            client,
            model,
            user_text,
            max_tokens=max_tokens,
            temperature=0.0,
        )
    except Exception as e:
        tail = str(e)
        code = getattr(e, "status_code", None)
        hint = ""
        if code == 401 or "401" in tail or "authentication" in tail.lower():
            hint = (
                "\n\nAuthentication hint: key and base URL must match (see ``induce.resolve_anthropic_config``). "
                "Try ``--debug-auth``, ``--no-claude-settings``, or ``--no-api-config``; use ``--dotenv-override`` if .env is ignored."
            )
        elif code == 400 and re.search(
            r"credit|billing|balance|purchase|Plans\s*&\s*Billing", tail, re.I
        ):
            hint = (
                "\n\nBilling hint: add credits at https://console.anthropic.com/ (Plans & Billing), or use another key/org."
            )
        raise RuntimeError(f"Verifier API error: {tail}{hint}") from e


def interpret_results(text: str, n: int) -> list[bool | None]:
    parsed = parse_json_from_model_text(text)
    results = parsed.get("results")
    if not isinstance(results, list):
        raise ValueError("Missing results array")
    out: list[bool | None] = [None] * n
    for i in range(min(n, len(results))):
        r = results[i]
        if isinstance(r, dict) and "pass" in r:
            out[i] = bool(r["pass"])
    return out


def evaluation_cache_key(criteria: list[str], file_blocks: list[tuple[str, str]]) -> str:
    """Hash rubrics + ordered path/content pairs so identical agent outputs are evaluated once."""
    h = hashlib.sha256()
    for c in criteria:
        h.update(c.encode("utf-8"))
        h.update(b"\0")
    for rel, content in file_blocks:
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(content.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def average_success_percent(passes: list[bool | None], n_criteria: int) -> float | None:
    if n_criteria <= 0:
        return None
    return 100.0 * sum(1 for p in passes if p is True) / n_criteria


def _attach_average_success(entry: dict[str, Any], n_criteria: int) -> None:
    lm = entry.get("lm")
    if not isinstance(lm, dict):
        return
    passes = lm.get("pass_per_criterion")
    if not isinstance(passes, list):
        return
    p = average_success_percent(passes, n_criteria)
    if p is not None:
        entry["average_success_pct"] = round(p, 1)


def print_average_success_summary(report: dict[str, Any]) -> None:
    """Human-readable per-version rates on stderr (xx.x% format)."""
    if report.get("error"):
        return
    uid = report.get("uuid") or ""
    name = report.get("name") or ""
    is_dry = bool(report.get("dry_run"))
    print("--- average success rate by solution version ---", file=sys.stderr)
    print(f"session: {uid}" + (f" — {name}" if name else ""), file=sys.stderr)
    stats = report.get("eval_cache_stats")
    if isinstance(stats, dict):
        print(
            f"eval cache: {stats.get('dedupe_hits', 0)} dedupe hits, "
            f"{stats.get('unique_agent_solutions_evaluated', 0)} unique (file+rubric) keys, "
            f"{stats.get('llm_calls', 0)} LLM calls",
            file=sys.stderr,
        )
    for task in report.get("tasks") or []:
        if not isinstance(task, dict):
            continue
        nid = str(task.get("node_id") or "")
        desc = str(task.get("description") or "")[:72]
        for ver in task.get("versions") or []:
            if not isinstance(ver, dict):
                continue
            vi = ver.get("version_index")
            si = ver.get("trajectory_step_index")
            pct = ver.get("average_success_pct")
            hit = ver.get("eval_cache_hit")
            suffix = " (deduplicated)" if hit else ""
            if isinstance(pct, (int, float)):
                print(f"  [{nid}] {desc} | v{vi} @ traj {si}: {float(pct):.1f}%{suffix}", file=sys.stderr)
            elif ver.get("error"):
                err = str(ver.get("error") or "")[:100]
                print(f"  [{nid}] {desc} | v{vi} @ traj {si}: N/A — {err}", file=sys.stderr)
            elif is_dry:
                print(f"  [{nid}] {desc} | v{vi} @ traj {si}: (dry-run, no score)", file=sys.stderr)
    print("--- end summary ---", file=sys.stderr)


def build_scatter_plot_data(report_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows for scatter: step index vs average success rate (only scored versions)."""
    rows: list[dict[str, Any]] = []
    for task in report_tasks:
        if not isinstance(task, dict):
            continue
        nid = task.get("node_id")
        desc = task.get("description")
        for ver in task.get("versions") or []:
            if not isinstance(ver, dict):
                continue
            step = ver.get("trajectory_step_index")
            pct = ver.get("average_success_pct")
            if not isinstance(step, int) or not isinstance(pct, (int, float)):
                continue
            rows.append(
                {
                    "trajectory_step_index": step,
                    "average_success_pct": float(pct),
                    "node_id": nid,
                    "task_description": desc,
                    "version_index_within_task": ver.get("version_index"),
                    "unique_eval_sequence": ver.get("unique_eval_sequence"),
                    "first_seen_trajectory_step_index": ver.get("first_seen_trajectory_step_index"),
                    "eval_cache_hit": ver.get("eval_cache_hit"),
                }
            )
    rows.sort(key=lambda r: (r["trajectory_step_index"], str(r.get("node_id") or "")))
    return rows


def try_load_existing_ratings_file(
    path: Path,
    *,
    expected_session_uuid: str | None,
) -> dict[str, Any] | None:
    """
    Load a prior ratings JSON. Returns None if missing, invalid, or uuid mismatch with ``--session``.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    tasks = raw.get("tasks")
    if not isinstance(tasks, list):
        return None
    if expected_session_uuid and expected_session_uuid.strip():
        file_uuid = raw.get("uuid")
        if file_uuid is not None and str(file_uuid) != str(expected_session_uuid).strip():
            print(
                f"Existing ratings {path} uuid={file_uuid!r} does not match --session {expected_session_uuid!r}; "
                "re-running LLM.",
                file=sys.stderr,
            )
            return None
    return raw


def save_success_rate_scatter_plot(report: dict[str, Any], out_path: Path) -> Path | None:
    """
    Scatter plot: x = trajectory step index, y = average success rate (%).
    Returns path written, or None if nothing to plot.
    """
    if report.get("error"):
        return None
    pts = report.get("scatter_plot_data") or build_scatter_plot_data(report.get("tasks") or [])
    if len(pts) < 1:
        print("Scatter plot skipped: no scored points.", file=sys.stderr)
        return None

    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mpl_dir = out_path.parent / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

    # Matplotlib logs INFO while scanning system fonts (harmless on macOS); hide noise.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:
        print(f"Scatter plot skipped: matplotlib not available ({exc}).", file=sys.stderr)
        return None

    xs = [p["trajectory_step_index"] for p in pts]
    ys = [p["average_success_pct"] for p in pts]
    node_ids = [str(p.get("node_id") or "") for p in pts]
    unique_nodes: list[str] = list(dict.fromkeys(node_ids))
    cmap = plt.get_cmap("tab10")
    node_color = {n: cmap(i % 10) for i, n in enumerate(unique_nodes)}
    colors = [node_color[n] for n in node_ids]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(xs, ys, c=colors, s=36, alpha=0.75, edgecolors="white", linewidths=0.4)
    ax.set_xlabel("Trajectory step index (agent snapshot)")
    ax.set_ylabel("Average success rate (%)")
    title = report.get("name") or report.get("uuid") or "Session"
    ax.set_title(f"Verifier success rate vs step — {title}")
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.3)
    if len(unique_nodes) <= 12:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=node_color[n],
                markersize=8,
                label=(n[:16] + "…") if len(n) > 16 else n,
            )
            for n in unique_nodes
        ]
        ax.legend(handles=handles, title="node_id", loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Scatter plot saved: {out_path}", file=sys.stderr)
    return out_path


def rate_session(
    session: dict[str, Any],
    *,
    client: Any | None,
    model: str,
    credential_meta: dict[str, str] | None,
    node_id_filter: str | None,
    leaves_only: bool,
    dry_run: bool,
    max_versions: int | None,
) -> dict[str, Any]:
    traj = session_trajectory(session)
    if not traj:
        raise ValueError("session has no trajectory or task_units")

    wf = last_workflow_in_trajectory(traj)
    if not wf:
        return {
            "uuid": session.get("uuid"),
            "name": session.get("name"),
            "error": "no workflow found in trajectory",
            "tasks": [],
            "scatter_plot_data": [],
        }

    nodes = flatten_workflow_nodes(wf)
    report_tasks: list[dict[str, Any]] = []
    eval_cache: dict[str, dict[str, Any]] = {}
    cache_hits = 0
    cache_misses = 0
    llm_calls = 0
    unique_eval_seq = 0

    for node in nodes:
        nid = str(node.get("id") or "")
        if leaves_only and not is_leaf_workflow_node(node):
            continue
        if node_id_filter and nid != node_id_filter:
            continue
        output_files = node.get("outputFiles") or []
        if not isinstance(output_files, list) or not output_files:
            continue
        output_files = [str(p) for p in output_files if p]
        criteria = verifier_strings(node.get("verifiers"))
        if not criteria:
            continue

        desc = str(node.get("description") or "")
        step_path = get_node_path(wf, nid)
        snapshots = collect_unique_snapshots_for_node(traj, output_files, agent_only=True)
        if max_versions is not None:
            snapshots = snapshots[:max_versions]

        version_results: list[dict[str, Any]] = []
        for v_idx, snap in enumerate(snapshots):
            file_blocks: list[tuple[str, str]] = list(snap["file_blocks"])
            ek = evaluation_cache_key(criteria, file_blocks)

            if ek in eval_cache:
                cache_hits += 1
                cached = eval_cache[ek]
                entry = {
                    "version_index": v_idx,
                    "trajectory_step_index": int(snap["trajectory_step_index"]),
                    "eval_cache_key_prefix": ek[:16],
                    "eval_cache_hit": True,
                    "unique_eval_sequence": cached.get("unique_eval_sequence"),
                    "first_seen_trajectory_step_index": cached.get("first_seen_trajectory_step_index"),
                    "lm": None,
                    "error": None,
                }
                if dry_run:
                    entry["prompt"] = cached.get("prompt")
                else:
                    err = cached.get("error")
                    if err:
                        entry["error"] = err
                    else:
                        entry["lm"] = copy.deepcopy(cached.get("lm"))
                _attach_average_success(entry, len(criteria))
                version_results.append(entry)
                continue

            cache_misses += 1
            unique_eval_seq += 1
            seq = unique_eval_seq
            first_step = int(snap["trajectory_step_index"])
            msg = build_labeler_user_message(
                step_path=step_path,
                step_description=desc,
                criteria=criteria,
                file_blocks=file_blocks,
            )
            entry = {
                "version_index": v_idx,
                "trajectory_step_index": first_step,
                "eval_cache_key_prefix": ek[:16],
                "eval_cache_hit": False,
                "unique_eval_sequence": seq,
                "first_seen_trajectory_step_index": first_step,
                "lm": None,
                "error": None,
            }
            cache_payload = {
                "unique_eval_sequence": seq,
                "first_seen_trajectory_step_index": first_step,
            }
            if dry_run:
                entry["prompt"] = msg
                eval_cache[ek] = {**cache_payload, "prompt": msg, "lm": None, "error": None}
                version_results.append(entry)
                continue
            if client is None:
                entry["error"] = "missing Anthropic client (credentials)"
                eval_cache[ek] = {**cache_payload, "prompt": None, "lm": None, "error": entry["error"]}
                version_results.append(entry)
                continue
            try:
                llm_calls += 1
                raw = call_verifier_llm(client, model, msg)
                passes = interpret_results(raw, len(criteria))
                entry["lm"] = {
                    "raw_text": raw,
                    "pass_per_criterion": passes,
                    "criteria": list(criteria),
                }
                eval_cache[ek] = {
                    **cache_payload,
                    "prompt": None,
                    "lm": copy.deepcopy(entry["lm"]),
                    "error": None,
                }
            except Exception as exc:  # noqa: BLE001 — surface per-version errors in report
                entry["error"] = str(exc)
                eval_cache[ek] = {**cache_payload, "prompt": None, "lm": None, "error": entry["error"]}
            _attach_average_success(entry, len(criteria))
            version_results.append(entry)

        report_tasks.append(
            {
                "node_id": nid,
                "description": desc,
                "output_files": output_files,
                "final_rubrics": criteria,
                "unique_agent_snapshots": len(snapshots),
                "versions": version_results,
            }
        )

    scatter_plot_data = build_scatter_plot_data(report_tasks)
    out: dict[str, Any] = {
        "uuid": session.get("uuid"),
        "name": session.get("name"),
        "dry_run": dry_run,
        "tasks": report_tasks,
        "scatter_plot_data": scatter_plot_data,
        "eval_cache_stats": {
            "dedupe_hits": cache_hits,
            "unique_agent_solutions_evaluated": cache_misses,
            "llm_calls": llm_calls,
            "cached_keys": len(eval_cache),
            "unique_eval_sequences": unique_eval_seq,
        },
    }
    if credential_meta is not None:
        out["auth_resolved"] = {**credential_meta, "llm": "induce.anthropic_user_text (Anthropic SDK)"}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="LM-rate file versions against final workflow rubrics.")
    p.add_argument("--json", "-j", type=Path, required=True, help="Session JSON path")
    p.add_argument(
        "--session",
        "-s",
        type=str,
        default=None,
        help="Session uuid (optional if JSON has exactly one session)",
    )
    p.add_argument(
        "--leaves-only",
        action="store_true",
        help="Only workflow nodes with no children (often matches leaf tasks)",
    )
    p.add_argument("--out", "-o", type=Path, default=None, help="Write JSON report to this path")
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run LLM even when -o file already exists (default: load existing ratings and skip API)",
    )
    p.add_argument("--node-id", type=str, default=None, help="Only rate this workflow node id")
    p.add_argument("--max-versions", type=int, default=None, help="Cap snapshots per task")
    p.add_argument("--dry-run", action="store_true", help="Build prompts only; do not call API")
    p.add_argument("--api-key", type=str, default=None, help="Override API key (else env / ~/.claude/settings.json)")
    p.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Load this dotenv file first (default: scripts/.env then ./.env if this flag omitted)",
    )
    p.add_argument(
        "--dotenv-override",
        action="store_true",
        help="Dotenv values override existing environment variables (fixes stale shell exports)",
    )
    p.add_argument(
        "--no-claude-settings",
        action="store_true",
        help="Skip ~/.claude/settings.json in induce.resolve_anthropic_config",
    )
    p.add_argument(
        "--no-api-config",
        action="store_true",
        help="Skip app api-config.json paths in induce.resolve_anthropic_config",
    )
    p.add_argument(
        "--debug-auth",
        action="store_true",
        help="Print masked key, base URL, model to stderr (induce + CLI merge)",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Anthropic model id (else ANTHROPIC_MODEL or ~/.claude/settings.json)",
    )
    p.add_argument(
        "--base-url",
        default=None,
        help="Anthropic API base URL (else ANTHROPIC_BASE_URL or ~/.claude/settings.json)",
    )
    p.add_argument(
        "--plot",
        nargs="?",
        const="__default__",
        default=None,
        metavar="PNG",
        help="Save scatter plot (step index vs avg success %%). "
        "Optional path; default: sibling of --out with _scatter.png or scripts/plots/rubric_success_scatter.png",
    )
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    report: dict[str, Any] | None = None
    loaded_from_ratings_file = False

    if (
        args.out
        and args.out.is_file()
        and not args.force
    ):
        report = try_load_existing_ratings_file(
            args.out.resolve(),
            expected_session_uuid=args.session,
        )
        if report is not None:
            loaded_from_ratings_file = True
            print(
                f"Using existing ratings file (no LLM): {args.out.resolve()}",
                file=sys.stderr,
            )
            if not report.get("scatter_plot_data"):
                report["scatter_plot_data"] = build_scatter_plot_data(report.get("tasks") or [])

    if report is None:
        if args.env_file is not None:
            load_dotenv(dotenv_path=args.env_file, override=args.dotenv_override)
        else:
            load_dotenv(dotenv_path=script_dir / ".env", override=args.dotenv_override)
            load_dotenv(dotenv_path=Path.cwd() / ".env", override=args.dotenv_override)

        cfg: induce.ResolvedAnthropicConfig | None
        try:
            cfg = induce.resolve_anthropic_config(
                skip_api_config=args.no_api_config,
                skip_claude_settings=args.no_claude_settings,
            )
        except induce.AnthropicConfigError as e:
            if args.dry_run:
                cfg = None
            else:
                raise SystemExit(str(e)) from e

        client: Any | None = None
        cred_meta: dict[str, str] | None = None
        model: str

        if cfg is not None:
            cfg = merge_resolved_config(
                cfg,
                cli_api_key=args.api_key,
                cli_base_url=args.base_url,
                cli_model=args.model,
            )
            model = cfg.model
            cred_meta = {
                "resolver": "induce.resolve_anthropic_config + CLI merge",
                "skip_api_config": str(args.no_api_config),
                "skip_claude_settings": str(args.no_claude_settings),
                "base_url_effective": cfg.base_url or "(Anthropic SDK default)",
                "model": cfg.model,
            }
            if args.debug_auth:
                bu = cfg.base_url or "https://api.anthropic.com"
                print("--- rate_file_versions auth debug (induce + SDK) ---", file=sys.stderr)
                print(f"  messages URL: {messages_api_url(bu)}", file=sys.stderr)
                print(f"  model: {cfg.model!r}", file=sys.stderr)
                print(f"  base_url: {cfg.base_url!r}", file=sys.stderr)
                print(f"  api_key: {mask_secret(cfg.api_key)}", file=sys.stderr)
                print("--- end auth debug ---", file=sys.stderr)
            if not args.dry_run:
                if cfg.api_key.startswith("tml-") or cfg.api_key.startswith("tml_"):
                    raise SystemExit(
                        "Resolved key looks like TINKER_API_KEY. Set ANTHROPIC_API_KEY for Anthropic (see induce.py)."
                    )
                client = induce.make_anthropic_client(cfg)
        else:
            model = (args.model or "").strip() or induce.DEFAULT_MODEL
            cred_meta = {"resolver": "dry-run (induce.resolve_anthropic_config unavailable)"}
            if args.debug_auth:
                print("--- auth debug: dry-run, no induce config ---", file=sys.stderr)

        sessions = load_sessions(args.json)
        session = find_session(sessions, args.session)

        report = rate_session(
            session,
            client=client,
            model=model,
            credential_meta=cred_meta,
            node_id_filter=args.node_id,
            leaves_only=args.leaves_only,
            dry_run=args.dry_run,
            max_versions=args.max_versions,
        )

    assert report is not None
    print_average_success_summary(report)

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        if not loaded_from_ratings_file:
            args.out.write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")

    if args.plot is not None:
        if args.plot == "__default__":
            if args.out:
                plot_path = args.out.with_name(args.out.stem + "_scatter.png")
            else:
                plot_path = script_dir / "plots" / "rubric_success_scatter.png"
        else:
            plot_path = Path(args.plot)
        save_success_rate_scatter_plot(report, plot_path)


if __name__ == "__main__":
    main()
