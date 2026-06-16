#!/usr/bin/env python3
"""
LM-rate unique file snapshots for **one deliverable per session**: the output file from the
latest workflow step that declares ``outputFiles`` (if that step lists several files, the last
listed file is used). Every version is graded against the **last** workflow step's verifiers.

Collects snapshots from agent steps **and** user/human-edit steps (e.g. agent ``abstract.md``
then a human-edited ``abstract.md``). Supports ``trajectory``, ``task_units`` (per-step
``trajectory``), and weight exports (``agent_trajectories`` + ``workflow_tree_final``).
Identical (rubrics + file contents) share one LLM call. Use ``--endpoints-only`` for first/last
snapshot only; ``--exported-status`` for human/exported marks (no LM). Per-version success %% is
summarized on stderr; ``--plot`` writes a scatter PNG.

Examples:
  python scripts/tools/rate_file_versions.py -j out.json -s <uuid> -o ratings.json --plot
  python scripts/tools/rate_file_versions.py -j out.json -s <uuid> --endpoints-only --dry-run
  python scripts/tools/rate_file_versions.py -j out.json -s <uuid> --exported-status
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterator

_scripts = Path(__file__).resolve().parent.parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

import induce  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from verifier_label_prompt import (  # noqa: E402
    format_numbered_lines,
    results_array_instructions,
    results_length_retry_hint,
    validate_results_length,
)

_TRUNCATE_LEN = 14_000
DEFAULT_MODEL = "claude-haiku-4-5"


# --- Session I/O ---


def load_sessions(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        return [raw]
    raise ValueError("JSON root must be an array of sessions or a single session object")


def resolve_session(sessions: list[dict[str, Any]], session_id: str | None) -> dict[str, Any]:
    if not sessions:
        raise SystemExit("No sessions in JSON")
    if session_id and str(session_id).strip():
        sid = str(session_id).strip()
        for s in sessions:
            if s.get("uuid") == sid:
                return s
        raise SystemExit(f"No session with uuid {sid!r}")
    if len(sessions) == 1:
        return sessions[0]
    raise SystemExit("Pass --session <uuid> when the JSON contains multiple sessions")


def _is_weight_task_units(session: dict[str, Any]) -> bool:
    units = session.get("task_units")
    if not isinstance(units, list):
        return False
    return any(isinstance(u, dict) and isinstance(u.get("agent_trajectories"), list) for u in units)


def workflow_tree_final(session: dict[str, Any]) -> list[dict[str, Any]] | None:
    for unit in session.get("task_units") or []:
        if not isinstance(unit, dict) or unit.get("intent") != "planning":
            continue
        wf = unit.get("workflow_tree_final")
        if isinstance(wf, list) and wf:
            return [n for n in wf if isinstance(n, dict)]
    return None


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _apply_messages_to_files(files: dict[str, str], messages: list[Any]) -> None:
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function")
            if not isinstance(fn, dict):
                continue
            name = str(fn.get("name") or "")
            args = _parse_tool_arguments(fn.get("arguments"))
            path = args.get("path")
            if not isinstance(path, str) or not path.strip():
                continue
            path = path.strip()
            if name == "write":
                content = args.get("content")
                if isinstance(content, str):
                    files[path] = content
            elif name == "edit" and path in files:
                text = files[path]
                for ed in args.get("edits") or []:
                    if not isinstance(ed, dict):
                        continue
                    old, new = ed.get("oldText"), ed.get("newText")
                    if isinstance(old, str) and isinstance(new, str):
                        text = text.replace(old, new, 1)
                files[path] = text


def _files_after_round(files: dict[str, str], rnd: dict[str, Any]) -> dict[str, str]:
    out = dict(files)
    for f in rnd.get("output_files") or []:
        if not isinstance(f, dict):
            continue
        path, content = f.get("path"), f.get("content")
        if isinstance(path, str) and path.strip() and isinstance(content, str) and content:
            out[path.strip()] = content
    _apply_messages_to_files(out, rnd.get("messages") or [])
    return out


def session_trajectory_weight(session: dict[str, Any]) -> list[dict[str, Any]]:
    """Synthetic agent steps from weight-format ``agent_trajectories`` rounds."""
    wf = workflow_tree_final(session) or []
    merged: list[dict[str, Any]] = []
    files: dict[str, str] = {}
    for unit in session.get("task_units") or []:
        if not isinstance(unit, dict) or unit.get("intent") == "planning":
            continue
        for rnd in unit.get("agent_trajectories") or []:
            if not isinstance(rnd, dict):
                continue
            files = _files_after_round(files, rnd)
            merged.append({"actor": "agent", "environment": {"file": dict(files), "workflow": wf}})
    return merged


def session_trajectory(session: dict[str, Any]) -> list[dict[str, Any]]:
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
        if isinstance(unit.get("environment"), dict):
            env_carry = unit["environment"]
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

    if merged:
        return merged
    if _is_weight_task_units(session):
        return session_trajectory_weight(session)
    return []


def session_meta(session: dict[str, Any]) -> dict[str, Any]:
    return {"uuid": session.get("uuid"), "name": session.get("name")}


def workflow_from_session(session: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    traj = session_trajectory(session)
    if not traj:
        raise ValueError("session has no trajectory or task_units")
    wf: list[dict[str, Any]] | None = None
    for step in traj:
        env = step.get("environment")
        if isinstance(env, dict):
            w = env.get("workflow")
            if isinstance(w, list) and w:
                wf = w
    if not wf:
        wf = workflow_tree_final(session)
    return traj, wf


# --- Workflow / files ---


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


def file_blocks_from_env(env: dict[str, Any]) -> list[tuple[str, str]]:
    ff = env.get("file") or env.get("files")
    blocks: list[tuple[str, str]] = []
    if isinstance(ff, dict):
        blocks = [(p, c) for p, c in ff.items() if isinstance(p, str) and isinstance(c, str) and c]
    elif isinstance(ff, list):
        for item in ff:
            if isinstance(item, dict):
                p, c = item.get("path"), item.get("content")
                if isinstance(p, str) and isinstance(c, str) and c:
                    blocks.append((p, c))
    blocks.sort(key=lambda x: x[0])
    return blocks


def file_snapshot(file_field: Any, filename: str) -> str | None:
    if isinstance(file_field, dict):
        val = file_field.get(filename)
        return val if isinstance(val, str) else None
    if isinstance(file_field, list):
        for item in file_field:
            if isinstance(item, dict) and item.get("path") == filename:
                c = item.get("content")
                return c if isinstance(c, str) else None
    return None


def parse_verifiers(verifiers: Any) -> list[tuple[str, str, bool]]:
    """(criterion, status, pass) per verifier entry."""
    if not isinstance(verifiers, list):
        return []
    out: list[tuple[str, str, bool]] = []
    for v in verifiers:
        if isinstance(v, dict) and v.get("criterion") is not None:
            c = str(v["criterion"]).strip()
            if c:
                st = v.get("status")
                if st is True:
                    out.append((c, "success", True))
                elif st is False:
                    out.append((c, "fail", False))
                else:
                    status = str(st or "")
                    out.append((c, status, status == "success"))
        elif isinstance(v, str) and v.strip():
            out.append((v.strip(), "", False))
    return out


def criteria_from_verifiers(verifiers: Any) -> list[str]:
    return [c for c, _, _ in parse_verifiers(verifiers)]


def last_workflow_node(wf: list[dict[str, Any]]) -> dict[str, Any] | None:
    nodes = flatten_workflow_nodes(wf)
    return nodes[-1] if nodes else None


def target_evaluation_node_and_file(wf: list[dict[str, Any]]) -> tuple[dict[str, Any], str] | None:
    """Last workflow node with output files and the single deliverable file to grade."""
    target_node: dict[str, Any] | None = None
    target_file: str | None = None
    for node in flatten_workflow_nodes(wf):
        files = [str(p) for p in (node.get("outputFiles") or []) if p]
        if files:
            target_node = node
            target_file = files[-1]
    if target_node is None or not target_file:
        return None
    return target_node, target_file


def last_workflow_rubrics(wf: list[dict[str, Any]]) -> list[str]:
    """Verifier criteria from the final workflow step (used for all file-version grading)."""
    node = last_workflow_node(wf)
    return criteria_from_verifiers(node.get("verifiers")) if node else []


def last_workflow_verifier_entries(wf: list[dict[str, Any]]) -> list[tuple[str, str, bool]]:
    node = last_workflow_node(wf)
    return parse_verifiers(node.get("verifiers")) if node else []


def get_node_path(tree: list[dict[str, Any]], node_id: str) -> str:
    path: list[str] = []

    def walk(nodes: list[dict[str, Any]]) -> bool:
        for node in nodes:
            path.append(str(node.get("description") or ""))
            if str(node.get("id") or "") == node_id:
                return True
            ch = node.get("children")
            if isinstance(ch, list) and walk(ch):
                return True
            path.pop()
        return False

    walk(tree)
    return " > ".join(path)


def iter_agent_steps(traj: list[dict[str, Any]]) -> Iterator[tuple[int, dict[str, Any]]]:
    for step_idx, step in enumerate(traj):
        if isinstance(step, dict) and step.get("actor") == "agent":
            yield step_idx, step


def iter_file_environment_steps(traj: list[dict[str, Any]]) -> Iterator[tuple[int, dict[str, Any]]]:
    """All trajectory steps whose environment carries a file snapshot (agent or user/human edits)."""
    for step_idx, step in enumerate(traj):
        if not isinstance(step, dict):
            continue
        env = step.get("environment")
        if isinstance(env, dict) and env.get("file") is not None:
            yield step_idx, step


def collect_unique_snapshots(
    traj: list[dict[str, Any]], output_files: list[str]
) -> list[dict[str, Any]]:
    seen: set[tuple[str, ...]] = set()
    snapshots: list[dict[str, Any]] = []
    for step_idx, step in iter_file_environment_steps(traj):
        env = step.get("environment")
        if not isinstance(env, dict):
            continue
        blocks = [(rel, file_snapshot(env.get("file"), rel) or "") for rel in output_files]
        if not any(text for _, text in blocks):
            continue
        key = tuple(text for _, text in blocks)
        if key in seen:
            continue
        seen.add(key)
        snapshots.append(
            {
                "trajectory_step_index": step_idx,
                "actor": str(step.get("actor") or ""),
                "file_blocks": blocks,
            }
        )
    return snapshots


def last_agent_file_snapshot(traj: list[dict[str, Any]]) -> tuple[list[tuple[str, str]], int | None]:
    blocks: list[tuple[str, str]] = []
    idx: int | None = None
    for step_idx, step in iter_agent_steps(traj):
        env = step.get("environment")
        if isinstance(env, dict):
            b = file_blocks_from_env(env)
            if b:
                blocks, idx = b, step_idx
    return blocks, idx


def last_file_snapshot(traj: list[dict[str, Any]], filename: str) -> tuple[str | None, int | None]:
    text: str | None = None
    idx: int | None = None
    for step_idx, step in iter_file_environment_steps(traj):
        env = step.get("environment")
        if not isinstance(env, dict):
            continue
        content = file_snapshot(env.get("file"), filename)
        if isinstance(content, str) and content:
            text, idx = content, step_idx
    return text, idx


# --- LM labeling ---


def truncate_file_text(text: str, max_len: int = _TRUNCATE_LEN) -> str:
    return text if len(text) <= max_len else text[:max_len] + "\n... [truncated]"


def build_labeler_user_message(
    *, step_path: str, step_description: str, criteria: list[str], file_blocks: list[tuple[str, str]]
) -> str:
    n = len(criteria)
    numbered = format_numbered_lines(criteria)
    rendered = [
        f"### {rel}\n\n{truncate_file_text(text) if text else '(file missing or empty at this snapshot)'}"
        for rel, text in file_blocks
    ]
    return "\n".join(
        [
            "You are an automated checker for a completed workflow step.",
            "Given verifier criteria and the current output files (below), decide whether each criterion is satisfied.",
            'Reply with ONLY a JSON object of this exact shape: {"results":[{"pass":true},{"pass":false},...]}',
            results_array_instructions(count=n, item_word="criterion"),
            "pass: true means the criterion is satisfied; false means it is not.",
            "",
            f"Step path: {step_path}",
            f"Step task: {step_description}",
            "",
            "Verifier criteria (in order):",
            numbered,
            "",
            "Output files and contents:",
            "\n\n---\n\n".join(rendered) if rendered else "(no output files listed)",
        ]
    )


def parse_json_from_model_text(text: str) -> dict[str, Any]:
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    raw = (fence.group(1) if fence else text).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object in model response")
    return json.loads(raw[start : end + 1])


def interpret_results(text: str, n: int) -> list[bool | None]:
    arr = parse_json_from_model_text(text).get("results")
    if not isinstance(arr, list):
        raise ValueError("Missing results array")
    validate_results_length(arr, n)
    out: list[bool | None] = [None] * n
    for i, row in enumerate(arr):
        if isinstance(row, dict) and "pass" in row:
            out[i] = bool(row["pass"])
    return out


def _call_verifier_llm_graded(
    client: Any, model: str, msg: str, n: int, *, stats: dict[str, int]
) -> tuple[str, list[bool | None]]:
    """Call verifier LLM; retry once if results length does not match criteria count."""
    stats["llm_calls"] += 1
    raw = call_verifier_llm(client, model, msg)
    try:
        return raw, interpret_results(raw, n)
    except ValueError as exc:
        if "Expected exactly" not in str(exc):
            raise
        stats["llm_calls"] += 1
        raw = call_verifier_llm(client, model, f"{msg}\n\n{results_length_retry_hint(n)}")
        return raw, interpret_results(raw, n)


def call_verifier_llm(client: Any, model: str, user_text: str, *, max_tokens: int = 1024) -> str:
    try:
        return induce.anthropic_user_text(client, model, user_text, max_tokens=max_tokens, temperature=0.0)
    except Exception as e:
        tail = str(e)
        code = getattr(e, "status_code", None)
        hint = ""
        if code == 401 or "401" in tail or "authentication" in tail.lower():
            hint = (
                "\n\nAuthentication hint: save Anthropic in app Settings (pi-agent/auth.json) or set "
                "ANTHROPIC_API_KEY. Try --debug-auth or --no-api-config for env-only."
            )
        elif code == 400 and re.search(r"credit|billing|balance|purchase|Plans\s*&\s*Billing", tail, re.I):
            hint = "\n\nBilling hint: add credits at https://console.anthropic.com/ (Plans & Billing)."
        raise RuntimeError(f"Verifier API error: {tail}{hint}") from e


def evaluation_cache_key(criteria: list[str], file_blocks: list[tuple[str, str]]) -> str:
    h = hashlib.sha256()
    for c in criteria:
        h.update(c.encode())
        h.update(b"\0")
    for rel, content in file_blocks:
        h.update(rel.encode())
        h.update(b"\0")
        h.update(content.encode())
        h.update(b"\0")
    return h.hexdigest()


def average_success_percent(passes: list[bool | None], n_criteria: int) -> float | None:
    if n_criteria <= 0:
        return None
    return 100.0 * sum(1 for p in passes if p is True) / n_criteria


def with_average_success(entry: dict[str, Any], n_criteria: int) -> dict[str, Any]:
    lm = entry.get("lm")
    if isinstance(lm, dict) and isinstance(lm.get("pass_per_criterion"), list):
        pct = average_success_percent(lm["pass_per_criterion"], n_criteria)
        if pct is not None:
            entry["average_success_pct"] = round(pct, 1)
    return entry


def _score_one_snapshot(
    *,
    v_idx: int,
    snap: dict[str, Any],
    criteria: list[str],
    step_path: str,
    desc: str,
    eval_cache: dict[str, dict[str, Any]],
    dry_run: bool,
    client: Any | None,
    model: str,
    stats: dict[str, int],
) -> dict[str, Any]:
    file_blocks = list(snap["file_blocks"])
    ek = evaluation_cache_key(criteria, file_blocks)
    step = int(snap["trajectory_step_index"])
    n = len(criteria)

    actor = str(snap.get("actor") or "")

    def base_entry(*, hit: bool, seq: int | None, cached: dict[str, Any] | None) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "version_index": v_idx,
            "trajectory_step_index": step,
            "actor": actor or None,
            "eval_cache_key_prefix": ek[:16],
            "eval_cache_hit": hit,
            "unique_eval_sequence": (cached or {}).get("unique_eval_sequence", seq),
            "first_seen_trajectory_step_index": (cached or {}).get("first_seen_trajectory_step_index", step),
            "lm": None,
            "error": (cached or {}).get("error") if cached else None,
        }
        if dry_run and cached:
            entry["prompt"] = cached.get("prompt")
        elif cached and not entry["error"]:
            lm = cached.get("lm")
            entry["lm"] = dict(lm) if isinstance(lm, dict) else None
        return with_average_success(entry, n)

    if ek in eval_cache:
        stats["hits"] += 1
        return base_entry(hit=True, seq=None, cached=eval_cache[ek])

    stats["misses"] += 1
    stats["unique_seq"] += 1
    seq = stats["unique_seq"]
    msg = build_labeler_user_message(step_path=step_path, step_description=desc, criteria=criteria, file_blocks=file_blocks)
    payload = {"unique_eval_sequence": seq, "first_seen_trajectory_step_index": step}

    if dry_run:
        eval_cache[ek] = {**payload, "prompt": msg, "lm": None, "error": None}
        entry = base_entry(hit=False, seq=seq, cached=eval_cache[ek])
        entry["prompt"] = msg
        return entry

    if client is None:
        eval_cache[ek] = {**payload, "prompt": None, "lm": None, "error": "missing Anthropic client (credentials)"}
        entry = base_entry(hit=False, seq=seq, cached=eval_cache[ek])
        entry["error"] = eval_cache[ek]["error"]
        return entry

    try:
        raw, passes = _call_verifier_llm_graded(client, model, msg, n, stats=stats)
        lm = {"raw_text": raw, "pass_per_criterion": passes, "criteria": list(criteria)}
        eval_cache[ek] = {**payload, "prompt": None, "lm": lm, "error": None}
        entry = base_entry(hit=False, seq=seq, cached=eval_cache[ek])
        entry["lm"] = lm
        return entry
    except Exception as exc:  # noqa: BLE001
        eval_cache[ek] = {**payload, "prompt": None, "lm": None, "error": str(exc)}
        entry = base_entry(hit=False, seq=seq, cached=eval_cache[ek])
        entry["error"] = str(exc)
        return entry


# --- Rating ---


def rate_session(
    session: dict[str, Any],
    *,
    client: Any | None,
    model: str,
    credential_meta: dict[str, str] | None,
    dry_run: bool,
    endpoints_only: bool,
) -> dict[str, Any]:
    traj, wf = workflow_from_session(session)
    if not wf:
        return {**session_meta(session), "error": "no workflow found in trajectory", "tasks": [], "scatter_plot_data": []}

    final_criteria = last_workflow_rubrics(wf)
    if not final_criteria:
        return {
            **session_meta(session),
            "error": "no verifier criteria on last workflow step",
            "tasks": [],
            "scatter_plot_data": [],
        }

    last_node = last_workflow_node(wf) or {}
    last_nid = str(last_node.get("id") or "")
    last_desc = str(last_node.get("description") or "")
    last_step_path = get_node_path(wf, last_nid)

    target = target_evaluation_node_and_file(wf)
    if target is None:
        return {
            **session_meta(session),
            "error": "no output files on workflow",
            "tasks": [],
            "scatter_plot_data": [],
        }

    eval_node, target_file = target
    nid = str(eval_node.get("id") or "")
    desc = str(eval_node.get("description") or "")

    report_tasks: list[dict[str, Any]] = []
    eval_cache: dict[str, dict[str, Any]] = {}
    stats = {"hits": 0, "misses": 0, "llm_calls": 0, "unique_seq": 0}

    snapshots = collect_unique_snapshots(traj, [target_file])
    if endpoints_only and len(snapshots) > 1:
        snapshots = [snapshots[0], snapshots[-1]]

    versions = [
        _score_one_snapshot(
            v_idx=i,
            snap=snap,
            criteria=final_criteria,
            step_path=last_step_path,
            desc=last_desc,
            eval_cache=eval_cache,
            dry_run=dry_run,
            client=client,
            model=model,
            stats=stats,
        )
        for i, snap in enumerate(snapshots)
    ]
    report_tasks.append(
        {
            "node_id": nid,
            "description": desc,
            "output_file": target_file,
            "output_files": [target_file],
            "final_rubrics": final_criteria,
            "graded_against_workflow_step": last_desc,
            "unique_agent_snapshots": len(snapshots),
            "versions": versions,
        }
    )

    out: dict[str, Any] = {
        **session_meta(session),
        "dry_run": dry_run,
        "grading_rubrics": final_criteria,
        "graded_against_workflow_step": last_desc,
        "evaluated_output_file": target_file,
        "tasks": report_tasks,
        "scatter_plot_data": build_scatter_plot_data(report_tasks),
        "eval_cache_stats": {
            "dedupe_hits": stats["hits"],
            "unique_agent_solutions_evaluated": stats["misses"],
            "llm_calls": stats["llm_calls"],
            "cached_keys": len(eval_cache),
            "unique_eval_sequences": stats["unique_seq"],
        },
    }
    if credential_meta:
        out["auth_resolved"] = {**credential_meta, "llm": "induce.anthropic_user_text (Anthropic SDK)"}
    return out


def rate_session_exported_status(session: dict[str, Any]) -> dict[str, Any]:
    base = {**session_meta(session), "scoring_method": "exported_status"}
    try:
        traj, wf = workflow_from_session(session)
    except ValueError:
        return {**base, "error": "no workflow found in trajectory"}
    if not wf:
        return {**base, "error": "no workflow found in trajectory"}

    entries = last_workflow_verifier_entries(wf)
    criteria = [c for c, _, _ in entries]
    passes = [p for _, _, p in entries]
    target = target_evaluation_node_and_file(wf)
    if target is None:
        return {**base, "error": "no output files on workflow"}
    _, target_file = target
    last_text, last_idx = last_file_snapshot(traj, target_file)

    out: dict[str, Any] = {
        **base,
        "file_environment_index": last_idx,
        "evaluated_output_file": target_file,
        "output_files": [target_file] if last_text else [],
        "final_rubrics": criteria,
        "verifier_statuses": [s for _, s, _ in entries],
        "pass_per_criterion": passes,
        "tasks": [],
        "scatter_plot_data": [],
        "error": None if criteria else "no verifier criteria in final workflow",
    }
    pct = average_success_percent(passes, len(criteria))
    if pct is not None:
        out["average_success_pct"] = round(pct, 1)
    return out


# --- Output / plots ---


def build_scatter_plot_data(report_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in report_tasks:
        if not isinstance(task, dict):
            continue
        for ver in task.get("versions") or []:
            if not isinstance(ver, dict):
                continue
            step, pct = ver.get("trajectory_step_index"), ver.get("average_success_pct")
            if isinstance(step, int) and isinstance(pct, (int, float)):
                rows.append(
                    {
                        "trajectory_step_index": step,
                        "average_success_pct": float(pct),
                        "node_id": task.get("node_id"),
                        "task_description": task.get("description"),
                        "version_index_within_task": ver.get("version_index"),
                        "actor": ver.get("actor"),
                        "unique_eval_sequence": ver.get("unique_eval_sequence"),
                        "first_seen_trajectory_step_index": ver.get("first_seen_trajectory_step_index"),
                        "eval_cache_hit": ver.get("eval_cache_hit"),
                    }
                )
    rows.sort(key=lambda r: (r["trajectory_step_index"], str(r.get("node_id") or "")))
    return rows


def try_load_existing_ratings(path: Path, *, expected_uuid: str | None) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("tasks"), list):
        return None
    if expected_uuid and str(raw.get("uuid") or "") != str(expected_uuid).strip():
        print(f"Ratings uuid mismatch for --session {expected_uuid!r}; re-running LLM.", file=sys.stderr)
        return None
    return raw


def _session_label(report: dict[str, Any]) -> str:
    uid, name = report.get("uuid") or "", report.get("name") or ""
    return f"session: {uid}" + (f" — {name}" if name else "")


def print_version_summary(report: dict[str, Any]) -> None:
    if report.get("error"):
        return
    print("--- average success rate by solution version ---", file=sys.stderr)
    print(_session_label(report), file=sys.stderr)
    st = report.get("eval_cache_stats")
    if isinstance(st, dict):
        print(
            f"eval cache: {st.get('dedupe_hits', 0)} dedupe hits, "
            f"{st.get('unique_agent_solutions_evaluated', 0)} unique keys, {st.get('llm_calls', 0)} LLM calls",
            file=sys.stderr,
        )
    dry = bool(report.get("dry_run"))
    for task in report.get("tasks") or []:
        if not isinstance(task, dict):
            continue
        nid, desc = str(task.get("node_id") or ""), str(task.get("description") or "")[:72]
        for ver in task.get("versions") or []:
            if not isinstance(ver, dict):
                continue
            vi, si, pct = ver.get("version_index"), ver.get("trajectory_step_index"), ver.get("average_success_pct")
            actor = str(ver.get("actor") or "").strip()
            actor_tag = f" ({actor})" if actor else ""
            dedupe = " (deduplicated)" if ver.get("eval_cache_hit") else ""
            if isinstance(pct, (int, float)):
                print(f"  [{nid}] {desc} | v{vi} @ traj {si}{actor_tag}: {float(pct):.1f}%{dedupe}", file=sys.stderr)
            elif ver.get("error"):
                print(f"  [{nid}] {desc} | v{vi} @ traj {si}{actor_tag}: N/A — {str(ver['error'])[:100]}", file=sys.stderr)
            elif dry:
                print(f"  [{nid}] {desc} | v{vi} @ traj {si}{actor_tag}: (dry-run)", file=sys.stderr)
    print("--- end summary ---", file=sys.stderr)


def print_exported_summary(report: dict[str, Any]) -> None:
    print(_session_label(report))
    print("scoring: exported_status (no LM)")
    if report.get("file_environment_index") is not None:
        print(f"last file @ trajectory index: {report['file_environment_index']}")
    if report.get("output_files"):
        print(f"output files: {', '.join(report['output_files'])}")
    print(f"criteria count: {len(report.get('final_rubrics') or [])}")
    if report.get("error"):
        print(f"error: {report['error']}")
        return
    criteria = report.get("final_rubrics") or []
    passes = report.get("pass_per_criterion") or []
    statuses = report.get("verifier_statuses") or []
    for i, c in enumerate(criteria):
        lab = passes[i] if i < len(passes) else None
        tag = "PASS" if lab is True else "FAIL" if lab is False else "?"
        st = statuses[i] if i < len(statuses) else ""
        print(f"  [{i:02d}] {tag}{f' [{st}]' if st else ''} — {c}")
    pct = report.get("average_success_pct")
    if isinstance(pct, (int, float)):
        n_pass = sum(1 for p in passes if p is True)
        print(f"\naverage_success_rate: {n_pass}/{len(criteria)} = {float(pct):.1f}%")


def save_scatter_plot(report: dict[str, Any], out_path: Path) -> Path | None:
    if report.get("error"):
        return None
    pts = report.get("scatter_plot_data") or build_scatter_plot_data(report.get("tasks") or [])
    if not pts:
        print("Scatter plot skipped: no scored points.", file=sys.stderr)
        return None

    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mpl_dir = out_path.parent / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:
        print(f"Scatter plot skipped: {exc}.", file=sys.stderr)
        return None

    xs = [p["trajectory_step_index"] for p in pts]
    ys = [p["average_success_pct"] for p in pts]
    node_ids = list(dict.fromkeys(str(p.get("node_id") or "") for p in pts))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(node_ids.index(str(p.get("node_id") or "")) % 10) for p in pts]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(xs, ys, c=colors, s=36, alpha=0.75, edgecolors="white", linewidths=0.4)
    ax.set_xlabel("Trajectory step index (agent snapshot)")
    ax.set_ylabel("Average success rate (%)")
    ax.set_title(f"Verifier success rate vs step — {report.get('name') or report.get('uuid') or 'Session'}")
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.3)
    if len(node_ids) <= 12:
        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=cmap(i % 10),
                    markersize=8,
                    label=n[:16] + ("…" if len(n) > 16 else ""),
                )
                for i, n in enumerate(node_ids)
            ],
            title="node_id",
            loc="best",
            fontsize=7,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Scatter plot saved: {out_path}", file=sys.stderr)
    return out_path


def load_dotenv_defaults(env_file: Path | None, override: bool) -> None:
    if env_file and env_file.is_file():
        load_dotenv(env_file, override=override)
        return
    scripts_env = _scripts / ".env"
    if scripts_env.is_file():
        load_dotenv(scripts_env, override=override)
    load_dotenv(override=override)


def resolve_client(args: argparse.Namespace) -> tuple[Any | None, str, dict[str, str] | None]:
    if args.exported_status:
        return None, "", None

    try:
        cfg = induce.resolve_anthropic_config(
            skip_api_config=args.no_api_config,
            skip_claude_settings=args.no_claude_settings,
        )
    except induce.AnthropicConfigError as e:
        if args.dry_run:
            return None, (args.model or "").strip() or DEFAULT_MODEL, {"resolver": "dry-run"}
        raise SystemExit(str(e)) from e

    key = (args.api_key or "").strip() or cfg.api_key
    key = key.strip().replace("\r", "").replace("\n", "")
    base = (args.base_url or "").strip().rstrip("/") or cfg.base_url
    model = (args.model or "").strip() or DEFAULT_MODEL
    cfg = induce.ResolvedAnthropicConfig(key, base, model)

    meta = {
        "resolver": "induce.resolve_anthropic_config + CLI",
        "model": model,
        "base_url_effective": base or "(Anthropic SDK default)",
    }
    if args.debug_auth:
        bu = base or "https://api.anthropic.com"
        url = f"{bu}/messages" if bu.endswith("/v1") else f"{bu.rstrip('/')}/v1/messages"
        masked = f"{key[:7]}…{key[-4:]} (len={len(key)})" if len(key) > 11 else f"(len={len(key)})"
        print("--- auth debug ---", file=sys.stderr)
        print(f"  messages URL: {url}", file=sys.stderr)
        print(f"  model: {model!r}", file=sys.stderr)
        print(f"  api_key: {masked}", file=sys.stderr)
        print("--- end ---", file=sys.stderr)

    if args.dry_run:
        return None, model, meta
    if key.startswith("tml-") or key.startswith("tml_"):
        raise SystemExit("Resolved key looks like TINKER_API_KEY; set ANTHROPIC_API_KEY.")
    return induce.make_anthropic_client(cfg), model, meta


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    session = resolve_session(load_sessions(args.json), args.session)
    if args.exported_status:
        return rate_session_exported_status(session)
    client, model, cred_meta = resolve_client(args)
    return rate_session(
        session,
        client=client,
        model=model,
        credential_meta=cred_meta,
        dry_run=args.dry_run,
        endpoints_only=args.endpoints_only,
    )


def plot_output_path(args: argparse.Namespace) -> Path:
    if args.plot != "__default__":
        return Path(args.plot)
    if args.out:
        return args.out.with_name(args.out.stem + "_scatter.png")
    return Path(__file__).resolve().parent / "plots" / "rubric_success_scatter.png"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-j", "--json", type=Path, required=True, dest="json")
    p.add_argument("-s", "--session", default=None, help="Session uuid (required if JSON has multiple)")
    p.add_argument("--exported-status", action="store_true", help="Use exported verifier marks (no LM)")
    p.add_argument("--endpoints-only", action="store_true", help="Rate only first and last snapshot per step")
    p.add_argument("-o", "--out", type=Path, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--api-key", default=None)
    p.add_argument("--env-file", type=Path, default=None)
    p.add_argument("--dotenv-override", action="store_true")
    p.add_argument("--no-claude-settings", action="store_true")
    p.add_argument("--no-api-config", action="store_true")
    p.add_argument("--debug-auth", action="store_true")
    p.add_argument("--model", default=None, help=f"Anthropic model (default: {DEFAULT_MODEL})")
    p.add_argument("--base-url", default=None)
    p.add_argument("--plot", nargs="?", const="__default__", default=None, metavar="PNG")
    args = p.parse_args()

    if args.exported_status and (args.dry_run or args.endpoints_only):
        raise SystemExit("--exported-status cannot combine with --dry-run or --endpoints-only")

    report: dict[str, Any] | None = None
    loaded_existing = False
    if args.out and args.out.is_file() and not args.force and not args.exported_status:
        report = try_load_existing_ratings(args.out.resolve(), expected_uuid=args.session)
        if report:
            loaded_existing = True
            print(f"Using existing ratings: {args.out.resolve()}", file=sys.stderr)
            if not report.get("scatter_plot_data"):
                report["scatter_plot_data"] = build_scatter_plot_data(report.get("tasks") or [])

    if report is None:
        load_dotenv_defaults(args.env_file, args.dotenv_override)
        report = run_report(args)

    (print_exported_summary if args.exported_status else print_version_summary)(report)

    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        if not loaded_existing:
            args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)

    if args.plot is not None:
        save_scatter_plot(report, plot_output_path(args))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
