#!/usr/bin/env python3
"""
Export task sessions from the Agent Cowork SQLite database to JSON.

Each exported session is:
  {
    "uuid": "<session id>",
    "name": "<title>",
    "trajectory": [
      {
        "actor": "user" | "agent",
        "action": str,
        "tool_result": optional str; present when the SDK tool outcome was merged into the prior tool-call step,
        "environment": { ... }  # omitted for user/agent steps that are only ``message("…")`` (no `` | ``)
      },
      ...
    ],
  }

The first trajectory step is always the user ``message({initial query})``; like later user
``message("…")`` steps, it omits ``environment``.
``user_prompt`` rows that are backend-built node instructions (``Proceed with: …`` + ``Task:``)
are omitted — they are LM input, not user chat. Only real follow-ups from the compose box appear
as later ``message("…")`` steps (the stored prompt is exactly what the user typed).

``edit_workflow()``, ``edit_verifier()``, and preview ``file_edit("…")`` rows all persist the same
``environment`` shape: ``workflow`` (nested steps + verifier criteria/status), ``file`` (output paths + content),
``memory`` and ``skill`` (each a map of ``file-name`` → file text as injected at snapshot time).

Per-step ``environment`` (workflow nodes, each node’s ``verifiers`` with ``status``, output files, memory/skills)
comes from ``messages.state_snapshot`` when recorded: human ``user_prompt``; SDK ``user`` tool
results; turn ``result``; and a synthetic ``verifier_label`` row after the verifier LM updates marks
(exported as agent ``verify("…")`` with the workflow node id). Pure ``message("…")`` steps (user or agent) omit ``environment``.
The export reapplies a carried-forward workflow tree from snapshots so, after ``edit_workflow`` removes
a step, later steps do not retain that step’s verifiers or output files (and ``file`` rows are aligned
to the current tree). Older DBs without ``state_snapshot`` fall back to one end-of-session snapshot.

The second is always the agent ``plan({initial query})``; its ``environment`` uses the workflow tree
as of the last persisted snapshot before the first ``edit_workflow`` (or WorkflowPlan tool input
with ``normalizeRoots``), not necessarily ``sessions.workflow_tree`` after later edits. Every step
``status`` is ``pending``; each verifier is ``{"criterion", "status": "failure"}``; ``file`` maps
paths to ``null``.

Database location (Electron userData):
- macOS: ~/Library/Application Support/Agent Cowork/sessions.db
- Windows: %APPDATA%\\Agent Cowork\\sessions.db
- Linux: ~/.config/Agent Cowork/sessions.db

Formats
-------
--format default (default)
  Human-readable action trajectory. Each step has actor, action, tool_result, and environment.
  Used by the existing context-export pipeline.

--format weight
  Raw SDK messages + human actions for training. Each session is split into task_units
  (one planning unit + one per workflow node). Each unit has:
    prompt_first_turn   : full prompt sent to the LM for the first turn (memoryPrefix included
                          when effective_prompt is persisted; otherwise reconstructed)
    agent_trajectory    : slimmed raw SDK messages (assistant/user/result/system/verifier_label)
    human_trajectory    : follow_up, file_edit, edit_workflow, edit_verifier actions
    verifiers           : final verifier criteria + pass/fail status
  Planning unit also has workflow_tree_generated and workflow_tree_final in LLM-native format.

Usage:
  conda activate code   # optional: use "code" env
  python export_task_sessions.py [--db PATH] [--output FILE] [--session-id ID] \\
    [--format {default,weight}]
  # Use AGENT_COWORK_DB to override DB path:
  AGENT_COWORK_DB=/path/to/sessions.db python export_task_sessions.py
"""

import argparse
import copy
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Cap per-file inlined content to keep exports bounded (bytes).
MAX_OUTPUT_FILE_BYTES = 500_000


def get_default_db_path() -> Optional[Path]:
    """Return default sessions.db path for this platform, or None if not found."""
    if os.environ.get("AGENT_COWORK_DB"):
        p = Path(os.environ["AGENT_COWORK_DB"])
        return p if p.exists() else None
    home = Path.home()
    if sys.platform == "darwin":
        candidates = [
            home / "Library" / "Application Support" / "Agent Cowork" / "sessions.db",
            home / "Library" / "Application Support" / "agent-cowork" / "sessions.db",
        ]
    elif sys.platform == "win32":
        appdata = os.environ.get("APPDATA", home / "AppData" / "Roaming")
        candidates = [
            Path(appdata) / "Agent Cowork" / "sessions.db",
            Path(appdata) / "agent-cowork" / "sessions.db",
        ]
    else:
        candidates = [
            home / ".config" / "Agent Cowork" / "sessions.db",
            home / ".config" / "agent-cowork" / "sessions.db",
        ]
    for p in candidates:
        if p.exists():
            return p
    return None


def parse_json_column(raw: Optional[str], default=None):
    if raw is None or raw == "":
        return default if default is not None else []
    try:
        out = json.loads(raw)
        return out if out is not None else (default if default is not None else [])
    except json.JSONDecodeError:
        return default if default is not None else []


def verifier_status(mark: Optional[str]) -> str:
    """Map DB verifier_marks to success/failure/unchecked."""
    if mark == "check":
        return "success"
    if mark == "cross":
        return "failure"
    return "unchecked"


def build_workflow_steps(
    steps: list,
    output_files: list,
    verification_criteria: list,
    verifier_marks: list,
) -> List[dict]:
    """Build list of workflow step objects with expected outputs and verifiers."""
    n = len(steps) if steps else 0
    result = []
    for i in range(n):
        step_desc = steps[i] if i < len(steps) else ""
        files = output_files[i] if i < len(output_files) and isinstance(output_files[i], list) else []
        criteria = verification_criteria[i] if i < len(verification_criteria) and isinstance(verification_criteria[i], list) else []
        marks = verifier_marks[i] if i < len(verifier_marks) and isinstance(verifier_marks[i], list) else []
        verifiers = []
        for j, c in enumerate(criteria):
            status = verifier_status(marks[j]) if j < len(marks) else "unchecked"
            verifiers.append({"criterion": c, "status": status})
        result.append({
            "step_index": i,
            "step_description": step_desc,
            "expected_output_files": files,
            "verifiers": verifiers,
        })
    return result


def _read_text_limited(abs_path: Path, max_bytes: int) -> Tuple[Optional[str], Optional[str]]:
    """Read up to max_bytes of UTF-8 text. Returns (text, error_message)."""
    try:
        if not abs_path.is_file():
            return None, "not_a_file"
        with abs_path.open("rb") as f:
            raw = f.read(max_bytes + 1)
        truncated = len(raw) > max_bytes
        chunk = raw[:max_bytes]
        text = chunk.decode("utf-8", errors="replace")
        if truncated:
            text += "\n[... export truncated: file larger than max bytes ...]"
        return text, None
    except OSError as e:
        return None, str(e)


def _collect_original_outputs_map(tree: Any) -> Dict[str, str]:
    """Map rel path -> content from nodes' originalOutputs (DB snapshot)."""
    out: Dict[str, str] = {}
    for _, node in iter_workflow_nodes_with_path(tree):
        oo = node.get("originalOutputs") if isinstance(node, dict) else None
        if not isinstance(oo, list):
            continue
        for item in oo:
            if not isinstance(item, dict):
                continue
            p = item.get("path")
            c = item.get("content")
            if isinstance(p, str) and isinstance(c, str) and p:
                out[p] = c
    return out


def _ordered_output_paths_from_tree(tree: Any) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for _, node in iter_workflow_nodes_with_path(tree):
        if not isinstance(node, dict):
            continue
        files = node.get("outputFiles")
        if not isinstance(files, list):
            continue
        for f in files:
            s = str(f).strip()
            if s and s not in seen:
                seen.add(s)
                ordered.append(s)
    return ordered


def _ordered_output_paths_legacy(output_files: list) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for row in output_files:
        if not isinstance(row, list):
            continue
        for f in row:
            s = str(f).strip()
            if s and s not in seen:
                seen.add(s)
                ordered.append(s)
    return ordered


def _build_output_file_entries(
    cwd: Optional[str],
    rel_paths: List[str],
    originals: Dict[str, str],
    max_bytes: int,
) -> List[dict]:
    base = Path(cwd).expanduser() if cwd else None
    entries: List[dict] = []
    for rel in rel_paths:
        item: dict = {"path": rel, "content": None, "content_source": None, "error": None}
        read_ok = False
        rel_path = Path(str(rel).strip()).expanduser() if rel else None
        if rel_path is not None and rel_path.is_absolute():
            try:
                abs_p = rel_path.resolve()
                if abs_p.is_file():
                    text, err = _read_text_limited(abs_p, max_bytes)
                    if text is not None:
                        item["content"] = text
                        item["content_source"] = "filesystem"
                        read_ok = True
                    elif err:
                        item["error"] = err
                else:
                    item["error"] = "not_a_file"
            except (OSError, ValueError):
                item["error"] = "resolve_or_read_failed"
        elif base is not None and rel:
            try:
                abs_p = (base / rel).resolve()
                base_r = base.resolve()
                if os.path.commonpath([str(base_r), str(abs_p)]) == str(base_r):
                    text, err = _read_text_limited(abs_p, max_bytes)
                    if text is not None:
                        item["content"] = text
                        item["content_source"] = "filesystem"
                        read_ok = True
                    elif err:
                        item["error"] = err
            except (OSError, ValueError):
                item["error"] = "resolve_or_read_failed"
        if not read_ok and rel in originals:
            item["content"] = originals[rel]
            item["content_source"] = "originalOutputs"
            item["error"] = None
        elif not read_ok and item["content"] is None and item["error"] is None:
            item["error"] = "no_cwd_or_missing_file" if base is None else "missing_or_unreadable"
        entries.append(item)
    return entries


def empty_environment() -> dict:
    return {"workflow": [], "file": [], "memory": {}, "skill": {}}


def _verifier_success_or_failure(mark: Optional[str], *, plan_snapshot: bool) -> str:
    """Export only ``success`` or ``failure`` (plan snapshot: all failure)."""
    if plan_snapshot:
        return "failure"
    if mark == "check":
        return "success"
    return "failure"


def _node_verifiers_for_export(
    criteria: list,
    marks: list,
    *,
    plan_snapshot: bool,
) -> List[dict]:
    if not isinstance(marks, list):
        marks = []
    out: List[dict] = []
    for j, c in enumerate(criteria):
        crit = str(c.get("criterion", "")) if isinstance(c, dict) else str(c)
        st = _verifier_success_or_failure(
            marks[j] if j < len(marks) else None,
            plan_snapshot=plan_snapshot,
        )
        out.append({"criterion": crit, "status": st})
    return out


def _legacy_step_verifiers_for_export(step: dict, *, plan_snapshot: bool) -> List[dict]:
    out: List[dict] = []
    for v in (step.get("verifiers") or []):
        if not isinstance(v, dict):
            continue
        crit = str(v.get("criterion", ""))
        if plan_snapshot:
            st = "failure"
        else:
            raw = v.get("status", "")
            st = "success" if raw == "success" else "failure"
        out.append({"criterion": crit, "status": st})
    return out


def _workflow_nested_nodes(nodes: Any, *, plan_snapshot: bool = False) -> List[dict]:
    out: List[dict] = []
    if not isinstance(nodes, list):
        return out
    for n in nodes:
        if not isinstance(n, dict):
            continue
        ch = n.get("children")
        if not isinstance(ch, list):
            ch = []
        crits = n.get("verifiers")
        if not isinstance(crits, list):
            crits = []
        marks_raw = n.get("verifierMarks")
        if not isinstance(marks_raw, list):
            marks_raw = []
        ofs = n.get("outputFiles")
        if not isinstance(ofs, list):
            ofs = []
        node_status: Any = "pending" if plan_snapshot else n.get("status")
        verifiers = _node_verifiers_for_export(crits, marks_raw, plan_snapshot=plan_snapshot)
        out.append({
            "id": n.get("id"),
            "description": str(n.get("description") or ""),
            "outputFiles": [str(x) for x in ofs],
            "verifiers": verifiers,
            "status": node_status,
            "children": _workflow_nested_nodes(ch, plan_snapshot=plan_snapshot),
        })
    return out


def workflow_nested_for_export(
    tree: Any,
    steps: list,
    output_files: list,
    verification_criteria: list,
    verifier_marks: list,
    *,
    plan_snapshot: bool = False,
) -> Any:
    if isinstance(tree, list) and len(tree) > 0:
        return _workflow_nested_nodes(tree, plan_snapshot=plan_snapshot)
    wsteps = build_workflow_steps(steps, output_files, verification_criteria, verifier_marks)
    return [
        {
            "id": f"legacy-step-{s.get('step_index')}",
            "description": str(s.get("step_description") or ""),
            "outputFiles": list(s.get("expected_output_files") or []),
            "verifiers": _legacy_step_verifiers_for_export(s, plan_snapshot=plan_snapshot),
            "status": "pending" if plan_snapshot else None,
            "children": [],
        }
        for s in wsteps
    ]


def _ordered_output_rel_paths(
    workflow_tree: Any,
    steps: list,
    output_files: list,
    verification_criteria: list,
    verifier_marks: list,
) -> List[str]:
    if isinstance(workflow_tree, list) and len(workflow_tree) > 0:
        return _ordered_output_paths_from_tree(workflow_tree)
    rel_paths = _ordered_output_paths_legacy(output_files)
    wsteps = build_workflow_steps(steps, output_files, verification_criteria, verifier_marks)
    for ws in wsteps:
        for f in ws.get("expected_output_files") or []:
            s = str(f).strip()
            if s and s not in rel_paths:
                rel_paths.append(s)
    return rel_paths


def build_environment_state(
    cwd: Optional[str],
    workflow_tree: Any,
    steps: list,
    output_files: list,
    verification_criteria: list,
    verifier_marks: list,
    *,
    include_files: bool,
    file_placeholder: bool = False,
    plan_snapshot: bool = False,
    max_file_bytes: int = MAX_OUTPUT_FILE_BYTES,
    memory: Optional[dict] = None,
    skill: Optional[dict] = None,
) -> dict:
    wf = workflow_nested_for_export(
        workflow_tree,
        steps,
        output_files,
        verification_criteria,
        verifier_marks,
        plan_snapshot=plan_snapshot,
    )
    rel_paths = _ordered_output_rel_paths(
        workflow_tree, steps, output_files, verification_criteria, verifier_marks
    )
    if include_files:
        originals = _collect_original_outputs_map(workflow_tree)
        files = _build_output_file_entries(cwd, rel_paths, originals, max_file_bytes)
    elif file_placeholder:
        files = {p: None for p in rel_paths}
    else:
        files = []
    mem = copy.deepcopy(memory) if isinstance(memory, dict) else {}
    sk = copy.deepcopy(skill) if isinstance(skill, dict) else {}
    return {"workflow": wf, "file": files, "memory": mem, "skill": sk}


def raw_has_workflow_tool(raw: dict) -> bool:
    """Match app MCP tool ``workflow``; SDK may expose as ``mcp__workflow__WorkflowPlan`` etc."""
    if raw.get("type") != "assistant":
        return False
    for b in (raw.get("message") or {}).get("content") or []:
        if not isinstance(b, dict) or b.get("type") != "tool_use":
            continue
        name = str(b.get("name") or "")
        if name == "workflow" or "WorkflowPlan" in name:
            return True
        nl = name.lower()
        if "workflow" in nl and "plan" in nl:
            return True
    return False


def is_tool_result_message(norm: dict) -> bool:
    if norm.get("role") != "agent":
        return False
    raw = norm.get("raw")
    if not isinstance(raw, dict) or raw.get("type") != "user":
        return False
    for b in (raw.get("message") or {}).get("content") or []:
        if isinstance(b, dict) and b.get("type") == "tool_result":
            return True
    return False


def _tool_result_blob(raw: dict) -> str:
    parts: List[str] = []
    for b in (raw.get("message") or {}).get("content") or []:
        if not isinstance(b, dict) or b.get("type") != "tool_result":
            continue
        c = b.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for item in c:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
    return "\n".join(parts).strip() or "(empty)"


def describe_human_action(norm: dict) -> str:
    prompt = norm.get("prompt", "")
    if isinstance(prompt, str):
        return f"message({json.dumps(prompt, ensure_ascii=False)})"
    return 'message("")'


def describe_agent_action(norm: dict) -> str:
    raw = norm.get("raw")
    if not isinstance(raw, dict):
        return "agent"
    t = raw.get("type")
    if t == "assistant":
        parts: List[str] = []
        for block in (raw.get("message") or {}).get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                txt = block.get("text", "")
                if isinstance(txt, str) and txt.strip():
                    parts.append(f"message({json.dumps(txt, ensure_ascii=False)})")
            elif block.get("type") == "tool_use":
                name = block.get("name", "?")
                inp = block.get("input", {})
                try:
                    inp_s = json.dumps(inp, ensure_ascii=False)
                except TypeError:
                    inp_s = str(inp)
                parts.append(f"{name}({inp_s})")
        return " | ".join(parts) if parts else "assistant"
    if t == "user":
        return f"tool_result({json.dumps(_tool_result_blob(raw), ensure_ascii=False)})"
    if t == "result":
        sub = raw.get("subtype", "")
        try:
            body = json.dumps(raw.get("result"), ensure_ascii=False, default=str)[:800]
        except TypeError:
            body = str(raw.get("result"))[:800]
        return f"result({sub},{body})"
    if t == "system":
        return f"system({raw.get('subtype', '')})"
    return str(t or "agent_event")


def _assistant_message_has_tool_use(m: dict) -> bool:
    """True if normalized message is an ``assistant`` SDK turn that includes at least one ``tool_use``."""
    if m.get("role") != "agent":
        return False
    raw = m.get("raw")
    if not isinstance(raw, dict) or raw.get("type") != "assistant":
        return False
    for block in (raw.get("message") or {}).get("content") or []:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            return True
    return False


def _assistant_tool_use_ids(m: dict) -> set:
    """Return the set of tool_use ids from an assistant message."""
    raw = m.get("raw")
    if not isinstance(raw, dict):
        return set()
    ids: set = set()
    for block in (raw.get("message") or {}).get("content") or []:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            tid = block.get("id")
            if tid:
                ids.add(tid)
    return ids


def _tool_result_message_ids(norm: dict) -> set:
    """Return the set of tool_use_ids referenced by tool_result blocks in a user message."""
    raw = norm.get("raw")
    if not isinstance(raw, dict) or raw.get("type") != "user":
        return set()
    ids: set = set()
    for b in (raw.get("message") or {}).get("content") or []:
        if isinstance(b, dict) and b.get("type") == "tool_result":
            tid = b.get("tool_use_id")
            if tid:
                ids.add(tid)
    return ids


def _assistant_text_only_payload(m: dict) -> Optional[str]:
    """
    If this normalized agent message is an ``assistant`` SDK message with only text blocks (no
    ``tool_use``), return the combined prose; otherwise None.
    """
    if m.get("role") != "agent":
        return None
    raw = m.get("raw")
    if not isinstance(raw, dict) or raw.get("type") != "assistant":
        return None
    texts: List[str] = []
    for block in (raw.get("message") or {}).get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use":
            return None
        if block.get("type") == "text":
            t = block.get("text", "")
            if isinstance(t, str) and t.strip():
                texts.append(t.strip())
    if not texts:
        return None
    return "\n\n".join(texts) if len(texts) > 1 else texts[0]


def _result_success_string_payload(m: dict) -> Optional[str]:
    """If this is a successful ``result`` message whose payload is a plain string, return it."""
    if m.get("role") != "agent":
        return None
    raw = m.get("raw")
    if not isinstance(raw, dict) or raw.get("type") != "result":
        return None
    if raw.get("subtype") != "success":
        return None
    r = raw.get("result")
    if isinstance(r, str):
        return r
    return None


def agent_export_action(msgs: List[dict], idx: int) -> Tuple[str, int]:
    """
    Serialized agent step. When an ``assistant`` text-only turn is immediately followed by a
    ``result(success)`` whose string payload matches the same prose (SDK duplicates the text in the
    envelope), emit a single ``message("…")`` and skip the result row (environment is unchanged).
    Returns ``(action, extra_skip)`` where ``extra_skip`` is 0 or 1.
    """
    m = msgs[idx]
    atext = _assistant_text_only_payload(m)
    if atext is not None and idx + 1 < len(msgs):
        nxt = msgs[idx + 1]
        rtext = _result_success_string_payload(nxt)
        if rtext is not None and (atext.strip() == rtext.strip()):
            return (f"message({json.dumps(atext, ensure_ascii=False)})", 1)
    return (describe_agent_action(m), 0)


def _step_omits_environment(actor: str, action: str, tool_result: Optional[str]) -> bool:
    """Omit env for user/agent prose-only steps (single ``message("…")``, no tool segment)."""
    if tool_result is not None:
        return False
    if actor not in ("user", "agent"):
        return False
    return action.startswith("message(") and " | " not in action


def _path_variant_strings(cwd: Optional[str], p: str) -> set[str]:
    """Path strings that may refer to the same file (for matching snapshot rows to workflow outputFiles)."""
    s = str(p).strip()
    out: set[str] = set()
    if not s:
        return out
    out.add(s.replace("\\", "/"))
    try:
        raw = Path(s)
        if raw.is_absolute():
            out.add(str(raw.resolve()).replace("\\", "/"))
        elif cwd:
            out.add(str((Path(cwd).expanduser() / s).resolve()).replace("\\", "/"))
    except (OSError, ValueError):
        pass
    return {x for x in out if x}


def _ordered_output_paths_nested_export_wf(wf: Any) -> List[str]:
    """Preorder merge of node ``outputFiles`` (same order as tree walk in TS export)."""
    seen: set[str] = set()
    ordered: List[str] = []

    def walk(nodes: Any) -> None:
        if not isinstance(nodes, list):
            return
        for n in nodes:
            if not isinstance(n, dict):
                continue
            for f in (n.get("outputFiles") or []):
                st = str(f).strip()
                if st and st not in seen:
                    seen.add(st)
                    ordered.append(st)
            walk(n.get("children"))

    walk(wf)
    return ordered


def _files_realigned_to_workflow(prior_files: Any, target_wf: list, cwd: Optional[str]) -> List[dict]:
    """Drop file rows not on ``target_wf``; order and path strings follow the workflow tree."""
    by_variants: Dict[str, dict] = {}
    if isinstance(prior_files, list):
        for f in prior_files:
            if not isinstance(f, dict):
                continue
            for k in _path_variant_strings(cwd, str(f.get("path", ""))):
                by_variants[k] = f
    out: List[dict] = []
    for p in _ordered_output_paths_nested_export_wf(target_wf):
        hit: Optional[dict] = None
        for k in _path_variant_strings(cwd, p):
            if k in by_variants:
                hit = by_variants[k]
                break
        if hit is not None:
            row = dict(hit)
            row["path"] = p
            out.append(row)
        else:
            out.append({"path": p, "content": None, "content_source": None, "error": None})
    return out


def _realign_env_to_workflow(base_env: dict, target_wf: list, cwd: Optional[str]) -> dict:
    return {
        "workflow": copy.deepcopy(target_wf),
        "file": _files_realigned_to_workflow(base_env.get("file"), target_wf, cwd),
    }


def _build_workflow_timeline(msgs: List[dict]) -> List[Optional[list]]:
    """
    Carry forward the latest nested ``workflow`` from each message's ``state_snapshot``.

    After ``edit_workflow`` (or any row that refreshes the snapshot), removed steps no longer appear
    in later indices; rows without a workflow in the snapshot keep the previous tree.
    """
    current: Optional[list] = None
    out: List[Optional[list]] = []
    for m in msgs:
        snap = m.get("state_snapshot")
        if isinstance(snap, dict):
            wf = snap.get("workflow")
            if isinstance(wf, list):
                current = copy.deepcopy(wf)
        out.append(copy.deepcopy(current) if current is not None else None)
    return out


def _build_memory_skill_timeline(msgs: List[dict]) -> Tuple[List[dict], List[dict]]:
    """
    Carry forward ``memory`` / ``skill`` filename→content maps from each message's ``state_snapshot``.

    Older snapshots without these keys keep the previous maps (or empty dict before the first snapshot
    that defines them).
    """
    mem_cur: dict = {}
    sk_cur: dict = {}
    mems_out: List[dict] = []
    sks_out: List[dict] = []
    for m in msgs:
        snap = m.get("state_snapshot")
        if isinstance(snap, dict):
            mp = snap.get("memory")
            if isinstance(mp, dict):
                mem_cur = copy.deepcopy(mp)
            sp = snap.get("skill")
            if isinstance(sp, dict):
                sk_cur = copy.deepcopy(sp)
        mems_out.append(copy.deepcopy(mem_cur))
        sks_out.append(copy.deepcopy(sk_cur))
    return mems_out, sks_out


def _environment_for_norm_merge(norm: dict, default_env: dict) -> Tuple[dict, bool]:
    """Merge snapshot file list with workflow from snapshot or fallback."""
    snap = norm.get("state_snapshot")
    if not isinstance(snap, dict):
        return default_env, False
    files = snap.get("file")
    if not isinstance(files, list):
        return default_env, False
    wf = snap.get("workflow")
    if isinstance(wf, list):
        return {"workflow": copy.deepcopy(wf), "file": copy.deepcopy(files)}, True
    wf_fb = default_env.get("workflow")
    if isinstance(wf_fb, list):
        return {"workflow": copy.deepcopy(wf_fb), "file": copy.deepcopy(files)}, True
    return default_env, False


def environment_for_norm(
    norm: dict,
    default_env: dict,
    *,
    cwd: Optional[str] = None,
    workflow_override: Optional[list] = None,
    memory: Optional[dict] = None,
    skill: Optional[dict] = None,
) -> Tuple[dict, bool]:
    """Return (environment dict, True if persisted ``state_snapshot`` contributed file content).

    Canonical shape is ``{"workflow": [...], "file": [...], "memory": {...}, "skill": {...}}``.
    ``workflow_override`` (per-message replayed tree) replaces the merged workflow so removed steps
    and their output files/verifiers do not appear in later steps. File rows are realigned to that tree.
    ``memory`` / ``skill`` are filename→content maps for the step (from carried-forward snapshots).
    """
    base, took_snap = _environment_for_norm_merge(norm, default_env)
    wf_target: Optional[list] = workflow_override
    if wf_target is None and isinstance(base.get("workflow"), list):
        wf_target = base["workflow"]
    if isinstance(wf_target, list):
        base = _realign_env_to_workflow(base, wf_target, cwd)
    mem = memory if isinstance(memory, dict) else {}
    sk = skill if isinstance(skill, dict) else {}
    base["memory"] = copy.deepcopy(mem)
    base["skill"] = copy.deepcopy(sk)
    return base, took_snap


def trajectory_row(
    actor: str,
    action: str,
    environment: dict,
    *,
    tool_result: Optional[str] = None,
) -> dict:
    """
    Build one trajectory object. User or agent steps that are only ``message("…")`` (no `` | ``)
    omit ``environment`` (keeps JSON small; state is on neighboring tool / verify / result rows).
    """
    row: Dict[str, Any] = {"actor": actor, "action": action}
    if tool_result is not None:
        row["tool_result"] = tool_result
    if not _step_omits_environment(actor, action, tool_result):
        row["environment"] = environment
    return row


def _prompts_equal(a: str, b: str) -> bool:
    return (a or "").strip() == (b or "").strip()


def is_backend_node_user_prompt(prompt: Any) -> bool:
    """
    True when this ``user_prompt`` is the app's node-solving instruction (``buildPromptForNode``),
    persisted for the LM — not text the human typed in the UI.
    """
    if not isinstance(prompt, str):
        return False
    p = prompt.strip()
    if not p.startswith("Proceed with: "):
        return False
    # Matches electron ``buildPromptForNode``: … path … \\n\\nTask: …
    return "\n\nTask: " in p


def build_full_session_trajectory(
    msgs: List[dict],
    initial_query: str,
    cwd_val: Optional[str],
    workflow_tree: Any,
    steps: list,
    output_files: list,
    verification_criteria: list,
    verifier_marks: list,
) -> List[dict]:
    empty_env = empty_environment()
    msgs = _merge_partial_assistant_messages(msgs)
    wf_timeline = _build_workflow_timeline(msgs)
    mem_timeline, sk_timeline = _build_memory_skill_timeline(msgs)
    plan_mem = mem_timeline[0] if mem_timeline else {}
    plan_sk = sk_timeline[0] if sk_timeline else {}
    final_mem = mem_timeline[-1] if mem_timeline else {}
    final_sk = sk_timeline[-1] if sk_timeline else {}

    plan_env = _build_plan_environment(
        msgs,
        workflow_tree,
        steps,
        output_files,
        verification_criteria,
        verifier_marks,
        memory=plan_mem,
        skill=plan_sk,
    )
    final_env = build_environment_state(
        cwd_val,
        workflow_tree,
        steps,
        output_files,
        verification_criteria,
        verifier_marks,
        include_files=True,
        memory=final_mem,
        skill=final_sk,
    )

    traj: List[dict] = [
        trajectory_row("user", f"message({json.dumps(initial_query, ensure_ascii=False)})", empty_env),
        trajectory_row("agent", f"plan({json.dumps(initial_query, ensure_ascii=False)})", plan_env),
    ]

    consumed_initial_user = False
    pending_skip_tool_result = False
    idx = 0
    while idx < len(msgs):
        m = msgs[idx]
        if m.get("role") == "user" and m.get("type") == "user_prompt":
            prompt = m.get("prompt", "")
            if isinstance(prompt, str) and _prompts_equal(prompt, initial_query) and not consumed_initial_user:
                consumed_initial_user = True
                idx += 1
                continue
            if is_backend_node_user_prompt(prompt):
                idx += 1
                continue

        raw = m.get("raw") if isinstance(m.get("raw"), dict) else {}
        if m.get("role") == "agent" and raw_has_workflow_tool(raw):
            pending_skip_tool_result = True
            idx += 1
            continue

        if pending_skip_tool_result and m.get("role") == "agent" and is_tool_result_message(m):
            pending_skip_tool_result = False
            idx += 1
            continue
        pending_skip_tool_result = False

        if m.get("type") == "verifier_label":
            wo_v = wf_timeline[idx] if idx < len(wf_timeline) else None
            m_v = mem_timeline[idx] if idx < len(mem_timeline) else {}
            s_v = sk_timeline[idx] if idx < len(sk_timeline) else {}
            step_env, _snap = environment_for_norm(
                m,
                final_env,
                cwd=cwd_val,
                workflow_override=wo_v,
                memory=m_v,
                skill=s_v,
            )
            nid_raw = m.get("nodeId", "")
            nid = str(nid_raw) if nid_raw is not None else ""
            act = f"verify({json.dumps(nid, ensure_ascii=False)})"
            traj.append(trajectory_row("agent", act, step_env))
            idx += 1
            continue

        if m.get("type") == "edit_workflow":
            wo_e = wf_timeline[idx] if idx < len(wf_timeline) else None
            m_e = mem_timeline[idx] if idx < len(mem_timeline) else {}
            s_e = sk_timeline[idx] if idx < len(sk_timeline) else {}
            step_env, _snap = environment_for_norm(
                m,
                final_env,
                cwd=cwd_val,
                workflow_override=wo_e,
                memory=m_e,
                skill=s_e,
            )
            traj.append(trajectory_row("user", "edit_workflow()", step_env))
            idx += 1
            continue
        if m.get("type") == "edit_verifier":
            wo_ev = wf_timeline[idx] if idx < len(wf_timeline) else None
            m_ev = mem_timeline[idx] if idx < len(mem_timeline) else {}
            s_ev = sk_timeline[idx] if idx < len(sk_timeline) else {}
            step_env, _snap = environment_for_norm(
                m,
                final_env,
                cwd=cwd_val,
                workflow_override=wo_ev,
                memory=m_ev,
                skill=s_ev,
            )
            traj.append(trajectory_row("user", "edit_verifier()", step_env))
            idx += 1
            continue

        if m.get("type") == "file_edit":
            wo_f = wf_timeline[idx] if idx < len(wf_timeline) else None
            m_f = mem_timeline[idx] if idx < len(mem_timeline) else {}
            s_f = sk_timeline[idx] if idx < len(sk_timeline) else {}
            step_env, _snap = environment_for_norm(
                m,
                final_env,
                cwd=cwd_val,
                workflow_override=wo_f,
                memory=m_f,
                skill=s_f,
            )
            p_raw = m.get("path", "")
            p = str(p_raw) if p_raw is not None else ""
            act = f"file_edit({json.dumps(p, ensure_ascii=False)})"
            traj.append(trajectory_row("user", act, step_env))
            idx += 1
            continue

        if m.get("role") == "user":
            wo_u = wf_timeline[idx] if idx < len(wf_timeline) else None
            m_u = mem_timeline[idx] if idx < len(mem_timeline) else {}
            s_u = sk_timeline[idx] if idx < len(sk_timeline) else {}
            u_env, _u_snap = environment_for_norm(
                m,
                final_env,
                cwd=cwd_val,
                workflow_override=wo_u,
                memory=m_u,
                skill=s_u,
            )
            traj.append(trajectory_row("user", describe_human_action(m), u_env))
        elif m.get("role") == "agent":
            action, extra = agent_export_action(msgs, idx)
            consume = 1 + extra
            merged_tool: Optional[str] = None
            if _assistant_message_has_tool_use(m):
                expected_ids = _assistant_tool_use_ids(m)
                tr_parts: List[str] = []
                scan = idx + consume
                while scan < len(msgs) and expected_ids:
                    if not is_tool_result_message(msgs[scan]):
                        break
                    rids = _tool_result_message_ids(msgs[scan])
                    if not rids or not rids.issubset(expected_ids):
                        break
                    tr_raw = msgs[scan].get("raw")
                    if isinstance(tr_raw, dict):
                        tr_parts.append(_tool_result_blob(tr_raw))
                    expected_ids -= rids
                    consume += 1
                    scan += 1
                if tr_parts:
                    merged_tool = "\n\n".join(p for p in tr_parts if p and p != "(empty)") or "(empty)"
            env_idx = min(idx + consume - 1, len(msgs) - 1) if (extra == 1 or merged_tool is not None) else idx
            wo_a = wf_timeline[env_idx] if env_idx < len(wf_timeline) else None
            m_a = mem_timeline[env_idx] if env_idx < len(mem_timeline) else {}
            s_a = sk_timeline[env_idx] if env_idx < len(sk_timeline) else {}
            step_env, _env_snap = environment_for_norm(
                msgs[env_idx],
                final_env,
                cwd=cwd_val,
                workflow_override=wo_a,
                memory=m_a,
                skill=s_a,
            )
            traj.append(
                trajectory_row(
                    "agent",
                    action,
                    step_env,
                    tool_result=merged_tool,
                )
            )
            idx += consume
            continue
        else:
            wo_x = wf_timeline[idx] if idx < len(wf_timeline) else None
            m_x = mem_timeline[idx] if idx < len(mem_timeline) else {}
            s_x = sk_timeline[idx] if idx < len(sk_timeline) else {}
            x_env, _ = environment_for_norm(
                {"role": "unknown"},
                final_env,
                cwd=cwd_val,
                workflow_override=wo_x,
                memory=m_x,
                skill=s_x,
            )
            traj.append(
                trajectory_row("user", json.dumps(m, ensure_ascii=False, default=str)[:400], x_env)
            )
        idx += 1

    return traj


def extract_initial_task_instruction(action_trajectory: List[dict], fallback: str) -> str:
    for m in action_trajectory:
        if m.get("role") == "user" and m.get("type") == "user_prompt":
            prompt = m.get("prompt", "")
            if is_backend_node_user_prompt(prompt):
                continue
            return (prompt if isinstance(prompt, str) else "") or fallback or ""
    return fallback or ""


def iter_workflow_nodes_with_path(tree: Any) -> List[tuple[str, dict]]:
    """Return list of (path, node) for a workflow tree. Path format matches UI getNodePath: 'A > B > C'."""
    if not isinstance(tree, list):
        return []

    out: List[tuple[str, dict]] = []

    def walk(nodes: Any, stack: List[str]) -> None:
        if not isinstance(nodes, list):
            return
        for n in nodes:
            if not isinstance(n, dict):
                continue
            desc = str(n.get("description") or "")
            stack.append(desc)
            out.append((" > ".join(stack), n))
            walk(n.get("children"), stack)
            stack.pop()

    walk(tree, [])
    return out


def normalize_message(msg: dict) -> dict:
    """Normalize a stored StreamMessage for JSON output (agent turn vs user message)."""
    if msg.get("type") == "user_prompt":
        return {"role": "user", "type": "user_prompt", "prompt": msg.get("prompt", "")}
    if msg.get("type") == "edit_workflow":
        return {"role": "user", "type": "edit_workflow"}
    if msg.get("type") == "edit_verifier":
        return {"role": "user", "type": "edit_verifier"}
    if msg.get("type") == "file_edit":
        return {"role": "user", "type": "file_edit", "path": msg.get("path", "")}
    if msg.get("type") == "verifier_label":
        return {
            "role": "agent",
            "type": "verifier_label",
            "nodeId": msg.get("nodeId", ""),
            "raw": msg,
        }
    # SDK message: could be assistant content, tool_use, tool_result, etc.
    return {"role": "agent", "raw": msg}


def _is_export_noise_message(msg: dict) -> bool:
    """
    Messages to omit from exported trajectories.

    ``stream_event``: partial streaming chunks.
    ``system`` (e.g. subtype init): SDK session/bootstrap; the export ``environment`` on each step is
    always derived from the stored session snapshot (workflow + files), not from these rows, so they
    only duplicate the same env as adjacent steps.
    """
    if msg.get("role") != "agent":
        return False
    raw = msg.get("raw")
    if not isinstance(raw, dict):
        return False
    t = raw.get("type")
    return t == "stream_event" or t == "system"


def filter_out_stream_events(trajectory: List[dict]) -> List[dict]:
    """Drop streaming chunks and SDK system/bootstrap messages before building trajectories."""
    return [msg for msg in trajectory if not _is_export_noise_message(msg)]


def extract_session(cursor: sqlite3.Cursor, session_id: str) -> Optional[dict]:
    row = cursor.execute(
        """SELECT id, title, last_prompt, workflow_tree, steps, output_files, verification_criteria, verifier_marks,
                  completed_step_indices, status, cwd, created_at, updated_at
           FROM sessions WHERE id = ?""",
        (session_id,),
    ).fetchone()
    if not row:
        return None
    (sid, title, last_prompt, workflow_tree_raw, steps_raw, output_files_raw, verification_criteria_raw, verifier_marks_raw,
     completed_indices_raw, status, cwd, created_at, updated_at) = row
    _ = parse_json_column(completed_indices_raw, [])
    workflow_tree = parse_json_column(workflow_tree_raw, [])
    steps = parse_json_column(steps_raw, [])
    output_files = parse_json_column(output_files_raw, [])
    verification_criteria = parse_json_column(verification_criteria_raw, [])
    verifier_marks = parse_json_column(verifier_marks_raw, [])

    try:
        messages_rows = cursor.execute(
            """SELECT data, state_snapshot, created_at FROM messages WHERE session_id = ? ORDER BY created_at ASC""",
            (session_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        messages_rows = [
            (r[0], None, r[1])
            for r in cursor.execute(
                "SELECT data, created_at FROM messages WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
        ]
    action_trajectory = []
    for data_str, snapshot_raw, _ in messages_rows:
        try:
            msg = json.loads(data_str)
            norm = normalize_message(msg)
            if snapshot_raw:
                try:
                    norm["state_snapshot"] = json.loads(snapshot_raw)
                except (json.JSONDecodeError, TypeError):
                    pass
            action_trajectory.append(norm)
        except json.JSONDecodeError:
            action_trajectory.append({"role": "unknown", "raw": data_str[:200]})
    action_trajectory = filter_out_stream_events(action_trajectory)

    initial_task_instruction = extract_initial_task_instruction(action_trajectory, last_prompt or "")

    cwd_val = cwd if isinstance(cwd, str) and cwd.strip() else None
    full_traj = build_full_session_trajectory(
        action_trajectory,
        initial_task_instruction,
        cwd_val,
        workflow_tree,
        steps,
        output_files,
        verification_criteria,
        verifier_marks,
    )

    return {
        "uuid": sid,
        "name": title or "",
        "trajectory": full_traj,
    }


# ────────────────────────────────────────────────────────────────────
# Weight-based export format
# ────────────────────────────────────────────────────────────────────


def _sdk_message_type(msg: dict) -> Optional[str]:
    """Return the SDK message type (system/assistant/user/result) or None for non-SDK."""
    raw = msg.get("raw") if msg.get("role") == "agent" else None
    if isinstance(raw, dict):
        return raw.get("type")
    return None


def _extract_workflow_tree_from_tool_use(agent_traj: List[dict]) -> List[dict]:
    """Extract the WorkflowPlan tool_use input.tasks from the planning trajectory."""
    for entry in agent_traj:
        raw = entry.get("raw", {})
        if raw.get("type") != "assistant":
            continue
        for block in (raw.get("message") or {}).get("content") or []:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            name = block.get("name", "")
            if "WorkflowPlan" in name or ("workflow" in name.lower() and "plan" in name.lower()):
                tasks = block.get("input", {}).get("tasks", [])
                if tasks:
                    return tasks
    return []


def normalize_workflow_plan_tasks(tasks: Any) -> List[dict]:
    """Match electron ``normalizeRoots``: unwrap a single wrapper root that only has children."""
    if not isinstance(tasks, list) or not tasks:
        return []
    roots: List[Any] = list(tasks)
    while len(roots) == 1:
        n = roots[0]
        if not isinstance(n, dict):
            break
        ch = n.get("children")
        if not isinstance(ch, list) or len(ch) == 0:
            break
        roots = ch
    return [x for x in roots if isinstance(x, dict)]


def _apply_plan_snapshot_visual(wf: Any) -> List[dict]:
    """Force plan-row semantics: every node ``pending``, every verifier ``failure``."""
    if not isinstance(wf, list):
        return []
    out: List[dict] = []
    for n in wf:
        if not isinstance(n, dict):
            continue
        ch = _apply_plan_snapshot_visual(n.get("children"))
        verifiers: List[dict] = []
        for v in (n.get("verifiers") or []):
            if isinstance(v, dict):
                c = str(v.get("criterion", ""))
                if c:
                    verifiers.append({"criterion": c, "status": "failure"})
            elif isinstance(v, str) and v.strip():
                verifiers.append({"criterion": v.strip(), "status": "failure"})
        out.append({
            "id": n.get("id"),
            "description": str(n.get("description") or ""),
            "outputFiles": [str(x) for x in (n.get("outputFiles") or [])],
            "verifiers": verifiers,
            "status": "pending",
            "children": ch,
        })
    return out


def _tool_tasks_to_export_nested_plan(tasks: Any) -> List[dict]:
    """Turn raw WorkflowPlan ``tasks`` (after ``normalize_workflow_plan_tasks``) into export workflow JSON."""
    id_counter = [0]

    def walk(nodes: Any) -> List[dict]:
        if not isinstance(nodes, list):
            return []
        out_local: List[dict] = []
        for n in nodes:
            if not isinstance(n, dict):
                continue
            nid = f"plan-tool-{id_counter[0]}"
            id_counter[0] += 1
            crits = n.get("verifiers") or []
            verifiers: List[dict] = []
            for c in crits:
                if isinstance(c, dict):
                    s = str(c.get("criterion", ""))
                else:
                    s = str(c) if c else ""
                if s:
                    verifiers.append({"criterion": s, "status": "failure"})
            ch_in = n.get("children")
            ch = walk(ch_in) if isinstance(ch_in, list) else []
            ofs = [str(x) for x in (n.get("outputFiles") or [])]
            out_local.append({
                "id": nid,
                "description": str(n.get("description") or ""),
                "outputFiles": ofs,
                "verifiers": verifiers,
                "status": "pending",
                "children": ch,
            })
        return out_local

    return walk(tasks)


def _snapshot_workflow_for_plan_row(msgs: List[dict]) -> Optional[list]:
    """Workflow for the synthetic ``plan(...)`` row: pre-edit snapshot or earliest post-plan snapshot."""
    first_edit_idx: Optional[int] = None
    for i, m in enumerate(msgs):
        if m.get("type") == "edit_workflow":
            first_edit_idx = i
            break
    if first_edit_idx is not None:
        for i in range(first_edit_idx - 1, -1, -1):
            snap = msgs[i].get("state_snapshot")
            if not isinstance(snap, dict):
                continue
            wf = snap.get("workflow")
            if isinstance(wf, list) and len(wf) > 0:
                return copy.deepcopy(wf)
        return None
    for m in msgs:
        snap = m.get("state_snapshot")
        if not isinstance(snap, dict):
            continue
        wf = snap.get("workflow")
        if isinstance(wf, list) and len(wf) > 0:
            return copy.deepcopy(wf)
    return None


def _build_plan_environment(
    msgs_merged: List[dict],
    workflow_tree: Any,
    steps: list,
    output_files: list,
    verification_criteria: list,
    verifier_marks: list,
    *,
    memory: Optional[dict] = None,
    skill: Optional[dict] = None,
) -> dict:
    """
    Environment for the synthetic ``plan(...)`` trajectory row: workflow as it was right after
    planning (from the last message snapshot before the first ``edit_workflow``, else the earliest
    snapshot with a workflow), not ``sessions.workflow_tree`` which may reflect later edits.
    Falls back to WorkflowPlan tool tasks (with ``normalize_workflow_plan_tasks``), then DB tree.
    """
    plan_wf: Any = None
    snap_wf = _snapshot_workflow_for_plan_row(msgs_merged)
    if snap_wf is not None:
        plan_wf = _apply_plan_snapshot_visual(snap_wf)
    else:
        raw_tasks = _extract_workflow_tree_from_tool_use(msgs_merged)
        normalized = normalize_workflow_plan_tasks(raw_tasks)
        if normalized:
            plan_wf = _tool_tasks_to_export_nested_plan(normalized)
    if plan_wf is None or (isinstance(plan_wf, list) and len(plan_wf) == 0):
        plan_wf = workflow_nested_for_export(
            workflow_tree,
            steps,
            output_files,
            verification_criteria,
            verifier_marks,
            plan_snapshot=True,
        )
    rel_paths = _ordered_output_paths_nested_export_wf(plan_wf)
    files = {p: None for p in rel_paths}
    mem = copy.deepcopy(memory) if isinstance(memory, dict) else {}
    sk = copy.deepcopy(skill) if isinstance(skill, dict) else {}
    return {"workflow": plan_wf, "file": files, "memory": mem, "skill": sk}


def _snapshot_workflow_tree(norm: dict) -> Optional[List[dict]]:
    """Extract workflow tree from a message's state_snapshot."""
    snap = norm.get("state_snapshot")
    if not isinstance(snap, dict):
        return None
    wf = snap.get("workflow")
    return wf if isinstance(wf, list) else None


def _snapshot_file_content(
    norm: dict, path: str, cwd: Optional[str] = None
) -> Optional[str]:
    """Get file content from a message's state_snapshot.

    ``path`` is the stored ``file_edit`` path (often cwd-relative after preview save); snapshot rows
    may use workflow basenames or absolute paths — match with path variants and basename fallback.
    """
    snap = norm.get("state_snapshot")
    if not isinstance(snap, dict):
        return None
    files = snap.get("file")
    if not isinstance(files, list):
        return None
    path_s = str(path).strip() if path is not None else ""
    if not path_s:
        return None
    cwd_opt = str(cwd).strip() if cwd else None
    wanted = _path_variant_strings(cwd_opt, path_s)
    for f in files:
        if not isinstance(f, dict):
            continue
        fp = str(f.get("path", "")).strip()
        if not fp:
            continue
        if fp == path_s:
            return f.get("content") if isinstance(f.get("content"), str) else None
        if wanted & _path_variant_strings(cwd_opt, fp):
            c = f.get("content")
            return c if isinstance(c, str) else None
    base_want = os.path.basename(path_s.replace("\\", "/"))
    if base_want:
        for f in files:
            if not isinstance(f, dict):
                continue
            fp = str(f.get("path", "")).strip()
            if not fp:
                continue
            if os.path.basename(fp.replace("\\", "/")) == base_want:
                c = f.get("content")
                return c if isinstance(c, str) else None
    return None


def _extract_verifier_criteria(tree_nodes: list) -> List[str]:
    """Extract verifier criterion strings from workflow tree nodes."""
    out = []
    for n in tree_nodes if isinstance(tree_nodes, list) else []:
        if not isinstance(n, dict):
            continue
        for v in n.get("verifiers") or []:
            if isinstance(v, dict):
                out.append(v.get("criterion", ""))
            elif isinstance(v, str):
                out.append(v)
    return out


def _extract_verifier_marks(tree_nodes: list) -> List[Optional[str]]:
    """Extract verifierMarks from workflow tree nodes."""
    out: List[Optional[str]] = []
    for n in tree_nodes if isinstance(tree_nodes, list) else []:
        if not isinstance(n, dict):
            continue
        for m in n.get("verifierMarks") or []:
            out.append(m)
    return out


def _find_node_in_tree(tree: Any, node_id: str) -> Optional[dict]:
    """Recursively find a node by id in a workflow tree."""
    if not isinstance(tree, list):
        return None
    for n in tree:
        if not isinstance(n, dict):
            continue
        if n.get("id") == node_id:
            return n
        found = _find_node_in_tree(n.get("children"), node_id)
        if found is not None:
            return found
    return None


def _flush_partial_group(partials: List[dict]) -> dict:
    """Merge a group of consecutive assistant partials into one entry."""
    if len(partials) == 1:
        return partials[0]
    merged_content: List[dict] = []
    for p in partials:
        for block in (p.get("raw", {}).get("message", {}).get("content") or []):
            merged_content.append(block)
    result = dict(partials[-1])
    final_raw = dict(result.get("raw", {}))
    final_msg = dict(final_raw.get("message", {}))
    final_msg["content"] = merged_content
    final_raw["message"] = final_msg
    result["raw"] = final_raw
    return result


def _merge_partial_assistant_messages(agent_traj: List[dict]) -> List[dict]:
    """Merge consecutive assistant messages that belong to one API call.

    Grouping rules (from design doc risk-5):
    - stop_reason=None  → partial message (includePartialMessages: true)
    - stop_reason="tool_use"/"end_turn" → final message of an API call

    Consecutive assistant messages are accumulated until either:
    1. A non-None stop_reason is seen (flush including it), or
    2. A non-assistant message arrives (flush all pending partials as one group).
    """
    out: List[dict] = []
    pending: List[dict] = []

    for entry in agent_traj:
        raw = entry.get("raw", {})
        if raw.get("type") != "assistant":
            if pending:
                out.append(_flush_partial_group(pending))
                pending = []
            out.append(entry)
            continue

        pending.append(entry)
        stop = raw.get("message", {}).get("stop_reason")
        if stop is not None:
            out.append(_flush_partial_group(pending))
            pending = []

    if pending:
        out.append(_flush_partial_group(pending))
    return out


def _to_llm_native_tree(tree: Any) -> List[dict]:
    """Convert an Electron-hydrated workflow tree to LLM native format.

    Electron hydrate adds: id, status, verifierMarks, children, and converts
    verifiers from strings to [{criterion, status}] objects.
    This function strips all of that back to the raw task format the LLM
    produced via the WorkflowPlan tool_use:
      {description, outputFiles, verifiers: [str], children?: [...]}
    Children are recursively converted and included only if non-empty.
    """
    if not isinstance(tree, list):
        return []
    out = []
    for n in tree:
        if not isinstance(n, dict):
            continue
        description = str(n.get("description") or "")
        output_files = [str(f) for f in (n.get("outputFiles") or [])]
        raw_verifiers = n.get("verifiers") or []
        verifiers: List[str] = []
        for v in raw_verifiers:
            if isinstance(v, dict):
                c = v.get("criterion", "")
                if c:
                    verifiers.append(str(c))
            elif isinstance(v, str) and v:
                verifiers.append(v)
        children = _to_llm_native_tree(n.get("children"))
        node: Dict[str, Any] = {
            "description": description,
            "outputFiles": output_files,
            "verifiers": verifiers,
        }
        if children:
            node["children"] = children
        out.append(node)
    return out


def _slim_tool_use_block(block: dict) -> dict:
    return {"type": "tool_use", "id": block.get("id", ""), "name": block.get("name", ""), "input": block.get("input", {})}


def _slim_tool_result_block(block: dict) -> dict:
    out: Dict[str, Any] = {"type": "tool_result", "tool_use_id": block.get("tool_use_id", "")}
    c = block.get("content")
    if isinstance(c, str):
        out["content"] = c
    elif isinstance(c, list):
        out["content"] = c
    return out


def _slim_content_block(block: dict) -> dict:
    t = block.get("type")
    if t == "tool_use":
        return _slim_tool_use_block(block)
    if t == "tool_result":
        return _slim_tool_result_block(block)
    if t == "text":
        return {"type": "text", "text": block.get("text", "")}
    return block


def _slim_raw_message(raw: dict) -> dict:
    """Strip infrastructure metadata from a raw SDK message, keeping only
    what LLM produced (assistant) or what LLM sees (user/tool_result)."""
    t = raw.get("type")

    if t == "assistant":
        msg = raw.get("message", {})
        content = [_slim_content_block(b) for b in (msg.get("content") or []) if isinstance(b, dict)]
        out: Dict[str, Any] = {"type": "assistant", "content": content}
        sr = msg.get("stop_reason")
        if sr is not None:
            out["stop_reason"] = sr
        return out

    if t == "user":
        msg = raw.get("message", {})
        raw_content = msg.get("content") or raw.get("content") or []
        content = [_slim_content_block(b) for b in raw_content if isinstance(b, dict)]
        return {"type": "user", "content": content}

    if t == "system":
        out = {"type": "system", "subtype": raw.get("subtype", "")}
        if raw.get("model"):
            out["model"] = raw["model"]
        return out

    if t == "result":
        out = {"type": "result", "subtype": raw.get("subtype", "")}
        r = raw.get("result")
        if r is not None:
            out["result"] = r
        return out

    if t == "verifier_label":
        return {"type": "verifier_label", "nodeId": raw.get("nodeId", "")}

    return raw


def _merge_parallel_tool_results(agent_traj: List[dict]) -> List[dict]:
    """Merge consecutive user/tool_result messages whose tool_use_ids all
    belong to the preceding assistant message's tool_use blocks."""
    out: List[dict] = []
    for entry in agent_traj:
        raw = entry.get("raw", {})
        if raw.get("type") != "user":
            out.append(entry)
            continue
        content = raw.get("content") or raw.get("message", {}).get("content") or []
        is_tool_result = all(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in content
        ) and len(content) > 0
        if not is_tool_result or not out:
            out.append(entry)
            continue
        prev = out[-1]
        prev_raw = prev.get("raw", {})
        prev_type = prev_raw.get("type")
        if prev_type == "user":
            prev_content = prev_raw.get("content") or prev_raw.get("message", {}).get("content") or []
            prev_is_tool_result = all(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in prev_content
            ) and len(prev_content) > 0
            if prev_is_tool_result:
                merged_raw = dict(prev_raw)
                merged_raw["content"] = list(prev_content) + list(content)
                if "message" in merged_raw:
                    del merged_raw["message"]
                out[-1] = {"raw": merged_raw}
                continue
        out.append(entry)
    return out


WORKFLOW_PLAN_INSTRUCTION = "\n".join([
    "",
    "IMPORTANT: You MUST call the mcp__workflow__WorkflowPlan tool as your very first action to register a structured plan.",
    "Do NOT write out steps as text. Use the tool with structured JSON input.",
    "Structure: Provide 3-5 main steps at the top level. Do NOT add a single wrapper root that repeats the task.",
    "Each main step (automation / level 0) must have a visually verifiable output: set outputFiles to file **names only** (e.g. position_slide.html, report.md, summary.txt)—no folders, no absolute paths, no ../ segments—or use verifiers to describe what the operator can check.",
    "For control mode (detailed view): add optional children to any main step to break it into detailed sub-steps; the number of sub-steps can depend on that step's complexity.",
    "Do NOT add separate validation/verification/testing steps — our system handles verification via verifier criteria on each node.",
    "Keep descriptions short but complete (under 10 words). Each node needs: description, outputFiles, verifiers, and optionally children.",
    "For outputFiles: use a single basename per entry (e.g. deliverable.md). Prefer .md for document-style output; use .txt when markdown does not apply.",
    "After calling the tool, STOP. Do NOT execute any steps yourself.",
    "The human operator will trigger each step individually.",
    "",
    "Task instruction:",
])


def build_weight_based_session(
    cursor: sqlite3.Cursor, session_id: str
) -> Optional[dict]:
    """Build the weight-based export for a single session."""
    row = cursor.execute(
        """SELECT id, title, workflow_tree, last_prompt, cwd
           FROM sessions WHERE id = ?""",
        (session_id,),
    ).fetchone()
    if not row:
        return None
    sid, title, workflow_tree_raw, last_prompt, cwd = row
    workflow_tree = parse_json_column(workflow_tree_raw, [])
    export_cwd: Optional[str] = None
    if cwd is not None:
        cs = str(cwd).strip()
        if cs:
            export_cwd = cs

    try:
        messages_rows = cursor.execute(
            """SELECT data, state_snapshot, created_at FROM messages
               WHERE session_id = ? ORDER BY created_at ASC""",
            (session_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        messages_rows = [
            (r[0], None, r[1])
            for r in cursor.execute(
                "SELECT data, created_at FROM messages WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
        ]

    all_msgs: List[dict] = []
    for data_str, snapshot_raw, ts in messages_rows:
        try:
            msg = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        t = msg.get("type")
        if t == "stream_event":
            continue
        norm = dict(msg)
        norm["_ts"] = ts
        if snapshot_raw:
            try:
                norm["state_snapshot"] = json.loads(snapshot_raw)
            except (json.JSONDecodeError, TypeError):
                pass
        all_msgs.append(norm)

    initial_task_instruction = ""
    for m in all_msgs:
        if m.get("type") == "user_prompt":
            p = m.get("prompt", "")
            if isinstance(p, str) and not is_backend_node_user_prompt(p):
                initial_task_instruction = p
                break
    if not initial_task_instruction:
        initial_task_instruction = last_prompt or ""

    # ── Segment messages into phases ──
    # Find boundaries: "Proceed with:" prompts mark node execution starts
    node_starts: List[Tuple[int, str]] = []  # (msg_index, node_description)
    path_nodes = iter_workflow_nodes_with_path(workflow_tree)
    path_to_node: Dict[str, dict] = {}
    for path, node in path_nodes:
        if isinstance(node.get("id"), str) and path:
            path_to_node[path] = node

    for i, m in enumerate(all_msgs):
        if m.get("type") != "user_prompt":
            continue
        p = m.get("prompt", "")
        if not isinstance(p, str) or not p.startswith("Proceed with: "):
            continue
        first_line = p.splitlines()[0]
        path = first_line.removeprefix("Proceed with: ").strip()
        if path in path_to_node:
            node_starts.append((i, path))

    # ── Build planning task_unit ──
    planning_end = node_starts[0][0] if node_starts else len(all_msgs)
    planning_msgs = all_msgs[:planning_end]

    planning_agent_traj_raw: List[dict] = []
    planning_human_traj: List[dict] = []
    prev_workflow_snapshot: Optional[List[dict]] = None

    for m in planning_msgs:
        t = m.get("type")
        if t in ("system", "assistant", "user", "result"):
            planning_agent_traj_raw.append({"raw": {k: v for k, v in m.items() if k not in ("_ts", "state_snapshot")}})

        elif t == "edit_workflow":
            wf_after = _snapshot_workflow_tree(m)
            entry: Dict[str, Any] = {
                "type": "edit_workflow",
                "round_index": None,
            }
            if prev_workflow_snapshot is not None:
                entry["workflow_tree_before"] = prev_workflow_snapshot
            if wf_after is not None:
                entry["workflow_tree_after"] = wf_after
                prev_workflow_snapshot = wf_after
            planning_human_traj.append(entry)

        elif t == "edit_verifier":
            wf_after = _snapshot_workflow_tree(m)
            entry = {
                "type": "edit_verifier",
                "round_index": None,
            }
            if prev_workflow_snapshot is not None:
                entry["verifiers_before"] = _extract_verifier_criteria(prev_workflow_snapshot)
            if wf_after is not None:
                entry["verifiers_after"] = _extract_verifier_criteria(wf_after)
                prev_workflow_snapshot = wf_after
            planning_human_traj.append(entry)

    planning_agent_traj_merged = _merge_partial_assistant_messages(planning_agent_traj_raw)
    workflow_tree_generated = _extract_workflow_tree_from_tool_use(planning_agent_traj_merged)
    workflow_tree_after_planning = prev_workflow_snapshot or workflow_tree
    planning_agent_traj_final = _merge_parallel_tool_results(planning_agent_traj_merged)
    planning_agent_traj = [{"raw": _slim_raw_message(e["raw"])} for e in planning_agent_traj_final]

    # Normalize planning human_trajectory workflow snapshots to LLM native format
    for h in planning_human_traj:
        if h.get("type") == "edit_workflow":
            if "workflow_tree_before" in h:
                h["workflow_tree_before"] = _to_llm_native_tree(h["workflow_tree_before"])
            if "workflow_tree_after" in h:
                h["workflow_tree_after"] = _to_llm_native_tree(h["workflow_tree_after"])

    # Reconstruct the planning first-turn prompt:
    # WORKFLOW_PLAN_INSTRUCTION + user's initial task instruction
    # NOTE: memoryPrefix is not available from DB; will be accurate once
    # effective_prompt is persisted (TODO: modify src/electron).
    planning_prompt = WORKFLOW_PLAN_INSTRUCTION + initial_task_instruction

    planning_unit: Dict[str, Any] = {
        "intent": "planning",
        "prompt_first_turn": planning_prompt,
        "agent_trajectory": planning_agent_traj,
        "human_trajectory": planning_human_traj,
        "verifiers": [],
        "workflow_tree_generated": workflow_tree_generated,
        "workflow_tree_final": _to_llm_native_tree(workflow_tree_after_planning),
    }

    # ── Build execution task_units ──
    task_units: List[dict] = [planning_unit]

    for seg_idx, (start_i, path) in enumerate(node_starts):
        end_i = node_starts[seg_idx + 1][0] if seg_idx + 1 < len(node_starts) else len(all_msgs)
        node_msgs = all_msgs[start_i:end_i]
        node = path_to_node[path]
        node_id = node.get("id", "")
        node_desc = node.get("description", "")

        agent_traj_raw: List[dict] = []
        human_traj: List[dict] = []
        round_counter = 0
        node_prompt_consumed = False
        node_first_turn_prompt: Optional[str] = None
        last_snapshot_msg: Optional[dict] = None

        for m in node_msgs:
            t = m.get("type")

            if t == "user_prompt":
                p = m.get("prompt", "")
                if isinstance(p, str) and is_backend_node_user_prompt(p):
                    if not node_prompt_consumed:
                        node_prompt_consumed = True
                        # Capture buildPromptForNode prompt as first-turn prompt.
                        # NOTE: memoryPrefix not included (TODO: persist effective_prompt).
                        node_first_turn_prompt = p
                        continue
                    human_traj.append({
                        "type": "follow_up",
                        "round_index": max(round_counter - 1, 0),
                        "prompt": p,
                    })
                    continue
                human_traj.append({
                    "type": "follow_up",
                    "round_index": max(round_counter - 1, 0),
                    "prompt": p if isinstance(p, str) else "",
                })
                continue

            if t in ("system", "assistant", "user", "result"):
                raw_clean = {k: v for k, v in m.items() if k not in ("_ts", "state_snapshot")}
                agent_traj_raw.append({"raw": raw_clean})
                if t == "system" and m.get("subtype") == "init":
                    round_counter += 1
                if m.get("state_snapshot"):
                    last_snapshot_msg = m

            elif t == "verifier_label":
                raw_clean = {k: v for k, v in m.items() if k not in ("_ts", "state_snapshot")}
                agent_traj_raw.append({"raw": raw_clean})
                if m.get("state_snapshot"):
                    last_snapshot_msg = m

            elif t == "file_edit":
                fe_path = m.get("path", "")
                edited_content = _snapshot_file_content(m, fe_path, export_cwd)
                original_content = None
                if last_snapshot_msg is not None:
                    original_content = _snapshot_file_content(last_snapshot_msg, fe_path, export_cwd)
                human_traj.append({
                    "type": "file_edit",
                    "round_index": None,
                    "path": fe_path,
                    "original": original_content,
                    "edited": edited_content,
                })

            elif t == "edit_workflow":
                wf_after = _snapshot_workflow_tree(m)
                entry = {"type": "edit_workflow", "round_index": None}
                if wf_after is not None:
                    entry["workflow_tree_after"] = wf_after
                human_traj.append(entry)

            elif t == "edit_verifier":
                wf_before = _snapshot_workflow_tree(last_snapshot_msg) if last_snapshot_msg else None
                wf_after = _snapshot_workflow_tree(m)
                entry: Dict[str, Any] = {"type": "edit_verifier", "round_index": None}
                if wf_before is not None:
                    before_node = _find_node_in_tree(wf_before, node_id)
                    if before_node:
                        entry["verifiers_before"] = before_node.get("verifiers", [])
                if wf_after is not None:
                    after_node = _find_node_in_tree(wf_after, node_id)
                    if after_node:
                        entry["verifiers_after"] = after_node.get("verifiers", [])
                if m.get("state_snapshot"):
                    last_snapshot_msg = m
                human_traj.append(entry)

        # Build verifiers from final workflow tree
        final_node = _find_node_in_tree(workflow_tree, node_id)
        verifiers: List[dict] = []
        if final_node:
            criteria = final_node.get("verifiers", [])
            marks = final_node.get("verifierMarks", [])
            for j, c in enumerate(criteria):
                crit = c.get("criterion", "") if isinstance(c, dict) else str(c)
                mark = marks[j] if j < len(marks) else None
                verifiers.append({
                    "criterion": crit,
                    "status": mark == "check",
                })

        agent_traj_merged = _merge_partial_assistant_messages(agent_traj_raw)
        agent_traj_with_results = _merge_parallel_tool_results(agent_traj_merged)
        agent_traj = [{"raw": _slim_raw_message(e["raw"])} for e in agent_traj_with_results]

        unit: Dict[str, Any] = {
            "intent": node_desc,
            "prompt_first_turn": node_first_turn_prompt or "",
            "agent_trajectory": agent_traj,
            "human_trajectory": human_traj,
            "verifiers": verifiers,
        }
        task_units.append(unit)

    return {
        "uuid": sid,
        "name": title or "",
        "initial_task_instruction": initial_task_instruction,
        "task_units": task_units,
    }


def extract_all_sessions_weight_based(cursor: sqlite3.Cursor) -> List[dict]:
    rows = cursor.execute("SELECT id FROM sessions ORDER BY updated_at DESC").fetchall()
    sessions = []
    for (session_id,) in rows:
        sess = build_weight_based_session(cursor, session_id)
        if sess:
            sessions.append(sess)
    return sessions


def extract_all_sessions(cursor: sqlite3.Cursor) -> List[dict]:
    rows = cursor.execute("SELECT id FROM sessions ORDER BY updated_at DESC").fetchall()
    out = []
    for (session_id,) in rows:
        sess = extract_session(cursor, session_id)
        if sess:
            out.append(sess)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Agent Cowork task sessions to JSON")
    parser.add_argument("--db", type=Path, help="Path to sessions.db (default: Electron userData location)")
    parser.add_argument("--output", "-o", type=Path, help="Output single JSON file (default: stdout)")
    parser.add_argument("--session-id", type=str, help="Export only this session ID")
    parser.add_argument(
        "--format",
        type=str,
        choices=["default", "weight"],
        default="default",
        help='Export format: "default" (human-readable trajectory) or "weight" (raw SDK messages + human_trajectory for training).',
    )
    args = parser.parse_args()

    db_path = args.db or get_default_db_path()
    if not db_path or not db_path.exists():
        print("Error: sessions.db not found. Set AGENT_COWORK_DB or pass --db PATH.", file=sys.stderr)
        if os.environ.get("AGENT_COWORK_DB"):
            print(f"  AGENT_COWORK_DB={os.environ['AGENT_COWORK_DB']}", file=sys.stderr)
        else:
            print("  Default paths tried (macOS): ~/Library/Application Support/Agent Cowork/sessions.db", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        if args.format == "weight":
            if args.session_id:
                sess = build_weight_based_session(cursor, args.session_id)
                if not sess:
                    print(f"Error: session not found: {args.session_id}", file=sys.stderr)
                    return 1
                payload = [sess]
            else:
                payload = extract_all_sessions_weight_based(cursor)
            json_str = json.dumps(payload, indent=2, ensure_ascii=False)
            if args.output:
                args.output.write_text(json_str + "\n", encoding="utf-8")
                n_units = sum(len(s.get("task_units", [])) for s in payload)
                print(f"Wrote {args.output} ({len(payload)} sessions, {n_units} task_units)", file=sys.stderr)
            else:
                print(json_str)
            return 0

        if args.session_id:
            payload = extract_session(cursor, args.session_id)
            if not payload:
                print(f"Error: session not found: {args.session_id}", file=sys.stderr)
                return 1
        else:
            payload = extract_all_sessions(cursor)

        json_str = json.dumps(payload, indent=2, ensure_ascii=False)
        if args.output:
            args.output.write_text(json_str + "\n", encoding="utf-8")
            print(f"Wrote {args.output}", file=sys.stderr)
        else:
            print(json_str)
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
