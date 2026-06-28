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

``brain_edit()`` rows record Brain dialog saves (memory + skill files on disk).
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
    human_trajectories  : follow_up, file_edit, brain_edit, edit_workflow, edit_verifier actions
                          (non-follow_up entries include a compact per-line ``diff``)
    verifiers           : final verifier criteria + pass/fail status
  Planning unit also has workflow_tree_generated and workflow_tree_final in LLM-native format.

  Per-message ``environment`` (Pi engine sessions): every OAI message inside
  ``agent_trajectories[].messages`` and every entry in ``human_trajectories`` carries an
  ``environment`` of the same shape as the default exporter:
      {workflow: [...nested nodes with verifier criterion+status...],
       file:     [{path, content, content_source, content_encoding, error}, ...],
       memory:   {<filename>: <content>, ...},
       skill:    {<filename>: <content>, ...}}
  The state is carried forward from each message's ``state_snapshot`` (workflow + file +
  memory + skill); messages without their own snapshot inherit the previous snapshot's
  state. The synthetic leading ``{role: user, content: prompt}`` message of each
  trajectory carries the start-of-segment environment (state at the moment the human
  posted that prompt). REINFORCE rewards read the final verifier statuses from the last
  message's ``environment.workflow[*].verifiers[*].status``; memory/skill induction takes
  diffs across successive ``environment.file`` snapshots within and between chunks.
  ``output_files`` (Pi-only, change-only artifacts) is retained alongside the full
  ``environment.file`` to preserve the existing "what was written this turn" signal.

Usage:
  conda activate code   # optional: use "code" env
  python export_task_sessions.py [--db PATH] [--output FILE] [--session-id ID] \\
    [--format {default,weight}]
  # Use AGENT_COWORK_DB to override DB path:
  AGENT_COWORK_DB=/path/to/sessions.db python export_task_sessions.py
"""

import argparse
import base64
import copy
import difflib
import json
import os
import re
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


def _read_file_content_limited(
    abs_path: Path, max_bytes: int
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Read up to ``max_bytes`` and return (content, content_encoding, error_message)."""
    try:
        if not abs_path.is_file():
            return None, None, "not_a_file"
        with abs_path.open("rb") as f:
            raw = f.read(max_bytes + 1)
        truncated = len(raw) > max_bytes
        chunk = raw[:max_bytes]
        try:
            text = chunk.decode("utf-8", errors="strict")
            if truncated:
                text += "\n[... export truncated: file larger than max bytes ...]"
            return text, "utf8", None
        except UnicodeDecodeError:
            return base64.b64encode(chunk).decode("ascii"), "base64", None
    except OSError as e:
        return None, None, str(e)


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
        item: dict = {"path": rel, "content": None, "content_source": None, "content_encoding": None, "error": None}
        read_ok = False
        rel_path = Path(str(rel).strip()).expanduser() if rel else None
        if rel_path is not None and rel_path.is_absolute():
            try:
                abs_p = rel_path.resolve()
                if abs_p.is_file():
                    content, content_encoding, err = _read_file_content_limited(abs_p, max_bytes)
                    if content is not None:
                        item["content"] = content
                        item["content_source"] = "filesystem"
                        item["content_encoding"] = content_encoding
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
                    content, content_encoding, err = _read_file_content_limited(abs_p, max_bytes)
                    if content is not None:
                        item["content"] = content
                        item["content_source"] = "filesystem"
                        item["content_encoding"] = content_encoding
                        read_ok = True
                    elif err:
                        item["error"] = err
            except (OSError, ValueError):
                item["error"] = "resolve_or_read_failed"
        if not read_ok and rel in originals:
            item["content"] = originals[rel]
            item["content_source"] = "originalOutputs"
            item["content_encoding"] = "utf8"
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
    blocks = (raw.get("message") or {}).get("content") or raw.get("blocks") or []
    for b in blocks:
        if not isinstance(b, dict) or b.get("type") != "tool_use":
            continue
        name = str(b.get("name") or "")
        if name in ("workflow", "workflow_plan") or "WorkflowPlan" in name:
            return True
        nl = name.lower()
        if "workflow" in nl and "plan" in nl:
            return True
    return False


def is_tool_result_message(norm: dict) -> bool:
    if norm.get("role") != "agent":
        return False
    raw = norm.get("raw")
    if not isinstance(raw, dict):
        return False
    # Pi format: standalone tool_result message
    if raw.get("type") == "tool_result":
        return True
    # Legacy format: user message with tool_result blocks
    if raw.get("type") != "user":
        return False
    for b in (raw.get("message") or {}).get("content") or []:
        if isinstance(b, dict) and b.get("type") == "tool_result":
            return True
    return False


def _tool_result_blob(raw: dict) -> str:
    # Pi format: standalone tool_result
    if raw.get("type") == "tool_result":
        c = raw.get("content", "")
        if isinstance(c, str):
            return c.strip() or "(empty)"
        if isinstance(c, list):
            texts = []
            for item in c:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(str(item.get("text", "")))
            return "\n".join(texts).strip() or "(empty)"
        return str(c).strip() or "(empty)"
    # Legacy format
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
    if norm.get("type") == "verifier_label":
        nid_raw = norm.get("nodeId", "")
        nid = str(nid_raw) if nid_raw is not None else ""
        return f"verify({json.dumps(nid, ensure_ascii=False)})"
    raw = norm.get("raw")
    if not isinstance(raw, dict):
        return "agent"
    t = raw.get("type")
    if t == "verifier_label":
        nid_raw = raw.get("nodeId", "")
        nid = str(nid_raw) if nid_raw is not None else ""
        return f"verify({json.dumps(nid, ensure_ascii=False)})"
    if t == "assistant":
        # Pi format uses top-level blocks; legacy uses raw.message.content
        blocks = raw.get("blocks") or (raw.get("message") or {}).get("content") or []
        parts: List[str] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                txt = block.get("text", "")
                if isinstance(txt, str) and txt.strip():
                    parts.append(f"message({json.dumps(txt, ensure_ascii=False)})")
            elif block.get("type") == "tool_use":
                name = block.get("name", "?")
                inp = block.get("input") or block.get("arguments") or {}
                try:
                    inp_s = json.dumps(inp, ensure_ascii=False)
                except TypeError:
                    inp_s = str(inp)
                parts.append(f"{name}({inp_s})")
        return " | ".join(parts) if parts else "assistant"
    if t == "tool_result":
        return f"tool_result({json.dumps(_tool_result_blob(raw), ensure_ascii=False)})"
    if t == "user":
        return f"tool_result({json.dumps(_tool_result_blob(raw), ensure_ascii=False)})"
    if t == "run_result":
        status = raw.get("status", "")
        return f"run_result({status})"
    if t == "system_init":
        model = raw.get("model", "")
        return f"system_init({model})"
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
    blocks = raw.get("blocks") or (raw.get("message") or {}).get("content") or []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            return True
    return False


def _assistant_tool_use_ids(m: dict) -> set:
    """Return the set of tool_use ids from an assistant message."""
    raw = m.get("raw")
    if not isinstance(raw, dict):
        return set()
    ids: set = set()
    blocks = raw.get("blocks") or (raw.get("message") or {}).get("content") or []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            tid = block.get("id")
            if tid:
                ids.add(tid)
    return ids


def _tool_result_message_ids(norm: dict) -> set:
    """Return the set of tool_use_ids referenced by tool_result blocks in a user message."""
    raw = norm.get("raw")
    if not isinstance(raw, dict):
        return set()
    # Pi format: standalone tool_result with toolUseId
    if raw.get("type") == "tool_result":
        tid = raw.get("toolUseId")
        return {tid} if tid else set()
    # Legacy format
    if raw.get("type") != "user":
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
    blocks = raw.get("blocks") or (raw.get("message") or {}).get("content") or []
    texts: List[str] = []
    for block in blocks:
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
            out.append({"path": p, "content": None, "content_source": None, "content_encoding": None, "error": None})
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


def _build_file_timeline(msgs: List[dict]) -> List[Optional[list]]:
    """
    Carry forward each snapshot's ``file`` list (path/content rows) across messages.

    Rows from the most recent snapshot persist until a later snapshot replaces them. Used by the
    weight-format exporter so messages without their own snapshot still see file content from the
    nearest preceding snapshot (instead of falling back to a fresh on-disk read that may not match
    the historical state).
    """
    current: Optional[list] = None
    out: List[Optional[list]] = []
    for m in msgs:
        snap = m.get("state_snapshot")
        if isinstance(snap, dict):
            files = snap.get("file")
            if isinstance(files, list):
                current = copy.deepcopy(files)
        out.append(copy.deepcopy(current) if current is not None else None)
    return out


def _build_message_environment(
    norm: dict,
    *,
    cwd: Optional[str],
    workflow_override: Optional[list],
    file_override: Optional[list],
    memory: dict,
    skill: dict,
) -> dict:
    """Per-message canonical 4-key environment for the weight format.

    Returns the genuine carried-forward snapshot state at this message — never the session's final
    on-disk / final-workflow state. This mirrors the default exporter, which seeds ``env_carry`` with
    :func:`empty_environment` and only advances it when a real ``state_snapshot`` is encountered, so
    the env attached to early messages (e.g., before the workflow is planned, before any file is
    written) accurately reflects "nothing exists yet" instead of leaking the post-completion state.

    Behavior:
    - ``workflow_override`` (carried-forward snapshot workflow) and ``file_override`` (carried-forward
      snapshot ``file`` rows) drive the result. File rows are realigned to the workflow tree so paths
      that no longer appear under the current tree drop out (matches :func:`_realign_env_to_workflow`).
    - When both are ``None`` (no snapshot has been recorded yet at this point in the timeline),
      returns an empty environment — *not* a fallback to session-level workflow_tree / on-disk reads.
    - ``memory`` / ``skill`` are passed through as filename → content maps.
    """
    if isinstance(workflow_override, list):
        files_in: List[dict] = file_override if isinstance(file_override, list) else []
        realigned = _realign_env_to_workflow(
            {"workflow": copy.deepcopy(workflow_override), "file": copy.deepcopy(files_in)},
            workflow_override,
            cwd,
        )
        realigned["memory"] = copy.deepcopy(memory) if isinstance(memory, dict) else {}
        realigned["skill"] = copy.deepcopy(skill) if isinstance(skill, dict) else {}
        return realigned
    return {
        "workflow": [],
        "file": copy.deepcopy(file_override) if isinstance(file_override, list) else [],
        "memory": copy.deepcopy(memory) if isinstance(memory, dict) else {},
        "skill": copy.deepcopy(skill) if isinstance(skill, dict) else {},
    }


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
    message: Optional[str] = None,
) -> dict:
    """
    Build one trajectory object. User or agent steps that are only ``message("…")`` (no `` | ``)
    omit ``environment`` (keeps JSON small; state is on neighboring tool / verify / result rows).
    """
    row: Dict[str, Any] = {"actor": actor, "action": action}
    if isinstance(message, str):
        row["message"] = message
    if tool_result is not None:
        row["tool_result"] = tool_result
    if not _step_omits_environment(actor, action, tool_result):
        row["environment"] = environment
    return row


def _parse_action_message_payload(action: str) -> Optional[str]:
    """Parse a leading ``message("...")`` segment and return decoded text."""
    if not isinstance(action, str):
        return None
    s = action.strip()
    if not (s.startswith("message(") and s.endswith(")")):
        return None
    inner = s[len("message("):-1].strip()
    if len(inner) >= 2 and inner[0] == '"' and inner[-1] == '"':
        try:
            return str(json.loads(inner))
        except json.JSONDecodeError:
            return inner[1:-1]
    return inner


def split_agent_action_and_message(action: str) -> Tuple[str, Optional[str]]:
    """
    For mixed rows like ``message("...") | Bash({...})``, export:
    - action: ``Bash({...})`` (tool-call chain)
    - message: decoded text from ``message("...")``
    """
    if not isinstance(action, str):
        return action, None
    parts = [p.strip() for p in action.split(" | ")]
    if len(parts) <= 1:
        return action, None
    head = parts[0]
    msg = _parse_action_message_payload(head)
    if msg is None:
        return action, None
    tool_action = " | ".join(p for p in parts[1:] if p)
    return (tool_action or action), msg


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


def is_human_edits_user_prompt(prompt: Any) -> bool:
    """True for compose-box prompts that only carry localized human file/verifier edits."""
    if not isinstance(prompt, str):
        return False
    p = prompt.strip()
    return (
        p.startswith("Human file edits (localized line changes):")
        or p.startswith("Human file edits (line diff")
        or p.startswith("Human verifier edits:")
        or p.startswith("Human comments on text files:")
    )


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

        if m.get("role") == "agent" and m.get("type") in ("update_verifiers", "edit_verifier"):
            wo_uv = wf_timeline[idx] if idx < len(wf_timeline) else None
            m_uv = mem_timeline[idx] if idx < len(mem_timeline) else {}
            s_uv = sk_timeline[idx] if idx < len(sk_timeline) else {}
            step_env, _snap = environment_for_norm(
                m,
                final_env,
                cwd=cwd_val,
                workflow_override=wo_uv,
                memory=m_uv,
                skill=s_uv,
            )
            traj.append(trajectory_row("agent", "edit_verifier()", step_env))
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
        if m.get("role") == "user" and m.get("type") == "edit_verifier":
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

        if m.get("type") == "brain_edit":
            wo_br = wf_timeline[idx] if idx < len(wf_timeline) else None
            m_br = mem_timeline[idx] if idx < len(mem_timeline) else {}
            s_br = sk_timeline[idx] if idx < len(sk_timeline) else {}
            step_env, _snap = environment_for_norm(
                m,
                final_env,
                cwd=cwd_val,
                workflow_override=wo_br,
                memory=m_br,
                skill=s_br,
            )
            traj.append(trajectory_row("user", "brain_edit()", step_env))
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
            action_out = action
            message_out: Optional[str] = None
            if " | " in action:
                action_out, message_out = split_agent_action_and_message(action)
            traj.append(
                trajectory_row(
                    "agent",
                    action_out,
                    step_env,
                    tool_result=merged_tool,
                    message=message_out,
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


# Matches src/electron/libs/runner.ts WORKFLOW_PLAN_APPEND_USER_PROMPT (appended on first turn).
_WORKFLOW_PLAN_USER_SUFFIX_MARKER = "Before doing anything else, you MUST call the workflow_plan"
_HEADLESS_USER_SUFFIX_MARKER = "\n\nHeadless execution only:"


def strip_interface_user_prompt(prompt: Any) -> str:
    """Return the human-typed task text without backend-appended planning/headless suffixes."""
    if not isinstance(prompt, str):
        return ""
    text = prompt.strip()
    if not text:
        return ""

    idx = text.find(_WORKFLOW_PLAN_USER_SUFFIX_MARKER)
    if idx >= 0:
        text = text[:idx].rstrip()

    idx = text.find(_HEADLESS_USER_SUFFIX_MARKER)
    if idx >= 0:
        text = text[:idx].rstrip()

    return text.strip()


def _extract_pi_jsonl_user_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "\n".join(p for p in parts if p)
    return ""


def load_initial_task_from_pi_session_file(path: Optional[str]) -> str:
    """First user turn from the Pi session jsonl (when planning user_prompt was not persisted)."""
    if not path or not isinstance(path, str):
        return ""
    p = Path(path.strip())
    if not p.is_file():
        return ""
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = obj.get("message") if isinstance(obj.get("message"), dict) else obj
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            stripped = strip_interface_user_prompt(_extract_pi_jsonl_user_text(msg.get("content")))
            if stripped:
                return stripped
    except OSError:
        pass
    return ""


def resolve_pi_session_file(
    db_pi_session_file: Optional[str],
    raw_msgs: Optional[List[dict]] = None,
) -> Optional[str]:
    if isinstance(db_pi_session_file, str) and db_pi_session_file.strip():
        return db_pi_session_file.strip()
    for m in raw_msgs or []:
        if m.get("type") == "system_init":
            sf = m.get("sessionFile")
            if isinstance(sf, str) and sf.strip():
                return sf.strip()
    return None


def extract_initial_task_instruction(
    action_trajectory: List[dict],
    fallback: str,
    *,
    pi_session_file: Optional[str] = None,
) -> str:
    for m in action_trajectory:
        if m.get("role") == "user" and m.get("type") == "user_prompt":
            prompt = m.get("prompt", "")
            if is_backend_node_user_prompt(prompt):
                continue
            if is_human_edits_user_prompt(prompt):
                continue
            stripped = strip_interface_user_prompt(prompt)
            if stripped:
                return stripped
    from_pi = load_initial_task_from_pi_session_file(pi_session_file)
    if from_pi:
        return from_pi
    fb = strip_interface_user_prompt(fallback)
    if fb and not is_backend_node_user_prompt(fb) and not is_human_edits_user_prompt(fb):
        return fb
    return ""


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


def segment_trajectory_by_resume_points(action_trajectory: List[dict], workflow_tree: Any) -> dict[str, List[dict]]:
    """
    Segment the session message stream into per-node slices using each node's resumePoint.uuid.

    In the app, node-solving prompts are broadcast to the UI but not persisted; however, each node
    stores resumePoint.uuid as the last SDK assistant UUID before the node run starts. We can use
    those UUIDs as stable boundaries in the persisted message log.

    Returns mapping: node_id -> list of normalized trajectory messages (slice for that node run).
    """
    # Index assistant UUID -> trajectory index
    uuid_to_index: dict[str, int] = {}
    for i, m in enumerate(action_trajectory):
        if m.get("role") != "agent":
            continue
        raw = m.get("raw")
        if not isinstance(raw, dict):
            continue
        u = raw.get("uuid")
        if isinstance(u, str):
            uuid_to_index[u] = i

    # Collect (start_index, node_id) for nodes with resumePoint.uuid
    starts: List[tuple[int, str]] = []
    for _, node in iter_workflow_nodes_with_path(workflow_tree):
        node_id = node.get("id")
        if not isinstance(node_id, str):
            continue
        rp = node.get("resumePoint")
        if not isinstance(rp, dict):
            continue
        u = rp.get("uuid")
        if not isinstance(u, str):
            continue
        idx = uuid_to_index.get(u)
        if idx is None:
            continue
        starts.append((idx + 1, node_id))

    # Order by occurrence in trajectory; build slices between boundaries
    starts.sort(key=lambda x: x[0])
    node_to_slice: dict[str, List[dict]] = {}
    for i, (start_i, node_id) in enumerate(starts):
        end_i = starts[i + 1][0] if i + 1 < len(starts) else len(action_trajectory)
        node_to_slice[node_id] = action_trajectory[start_i:end_i]
    return node_to_slice


def segment_trajectory_by_persisted_node_prompts(action_trajectory: List[dict], workflow_tree: Any) -> dict[str, List[dict]]:
    """
    Segment by persisted node-solving prompts (preferred when available).

    After a node solve starts, the app emits a `user_prompt` containing the nodePrompt built from:
      buildPromptForNode(node.description, pathContext, ...)
    We can recover the node by matching the pathContext, which is the first line: "Proceed with: {path}".
    """
    path_nodes = iter_workflow_nodes_with_path(workflow_tree)
    path_to_node_id: dict[str, str] = {}
    for path, node in path_nodes:
        node_id = node.get("id")
        if isinstance(node_id, str) and path:
            path_to_node_id[path] = node_id

    runs: List[tuple[int, str]] = []
    for i, m in enumerate(action_trajectory):
        if m.get("role") != "user" or m.get("type") != "user_prompt":
            continue
        prompt = m.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.startswith("Proceed with: "):
            continue
        first_line = prompt.splitlines()[0]
        path = first_line.removeprefix("Proceed with: ").strip()
        node_id = path_to_node_id.get(path)
        if node_id:
            runs.append((i, node_id))

    if not runs:
        return {}

    node_to_slice: dict[str, List[dict]] = {}
    for idx, (start_i, node_id) in enumerate(runs):
        end_i = runs[idx + 1][0] if idx + 1 < len(runs) else len(action_trajectory)
        node_to_slice[node_id] = action_trajectory[start_i:end_i]
    return node_to_slice


def normalize_legacy_message(msg: dict) -> dict:
    """Normalize a stored StreamMessage for JSON output (agent turn vs user message)."""
    if msg.get("type") == "user_prompt":
        return {"role": "user", "type": "user_prompt", "prompt": msg.get("prompt", "")}
    if msg.get("type") == "edit_workflow":
        return {"role": "user", "type": "edit_workflow"}
    if msg.get("type") == "edit_verifier":
        return {"role": "user", "type": "edit_verifier"}
    if msg.get("type") == "update_verifiers":
        return {"role": "agent", "type": "edit_verifier", "nodeId": msg.get("nodeId", ""), "raw": msg}
    if msg.get("type") == "file_edit":
        return {"role": "user", "type": "file_edit", "path": msg.get("path", "")}
    if msg.get("type") == "brain_edit":
        return {"role": "user", "type": "brain_edit"}
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
    ``system`` / ``system_init``: SDK session/bootstrap metadata.
    ``run_result``: Pi run completion metadata (status/usage only, no LLM content).
    ``node_completed``: Pi node lifecycle event.
    ``assistant`` with ``stopReason: "error"``: truncated/failed Pi retry attempts.
    """
    if msg.get("role") != "agent":
        return False
    raw = msg.get("raw")
    if not isinstance(raw, dict):
        return False
    t = raw.get("type")
    if t in ("stream_event", "system", "system_init", "run_result", "node_completed"):
        return True
    if t == "assistant" and raw.get("stopReason") == "error":
        return True
    return False


def normalize_pi_message(msg: dict) -> dict:
    msg_type = msg.get("type")
    if msg_type == "user_prompt":
        return {"role": "user", "type": "user_prompt", "prompt": msg.get("prompt", "")}
    if msg_type == "node_completed":
        return {"role": "agent", "type": "node_completed", "raw": msg}
    if msg_type == "verifier_label":
        return {
            "role": "agent",
            "type": "verifier_label",
            "nodeId": msg.get("nodeId", ""),
            "raw": msg,
        }
    if msg_type in ("system_init", "assistant", "tool_result", "run_result"):
        return {"role": "agent", "type": msg_type, "raw": msg}
    return {"role": "agent", "raw": msg}


def filter_out_stream_events(trajectory: List[dict]) -> List[dict]:
    """Drop streaming chunks and SDK system/bootstrap messages before building trajectories."""
    return [msg for msg in trajectory if not _is_export_noise_message(msg)]


def extract_session(cursor: sqlite3.Cursor, session_id: str) -> Optional[dict]:
    try:
        row = cursor.execute(
            """SELECT id, title, engine, last_prompt, workflow_tree, steps, output_files, verification_criteria, verifier_marks,
                      completed_step_indices, status, cwd, created_at, updated_at, pi_session_file
               FROM sessions WHERE id = ?""",
            (session_id,),
        ).fetchone()
    except sqlite3.OperationalError:
        row = cursor.execute(
            """SELECT id, title, engine, last_prompt, workflow_tree, steps, output_files, verification_criteria, verifier_marks,
                      completed_step_indices, status, cwd, created_at, updated_at
               FROM sessions WHERE id = ?""",
            (session_id,),
        ).fetchone()
        if row:
            row = (*row, None)
    if not row:
        return None
    (sid, title, engine, last_prompt, workflow_tree_raw, steps_raw, output_files_raw, verification_criteria_raw, verifier_marks_raw,
     completed_indices_raw, status, cwd, created_at, updated_at, pi_session_file_raw) = row
    _ = parse_json_column(completed_indices_raw, [])
    engine = engine or "legacy-claude"
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
            norm = normalize_pi_message(msg) if engine == "pi" else normalize_legacy_message(msg)
            if snapshot_raw:
                try:
                    norm["state_snapshot"] = json.loads(snapshot_raw)
                except (json.JSONDecodeError, TypeError):
                    pass
            action_trajectory.append(norm)
        except json.JSONDecodeError:
            action_trajectory.append({"role": "unknown", "raw": data_str[:200]})
    if engine == "pi":
        action_trajectory = [m for m in action_trajectory if not _is_export_noise_message(m)]
    else:
        action_trajectory = filter_out_stream_events(action_trajectory)

    raw_msgs_for_sf: List[dict] = []
    for data_str, _, _ in messages_rows:
        try:
            raw_msgs_for_sf.append(json.loads(data_str))
        except json.JSONDecodeError:
            pass
    initial_task_instruction = extract_initial_task_instruction(
        action_trajectory,
        last_prompt or "",
        pi_session_file=resolve_pi_session_file(pi_session_file_raw, raw_msgs_for_sf),
    )
    node_id_to_segment = segment_trajectory_by_persisted_node_prompts(action_trajectory, workflow_tree)
    if engine != "pi" and not node_id_to_segment:
        node_id_to_segment = segment_trajectory_by_resume_points(action_trajectory, workflow_tree)

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
        # Legacy: raw.message.content[];  Pi: raw.blocks[]
        blocks = (raw.get("message") or {}).get("content") or raw.get("blocks") or []
        for block in blocks:
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


def _brain_edit_human_entry(m: dict) -> dict:
    """Weight-format row for Brain dialog save (compact payload; use ``diff`` for changes)."""
    return {"type": "brain_edit", "round_index": None}


def _prior_snapshot_message(msgs: List[dict], before_idx: int) -> Optional[dict]:
    """Latest message before ``before_idx`` that carries a ``state_snapshot``."""
    for j in range(before_idx - 1, -1, -1):
        if isinstance(msgs[j].get("state_snapshot"), dict):
            return msgs[j]
    return None


def _snapshot_brain_map(norm: Optional[dict], key: str) -> Dict[str, str]:
    """Filename → utf-8 text for ``memory`` or ``skill`` from a message snapshot."""
    if not isinstance(norm, dict):
        return {}
    snap = norm.get("state_snapshot")
    if not isinstance(snap, dict):
        return {}
    mp = snap.get(key)
    if not isinstance(mp, dict):
        return {}
    out: Dict[str, str] = {}
    for fn, content in mp.items():
        name = str(fn).strip()
        if not name:
            continue
        out[name] = content if isinstance(content, str) else ""
    return out


_SENTENCE_BREAK_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _lines_for_diff_sentence_split(text: str) -> List[str]:
    """Split text into one line per sentence (with trailing ``\\n``) for change alignment."""
    if not text or not str(text).strip():
        return []
    lines_out: List[str] = []
    for para in re.split(r"\n\s*\n", str(text).strip()):
        para_flat = " ".join(para.split())
        if not para_flat:
            continue
        for sent in _SENTENCE_BREAK_SPLIT_RE.split(para_flat):
            s = sent.strip()
            if s:
                lines_out.append(s + "\n")
    return lines_out if lines_out else [str(text).strip() + "\n"]


def _trunc_anno(s: str, max_len: int = 160) -> str:
    t = str(s).strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3].rstrip() + "..."


def _word_change_snippets(before_line: str, after_line: str, *, trunc: int = 100) -> str:
    """Short token-level edits between two strings (whitespace tokenization)."""
    wa = before_line.split()
    wb = after_line.split()
    if wa == wb:
        return "(same tokens)"
    sm = difflib.SequenceMatcher(a=wa, b=wb, autojunk=False)
    bits: List[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            left = " ".join(wa[i1:i2])
            right = " ".join(wb[j1:j2])
            bits.append(f"{_trunc_anno(left, trunc)} → {_trunc_anno(right, trunc)}")
        elif tag == "delete":
            left = " ".join(wa[i1:i2])
            bits.append(f"- {_trunc_anno(left, trunc)}")
        elif tag == "insert":
            right = " ".join(wb[j1:j2])
            bits.append(f"+ {_trunc_anno(right, trunc)}")
    return ", ".join(bits) if bits else "(no token edits)"


def _compact_file_edit_annotation(before_s: str, after_s: str, path_disp: str) -> str:
    """Localized edit summary: align sentences, then token-level snippets only (no full-file diff)."""
    if before_s == after_s:
        return f"(no textual change) path={path_disp}"

    bl = [ln.rstrip("\n\r") for ln in _lines_for_diff_sentence_split(before_s)]
    al = [ln.rstrip("\n\r") for ln in _lines_for_diff_sentence_split(after_s)]

    sm = difflib.SequenceMatcher(a=bl, b=al, autojunk=False)
    parts: List[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            b_join = " ".join(bl[i1:i2])
            a_join = " ".join(al[j1:j2])
            if b_join.strip() == a_join.strip():
                continue
            ws = _word_change_snippets(b_join, a_join)
            if ws == "(same tokens)":
                continue
            parts.append(f"• {_trunc_anno(ws, 700)}")
        elif tag == "delete":
            b_join = " ".join(bl[i1:i2])
            parts.append(f"• removed: {_trunc_anno(b_join, 200)}")
        elif tag == "insert":
            a_join = " ".join(al[j1:j2])
            parts.append(f"• added: {_trunc_anno(a_join, 200)}")

    if not parts:
        return f"(no localized changes detected) path={path_disp}"

    body = "\n".join(parts)
    max_out = 8000
    if len(body) > max_out:
        body = body[: max_out - 50].rstrip() + f"\n… (annotation truncated, path={path_disp})"
    return f"path={path_disp}\n{body}"


def _file_edit_diff_from_raw_strings(
    before_raw: Optional[str],
    after_raw: Optional[str],
    path: str,
) -> str:
    """Compact localized edit annotation (sentence align + token-level snippets), not a full unified diff."""
    before_s = before_raw if isinstance(before_raw, str) else ""
    after_s = after_raw if isinstance(after_raw, str) else ""
    path_disp = str(path).strip() or "(path)"
    return _compact_file_edit_annotation(before_s, after_s, path_disp)


def _brain_edit_map_diff_sections(
    category: str,
    before_map: Dict[str, str],
    after_map: Dict[str, str],
) -> List[str]:
    """Per-file compact diffs for a brain map (memory or skill)."""
    sections: List[str] = []
    for fn in sorted(set(before_map) | set(after_map)):
        before_s = before_map.get(fn, "")
        after_s = after_map.get(fn, "")
        if before_s == after_s:
            continue
        path_disp = f"{category}/{fn}"
        sections.append(_file_edit_diff_from_raw_strings(before_s, after_s, path_disp))
    return sections


def _brain_edit_memory_skill_diff(
    prior: Optional[dict],
    current: dict,
    *,
    max_chars: int = 12_000,
) -> str:
    """Compact brain-edit summary: memory and skill markdown only (vs prior snapshot)."""
    mem_before = _snapshot_brain_map(prior, "memory")
    mem_after = _snapshot_brain_map(current, "memory")
    sk_before = _snapshot_brain_map(prior, "skill")
    sk_after = _snapshot_brain_map(current, "skill")

    sections: List[str] = []
    sections.extend(_brain_edit_map_diff_sections("memory", mem_before, mem_after))
    sections.extend(_brain_edit_map_diff_sections("skill", sk_before, sk_after))

    if not sections:
        return "(no textual change) path=brain"

    body = "\n\n".join(sections)
    if len(body) > max_chars:
        body = body[: max_chars - 50].rstrip() + "\n… (annotation truncated, path=brain)"
    return body


def _workflow_tree_json_for_diff(wf: Any) -> str:
    """Stable JSON text for workflow tree diffing (LLM-native step shape)."""
    if not isinstance(wf, list):
        return ""
    return json.dumps(wf, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _edit_workflow_diff_from_trees(wf_before: Any, wf_after: Any) -> str:
    """Diff serialized workflow steps (plan edit)."""
    before_native = _to_llm_native_tree(wf_before) if wf_before is not None else []
    after_native = _to_llm_native_tree(wf_after) if wf_after is not None else []
    return _file_edit_diff_from_raw_strings(
        _workflow_tree_json_for_diff(before_native),
        _workflow_tree_json_for_diff(after_native),
        "workflow",
    )


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


def _verifiers_from_snapshot(snapshot_msg: Optional[dict], node_id: str) -> Optional[List[dict]]:
    """Extract per-criterion verifier results for *node_id* from a snapshot message.

    Returns a list like [{"criterion": "...", "status": True/False}, …], or None
    if no snapshot / node is found.  The snapshot stores each criterion as
    ``{"criterion": str, "status": "success" | "failure"}`` (``buildExportEnvironmentSnapshot``
    collapses the in-memory ``verifierMarks`` ("check"/"cross"/undefined) into
    these strings via ``verifierStatusForExport`` in message-state-snapshot.ts),
    so we read ``status`` directly instead of looking up ``verifierMarks``.
    """
    if snapshot_msg is None:
        return None
    tree = _snapshot_workflow_tree(snapshot_msg)
    if tree is None:
        return None
    node = _find_node_in_tree(tree, node_id)
    if node is None:
        return None
    criteria = node.get("verifiers") or []
    result = []
    for c in criteria:
        if isinstance(c, dict):
            crit = str(c.get("criterion", ""))
            status = c.get("status") == "success"
        else:
            crit = str(c)
            status = False
        result.append({"criterion": crit, "status": status})
    return result if result else None


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
        # Legacy: stop_reason nested under message; Pi: stopReason at top level
        stop = raw.get("message", {}).get("stop_reason") or raw.get("stopReason")
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


# ── Pi engine constants ──────────────────────────────────────────────────────
PI_AGENT_TYPES = {"system_init", "assistant", "tool_result", "run_result"}
LEGACY_AGENT_TYPES = {"system", "assistant", "user", "result"}
ALL_AGENT_MSG_TYPES = PI_AGENT_TYPES | LEGACY_AGENT_TYPES


def _slim_pi_block(b: dict) -> dict:
    bt = b.get("type")
    if bt == "text":
        return {"type": "text", "text": b.get("text", "")}
    if bt == "thinking":
        return {"type": "thinking", "thinking": b.get("thinking", "")}
    if bt == "tool_use":
        return {"type": "tool_use", "id": b.get("id", ""), "name": b.get("name", ""), "input": b.get("input", {})}
    return b


def _is_pi_engine(all_msgs: List[dict]) -> bool:
    for m in all_msgs:
        if m.get("engine") == "pi":
            return True
        if m.get("type") == "system_init" and m.get("engine") == "pi":
            return True
    return False


def _slim_raw_message(raw: dict) -> dict:
    """Strip infrastructure metadata from a raw SDK message, keeping only
    what LLM produced (assistant) or what LLM sees (user/tool_result)."""
    t = raw.get("type")
    engine = raw.get("engine")

    # ── Pi engine format ──
    if engine == "pi":
        if t == "assistant":
            blocks = [_slim_pi_block(b) for b in (raw.get("blocks") or []) if isinstance(b, dict)]
            out: Dict[str, Any] = {"type": "assistant", "engine": "pi", "blocks": blocks}
            sr = raw.get("stopReason")
            if sr is not None:
                out["stopReason"] = sr
            ts = raw.get("timestamp")
            if ts is not None:
                out["timestamp"] = ts
            return out
        if t == "tool_result":
            out: Dict[str, Any] = {
                "type": "tool_result",
                "toolUseId": raw.get("toolUseId", ""),
                "toolName": raw.get("toolName", ""),
                "content": raw.get("content", ""),
                "isError": raw.get("isError", False),
            }
            ts = raw.get("timestamp")
            if ts is not None:
                out["timestamp"] = ts
            return out
        if t == "system_init":
            out = {"type": "system_init", "engine": "pi"}
            if raw.get("model"):
                out["model"] = raw["model"]
            if raw.get("provider"):
                out["provider"] = raw["provider"]
            return out
        if t == "run_result":
            out = {"type": "run_result", "status": raw.get("status", "")}
            if raw.get("usage"):
                out["usage"] = raw["usage"]
            return out
        # fall through for verifier_label etc.

    # ── Legacy format ──
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
    if t == "update_verifiers":
        return {"type": "update_verifiers", "nodeId": raw.get("nodeId", "")}

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


# ── Pi system prompt & tool schemas (hardcoded from Pi mono source) ───────────
# These are code constants that never appear in DB; we splice them into exports.

PI_SYSTEM_PROMPT_TEMPLATE = (
    "You are an expert coding assistant operating inside pi, a coding agent harness. "
    "You help users by reading files, executing commands, editing code, and writing new files.\n\n"
    "Available tools:\n"
    "- read: Read file contents\n"
    "- bash: Execute bash commands (ls, grep, find, etc.)\n"
    "- edit: Edit a file using exact text replacement\n"
    "- write: Write content to a file\n"
    "- grep: Search file contents for patterns (respects .gitignore)\n"
    "- find: Find files by glob pattern (respects .gitignore)\n"
    "- ls: List directory contents\n\n"
    "In addition to the tools above, you may have access to other custom tools depending on the project.\n\n"
    "Guidelines:\n"
    "- Prefer grep/find/ls tools over bash for file exploration (faster, respects .gitignore)\n"
    "- Be concise in your responses\n"
    "- Show file paths clearly when working with files"
)


def _pi_system_prompt(cwd: Optional[str] = None) -> str:
    prompt = PI_SYSTEM_PROMPT_TEMPLATE
    import datetime
    date_str = datetime.date.today().isoformat()
    prompt += f"\nCurrent date: {date_str}"
    if cwd:
        prompt += f"\nCurrent working directory: {cwd}"
    return prompt


PI_TOOL_SCHEMAS: List[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
                    "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
                    "limit": {"type": "number", "description": "Maximum number of lines to read"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
                    "content": {"type": "string", "description": "Content to write to the file"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Edit a single file using exact text replacement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
                    "edits": {
                        "type": "array",
                        "description": "One or more targeted replacements.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "oldText": {"type": "string", "description": "Exact text to find."},
                                "newText": {"type": "string", "description": "Replacement text."},
                            },
                            "required": ["oldText", "newText"],
                        },
                    },
                },
                "required": ["path", "edits"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command in the current working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Bash command to execute"},
                    "timeout": {"type": "number", "description": "Timeout in seconds (optional)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents for a pattern. Respects .gitignore.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Search pattern (regex or literal string)"},
                    "path": {"type": "string", "description": "Directory or file to search (default: current directory)"},
                    "glob": {"type": "string", "description": "Filter files by glob pattern, e.g. '*.ts'"},
                    "ignoreCase": {"type": "boolean", "description": "Case-insensitive search (default: false)"},
                    "literal": {"type": "boolean", "description": "Treat pattern as literal string (default: false)"},
                    "context": {"type": "number", "description": "Lines of context before and after each match"},
                    "limit": {"type": "number", "description": "Maximum number of matches to return"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find",
            "description": "Search for files by glob pattern. Respects .gitignore.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern to match files"},
                    "path": {"type": "string", "description": "Directory to search in (default: current directory)"},
                    "limit": {"type": "number", "description": "Maximum number of results"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ls",
            "description": "List directory contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory to list (default: current directory)"},
                    "limit": {"type": "number", "description": "Maximum number of entries to return"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workflow_plan",
            "description": "Register a hierarchical workflow plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "description": "Top-level workflow steps",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "outputFiles": {"type": "array", "items": {"type": "string"}},
                                "verifiers": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {"criterion": {"type": "string"}},
                                    },
                                },
                                "children": {"type": "array", "items": {"type": "object"}},
                            },
                            "required": ["description"],
                        },
                    },
                },
                "required": ["tasks"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user_question",
            "description": "Ask the operator a structured question and wait for the answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "string"},
                                "header": {"type": "string"},
                                "options": {"type": "array", "items": {"type": "string"}},
                                "multiSelect": {"type": "boolean"},
                            },
                            "required": ["question"],
                        },
                    },
                },
                "required": ["questions"],
            },
        },
    },
]


def pi_messages_to_openai(all_msgs: List[dict]) -> List[dict]:
    """Convert Pi DB messages to OpenAI chat format.

    Follows the same field mapping as tinker-provider.ts contextToBridgeMessages():
      - assistant.blocks[] → role:assistant content + tool_calls (+ thinking if present)
      - tool_result → role:tool with tool_call_id
      - user_prompt → role:user
      - system_init / run_result → skipped (metadata, not LLM turns)
    """
    oai: List[dict] = []
    for m in all_msgs:
        t = m.get("type")

        if t == "user_prompt":
            oai.append({"role": "user", "content": m.get("prompt", "")})
            continue

        if t == "assistant":
            if m.get("stopReason") == "error":
                continue
            blocks = m.get("blocks") or []
            text_parts: List[str] = []
            tool_calls: List[dict] = []
            thinking_parts: List[str] = []
            for b in blocks:
                bt = b.get("type")
                if bt == "text":
                    text_parts.append(b.get("text", ""))
                elif bt == "thinking":
                    thinking_parts.append(b.get("thinking", ""))
                elif bt == "tool_use":
                    inp = b.get("input", {})
                    tool_calls.append({
                        "id": b.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": b.get("name", ""),
                            "arguments": json.dumps(inp, ensure_ascii=False) if isinstance(inp, dict) else str(inp),
                        },
                    })
            msg: Dict[str, Any] = {"role": "assistant"}
            content_str = "\n".join(text_parts).strip()
            if content_str:
                msg["content"] = content_str
            else:
                msg["content"] = None
            if tool_calls:
                msg["tool_calls"] = tool_calls
            if thinking_parts:
                msg["thinking"] = "\n".join(thinking_parts)
            oai.append(msg)
            continue

        if t == "tool_result":
            content = m.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    p.get("text", str(p)) if isinstance(p, dict) else str(p)
                    for p in content
                )
            oai.append({
                "role": "tool",
                "tool_call_id": m.get("toolUseId", ""),
                "name": m.get("toolName", ""),
                "content": str(content),
            })
            continue

    return oai


def pi_messages_to_openai_with_envs(
    raw_env_pairs: List[Tuple[dict, dict]],
) -> List[dict]:
    """Like ``pi_messages_to_openai`` but each (raw, env) pair contributes ``env`` to its OAI message.

    The mapping is one-to-one for ``assistant`` (non-error) / ``tool_result`` / ``user_prompt`` rows;
    ``assistant`` rows with ``stopReason == "error"`` and ``system_init`` / ``run_result`` are still
    skipped. ``env`` is attached as the canonical 4-key dict under the ``environment`` key on the
    emitted OAI message, so REINFORCE can read final verifier statuses and memory/skill induction can
    diff successive ``environment.file`` snapshots without consulting other arrays.
    """
    oai: List[dict] = []
    for raw, env in raw_env_pairs:
        t = raw.get("type")

        if t == "user_prompt":
            msg: Dict[str, Any] = {"role": "user", "content": raw.get("prompt", "")}
            if isinstance(env, dict):
                msg["environment"] = env
            oai.append(msg)
            continue

        if t == "assistant":
            if raw.get("stopReason") == "error":
                continue
            blocks = raw.get("blocks") or []
            text_parts: List[str] = []
            tool_calls: List[dict] = []
            thinking_parts: List[str] = []
            for b in blocks:
                bt = b.get("type")
                if bt == "text":
                    text_parts.append(b.get("text", ""))
                elif bt == "thinking":
                    thinking_parts.append(b.get("thinking", ""))
                elif bt == "tool_use":
                    inp = b.get("input", {})
                    tool_calls.append({
                        "id": b.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": b.get("name", ""),
                            "arguments": json.dumps(inp, ensure_ascii=False) if isinstance(inp, dict) else str(inp),
                        },
                    })
            msg = {"role": "assistant"}
            content_str = "\n".join(text_parts).strip()
            msg["content"] = content_str if content_str else None
            if tool_calls:
                msg["tool_calls"] = tool_calls
            if thinking_parts:
                msg["thinking"] = "\n".join(thinking_parts)
            if isinstance(env, dict):
                msg["environment"] = env
            oai.append(msg)
            continue

        if t == "tool_result":
            content = raw.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    p.get("text", str(p)) if isinstance(p, dict) else str(p)
                    for p in content
                )
            msg = {
                "role": "tool",
                "tool_call_id": raw.get("toolUseId", ""),
                "name": raw.get("toolName", ""),
                "content": str(content),
            }
            if isinstance(env, dict):
                msg["environment"] = env
            oai.append(msg)
            continue

    return oai


def _collect_written_paths_from_segment(seg: List[dict]) -> set[str]:
    """Return paths the agent **successfully** invoked ``write`` / ``edit`` on.

    Two filters are applied:
    1. Only ``write`` / ``edit`` tool_use calls (not ``read``, ``bash``, etc.).
    2. Only calls whose corresponding ``tool_result`` did NOT have ``isError=true``.
       If a ``write``/``edit`` failed (e.g. "Could not find the exact text"), the
       path is excluded so a stale on-disk file is not mistakenly treated as a
       produced artifact.

    The ``toolUseId`` from the assistant block is matched against the nearest
    ``tool_result`` message in the segment.
    """
    # Pass 1: collect (toolUseId → path) for write/edit calls
    id_to_path: Dict[str, str] = {}
    for e in seg:
        raw = e.get("raw") if isinstance(e, dict) else None
        if not isinstance(raw, dict):
            continue
        if raw.get("type") != "assistant":
            continue
        blocks = raw.get("blocks") or (raw.get("message") or {}).get("content") or []
        for b in blocks:
            if not isinstance(b, dict) or b.get("type") != "tool_use":
                continue
            name = str(b.get("name") or "").lower()
            if name not in ("write", "edit"):
                continue
            inp = b.get("input")
            if inp is None:
                inp = b.get("arguments")
            if isinstance(inp, str):
                try:
                    inp = json.loads(inp)
                except json.JSONDecodeError:
                    continue
            if not isinstance(inp, dict):
                continue
            p = inp.get("path")
            tid = b.get("id") or b.get("toolUseId")
            if isinstance(p, str) and p.strip():
                if isinstance(tid, str) and tid:
                    id_to_path[tid] = p.strip()
                else:
                    # No id tracking possible; still collect but can't filter by error
                    id_to_path[f"_noid_{p.strip()}"] = p.strip()

    # Pass 2: remove paths whose tool_result has isError=true
    failed_ids: set[str] = set()
    for e in seg:
        raw = e.get("raw") if isinstance(e, dict) else None
        if not isinstance(raw, dict):
            continue
        # Pi format: standalone tool_result message
        if raw.get("type") == "tool_result":
            if raw.get("isError"):
                tid = raw.get("toolUseId") or raw.get("tool_use_id") or ""
                if tid:
                    failed_ids.add(tid)
        # Legacy format: user message wrapping tool_result blocks
        elif raw.get("type") == "user":
            for block in (raw.get("message") or {}).get("content") or []:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                # Legacy tool_results mark errors via non-empty "is_error" or content prefix
                if block.get("is_error"):
                    tid = block.get("tool_use_id") or ""
                    if tid:
                        failed_ids.add(tid)

    paths: set[str] = {
        path for tid, path in id_to_path.items()
        if tid not in failed_ids
    }
    return paths


def _output_files_for_segment(
    written_paths: set[str],
    snap_start: Optional[dict],
    snap_end: Optional[dict],
) -> List[dict]:
    """Filter ``snapshot.file`` rows to files that changed during this segment.

    Two conditions must both hold:
    1. The path is in ``written_paths`` (agent called write/edit on it successfully).
    2. The file content at segment-end differs from content at segment-start.
       This excludes corner cases where ``write`` succeeded but wrote the same bytes
       (content unchanged). Segment-start snapshot is taken from the **previous**
       segment's end, or None for the first segment.

    Matches by exact path or basename to handle cwd-relative vs absolute mix.
    Skips rows without text content.
    """
    if not written_paths or not isinstance(snap_end, dict):
        return []
    files_end = snap_end.get("file")
    if not isinstance(files_end, list):
        return []

    # Build lookup for start-of-segment content; None means "file didn't exist before"
    start_content: Dict[str, Optional[str]] = {}
    if isinstance(snap_start, dict):
        for f in (snap_start.get("file") or []):
            if not isinstance(f, dict):
                continue
            fp = str(f.get("path", "")).strip()
            if fp:
                start_content[fp] = f.get("content") if isinstance(f.get("content"), str) else None

    wanted_basenames = {os.path.basename(p.replace("\\", "/")) for p in written_paths}
    out: List[dict] = []
    for f in files_end:
        if not isinstance(f, dict):
            continue
        fp = str(f.get("path", "")).strip()
        if not fp:
            continue
        if fp not in written_paths and os.path.basename(fp.replace("\\", "/")) not in wanted_basenames:
            continue
        content = f.get("content")
        if not isinstance(content, str):
            continue
        # Skip if content is identical to what it was at segment start
        prev = start_content.get(fp)
        if prev is not None and prev == content:
            continue
        entry: Dict[str, Any] = {"path": fp, "content": content}
        cs = f.get("content_source")
        if isinstance(cs, str):
            entry["content_source"] = cs
        out.append(entry)
    return out


def _last_snapshot_in_range(
    snapshots: List[Optional[dict]], start: int, end: int
) -> Optional[dict]:
    """Return the latest non-null snapshot in ``snapshots[start:end]``."""
    end = min(end, len(snapshots))
    for i in range(end - 1, max(start, 0) - 1, -1):
        snap = snapshots[i]
        if isinstance(snap, dict):
            return snap
    return None


def _is_pi_raw_noise(msg: dict) -> bool:
    """Pi rows omitted when running synthetic file-edit detection (raw DB shape)."""
    t = msg.get("type")
    if t in ("stream_event", "system_init", "run_result", "node_completed", "llm_debug"):
        return True
    if t == "assistant" and msg.get("stopReason") == "error":
        return True
    return False


def _insert_synthetic_file_edits(
    msgs: List[dict], cwd: Optional[str], *, is_pi: bool
) -> List[dict]:
    """Insert preview ``file_edit`` rows missing from the DB (matches default exporter)."""
    if not msgs:
        return msgs
    try:
        from export_task_sessions_context import (  # type: ignore
            inject_synthetic_human_edits,
            reclassify_auto_verifier_edits,
        )
    except ImportError:
        from tasks.export_task_sessions_context import (  # type: ignore
            inject_synthetic_human_edits,
            reclassify_auto_verifier_edits,
        )
    if not is_pi:
        return inject_synthetic_human_edits(filter_out_stream_events(msgs), cwd)

    non_noise_idx = [i for i, m in enumerate(msgs) if not _is_pi_raw_noise(m)]
    norm_msgs: List[dict] = []
    for i in non_noise_idx:
        norm = normalize_pi_message(msgs[i])
        if "state_snapshot" in msgs[i]:
            norm["state_snapshot"] = msgs[i]["state_snapshot"]
        norm_msgs.append(norm)
    injected = inject_synthetic_human_edits(reclassify_auto_verifier_edits(norm_msgs), cwd)
    inserts: List[Tuple[int, dict]] = []
    ni = 0
    for im in injected:
        if im.get("_synthetic_before") is not None:
            inserts.append((non_noise_idx[ni], dict(im)))
        elif not im.get("_synthetic"):
            ni += 1
    out = list(msgs)
    for idx, row in reversed(inserts):
        out.insert(idx, row)
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
    try:
        row = cursor.execute(
            """SELECT id, title, workflow_tree, last_prompt, cwd, engine, pi_session_file
               FROM sessions WHERE id = ?""",
            (session_id,),
        ).fetchone()
    except sqlite3.OperationalError:
        row = cursor.execute(
            """SELECT id, title, workflow_tree, last_prompt, cwd
               FROM sessions WHERE id = ?""",
            (session_id,),
        ).fetchone()
        if row:
            row = (*row, None, None)
    if not row:
        return None
    sid, title, workflow_tree_raw, last_prompt, cwd, db_engine, pi_session_file_raw = row
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

    is_pi = _is_pi_engine(all_msgs)
    all_msgs = _insert_synthetic_file_edits(all_msgs, export_cwd, is_pi=is_pi)

    # ── Per-message timelines (carry-forward state from snapshots) ──
    # ``wf_timeline_all[i]`` / ``file_timeline_all[i]`` / ``mem_timeline_all[i]`` /
    # ``sk_timeline_all[i]`` reflect the state right after applying message ``i``'s
    # ``state_snapshot`` (or the previous carried-forward state when ``i`` has no snapshot).
    # These power the per-OAI-message ``environment`` and per-human_trajectory ``environment``
    # so REINFORCE rewards can read final verifier statuses and memory/skill induction can diff
    # successive ``environment.file`` snapshots without consulting other arrays.
    wf_timeline_all = _build_workflow_timeline(all_msgs)
    file_timeline_all = _build_file_timeline(all_msgs)
    mem_timeline_all, sk_timeline_all = _build_memory_skill_timeline(all_msgs)

    def _env_at(idx: int) -> dict:
        """Return the canonical {workflow, file, memory, skill} environment at ``all_msgs[idx]``.

        ``idx < 0`` (e.g., before any message) yields an empty environment so the synthetic
        leading user prompt of the very first segment still gets a well-typed shape.
        """
        if idx < 0 or idx >= len(all_msgs):
            return {"workflow": [], "file": [], "memory": {}, "skill": {}}
        return _build_message_environment(
            all_msgs[idx],
            cwd=export_cwd,
            workflow_override=wf_timeline_all[idx],
            file_override=file_timeline_all[idx],
            memory=mem_timeline_all[idx],
            skill=sk_timeline_all[idx],
        )

    initial_user_msgs = [
        {"role": "user", "type": "user_prompt", "prompt": m.get("prompt", "")}
        for m in all_msgs
        if m.get("type") == "user_prompt"
    ]
    initial_task_instruction = extract_initial_task_instruction(
        initial_user_msgs,
        last_prompt or "",
        pi_session_file=resolve_pi_session_file(pi_session_file_raw, all_msgs),
    )

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
        if path:
            node_starts.append((i, path))

    # ── Build planning task_unit ──
    planning_end = node_starts[0][0] if node_starts else len(all_msgs)
    planning_msgs = all_msgs[:planning_end]

    planning_agent_traj_raw: List[dict] = []
    # Absolute index in ``all_msgs`` for each entry pushed to ``planning_agent_traj_raw`` (Pi only;
    # Legacy merging shifts indices, so the env-aware OAI mapping below is gated on ``is_pi``).
    planning_msg_indices: List[int] = []
    planning_human_traj: List[dict] = []
    prev_workflow_snapshot: Optional[List[dict]] = None
    planning_initial_prompt_consumed = False
    planning_follow_up_counter = 0
    last_snapshot_msg: Optional[dict] = None

    for local_i, m in enumerate(planning_msgs):
        i_abs = local_i  # planning_msgs starts at all_msgs[0]
        t = m.get("type")
        if t in ALL_AGENT_MSG_TYPES:
            planning_agent_traj_raw.append({"raw": {k: v for k, v in m.items() if k not in ("_ts", "state_snapshot")}})
            planning_msg_indices.append(i_abs)
            snap = m.get("state_snapshot")
            if isinstance(snap, dict):
                last_snapshot_msg = m

        elif t == "user_prompt":
            p = m.get("prompt", "")
            if isinstance(p, str) and is_backend_node_user_prompt(p):
                continue
            if isinstance(p, str) and not planning_initial_prompt_consumed:
                if _prompts_equal(strip_interface_user_prompt(p), initial_task_instruction):
                    planning_initial_prompt_consumed = True
                    continue
            planning_human_traj.append({
                "type": "follow_up",
                "round_index": planning_follow_up_counter,
                "prompt": p if isinstance(p, str) else "",
                "environment": _env_at(i_abs),
            })
            planning_follow_up_counter += 1

        elif t == "file_edit":
            fe_path = m.get("path", "")
            edited_content = _snapshot_file_content(m, fe_path, export_cwd)
            original_content = None
            synthetic_before = m.get("_synthetic_before")
            if isinstance(synthetic_before, str):
                original_content = synthetic_before
            else:
                prior = _prior_snapshot_message(all_msgs, i_abs)
                if prior is not None:
                    original_content = _snapshot_file_content(prior, fe_path, export_cwd)
            planning_human_traj.append({
                "type": "file_edit",
                "round_index": None,
                "path": fe_path,
                "diff": _file_edit_diff_from_raw_strings(original_content, edited_content, fe_path),
                "environment": _env_at(i_abs),
            })

        elif t == "edit_workflow":
            wf_after = _snapshot_workflow_tree(m)
            entry: Dict[str, Any] = {
                "type": "edit_workflow",
                "round_index": None,
            }
            wf_before = prev_workflow_snapshot
            if wf_before is not None:
                entry["workflow_tree_before"] = wf_before
            if wf_after is not None:
                entry["workflow_tree_after"] = wf_after
                prev_workflow_snapshot = wf_after
            entry["environment"] = _env_at(i_abs)
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
            entry["environment"] = _env_at(i_abs)
            planning_human_traj.append(entry)

        elif t == "brain_edit":
            be_entry = _brain_edit_human_entry(m)
            be_entry["diff"] = _brain_edit_memory_skill_diff(
                _prior_snapshot_message(all_msgs, i_abs), m
            )
            be_entry["environment"] = _env_at(i_abs)
            planning_human_traj.append(be_entry)
            if m.get("state_snapshot"):
                last_snapshot_msg = m

    if is_pi:
        planning_agent_traj_merged = planning_agent_traj_raw
    else:
        planning_agent_traj_merged = _merge_partial_assistant_messages(planning_agent_traj_raw)
    workflow_tree_generated = _extract_workflow_tree_from_tool_use(planning_agent_traj_merged)
    workflow_tree_after_planning = prev_workflow_snapshot or workflow_tree
    if is_pi:
        planning_agent_traj_final = planning_agent_traj_merged
    else:
        planning_agent_traj_final = _merge_parallel_tool_results(planning_agent_traj_merged)
    planning_agent_traj = [{"raw": _slim_raw_message(e["raw"])} for e in planning_agent_traj_final]

    # Normalize planning human_trajectories workflow snapshots to LLM native format
    for h in planning_human_traj:
        if h.get("type") == "edit_workflow":
            if "workflow_tree_before" in h:
                h["workflow_tree_before"] = _to_llm_native_tree(h["workflow_tree_before"])
            if "workflow_tree_after" in h:
                h["workflow_tree_after"] = _to_llm_native_tree(h["workflow_tree_after"])
            h["diff"] = _edit_workflow_diff_from_trees(
                h.get("workflow_tree_before"),
                h.get("workflow_tree_after"),
            )

    # Reconstruct the planning first-turn prompt:
    # WORKFLOW_PLAN_INSTRUCTION + user's initial task instruction
    # NOTE: memoryPrefix is not available from DB; will be accurate once
    # effective_prompt is persisted (TODO: modify src/electron).
    planning_prompt = WORKFLOW_PLAN_INSTRUCTION + initial_task_instruction

    # Build planning trajectories (planning is always a single trajectory).
    # The synthetic leading user message gets the start-of-segment environment so memory/skill
    # induction can compare it against the first assistant turn's post-state.
    planning_user_env = _env_at(0) if planning_msgs else {"workflow": [], "file": [], "memory": {}, "skill": {}}
    leading_user_msg: Dict[str, Any] = {"role": "user", "content": planning_prompt}
    if isinstance(planning_user_env, dict):
        leading_user_msg["environment"] = planning_user_env
    if is_pi:
        # Pi: no merging happens, so ``planning_agent_traj`` order matches ``planning_agent_traj_raw``
        # one-for-one and ``planning_msg_indices`` aligns with each entry.
        pi_pairs: List[Tuple[dict, dict]] = []
        for k, e in enumerate(planning_agent_traj):
            raw = e["raw"]
            if raw.get("type") not in ("assistant", "tool_result", "user_prompt"):
                continue
            idx_abs = planning_msg_indices[k] if k < len(planning_msg_indices) else -1
            pi_pairs.append((raw, _env_at(idx_abs)))
        planning_messages = [leading_user_msg] + pi_messages_to_openai_with_envs(pi_pairs)
    else:
        planning_messages = [leading_user_msg]
    planning_traj_entry: Dict[str, Any] = {
        "prompt": planning_prompt,
        "messages": planning_messages,
    }

    planning_unit: Dict[str, Any] = {
        "intent": "planning",
        "agent_trajectories": [planning_traj_entry],
        "human_trajectories": planning_human_traj,
        "verifiers": [],
        "workflow_tree_generated": workflow_tree_generated,
        "workflow_tree_final": _to_llm_native_tree(workflow_tree_after_planning),
    }

    # ── Build execution task_units ──
    task_units: List[dict] = [planning_unit]

    for seg_idx, (start_i, path) in enumerate(node_starts):
        end_i = node_starts[seg_idx + 1][0] if seg_idx + 1 < len(node_starts) else len(all_msgs)
        node_msgs = all_msgs[start_i:end_i]
        node = path_to_node.get(path)
        if node is None:
            task_desc = path
            for line in (node_msgs[0].get("prompt", "") if node_msgs else "").splitlines():
                if line.startswith("Task: "):
                    task_desc = line.removeprefix("Task: ").strip() or path
                    break
            node = {
                "id": "",
                "description": task_desc,
                "outputFiles": [],
                "verifiers": [],
            }
        node_id = node.get("id", "")
        node_desc = node.get("description", "")

        agent_traj_raw: List[dict] = []
        # Parallel to ``agent_traj_raw``: per-message ``state_snapshot`` (or None).
        # Used to recover the segment-end on-disk file state for artifact-only
        # completion construction. Pi: indices align with ``agent_traj`` since no
        # merging happens; Legacy: indices may shift after merging — we therefore
        # only emit ``output_files`` for Pi sessions (gated below).
        agent_traj_snapshots: List[Optional[dict]] = []
        # Absolute index in ``all_msgs`` for each entry pushed to ``agent_traj_raw`` (Pi only;
        # Legacy merging shifts indices, so per-OAI-message env mapping below is gated on ``is_pi``).
        agent_msg_indices: List[int] = []
        human_traj: List[dict] = []
        round_counter = 0
        node_prompt_consumed = False
        node_first_turn_prompt: Optional[str] = None
        # Absolute index of the node's first-turn ``Proceed with: …`` prompt (start-of-segment).
        node_first_turn_idx: Optional[int] = None
        last_snapshot_msg: Optional[dict] = None
        # Track where follow_up prompts split the trajectory.
        # Each entry: (index into agent_traj_raw at time of split, prompt text, abs idx of prompt msg)
        follow_up_splits: List[Tuple[int, str, int]] = []

        for local_i, m in enumerate(node_msgs):
            i_abs = start_i + local_i
            t = m.get("type")

            if t == "user_prompt":
                p = m.get("prompt", "")
                if isinstance(p, str) and is_backend_node_user_prompt(p):
                    if not node_prompt_consumed:
                        node_prompt_consumed = True
                        node_first_turn_prompt = p
                        node_first_turn_idx = i_abs
                        continue
                    follow_up_splits.append((len(agent_traj_raw), p, i_abs))
                    human_traj.append({
                        "type": "follow_up",
                        # 0-based index of this follow-up (aligns with trajectories[1], [2], …)
                        "round_index": len(follow_up_splits) - 1,
                        "prompt": p,
                        "environment": _env_at(i_abs),
                    })
                    continue
                follow_up_splits.append((len(agent_traj_raw), p if isinstance(p, str) else "", i_abs))
                human_traj.append({
                    "type": "follow_up",
                    "round_index": len(follow_up_splits) - 1,
                    "prompt": p if isinstance(p, str) else "",
                    "environment": _env_at(i_abs),
                })
                continue

            if t in ALL_AGENT_MSG_TYPES:
                raw_clean = {k: v for k, v in m.items() if k not in ("_ts", "state_snapshot")}
                agent_traj_raw.append({"raw": raw_clean})
                snap = m.get("state_snapshot")
                agent_traj_snapshots.append(snap if isinstance(snap, dict) else None)
                agent_msg_indices.append(i_abs)
                if t == "system" and m.get("subtype") == "init":
                    round_counter += 1
                if t == "system_init":
                    round_counter += 1
                if isinstance(snap, dict):
                    last_snapshot_msg = m

            elif t == "verifier_label" or t == "update_verifiers":
                raw_clean = {k: v for k, v in m.items() if k not in ("_ts", "state_snapshot")}
                agent_traj_raw.append({"raw": raw_clean})
                snap = m.get("state_snapshot")
                agent_traj_snapshots.append(snap if isinstance(snap, dict) else None)
                agent_msg_indices.append(i_abs)
                if isinstance(snap, dict):
                    last_snapshot_msg = m

            elif t == "file_edit":
                fe_path = m.get("path", "")
                edited_content = _snapshot_file_content(m, fe_path, export_cwd)
                original_content = None
                synthetic_before = m.get("_synthetic_before")
                if isinstance(synthetic_before, str):
                    original_content = synthetic_before
                else:
                    prior = _prior_snapshot_message(all_msgs, i_abs)
                    if prior is not None:
                        original_content = _snapshot_file_content(prior, fe_path, export_cwd)
                human_traj.append({
                    "type": "file_edit",
                    "round_index": None,
                    "path": fe_path,
                    "diff": _file_edit_diff_from_raw_strings(original_content, edited_content, fe_path),
                    "environment": _env_at(i_abs),
                })

            elif t == "edit_workflow":
                wf_after = _snapshot_workflow_tree(m)
                prior = _prior_snapshot_message(all_msgs, i_abs)
                wf_before = _snapshot_workflow_tree(prior) if prior else None
                entry = {"type": "edit_workflow", "round_index": None}
                if wf_before is not None:
                    entry["workflow_tree_before"] = _to_llm_native_tree(wf_before)
                if wf_after is not None:
                    entry["workflow_tree_after"] = _to_llm_native_tree(wf_after)
                entry["diff"] = _edit_workflow_diff_from_trees(wf_before, wf_after)
                entry["environment"] = _env_at(i_abs)
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
                entry["environment"] = _env_at(i_abs)
                human_traj.append(entry)

            elif t == "brain_edit":
                be_entry = _brain_edit_human_entry(m)
                be_entry["diff"] = _brain_edit_memory_skill_diff(
                    _prior_snapshot_message(all_msgs, i_abs), m
                )
                be_entry["environment"] = _env_at(i_abs)
                human_traj.append(be_entry)
                if m.get("state_snapshot"):
                    last_snapshot_msg = m

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

        if is_pi:
            agent_traj_final = agent_traj_raw
        else:
            agent_traj_merged = _merge_partial_assistant_messages(agent_traj_raw)
            agent_traj_final = _merge_parallel_tool_results(agent_traj_merged)
        agent_traj = [{"raw": _slim_raw_message(e["raw"])} for e in agent_traj_final]

        # ── Split agent_trajectory into per-trajectory segments ──
        # ``follow_up_splits`` was populated during the message loop:
        # each entry is (index_in_agent_traj_raw, prompt_text, abs_msg_idx).
        # Since merging / slimming preserves ordering and count for Pi
        # (no merging) and approximately for Legacy, the indices still
        # align with agent_traj.
        # verifier_label rows survive replaceMessages(); order matches trajectory order for this node
        vl_snaps_for_node = [
            m for m in all_msgs
            if m.get("type") == "verifier_label"
            and m.get("nodeId") == node_id
            and m.get("state_snapshot")
        ]

        # Build raw (prompt, segment) pairs first so we can align vl → traj by
        # time window before building entries. Track each segment's [start, end)
        # range over ``agent_traj_raw`` so the parallel ``agent_traj_snapshots``
        # array (Pi only) and tool_use scan can be sliced consistently.
        # ``seg_prompt_indices[i]`` is the absolute ``all_msgs`` index of the user
        # prompt that opened segment ``i`` (the node's first ``Proceed with: …`` row
        # for segment 0, and each follow-up's user_prompt for later segments). This
        # lets the synthetic leading user message in each ``messages`` list carry the
        # carried-forward environment as it stood when the human typed that prompt.
        raw_segments: List[Tuple[str, List[dict]]] = []
        seg_ranges: List[Tuple[int, int]] = []
        seg_prompt_indices: List[int] = []
        if not follow_up_splits:
            raw_segments.append((node_first_turn_prompt or "", agent_traj))
            seg_ranges.append((0, len(agent_traj)))
            seg_prompt_indices.append(node_first_turn_idx if node_first_turn_idx is not None else start_i)
        else:
            first_end = follow_up_splits[0][0]
            raw_segments.append((node_first_turn_prompt or "", agent_traj[:first_end]))
            seg_ranges.append((0, first_end))
            seg_prompt_indices.append(node_first_turn_idx if node_first_turn_idx is not None else start_i)
            for si, (split_idx, follow_prompt, fu_abs_idx) in enumerate(follow_up_splits):
                next_end = follow_up_splits[si + 1][0] if si + 1 < len(follow_up_splits) else len(agent_traj)
                raw_segments.append((follow_prompt, agent_traj[split_idx:next_end]))
                seg_ranges.append((split_idx, next_end))
                seg_prompt_indices.append(fu_abs_idx)

        # For each segment, compute the max Pi ``timestamp`` among its
        # assistant / tool_result entries.  This is the real LLM call time
        # and survives replaceMessages() (unlike DB created_at, which is
        # rewritten).  Used only for the precise alignment path below.
        def _seg_end_ts(seg: List[dict]) -> Optional[int]:
            best: Optional[int] = None
            for e in seg:
                raw = e.get("raw") or {}
                if raw.get("type") not in ("assistant", "tool_result"):
                    continue
                ts = raw.get("timestamp")
                if isinstance(ts, (int, float)):
                    if best is None or ts > best:
                        best = int(ts)
            return best

        seg_end_ts_list = [_seg_end_ts(seg) for (_, seg) in raw_segments]

        # Align vl → segment.  New sessions: each vl carries
        # ``runEndTimestamp`` (set right after labelVerifiers writes it).
        # Old sessions: fall back to end-aligned mapping (last vl → last
        # segment, second-last vl → second-last segment, …) which is what
        # the user actually cares about for training.
        vl_for_segment: List[Optional[dict]] = [None] * len(raw_segments)
        use_timestamp_alignment = any(
            isinstance(vl.get("runEndTimestamp"), (int, float)) for vl in vl_snaps_for_node
        )
        if use_timestamp_alignment:
            for vl in vl_snaps_for_node:
                run_end = vl.get("runEndTimestamp")
                if not isinstance(run_end, (int, float)):
                    continue
                # Pick the last segment whose end_ts is < run_end; ignore
                # segments that have no Pi timestamps (stuck trajectories).
                chosen: Optional[int] = None
                for k, end_ts in enumerate(seg_end_ts_list):
                    if end_ts is None:
                        continue
                    if end_ts < run_end:
                        chosen = k
                if chosen is not None:
                    vl_for_segment[chosen] = vl
        else:
            N = len(raw_segments)
            M = len(vl_snaps_for_node)
            shift = N - M
            for k in range(N):
                idx = k - shift
                if 0 <= idx < M:
                    vl_for_segment[k] = vl_snaps_for_node[idx]

        def _build_traj_entry(
            prompt: str,
            seg: List[dict],
            vl_snap: Optional[dict],
            *,
            seg_msg_indices: List[int],
            prompt_msg_idx: int,
        ) -> Dict[str, Any]:
            # Synthetic leading user message: state at the moment the human posted ``prompt``.
            leading_msg: Dict[str, Any] = {"role": "user", "content": prompt}
            leading_env = _env_at(prompt_msg_idx)
            if isinstance(leading_env, dict):
                leading_msg["environment"] = leading_env
            if is_pi:
                # Pi: ``seg`` aligns with ``seg_msg_indices`` one-for-one (no merging upstream).
                pi_pairs: List[Tuple[dict, dict]] = []
                for k, e in enumerate(seg):
                    raw = e["raw"]
                    if raw.get("type") not in ("assistant", "tool_result", "user_prompt"):
                        continue
                    idx_abs = seg_msg_indices[k] if k < len(seg_msg_indices) else -1
                    pi_pairs.append((raw, _env_at(idx_abs)))
                messages = [leading_msg] + pi_messages_to_openai_with_envs(pi_pairs)
            else:
                messages = [leading_msg]
            entry: Dict[str, Any] = {"prompt": prompt, "messages": messages}
            v = _verifiers_from_snapshot(vl_snap, node_id)
            if v is not None:
                entry["verifiers"] = v
            return entry

        agent_trajectories: List[dict] = []
        prev_seg_snap: Optional[dict] = None  # end-of-previous-segment snapshot for diff
        for i, (prompt, seg) in enumerate(raw_segments):
            s_start_i, s_end_i = seg_ranges[i]
            seg_indices_slice = agent_msg_indices[s_start_i:s_end_i]
            entry = _build_traj_entry(
                prompt,
                seg,
                vl_for_segment[i],
                seg_msg_indices=seg_indices_slice,
                prompt_msg_idx=seg_prompt_indices[i],
            )
            # Artifact-only completion support (Pi only — Legacy merge shifts
            # ``agent_traj`` indices relative to the snapshot array). For each
            # segment:
            #  - ``snap_start``: last snapshot from the *previous* segment (None for k=0)
            #  - ``snap_end``: last snapshot within this segment
            #  - ``written_paths``: only paths the agent called write/edit on **successfully**
            #    (isError calls are excluded); combined with a content-diff vs snap_start
            #    to avoid including files that didn't actually change.
            if is_pi:
                s_start, s_end = seg_ranges[i]
                snap_end = _last_snapshot_in_range(agent_traj_snapshots, s_start, s_end)
                written = _collect_written_paths_from_segment(agent_traj_raw[s_start:s_end])
                output_files = _output_files_for_segment(written, prev_seg_snap, snap_end)
                if output_files:
                    entry["output_files"] = output_files
                # Advance the "previous" snapshot so the next segment can diff against it
                if snap_end is not None:
                    prev_seg_snap = snap_end
            agent_trajectories.append(entry)

        unit: Dict[str, Any] = {
            "intent": node_desc,
            "agent_trajectories": agent_trajectories,
            "human_trajectories": human_traj,
            "verifiers": verifiers,
        }
        task_units.append(unit)

    # ── Extract model name from messages ──
    model_name = ""
    for m in all_msgs:
        if m.get("type") == "system_init" and m.get("model"):
            model_name = m["model"]
            break
        if m.get("type") == "assistant" and m.get("model"):
            model_name = m["model"]
            break
        if m.get("type") == "system" and m.get("model"):
            model_name = m["model"]
            break

    result: Dict[str, Any] = {
        "uuid": sid,
        "name": title or "",
        "initial_task_instruction": initial_task_instruction,
        "model": model_name,
        "task_units": task_units,
    }
    if is_pi:
        result["system_prompt"] = _pi_system_prompt(export_cwd)
        result["tool_schemas"] = PI_TOOL_SCHEMAS
    return result


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
        default="weight",
        help=(
            'Export format: "default" (human-readable trajectory) or "weight" '
            "(OAI messages + human_trajectories for training; every OAI message in "
            "agent_trajectories[].messages and every human_trajectories entry carries "
            "an `environment` dict of {workflow,file,memory,skill} so REINFORCE rewards "
            "can read final verifier statuses and memory/skill induction can diff "
            "successive environment.file snapshots)."
        ),
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
