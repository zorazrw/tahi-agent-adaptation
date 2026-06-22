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
        "tool_result": optional str; SDK tool outcomes, file-edit annotations, or compact verifier-line
        annotations for ``edit_verifier()`` (one ``{criterion}: {status}`` line per verifier, then targeted ``•`` edits),
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

``brain_edit()`` rows record Brain dialog saves (memory + skill files on disk) and also set
``tool_result``: memory and skill maps are compared to the prior snapshot per file, and the rest of the
step ``environment`` (workflow tree + output files) is diffed the same way — compact ``path=…`` /
``•`` annotations like file ``edit("…")`` tool results.
``edit_workflow()``, ``edit_verifier()``, and preview ``edit("…")`` (file save) rows persist the same
``environment`` shape: ``workflow`` (nested steps; each node has its own ``verifiers`` with ``status``),
``file`` (output paths + content), and ``memory`` / ``skill`` (each a map of ``file-name`` → file text
as injected at snapshot time). ``edit_verifier()`` rows also set ``tool_result``: each verifier is flattened
to one line ``{criterion}: {status}``, then a compact ``path=verifiers`` /
``•`` annotation like file ``edit("…")`` tool results.

Per-step ``environment`` (workflow nodes, each node’s ``verifiers`` with ``status``, output files, memory/skills)
comes from ``messages.state_snapshot`` when recorded: human ``user_prompt``; SDK ``user`` tool
results; turn ``result``; and a synthetic ``verifier_label`` row after the verifier LM updates marks
(exported as agent ``verify("…")`` with the workflow node id). Pure ``message("…")`` steps (user or agent) omit ``environment``.
The export reapplies a carried-forward workflow tree from snapshots so, after ``edit_workflow`` removes
a step, later steps do not retain that step’s verifiers or output files (and ``file`` rows are aligned
to the current tree). Older DBs without ``state_snapshot`` fall back to one end-of-session snapshot.

The second is always the agent ``workflow_plan({initial query})``; its ``environment`` uses the workflow tree
as of the last persisted snapshot before the first ``edit_workflow`` (or WorkflowPlan tool input
with ``normalizeRoots``), not necessarily ``sessions.workflow_tree`` after later edits. Every step
``status`` is ``pending``; each verifier is ``{"criterion", "status": "failure"}``; ``file`` maps
paths to ``null``.

Database location (Electron userData):
- macOS: ~/Library/Application Support/Agent Cowork/sessions.db
- Windows: %APPDATA%\\Agent Cowork\\sessions.db
- Linux: ~/.config/Agent Cowork/sessions.db

Output format
-------------
Exports ``uuid``, ``name``, ``model``, ``task``, ``system_prompt``, ``tool_schemas``, and ``task_units``.
Each task_unit has ``actor`` (``user`` or ``agent``), ``trajectory`` (default-style action rows without
per-step ``actor`` or ``environment``), and ``environment`` (workflow, files, memory, skill). Agent units also include
``prompt`` (full first-turn instruction for planning, then user or node prompts). The synthetic ``workflow_plan(...)``
step is its own agent task_unit, followed by execution agent units.

Usage:
  conda activate code   # optional: use "code" env
  python export_task_sessions.py [--db PATH] [--output FILE] [--session-id ID] [--task-category CATEGORY]
  # Use AGENT_COWORK_DB to override DB path:
  AGENT_COWORK_DB=/path/to/sessions.db python export_task_sessions.py
  # Only abstract-writing sessions:
  python export_task_sessions.py --task-category abstract-writing -o out.json
"""

from __future__ import annotations

import argparse
import base64
import copy
import difflib
import json
import logging
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

# Cap per-file inlined content to keep exports bounded (bytes).
MAX_OUTPUT_FILE_BYTES = 500_000
_FILE_EDIT_DESCRIBE_MAX_CHARS = 12_000

_FILE_EDIT_DESCRIBE_SYSTEM = """You summarize direct human edits to a file, for a training log.

Part 1 — what changed: describe in plain language what would look different on the rendered result (page, chart, document), not as abstract developer jargon.

Good summary: "Moved titles lower", "Removed gridlines and y-axis ticks", "Colored the All/750 bars light blue".
Bad summary: "Updated layout mode attributes", "Applied inline transform styles".

Part 2 — why: infer the user's preference or intention behind the edit — what they seem to want, care about, or be correcting (visual style, layout, emphasis, accuracy, polish, etc.). Ground this in the changes but state it as reusable intent, not a repeat of Part 1.

Good intention: "Wants a cleaner chart with less clutter and more readable labels", "Prefers light blue for the baseline All/750 bar to distinguish it from the rest".
Bad intention: "Edited the HTML file", "Changed some CSS properties".

Rules:
- Part 1: 1-3 short sentences, concrete (colors, sizes, positions, labels, elements added/removed).
- Part 2: 1-2 short sentences on preference/intent; plain language, no code or markup.
- Plain text only — no markdown, no bullet lists, no code fences, no reasoning or analysis steps.
- Reply on exactly two lines:
  Part 1: <what changed>
  Part 2: <inferred preference or intent>"""


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
    """Match ``verifier_status`` / Electron snapshot: check→success, cross→failure, else unchecked."""
    if plan_snapshot:
        return "failure"
    if mark == "check":
        return "success"
    if mark == "cross":
        return "failure"
    return "unchecked"


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
    return {
        "workflow": wf,
        "file": files,
        "memory": mem,
        "skill": sk,
    }


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
        merged: Dict[str, Any] = {"workflow": copy.deepcopy(wf), "file": copy.deepcopy(files)}
    else:
        wf_fb = default_env.get("workflow")
        if isinstance(wf_fb, list):
            merged = {"workflow": copy.deepcopy(wf_fb), "file": copy.deepcopy(files)}
        else:
            return default_env, False
    return merged, True


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
    Verifier criteria and status live on each node under ``workflow[].verifiers``.
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
    base.pop("verifier", None)
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


def _first_backend_node_prompt(norm_msgs: List[dict]) -> str:
    """First persisted node instruction (Proceed with: … + Task:), for LM prompts after planning."""
    for m in norm_msgs:
        if m.get("role") != "user" or m.get("type") != "user_prompt":
            continue
        p = m.get("prompt", "")
        if isinstance(p, str) and is_backend_node_user_prompt(p):
            return p
    return ""


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
        trajectory_row("agent", f"workflow_plan({json.dumps(initial_query, ensure_ascii=False)})", plan_env),
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
            snap_wf_uv = _snapshot_workflow_tree(m)
            wf_after_uv = snap_wf_uv if isinstance(snap_wf_uv, list) else wo_uv
            step_env, _snap = environment_for_norm(
                m,
                final_env,
                cwd=cwd_val,
                workflow_override=wf_after_uv,
                memory=m_uv,
                skill=s_uv,
            )
            wf_prev = wf_timeline[idx - 1] if idx > 0 else None
            ev_tr = _edit_verifier_tool_result_from_snapshots(wf_prev, wf_after_uv)
            traj.append(trajectory_row("agent", "edit_verifier()", step_env, tool_result=ev_tr))
            idx += 1
            continue

        if m.get("type") in ("edit_workflow", "edit_plan"):
            wo_e = wf_timeline[idx] if idx < len(wf_timeline) else None
            m_e = mem_timeline[idx] if idx < len(mem_timeline) else {}
            s_e = sk_timeline[idx] if idx < len(sk_timeline) else {}
            snap_wf_e = _snapshot_workflow_tree(m)
            wf_after_e = snap_wf_e if isinstance(snap_wf_e, list) else wo_e
            step_env, _snap = environment_for_norm(
                m,
                final_env,
                cwd=cwd_val,
                workflow_override=wf_after_e,
                memory=m_e,
                skill=s_e,
            )
            wf_prev_e = m.get("_synthetic_wf_before")
            if not isinstance(wf_prev_e, list):
                wf_prev_e = wf_timeline[idx - 1] if idx > 0 else None
            ew_tr = _edit_workflow_tool_result_from_snapshots(wf_prev_e, wf_after_e)
            traj.append(trajectory_row("user", "edit_workflow()", step_env, tool_result=ew_tr))
            idx += 1
            continue
        if m.get("role") == "user" and m.get("type") == "edit_verifier":
            wo_ev = wf_timeline[idx] if idx < len(wf_timeline) else None
            m_ev = mem_timeline[idx] if idx < len(mem_timeline) else {}
            s_ev = sk_timeline[idx] if idx < len(sk_timeline) else {}
            snap_wf_ev = _snapshot_workflow_tree(m)
            # Prefer this row's persisted snapshot as "after" (matches DB write order); timeline should match.
            wf_after_ev = snap_wf_ev if isinstance(snap_wf_ev, list) else wo_ev
            step_env, _snap = environment_for_norm(
                m,
                final_env,
                cwd=cwd_val,
                workflow_override=wf_after_ev,
                memory=m_ev,
                skill=s_ev,
            )
            wf_prev_u = m.get("_synthetic_wf_before")
            if not isinstance(wf_prev_u, list):
                wf_prev_u = wf_timeline[idx - 1] if idx > 0 else None
            ev_tr_u = _edit_verifier_tool_result_from_snapshots(wf_prev_u, wf_after_ev)
            traj.append(trajectory_row("user", "edit_verifier()", step_env, tool_result=ev_tr_u))
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
            act = f"edit({json.dumps(p, ensure_ascii=False)})"
            synthetic_before = m.get("_synthetic_before")
            if isinstance(synthetic_before, str):
                before_raw = synthetic_before
            else:
                prior = _prior_snapshot_message(msgs, idx)
                before_raw = _snapshot_file_content(prior, p, cwd_val) if prior else None
            after_raw = _snapshot_file_content(m, p, cwd_val)
            diff_blob = _file_edit_diff_from_raw_strings(before_raw, after_raw, p)
            traj.append(trajectory_row("user", act, step_env, tool_result=diff_blob))
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
            prior_br = _prior_snapshot_message(msgs, idx)
            be_tr = _brain_edit_tool_result_from_snapshots(prior_br, m, cwd=cwd_val)
            traj.append(trajectory_row("user", "brain_edit()", step_env, tool_result=be_tr))
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
    *,
    db_root: Optional[Path] = None,
    session_id: Optional[str] = None,
) -> Optional[str]:
    """Resolve pi jsonl on disk; remap absolute paths from another machine to bundle layout."""
    raw_paths: List[str] = []
    if isinstance(db_pi_session_file, str) and db_pi_session_file.strip():
        raw_paths.append(db_pi_session_file.strip())
    for m in raw_msgs or []:
        if m.get("type") == "system_init":
            sf = m.get("sessionFile")
            if isinstance(sf, str) and sf.strip():
                raw_paths.append(sf.strip())

    tried: Set[str] = set()
    bundle_root = db_root.resolve() if db_root is not None else None

    for raw in raw_paths:
        if raw in tried:
            continue
        tried.add(raw)
        p = Path(raw)
        if p.is_file():
            return str(p)

        if bundle_root is None:
            continue

        parts = p.parts
        for i, part in enumerate(parts):
            if part == "pi-agent":
                candidate = bundle_root.joinpath(*parts[i:])
                if candidate.is_file():
                    return str(candidate)

        if session_id:
            candidate = bundle_root / "pi-agent" / "sessions" / session_id / p.name
            if candidate.is_file():
                return str(candidate)

        if not p.is_absolute():
            candidate = bundle_root / p
            if candidate.is_file():
                return str(candidate)

    if bundle_root and session_id:
        session_dir = bundle_root / "pi-agent" / "sessions" / session_id
        if session_dir.is_dir():
            for child in sorted(session_dir.glob("*.jsonl")):
                return str(child)

    return None


def extract_initial_task_instruction(
    action_trajectory: List[dict],
    fallback: str,
    *,
    pi_session_file: Optional[str] = None,
    stored_initial_prompt: Optional[str] = None,
) -> str:
    if isinstance(stored_initial_prompt, str):
        stripped = strip_interface_user_prompt(stored_initial_prompt)
        if stripped and not is_backend_node_user_prompt(stripped):
            return stripped

    from_pi = load_initial_task_from_pi_session_file(pi_session_file)
    if from_pi:
        return from_pi

    for m in action_trajectory:
        if m.get("role") == "user" and m.get("type") == "user_prompt":
            prompt = m.get("prompt", "")
            if is_backend_node_user_prompt(prompt):
                continue
            stripped = strip_interface_user_prompt(prompt)
            if stripped:
                return stripped
    fb = strip_interface_user_prompt(fallback)
    if fb and not is_backend_node_user_prompt(fb):
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
    if msg.get("type") in ("edit_workflow", "edit_plan"):
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
    ``llm_debug``: Tinker bridge request/response telemetry (not part of the agent conversation).
    ``assistant`` with ``stopReason: "error"``: truncated/failed Pi retry attempts.
    """
    if msg.get("role") != "agent":
        return False
    raw = msg.get("raw")
    if not isinstance(raw, dict):
        return False
    t = raw.get("type")
    if t in ("stream_event", "system", "system_init", "run_result", "node_completed", "llm_debug"):
        return True
    if t == "assistant" and raw.get("stopReason") == "error":
        return True
    return False


def _edit_verifier_is_agent_driven(msgs: List[dict], idx: int) -> bool:
    """
  True when ``edit_verifier`` was almost certainly emitted by auto-refinement (after
  user_prompt / file_edit / brain_edit), not a manual sidebar verifier edit.
  """
    if msgs[idx].get("type") != "edit_verifier":
        return False
    for j in range(idx - 1, -1, -1):
        prev = msgs[j]
        pt = prev.get("type")
        if pt == "edit_verifier":
            continue
        if pt in ("user_prompt", "file_edit", "brain_edit"):
            return True
        if pt in ("edit_workflow", "edit_plan"):
            return False
        if pt == "update_verifiers":
            return True
    return False


def reclassify_auto_verifier_edits(msgs: List[dict]) -> List[dict]:
    """Map auto-refinement ``edit_verifier`` rows to agent ``update_verifiers`` for export."""
    out: List[dict] = []
    for i, m in enumerate(msgs):
        row = dict(m)
        if row.get("type") == "edit_verifier" and _edit_verifier_is_agent_driven(msgs, i):
            row["role"] = "agent"
            row["type"] = "update_verifiers"
        out.append(row)
    return out


def normalize_pi_message(msg: dict) -> dict:
    msg_type = msg.get("type")
    role = msg.get("role")
    if msg_type == "user_prompt":
        return {"role": "user", "type": "user_prompt", "prompt": msg.get("prompt", "")}
    if msg_type in ("edit_workflow", "edit_plan"):
        return {"role": "user", "type": "edit_workflow"}
    if msg_type == "update_verifiers":
        return {"role": "agent", "type": "update_verifiers"}
    if msg_type == "edit_verifier":
        return {"role": "user", "type": "edit_verifier"}
    if msg_type == "file_edit":
        return {"role": "user", "type": "file_edit", "path": msg.get("path", "")}
    if msg_type == "brain_edit":
        return {"role": "user", "type": "brain_edit"}
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
    if role == "user":
        return {"role": "user", "type": msg_type, "raw": msg}
    return {"role": "agent", "type": msg_type, "raw": msg}


def filter_out_stream_events(trajectory: List[dict]) -> List[dict]:
    """Drop streaming chunks and SDK system/bootstrap messages before building trajectories."""
    return [msg for msg in trajectory if not _is_export_noise_message(msg)]


def _session_initial_prompt(cursor: sqlite3.Cursor, session_id: str) -> Optional[str]:
    try:
        row = cursor.execute(
            "SELECT initial_prompt FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row and row[0]:
            return str(row[0])
    except sqlite3.OperationalError:
        pass
    return None


def extract_session(
    cursor: sqlite3.Cursor, session_id: str, *, db_root: Optional[Path] = None
) -> Optional[dict]:
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
        pi_session_file=resolve_pi_session_file(
            pi_session_file_raw, raw_msgs_for_sf, db_root=db_root, session_id=session_id
        ),
        stored_initial_prompt=_session_initial_prompt(cursor, session_id),
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
    """Workflow for the synthetic ``workflow_plan(...)`` row: pre-edit snapshot or earliest post-plan snapshot."""
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
    Environment for the synthetic ``workflow_plan(...)`` trajectory row: workflow as it was right after
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
    return {
        "workflow": plan_wf,
        "file": files,
        "memory": mem,
        "skill": sk,
    }


def _snapshot_workflow_tree(norm: dict) -> Optional[List[dict]]:
    """Extract workflow tree from a message's state_snapshot."""
    snap = norm.get("state_snapshot")
    if not isinstance(snap, dict):
        return None
    wf = snap.get("workflow")
    return wf if isinstance(wf, list) else None


def _brain_edit_human_entry(m: dict) -> dict:
    """Weight-format row for Brain dialog save; includes memory/skill maps from the step snapshot."""
    snap = m.get("state_snapshot")
    mem: dict = {}
    sk: dict = {}
    if isinstance(snap, dict):
        mp = snap.get("memory")
        if isinstance(mp, dict):
            mem = copy.deepcopy(mp)
        sp = snap.get("skill")
        if isinstance(sp, dict):
            sk = copy.deepcopy(sp)
    return {"type": "brain_edit", "round_index": None, "memory": mem, "skill": sk}


def _prior_snapshot_message(msgs: List[dict], before_idx: int) -> Optional[dict]:
    """Latest message before ``before_idx`` that carries a ``state_snapshot`` (for file before/after)."""
    for j in range(before_idx - 1, -1, -1):
        if isinstance(msgs[j].get("state_snapshot"), dict):
            return msgs[j]
    return None


def _snapshot_files_content_map(norm: dict, cwd: Optional[str]) -> Dict[str, str]:
    """Map canonical path key -> utf-8 file content from a message snapshot."""
    snap = norm.get("state_snapshot")
    if not isinstance(snap, dict):
        return {}
    files = snap.get("file")
    if not isinstance(files, list):
        return {}
    out: Dict[str, str] = {}
    cwd_opt = str(cwd).strip() if cwd else None
    for f in files:
        if not isinstance(f, dict):
            continue
        if f.get("content_encoding") == "base64":
            continue
        content = f.get("content")
        if not isinstance(content, str):
            continue
        path_s = str(f.get("path", "")).strip()
        if not path_s:
            continue
        key = next(iter(_path_variant_strings(cwd_opt, path_s)), path_s.replace("\\", "/"))
        out[key] = content
    return out


def _display_path_from_key(path_key: str) -> str:
    return os.path.basename(path_key.replace("\\", "/")) or path_key


def _workflow_structure_signature(wf: Optional[list]) -> str:
    """Workflow tree shape for human edit detection (ignores verifier status)."""

    def walk(nodes: Any) -> List[Any]:
        if not isinstance(nodes, list):
            return []
        rows: List[Any] = []
        for n in nodes:
            if not isinstance(n, dict):
                continue
            rows.append(
                {
                    "description": str(n.get("description", "")),
                    "outputFiles": [str(x) for x in (n.get("outputFiles") or [])],
                    "children": walk(n.get("children")),
                }
            )
        return rows

    return json.dumps(walk(wf) if isinstance(wf, list) else [], sort_keys=True)


def _human_verifier_signature(wf: Optional[list]) -> str:
    """Verifier criterion text + marks only (ignore pass/fail status churn)."""

    def walk(nodes: Any) -> List[Any]:
        if not isinstance(nodes, list):
            return []
        rows: List[Any] = []
        for n in nodes:
            if not isinstance(n, dict):
                continue
            crits = n.get("verifiers") or []
            marks = n.get("verifierMarks") or []
            ver_rows: List[Any] = []
            for i, v in enumerate(crits):
                if isinstance(v, dict):
                    crit = _one_line_verifier_field(v.get("criterion", ""))
                elif isinstance(v, str):
                    crit = _one_line_verifier_field(v)
                else:
                    crit = ""
                mark = marks[i] if i < len(marks) and marks[i] is not None else None
                ver_rows.append({"criterion": crit, "mark": mark})
            rows.append({"id": n.get("id"), "verifiers": ver_rows, "children": walk(n.get("children"))})
        return rows

    return json.dumps(walk(wf) if isinstance(wf, list) else [], sort_keys=True)


def _message_agent_wrote_file_basename(m: dict, path_key: str) -> bool:
    """True if this agent row includes a write/edit tool call for ``path_key``'s basename."""
    raw = m.get("raw")
    if not isinstance(raw, dict):
        return False
    want_base = os.path.basename(path_key.replace("\\", "/"))
    if not want_base:
        return False
    blocks = raw.get("blocks") or (raw.get("message") or {}).get("content") or []
    if not isinstance(blocks, list):
        return False
    for b in blocks:
        if not isinstance(b, dict):
            continue
        if b.get("type") not in ("tool_use", "toolCall"):
            continue
        name = str(b.get("name", "")).lower()
        if name not in ("write", "edit"):
            continue
        inp = b.get("input") or b.get("arguments") or {}
        if not isinstance(inp, dict):
            continue
        fp = str(inp.get("path") or inp.get("file_path") or "").strip()
        if fp and os.path.basename(fp.replace("\\", "/")) == want_base:
            return True
    return False


def inject_synthetic_human_edits(msgs: List[dict], cwd: Optional[str]) -> List[dict]:
    """
    Insert synthetic human-action rows when snapshots show edits but the DB has no
    ``file_edit`` / ``edit_workflow`` / ``edit_verifier`` message (preview save without
    recording, or verifier edits lost between snapshots).
    """
    if not msgs:
        return msgs
    out: List[dict] = []
    prev_files: Dict[str, str] = {}
    prev_wf_struct: Optional[str] = None
    prev_ver_sig: Optional[str] = None
    prev_wf_tree: Optional[list] = None

    for i, m in enumerate(msgs):
        m_type = m.get("type")
        curr_files = _snapshot_files_content_map(m, cwd)
        curr_wf = _snapshot_workflow_tree(m)
        curr_wf_struct = _workflow_structure_signature(curr_wf) if curr_wf is not None else None
        curr_ver_sig = _human_verifier_signature(curr_wf) if curr_wf is not None else None

        if m_type != "file_edit":
            for path_key, after_content in curr_files.items():
                before_content = prev_files.get(path_key)
                if before_content is None or before_content == after_content:
                    continue
                # Only skip when the immediately prior row is the agent write/edit for this file.
                if i > 0 and _message_agent_wrote_file_basename(msgs[i - 1], path_key):
                    continue
                out.append(
                    {
                        "role": "user",
                        "type": "file_edit",
                        "path": _display_path_from_key(path_key),
                        "state_snapshot": copy.deepcopy(m.get("state_snapshot")),
                        "_synthetic_before": before_content,
                    }
                )

        if m_type not in ("edit_workflow", "edit_plan") and curr_wf_struct is not None:
            if prev_wf_struct is not None and curr_wf_struct != prev_wf_struct:
                out.append(
                    {
                        "role": "user",
                        "type": "edit_workflow",
                        "state_snapshot": copy.deepcopy(m.get("state_snapshot")),
                        "_synthetic": True,
                    }
                )
        if (
            m_type not in ("edit_verifier", "update_verifiers")
            and curr_ver_sig is not None
            and prev_ver_sig is not None
            and curr_ver_sig != prev_ver_sig
            and (prev_wf_struct is None or curr_wf_struct == prev_wf_struct)
        ):
            prev_m = msgs[i - 1] if i > 0 else None
            agent_driven = m.get("role") == "agent" or (
                prev_m is not None
                and prev_m.get("type") in ("file_edit", "brain_edit", "user_prompt")
            )
            out.append(
                {
                    "role": "agent" if agent_driven else "user",
                    "type": "update_verifiers" if agent_driven else "edit_verifier",
                    "state_snapshot": copy.deepcopy(m.get("state_snapshot")),
                    "_synthetic": True,
                    "_synthetic_wf_before": copy.deepcopy(prev_wf_tree),
                }
            )

        out.append(m)
        prev_files = curr_files if curr_files else prev_files
        if curr_wf_struct is not None:
            prev_wf_struct = curr_wf_struct
        if curr_ver_sig is not None:
            prev_ver_sig = curr_ver_sig
        if curr_wf is not None:
            prev_wf_tree = copy.deepcopy(curr_wf)

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


def _clip_file_content_for_lm(text: Optional[str], *, max_chars: int = _FILE_EDIT_DESCRIBE_MAX_CHARS) -> str:
    s = text if isinstance(text, str) else ""
    s = s.strip()
    if not s:
        return "(empty)"
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 20].rstrip() + "\n...[truncated]"


def _clean_file_edit_description(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _extract_part_line_content(line: str) -> Optional[str]:
    """Return text after ``Part 1:`` or ``Part 2:`` if the label appears in the first 15 chars."""
    head = line[:15].lower()
    for label in ("part 1:", "part 2:"):
        idx = head.find(label)
        if idx < 0:
            continue
        return line[idx + len(label) :].strip() or None
    return None


def _parse_file_edit_description(raw: str) -> Optional[str]:
    """Merge ``Part 1:`` / ``Part 2:`` lines (label in first 15 chars) into one description."""
    text = _clean_file_edit_description(raw)
    chunks: List[str] = []
    for line in text.splitlines():
        content = _extract_part_line_content(line)
        if content:
            chunks.append(content)
    return " ".join(chunks) if chunks else None


def describe_file_edit_with_llm(
    runtime: Any,
    path: str,
    original: Optional[str],
    edited: Optional[str],
) -> Optional[str]:
    """Ask the Pi runtime LM to summarize a human file edit (original vs edited)."""
    from dataclasses import replace

    from pi_llm import runtime_llm_text

    path_disp = str(path).strip() or "(path)"
    user = (
        f"File path: {path_disp}\n"
        "1) Describe what would look different on the rendered page or output.\n"
        "2) Infer the user's preference or intention behind the edit.\n\n"
        f"BEFORE:\n{_clip_file_content_for_lm(original)}\n\n"
        f"AFTER:\n{_clip_file_content_for_lm(edited)}"
    )
    runtime_plain = (
        replace(runtime, reasoning=None)
        if getattr(runtime, "reasoning", None)
        else runtime
    )
    try:
        raw = runtime_llm_text(
            runtime_plain, _FILE_EDIT_DESCRIBE_SYSTEM, user, max_tokens=1024
        )
    except Exception:
        logger.exception("File-edit description LM failed for %s", path_disp)
        return None
    print(f"File-edit description raw ({path_disp}):\n{raw}\n", file=sys.stderr)
    desc = _parse_file_edit_description(raw)
    if desc is None:
        logger.warning(
            "No Part 1:/Part 2: lines in file-edit LM response for %s", path_disp
        )
    return desc


def file_edit_human_entry(
    *,
    path: str,
    original_content: Optional[str],
    edited_content: Optional[str],
    environment: dict,
    describe_runtime: Optional[Any] = None,
) -> Dict[str, Any]:
    diff = _file_edit_diff_from_raw_strings(original_content, edited_content, path)
    entry: Dict[str, Any] = {
        "type": "file_edit",
        "round_index": None,
        "path": path,
        "original": original_content,
        "edited": edited_content,
        "diff": diff,
        "environment": environment,
    }
    if describe_runtime is not None:
        desc = describe_file_edit_with_llm(
            describe_runtime,
            path,
            original_content,
            edited_content,
        )
        if desc is not None:
            entry["description"] = desc
    return entry


def _one_line_verifier_field(val: Any) -> str:
    """Collapse internal newlines/whitespace for a single-line verifier representation."""
    if val is None:
        return ""
    s = val if isinstance(val, str) else str(val)
    return " ".join(s.split())


def _flatten_workflow_verifier_lines(wf: Optional[list]) -> List[str]:
    """
    Preorder flatten: one line per verifier, ``{criterion}: {status}`` (no node id; criterion is one line).
    """
    out: List[str] = []

    def walk(nodes: Any) -> None:
        if not isinstance(nodes, list):
            return
        for n in nodes:
            if not isinstance(n, dict):
                continue
            for v in n.get("verifiers") or []:
                crit = ""
                st = ""
                if isinstance(v, dict):
                    crit = _one_line_verifier_field(v.get("criterion", ""))
                    st = _one_line_verifier_field(v.get("status", ""))
                elif isinstance(v, str) and v.strip():
                    crit = _one_line_verifier_field(v)
                line = f"{crit}: {st}" if crit else f"(empty criterion): {st}"
                out.append(line)
            walk(n.get("children"))

    if isinstance(wf, list):
        walk(wf)
    return out


def _verifier_line_criterion_key(line: str) -> str:
    """Criterion portion of a flattened verifier line (everything before the last ``': '`` status suffix)."""
    s = line.rstrip("\n\r")
    if ": " not in s:
        return s
    return s.rsplit(": ", 1)[0]


def _verifier_line_status_tail(line: str) -> str:
    """Status token(s) after the last ``': '`` on a flattened verifier line."""
    s = line.rstrip("\n\r")
    if ": " not in s:
        return ""
    return s.rsplit(": ", 1)[1]


def _compact_verifier_lines_annotation(
    before_lines: List[str],
    after_lines: List[str],
    *,
    path_disp: str = "verifiers",
    max_out: int = 8000,
) -> str:
    """Diff flattened ``{criterion}: {status}`` lines by criterion key; targeted ``•`` bullets like file edits."""
    bl = [ln.rstrip("\n\r") for ln in before_lines]
    al = [ln.rstrip("\n\r") for ln in after_lines]
    if bl == al:
        return f"(no textual change) path={path_disp}"

    kb = {_verifier_line_criterion_key(l): _verifier_line_status_tail(l) for l in bl}
    ka = {_verifier_line_criterion_key(l): _verifier_line_status_tail(l) for l in al}
    keys_b, keys_a = set(kb), set(ka)
    parts: List[str] = []

    for k in sorted(keys_b & keys_a):
        stb, sta = kb[k], ka[k]
        if stb == sta:
            continue
        # Same criterion key: only the trailing ``status`` changed.
        parts.append(f"• {_trunc_anno(k, 520)}: {stb} → {sta}")

    for k in sorted(keys_b - keys_a):
        parts.append(f"• removed: {_trunc_anno(f'{k}: {kb[k]}', 220)}")
    for k in sorted(keys_a - keys_b):
        parts.append(f"• added: {_trunc_anno(f'{k}: {ka[k]}', 220)}")

    if not parts:
        return f"(no localized changes detected) path={path_disp}"

    body = "\n".join(parts)
    if len(body) > max_out:
        body = body[: max_out - 50].rstrip() + f"\n… (annotation truncated, path={path_disp})"
    return f"path={path_disp}\n{body}"


def _edit_workflow_tool_result_from_snapshots(
    wf_before: Optional[list],
    wf_after: Optional[list],
) -> str:
    """Compact workflow plan diff for ``edit_workflow`` / ``edit_plan`` (Progress region)."""

    def _norm_wf(wf: Optional[list]) -> list:
        if wf is None:
            return []
        return wf if isinstance(wf, list) else []

    before_native = _to_llm_native_tree(_norm_wf(wf_before))
    after_native = _to_llm_native_tree(_norm_wf(wf_after))
    before_json = json.dumps(before_native, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    after_json = json.dumps(after_native, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if before_json == after_json:
        return (
            "(no changes to workflow plan between prior carried-forward snapshot "
            "and this edit_workflow row)"
        )
    return _file_edit_diff_from_raw_strings(before_json, after_json, "workflow")


def _edit_verifier_tool_result_from_snapshots(
    wf_before: Optional[list],
    wf_after: Optional[list],
    *,
    max_chars: int = 12_000,
) -> str:
    """Compact verifier edit summary: flatten to ``criterion: status`` lines, then targeted ``•`` edits."""
    def _norm_wf(wf: Optional[list]) -> list:
        if wf is None:
            return []
        return wf if isinstance(wf, list) else []

    blines = _flatten_workflow_verifier_lines(_norm_wf(wf_before))
    alines = _flatten_workflow_verifier_lines(_norm_wf(wf_after))
    if blines == alines:
        return (
            "(no changes to verifier criteria/status between prior carried-forward snapshot "
            "and this edit_verifier row)"
        )
    cap = min(max_chars, 8000)
    return _compact_verifier_lines_annotation(blines, alines, path_disp="verifiers", max_out=cap)


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


def _snapshot_workflow_json(norm: Optional[dict]) -> str:
    """Stable JSON text for workflow tree diffing."""
    if not isinstance(norm, dict):
        return ""
    snap = norm.get("state_snapshot")
    if not isinstance(snap, dict):
        return ""
    wf = snap.get("workflow")
    if not isinstance(wf, list):
        return ""
    return json.dumps(wf, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


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


def _brain_edit_environment_diff_sections(
    prior: Optional[dict],
    current: dict,
    *,
    cwd: Optional[str] = None,
) -> List[str]:
    """Diff workflow tree JSON and output file contents between snapshots."""
    sections: List[str] = []
    wf_before = _snapshot_workflow_json(prior)
    wf_after = _snapshot_workflow_json(current)
    if wf_before != wf_after:
        sections.append(
            _file_edit_diff_from_raw_strings(wf_before, wf_after, "environment/workflow")
        )

    files_before = _snapshot_files_content_map(prior or {}, cwd)
    files_after = _snapshot_files_content_map(current, cwd)
    for path_key in sorted(set(files_before) | set(files_after)):
        before_s = files_before.get(path_key, "")
        after_s = files_after.get(path_key, "")
        if before_s == after_s:
            continue
        path_disp = f"environment/{_display_path_from_key(path_key)}"
        sections.append(_file_edit_diff_from_raw_strings(before_s, after_s, path_disp))
    return sections


def _brain_edit_tool_result_from_snapshots(
    prior: Optional[dict],
    current: dict,
    *,
    cwd: Optional[str] = None,
    max_chars: int = 12_000,
) -> str:
    """Compact brain-edit summary: memory, skill, and environment vs prior snapshot."""
    mem_before = _snapshot_brain_map(prior, "memory")
    mem_after = _snapshot_brain_map(current, "memory")
    sk_before = _snapshot_brain_map(prior, "skill")
    sk_after = _snapshot_brain_map(current, "skill")

    sections: List[str] = []
    sections.extend(_brain_edit_map_diff_sections("memory", mem_before, mem_after))
    sections.extend(_brain_edit_map_diff_sections("skill", sk_before, sk_after))
    sections.extend(_brain_edit_environment_diff_sections(prior, current, cwd=cwd))

    if not sections:
        return "(no textual change) path=brain"

    body = "\n\n".join(sections)
    if len(body) > max_chars:
        body = body[: max_chars - 50].rstrip() + "\n… (annotation truncated, path=brain)"
    return body


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
            return out
        if t == "tool_result":
            return {
                "type": "tool_result",
                "toolUseId": raw.get("toolUseId", ""),
                "toolName": raw.get("toolName", ""),
                "content": raw.get("content", ""),
                "isError": raw.get("isError", False),
            }
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
            "description": (
                "Read the contents of a file. Supports text files and images (jpg, png, gif, webp). "
                "Images are sent as attachments. For text files, output is truncated to 2000 lines or 50KB "
                "(whichever is hit first). Use offset/limit for large files. When you need the full file, "
                "continue with offset until complete."
            ),
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
            "description": (
                "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
                "Automatically creates parent directories."
            ),
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
            "description": (
                "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, "
                "non-overlapping region of the original file. If two changes affect the same block or nearby lines, "
                "merge them into one edit instead of emitting overlapping edits. Do not include large unchanged "
                "regions just to connect distant changes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
                    "edits": {
                        "type": "array",
                        "description": (
                            "One or more targeted replacements. Each edit is matched against the original file, "
                            "not incrementally. Do not include overlapping or nested edits. If two changes touch "
                            "the same block or nearby lines, merge them into one edit instead."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "oldText": {
                                    "type": "string",
                                    "description": (
                                        "Exact text for one targeted replacement. It must be unique in the "
                                        "original file and must not overlap with any other edits[].oldText in "
                                        "the same call."
                                    ),
                                },
                                "newText": {
                                    "type": "string",
                                    "description": "Replacement text for this targeted edit.",
                                },
                            },
                            "required": ["oldText", "newText"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["path", "edits"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a bash command in the current working directory. Returns stdout and stderr. "
                "Output is truncated to last 2000 lines or 50KB (whichever is hit first). If truncated, full "
                "output is saved to a temp file. Optionally provide a timeout in seconds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Bash command to execute"},
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (optional, no default timeout)",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search file contents for a pattern. Returns matching lines with file paths and line numbers. "
                "Respects .gitignore. Output is truncated to 100 matches or 50KB (whichever is hit first). "
                "Long lines are truncated to 500 chars."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Search pattern (regex or literal string)"},
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search (default: current directory)",
                    },
                    "glob": {
                        "type": "string",
                        "description": "Filter files by glob pattern, e.g. '*.ts' or '**/*.spec.ts'",
                    },
                    "ignoreCase": {"type": "boolean", "description": "Case-insensitive search (default: false)"},
                    "literal": {
                        "type": "boolean",
                        "description": "Treat pattern as literal string instead of regex (default: false)",
                    },
                    "context": {
                        "type": "number",
                        "description": "Number of lines to show before and after each match (default: 0)",
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of matches to return (default: 100)",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find",
            "description": (
                "Search for files by glob pattern. Returns matching file paths relative to the search directory. "
                "Respects .gitignore. Output is truncated to 1000 results or 50KB (whichever is hit first)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": (
                            "Glob pattern to match files, e.g. '*.ts', '**/*.json', or 'src/**/*.spec.ts'"
                        ),
                    },
                    "path": {"type": "string", "description": "Directory to search in (default: current directory)"},
                    "limit": {"type": "number", "description": "Maximum number of results (default: 1000)"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ls",
            "description": (
                "List directory contents. Returns entries sorted alphabetically, with '/' suffix for directories. "
                "Includes dotfiles. Output is truncated to 500 entries or 50KB (whichever is hit first)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory to list (default: current directory)"},
                    "limit": {"type": "number", "description": "Maximum number of entries to return (default: 500)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workflow_plan",
            "description": (
                "Register a hierarchical workflow plan. Provide 3-5 main steps at the top level with description, "
                "outputFiles, verifiers, and optional children."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "description": "Top-level workflow steps",
                        "items": {"$ref": "#/$defs/WorkflowNode"},
                    },
                },
                "required": ["tasks"],
                "$defs": {
                    "WorkflowNode": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "outputFiles": {"type": "array", "items": {"type": "string"}},
                            "verifiers": {"type": "array", "items": {"type": "string"}},
                            "children": {"type": "array", "items": {"$ref": "#/$defs/WorkflowNode"}},
                        },
                        "required": ["description", "outputFiles", "verifiers"],
                        "additionalProperties": False,
                    },
                },
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
    cursor: sqlite3.Cursor, session_id: str, *, db_root: Optional[Path] = None
) -> Optional[dict]:
    """Build the weight-based export for a single session."""
    try:
        row = cursor.execute(
            """SELECT id, title, workflow_tree, last_prompt, cwd, engine, steps, output_files, verification_criteria, verifier_marks, expertise_task, pi_session_file
               FROM sessions WHERE id = ?""",
            (session_id,),
        ).fetchone()
    except sqlite3.OperationalError:
        try:
            row = cursor.execute(
                """SELECT id, title, workflow_tree, last_prompt, cwd, engine, steps, output_files, verification_criteria, verifier_marks, expertise_task
                   FROM sessions WHERE id = ?""",
                (session_id,),
            ).fetchone()
            if row:
                row = (*row, None)
        except sqlite3.OperationalError:
            try:
                row = cursor.execute(
                    """SELECT id, title, workflow_tree, last_prompt, cwd, engine, steps, output_files, verification_criteria, verifier_marks
                       FROM sessions WHERE id = ?""",
                    (session_id,),
                ).fetchone()
                if row:
                    row = (*row, None, None)
            except sqlite3.OperationalError:
                row = cursor.execute(
                    """SELECT id, title, workflow_tree, last_prompt, cwd, engine
                       FROM sessions WHERE id = ?""",
                    (session_id,),
                ).fetchone()
                if row:
                    row = (*row, None, None, None, None, None, None)
    if not row:
        return None
    sid, title, workflow_tree_raw, last_prompt, cwd, db_engine, steps_raw, output_files_raw, verification_criteria_raw, verifier_marks_raw, expertise_task_raw, pi_session_file_raw = row
    workflow_tree = parse_json_column(workflow_tree_raw, [])
    steps = parse_json_column(steps_raw, [])
    output_files = parse_json_column(output_files_raw, [])
    verification_criteria = parse_json_column(verification_criteria_raw, [])
    verifier_marks = parse_json_column(verifier_marks_raw, [])
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

    engine = db_engine or "legacy-claude"
    action_trajectory: List[dict] = []
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

    initial_task_instruction = extract_initial_task_instruction(
        action_trajectory,
        last_prompt or "",
        pi_session_file=resolve_pi_session_file(
            pi_session_file_raw, all_msgs, db_root=db_root, session_id=session_id
        ),
        stored_initial_prompt=_session_initial_prompt(cursor, session_id),
    )
    action_trajectory = reclassify_auto_verifier_edits(action_trajectory)
    action_trajectory = inject_synthetic_human_edits(action_trajectory, export_cwd)
    full_traj = build_full_session_trajectory(
        action_trajectory,
        initial_task_instruction,
        export_cwd,
        workflow_tree,
        steps,
        output_files,
        verification_criteria,
        verifier_marks,
    )

    # Ensure each exported step has environment, then group consecutive steps
    # by actor into task_units.
    env_carry = empty_environment()
    full_traj_with_env: List[dict] = []
    for step in full_traj:
        step_out = copy.deepcopy(step)
        step_env = step_out.get("environment")
        if isinstance(step_env, dict):
            merged_env = copy.deepcopy(step_env)
            merged_env.pop("verifier", None)
            env_carry = merged_env
            step_out["environment"] = copy.deepcopy(merged_env)
        else:
            step_out["environment"] = copy.deepcopy(env_carry)
        full_traj_with_env.append(step_out)

    task_units: List[dict] = []
    current_actor: Optional[str] = None
    current_steps: List[dict] = []
    current_agent_prompt: str = ""
    pending_agent_prompt: str = WORKFLOW_PLAN_INSTRUCTION + initial_task_instruction
    first_plan_split_done = False
    skip_indices: Set[int] = set()

    def _flush_group() -> None:
        nonlocal current_actor, current_steps, current_agent_prompt
        if not current_steps or current_actor is None:
            return
        last_env = copy.deepcopy(current_steps[-1].get("environment")) or empty_environment()
        traj_out: List[dict] = []
        for s in current_steps:
            row = copy.deepcopy(s)
            row.pop("environment", None)
            row.pop("actor", None)
            traj_out.append(row)
        unit: Dict[str, Any] = {
            "actor": "agent" if current_actor == "agent" else "user",
            "trajectory": traj_out,
            "environment": last_env,
        }
        if current_actor == "agent":
            unit["prompt"] = current_agent_prompt
        task_units.append(unit)
        current_actor = None
        current_steps = []
        current_agent_prompt = ""

    for i, step in enumerate(full_traj_with_env):
        if i in skip_indices:
            continue
        actor = str(step.get("actor", "user"))
        if actor == "user":
            p = _parse_action_message_payload(str(step.get("action", "")))
            if isinstance(p, str) and not _prompts_equal(p, initial_task_instruction):
                pending_agent_prompt = p

        if current_actor is None:
            current_actor = actor
        if actor != current_actor:
            _flush_group()
            current_actor = actor
            current_steps = []
            if current_actor == "agent":
                current_agent_prompt = pending_agent_prompt
        elif current_actor == "agent" and not current_steps:
            current_agent_prompt = pending_agent_prompt

        current_steps.append(step)

        act = str(step.get("action", ""))
        if (
            current_actor == "agent"
            and (act.startswith("workflow_plan(") or act.startswith("plan("))
            and not first_plan_split_done
        ):
            first_plan_split_done = True
            if i + 1 < len(full_traj_with_env):
                nxt = full_traj_with_env[i + 1]
                if str(nxt.get("actor", "")) == "agent":
                    nxt_act = str(nxt.get("action", ""))
                    if nxt_act.startswith("message("):
                        ack = _parse_action_message_payload(nxt_act)
                        if isinstance(ack, str) and ack.strip():
                            current_steps[-1]["tool_result"] = ack
                            skip_indices.add(i + 1)
            _flush_group()
            current_actor = "agent"
            current_steps = []
            nxt = _first_backend_node_prompt(action_trajectory)
            pending_agent_prompt = nxt
            current_agent_prompt = nxt

    _flush_group()

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
        "task": initial_task_instruction,
        "model": model_name,
        "task_units": task_units,
    }
    if expertise_task_raw is not None:
        et = str(expertise_task_raw).strip()
        if et:
            result["expertise_task"] = et
    result["system_prompt"] = _pi_system_prompt(export_cwd) if is_pi else ""
    result["tool_schemas"] = PI_TOOL_SCHEMAS if is_pi else []
    return result


def _last_workflow_node_verifiers(workflow: Any) -> Tuple[Tuple[str, str], ...]:
    """Verifiers on the last top-level ``workflow`` node (criterion + status)."""
    if not isinstance(workflow, list) or not workflow:
        return ()
    last = workflow[-1]
    if not isinstance(last, dict):
        return ()
    vs = last.get("verifiers") or []
    return tuple(
        (str(v.get("criterion", "")), str(v.get("status", "")))
        for v in vs
        if isinstance(v, dict)
    )


def _last_workflow_node_description(workflow: Any) -> str:
    if not isinstance(workflow, list) or not workflow:
        return ""
    last = workflow[-1]
    if not isinstance(last, dict):
        return ""
    return str(last.get("description", "")).strip()


def _last_workflow_node_criteria(workflow: Any) -> Tuple[str, ...]:
    return tuple(c for c, _ in _last_workflow_node_verifiers(workflow))


def _collect_last_node_verifier_versions(session: dict) -> List[Tuple[Tuple[str, str], ...]]:
    """Distinct last-node verifier snapshots in task_unit order (first appearance per criteria set)."""
    seen: Set[Tuple[str, ...]] = set()
    versions: List[Tuple[Tuple[str, str], ...]] = []
    for unit in session.get("task_units") or []:
        if not isinstance(unit, dict):
            continue
        env = unit.get("environment")
        if not isinstance(env, dict):
            continue
        sig = _last_workflow_node_verifiers(env.get("workflow"))
        if not sig:
            continue
        crit = tuple(c for c, _ in sig)
        if crit in seen:
            continue
        seen.add(crit)
        versions.append(sig)
    return versions


def _verifier_status_glyph(status: str) -> str:
    s = status.strip().lower()
    if s == "success":
        return "✓"
    if s == "failure":
        return "✗"
    if s == "unchecked":
        return "○"
    return "·"


def log_exported_session_verifier_versions(sessions: List[dict]) -> None:
    """Pretty-print each exported session's uuid and last-workflow-node verifier versions."""
    if not sessions:
        logger.info("No exported sessions.")
        return

    bar = "═" * 72
    for sess in sessions:
        uuid = str(sess.get("uuid", "?"))
        name = str(sess.get("name", "")).strip()
        versions = _collect_last_node_verifier_versions(sess)

        step_desc = ""
        for unit in sess.get("task_units") or []:
            if not isinstance(unit, dict):
                continue
            env = unit.get("environment")
            if not isinstance(env, dict):
                continue
            desc = _last_workflow_node_description(env.get("workflow"))
            if desc:
                step_desc = desc
                break

        logger.info("")
        logger.info(bar)
        logger.info("Session %s", uuid)
        if name:
            logger.info("  %s", name)
        if step_desc:
            logger.info("  Last workflow step: %s", step_desc)
        logger.info("  Verifier versions: %d", len(versions))

        if not versions:
            logger.info("  (no verifiers on last workflow node)")
            continue

        for i, verifiers in enumerate(versions, start=1):
            logger.info("")
            logger.info("  ── Version %d (%d criteria) ──", i, len(verifiers))
            for criterion, status in verifiers:
                glyph = _verifier_status_glyph(status)
                logger.info("    %s [%s] %s", glyph, status, criterion)

    logger.info("")
    logger.info(bar)
    logger.info("Logged verifier versions for %d exported session(s)", len(sessions))


def _session_last_node_verifiers_unchanged_first_to_last(session: dict) -> bool:
    """True when workflow[-1] criteria at the first and last task_unit snapshots match."""
    first: Tuple[str, ...] = ()
    last: Tuple[str, ...] = ()
    for unit in session.get("task_units") or []:
        if not isinstance(unit, dict):
            continue
        env = unit.get("environment")
        if not isinstance(env, dict):
            continue
        crit = _last_workflow_node_criteria(env.get("workflow"))
        if not crit:
            continue
        if not first:
            first = crit
        last = crit
    if not first:
        return True
    return first == last


def filter_sessions_with_unchanged_workflow_verifiers(sessions: List[dict]) -> List[dict]:
    """Drop sessions whose last workflow node's verifiers are unchanged (first vs last snapshot)."""
    kept: List[dict] = []
    removed = 0
    for sess in sessions:
        if _session_last_node_verifiers_unchanged_first_to_last(sess):
            removed += 1
            logger.info(
                "Skipping session %s (%s): last workflow node verifiers unchanged "
                "(first and last snapshot match)",
                sess.get("uuid", "?"),
                sess.get("name", ""),
            )
        else:
            kept.append(sess)
    if removed:
        logger.info(
            "Filtered %d session(s) with unchanged last-node verifiers (%d kept)",
            removed,
            len(kept),
        )
    return kept


def _session_user_unit_count(session: dict) -> int:
    return sum(
        1
        for unit in session.get("task_units") or []
        if isinstance(unit, dict) and unit.get("actor") == "user"
    )


def filter_sessions_without_follow_up_user_actions(sessions: List[dict]) -> List[dict]:
    """Drop sessions with only the initial user message (fewer than 2 user task_units)."""
    kept: List[dict] = []
    removed = 0
    for sess in sessions:
        n_user = _session_user_unit_count(sess)
        if n_user < 2:
            removed += 1
            logger.info(
                "Skipping session %s (%s): only %d user action(s) (need at least 2)",
                sess.get("uuid", "?"),
                sess.get("name", ""),
                n_user,
            )
        else:
            kept.append(sess)
    if removed:
        logger.info(
            "Filtered %d session(s) without follow-up user actions (%d kept)",
            removed,
            len(kept),
        )
    return kept


TASK_CATEGORIES = ("abstract-writing", "data-viz-html")


def filter_sessions_by_task_category(sessions: List[dict], category: str) -> List[dict]:
    """Keep only sessions whose ``expertise_task`` matches ``category``."""
    kept: List[dict] = []
    removed = 0
    for sess in sessions:
        et = sess.get("expertise_task")
        if isinstance(et, str) and et.strip() == category:
            kept.append(sess)
        else:
            removed += 1
            logger.info(
                "Skipping session %s (%s): expertise_task=%r (want %r)",
                sess.get("uuid", "?"),
                sess.get("name", ""),
                et,
                category,
            )
    if removed:
        logger.info(
            "Filtered %d session(s) outside task category %r (%d kept)",
            removed,
            category,
            len(kept),
        )
    return kept


def extract_all_sessions_weight_based(
    cursor: sqlite3.Cursor, *, db_root: Optional[Path] = None
) -> List[dict]:
    rows = cursor.execute("SELECT id FROM sessions ORDER BY updated_at DESC").fetchall()
    sessions = []
    for (session_id,) in rows:
        sess = build_weight_based_session(cursor, session_id, db_root=db_root)
        if sess:
            sessions.append(sess)
    return sessions


def extract_all_sessions(
    cursor: sqlite3.Cursor, *, db_root: Optional[Path] = None
) -> List[dict]:
    rows = cursor.execute("SELECT id FROM sessions ORDER BY updated_at DESC").fetchall()
    out = []
    for (session_id,) in rows:
        sess = extract_session(cursor, session_id, db_root=db_root)
        if sess:
            out.append(sess)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Agent Cowork task sessions to JSON")
    parser.add_argument("--db", type=Path, help="Path to sessions.db (default: Electron userData location)")
    parser.add_argument("--output", "-o", type=Path, help="Output single JSON file (default: stdout)")
    parser.add_argument("--session-id", type=str, help="Export only this session ID")
    parser.add_argument(
        "--task-category",
        choices=TASK_CATEGORIES,
        help="Only export sessions with this expertise_task value (default: all categories)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

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
    db_root = db_path.parent

    try:
        if args.session_id:
            sess = build_weight_based_session(cursor, args.session_id, db_root=db_root)
            if not sess:
                print(f"Error: session not found: {args.session_id}", file=sys.stderr)
                return 1
            payload = [sess]
        else:
            payload = extract_all_sessions_weight_based(cursor, db_root=db_root)

        payload = filter_sessions_with_unchanged_workflow_verifiers(payload)
        payload = filter_sessions_without_follow_up_user_actions(payload)
        if args.task_category:
            payload = filter_sessions_by_task_category(payload, args.task_category)

        log_exported_session_verifier_versions(payload)

        json_str = json.dumps(payload, indent=2, ensure_ascii=False)
        if args.output:
            args.output.write_text(json_str + "\n", encoding="utf-8")
            n_units = sum(len(s.get("task_units", [])) for s in payload)
            print(f"Wrote {args.output} ({len(payload)} sessions, {n_units} task_units)", file=sys.stderr)
        else:
            print(json_str)
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
