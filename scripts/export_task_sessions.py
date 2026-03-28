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

Per-step ``environment`` (workflow nodes, each node’s ``verifiers`` with ``status``, and output files)
comes from ``messages.state_snapshot`` when recorded: human ``user_prompt``; SDK ``user`` tool
results; turn ``result``; and a synthetic ``verifier_label`` row after the verifier LM updates marks
(exported as agent ``verify({"nodeId":...})``). Pure ``message("…")`` steps (user or agent) omit ``environment``.
Older DBs without ``state_snapshot`` fall back to one end-of-session snapshot.

The second is always the agent ``plan({initial query})``; its ``environment`` has the workflow tree
(with every step ``status`` ``pending``, not DB completion state); each verifier is
``{"criterion", "status": "failure"}``; ``file`` maps paths to ``null``.

Database location (Electron userData):
- macOS: ~/Library/Application Support/Agent Cowork/sessions.db
- Windows: %APPDATA%\\Agent Cowork\\sessions.db
- Linux: ~/.config/Agent Cowork/sessions.db

Usage:
  conda activate code   # optional: use "code" env
  python export_task_sessions.py [--db PATH] [--output FILE] [--session-id ID] \\
    [--tasks-dir DIR [--task-unit-id NODE_UUID]] [--granularity {all,automation,control}]
  # Per-task files: --tasks-dir requires --session-id. Each task unit is written as tasks/{unit-id}.json
  # (unit id is the workflow node id, a UUID). With --task-unit-id, only that file is updated.
  # Use AGENT_COWORK_DB to override DB path:
  AGENT_COWORK_DB=/path/to/sessions.db python export_task_sessions.py
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

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
        if base is not None and rel and not str(rel).startswith(("/", "\\")):
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
    return {"workflow": [], "file": []}


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
    return {"workflow": wf, "file": files}


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


def environment_for_norm(norm: dict, default_env: dict) -> Tuple[dict, bool]:
    """Return (environment dict, True if taken from persisted ``state_snapshot`` on this message)."""
    snap = norm.get("state_snapshot")
    if isinstance(snap, dict) and "workflow" in snap and "file" in snap:
        return snap, True
    return default_env, False


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
    final_env = build_environment_state(
        cwd_val,
        workflow_tree,
        steps,
        output_files,
        verification_criteria,
        verifier_marks,
        include_files=True,
    )
    plan_env = build_environment_state(
        cwd_val,
        workflow_tree,
        steps,
        output_files,
        verification_criteria,
        verifier_marks,
        include_files=False,
        file_placeholder=True,
        plan_snapshot=True,
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
            step_env, _snap = environment_for_norm(m, final_env)
            nid_raw = m.get("nodeId", "")
            nid = str(nid_raw) if nid_raw is not None else ""
            act = f"verify({json.dumps({'nodeId': nid}, ensure_ascii=False)})"
            traj.append(trajectory_row("agent", act, step_env))
            idx += 1
            continue

        if m.get("role") == "user":
            u_env, _u_snap = environment_for_norm(m, final_env)
            traj.append(trajectory_row("user", describe_human_action(m), u_env))
        elif m.get("role") == "agent":
            action, extra = agent_export_action(msgs, idx)
            consume = 1 + extra
            merged_tool: Optional[str] = None
            if _assistant_message_has_tool_use(m):
                tail_i = idx + consume
                if tail_i < len(msgs) and is_tool_result_message(msgs[tail_i]):
                    tr_raw = msgs[tail_i].get("raw")
                    if isinstance(tr_raw, dict):
                        merged_tool = _tool_result_blob(tr_raw)
                    consume += 1
            env_idx = idx + 1 if (extra == 1 or merged_tool is not None) else idx
            step_env, _env_snap = environment_for_norm(msgs[env_idx], final_env)
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
            traj.append(
                trajectory_row("user", json.dumps(m, ensure_ascii=False, default=str)[:400], final_env)
            )
        idx += 1

    return traj


def build_slice_trajectory(
    msgs: List[dict],
    cwd_val: Optional[str],
    workflow_tree: Any,
    steps: list,
    output_files: list,
    verification_criteria: list,
    verifier_marks: list,
) -> List[dict]:
    """Messages for one workflow node run, in order, with final session environment."""
    final_env = build_environment_state(
        cwd_val,
        workflow_tree,
        steps,
        output_files,
        verification_criteria,
        verifier_marks,
        include_files=True,
    )
    traj: List[dict] = []
    idx = 0
    while idx < len(msgs):
        m = msgs[idx]
        if m.get("type") == "verifier_label":
            step_env, _snap = environment_for_norm(m, final_env)
            nid_raw = m.get("nodeId", "")
            nid = str(nid_raw) if nid_raw is not None else ""
            act = f"verify({json.dumps({'nodeId': nid}, ensure_ascii=False)})"
            traj.append(trajectory_row("agent", act, step_env))
            idx += 1
            continue

        if m.get("role") == "user":
            if m.get("type") == "user_prompt" and is_backend_node_user_prompt(m.get("prompt")):
                idx += 1
                continue
            u_env, _u_snap = environment_for_norm(m, final_env)
            traj.append(trajectory_row("user", describe_human_action(m), u_env))
        elif m.get("role") == "agent":
            action, extra = agent_export_action(msgs, idx)
            consume = 1 + extra
            merged_tool: Optional[str] = None
            if _assistant_message_has_tool_use(m):
                tail_i = idx + consume
                if tail_i < len(msgs) and is_tool_result_message(msgs[tail_i]):
                    tr_raw = msgs[tail_i].get("raw")
                    if isinstance(tr_raw, dict):
                        merged_tool = _tool_result_blob(tr_raw)
                    consume += 1
            env_idx = idx + 1 if (extra == 1 or merged_tool is not None) else idx
            step_env, _env_snap = environment_for_norm(msgs[env_idx], final_env)
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
            traj.append(
                trajectory_row("user", json.dumps(m, ensure_ascii=False, default=str)[:400], final_env)
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


Granularity = Literal["all", "automation", "control"]


def flatten_workflow_tree(tree: Any, granularity: Granularity) -> List[dict]:
    """
    Flatten stored workflow_tree (WorkflowNode[]) into a stable list.

    Expected node shape (from UI):
      { id, description, outputFiles, verifiers, verifierMarks, children, ... }
    """
    if not isinstance(tree, list):
        return []

    out: List[dict] = []

    def include_node(node: dict) -> bool:
        if granularity == "all":
            return True
        depth = node.get("depth")
        if not isinstance(depth, int):
            return False
        if granularity == "automation":
            return depth == 0
        # control
        return depth > 0

    def walk(nodes: Any) -> None:
        if not isinstance(nodes, list):
            return
        for n in nodes:
            if not isinstance(n, dict):
                continue
            if include_node(n):
                out.append(n)
            walk(n.get("children"))

    walk(tree)
    return out


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


def normalize_message(msg: dict) -> dict:
    """Normalize a stored StreamMessage for JSON output (agent turn vs user message)."""
    if msg.get("type") == "user_prompt":
        return {"role": "user", "type": "user_prompt", "prompt": msg.get("prompt", "")}
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


def extract_session(
    cursor: sqlite3.Cursor, session_id: str, granularity: Granularity
) -> Optional[Tuple[dict, List[dict]]]:
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
    node_id_to_segment = segment_trajectory_by_persisted_node_prompts(action_trajectory, workflow_tree)
    if not node_id_to_segment:
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

    public: dict = {
        "uuid": sid,
        "name": title or "",
        "trajectory": full_traj,
    }

    unit_payloads: List[dict] = []
    tree_nodes = flatten_workflow_tree(workflow_tree, granularity)
    if tree_nodes:
        for node in tree_nodes:
            node_id = node.get("id") if isinstance(node.get("id"), str) else None
            if not node_id:
                continue
            segment = node_id_to_segment.get(node_id, [])
            intent = str(node.get("description") or "")
            ut = build_slice_trajectory(
                segment,
                cwd_val,
                workflow_tree,
                steps,
                output_files,
                verification_criteria,
                verifier_marks,
            )
            display_name = f"{title} — {intent}" if title else intent
            unit_payloads.append({"uuid": node_id, "name": display_name, "trajectory": ut})
    else:
        # Legacy fallback: flat steps grid columns
        if granularity in ("all", "automation"):
            workflow_steps = build_workflow_steps(steps, output_files, verification_criteria, verifier_marks)
            for i, step in enumerate(workflow_steps):
                step_idx = step.get("step_index", i)
                unit_id = f"{sid}-step-{step_idx}"
                intent = step.get("step_description", "")
                display_name = f"{title} — {intent}" if title else str(intent)
                unit_payloads.append({"uuid": unit_id, "name": display_name, "trajectory": []})

    return public, unit_payloads


def write_session_to_tasks_dir(
    unit_payloads: List[dict],
    tasks_dir: Path,
    *,
    only_unit_id: Optional[str] = None,
    pretty: bool = False,
) -> int:
    """Write each unit payload to ``tasks_dir / f\"{uuid}.json\"``."""
    tasks_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for payload in unit_payloads:
        uid = payload.get("uuid")
        if not isinstance(uid, str) or not uid:
            continue
        if only_unit_id is not None and uid != only_unit_id:
            continue
        path = tasks_dir / f"{uid}.json"
        body = json.dumps(payload, indent=2 if pretty else None, ensure_ascii=False)
        if not body.endswith("\n"):
            body += "\n"
        path.write_text(body, encoding="utf-8")
        written += 1
    return written


def extract_all_sessions(cursor: sqlite3.Cursor, granularity: Granularity) -> List[dict]:
    rows = cursor.execute("SELECT id FROM sessions ORDER BY updated_at DESC").fetchall()
    out = []
    for (session_id,) in rows:
        sess = extract_session(cursor, session_id, granularity)
        if sess:
            out.append(sess[0])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Agent Cowork task sessions to JSON")
    parser.add_argument("--db", type=Path, help="Path to sessions.db (default: Electron userData location)")
    parser.add_argument("--output", "-o", type=Path, help="Output single JSON file (default: stdout). Ignored if --tasks-dir is set.")
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        help="Write one file per task unit: {unit-id}.json (requires --session-id). Use --task-unit-id to update only one file.",
    )
    parser.add_argument(
        "--task-unit-id",
        type=str,
        help="With --tasks-dir, only write/update this task unit's JSON (workflow node id / unit id).",
    )
    parser.add_argument("--session-id", type=str, help="Export only this session ID")
    parser.add_argument(
        "--granularity",
        type=str,
        choices=["all", "automation", "control"],
        default="automation",
        help='Which workflow level to export from workflow_tree: "automation" (depth=0), "control" (depth>0), or "all".',
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
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
        if args.tasks_dir:
            if not args.session_id:
                print("Error: --tasks-dir requires --session-id", file=sys.stderr)
                return 1
            packed = extract_session(cursor, args.session_id, args.granularity)
            if not packed:
                print(f"Error: session not found: {args.session_id}", file=sys.stderr)
                return 1
            _public, unit_payloads = packed
            n = write_session_to_tasks_dir(
                unit_payloads,
                args.tasks_dir,
                only_unit_id=args.task_unit_id,
                pretty=args.pretty,
            )
            if args.task_unit_id and n == 0:
                print(
                    f"Error: no task unit with id {args.task_unit_id!r} in this session export",
                    file=sys.stderr,
                )
                return 1
            print(f"Wrote {n} task file(s) under {args.tasks_dir}", file=sys.stderr)
        elif args.session_id:
            packed = extract_session(cursor, args.session_id, args.granularity)
            if not packed:
                print(f"Error: session not found: {args.session_id}", file=sys.stderr)
                return 1
            payload, _units = packed
            json_str = json.dumps(payload, indent=2 if args.pretty else None, ensure_ascii=False)
            if args.output:
                args.output.write_text(json_str, encoding="utf-8")
                print(f"Wrote {args.output}", file=sys.stderr)
            else:
                print(json_str)
        else:
            payload = {"sessions": extract_all_sessions(cursor, args.granularity)}
            json_str = json.dumps(payload, indent=2 if args.pretty else None, ensure_ascii=False)
            if args.output:
                args.output.write_text(json_str, encoding="utf-8")
                print(f"Wrote {args.output}", file=sys.stderr)
            else:
                print(json_str)
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
