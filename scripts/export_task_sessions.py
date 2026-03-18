#!/usr/bin/env python3
"""
Export task sessions from the Agent Cowork SQLite database to JSON.

Each task session contains:
- session_id: unique identifier
- title: session title
- initial_task_instruction: the user's first prompt (natural language)
- task_units: list of units, each with:
  - intent
  - agent_trajectory
  - verifiers: list of { criterion, status } where status is bool
  - human_trajectory
  - expected_output_files

Database location (Electron userData):
- macOS: ~/Library/Application Support/Agent Cowork/sessions.db
- Windows: %APPDATA%\\Agent Cowork\\sessions.db
- Linux: ~/.config/Agent Cowork/sessions.db

Usage:
  conda activate code   # optional: use "code" env
  python export_task_sessions.py [--db PATH] [--output FILE] [--session-id ID] [--granularity {all,automation,control}]
  # Use AGENT_COWORK_DB to override DB path:
  AGENT_COWORK_DB=/path/to/sessions.db python export_task_sessions.py
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, List, Literal, Optional


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


def verifier_status_bool(mark: Optional[str]) -> bool:
    """Map DB verifier_marks to bool (only 'check' is True)."""
    return mark == "check"


def split_trajectory(action_trajectory: List[dict]) -> tuple[List[dict], List[dict]]:
    """Split normalized trajectory into (human_trajectory, agent_trajectory)."""
    human = [m for m in action_trajectory if m.get("role") == "user"]
    agent = [m for m in action_trajectory if m.get("role") == "agent"]
    return human, agent


def extract_initial_task_instruction(action_trajectory: List[dict], fallback: str) -> str:
    for m in action_trajectory:
        if m.get("role") == "user" and m.get("type") == "user_prompt":
            return m.get("prompt", "") or fallback or ""
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
    # SDK message: could be assistant content, tool_use, tool_result, etc.
    return {"role": "agent", "raw": msg}


def filter_out_stream_events(trajectory: List[dict]) -> List[dict]:
    """Drop stream_event messages from the trajectory (keep user prompts and other agent messages)."""
    return [
        msg for msg in trajectory
        if not (msg.get("role") == "agent" and (msg.get("raw") or {}).get("type") == "stream_event")
    ]


def extract_session(cursor: sqlite3.Cursor, session_id: str, granularity: Granularity) -> Optional[dict]:
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
    workflow_tree = parse_json_column(workflow_tree_raw, [])
    steps = parse_json_column(steps_raw, [])
    output_files = parse_json_column(output_files_raw, [])
    verification_criteria = parse_json_column(verification_criteria_raw, [])
    verifier_marks = parse_json_column(verifier_marks_raw, [])
    completed_step_indices = parse_json_column(completed_indices_raw, [])

    messages_rows = cursor.execute(
        "SELECT data, created_at FROM messages WHERE session_id = ? ORDER BY created_at ASC",
        (session_id,),
    ).fetchall()
    action_trajectory = []
    for data_str, _ in messages_rows:
        try:
            msg = json.loads(data_str)
            action_trajectory.append(normalize_message(msg))
        except json.JSONDecodeError:
            action_trajectory.append({"role": "unknown", "raw": data_str[:200]})
    action_trajectory = filter_out_stream_events(action_trajectory)

    initial_task_instruction = extract_initial_task_instruction(action_trajectory, last_prompt or "")
    node_id_to_segment = segment_trajectory_by_persisted_node_prompts(action_trajectory, workflow_tree)
    if not node_id_to_segment:
        node_id_to_segment = segment_trajectory_by_resume_points(action_trajectory, workflow_tree)

    task_units = []
    tree_nodes = flatten_workflow_tree(workflow_tree, granularity)
    if tree_nodes:
        for node in tree_nodes:
            node_id = node.get("id") if isinstance(node.get("id"), str) else None
            segment = node_id_to_segment.get(node_id, []) if node_id else []
            human_trajectory, agent_trajectory = split_trajectory(segment)

            intent = str(node.get("description") or "")
            expected_output_files = node.get("outputFiles")
            if not isinstance(expected_output_files, list):
                expected_output_files = []
            expected_output_files = [str(p) for p in expected_output_files]

            criteria = node.get("verifiers")
            if not isinstance(criteria, list):
                criteria = []
            marks = node.get("verifierMarks")
            if not isinstance(marks, list):
                marks = []
            verifiers_bool = []
            for i, c in enumerate(criteria):
                mark = marks[i] if i < len(marks) else None
                verifiers_bool.append({"criterion": str(c), "status": verifier_status_bool(mark)})

            task_units.append(
                {
                    "intent": intent,
                    "agent_trajectory": agent_trajectory,
                    "verifiers": verifiers_bool,
                    "human_trajectory": human_trajectory,
                    "expected_output_files": expected_output_files,
                }
            )
    else:
        # Legacy fallback: flat steps grid columns
        # Legacy sessions effectively only have a single "automation" level (flat steps).
        if granularity in ("all", "automation"):
            workflow_steps = build_workflow_steps(steps, output_files, verification_criteria, verifier_marks)
            for step in workflow_steps:
                # Legacy: we don't have per-step boundaries in messages, so keep empty per-step trajectories.
                human_trajectory, agent_trajectory = [], []
                verifiers_bool = [{"criterion": v.get("criterion", ""), "status": v.get("status") == "success"} for v in step.get("verifiers", [])]
                task_units.append(
                    {
                        "intent": step.get("step_description", ""),
                        "agent_trajectory": agent_trajectory,
                        "verifiers": verifiers_bool,
                        "human_trajectory": human_trajectory,
                        "expected_output_files": step.get("expected_output_files", []),
                    }
                )

    return {
        "session_id": sid,
        "title": title or "",
        "initial_task_instruction": initial_task_instruction,
        "task_units": task_units,
    }


def extract_all_sessions(cursor: sqlite3.Cursor, granularity: Granularity) -> List[dict]:
    rows = cursor.execute("SELECT id FROM sessions ORDER BY updated_at DESC").fetchall()
    out = []
    for (session_id,) in rows:
        sess = extract_session(cursor, session_id, granularity)
        if sess:
            out.append(sess)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Agent Cowork task sessions to JSON")
    parser.add_argument("--db", type=Path, help="Path to sessions.db (default: Electron userData location)")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file (default: stdout)")
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
        if args.session_id:
            data = extract_session(cursor, args.session_id, args.granularity)
            if not data:
                print(f"Error: session not found: {args.session_id}", file=sys.stderr)
                return 1
            payload = data
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
