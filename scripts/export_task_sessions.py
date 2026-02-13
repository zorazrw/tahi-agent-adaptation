#!/usr/bin/env python3
"""
Export task sessions from the Agent Cowork SQLite database to JSON.

Each task session contains:
- initial_task_instruction: the user's first prompt (natural language)
- workflow_steps: list of steps, each with:
  - step_description
  - expected_output_files
  - verifiers: list of { criterion, status } (success/failure/unchecked)
  - (trajectory is at session level; see action_trajectory)
- action_trajectory: chronological list of agent/user message turns

Database location (Electron userData):
- macOS: ~/Library/Application Support/Agent Cowork/sessions.db
- Windows: %APPDATA%\\Agent Cowork\\sessions.db
- Linux: ~/.config/Agent Cowork/sessions.db

Usage:
  conda activate code   # optional: use "code" env
  python export_task_sessions.py [--db PATH] [--output FILE] [--session-id ID]
  # Use AGENT_COWORK_DB to override DB path:
  AGENT_COWORK_DB=/path/to/sessions.db python export_task_sessions.py
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional


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


def extract_session(cursor: sqlite3.Cursor, session_id: str) -> Optional[dict]:
    row = cursor.execute(
        """SELECT id, title, last_prompt, steps, output_files, verification_criteria, verifier_marks,
                  completed_step_indices, status, cwd, created_at, updated_at
           FROM sessions WHERE id = ?""",
        (session_id,),
    ).fetchone()
    if not row:
        return None
    (sid, title, last_prompt, steps_raw, output_files_raw, verification_criteria_raw, verifier_marks_raw,
     completed_indices_raw, status, cwd, created_at, updated_at) = row
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

    workflow_steps = build_workflow_steps(steps, output_files, verification_criteria, verifier_marks)

    return {
        "session_id": sid,
        "title": title or "",
        "initial_task_instruction": last_prompt or "",
        "workflow_steps": workflow_steps,
        "completed_step_indices": completed_step_indices,
        "action_trajectory": action_trajectory,
        "meta": {
            "status": status,
            "cwd": cwd,
            "created_at": created_at,
            "updated_at": updated_at,
        },
    }


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
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file (default: stdout)")
    parser.add_argument("--session-id", type=str, help="Export only this session ID")
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
            data = extract_session(cursor, args.session_id)
            if not data:
                print(f"Error: session not found: {args.session_id}", file=sys.stderr)
                return 1
            payload = data
        else:
            payload = {"sessions": extract_all_sessions(cursor)}
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
