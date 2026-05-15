#!/usr/bin/env python3
"""Re-export every Agent Cowork session in place using the weight format."""

from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
from pathlib import Path


def default_user_data_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "agent-cowork"
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        return base / "agent-cowork"
    return Path.home() / ".config" / "agent-cowork"


def session_ids(db_path: Path) -> list[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("select id from sessions order by updated_at desc").fetchall()
    finally:
        conn.close()
    return [str(row[0]) for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-export all stored Agent Cowork sessions as weight-format JSON."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=default_user_data_dir() / "sessions.db",
        help="Path to sessions.db.",
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=None,
        help="Directory to overwrite JSON exports in. Defaults to <userData>/tasks.",
    )
    parser.add_argument(
        "--exporter",
        type=Path,
        default=Path(__file__).resolve().parent / "scripts" / "export_task_sessions.py",
        help="Path to export_task_sessions.py.",
    )
    args = parser.parse_args()

    db_path = args.db.expanduser().resolve()
    exporter = args.exporter.expanduser().resolve()
    tasks_dir = (
        args.tasks_dir.expanduser().resolve()
        if args.tasks_dir is not None
        else db_path.parent / "tasks"
    )

    if not db_path.exists():
        print(f"sessions.db not found: {db_path}", file=sys.stderr)
        return 1
    if not exporter.exists():
        print(f"exporter not found: {exporter}", file=sys.stderr)
        return 1

    tasks_dir.mkdir(parents=True, exist_ok=True)
    ids = session_ids(db_path)
    print(f"Re-exporting {len(ids)} session(s) to {tasks_dir}", file=sys.stderr)

    for idx, session_id in enumerate(ids, start=1):
        output_path = tasks_dir / f"{session_id}-workflow-full.json"
        cmd = [
            sys.executable,
            str(exporter),
            "--db",
            str(db_path),
            "--session-id",
            session_id,
            "--output",
            str(output_path),
            "--format",
            "weight",
        ]
        subprocess.run(cmd, check=True)
        print(f"[{idx}/{len(ids)}] wrote {output_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
