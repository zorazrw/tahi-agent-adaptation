#!/usr/bin/env python3
"""
Aggregate prediction-tab telemetry from the Agent Cowork SQLite database.

The renderer logs one row per prediction lifecycle event to the
``prediction_events`` table:

  - shown      : a prediction surfaced in the prompt-input tab
  - accepted   : user clicked Accept (or autofill mode consumed it)
  - dismissed  : user clicked Dismiss / pressed Escape
  - ignored    : prediction was cleared without an explicit resolve
                 (session change, new run, prediction-mode toggled off, etc.)

Each ``prediction_id`` is unique per *shown* prediction; subsequent events for
the same prediction share that id, so accept / dismiss / ignore are
mutually exclusive per prediction (the renderer guards with a ``resolved``
flag).

Usage:
  python scripts/prediction_stats.py [--db PATH] [--since-days N]
                                     [--session-id ID] [--json]

  AGENT_COWORK_DB=/path/to/sessions.db python scripts/prediction_stats.py

Default DB location (Electron userData):
  macOS:   ~/Library/Application Support/agent-cowork/sessions.db
  Windows: %APPDATA%\\agent-cowork\\sessions.db
  Linux:   ~/.config/agent-cowork/sessions.db
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional


def get_default_db_path() -> Optional[Path]:
    """Resolve the default sessions.db path for this platform."""
    if os.environ.get("AGENT_COWORK_DB"):
        p = Path(os.environ["AGENT_COWORK_DB"])
        return p if p.exists() else None
    home = Path.home()
    if sys.platform == "darwin":
        candidates = [
            home / "Library" / "Application Support" / "agent-cowork" / "sessions.db",
            home / "Library" / "Application Support" / "Agent Cowork" / "sessions.db",
        ]
    elif sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else (home / "AppData" / "Roaming")
        candidates = [
            base / "agent-cowork" / "sessions.db",
            base / "Agent Cowork" / "sessions.db",
        ]
    else:
        candidates = [
            home / ".config" / "agent-cowork" / "sessions.db",
            home / ".config" / "Agent Cowork" / "sessions.db",
        ]
    for p in candidates:
        if p.exists():
            return p
    return None


def fetch_events(
    db_path: Path,
    since_ms: Optional[int],
    session_id: Optional[str],
) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        # Confirm the table exists (older builds won't have it).
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='prediction_events'"
        ).fetchone()
        if not row:
            print(
                "prediction_events table not found. Open the app once to run the migration.",
                file=sys.stderr,
            )
            sys.exit(2)

        clauses: list[str] = []
        params: list = []
        if since_ms is not None:
            clauses.append("created_at >= ?")
            params.append(since_ms)
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            "SELECT id, session_id, prediction_id, event, action_type, confidence, "
            "       draft_text, rationale, metadata, created_at "
            f"FROM prediction_events {where} ORDER BY created_at ASC"
        )
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def _pct(n: int, d: int) -> str:
    if d <= 0:
        return "  —  "
    return f"{(100.0 * n / d):5.1f}%"


def summarize(events: list[dict]) -> dict:
    """Roll events up by prediction_id, then aggregate."""
    by_pred: dict[str, dict] = {}
    for e in events:
        pid = e["prediction_id"]
        bucket = by_pred.setdefault(
            pid,
            {
                "action_type": e["action_type"],
                "confidence": e.get("confidence"),
                "session_id": e["session_id"],
                "shown_at": None,
                "resolved_at": None,
                "outcome": None,  # accepted | dismissed | ignored | unresolved
                "auto": False,
            },
        )
        if e["event"] == "shown" and bucket["shown_at"] is None:
            bucket["shown_at"] = e["created_at"]
            bucket["action_type"] = e["action_type"]
            bucket["confidence"] = e.get("confidence")
        elif e["event"] in ("accepted", "dismissed", "ignored") and bucket["outcome"] is None:
            bucket["outcome"] = e["event"]
            bucket["resolved_at"] = e["created_at"]
            meta = e.get("metadata")
            if meta:
                try:
                    parsed = json.loads(meta)
                    if isinstance(parsed, dict) and parsed.get("auto"):
                        bucket["auto"] = True
                except (TypeError, ValueError):
                    pass

    totals = {"shown": 0, "accepted": 0, "dismissed": 0, "ignored": 0, "unresolved": 0, "auto_accepted": 0}
    by_action: dict[str, dict[str, int]] = {}
    latencies_ms: list[int] = []

    for bucket in by_pred.values():
        if bucket["shown_at"] is None:
            # Event with no preceding 'shown' (stale prediction_id) — skip.
            continue
        totals["shown"] += 1
        action = bucket["action_type"] or "unknown"
        slot = by_action.setdefault(
            action,
            {"shown": 0, "accepted": 0, "dismissed": 0, "ignored": 0, "unresolved": 0, "auto_accepted": 0},
        )
        slot["shown"] += 1
        outcome = bucket["outcome"] or "unresolved"
        totals[outcome] += 1
        slot[outcome] += 1
        if outcome == "accepted" and bucket["auto"]:
            totals["auto_accepted"] += 1
            slot["auto_accepted"] += 1
        if outcome in ("accepted", "dismissed") and bucket["resolved_at"] is not None:
            latencies_ms.append(bucket["resolved_at"] - bucket["shown_at"])

    return {
        "totals": totals,
        "by_action": by_action,
        "latencies_ms": latencies_ms,
        "raw_event_count": len(events),
        "prediction_count": len(by_pred),
    }


def format_text(summary: dict, db_path: Path, since_ms: Optional[int]) -> str:
    t = summary["totals"]
    lines: list[str] = []
    lines.append(f"DB: {db_path}")
    if since_ms is not None:
        lines.append(f"Window: since epoch_ms={since_ms}")
    lines.append(f"Raw events: {summary['raw_event_count']}    Predictions: {summary['prediction_count']}")
    lines.append("")
    lines.append("Overall")
    lines.append(f"  shown           {t['shown']:>5}")
    lines.append(f"  accepted        {t['accepted']:>5}   {_pct(t['accepted'], t['shown'])}   (auto: {t['auto_accepted']})")
    lines.append(f"  dismissed       {t['dismissed']:>5}   {_pct(t['dismissed'], t['shown'])}")
    lines.append(f"  ignored         {t['ignored']:>5}   {_pct(t['ignored'], t['shown'])}")
    lines.append(f"  unresolved      {t['unresolved']:>5}   {_pct(t['unresolved'], t['shown'])}")

    if summary["latencies_ms"]:
        ls = sorted(summary["latencies_ms"])
        median = ls[len(ls) // 2]
        p90 = ls[max(0, int(len(ls) * 0.9) - 1)]
        lines.append(f"  decision latency  median={median}ms  p90={p90}ms  (n={len(ls)})")

    by_action = summary["by_action"]
    if by_action:
        lines.append("")
        lines.append("By predicted action_type")
        header = f"  {'action_type':<14}  {'shown':>5}  {'accept':>6}  {'dismiss':>7}  {'ignore':>6}  {'unrslv':>6}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for action in sorted(by_action.keys()):
            s = by_action[action]
            lines.append(
                f"  {action:<14}  {s['shown']:>5}  "
                f"{s['accepted']:>3}/{_pct(s['accepted'], s['shown']).strip():<5} "
                f"{s['dismissed']:>3}/{_pct(s['dismissed'], s['shown']).strip():<5} "
                f"{s['ignored']:>3}/{_pct(s['ignored'], s['shown']).strip():<5} "
                f"{s['unresolved']:>3}"
            )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize user-prediction accept/decline/ignore telemetry."
    )
    parser.add_argument("--db", type=Path, help="Path to sessions.db (default: Electron userData location)")
    parser.add_argument(
        "--since-days",
        type=float,
        default=None,
        help="Only include events from the last N days.",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Restrict to a single session id.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args()

    db_path = args.db or get_default_db_path()
    if not db_path or not Path(db_path).exists():
        print("Error: sessions.db not found. Set AGENT_COWORK_DB or pass --db PATH.", file=sys.stderr)
        return 2

    since_ms: Optional[int] = None
    if args.since_days is not None:
        since_ms = int(time.time() * 1000) - int(args.since_days * 86_400_000)

    events = fetch_events(Path(db_path), since_ms, args.session_id)
    summary = summarize(events)

    if args.json:
        out = {
            "db": str(db_path),
            "since_ms": since_ms,
            "session_id": args.session_id,
            **summary,
        }
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(format_text(summary, Path(db_path), since_ms))
    return 0


if __name__ == "__main__":
    sys.exit(main())
