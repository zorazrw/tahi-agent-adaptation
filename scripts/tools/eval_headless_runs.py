#!/usr/bin/env python3
"""
Re-evaluate existing headless task sessions.

Given a runs root, a single run directory, or a task_XXX directory, this script finds
task_*/session.json files, overwrites each task's ratings.json via grade_redo.py,
then rebuilds summary.json and scores.csv for every touched run directory.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent.parent


def task_sort_key(path: Path) -> tuple[str, int, str]:
    name = path.parent.name
    try:
        number = int(name.removeprefix("task_"))
    except ValueError:
        number = 10**9
    return (str(path.parent.parent), number, name)


def find_sessions(path: Path) -> list[Path]:
    path = path.resolve()
    if path.is_file():
        if path.name != "session.json":
            raise SystemExit(f"Expected a session.json file, got: {path}")
        return [path]
    if not path.exists():
        raise SystemExit(f"Input not found: {path}")

    direct = path / "session.json"
    if direct.exists() and path.name.startswith("task_"):
        return [direct]

    return sorted(
        (p for p in path.rglob("session.json") if p.parent.name.startswith("task_")),
        key=task_sort_key,
    )


def run_command(args: list[str], *, log_path: Path | None = None) -> int:
    if log_path is None:
        return subprocess.run(args, cwd=REPO_ROOT).returncode

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(args, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="runs root, run directory, task_XXX directory, or session.json")
    parser.add_argument("--verifiers", type=Path, default=Path("scripts/verifiers.json"))
    parser.add_argument("--backend", choices=["anthropic", "openai"], default="openai")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL override")
    parser.add_argument("--api-key", help="OpenAI backend API key override")
    parser.add_argument("--request-timeout", type=float)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--no-rebuild-summary", action="store_true")
    args = parser.parse_args()

    verifiers = args.verifiers.resolve()
    if not verifiers.exists():
        raise SystemExit(
            "Verifier catalog not found: "
            f"{verifiers}\n"
            "Create it first, for example:\n"
            "  python scripts/tools/extract_verifiers.py out.json -o scripts/verifiers.json"
        )

    sessions: list[Path] = []
    seen: set[Path] = set()
    for path in args.paths:
        for session in find_sessions(path):
            if session not in seen:
                sessions.append(session)
                seen.add(session)

    if not sessions:
        raise SystemExit("No task_*/session.json files found.")

    run_dirs = sorted({session.parent.parent for session in sessions})
    failures: list[tuple[Path, int]] = []
    print(f"Found {len(sessions)} session(s) across {len(run_dirs)} run dir(s).")
    for index, session in enumerate(sessions, start=1):
        task_dir = session.parent
        ratings = task_dir / "ratings.json"
        log_path = task_dir / "logs" / "eval.log"
        command = [
            sys.executable,
            str(TOOLS_DIR / "grade_redo.py"),
            "--session-json",
            str(session),
            "--verifiers",
            str(verifiers),
            "--backend",
            args.backend,
            "--model",
            args.model,
            "--json-out",
            str(ratings),
        ]
        if args.base_url:
            command.extend(["--base-url", args.base_url])
        if args.api_key:
            command.extend(["--api-key", args.api_key])
        if args.request_timeout is not None:
            command.extend(["--request-timeout", str(args.request_timeout)])
        if args.max_retries is not None:
            command.extend(["--max-retries", str(args.max_retries)])

        print(f"[{index}/{len(sessions)}] grading {session}")
        code = run_command(command, log_path=log_path)
        if code != 0:
            failures.append((session, code))
            print(f"  failed with exit code {code}; see {log_path}")

    if not args.no_rebuild_summary:
        for run_dir in run_dirs:
            code = run_command([
                sys.executable,
                str(TOOLS_DIR / "rebuild_headless_summary.py"),
                "--out-dir",
                str(run_dir),
            ])
            if code != 0:
                failures.append((run_dir, code))

    if failures:
        print()
        print(f"Completed with {len(failures)} failure(s).")
        return 1

    print("Completed re-evaluation successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
