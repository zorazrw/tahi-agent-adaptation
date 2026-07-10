#!/usr/bin/env python3
"""
Add task ``id`` fields and reorder session-style JSON exports by tasks.json order.

Matches each item's ``task`` or ``instruction`` text against
``expertise-examples/<expertise_task>/tasks.json``, then writes items in descending
task id order (larger ids first). Items with the same id keep their original
relative order. Unmatched items are appended at the end without an ``id``.

Examples:
  python scripts/tools/add_task_ids.py scripts/log_dataviz/out_dataviz_23-context.json
  python scripts/tools/add_task_ids.py scripts/log_dataviz --dry-run
  python scripts/tools/add_task_ids.py scripts/log_dataviz/verifiers_22-offline.json
  python scripts/tools/add_task_ids.py scripts/log_dataviz --expertise-task data-viz-html
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

_tools = Path(__file__).resolve().parent
_scripts = _tools.parent
_repo = _scripts.parent

DEFAULT_GLOBS = (
    "out_dataviz_*.json",
    "redo_dataviz_*.json",
    "baseline_dataviz.json",
    "totrain_dataviz_*.json",
    "verifiers*.json",
)

_TASK1_PREFIX = "Visualize data in HTML based on the following instructions:\n"
_FUZZY_MIN_OVERLAP = 0.95


def resolve_path(path: Path) -> Path:
    if path.is_absolute() and path.exists():
        return path
    for base in (Path.cwd(), _scripts, _repo):
        cand = base / path
        if cand.exists():
            return cand
    return path


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def load_tasks(tasks_path: Path) -> list[dict[str, Any]]:
    raw = json.loads(tasks_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{tasks_path}: expected a JSON array")
    tasks = [x for x in raw if isinstance(x, dict) and isinstance(x.get("id"), int)]
    if not tasks:
        raise ValueError(f"{tasks_path}: no task rows with integer id")
    return tasks


def build_instruction_index(tasks: list[dict[str, Any]]) -> dict[str, int]:
    index: dict[str, int] = {}
    for task in tasks:
        instruction = task.get("instruction")
        if not isinstance(instruction, str):
            continue
        index[instruction] = task["id"]
        index[normalize_text(instruction)] = task["id"]
        if instruction.startswith(_TASK1_PREFIX):
            suffix = instruction.split("\n", 1)[1]
            index[suffix] = task["id"]
            index[normalize_text(suffix)] = task["id"]
    return index


def item_instruction(item: dict[str, Any]) -> str:
    for key in ("task", "instruction"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def instruction_overlap(a: str, b: str) -> float:
    left, right = a.strip(), b.strip()
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    pairs = [
        (left, right),
        (normalize_text(left), normalize_text(right)),
    ]
    return max(SequenceMatcher(None, x, y).ratio() for x, y in pairs if x and y)


def match_task_id(
    item: dict[str, Any],
    tasks: list[dict[str, Any]],
    instruction_index: dict[str, int],
) -> int | None:
    instruction = item_instruction(item)
    if not instruction:
        return None

    if instruction in instruction_index:
        return instruction_index[instruction]

    normalized = normalize_text(instruction)
    if normalized in instruction_index:
        return instruction_index[normalized]

    name = item.get("name")
    name_str = name if isinstance(name, str) else ""
    for task in tasks:
        title = task.get("title")
        if isinstance(title, str) and title and (title in instruction or title in name_str or name_str in title):
            return task["id"]

    best_id: int | None = None
    best_score = 0.0
    for task in tasks:
        catalog_instruction = task.get("instruction")
        if not isinstance(catalog_instruction, str):
            continue
        score = instruction_overlap(instruction, catalog_instruction)
        if score > best_score:
            best_score = score
            best_id = task["id"]
    if best_id is not None and best_score >= _FUZZY_MIN_OVERLAP:
        return best_id
    return None


def reorder_items(
    items: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    instruction_index: dict[str, int],
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    annotated: list[tuple[int, int, int | None, dict[str, Any]]] = []
    for index, item in enumerate(items):
        task_id = match_task_id(item, tasks, instruction_index)
        annotated.append((index, task_id if task_id is not None else -1, task_id, item))

    unmatched = [row for row in annotated if row[2] is None]
    if unmatched:
        warnings.append(f"{len(unmatched)} unmatched item(s)")

    duplicate_ids = [task_id for task_id, count in Counter(row[2] for row in annotated if row[2] is not None).items() if count > 1]
    if duplicate_ids:
        warnings.append(f"duplicate task ids: {sorted(duplicate_ids)}")

    matched = [row for row in annotated if row[2] is not None]
    matched.sort(key=lambda row: (-row[2], row[0]))

    reordered: list[dict[str, Any]] = []
    for _, _, task_id, item in matched:
        new_item = {"id": task_id}
        new_item.update(item)
        reordered.append(new_item)

    for _, _, _, item in unmatched:
        new_item = dict(item)
        new_item.pop("id", None)
        reordered.append(new_item)

    return reordered, warnings


def infer_expertise_task(items: list[dict[str, Any]], fallback: str) -> str:
    counts = Counter(
        str(item["expertise_task"]).strip()
        for item in items
        if isinstance(item.get("expertise_task"), str) and str(item["expertise_task"]).strip()
    )
    if counts:
        return counts.most_common(1)[0][0]
    return fallback


def discover_files(path: Path, globs: tuple[str, ...]) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    files: list[Path] = []
    for pattern in globs:
        files.extend(path.glob(pattern))
    return sorted({fp.resolve() for fp in files if fp.is_file()})


def process_file(
    path: Path,
    *,
    expertise_task: str,
    expertise_root: Path,
    dry_run: bool,
) -> tuple[bool, list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        return False, [f"{path.name}: skipped (empty or not a JSON array)"]
    if not isinstance(raw[0], dict):
        return False, [f"{path.name}: skipped (array items are not objects)"]
    if "task" not in raw[0] and "instruction" not in raw[0]:
        return False, [f"{path.name}: skipped (no task/instruction field)"]

    task_name = infer_expertise_task(raw, expertise_task)
    tasks_path = expertise_root / task_name / "tasks.json"
    if not tasks_path.is_file():
        return False, [f"{path.name}: tasks catalog not found at {tasks_path}"]

    tasks = load_tasks(tasks_path)
    instruction_index = build_instruction_index(tasks)
    reordered, warnings = reorder_items(raw, tasks, instruction_index)

    messages = [f"{path.name}: {len(reordered)} item(s), expertise_task={task_name}"]
    messages.extend(f"  warning: {warning}" for warning in warnings)

    if not dry_run:
        path.write_text(json.dumps(reordered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        messages.append(f"  wrote {path}")

    return bool(warnings), messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("scripts/log_dataviz"),
        help="Session JSON file or directory (default: scripts/log_dataviz)",
    )
    parser.add_argument(
        "--expertise-task",
        default="data-viz-html",
        help="Fallback expertise task when items lack expertise_task (default: data-viz-html)",
    )
    parser.add_argument(
        "--expertise-root",
        type=Path,
        default=_repo / "expertise-examples",
        help="Root directory containing expertise task folders (default: expertise-examples)",
    )
    parser.add_argument(
        "--glob",
        action="append",
        dest="globs",
        help=f"Glob pattern when input is a directory (default: {', '.join(DEFAULT_GLOBS)})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing files")
    args = parser.parse_args(argv)

    input_path = resolve_path(args.input)
    if not input_path.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    globs = tuple(args.globs) if args.globs else DEFAULT_GLOBS
    files = discover_files(input_path, globs)
    if not files:
        print(f"No matching JSON files under {input_path}", file=sys.stderr)
        return 1

    had_warnings = False
    for path in files:
        warned, messages = process_file(
            path,
            expertise_task=args.expertise_task,
            expertise_root=resolve_path(args.expertise_root),
            dry_run=args.dry_run,
        )
        had_warnings = had_warnings or warned
        for message in messages:
            print(message)

    if args.dry_run:
        print("Dry run: no files written", file=sys.stderr)

    return 1 if had_warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
