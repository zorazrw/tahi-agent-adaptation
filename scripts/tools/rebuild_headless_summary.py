#!/usr/bin/env python3
"""
Rebuild a headless run summary from existing task directories.

Reads each ``task_XXX/task_summary.json`` and matching ``ratings.json`` under ``--out-dir``,
recomputes per-task scores from cached ratings, then rewrites ``summary.json`` and ``scores.csv``.
No model calls are made.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_task_score(ratings: dict[str, Any]) -> float | None:
    average_success_rate = ratings.get("average_success_rate")
    if isinstance(average_success_rate, (int, float)):
        return float(average_success_rate) * 100.0

    tasks = ratings.get("tasks")
    if not isinstance(tasks, list):
        return None

    scores: list[float] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        versions = task.get("versions")
        if not isinstance(versions, list):
            continue
        for version in reversed(versions):
            if not isinstance(version, dict):
                continue
            score = version.get("average_success_pct")
            if isinstance(score, (int, float)):
                scores.append(float(score))
                break

    if not scores:
        return None
    return sum(scores) / len(scores)


def main() -> None:
    p = argparse.ArgumentParser(description="Rebuild headless summary.json from existing ratings.")
    p.add_argument("--out-dir", required=True, type=Path, help="Headless output directory containing task_XXX subdirs")
    p.add_argument(
        "--include-skipped",
        action="store_true",
        help="Include skipped tasks in summary.json even if they have no score",
    )
    args = p.parse_args()

    out_dir = args.out_dir.resolve()
    task_dirs = sorted([p for p in out_dir.glob("task_*") if p.is_dir()])

    summaries: list[dict[str, Any]] = []
    scored: list[float] = []
    zero_score_task_ids: list[Any] = []
    unscored_task_ids: list[Any] = []

    for task_dir in task_dirs:
        summary_path = task_dir / "task_summary.json"
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        if not isinstance(summary, dict):
            continue

        ratings_path = task_dir / "ratings.json"
        if ratings_path.exists():
            ratings = load_json(ratings_path)
            if isinstance(ratings, dict):
                score = compute_task_score(ratings)
                summary["ratings_json"] = str(ratings_path)
                summary["score"] = score
                if score is not None:
                    scored.append(score)
            else:
                summary["score"] = None
        else:
            summary["score"] = None

        if summary.get("score") is None:
            unscored_task_ids.append(summary.get("task_id"))
        elif summary.get("score") == 0:
            zero_score_task_ids.append(summary.get("task_id"))

        if args.include_skipped or summary.get("status") != "skipped" or summary.get("score") is not None:
            summaries.append(summary)

        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    overall_score = sum(scored) / len(scored) if scored else None
    report = {
        "overall_score": overall_score,
        "scored_task_count": len(scored),
        "unscored_task_count": len(summaries) - len(scored),
        "total_task_count": len(summaries),
        "zero_score_task_ids": zero_score_task_ids,
        "unscored_task_ids": unscored_task_ids,
        "tasks": summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rows = [["task_id", "status", "score", "session_id", "error"]]
    for task in summaries:
        rows.append(
            [
                json.dumps(task.get("task_id")),
                str(task.get("status") or ""),
                "" if task.get("score") is None else str(task.get("score")),
                str(task.get("session_id") or ""),
                json.dumps(task.get("error") or ""),
            ]
        )
    with (out_dir / "scores.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Rebuilt summary for {len(summaries)} tasks under {out_dir}")
    if overall_score is None:
        print("overall_score: N/A")
    else:
        print(f"overall_score: {overall_score:.1f}")
    if zero_score_task_ids:
        print(f"zero_score_task_ids: {zero_score_task_ids}")
    if unscored_task_ids:
        print(f"unscored_task_ids: {unscored_task_ids}")


if __name__ == "__main__":
    main()
