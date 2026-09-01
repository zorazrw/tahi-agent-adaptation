#!/usr/bin/env python3
"""
Summarize joint general verifiers from user history + human guidelines.

Takes high-quality (manual) verifiers across past tasks and a fixed human guideline
list, then asks an LLM for one reusable general verifier list that covers every
human guideline. Writes output in the same shape as the human guidelines file:
an array of ``{uuid, instruction, verifiers}`` with the same summarized verifier
set on every task.

Examples:
  python scripts/tools/create_verifiers.py \\
    --manual scripts/log_dataviz/verifiers_evolve/verifiers-manual_17-context.json \\
    --human scripts/log_dataviz/verifiers_human/verifiers-human_17-context.json \\
    -o scripts/log_dataviz/verifiers_summary/verifiers-summary_17-context.json

  # Inspect the prompt without calling the LM
  python scripts/tools/create_verifiers.py \\
    --manual ... --human ... --print-prompt

  # Call the LM and print results without writing
  python scripts/tools/create_verifiers.py \\
    --manual ... --human ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import uuid
from pathlib import Path
from typing import Any

_tools = Path(__file__).resolve().parent
_scripts = _tools.parent
_repo = _scripts.parent
for p in (_scripts, _tools):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import induce  # noqa: E402
from pi_llm import runtime_llm_text  # noqa: E402

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]  # noqa: E402
except ImportError:
    load_dotenv = None  # type: ignore[assignment]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = induce.DEFAULT_MODEL

SUMMARIZE_SYSTEM = """You distill grading rubrics for HTML data-visualization tasks.

You are given:
(A) High-quality, task-specific verifiers written/edited by a user across many past tasks.
(B) A fixed list of human-written general guidelines that every visualization should satisfy.

Your job: produce a JOINT GENERAL verifier list for this expertise overall.

Rules:
- Capture recurring themes from the user's history (layout, labels, legends, annotations,
  spacing/overlap, axes/scales, colors, HTML validity, browser renderability,
  chart-type-specific checks when they recur).
- Phrase items as general, reusable criteria (not tied to one past task's numbers/labels).
- Do NOT include criteria whose only job is checking that plotted numeric values match
  an instruction (those are task-specific). Prefer structural/visual quality checks.
- MUST cover every human guideline in (B). Either keep them (possibly tightened) or rewrite
  them so the same requirement is clearly present. Do not drop a human guideline.
- Prefer the user's concrete, falsifiable style from (A) over vague wording.
- Deduplicate near-duplicates; keep one clear criterion per idea.
- Typically 10–20 items. Prefer fewer clear lines over a long union.

Reply with JSON only (no markdown fences, no commentary):
{
  "general_verifiers": ["...", "..."],
  "human_coverage": [
    {"human": "<exact human guideline>", "covered_by": "<matching general_verifiers item>"}
  ]
}
Every human guideline must appear exactly once in human_coverage."""


def resolve_path(path: Path) -> Path:
    if path.exists():
        return path
    for base in (Path.cwd(), _scripts, _repo):
        cand = base / path
        if cand.exists():
            return cand
    return path


def load_json(path: Path) -> Any:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise SystemExit(f"File not found: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def parse_json_from_model_text(text: str) -> dict[str, Any]:
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    raw = (fence.group(1) if fence else text).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        preview = raw[:400] + ("..." if len(raw) > 400 else "")
        raise ValueError(f"No JSON object in model response. Preview: {preview!r}")
    parsed = json.loads(raw[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed


def normalize_str_list(raw: Any, *, field: str) -> list[str]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Expected non-empty list for {field!r}")
    out: list[str] = []
    for item in raw:
        s = str(item).strip()
        if s:
            out.append(s)
    if not out:
        raise ValueError(f"Empty {field} after stripping")
    return out


def load_human_catalog(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load human catalog; return (all task entries, guidelines from first entry).

    Guideline lists are identical across tasks, so only the first entry's verifiers
    are used as the guideline set. Task entries keep uuid + instruction for output.
    """
    data = load_json(path)
    if isinstance(data, list) and data and all(isinstance(x, str) for x in data):
        guidelines = [str(v).strip() for v in data if str(v).strip()]
        if not guidelines:
            raise SystemExit(f"No human guidelines found in {path}")
        return [], guidelines

    if isinstance(data, dict):
        guidelines = [
            str(v).strip()
            for v in (data.get("verifiers") or data.get("human_guidelines") or [])
            if str(v).strip()
        ]
        if not guidelines:
            raise SystemExit(f"No human guidelines found in {path}")
        return [], guidelines

    if not isinstance(data, list) or not data:
        raise SystemExit(f"--human must be a non-empty JSON array or object: {path}")

    entries: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        instruction = str(item.get("instruction") or "").strip()
        if not instruction:
            continue
        entries.append(
            {
                "uuid": str(item["uuid"]) if item.get("uuid") else None,
                "instruction": instruction,
            }
        )

    if not entries:
        raise SystemExit(f"No task entries with instructions in {path}")

    first_verifiers = data[0].get("verifiers") if isinstance(data[0], dict) else None
    if not isinstance(first_verifiers, list) or not first_verifiers:
        raise SystemExit(f"No human guidelines found in first entry of {path}")
    guidelines = [str(v).strip() for v in first_verifiers if str(v).strip()]
    if not guidelines:
        raise SystemExit(f"No human guidelines found in {path}")

    logger.info(
        "Using human guidelines from first catalog entry only "
        "(same set applied to all %d task entries)",
        len(entries),
    )
    return entries, guidelines


def format_summary_catalog(
    task_entries: list[dict[str, Any]],
    verifiers: list[str],
) -> list[dict[str, Any]]:
    """Same shape as human guidelines: [{uuid, instruction, verifiers}, ...] with one shared set."""
    out: list[dict[str, Any]] = []
    for entry in task_entries:
        uid = entry.get("uuid") or str(uuid.uuid4())
        out.append(
            {
                "uuid": str(uid),
                "instruction": entry["instruction"],
                "verifiers": list(verifiers),
            }
        )
    return out


def format_manual_history(
    manual_catalog: list[dict[str, Any]], *, max_tasks: int | None = None
) -> str:
    parts: list[str] = []
    entries = manual_catalog if max_tasks is None else manual_catalog[:max_tasks]
    for i, entry in enumerate(entries, start=1):
        instruction = str(entry.get("instruction") or "").strip()
        verifiers = [
            str(v).strip() for v in (entry.get("verifiers") or []) if str(v).strip()
        ]
        uid = entry.get("uuid") or entry.get("id") or i
        parts.append(f"### Past task {i} (id={uid})")
        parts.append("Instruction:")
        parts.append(instruction)
        parts.append("User verifiers:")
        for j, v in enumerate(verifiers, start=1):
            parts.append(f"  {j}. {v}")
        parts.append("")
    return "\n".join(parts).strip()


def build_summarize_user_prompt(
    manual_catalog: list[dict[str, Any]],
    human_guidelines: list[str],
    *,
    max_history_tasks: int | None = None,
) -> str:
    parts = [
        f"You are given {len(manual_catalog)} past tasks with high-quality user verifiers "
        "and a fixed human guideline list.",
        "",
        "## (A) User high-quality verifier history",
        format_manual_history(manual_catalog, max_tasks=max_history_tasks),
        "",
        "## (B) Human-written guidelines (must all be covered)",
    ]
    for i, g in enumerate(human_guidelines, start=1):
        parts.append(f"  {i}. {g}")
    parts.append("")
    parts.append(
        "Produce the joint general verifier list now as JSON "
        "(general_verifiers + human_coverage)."
    )
    return "\n".join(parts)


def call_llm(system: str, user: str, *, model: str | None, max_tokens: int) -> str:
    runtime = induce.resolve_induce_llm(model_override=model)
    logger.info("Calling %s/%s …", runtime.provider, runtime.model)
    return runtime_llm_text(runtime, system, user, max_tokens=max_tokens)


def _missing_human_coverage(human_guidelines: list[str], coverage: Any) -> list[str]:
    if not isinstance(coverage, list):
        return list(human_guidelines)
    covered: set[str] = set()
    for item in coverage:
        if not isinstance(item, dict):
            continue
        h = str(item.get("human") or "").strip().lower()
        if h:
            covered.add(h)
    return [g for g in human_guidelines if g.lower() not in covered]


def summarize_general_verifiers(
    manual_catalog: list[dict[str, Any]],
    human_guidelines: list[str],
    *,
    model: str | None,
    max_tokens: int,
    max_history_tasks: int | None = None,
) -> dict[str, Any]:
    user = build_summarize_user_prompt(
        manual_catalog, human_guidelines, max_history_tasks=max_history_tasks
    )
    raw = call_llm(SUMMARIZE_SYSTEM, user, model=model, max_tokens=max_tokens)
    parsed = parse_json_from_model_text(raw)
    general = normalize_str_list(parsed.get("general_verifiers"), field="general_verifiers")
    coverage = parsed.get("human_coverage")
    missing = _missing_human_coverage(human_guidelines, coverage)
    if missing:
        logger.warning(
            "Model left %d human guideline(s) uncovered; appending them to general_verifiers",
            len(missing),
        )
        existing = {g.lower() for g in general}
        for g in missing:
            if g.lower() not in existing:
                general.append(g)
                existing.add(g.lower())
    return {
        "general_verifiers": general,
        "human_coverage": coverage if isinstance(coverage, list) else [],
        "human_guidelines": human_guidelines,
        "source_manual_tasks": len(manual_catalog),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def abs_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Summarize joint general verifiers from user history + human guidelines"
        )
    )
    p.add_argument(
        "--manual",
        type=Path,
        required=True,
        help="High-quality user verifiers JSON (array of {instruction, verifiers})",
    )
    p.add_argument(
        "--human",
        type=Path,
        required=True,
        help=(
            "Human-written guideline verifiers JSON; first entry's verifiers are the "
            "guideline set; output mirrors this file's {uuid, instruction, verifiers} "
            "shape with the summarized set on every task"
        ),
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output summary JSON path (required unless --print-prompt or --dry-run)",
    )
    p.add_argument(
        "--max-history-tasks",
        type=int,
        default=None,
        help="Limit how many past manual tasks are included in the prompt (default: all)",
    )
    p.add_argument(
        "--model",
        default=None,
        help=f"Override model (default: {DEFAULT_MODEL}; Anthropic via induce auth)",
    )
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--print-prompt", action="store_true", help="Print prompt and exit")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Call the LM and print results; do not write -o",
    )
    p.add_argument("--env-file", type=Path)
    p.add_argument("--dotenv-override", action="store_true")
    args = p.parse_args(argv)

    if not args.print_prompt and not args.dry_run and not args.output:
        p.error("Provide -o/--output, or use --print-prompt / --dry-run")

    env_file = args.env_file or (_scripts / ".env")
    if load_dotenv is not None and env_file.is_file():
        load_dotenv(env_file, override=args.dotenv_override)

    manual_catalog = load_json(args.manual)
    if not isinstance(manual_catalog, list) or not manual_catalog:
        raise SystemExit(f"--manual must be a non-empty JSON array: {args.manual}")
    human_entries, human_guidelines = load_human_catalog(args.human)
    # Fall back to manual task skeletons if --human is only a bare verifier list.
    task_entries = human_entries or [
        {
            "uuid": str(t["uuid"]) if t.get("uuid") else None,
            "instruction": str(t.get("instruction") or "").strip(),
        }
        for t in manual_catalog
        if str(t.get("instruction") or "").strip()
    ]
    logger.info(
        "Loaded %d manual tasks, %d human guidelines, %d output task slots",
        len(manual_catalog),
        len(human_guidelines),
        len(task_entries),
    )

    summarize_user = build_summarize_user_prompt(
        manual_catalog, human_guidelines, max_history_tasks=args.max_history_tasks
    )

    if args.print_prompt:
        print("=== SYSTEM ===")
        print(SUMMARIZE_SYSTEM)
        print("\n=== USER ===")
        print(summarize_user)
        return 0

    summary = summarize_general_verifiers(
        manual_catalog,
        human_guidelines,
        model=args.model,
        max_tokens=args.max_tokens,
        max_history_tasks=args.max_history_tasks,
    )
    general = summary["general_verifiers"]
    logger.info("Joint general verifiers: %d", len(general))
    for i, v in enumerate(general, start=1):
        print(f"{i}. {v}")

    catalog = format_summary_catalog(task_entries, general)
    if args.dry_run:
        print(f"\nWould write {len(catalog)} task entries with the same verifier set")
        return 0

    out = abs_path(args.output)
    write_json(out, catalog)
    logger.info(
        "Wrote %s (%d tasks, %d shared verifiers each)",
        out,
        len(catalog),
        len(general),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())