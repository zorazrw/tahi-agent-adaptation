#!/usr/bin/env python3
"""
Pairwise LM judge for heldout score exports.

Given a scores JSON (``expertise_task``, ``id``, ``artifacts`` per row) and two setup
names (e.g. ``baseline`` vs ``context``), loads the task instruction from
``expertise-examples/<expertise_task>/heldout.json`` and asks a language model which
artifact is better. Reports per-instance winners and aggregate win rate.

Examples:
  python scripts/tools/calc_win_rate.py \\
    -j scripts/scores/heldout_writing_3-offline.json \\
    --setup-a baseline --setup-b context

  # Single heldout instance by task id or array index
  python scripts/tools/calc_win_rate.py \\
    -j scripts/scores/heldout_writing_3-offline.json --id 1 \\
    --setup-a baseline --setup-b context --dry-run

  python scripts/tools/calc_win_rate.py \\
    -j scripts/scores/heldout_writing_3-offline.json --index 0 \\
    --setup-a baseline --setup-b context -o win_rate.json

  # Include user writing preferences from a memory file
  python scripts/tools/calc_win_rate.py \\
    -j scripts/scores/heldout_writing_3-offline.json \\
    --memory scripts/brain/writing_3-offline/memories/abstract-writing.md \\
    --setup-a baseline --setup-b context
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

_tools = Path(__file__).resolve().parent
_scripts = _tools.parent
_repo = _scripts.parent
for p in (_scripts, _tools):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import induce  # noqa: E402

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]  # noqa: E402
except ImportError:
    load_dotenv = None  # type: ignore[assignment]

_TRUNCATE_LEN = 14_000
DEFAULT_MODEL = "claude-sonnet-4-5"


def resolve_path(path: Path) -> Path:
    if path.exists():
        return path
    for base in (_scripts, _repo, Path.cwd()):
        cand = base / path
        if cand.exists():
            return cand
    return path


def load_scores(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit(f"{path}: expected a JSON array")
    rows = [x for x in raw if isinstance(x, dict)]
    if not rows:
        raise SystemExit(f"{path}: no score rows found")
    return rows


def select_instances(
    rows: list[dict[str, Any]],
    *,
    task_id: int | None,
    index: int | None,
) -> list[dict[str, Any]]:
    if task_id is not None and index is not None:
        raise SystemExit("Pass only one of --id or --index")
    if task_id is not None:
        matches = [r for r in rows if r.get("id") == task_id]
        if not matches:
            raise SystemExit(f"No row with id={task_id!r}")
        return matches
    if index is not None:
        if index < 0 or index >= len(rows):
            raise SystemExit(f"--index {index} out of range (0..{len(rows) - 1})")
        return [rows[index]]
    return rows


def heldout_path(expertise_task: str, heldout_root: Path) -> Path:
    return heldout_root / expertise_task / "heldout.json"


def load_heldout_catalog(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"heldout file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path.name}: expected a JSON array")
    out: dict[int, dict[str, Any]] = {}
    for row in raw:
        if isinstance(row, dict) and isinstance(row.get("id"), int):
            out[int(row["id"])] = row
    return out


def truncate_text(text: str, max_len: int = _TRUNCATE_LEN) -> str:
    return text if len(text) <= max_len else text[:max_len] + "\n... [truncated]"


def load_memory(path: Path) -> str:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise SystemExit(f"--memory file not found: {path}")
    text = resolved.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(f"--memory file is empty: {resolved}")
    return text


def memory_preferences_block(memory_text: str | None) -> str | None:
    if not memory_text or not memory_text.strip():
        return None
    return truncate_text(memory_text.strip())


def task_context_block(heldout_row: dict[str, Any] | None) -> str:
    if not heldout_row:
        return "(Task instruction unavailable.)"
    parts: list[str] = []
    task_type = str(heldout_row.get("type") or "").strip()
    if task_type:
        parts.append(f"Task type: {task_type}")
    for key in ("title", "figure"):
        val = str(heldout_row.get(key) or "").strip()
        if val:
            parts.append(f"{key.replace('_', ' ').title()}: {val}")
    instruction = str(heldout_row.get("instruction") or "").strip()
    if instruction:
        parts.append(f"Instruction:\n{truncate_text(instruction)}")
    return "\n\n".join(parts) if parts else "(Task instruction unavailable.)"


def build_comparison_prompt(
    *,
    setup_a: str,
    setup_b: str,
    content_a: str,
    content_b: str,
    heldout_row: dict[str, Any] | None,
    randomized_order: list[str],
    memory_text: str | None = None,
) -> str:
    blocks: list[str] = []
    for name in randomized_order:
        content = content_a if name == setup_a else content_b
        blocks.append(f"### {name}\n\n{truncate_text(content) if content.strip() else '(empty)'}")

    parts = [
        "You are an expert judge comparing two candidate outputs for the same task.",
        "Pick the output that better satisfies the task instruction overall.",
        "Consider accuracy, completeness, clarity, and adherence to the requested format.",
        "",
        "Task context:",
        task_context_block(heldout_row),
    ]

    prefs = memory_preferences_block(memory_text)
    if prefs:
        parts.extend(
            [
                "",
                "User preferences for writing the abstract:",
                "The user has accumulated the following preferences from prior editing sessions.",
                "Prefer the candidate that better follows these preferences, in addition to the task instruction.",
                prefs,
            ]
        )

    parts.extend(
        [
            "",
            "Candidate outputs (order is randomized):",
            "\n\n---\n\n".join(blocks),
            "",
            "Reply with ONLY a JSON object of this exact shape:",
            '{"winner": "<setup_name_or_tie>", "reason": "<one short paragraph>"}',
            f'Use exactly one of these winner values: "{setup_a}", "{setup_b}", or "tie".',
        ]
    )
    return "\n".join(parts)


def parse_json_from_model_text(text: str) -> dict[str, Any]:
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    raw = (fence.group(1) if fence else text).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object in model response")
    parsed = json.loads(raw[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed


def normalize_winner(raw: Any, setup_a: str, setup_b: str) -> str:
    if raw is None:
        raise ValueError("Missing winner field")
    winner = str(raw).strip().lower()
    if winner in {"tie", "draw", "equal", "same"}:
        return "tie"
    a_lower, b_lower = setup_a.lower(), setup_b.lower()
    if winner == a_lower or winner == setup_a:
        return setup_a
    if winner == b_lower or winner == setup_b:
        return setup_b
    raise ValueError(f"winner must be {setup_a!r}, {setup_b!r}, or 'tie'; got {raw!r}")


def call_judge_llm(client: Any, model: str, prompt: str, *, max_tokens: int) -> str:
    try:
        return induce.anthropic_user_text(client, model, prompt, max_tokens=max_tokens, temperature=0.0)
    except Exception as exc:
        tail = str(exc)
        hint = ""
        code = getattr(exc, "status_code", None)
        if code == 401 or "401" in tail or "authentication" in tail.lower():
            hint = (
                "\n\nAuthentication hint: save Anthropic in app Settings (pi-agent/auth.json) or set "
                "ANTHROPIC_API_KEY."
            )
        raise RuntimeError(f"Judge API error: {tail}{hint}") from exc


def compare_instance(
    row: dict[str, Any],
    *,
    setup_a: str,
    setup_b: str,
    heldout_root: Path,
    heldout_cache: dict[str, dict[int, dict[str, Any]]],
    client: Any | None,
    model: str,
    dry_run: bool,
    max_tokens: int,
    seed: int | None,
    memory_text: str | None = None,
) -> dict[str, Any]:
    task_id = row.get("id")
    expertise_task = str(row.get("expertise_task") or "").strip()
    artifacts = row.get("artifacts")
    if not isinstance(artifacts, dict):
        return {
            "id": task_id,
            "expertise_task": expertise_task,
            "error": "row has no artifacts object",
        }

    content_a = artifacts.get(setup_a)
    content_b = artifacts.get(setup_b)
    if not isinstance(content_a, str):
        return {
            "id": task_id,
            "expertise_task": expertise_task,
            "error": f"missing or non-string artifact {setup_a!r}",
        }
    if not isinstance(content_b, str):
        return {
            "id": task_id,
            "expertise_task": expertise_task,
            "error": f"missing or non-string artifact {setup_b!r}",
        }
    if not content_a.strip() and not content_b.strip():
        return {
            "id": task_id,
            "expertise_task": expertise_task,
            "error": "both artifacts are empty",
        }

    heldout_row: dict[str, Any] | None = None
    if expertise_task and isinstance(task_id, int):
        if expertise_task not in heldout_cache:
            path = heldout_path(expertise_task, heldout_root)
            try:
                heldout_cache[expertise_task] = load_heldout_catalog(path)
            except (OSError, ValueError, FileNotFoundError) as exc:
                heldout_cache[expertise_task] = {}
                return {
                    "id": task_id,
                    "expertise_task": expertise_task,
                    "error": f"failed to load heldout catalog: {exc}",
                }
        heldout_row = heldout_cache[expertise_task].get(task_id)

    rng = random.Random(seed if seed is not None else task_id)
    randomized_order = [setup_a, setup_b]
    rng.shuffle(randomized_order)

    prompt = build_comparison_prompt(
        setup_a=setup_a,
        setup_b=setup_b,
        content_a=content_a,
        content_b=content_b,
        heldout_row=heldout_row,
        randomized_order=randomized_order,
        memory_text=memory_text,
    )

    result: dict[str, Any] = {
        "id": task_id,
        "expertise_task": expertise_task,
        "setup_a": setup_a,
        "setup_b": setup_b,
        "randomized_order": randomized_order,
        "heldout_found": heldout_row is not None,
    }

    if dry_run:
        result["prompt"] = prompt
        result["winner"] = None
        return result

    if client is None:
        result["error"] = "missing Anthropic client (credentials)"
        return result

    try:
        raw = call_judge_llm(client, model, prompt, max_tokens=max_tokens)
        parsed = parse_json_from_model_text(raw)
        winner = normalize_winner(parsed.get("winner"), setup_a, setup_b)
        result["winner"] = winner
        result["reason"] = str(parsed.get("reason") or "").strip() or None
        result["raw_text"] = raw
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)

    return result


def summarize_results(results: list[dict[str, Any]], *, setup_a: str, setup_b: str) -> dict[str, Any]:
    setup_a_wins = setup_b_wins = ties = errors = 0
    for row in results:
        if row.get("error"):
            errors += 1
            continue
        winner = row.get("winner")
        if winner == setup_a:
            setup_a_wins += 1
        elif winner == setup_b:
            setup_b_wins += 1
        elif winner == "tie":
            ties += 1

    decided = setup_a_wins + setup_b_wins
    win_rate_b = (setup_b_wins / decided) if decided else None
    return {
        "total": len(results),
        "setup_a": setup_a,
        "setup_b": setup_b,
        f"{setup_a}_wins": setup_a_wins,
        f"{setup_b}_wins": setup_b_wins,
        "ties": ties,
        "errors": errors,
        f"{setup_b}_win_rate_excluding_ties": win_rate_b,
    }


def print_summary(summary: dict[str, Any]) -> None:
    setup_a = str(summary.get("setup_a") or "setup_a")
    setup_b = str(summary.get("setup_b") or "setup_b")
    print(f"--- win rate: {setup_b} vs {setup_a} ---", file=sys.stderr)
    print(
        f"  {setup_b} wins: {summary.get(f'{setup_b}_wins', 0)} | "
        f"{setup_a} wins: {summary.get(f'{setup_a}_wins', 0)} | "
        f"ties: {summary.get('ties', 0)} | errors: {summary.get('errors', 0)}",
        file=sys.stderr,
    )
    rate = summary.get(f"{setup_b}_win_rate_excluding_ties")
    if isinstance(rate, (int, float)):
        print(f"  {setup_b} win rate (excluding ties): {rate * 100:.1f}%", file=sys.stderr)
    print("--- end summary ---", file=sys.stderr)


def resolve_client(args: argparse.Namespace) -> tuple[Any | None, str]:
    if args.dry_run:
        return None, (args.model or "").strip() or DEFAULT_MODEL
    try:
        cfg = induce.resolve_anthropic_config(
            skip_api_config=args.no_api_config,
            skip_claude_settings=args.no_claude_settings,
        )
    except induce.AnthropicConfigError as exc:
        raise SystemExit(str(exc)) from exc

    key = (args.api_key or "").strip() or cfg.api_key
    key = key.strip().replace("\r", "").replace("\n", "")
    base = (args.base_url or "").strip().rstrip("/") or cfg.base_url
    model = (args.model or "").strip() or DEFAULT_MODEL
    cfg = induce.ResolvedAnthropicConfig(key, base, model)
    if key.startswith("tml-") or key.startswith("tml_"):
        raise SystemExit("Resolved key looks like TINKER_API_KEY; set ANTHROPIC_API_KEY.")
    return induce.make_anthropic_client(cfg), model


def load_dotenv_defaults(env_file: Path | None, override: bool) -> None:
    if load_dotenv is None:
        return
    if env_file and env_file.is_file():
        load_dotenv(env_file, override=override)
        return
    scripts_env = _scripts / ".env"
    if scripts_env.is_file():
        load_dotenv(scripts_env, override=override)
    load_dotenv(override=override)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-j", "--json", type=Path, required=True, help="Scores JSON array")
    p.add_argument("--id", type=int, default=None, help="Heldout task id (field ``id`` in scores row)")
    p.add_argument("--index", type=int, default=None, help="0-based index into the scores JSON array")
    p.add_argument("--setup-a", default="baseline", help="First setup name (default: baseline)")
    p.add_argument("--setup-b", default="context", help="Second setup name (default: context)")
    p.add_argument(
        "--heldout-root",
        type=Path,
        default=_repo / "expertise-examples",
        help="Root directory containing per-task heldout.json files",
    )
    p.add_argument(
        "--memory",
        type=Path,
        default=None,
        help="Memory file with user preferences for abstract writing (e.g. brain memories/*.md)",
    )
    p.add_argument("-o", "--out", type=Path, default=None, help="Write full report JSON here")
    p.add_argument("--dry-run", action="store_true", help="Build prompts only; do not call the LM")
    p.add_argument("--model", default=None, help=f"Anthropic model (default: {DEFAULT_MODEL})")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=None, help="Random seed for presentation order")
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--env-file", type=Path, default=None)
    p.add_argument("--dotenv-override", action="store_true")
    p.add_argument("--no-claude-settings", action="store_true")
    p.add_argument("--no-api-config", action="store_true")
    args = p.parse_args()

    setup_a = str(args.setup_a).strip()
    setup_b = str(args.setup_b).strip()
    if not setup_a or not setup_b:
        raise SystemExit("--setup-a and --setup-b must be non-empty")
    if setup_a == setup_b:
        raise SystemExit("--setup-a and --setup-b must differ")

    load_dotenv_defaults(args.env_file, args.dotenv_override)
    scores_path = resolve_path(args.json)
    rows = load_scores(scores_path)
    selected = select_instances(rows, task_id=args.id, index=args.index)

    client, model = resolve_client(args)
    heldout_cache: dict[str, dict[int, dict[str, Any]]] = {}
    heldout_root = resolve_path(args.heldout_root)
    memory_text: str | None = load_memory(args.memory) if args.memory else None
    memory_file = str(resolve_path(args.memory)) if args.memory else None

    results = [
        compare_instance(
            row,
            setup_a=setup_a,
            setup_b=setup_b,
            heldout_root=heldout_root,
            heldout_cache=heldout_cache,
            client=client,
            model=model,
            dry_run=args.dry_run,
            max_tokens=args.max_tokens,
            seed=args.seed,
            memory_text=memory_text,
        )
        for row in selected
    ]

    report: dict[str, Any] = {
        "scores_file": str(scores_path),
        "setup_a": setup_a,
        "setup_b": setup_b,
        "model": model,
        "memory_file": memory_file,
        "dry_run": args.dry_run,
        "results": results,
        "summary": summarize_results(results, setup_a=setup_a, setup_b=setup_b),
    }

    print_summary(report["summary"])

    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote report to {args.out.resolve()}", file=sys.stderr)
    else:
        sys.stdout.write(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
