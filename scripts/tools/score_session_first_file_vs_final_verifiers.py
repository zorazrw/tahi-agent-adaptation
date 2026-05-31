#!/usr/bin/env python3
"""
Score a task session's output-file snapshot against its **final** verifier criteria.

Walks exported session JSON (legacy ``trajectory`` or ``task_units`` from ``export_task_sessions``),
loads the **first** ``environment.file`` snapshot and calls an LM to label pass/fail per criterion, or
with ``--last-file`` reads the final snapshot's exported ``verifier.status`` values (no LM; pass when
``status`` is ``success``). Verifier criteria always come from the last non-empty
``environment.workflow[*].verifiers`` set.

Credentials / dotenv (``--first-file`` / LM path only) match ``scripts/induce.py``.

Examples:
  # All sessions in the file (omit ``-s``):
  python scripts/tools/score_session_first_file_vs_final_verifiers.py -j scripts/out.json

  python scripts/tools/score_session_first_file_vs_final_verifiers.py \\
    -j scripts/out.json -s 955c6bba-ec75-4f00-a706-85945b50e4d5

  python scripts/tools/score_session_first_file_vs_final_verifiers.py \\
    -j scripts/out.json --json-out -o scores.json

  # Final file + exported verifier statuses (no LM):
  python scripts/tools/score_session_first_file_vs_final_verifiers.py -j scripts/out.json --last-file
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator, Literal

FileVersion = Literal["first", "last"]

_scripts_dir = Path(__file__).resolve().parent.parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import induce  # noqa: E402
from dotenv import load_dotenv  # noqa: E402


def load_sessions(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        if isinstance(raw.get("sessions"), list):
            return [x for x in raw["sessions"] if isinstance(x, dict)]
        return [raw]
    raise ValueError("JSON root must be a session object or an array of sessions")


def select_sessions(sessions: list[dict[str, Any]], session_id: str | None) -> list[dict[str, Any]]:
    """One session when ``-s`` is set; otherwise every session in the file."""
    if session_id is not None and str(session_id).strip():
        sid = str(session_id).strip()
        for s in sessions:
            if s.get("uuid") == sid:
                return [s]
        raise SystemExit(f"No session with uuid {sid!r}")
    if not sessions:
        raise SystemExit("No sessions in JSON")
    return sessions


def iter_environments_chronological(session: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield ``environment`` dicts in session order (``task_units`` or flat ``trajectory``)."""
    units = session.get("task_units")
    if isinstance(units, list) and units:
        for unit in units:
            if not isinstance(unit, dict):
                continue
            env = unit.get("environment")
            if isinstance(env, dict):
                yield env
        return

    traj = session.get("trajectory")
    if not isinstance(traj, list):
        return
    for step in traj:
        if not isinstance(step, dict):
            continue
        env = step.get("environment")
        if isinstance(env, dict):
            yield env


def _file_field(env: dict[str, Any]) -> Any:
    ff = env.get("file")
    if ff is None:
        ff = env.get("files")
    return ff


def file_blocks_from_env(env: dict[str, Any]) -> list[tuple[str, str]]:
    """Sorted ``(path, content)`` pairs with non-empty string content."""
    ff = _file_field(env)
    blocks: list[tuple[str, str]] = []
    if isinstance(ff, dict):
        for path, content in ff.items():
            if isinstance(path, str) and isinstance(content, str) and content:
                blocks.append((path, content))
    elif isinstance(ff, list):
        for item in ff:
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            content = item.get("content")
            if isinstance(path, str) and isinstance(content, str) and content:
                blocks.append((path, content))
    blocks.sort(key=lambda x: x[0])
    return blocks


def file_blocks_for_session(
    session: dict[str, Any],
    *,
    version: FileVersion = "first",
) -> tuple[list[tuple[str, str]], int | None]:
    """
    First or last environment snapshot that includes at least one file with string content.

    Returns ``(blocks, environment_index)`` where ``environment_index`` is 0-based over
    chronological environments (``task_units`` order).
    """
    last_blocks: list[tuple[str, str]] = []
    last_idx: int | None = None
    for idx, env in enumerate(iter_environments_chronological(session)):
        blocks = file_blocks_from_env(env)
        if not blocks:
            continue
        if version == "first":
            return blocks, idx
        last_blocks = blocks
        last_idx = idx
    if version == "last":
        return last_blocks, last_idx
    return [], None


def verifier_entries_from_env(env: dict[str, Any]) -> list[dict[str, Any]]:
    """``{criterion, status, pass}`` from top-level ``environment.workflow`` nodes."""
    wf = env.get("workflow")
    if not isinstance(wf, list):
        return []
    out: list[dict[str, Any]] = []
    for node in wf:
        if not isinstance(node, dict):
            continue
        for v in node.get("verifiers") or []:
            if isinstance(v, dict):
                c = v.get("criterion")
                if not isinstance(c, str) or not c.strip():
                    continue
                status = str(v.get("status") or "")
                out.append(
                    {
                        "criterion": c.strip(),
                        "status": status,
                        "pass": status == "success",
                    }
                )
            elif isinstance(v, str) and v.strip():
                out.append({"criterion": v.strip(), "status": "", "pass": False})
    return out


def final_verifier_entries(session: dict[str, Any]) -> tuple[list[dict[str, Any]], int | None]:
    """Last non-empty verifier list in chronological order."""
    last: list[dict[str, Any]] = []
    last_idx: int | None = None
    for idx, env in enumerate(iter_environments_chronological(session)):
        entries = verifier_entries_from_env(env)
        if entries:
            last = entries
            last_idx = idx
    return last, last_idx


def final_verifier_criteria(session: dict[str, Any]) -> tuple[list[str], int | None]:
    entries, idx = final_verifier_entries(session)
    return [str(e["criterion"]) for e in entries], idx


def truncate_file_text(text: str, max_len: int = 14_000) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n... [truncated]"


def build_labeler_user_message(criteria: list[str], file_blocks: list[tuple[str, str]]) -> str:
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(criteria))
    rendered: list[str] = []
    for rel, text in file_blocks:
        body = truncate_file_text(text)
        rendered.append(f"### {rel}\n\n{body}")
    files_joined = "\n\n---\n\n".join(rendered) if rendered else "(no output files)"

    return "\n".join(
        [
            "You are an automated checker for completed task output files.",
            "Given verifier criteria and the current output files (below), decide whether each criterion is satisfied.",
            'Reply with ONLY a JSON object of this exact shape: {"results":[{"pass":true},{"pass":false},...]}',
            "The results array must have exactly one object per verifier line, in the same order (indices 0 .. n-1).",
            "pass: true means the criterion is satisfied; false means it is not.",
            "",
            "Verifier criteria (in order):",
            numbered,
            "",
            "Output files and contents:",
            files_joined,
        ]
    )


def parse_json_from_model_text(text: str) -> dict[str, Any]:
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    raw = (fence.group(1) if fence else text).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object in model response")
    return json.loads(raw[start : end + 1])


def interpret_results(text: str, n: int) -> list[bool | None]:
    parsed = parse_json_from_model_text(text)
    results = parsed.get("results")
    if not isinstance(results, list):
        raise ValueError("Missing results array")
    out: list[bool | None] = [None] * n
    for i in range(min(n, len(results))):
        r = results[i]
        if isinstance(r, dict) and "pass" in r:
            out[i] = bool(r["pass"])
    return out


def average_success_percent(passes: list[bool | None], n_criteria: int) -> float | None:
    if n_criteria <= 0:
        return None
    return 100.0 * sum(1 for p in passes if p is True) / n_criteria


def score_session(
    session: dict[str, Any],
    *,
    client: Any | None,
    model: str,
    dry_run: bool,
    file_version: FileVersion = "first",
) -> dict[str, Any]:
    entries, verifier_env_idx = final_verifier_entries(session)
    criteria = [str(e["criterion"]) for e in entries]
    files, file_env_idx = file_blocks_for_session(session, version=file_version)
    use_exported_status = file_version == "last"

    out: dict[str, Any] = {
        "uuid": session.get("uuid"),
        "name": session.get("name"),
        "file_version": file_version,
        "scoring_method": "verifier_status" if use_exported_status else "lm",
        "file_environment_index": file_env_idx,
        "first_file_environment_index": file_env_idx if file_version == "first" else None,
        "final_verifier_environment_index": verifier_env_idx,
        "output_files": [p for p, _ in files],
        "final_rubrics": criteria,
        "dry_run": dry_run,
        "lm": None,
        "error": None,
        "average_success_pct": None,
    }

    if not criteria:
        out["error"] = "no final verifier criteria found in session environments"
        return out

    if use_exported_status:
        passes = [bool(e.get("pass")) for e in entries]
        statuses = [str(e.get("status") or "") for e in entries]
        out["verifier_statuses"] = statuses
        out["pass_per_criterion"] = passes
        pct = average_success_percent(passes, len(criteria))
        if pct is not None:
            out["average_success_pct"] = round(pct, 1)
        return out

    if not files:
        out["error"] = "no file content found in session environments"
        return out

    prompt = build_labeler_user_message(criteria, files)
    out["prompt"] = prompt if dry_run else None

    if dry_run:
        return out

    if client is None:
        out["error"] = "missing Anthropic client (credentials)"
        return out

    try:
        raw = induce.anthropic_user_text(client, model, prompt, max_tokens=1024, temperature=0.0)
        passes = interpret_results(raw, len(criteria))
        out["lm"] = {
            "raw_text": raw,
            "pass_per_criterion": passes,
            "criteria": list(criteria),
        }
        pct = average_success_percent(passes, len(criteria))
        if pct is not None:
            out["average_success_pct"] = round(pct, 1)
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)

    return out


def print_human_summary(report: dict[str, Any], *, compact: bool = False) -> None:
    uid = report.get("uuid") or ""
    name = report.get("name") or ""
    if compact:
        pct = report.get("average_success_pct")
        pct_s = f"{float(pct):.1f}%" if isinstance(pct, (int, float)) else "N/A"
        err = report.get("error")
        label = (name or uid)[:56]
        if err:
            print(f"  {label}: ERROR — {err}")
        elif report.get("dry_run"):
            n = len(report.get("final_rubrics") or [])
            print(f"  {label}: dry-run ({n} criteria, {len(report.get('output_files') or [])} file(s))")
        else:
            method = report.get("scoring_method") or ""
            suffix = " (exported status)" if method == "verifier_status" else ""
            print(f"  {label}: {pct_s}{suffix}")
        return

    print(f"session: {uid}" + (f" — {name}" if name else ""))
    fv = report.get("file_version") or "first"
    method = report.get("scoring_method") or ""
    print(f"scoring: {method}" + (" (no LM)" if method == "verifier_status" else ""))
    if report.get("file_environment_index") is not None:
        print(f"{fv} file @ environment index: {report.get('file_environment_index')}")
    print(f"final verifiers @ environment index: {report.get('final_verifier_environment_index')}")
    if report.get("output_files"):
        print(f"output files: {', '.join(report.get('output_files') or [])}")
    print(f"criteria count: {len(report.get('final_rubrics') or [])}")

    if report.get("error"):
        print(f"error: {report['error']}")
        return

    if report.get("dry_run"):
        print("(dry-run — no LM call)")
        return

    criteria = report.get("final_rubrics") or []
    passes = report.get("pass_per_criterion")
    statuses = report.get("verifier_statuses")
    if not isinstance(passes, list):
        lm = report.get("lm")
        if isinstance(lm, dict):
            criteria = lm.get("criteria") or criteria
            passes = lm.get("pass_per_criterion") or []
        else:
            passes = []

    for i, c in enumerate(criteria):
        lab = passes[i] if i < len(passes) else None
        tag = "PASS" if lab is True else "FAIL" if lab is False else "?"
        st = statuses[i] if isinstance(statuses, list) and i < len(statuses) else ""
        st_s = f" [{st}]" if st and method == "verifier_status" else ""
        print(f"  [{i:02d}] {tag}{st_s} — {c}")

    pct = report.get("average_success_pct")
    passed = sum(1 for p in passes if p is True)
    total = len(criteria)
    if isinstance(pct, (int, float)):
        print(f"\naverage_success_rate: {passed}/{total} = {float(pct):.1f}%")
    else:
        print(f"\naverage_success_rate: {passed}/{total}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="LM-score session file snapshot (first or last) against final verifier criteria."
    )
    p.add_argument(
        "--last-file",
        action="store_true",
        help=(
            "Score using exported verifier status from the final rubric set (pass=success); "
            "no LM call. Still reports last file snapshot index when present."
        ),
    )
    p.add_argument("--json", "-j", type=Path, required=True, help="Session JSON path")
    p.add_argument(
        "--session",
        "-s",
        type=str,
        default=None,
        help="Score only this session uuid (default: all sessions in the JSON file)",
    )
    p.add_argument("--out", "-o", type=Path, default=None, help="Write JSON report to this path")
    p.add_argument("--json-out", action="store_true", help="Print full JSON report to stdout")
    p.add_argument("--dry-run", action="store_true", help="Build prompt only; do not call API")
    p.add_argument("--model", type=str, default=None, help="Override Anthropic model")
    p.add_argument("--no-api-config", action="store_true")
    p.add_argument("--no-claude-settings", action="store_true")
    p.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Dotenv file (default: scripts/.env then ./.env)",
    )
    p.add_argument("--dotenv-override", action="store_true")
    args = p.parse_args()

    if args.env_file is not None:
        load_dotenv(args.env_file, override=bool(args.dotenv_override))
    else:
        scripts_env = _scripts_dir / ".env"
        if scripts_env.is_file():
            load_dotenv(scripts_env, override=bool(args.dotenv_override))
        load_dotenv(override=bool(args.dotenv_override))

    sessions = load_sessions(args.json)
    targets = select_sessions(sessions, args.session)

    file_version: FileVersion = "last" if args.last_file else "first"
    need_llm = file_version == "first" and not args.dry_run

    client = None
    model = args.model or "claude-sonnet-4-20250514"
    if need_llm:
        cfg = induce.resolve_anthropic_config(
            skip_api_config=bool(args.no_api_config),
            skip_claude_settings=bool(args.no_claude_settings),
        )
        model = args.model or cfg.model
        client = induce.make_anthropic_client(cfg)

    reports: list[dict[str, Any]] = []
    for session in targets:
        reports.append(
            score_session(
                session,
                client=client,
                model=model,
                dry_run=bool(args.dry_run),
                file_version=file_version,
            )
        )

    multi = len(reports) > 1
    payload: dict[str, Any] | list[dict[str, Any]]
    if multi:
        scored = [r for r in reports if isinstance(r.get("average_success_pct"), (int, float))]
        mean_pct = (
            round(sum(float(r["average_success_pct"]) for r in scored) / len(scored), 1) if scored else None
        )
        payload = {
            "file_version": file_version,
            "scoring_method": reports[0].get("scoring_method") if reports else None,
            "session_count": len(reports),
            "mean_average_success_pct": mean_pct,
            "sessions": reports,
        }
    else:
        payload = reports[0]

    if args.out is not None:
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.stderr)

    if args.json_out:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif multi:
        print(f"Scored {len(reports)} session(s) from {args.json} ({file_version} file snapshot)")
        for report in reports:
            print_human_summary(report, compact=True)
        mean = payload.get("mean_average_success_pct") if isinstance(payload, dict) else None
        if isinstance(mean, (int, float)):
            print(f"\nmean average success across sessions: {float(mean):.1f}%")
    else:
        print_human_summary(reports[0])

    return 1 if any(r.get("error") for r in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
