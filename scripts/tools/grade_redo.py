#!/usr/bin/env python3
"""
Grade redo session final artifacts against rubrics from ``verifiers.json``.

Matches rubrics by instruction overlap. Sends only the file(s) required by the **last**
workflow step (walking back one step if that step has no ``outputFiles``). By default uses
the **last** snapshot of each file; pass ``--eval-first`` to use the first snapshot instead.

Grades every session in the JSON by default. Pass ``--session-id`` to grade one session.

Examples:
  python scripts/tools/grade_redo.py -j runs/.../session.json --verifiers verifiers.json
  python scripts/tools/grade_redo.py -j sessions.json --dry-run --log-file grade_report.txt
  python scripts/tools/grade_redo.py -j sessions.json --session-id <uuid> --verifiers verifiers.json
  python scripts/tools/grade_redo.py -j session.json --verifiers verifiers.json \
      --backend openai --model gpt-4.1-mini --json-out ratings.json
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sys
from contextlib import contextmanager
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, TextIO

_tools = Path(__file__).resolve().parent
_scripts = _tools.parent
for p in (_scripts, _tools):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import extract_verifiers  # noqa: E402
from verifier_label_prompt import (  # noqa: E402
    format_numbered_lines,
    results_array_instructions,
    results_length_retry_hint,
    validate_results_length,
)

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]  # noqa: E402
except ImportError:  # pragma: no cover - optional for dry-run/OpenAI-only environments.
    load_dotenv = None  # type: ignore[assignment]

_PLAN_SUFFIX = "Before doing anything else, you MUST call the workflow_plan"
DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
_TRUNCATE_LEN = 14_000
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\n\r]+$")


@contextmanager
def tee_stdout(log_path: Path | None):
    if not log_path:
        yield
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_f:
        orig = sys.stdout

        class Tee(TextIO):
            def write(self, data: str) -> int:
                orig.write(data)
                log_f.write(data)
                return len(data)

            def flush(self) -> None:
                orig.flush()
                log_f.flush()

            def isatty(self) -> bool:
                return orig.isatty()

        sys.stdout = Tee()  # type: ignore[assignment]
        try:
            yield
        finally:
            sys.stdout = orig


def resolve_path(path: Path) -> Path:
    if path.exists():
        return path
    for base in (_scripts, Path.cwd()):
        cand = base / path
        if cand.exists():
            return cand
    return path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_sessions(sessions: list[dict[str, Any]], session_id: str | None) -> list[dict[str, Any]]:
    if not sessions:
        raise SystemExit("No sessions in JSON.")
    if session_id and str(session_id).strip():
        sid = str(session_id).strip()
        for session in sessions:
            if session.get("uuid") == sid:
                return [session]
        raise SystemExit(f"No session with uuid {sid!r}")
    return sessions


def load_catalog(path: Path) -> list[dict[str, Any]]:
    raw = load_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"{path.name}: expected a JSON array")
    return [x for x in raw if isinstance(x, dict)]


def session_instruction(session: dict[str, Any]) -> str:
    task = session.get("task")
    if isinstance(task, str) and task.strip():
        return task.strip()
    return extract_verifiers.session_instruction(session)


def instruction_overlap(a: str, b: str) -> float:
    a, b = a.strip(), b.strip()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    def core(s: str) -> str:
        i = s.find(_PLAN_SUFFIX)
        return s[:i].strip() if i >= 0 else s

    def normalize(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip())

    pairs = [
        (a, b),
        (core(a), core(b)),
        (normalize(a), normalize(b)),
        (normalize(core(a)), normalize(core(b))),
    ]
    return max(SequenceMatcher(None, left, right).ratio() for left, right in pairs if left and right)


def match_rubrics(catalog: list[dict[str, Any]], instruction: str, min_overlap: float) -> tuple[dict[str, Any], float]:
    scored = [
        (e, instruction_overlap(instruction, ins))
        for e in catalog
        if isinstance(ins := e.get("instruction"), str)
    ]
    if not scored:
        raise SystemExit("No verifier entries in catalog.")
    best, score = max(scored, key=lambda x: x[1])
    if score < min_overlap:
        raise SystemExit(f"No rubric match (best overlap {score:.3f} < {min_overlap}).")
    return best, score


def file_blocks_from_env(env: dict[str, Any]) -> list[tuple[str, str]]:
    ff = env.get("file") or env.get("files")
    out: list[tuple[str, str]] = []
    if isinstance(ff, dict):
        out = [(p, c) for p, c in ff.items() if isinstance(p, str) and isinstance(c, str) and c]
    elif isinstance(ff, list):
        for item in ff:
            if isinstance(item, dict):
                p, c = item.get("path"), item.get("content")
                if isinstance(p, str) and isinstance(c, str) and c:
                    out.append((p, c))
    return out


def _node_output_files(node: dict[str, Any]) -> list[str]:
    raw = node.get("outputFiles") or node.get("expected_output_files") or []
    if not isinstance(raw, list):
        return []
    return [str(f).strip() for f in raw if str(f).strip()]


def final_workflow_nodes(session: dict[str, Any]) -> list[dict[str, Any]]:
    """Top-level workflow nodes from the last snapshot, or planning ``workflow_tree_final``."""
    last_wf: list[Any] | None = None
    units = session.get("task_units")
    if isinstance(units, list):
        for unit in units:
            if not isinstance(unit, dict):
                continue
            env = unit.get("environment")
            if isinstance(env, dict):
                wf = env.get("workflow")
                if isinstance(wf, list) and wf:
                    last_wf = wf
        if not last_wf:
            planning = next((u for u in units if u.get("intent") == "planning"), None)
            if isinstance(planning, dict):
                wf = planning.get("workflow_tree_final")
                if isinstance(wf, list) and wf:
                    last_wf = wf
    else:
        for step in session.get("trajectory") or []:
            if isinstance(step, dict):
                env = step.get("environment")
                if isinstance(env, dict):
                    wf = env.get("workflow")
                    if isinstance(wf, list) and wf:
                        last_wf = wf
    return [n for n in (last_wf or []) if isinstance(n, dict)]


def required_paths_last_workflow_step(nodes: list[dict[str, Any]]) -> list[str]:
    """Output file paths for the last step; if that step has none, walk backward."""
    for node in reversed(nodes):
        paths = _node_output_files(node)
        if paths:
            return paths
    return []


def _basename(path: str) -> str:
    return Path(path.replace("\\", "/")).name


def all_artifact_blocks(session: dict[str, Any], *, eval_first: bool = False) -> dict[str, str]:
    """Content per path across environment snapshots (first or last version per path)."""
    merged: dict[str, str] = {}
    units = session.get("task_units")
    if isinstance(units, list):
        sources = (u.get("environment") for u in units if isinstance(u, dict))
    else:
        sources = (s.get("environment") for s in (session.get("trajectory") or []) if isinstance(s, dict))
    for env in sources:
        if isinstance(env, dict):
            for path, content in file_blocks_from_env(env):
                if eval_first:
                    merged.setdefault(path, content)
                else:
                    merged[path] = content
    return merged


def grading_artifact_blocks(
    session: dict[str, Any], *, eval_first: bool = False
) -> tuple[list[tuple[str, str]], list[str], str | None]:
    """
    File blocks for the LM grader: only artifacts required by the last workflow step
    (walking back if the last step lists no output files).
    """
    nodes = final_workflow_nodes(session)
    required = required_paths_last_workflow_step(nodes)
    if not required:
        return [], [], "Could not determine output files for the last workflow step."

    required_names = {_basename(p) for p in required}
    merged = all_artifact_blocks(session, eval_first=eval_first)
    blocks = sorted((p, c) for p, c in merged.items() if _basename(p) in required_names)
    if not blocks:
        available = ", ".join(sorted(_basename(p) for p in merged)) or "(none)"
        need = ", ".join(required)
        return [], required, f"No content for last-step file(s) [{need}] in session snapshots (available: {available})."
    return blocks, required, None


def criteria_list(entry: dict[str, Any]) -> list[str]:
    raw = entry.get("verifiers")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for v in raw:
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
        elif isinstance(v, dict) and isinstance(v.get("criterion"), str) and v["criterion"].strip():
            out.append(v["criterion"].strip())
    return out


def label_tag(passed: bool | None) -> str:
    return "PASS" if passed is True else "FAIL" if passed is False else "UNKNOWN"


def _truncate_text(text: str, max_len: int = _TRUNCATE_LEN) -> str:
    return text if len(text) <= max_len else text[:max_len] + "\n... [truncated]"


def _looks_like_base64(s: str, *, min_len: int = 256) -> bool:
    t = s.strip()
    return len(t) >= min_len and _BASE64_RE.fullmatch(t) is not None


def _extract_image_data(path: str, text: str) -> tuple[str, str] | None:
    media_type, _ = mimetypes.guess_type(path)
    if not media_type or not media_type.startswith("image/"):
        return None
    raw = text.strip()
    if _looks_like_base64(raw):
        return media_type, raw
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        c = parsed.get("content")
        if isinstance(c, str) and _looks_like_base64(c):
            return media_type, c.strip()
    return None


def build_message_content(criteria: list[str], file_blocks: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Anthropic message blocks for verifier labeling (matches app verifier-labeler contract)."""
    n = len(criteria)
    numbered = format_numbered_lines(criteria)
    header = "\n".join(
        [
            "You are an automated checker for completed task output files.",
            "Given verifier criteria and current output files, decide whether each criterion is satisfied.",
            'Reply with ONLY a JSON object of this exact shape: {"results":[{"pass":true},{"pass":false},...]}',
            results_array_instructions(count=n, item_word="criterion"),
            "",
            "Verifier criteria (in order):",
            numbered,
            "",
            "Output files and contents:",
        ]
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": header}]
    if not file_blocks:
        content.append({"type": "text", "text": "(no output files present)"})
        return content
    for rel, text in file_blocks:
        content.append({"type": "text", "text": f"### {rel}"})
        img = _extract_image_data(rel, text)
        if img is not None:
            mt, data = img
            content.append({"type": "image", "source": {"type": "base64", "media_type": mt, "data": data}})
            content.append({"type": "text", "text": "(image content attached above)"})
        else:
            content.append({"type": "text", "text": _truncate_text(text)})
        content.append({"type": "text", "text": "---"})
    return content


def _parse_json_from_model_text(text: str) -> Any:
    def scan_candidates(source: str) -> list[str]:
        candidates: list[str] = []
        start: int | None = None
        opener: str | None = None
        depth = 0
        in_string = False
        escape = False
        for i, ch in enumerate(source):
            if start is None:
                if ch in "{[":
                    start = i
                    opener = ch
                    depth = 1
                    in_string = False
                    escape = False
                continue

            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif opener == "{" and ch == "{":
                depth += 1
            elif opener == "[" and ch == "[":
                depth += 1
            elif (opener == "{" and ch == "}") or (opener == "[" and ch == "]"):
                depth -= 1
                if depth == 0:
                    candidates.append(source[start : i + 1])
                    start = None
                    opener = None
        return candidates

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    sources: list[str] = []
    if fence and fence.group(1):
        sources.append(fence.group(1).strip())
    sources.append(text.strip())

    last_error: Exception | None = None
    seen: set[str] = set()
    for source in sources:
        for candidate in reversed(scan_candidates(source)):
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            if isinstance(parsed, (dict, list)):
                return parsed

    if last_error is not None:
        raise ValueError(f"Unable to parse JSON object in model response: {last_error}") from last_error
    preview = text.strip()
    if len(preview) > 400:
        preview = preview[:400] + "... [truncated]"
    raise ValueError(f"No JSON object found in model response. Raw preview: {preview!r}")


def interpret_results(text: str, n: int) -> list[bool | None]:
    parsed = _parse_json_from_model_text(text)
    if isinstance(parsed, dict):
        arr = parsed.get("results")
    elif isinstance(parsed, list):
        arr = parsed
    else:
        arr = None
    if not isinstance(arr, list):
        raise ValueError("Missing results array")
    validate_results_length(arr, n)
    out: list[bool | None] = [None] * n
    for i, row in enumerate(arr):
        if isinstance(row, bool):
            out[i] = row
        elif isinstance(row, dict) and "pass" in row:
            value = row["pass"]
            if isinstance(value, bool):
                out[i] = value
            elif isinstance(value, str) and value.strip().lower() in {"true", "false"}:
                out[i] = value.strip().lower() == "true"
    return out


def grade(
    criteria: list[str],
    files: list[tuple[str, str]],
    *,
    client: Any,
    model: str,
    max_tokens: int,
) -> tuple[list[bool | None], str]:
    content = build_message_content(criteria, files)
    n = len(criteria)

    def fetch(extra: str = "") -> str:
        blocks = list(content)
        if extra:
            blocks.append({"type": "text", "text": extra})
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[{"role": "user", "content": blocks}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")

    raw = fetch()
    try:
        return interpret_results(raw, n), raw
    except ValueError as exc:
        if "Expected exactly" not in str(exc):
            raise
        raw = fetch(results_length_retry_hint(n))
        return interpret_results(raw, n), raw


def print_summary(
    *,
    session: dict[str, Any],
    entry: dict[str, Any],
    overlap: float,
    files: list[tuple[str, str]],
    criteria_count: int,
    model: str | None = None,
) -> None:
    name = session.get("name") or session.get("uuid")
    print(f"session: {name}")
    print(f"matched rubrics: {entry.get('uuid')} (overlap {overlap:.3f})")
    if model:
        print(f"model: {model}")
    print(f"criteria: {criteria_count}")
    print(f"artifacts: {', '.join(p for p, _ in files) or '(none)'}")


def build_openai_input(criteria: list[str], file_blocks: list[tuple[str, str]]) -> list[dict[str, Any]]:
    anthropic_blocks = build_message_content(criteria, file_blocks)
    content: list[dict[str, Any]] = []
    for block in anthropic_blocks:
        if block.get("type") == "text":
            content.append({"type": "input_text", "text": str(block.get("text") or "")})
        elif block.get("type") == "image":
            src = block.get("source")
            if not isinstance(src, dict):
                continue
            media_type = str(src.get("media_type") or "image/png")
            data = str(src.get("data") or "").strip()
            if data:
                content.append({"type": "input_image", "image_url": f"data:{media_type};base64,{data}"})
    return [{"role": "user", "content": content}]


def grade_openai(
    criteria: list[str],
    files: list[tuple[str, str]],
    *,
    model: str,
    api_key: str | None,
    base_url: str | None,
    request_timeout: float,
    max_retries: int,
    max_tokens: int,
    debug_prompts: bool,
) -> tuple[list[bool | None], str]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("OpenAI backend requires the openai package.") from exc

    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_api_key:
        raise SystemExit("OpenAI backend requires OPENAI_API_KEY or --api-key.")

    client = OpenAI(
        api_key=resolved_api_key,
        base_url=base_url,
        timeout=request_timeout,
        max_retries=max_retries,
    )
    input_payload = build_openai_input(criteria, files)
    if debug_prompts:
        print("=== openai verifier request blocks ===", file=sys.stderr)
        for block in input_payload[0]["content"]:
            if block["type"] == "input_text":
                print(f"text:\n{block['text']}\n", file=sys.stderr)
            elif block["type"] == "input_image":
                url = str(block["image_url"])
                print(f"image_url len={len(url)} preview={url[:72]}...", file=sys.stderr)
        print("=== end request blocks ===", file=sys.stderr)

    n = len(criteria)
    base_content = list(input_payload[0]["content"])

    def fetch(extra: str = "") -> str:
        content = list(base_content)
        if extra:
            content.append({"type": "input_text", "text": extra})
        response = client.responses.create(
            model=model,
            temperature=0.0,
            max_output_tokens=max_tokens,
            input=[{"role": "user", "content": content}],
        )
        raw = getattr(response, "output_text", None)
        return raw if isinstance(raw, str) else str(response)

    raw = fetch()
    try:
        return interpret_results(raw, n), raw
    except ValueError as exc:
        if "Expected exactly" not in str(exc):
            raise
        raw = fetch(results_length_retry_hint(n))
        return interpret_results(raw, n), raw


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-j", "--session-json", type=Path, required=True)
    p.add_argument(
        "--session-id",
        help="Grade only the session with this uuid (default: grade all sessions in the JSON)",
    )
    p.add_argument("--verifiers", type=Path, default=Path("verifiers.json"))
    p.add_argument("--min-overlap", type=float, default=0.55)
    p.add_argument("--backend", choices=["anthropic", "openai"], default="anthropic")
    p.add_argument("--model", help=f"Judge model (default: {DEFAULT_MODEL} for Anthropic, {DEFAULT_OPENAI_MODEL} for OpenAI)")
    p.add_argument("--api-key", help="OpenAI backend API key override (default: OPENAI_API_KEY)")
    p.add_argument("--base-url", help="OpenAI-compatible base URL override")
    p.add_argument("--request-timeout", type=float, default=120.0)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--no-api-config", action="store_true")
    p.add_argument("--no-claude-settings", action="store_true")
    p.add_argument("--env-file", type=Path)
    p.add_argument("--dotenv-override", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--eval-first",
        action="store_true",
        help="Grade the first file version per path instead of the last (default: last)",
    )
    p.add_argument("--json-out", type=Path)
    p.add_argument("--log-file", type=Path, help="Also write stdout to this .txt file")
    p.add_argument("--debug-prompts", action="store_true")
    args = p.parse_args(argv)

    env_file = args.env_file or (_scripts / ".env")
    if load_dotenv is not None and env_file.is_file():
        load_dotenv(env_file, override=args.dotenv_override)

    session_path = resolve_path(args.session_json)
    verifiers_path = resolve_path(args.verifiers)
    sessions = resolve_sessions(extract_verifiers.load_sessions_from_path(session_path), args.session_id)
    catalog = load_catalog(verifiers_path)

    anthropic_client: Any | None = None
    if not args.dry_run and args.backend == "anthropic":
        import induce  # noqa: PLC0415

        cfg = induce.resolve_anthropic_config(
            skip_api_config=args.no_api_config,
            skip_claude_settings=args.no_claude_settings,
        )
        anthropic_client = induce.make_anthropic_client(cfg)

    reports: list[dict[str, Any]] = []
    with tee_stdout(args.log_file):
        for i, session in enumerate(sessions):
            if i:
                print("\n" + "=" * 72 + "\n")

            instruction = session_instruction(session)
            if not instruction:
                raise SystemExit("No initial task instruction in session JSON.")

            entry, overlap = match_rubrics(catalog, instruction, args.min_overlap)
            criteria = criteria_list(entry)
            files, required_paths, artifact_issue = grading_artifact_blocks(session, eval_first=args.eval_first)
            if not criteria:
                raise SystemExit("Matched rubric entry has no criteria.")

            last_step = (final_workflow_nodes(session) or [None])[-1]
            last_step_desc = ""
            if isinstance(last_step, dict):
                last_step_desc = str(last_step.get("description") or "").strip()

            report: dict[str, Any] = {
                "session_uuid": session.get("uuid"),
                "session_name": session.get("name"),
                "matched_verifier_uuid": entry.get("uuid"),
                "instruction_overlap": round(overlap, 4),
                "last_workflow_step": last_step_desc,
                "required_output_files": required_paths,
                "grading_files": [path for path, _ in files],
                "verifiers": criteria,
            }
            if artifact_issue:
                report["artifact_issue"] = artifact_issue

            if args.dry_run:
                print_summary(
                    session=session, entry=entry, overlap=overlap, files=files, criteria_count=len(criteria)
                )
                if artifact_issue:
                    print(f"artifact_issue: {artifact_issue}")
            else:
                if artifact_issue:
                    labels = [False] * len(criteria)
                    raw_response = f"Skipped verifier model call: {artifact_issue}"
                    model = args.model or (DEFAULT_OPENAI_MODEL if args.backend == "openai" else DEFAULT_MODEL)
                elif args.backend == "openai":
                    model = args.model or DEFAULT_OPENAI_MODEL
                    labels, raw_response = grade_openai(
                        criteria,
                        files,
                        model=model,
                        api_key=args.api_key,
                        base_url=args.base_url,
                        request_timeout=args.request_timeout,
                        max_retries=args.max_retries,
                        max_tokens=args.max_tokens,
                        debug_prompts=args.debug_prompts,
                    )
                else:
                    model = args.model or DEFAULT_MODEL

                    if args.debug_prompts:
                        for j, block in enumerate(build_message_content(criteria, files)):
                            print(f"[{j}] {block.get('type')}", file=sys.stderr)

                    labels, raw_response = grade(
                        criteria, files, client=anthropic_client, model=model, max_tokens=args.max_tokens
                    )
                results = [{"index": j, "criterion": c, "pass": labels[j]} for j, c in enumerate(criteria)]

                n_pass = sum(1 for x in labels if x is True)
                n_fail = sum(1 for x in labels if x is False)
                n_unknown = len(labels) - n_pass - n_fail
                total = len(criteria)
                rate = n_pass / total if total else 0.0

                for j, (c, lab) in enumerate(zip(criteria, labels)):
                    print(f"[{j:02d}] {label_tag(lab)} - {c}")

                print()
                print_summary(
                    session=session,
                    entry=entry,
                    overlap=overlap,
                    files=files,
                    criteria_count=total,
                    model=model,
                )
                if artifact_issue:
                    print(f"artifact_issue: {artifact_issue}")
                print(f"pass/fail/unknown: {n_pass}/{n_fail}/{n_unknown} of {total}")
                print(f"average_success_rate: {n_pass}/{total} = {rate:.4f}")
                if n_unknown and (n_pass + n_fail):
                    print(
                        f"average_success_rate (scored only): {n_pass}/{n_pass + n_fail} = {n_pass / (n_pass + n_fail):.4f}"
                    )

                report.update(
                    {
                        "backend": args.backend,
                        "model": model,
                        "raw_response": raw_response,
                        "results": results,
                        "n_pass": n_pass,
                        "n_fail": n_fail,
                        "n_unknown": n_unknown,
                        "average_success_rate": round(rate, 4),
                    }
                )

            reports.append(report)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload: Any = reports[0] if len(reports) == 1 else reports
        args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote JSON to {args.json_out}", file=sys.stderr)

    if args.log_file:
        print(f"Wrote log to {args.log_file.resolve()}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
