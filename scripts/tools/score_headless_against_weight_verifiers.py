#!/usr/bin/env python3
"""Score one headless session against task-level verifiers from out_weight.json."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sys
from pathlib import Path
from typing import Any

_tools_dir = Path(__file__).resolve().parent
_scripts_dir = _tools_dir.parent
for _path in (_tools_dir, _scripts_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import induce  # noqa: E402
from dotenv import load_dotenv  # noqa: E402


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
DEFAULT_TINKER_MODEL = "Qwen/Qwen3.5-35B-A3B"

_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\n\r]+$")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_headless_session(path: Path) -> dict[str, Any]:
    raw = _load_json(path)
    if not isinstance(raw, dict) or not isinstance(raw.get("trajectory"), list):
        raise SystemExit(f"Expected trajectory-style session JSON in {path}")
    return raw


def _load_weight_sessions(path: Path) -> list[dict[str, Any]]:
    raw = _load_json(path)
    if not isinstance(raw, list):
        raise SystemExit(f"Expected weight-format JSON array in {path}")
    return [item for item in raw if isinstance(item, dict)]


def _task_key(value: str) -> str:
    return str(value).strip().lower()


def _canonical_task_name(value: str) -> str:
    raw = _task_key(value)
    match = re.fullmatch(r"(?:task[_-]?)?0*([1-9]\d*|0)", raw)
    if match:
        return f"task{int(match.group(1))}"
    return raw


def _normalize_instruction(value: str) -> str:
    text = " ".join(str(value).split())
    return text.strip().lower()


def _find_weight_session(
    sessions: list[dict[str, Any]],
    *,
    task_id: str,
    instruction: str | None,
) -> dict[str, Any]:
    want_name = _canonical_task_name(task_id)
    for session in sessions:
        name = str(session.get("name") or "")
        session_key = _canonical_task_name(name.split(";", 1)[0])
        if session_key == want_name:
            return session

    if instruction:
        want_instruction = _normalize_instruction(instruction)
        for session in sessions:
            candidate = session.get("initial_task_instruction")
            if isinstance(candidate, str) and _normalize_instruction(candidate) == want_instruction:
                return session

    raise SystemExit(f'No weight-format session matched task "{task_id}"')


def _criterion_strings(verifiers: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(verifiers, list):
        return out
    for item in verifiers:
        if not isinstance(item, dict):
            continue
        criterion = item.get("criterion")
        if isinstance(criterion, str) and criterion.strip():
            out.append(criterion.strip())
    return out


def _select_final_task_unit(weight_session: dict[str, Any]) -> tuple[int, dict[str, Any], list[str]]:
    task_units = weight_session.get("task_units")
    if not isinstance(task_units, list):
        raise SystemExit("Weight-format session has no task_units array")

    best_index = -1
    best_unit: dict[str, Any] | None = None
    best_criteria: list[str] = []
    for index, unit in enumerate(task_units):
        if not isinstance(unit, dict):
            continue
        criteria = _criterion_strings(unit.get("verifiers"))
        if criteria:
            best_index = index
            best_unit = unit
            best_criteria = criteria

    if best_unit is None:
        raise SystemExit("No non-empty verifiers found in weight-format task_units")
    return best_index, best_unit, best_criteria


def _final_agent_file_blocks(session: dict[str, Any]) -> tuple[int, list[tuple[str, str]]]:
    traj = session.get("trajectory")
    if not isinstance(traj, list):
        return -1, []
    for step_index in range(len(traj) - 1, -1, -1):
        step = traj[step_index]
        if not isinstance(step, dict) or step.get("actor") != "agent":
            continue
        env = step.get("environment")
        if not isinstance(env, dict):
            continue
        file_field = env.get("file")
        if isinstance(file_field, dict):
            blocks = [(path, content) for path, content in file_field.items() if isinstance(path, str) and isinstance(content, str)]
            if blocks:
                blocks.sort(key=lambda item: item[0])
                return step_index, blocks
        if isinstance(file_field, list):
            blocks: list[tuple[str, str]] = []
            for item in file_field:
                if not isinstance(item, dict):
                    continue
                path = item.get("path")
                content = item.get("content")
                if isinstance(path, str) and isinstance(content, str):
                    blocks.append((path, content))
            if blocks:
                blocks.sort(key=lambda item: item[0])
                return step_index, blocks
    return -1, []


def _truncate_text(text: str, max_len: int = 14_000) -> str:
    return text if len(text) <= max_len else text[:max_len] + "\n... [truncated]"


def _looks_like_base64_data(value: str, *, min_len: int = 256) -> bool:
    text = value.strip()
    if len(text) < min_len:
        return False
    return _BASE64_RE.fullmatch(text) is not None


def _guess_image_media_type(path: str) -> str | None:
    media_type, _ = mimetypes.guess_type(path)
    if media_type and media_type.startswith("image/"):
        return media_type
    return None


def _extract_image_data(rel: str, text: str) -> tuple[str, str] | None:
    media_type = _guess_image_media_type(rel)
    if media_type is None:
        return None
    raw = text.strip()
    if raw.startswith("data:image/"):
        prefix, _, data = raw.partition(",")
        if prefix and data:
            return media_type, data
        return None
    if _looks_like_base64_data(raw):
        return media_type, re.sub(r"\s+", "", raw)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        content = parsed.get("content")
        if isinstance(content, str) and _looks_like_base64_data(content):
            return media_type, re.sub(r"\s+", "", content)
    return None


def _build_prompt_header(criteria: list[str]) -> str:
    numbered = "\n".join(f"{idx}. {criterion}" for idx, criterion in enumerate(criteria))
    return "\n".join(
        [
            "You are an automated checker for completed task output files.",
            "Given verifier criteria and current output files, decide whether each criterion is satisfied.",
            'Reply with ONLY a JSON object of this exact shape: {"results":[{"pass":true},{"pass":false},...]}',
            "The results array must have exactly one object per verifier line, in the same order.",
            "The current output filenames may differ from the filenames mentioned in the verifier criteria.",
            "Judge primarily by artifact content and task intent, not by exact filename, unless a criterion explicitly depends on the filename itself.",
            "",
            "Verifier criteria (in order):",
            numbered,
            "",
            "Output files and contents:",
        ]
    )


def _build_anthropic_message_content(criteria: list[str], file_blocks: list[tuple[str, str]]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": _build_prompt_header(criteria)}]
    if not file_blocks:
        content.append({"type": "text", "text": "(no output files present)"})
        return content
    for rel, text in file_blocks:
        content.append({"type": "text", "text": f"### {rel}"})
        image = _extract_image_data(rel, text)
        if image is not None:
            media_type, data = image
            content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}})
            content.append({"type": "text", "text": "(image content attached above)"})
        else:
            content.append({"type": "text", "text": _truncate_text(text)})
        content.append({"type": "text", "text": "---"})
    return content


def _build_openai_input(criteria: list[str], file_blocks: list[tuple[str, str]]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": _build_prompt_header(criteria)}]
    if not file_blocks:
        content.append({"type": "input_text", "text": "(no output files present)"})
        return [{"role": "user", "content": content}]
    for rel, text in file_blocks:
        content.append({"type": "input_text", "text": f"### {rel}"})
        image = _extract_image_data(rel, text)
        if image is not None:
            media_type, data = image
            content.append({"type": "input_image", "image_url": f"data:{media_type};base64,{data}"})
            content.append({"type": "input_text", "text": "(image content attached above)"})
        else:
            content.append({"type": "input_text", "text": _truncate_text(text)})
        content.append({"type": "input_text", "text": "---"})
    return [{"role": "user", "content": content}]


def _build_text_prompt(criteria: list[str], file_blocks: list[tuple[str, str]]) -> str:
    rendered_blocks: list[str] = []
    if not file_blocks:
        rendered_blocks.append("(no output files present)")
    for rel, text in file_blocks:
        image = _extract_image_data(rel, text)
        if image is not None:
            rendered_blocks.append(f"### {rel}\n\n(image artifact omitted from text-only backend)")
        else:
            rendered_blocks.append(f"### {rel}\n\n{_truncate_text(text)}")
    return _build_prompt_header(criteria) + "\n" + "\n\n".join(rendered_blocks)


def _parse_json_from_model_text(text: str) -> dict[str, Any]:
    def iter_candidates(source: str) -> list[str]:
        candidates: list[str] = []
        start: int | None = None
        depth = 0
        in_string = False
        escape = False
        for idx, ch in enumerate(source):
            if start is None:
                if ch == "{":
                    start = idx
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
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(source[start : idx + 1])
                    start = None
        return candidates

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    sources = [text.strip()]
    if fence:
        sources.insert(0, fence.group(1).strip())
    for source in sources:
        for candidate in reversed(iter_candidates(source)):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    raise ValueError("No JSON object found in model response")


def _interpret_results(text: str, count: int) -> list[bool | None]:
    parsed = _parse_json_from_model_text(text)
    rows = parsed.get("results")
    if not isinstance(rows, list):
        raise ValueError("Missing results array")
    output: list[bool | None] = [None] * count
    for index in range(min(count, len(rows))):
        row = rows[index]
        if isinstance(row, dict) and "pass" in row:
            output[index] = bool(row["pass"])
    return output


def _average_success_pct(labels: list[bool | None], total: int) -> float | None:
    if total <= 0:
        return None
    return round(100.0 * sum(1 for item in labels if item is True) / total, 1)


def _load_dotenvs(env_file: Path | None, override: bool) -> None:
    if env_file is not None:
        load_dotenv(dotenv_path=env_file, override=override)
        return
    load_dotenv(dotenv_path=_scripts_dir / ".env", override=override)
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=override)


def _make_openai_client(*, api_key: str, base_url: str, timeout: float, max_retries: int) -> Any:
    try:
        from openai import OpenAI
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("OpenAI-compatible eval requires the openai package.") from exc
    return OpenAI(api_key=api_key, base_url=base_url.rstrip("/"), timeout=timeout, max_retries=max_retries)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a headless session with task-level verifiers from out_weight.json.")
    parser.add_argument("--verifiers-json", type=Path, default=Path("out_weight.json"))
    parser.add_argument("--session-json", type=Path, required=True)
    parser.add_argument("--task-id", required=True, help='Task id, e.g. "1"')
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--backend", choices=["anthropic", "openai", "tinker"], default="openai")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--dotenv-override", action="store_true")
    parser.add_argument("--no-api-config", action="store_true")
    parser.add_argument("--no-claude-settings", action="store_true")
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--debug-prompts", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.out and args.out.exists() and not args.force:
        print(f"Using existing ratings file (no LLM): {args.out.resolve()}", file=sys.stderr)
        return 0

    _load_dotenvs(args.env_file, args.dotenv_override)

    headless_session = _load_headless_session(args.session_json)
    verifier_sessions = _load_weight_sessions(args.verifiers_json)
    matched_weight_session = _find_weight_session(
        verifier_sessions,
        task_id=str(args.task_id),
        instruction=headless_session.get("task") if isinstance(headless_session.get("task"), str) else None,
    )
    task_unit_index, task_unit, criteria = _select_final_task_unit(matched_weight_session)
    trajectory_step_index, file_blocks = _final_agent_file_blocks(headless_session)
    if not file_blocks:
        raise SystemExit("No final agent file outputs found in headless session.")

    raw_text: str
    auth_resolved: dict[str, Any]
    if args.backend == "anthropic":
        cfg = induce.resolve_anthropic_config(
            skip_api_config=bool(args.no_api_config),
            skip_claude_settings=bool(args.no_claude_settings),
        )
        model = args.model or cfg.model
        if args.api_key:
            cfg = induce.ResolvedAnthropicConfig(
                api_key=args.api_key.strip(),
                base_url=(args.base_url.strip().rstrip("/") if args.base_url else cfg.base_url),
                model=model,
            )
        elif args.base_url or args.model:
            cfg = induce.ResolvedAnthropicConfig(
                api_key=cfg.api_key,
                base_url=(args.base_url.strip().rstrip("/") if args.base_url else cfg.base_url),
                model=model,
            )
        client = induce.make_anthropic_client(cfg)
        content = _build_anthropic_message_content(criteria, file_blocks)
        if args.debug_prompts:
            print(json.dumps(content, ensure_ascii=False)[:12000], file=sys.stderr)
        msg = client.messages.create(
            model=cfg.model,
            max_tokens=int(args.max_tokens),
            temperature=0.0,
            messages=[{"role": "user", "content": content}],
        )
        raw_text = "".join(block.text for block in msg.content if getattr(block, "type", None) == "text")
        auth_resolved = {
            "backend": "anthropic",
            "model": cfg.model,
            "base_url_effective": cfg.base_url or "(Anthropic SDK default)",
            "llm": "Anthropic Messages API",
        }
    else:
        if args.backend == "openai":
            api_key = (args.api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
            base_url = (args.base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL).strip().rstrip("/")
            model = (args.model or os.environ.get("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL).strip()
            if not api_key:
                raise SystemExit("Missing OPENAI_API_KEY. Set it in scripts/.env or pass --api-key.")
            client = _make_openai_client(
                api_key=api_key,
                base_url=base_url,
                timeout=args.request_timeout,
                max_retries=max(0, args.max_retries),
            )
            request_input = _build_openai_input(criteria, file_blocks)
            if args.debug_prompts:
                print(json.dumps(request_input, ensure_ascii=False)[:12000], file=sys.stderr)
            response = client.responses.create(
                model=model,
                input=request_input,
                max_output_tokens=int(args.max_tokens),
            )
            raw_text = getattr(response, "output_text", None) or ""
            auth_resolved = {
                "backend": "openai",
                "model": model,
                "base_url_effective": base_url,
                "llm": "OpenAI Responses API with image inputs when available",
            }
        else:
            api_key = (args.api_key or os.environ.get("TINKER_API_KEY") or "").strip()
            base_url = (args.base_url or os.environ.get("TINKER_BASE_URL") or DEFAULT_TINKER_BASE_URL).strip().rstrip("/")
            model = (args.model or os.environ.get("TINKER_MODEL") or DEFAULT_TINKER_MODEL).strip()
            if not api_key:
                raise SystemExit("Missing TINKER_API_KEY. Set it in scripts/.env or pass --api-key.")
            client = _make_openai_client(
                api_key=api_key,
                base_url=base_url,
                timeout=args.request_timeout,
                max_retries=max(0, args.max_retries),
            )
            prompt = _build_text_prompt(criteria, file_blocks)
            if args.debug_prompts:
                print(prompt[:12000], file=sys.stderr)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=int(args.max_tokens),
            )
            choice = response.choices[0] if response.choices else None
            raw_text = getattr(getattr(choice, "message", None), "content", None) or ""
            auth_resolved = {
                "backend": "tinker",
                "model": model,
                "base_url_effective": base_url,
                "llm": "OpenAI-compatible chat completions",
            }

    if not raw_text.strip():
        raise SystemExit("Verifier model returned no text content.")

    labels = _interpret_results(raw_text, len(criteria))
    average_success_pct = _average_success_pct(labels, len(criteria))
    output_files = [path for path, _ in file_blocks]
    image_artifacts = [path for path, text in file_blocks if _extract_image_data(path, text) is not None]

    report = {
        "uuid": headless_session.get("uuid"),
        "name": headless_session.get("name"),
        "dry_run": False,
        "task_id": args.task_id,
        "verifier_source": {
            "uuid": matched_weight_session.get("uuid"),
            "name": matched_weight_session.get("name"),
            "task_unit_index": task_unit_index,
            "intent": task_unit.get("intent"),
        },
        "tasks": [
            {
                "node_id": f"task{args.task_id}:weight_final",
                "description": str(task_unit.get("intent") or f"task{args.task_id} final verifiers"),
                "output_files": output_files,
                "final_rubrics": criteria,
                "unique_agent_snapshots": 1,
                "versions": [
                    {
                        "version_index": 0,
                        "trajectory_step_index": trajectory_step_index,
                        "eval_cache_key_prefix": None,
                        "eval_cache_hit": False,
                        "unique_eval_sequence": 1,
                        "first_seen_trajectory_step_index": trajectory_step_index,
                        "attached_image_artifacts": image_artifacts,
                        "lm": {
                            "raw_text": raw_text,
                            "pass_per_criterion": labels,
                            "criteria": criteria,
                        },
                        "error": None,
                        "raw_text": raw_text,
                        "average_success_pct": average_success_pct,
                    }
                ],
            }
        ],
        "scatter_plot_data": [
            {
                "trajectory_step_index": trajectory_step_index,
                "average_success_pct": average_success_pct,
                "node_id": f"task{args.task_id}:weight_final",
                "task_description": str(task_unit.get("intent") or ""),
                "version_index_within_task": 0,
                "unique_eval_sequence": 1,
                "first_seen_trajectory_step_index": trajectory_step_index,
                "eval_cache_hit": False,
            }
        ] if average_success_pct is not None else [],
        "eval_cache_stats": {
            "dedupe_hits": 0,
            "unique_agent_solutions_evaluated": 1,
            "llm_calls": 1,
            "cached_keys": 1,
            "unique_eval_sequences": 1,
        },
        "auth_resolved": auth_resolved,
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
