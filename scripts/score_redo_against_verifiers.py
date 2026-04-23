#!/usr/bin/env python3
"""Score final outputs in one export against final verifiers from another."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
from pathlib import Path
from typing import Any

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

import induce  # noqa: E402


def _load_sessions(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        if isinstance(raw.get("sessions"), list):
            return [x for x in raw["sessions"] if isinstance(x, dict)]
        if isinstance(raw.get("trajectory"), list):
            return [raw]
    return []


def _task_key(name: str) -> str:
    # e.g. "task2; before; qwen..." / "task2; after; qwen..." -> "task2"
    left = name.split(";", 1)[0].strip().lower()
    return left


def _top_level_workflow(step: dict[str, Any]) -> list[dict[str, Any]]:
    env = step.get("environment") if isinstance(step, dict) else None
    wf = env.get("workflow") if isinstance(env, dict) else None
    if not isinstance(wf, list):
        return []
    return [n for n in wf if isinstance(n, dict)]


def _final_verifiers_from_session(session: dict[str, Any]) -> list[str]:
    traj = session.get("trajectory")
    if not isinstance(traj, list):
        return []
    for step in reversed(traj):
        wf = _top_level_workflow(step)
        if not wf:
            continue
        out: list[str] = []
        for node in wf:
            for v in node.get("verifiers") or []:
                if isinstance(v, dict):
                    c = v.get("criterion")
                    if isinstance(c, str) and c.strip():
                        out.append(c)
        if out:
            return out
    return []


def _final_agent_file_blocks(session: dict[str, Any]) -> list[tuple[str, str]]:
    traj = session.get("trajectory")
    if not isinstance(traj, list):
        return []
    for step in reversed(traj):
        if not isinstance(step, dict) or step.get("actor") != "agent":
            continue
        env = step.get("environment")
        if not isinstance(env, dict):
            continue
        ff = env.get("file")
        if isinstance(ff, dict):
            blocks = [(k, v) for k, v in ff.items() if isinstance(k, str) and isinstance(v, str)]
            if blocks:
                blocks.sort(key=lambda x: x[0])
                return blocks
        if isinstance(ff, list):
            blocks: list[tuple[str, str]] = []
            for item in ff:
                if not isinstance(item, dict):
                    continue
                p = item.get("path")
                c = item.get("content")
                if isinstance(p, str) and isinstance(c, str):
                    blocks.append((p, c))
            if blocks:
                blocks.sort(key=lambda x: x[0])
                return blocks
    return []


def _truncate_text(text: str, max_len: int = 14_000) -> str:
    return text if len(text) <= max_len else text[:max_len] + "\n... [truncated]"


_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\n\r]+$")


def _looks_like_base64_data(s: str, *, min_len: int = 256) -> bool:
    t = s.strip()
    if len(t) < min_len:
        return False
    return _BASE64_RE.fullmatch(t) is not None


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
    if _looks_like_base64_data(raw):
        return media_type, raw
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        c = parsed.get("content")
        if isinstance(c, str) and _looks_like_base64_data(c):
            return media_type, c.strip()
    return None


def _build_prompt_header(criteria: list[str]) -> str:
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(criteria))
    return "\n".join(
        [
            "You are an automated checker for completed task output files.",
            "Given verifier criteria and current output files, decide whether each criterion is satisfied.",
            'Reply with ONLY a JSON object of this exact shape: {"results":[{"pass":true},{"pass":false},...]}',
            "The results array must have exactly one object per verifier line, in the same order (indices 0 .. n-1).",
            "",
            "Verifier criteria (in order):",
            numbered,
            "",
            "Output files and contents:",
        ]
    )


def _build_message_content(criteria: list[str], file_blocks: list[tuple[str, str]]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": _build_prompt_header(criteria)}]
    if not file_blocks:
        content.append({"type": "text", "text": "(no output files present)"})
        return content
    for rel, text in file_blocks:
        content.append({"type": "text", "text": f"### {rel}"})
        img = _extract_image_data(rel, text)
        if img is not None:
            mt, data = img
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": mt, "data": data},
                }
            )
            content.append({"type": "text", "text": "(image content attached above)"})
        else:
            content.append({"type": "text", "text": _truncate_text(text)})
        content.append({"type": "text", "text": "---"})
    return content


def _parse_json_from_model_text(text: str) -> dict[str, Any]:
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    raw = (fence.group(1) if fence else text).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response")
    return json.loads(raw[start : end + 1])


def _interpret_results(text: str, n: int) -> list[bool | None]:
    parsed = _parse_json_from_model_text(text)
    arr = parsed.get("results")
    if not isinstance(arr, list):
        raise ValueError("Missing results array")
    out: list[bool | None] = [None] * n
    for i in range(min(n, len(arr))):
        row = arr[i]
        if isinstance(row, dict) and "pass" in row:
            out[i] = bool(row["pass"])
    return out


def _find_session_by_task(sessions: list[dict[str, Any]], task: str) -> dict[str, Any] | None:
    key = _task_key(task)
    for s in sessions:
        n = s.get("name")
        if isinstance(n, str) and _task_key(n) == key:
            return s
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="Score redo outputs with verifiers from baseline export.")
    p.add_argument("--verifiers-json", type=Path, default=Path("out.json"))
    p.add_argument("--outputs-json", type=Path, default=Path("out_redo.json"))
    p.add_argument("--task", required=True, help='Task name/key, e.g. "task2"')
    p.add_argument("--model", default=None, help="Override Anthropic model")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--no-api-config", action="store_true")
    p.add_argument("--no-claude-settings", action="store_true")
    p.add_argument("--debug-prompts", action="store_true")
    args = p.parse_args()

    ver_sessions = _load_sessions(args.verifiers_json)
    out_sessions = _load_sessions(args.outputs_json)
    ver_s = _find_session_by_task(ver_sessions, args.task)
    out_s = _find_session_by_task(out_sessions, args.task)
    if ver_s is None:
        raise SystemExit(f'No verifier session matched task "{args.task}" in {args.verifiers_json}')
    if out_s is None:
        raise SystemExit(f'No output session matched task "{args.task}" in {args.outputs_json}')

    criteria = _final_verifiers_from_session(ver_s)
    if not criteria:
        raise SystemExit("No final verifier criteria found in verifier source session.")
    files = _final_agent_file_blocks(out_s)
    if not files:
        raise SystemExit("No final agent file outputs found in redo source session.")

    cfg = induce.resolve_anthropic_config(
        skip_api_config=bool(args.no_api_config),
        skip_claude_settings=bool(args.no_claude_settings),
    )
    model = args.model or cfg.model
    client = induce.make_anthropic_client(cfg)

    content = _build_message_content(criteria, files)
    if args.debug_prompts:
        print("=== verifier request blocks ===", file=sys.stderr)
        for i, block in enumerate(content):
            t = block.get("type")
            if t == "text":
                txt = str(block.get("text") or "")
                print(f"[{i}] text:\n{txt}\n", file=sys.stderr)
            elif t == "image":
                src = block.get("source") or {}
                data = str(src.get("data") or "")
                print(
                    f"[{i}] image media_type={src.get('media_type')} len={len(data)} preview={data[:48]}...",
                    file=sys.stderr,
                )
            else:
                print(f"[{i}] {block}", file=sys.stderr)
        print("=== end request blocks ===", file=sys.stderr)

    msg = client.messages.create(
        model=model,
        max_tokens=int(args.max_tokens),
        temperature=0.0,
        messages=[{"role": "user", "content": content}],
    )
    raw = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    labels = _interpret_results(raw, len(criteria))

    passed = sum(1 for x in labels if x is True)
    total = len(criteria)
    rate = passed / total if total else 0.0

    print(f'task: {args.task}')
    print(f'verifier_source: {ver_s.get("name")}')
    print(f'output_source: {out_s.get("name")}')
    print(f'model: {model}')
    print("")
    for i, (c, lab) in enumerate(zip(criteria, labels)):
        tag = "PASS" if lab is True else "FAIL"
        print(f"[{i:02d}] {tag} - {c}")
    print("")
    print(f"overall_success_rate: {passed}/{total} = {rate:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

