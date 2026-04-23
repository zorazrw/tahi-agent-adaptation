#!/usr/bin/env python3
"""REINFORCE-style export with LLM-rated verifier rewards."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

import induce  # noqa: E402
import session_export_common as s  # noqa: E402


def _strip_environment(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _strip_environment(v) for k, v in x.items() if k != "environment"}
    if isinstance(x, list):
        return [_strip_environment(i) for i in x]
    return x


def _top_level_workflow(step: dict) -> list[dict]:
    env = step.get("environment") if isinstance(step, dict) else None
    wf = env.get("workflow") if isinstance(env, dict) else None
    if not isinstance(wf, list):
        return []
    return [n for n in wf if isinstance(n, dict)]


def _verifier_criterion(v: dict) -> str | None:
    crit = v.get("criterion")
    return crit if isinstance(crit, str) and crit.strip() else None


def _final_verifier_criteria(traj: list) -> list[str]:
    """Final top-level verifier criteria text in-order."""
    for step in reversed(traj):
        wf = _top_level_workflow(step)
        if not wf:
            continue
        out: list[str] = []
        for node in wf:
            for raw_v in node.get("verifiers") or []:
                if isinstance(raw_v, dict):
                    c = _verifier_criterion(raw_v)
                    if c is not None:
                        out.append(c)
        if out:
            return out
    return []


def _latest_agent_file_blocks(agent_steps: list[dict]) -> list[tuple[str, str]]:
    for step in reversed(agent_steps):
        env = step.get("environment") if isinstance(step, dict) else None
        if not isinstance(env, dict):
            continue
        file_field = env.get("file")
        if isinstance(file_field, dict):
            blocks = [(k, v) for k, v in file_field.items() if isinstance(k, str) and isinstance(v, str)]
            if blocks:
                blocks.sort(key=lambda x: x[0])
                return blocks
        if isinstance(file_field, list):
            blocks_list: list[tuple[str, str]] = []
            for item in file_field:
                if not isinstance(item, dict):
                    continue
                rel = item.get("path")
                content = item.get("content")
                if isinstance(rel, str) and isinstance(content, str):
                    blocks_list.append((rel, content))
            if blocks_list:
                blocks_list.sort(key=lambda x: x[0])
                return blocks_list
    return []


def _truncate_file_text(text: str, max_len: int = 14_000) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n... [truncated]"


def _redact_image_bytes(text: str, max_keep: int = 160) -> str:
    """
    Redact likely inline image bytes / base64 blobs to keep debug prompts readable.

    Targets large JSON string values like:
      "content": "iVBORw0KGgoAAA...."
    """

    def repl(m: re.Match[str]) -> str:
        raw = m.group(1)
        if len(raw) <= max_keep:
            return m.group(0)
        preview = raw[: min(48, len(raw))]
        return f'"content": "[image-bytes-truncated len={len(raw)} preview={preview}...]"'

    # Very long base64-like strings are treated as binary payloads.
    return re.sub(r'"content"\s*:\s*"([A-Za-z0-9+/=\n\r]{800,})"', repl, text)


def _build_labeler_prompt_text(*, criteria: list[str]) -> str:
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
        ]
    )


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

    # Also support JSON wrappers that include base64 in "content".
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        content = parsed.get("content")
        if isinstance(content, str) and _looks_like_base64_data(content):
            return media_type, content.strip()
    return None


def _build_labeler_message_content(*, criteria: list[str], file_blocks: list[tuple[str, str]]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": _build_labeler_prompt_text(criteria=criteria)}]
    content.append({"type": "text", "text": "Output files and contents:"})
    if not file_blocks:
        content.append({"type": "text", "text": "(no output files present)"})
        return content

    for rel, text in file_blocks:
        content.append({"type": "text", "text": f"### {rel}"})
        image_payload = _extract_image_data(rel, text)
        if image_payload is not None:
            media_type, data = image_payload
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    },
                }
            )
            content.append({"type": "text", "text": "(image content attached above)"})
        else:
            cleaned = _redact_image_bytes(text)
            content.append({"type": "text", "text": _truncate_file_text(cleaned)})
        content.append({"type": "text", "text": "---"})
    return content


def _debug_render_message_content(content: list[dict[str, Any]], image_preview_chars: int = 48) -> str:
    out: list[str] = []
    for i, block in enumerate(content):
        btype = block.get("type")
        if btype == "text":
            out.append(f"[{i}] text:\n{block.get('text', '')}")
        elif btype == "image":
            src = block.get("source") if isinstance(block.get("source"), dict) else {}
            data = str(src.get("data") or "")
            out.append(
                f"[{i}] image: media_type={src.get('media_type')} "
                f"len={len(data)} preview={data[:image_preview_chars]}..."
            )
        else:
            out.append(f"[{i}] {json.dumps(block, ensure_ascii=False)}")
    return "\n".join(out)


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
    results = parsed.get("results")
    if not isinstance(results, list):
        raise ValueError("Missing results array")
    out: list[bool | None] = [None] * n
    for i in range(min(n, len(results))):
        row = results[i]
        if isinstance(row, dict) and "pass" in row:
            out[i] = bool(row["pass"])
    return out


def _eval_cache_key(criteria: list[str], file_blocks: list[tuple[str, str]]) -> str:
    h = hashlib.sha256()
    for c in criteria:
        h.update(c.encode("utf-8"))
        h.update(b"\0")
    for rel, content in file_blocks:
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(content.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


@dataclass
class LLMScorer:
    client: Any
    model: str
    max_tokens: int
    cache: dict[str, float]
    debug_prompts: bool

    def score(self, criteria: list[str], file_blocks: list[tuple[str, str]]) -> float:
        if not criteria or not file_blocks:
            return 0.0
        key = _eval_cache_key(criteria, file_blocks)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        content = _build_labeler_message_content(criteria=criteria, file_blocks=file_blocks)
        if self.debug_prompts:
            print("\n=== verifier_prompt_begin ===", file=sys.stderr)
            print(_debug_render_message_content(content), file=sys.stderr)
            print("=== verifier_prompt_end ===\n", file=sys.stderr)
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.0,
            messages=[{"role": "user", "content": content}],
        )
        raw = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        judged = _interpret_results(raw, len(criteria))
        score = round(sum(1 for p in judged if p is True) / len(criteria), 4)
        self.cache[key] = score
        return score


def _chunk_reward(
    agent_steps: list[dict],
    human_steps: list[dict],
    final_criteria: list[str],
    scorer: LLMScorer,
) -> dict[str, float | int]:
    files = _latest_agent_file_blocks(agent_steps)
    return {"verifier": scorer.score(final_criteria, files), "human": len(human_steps)}


def _units(traj: list, scorer: LLMScorer) -> tuple[dict | None, list[dict]]:
    initial, pairs = s.pair_segments(traj)
    if not pairs:
        return initial, []
    final_criteria = _final_verifier_criteria(traj)
    prefix = list(initial["steps"]) if initial else []
    agent_history: list[dict] = []
    out: list[dict] = []
    for j, (ag, hm) in enumerate(pairs):
        ag, hm = list(ag), list(hm)
        # Score each unit using file state at the end of this chunk in sequence.
        # If this chunk has no new file update, we fall back to the latest prior agent file state.
        agent_history.extend(ag)
        out.append(
            {
                "index": j,
                "user_messages": [s.user_text(x) for x in prefix],
                "agent_trajectory": ag,
                "human_trajectory": hm,
                "reward": _chunk_reward(agent_history, hm, final_criteria, scorer),
            }
        )
        prefix.extend(hm)
    return initial, out


def _session(blob: dict, scorer: LLMScorer) -> dict:
    meta = {"uuid": blob.get("uuid"), "name": blob.get("name")}
    traj = blob.get("trajectory")
    if not isinstance(traj, list):
        return s.strip_actor({**meta, "initial_message": None, "learning_units": [], "error": "bad_trajectory"})
    initial, units = _units(traj, scorer)
    return s.strip_actor(_strip_environment({**meta, "initial_message": initial, "learning_units": units}))


def main() -> None:
    p = argparse.ArgumentParser(description="Export trajectories with LLM-rated REINFORCE rewards.")
    p.add_argument("input", nargs="?", default="-")
    p.add_argument("-o", "--output", default="-")
    p.add_argument("--model", default=None, help="Override Anthropic model name")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--no-api-config", action="store_true")
    p.add_argument("--no-claude-settings", action="store_true")
    p.add_argument(
        "--debug-prompts",
        action="store_true",
        help="Print full verifier prompt before each model call (image bytes are truncated)",
    )
    args = p.parse_args()

    cfg = induce.resolve_anthropic_config(
        skip_api_config=bool(args.no_api_config),
        skip_claude_settings=bool(args.no_claude_settings),
    )
    scorer = LLMScorer(
        client=induce.make_anthropic_client(cfg),
        model=args.model or cfg.model,
        max_tokens=int(args.max_tokens),
        cache={},
        debug_prompts=bool(args.debug_prompts),
    )

    raw = json.load(sys.stdin) if args.input == "-" else json.loads(Path(args.input).read_text(encoding="utf-8"))
    results = [_session(b, scorer) for b in s.blobs(raw)]
    payload = results[0] if len(results) == 1 else {"sessions": results}
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output == "-":
        sys.stdout.write(text)
    else:
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
