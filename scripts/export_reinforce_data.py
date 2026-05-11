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

DEFAULT_VERIFIER_MODEL = "claude-haiku-4-5-20251001"
_ACTION_CALL_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\(([\s\S]*)\)$")


def _strip_environment(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _strip_environment(v) for k, v in x.items() if k != "environment"}
    if isinstance(x, list):
        return [_strip_environment(i) for i in x]
    return x


def _parse_action_call(action: str) -> tuple[str, Any] | None:
    m = _ACTION_CALL_RE.match(action)
    if m is None:
        return None
    name, payload = m.group(1), m.group(2)
    name_key = name.lower()
    payload = payload.strip()
    if not payload:
        return name_key, None
    try:
        if name_key in {"write", "edit"}:
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                return name_key, parsed
            return None
        if name_key == "message":
            text = json.loads(payload)
            if isinstance(text, str):
                return name_key, text
            return None
    except json.JSONDecodeError:
        return None
    return name_key, payload


def _make_write_action(path: str, content: str) -> str:
    payload = {"path": path, "content": content}
    return f"write({json.dumps(payload, ensure_ascii=False)})"


def _file_path_from_write_payload(payload: dict[str, Any]) -> str | None:
    p = payload.get("path")
    if isinstance(p, str):
        return p
    fp = payload.get("file_path")
    return fp if isinstance(fp, str) else None


def _apply_edit_content(content: str, edit_payload: dict[str, Any]) -> str:
    edits = edit_payload.get("edits")
    if isinstance(edits, list) and len(edits) > 0:
        out = content
        for edit in edits:
            if not isinstance(edit, dict):
                continue
            old_text = edit.get("oldText")
            new_text = edit.get("newText")
            if isinstance(old_text, str) and isinstance(new_text, str) and old_text in out:
                out = out.replace(old_text, new_text, 1)
        return out

    # Claude / Anthropic style: old_string, new_string, replace_all
    old_s = edit_payload.get("old_string")
    new_s = edit_payload.get("new_string")
    if isinstance(old_s, str) and isinstance(new_s, str):
        if edit_payload.get("replace_all"):
            return content.replace(old_s, new_s)
        if old_s in content:
            return content.replace(old_s, new_s, 1)
    return content


def _step_env_file_map(step: dict) -> dict[str, str]:
    env = step.get("environment") if isinstance(step, dict) else None
    if not isinstance(env, dict):
        return {}
    file_field = env.get("file")
    if isinstance(file_field, dict):
        return {
            k: v
            for k, v in file_field.items()
            if isinstance(k, str) and isinstance(v, str)
        }
    if isinstance(file_field, list):
        out: dict[str, str] = {}
        for item in file_field:
            if not isinstance(item, dict):
                continue
            rel = item.get("path")
            content = item.get("content")
            if isinstance(rel, str) and isinstance(content, str):
                out[rel] = content
        return out
    return {}


def _rewrite_agent_trajectory(agent_steps: list[dict]) -> list[dict]:
    final_message_idx: int | None = None
    file_content_by_path: dict[str, str] = {}
    first_file_action_index: dict[str, int] = {}
    latest_env_content_by_path: dict[str, str] = {}
    latest_nonempty_env_content_by_path: dict[str, str] = {}

    for idx, step in enumerate(agent_steps):
        step_env_files = _step_env_file_map(step)
        latest_env_content_by_path.update(step_env_files)
        for path, content in step_env_files.items():
            if content.strip():
                latest_nonempty_env_content_by_path[path] = content
        action = step.get("action")
        if not isinstance(action, str):
            continue
        parsed = _parse_action_call(action)
        if parsed is None:
            continue
        kind, payload = parsed
        if kind == "message":
            final_message_idx = idx
            continue
        if kind == "write" and isinstance(payload, dict):
            path = _file_path_from_write_payload(payload)
            content = payload.get("content")
            if isinstance(path, str) and isinstance(content, str):
                if path not in first_file_action_index:
                    first_file_action_index[path] = idx
                file_content_by_path[path] = content
            continue
        if kind == "edit" and isinstance(payload, dict):
            path = _file_path_from_write_payload(payload)
            if isinstance(path, str):
                if path not in first_file_action_index:
                    first_file_action_index[path] = idx
                env_content = latest_nonempty_env_content_by_path.get(path) or latest_env_content_by_path.get(
                    path
                )
                if env_content is not None:
                    # Prefer environment snapshot of full post-edit file content.
                    file_content_by_path[path] = env_content
                else:
                    prior = file_content_by_path.get(path, "")
                    file_content_by_path[path] = _apply_edit_content(prior, payload)

    # Finalize touched file content using latest available environment snapshots.
    for path in list(first_file_action_index.keys()):
        env_content = latest_nonempty_env_content_by_path.get(path) or latest_env_content_by_path.get(path)
        if env_content is not None:
            file_content_by_path[path] = env_content

    merged_writes_by_index: dict[int, list[dict]] = {}
    for path, idx in first_file_action_index.items():
        content = file_content_by_path.get(path)
        if content is None:
            continue
        merged_writes_by_index.setdefault(idx, []).append({"action": _make_write_action(path, content)})

    out: list[dict] = []
    for idx, step in enumerate(agent_steps):
        action = step.get("action")
        parsed = _parse_action_call(action) if isinstance(action, str) else None

        if idx in merged_writes_by_index:
            merged = sorted(merged_writes_by_index[idx], key=lambda s: str(s["action"]))
            out.extend(merged)

        if parsed is None:
            out.append(step)
            continue

        kind, _ = parsed
        if kind in {"write", "edit"}:
            continue
        if kind == "message" and idx != final_message_idx:
            continue
        out.append(step)

    return out


def _top_level_workflow(step: dict) -> list[dict]:
    env = step.get("environment") if isinstance(step, dict) else None
    wf = env.get("workflow") if isinstance(env, dict) else None
    if not isinstance(wf, list):
        return []
    return [n for n in wf if isinstance(n, dict)]


def _verifier_criterion(v: dict) -> str | None:
    crit = v.get("criterion")
    return crit if isinstance(crit, str) and crit.strip() else None


def _workflow_verifier_lines_flat(wf: list[dict]) -> list[str]:
    """All verifier criterion strings from a workflow tree (pre-order: node then children)."""
    out: list[str] = []
    for node in wf:
        if not isinstance(node, dict):
            continue
        for raw_v in node.get("verifiers") or []:
            if isinstance(raw_v, dict):
                c = _verifier_criterion(raw_v)
                if c is not None:
                    out.append(c)
        children = node.get("children")
        if isinstance(children, list):
            child_nodes = [c for c in children if isinstance(c, dict)]
            out.extend(_workflow_verifier_lines_flat(child_nodes))
    return out


def _final_verifier_criteria(traj: list) -> list[str]:
    """Verifier criteria from the newest trajectory step that carries a workflow tree."""
    for step in reversed(traj):
        wf = _top_level_workflow(step)
        if not wf:
            continue
        lines = _workflow_verifier_lines_flat(wf)
        if lines:
            return lines
    return []


def _latest_agent_file_blocks(agent_steps: list[dict]) -> list[tuple[str, str]]:
    for step in reversed(agent_steps):
        file_map = _step_env_file_map(step)
        if file_map:
            blocks = list(file_map.items())
            blocks.sort(key=lambda x: x[0])
            return blocks
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


def _parse_passes_fallback(text: str, n: int) -> list[bool | None]:
    """
    Best-effort fallback when model returns malformed JSON.
    Extracts pass booleans from patterns like:
      "pass": true / "pass": false
    """
    vals = re.findall(r'"pass"\s*:\s*(true|false)', text, flags=re.IGNORECASE)
    out: list[bool | None] = [None] * n
    for i, tok in enumerate(vals[:n]):
        out[i] = tok.lower() == "true"
    return out


def _interpret_results(text: str, n: int) -> list[bool | None]:
    try:
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
    except (ValueError, json.JSONDecodeError, TypeError, KeyError):
        return _parse_passes_fallback(text, n)


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
    model: str
    max_tokens: int
    cache: dict[str, float]
    debug_prompts: bool
    client: Any | None = None
    no_llm_reward: bool = False
    max_retries: int = 1

    def score(self, criteria: list[str], file_blocks: list[tuple[str, str]]) -> float:
        if self.no_llm_reward:
            return 0.5
        if not criteria or not file_blocks:
            return 0.0
        if self.client is None:
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
        judged: list[bool | None] | None = None
        last_raw = ""
        attempts = 1 + max(0, self.max_retries)
        for attempt in range(attempts):
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.0,
                messages=[{"role": "user", "content": content}],
            )
            raw = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
            last_raw = raw
            judged_try = _interpret_results(raw, len(criteria))
            if all(v is not None for v in judged_try):
                judged = judged_try
                break
            # Retry once with a strict repair prompt if parse was partial.
            if attempt + 1 < attempts:
                content = [
                    {
                        "type": "text",
                        "text": (
                            "Your previous output was not valid JSON for this schema.\n"
                            'Return ONLY: {"results":[{"pass":true|false}, ...]} with exactly '
                            f"{len(criteria)} entries."
                        ),
                    },
                    {"type": "text", "text": "Previous output:"},
                    {"type": "text", "text": _truncate_file_text(raw, max_len=4000)},
                ]
                continue
            judged = judged_try
        if judged is None:
            judged = [None] * len(criteria)
        # Conservative default: unresolved entries count as failure.
        judged = [False if v is None else v for v in judged]
        if self.debug_prompts and any(v is False for v in judged) and "pass" not in last_raw:
            print("[warn] verifier response parse fallback used; unresolved entries marked false", file=sys.stderr)
        score = round(sum(1 for p in judged if p is True) / len(criteria), 4)
        self.cache[key] = score
        return score


def _chunk_reward(
    agent_steps: list[dict],
    human_suffix_count: int,
    final_criteria: list[str],
    scorer: LLMScorer,
) -> dict[str, float | int]:
    files = _latest_agent_file_blocks(agent_steps)
    return {"verifier": scorer.score(final_criteria, files), "human": human_suffix_count}


def _plan_steps(agent_steps: list[dict]) -> list[dict]:
    out: list[dict] = []
    for step in agent_steps:
        action = step.get("action")
        parsed = _parse_action_call(action) if isinstance(action, str) else None
        if parsed is None:
            continue
        kind, _ = parsed
        if kind == "plan":
            out.append(step)
    return out


def _first_plan_step_index(traj: list, plan_action: str) -> int | None:
    for i, st in enumerate(traj):
        if not isinstance(st, dict):
            continue
        if st.get("actor") == "agent" and st.get("action") == plan_action:
            return i
    return None


def _latest_workflow_between(
    traj: list,
    plan_action: str,
    end_idx: int,
) -> list[Any] | None:
    """Last ``environment.workflow`` from the matching plan step through ``end_idx`` (inclusive).

    ``end_idx`` should be the index in ``traj`` of the last step of the current learning unit's
    agent segment so each unit's ``plan`` tool_result reflects workflow state after that chunk,
    not necessarily the end of the whole session.
    """
    start = _first_plan_step_index(traj, plan_action)
    if start is None or start > end_idx:
        return None
    last: list[Any] | None = None
    for st in traj[start : end_idx + 1]:
        if not isinstance(st, dict):
            continue
        env = st.get("environment")
        if isinstance(env, dict) and isinstance(env.get("workflow"), list):
            last = env["workflow"]
    return last


def _inject_plan_workflow_tool_results(
    agent_steps: list[dict],
    traj: list | None = None,
    *,
    workflow_upto_idx: int | None = None,
) -> None:
    """Attach tool_result to plan steps from ``environment.workflow`` (before env is stripped).

    Resolves workflow via ``traj`` from the plan row through ``workflow_upto_idx`` (defaults to
    end of session). Falls back to the plan step's own ``environment`` if lookup fails.
    """
    end_idx = workflow_upto_idx if workflow_upto_idx is not None else (len(traj) - 1 if traj else -1)
    for step in agent_steps:
        action = step.get("action")
        if not isinstance(action, str):
            continue
        parsed = _parse_action_call(action)
        if parsed is None or parsed[0] != "plan":
            continue
        wf: Any = None
        if traj is not None and end_idx >= 0:
            wf = _latest_workflow_between(traj, action, end_idx)
        if not isinstance(wf, list):
            env = step.get("environment")
            if isinstance(env, dict):
                wf = env.get("workflow")
        if isinstance(wf, list):
            step["tool_result"] = json.dumps(wf, ensure_ascii=False)
        else:
            step["tool_result"] = "[]"


def _units(
    traj: list,
    scorer: LLMScorer,
    agent_trajectory_style: str = "default",
    merge_file_actions: bool = True,
) -> tuple[dict | None, list[dict]]:
    initial, pairs = s.pair_segments(traj)
    if not pairs:
        return initial, []
    final_criteria = _final_verifier_criteria(traj)
    human_counts = [len(hm) for _, hm in pairs]
    human_suffix_counts = [0] * len(human_counts)
    running = 0
    for i in range(len(human_counts) - 1, -1, -1):
        running += human_counts[i]
        human_suffix_counts[i] = running
    prefix = list(initial["steps"]) if initial else []
    agent_history: list[dict] = []
    initial_plan_steps: list[dict] = []
    out: list[dict] = []
    for j, (ag, hm) in enumerate(pairs):
        ag, hm = list(ag), list(hm)
        if merge_file_actions:
            ag_rewritten = _rewrite_agent_trajectory(ag)
        else:
            ag_rewritten = ag
        if j == 0:
            initial_plan_steps = _plan_steps(ag_rewritten)
        # Score each unit using file state at the end of this chunk in sequence.
        # If this chunk has no new file update, we fall back to the latest prior agent file state.
        agent_history.extend(ag)
        if agent_trajectory_style == "aggregate":
            if merge_file_actions:
                agent_traj_out = _rewrite_agent_trajectory(list(agent_history))
            else:
                agent_traj_out = list(agent_history)
        else:
            agent_traj_out = ag_rewritten
        if j > 0 and initial_plan_steps:
            plan_actions = [s.get("action") for s in initial_plan_steps]
            existing_prefix = [s.get("action") for s in agent_traj_out[: len(initial_plan_steps)]]
            if existing_prefix != plan_actions:
                agent_traj_out = [*initial_plan_steps, *agent_traj_out]
        if hm:
            segment_tail = hm[-1]
        elif ag:
            segment_tail = ag[-1]
        else:
            segment_tail = None
        try:
            wf_upto = traj.index(segment_tail) if segment_tail is not None else (len(traj) - 1 if traj else -1)
        except ValueError:
            wf_upto = len(traj) - 1 if traj else -1
        _inject_plan_workflow_tool_results(agent_traj_out, traj, workflow_upto_idx=wf_upto)
        out.append(
            {
                "index": j,
                "user_messages": [s.user_text(x) for x in prefix],
                "agent_trajectory": agent_traj_out,
                "human_trajectory": hm,
                "reward": _chunk_reward(agent_history, human_suffix_counts[j], final_criteria, scorer),
            }
        )
        prefix.extend(hm)
    return initial, out


def _rescale_session_verifier_rewards(units: list[dict]) -> None:
    """Rescale verifier rewards within a session.

    Policy:
    - First learning unit verifier is set to 0.0.
    - Last learning unit verifier is kept unchanged.
    - Middle units are linearly rescaled proportional to their original values
      between the original first and original last verifier anchors.

    If there is only one learning unit, rewards are left unchanged (nothing to anchor).
    """
    if not units:
        return

    def _get_verifier(unit: dict) -> float | None:
        reward = unit.get("reward")
        if not isinstance(reward, dict):
            return None
        v = reward.get("verifier")
        if isinstance(v, (int, float)):
            return float(v)
        return None

    def _set_verifier(unit: dict, value: float) -> None:
        reward = unit.get("reward")
        if not isinstance(reward, dict):
            reward = {}
            unit["reward"] = reward
        reward["verifier"] = round(float(value), 4)

    n = len(units)
    if n == 1:
        return

    first_orig = _get_verifier(units[0])
    last_orig = _get_verifier(units[-1])
    if first_orig is None or last_orig is None:
        return

    _set_verifier(units[0], 0.0)
    _set_verifier(units[-1], last_orig)

    if n <= 2:
        return

    denom = last_orig - first_orig
    for i in range(1, n - 1):
        current = _get_verifier(units[i])
        if current is None:
            continue
        if abs(denom) < 1e-12:
            scaled = 0.0
        else:
            scaled = ((current - first_orig) / denom) * last_orig
        # Keep rewards in [0, 1] range.
        scaled = max(0.0, min(1.0, scaled))
        _set_verifier(units[i], scaled)


def _session(
    blob: dict,
    scorer: LLMScorer,
    agent_trajectory_style: str = "default",
    merge_file_actions: bool = True,
) -> dict:
    meta = {"uuid": blob.get("uuid"), "name": blob.get("name")}
    traj = blob.get("trajectory")
    if not isinstance(traj, list):
        return s.strip_actor({**meta, "initial_message": None, "learning_units": [], "error": "bad_trajectory"})
    initial, units = _units(
        traj,
        scorer,
        agent_trajectory_style=agent_trajectory_style,
        merge_file_actions=merge_file_actions,
    )
    _rescale_session_verifier_rewards(units)
    return s.strip_actor(_strip_environment({**meta, "initial_message": initial, "learning_units": units}))


def main() -> None:
    p = argparse.ArgumentParser(description="Export trajectories with LLM-rated REINFORCE rewards.")
    p.add_argument("input", nargs="?", default="-")
    p.add_argument("-o", "--output", default="-")
    p.add_argument(
        "--model",
        default=DEFAULT_VERIFIER_MODEL,
        help=f'Anthropic model name for verifier scoring (default: {DEFAULT_VERIFIER_MODEL})',
    )
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument(
        "--agent_trajectory_style",
        choices=["default", "aggregate"],
        default="default",
        help=(
            "default: per-unit agent steps only; aggregate: each unit's agent_trajectory is the "
            "cumulative prefix of all agent steps so far. Works together with --merge_file_actions: "
            "when merge is on, that cumulative list is rewritten so Write/Edit (any casing) collapse "
            "into consolidated write(...) steps."
        ),
    )
    p.add_argument(
        "--merge_file_actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Rewrite trajectories to merge file edits into synthetic write(path, content) at first "
            "touch per path. Supports write/Write/edit/Edit and file_path or path. Composes with "
            "--agent_trajectory_style aggregate (rewrite runs on the full cumulative agent history)."
        ),
    )
    p.add_argument("--no-api-config", action="store_true")
    p.add_argument("--no-claude-settings", action="store_true")
    p.add_argument(
        "--no_llm_reward",
        action="store_true",
        help="Skip verifier LM calls and use a fixed default verifier reward of 0.5",
    )
    p.add_argument(
        "--debug-prompts",
        action="store_true",
        help="Print full verifier prompt before each model call (image bytes are truncated)",
    )
    args = p.parse_args()

    base_scorer_kw = {
        "model": args.model,
        "max_tokens": int(args.max_tokens),
        "cache": {},
        "debug_prompts": bool(args.debug_prompts),
    }
    if args.no_llm_reward:
        scorer = LLMScorer(**base_scorer_kw, client=None, no_llm_reward=True)
    else:
        cfg = induce.resolve_anthropic_config(
            skip_api_config=bool(args.no_api_config),
            skip_claude_settings=bool(args.no_claude_settings),
        )
        scorer = LLMScorer(
            **base_scorer_kw,
            client=induce.make_anthropic_client(cfg),
            no_llm_reward=False,
        )

    raw = json.load(sys.stdin) if args.input == "-" else json.loads(Path(args.input).read_text(encoding="utf-8"))
    results = [
        _session(
            b,
            scorer,
            agent_trajectory_style=args.agent_trajectory_style,
            merge_file_actions=bool(args.merge_file_actions),
        )
        for b in s.blobs(raw)
    ]
    payload = results[0] if len(results) == 1 else {"sessions": results}
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output == "-":
        sys.stdout.write(text)
    else:
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
