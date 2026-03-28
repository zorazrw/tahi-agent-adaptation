"""
Build structured text summaries from raw human-agent session logs (out.json)
for context-based adaptation (memory + skill extraction).

Each task_unit with a non-empty agent_trajectory is converted into a dict
containing the task description, intent, verifier status, and a human-readable
trajectory text with agent actions, observations, and human messages interleaved.
"""

from __future__ import annotations

import json
import logging
from typing import Any


logger = logging.getLogger(__name__)

MAX_OBSERVATION_CHARS = 500

def _extract_tool_result(raw: dict) -> str:
    """Extract tool result text from a user/tool_result entry."""
    parts: list[str] = []

    msg = raw.get("message", {})
    if isinstance(msg, str):
        if msg.strip():
            parts.append(msg)
        content: Any = []
    elif isinstance(msg, dict):
        content = msg.get("content", [])
    else:
        content = []

    if isinstance(content, str):
        if content.strip():
            parts.append(content)
        blocks: list[Any] = []
    elif isinstance(content, list):
        blocks = content
    else:
        blocks = []

    for block in blocks:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            text = block.get("content", block.get("text", ""))
            if isinstance(text, str):
                parts.append(text)

    tur = raw.get("tool_use_result")
    if isinstance(tur, str):
        if tur.strip():
            parts.append(tur)
    elif isinstance(tur, dict) and tur:
        stdout = tur.get("stdout", "")
        stderr = tur.get("stderr", "")
        content_field = tur.get("content", "")
        combined = "\n".join(
            str(x)
            for x in (stdout, stderr, content_field)
            if x not in (None, "")
        )
        if combined:
            parts.append(combined)

    return "\n".join(parts).strip() or "(empty result)"


def _truncate(text: str, limit: int = MAX_OBSERVATION_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _format_verifiers(verifiers: list[dict]) -> str:
    if not verifiers:
        return "(none)"
    parts = []
    for v in verifiers:
        mark = "✓" if v.get("status") else "✗"
        parts.append(f"{mark} {v.get('criterion', '')}")
    return "\n".join(parts)


def build_unit_trajectory_text(unit: dict) -> str:
    """Convert a task_unit's agent_trajectory + human_trajectory into readable text.

    Human messages are inserted at round boundaries (after a ``result`` and
    before the next ``system/init``), which mirrors the actual interaction order.
    """
    agent_traj = unit.get("agent_trajectory", [])
    human_msgs = list(unit.get("human_trajectory", []))
    human_idx = 0
    round_num = 0
    lines: list[str] = []

    for entry in agent_traj:
        raw = entry.get("raw", {})
        entry_type = raw.get("type")

        if entry_type == "system":
            if round_num > 0 and human_idx < len(human_msgs):
                prompt = human_msgs[human_idx].get("prompt", "")
                lines.append(f"[Human] {prompt}")
                human_idx += 1
            round_num += 1
            lines.append(f"\n--- Round {round_num} ---")

        elif entry_type == "result":
            subtype = raw.get("subtype", "")
            result_text = _truncate(raw.get("result", ""), 300)
            lines.append(f"[Result] {subtype}: {result_text}")

        elif entry_type == "assistant":
            content_blocks = raw.get("message", {}).get("content", [])
            for block in content_blocks:
                if block.get("type") == "text":
                    lines.append(f"[Agent] {block['text']}")
                elif block.get("type") == "tool_use":
                    name = block.get("name", "?")
                    inp = json.dumps(block.get("input", {}), ensure_ascii=False)
                    lines.append(f"[Agent] Tool call: {name}({_truncate(inp, 300)})")

        elif entry_type == "user":
            result_text = _extract_tool_result(raw)
            lines.append(f"[Observation] {_truncate(result_text)}")

    while human_idx < len(human_msgs):
        prompt = human_msgs[human_idx].get("prompt", "")
        lines.append(f"[Human] {prompt}")
        human_idx += 1

    return "\n".join(lines)


def normalize_sessions_list(data: Any) -> list[dict[str, Any]]:
    """Accept export_task_sessions shapes: ``{"sessions": [...]}`` or a single session object."""
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if not isinstance(data, dict):
        return []
    if isinstance(data.get("sessions"), list):
        return [x for x in data["sessions"] if isinstance(x, dict)]
    if "task_units" in data or "session_id" in data:
        return [data]
    return []


def build_context_inputs(sessions_payload: Any) -> list[dict[str, Any]]:
    """Return a list of structured dicts, one per task_unit with execution data.

    Only units whose ``agent_trajectory`` is non-empty are included.

    Each dict contains:
      task              – initial_task_instruction
      intent            – unit intent
      verifiers         – formatted verifier string
      trajectory_text   – human-readable trajectory
      source            – "session_{i}/unit_{j}"
    """
    inputs: list[dict[str, Any]] = []
    session_list = normalize_sessions_list(sessions_payload)

    for si, session in enumerate(session_list):
        task = session.get("initial_task_instruction", "")
        for ui, unit in enumerate(session.get("task_units", [])):
            agent_traj = unit.get("agent_trajectory", [])
            if not agent_traj:
                continue

            trajectory_text = build_unit_trajectory_text(unit)
            verifiers_text = _format_verifiers(unit.get("verifiers", []))

            inputs.append({
                "task": task,
                "intent": unit.get("intent", ""),
                "verifiers": verifiers_text,
                "trajectory_text": trajectory_text,
                "source": f"session_{si}/unit_{ui}",
            })

    logger.info("Built %d context inputs from sessions", len(inputs))
    return inputs
