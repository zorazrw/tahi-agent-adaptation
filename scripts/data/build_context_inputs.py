"""
Build structured text summaries from raw human-agent session logs (out.json)
for context-based adaptation (memory + skill extraction).

Supports:
- New export shape: ``{"uuid", "name", "trajectory"}`` (see export_task_sessions.py)
- Legacy shape: ``task_units`` with ``agent_trajectory`` / ``human_trajectory``
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


def _format_verifiers_export(verifiers: list[dict]) -> str:
    """Statuses are strings: success | failure | unchecked."""
    if not verifiers:
        return "(none)"
    parts = []
    for v in verifiers:
        st = v.get("status")
        if st == "success":
            mark = "✓"
        elif st == "failure":
            mark = "✗"
        else:
            mark = "?"
        parts.append(f"{mark} {v.get('criterion', '')}")
    return "\n".join(parts)


def build_export_trajectory_text(trajectory: list[dict]) -> str:
    lines: list[str] = []
    for step in trajectory:
        actor = step.get("actor", "?")
        action = step.get("action", "")
        line = f"[{actor}] {action}"
        tr = step.get("tool_result")
        if isinstance(tr, str) and tr.strip():
            obs = tr.strip()
            if len(obs) > MAX_OBSERVATION_CHARS:
                obs = obs[:MAX_OBSERVATION_CHARS] + "…"
            line = f"{line}\n[{actor}] tool_result: {obs}"
        lines.append(line)
    return "\n".join(lines)


def _task_from_first_message_action(trajectory: list[dict]) -> str:
    for step in trajectory:
        if step.get("actor") not in ("human", "user"):
            continue
        act = step.get("action", "")
        if act.startswith("message(") and act.endswith(")"):
            inner = act[len("message("): -1]
            try:
                return str(json.loads(inner))
            except json.JSONDecodeError:
                return inner
    return ""


def _flatten_verifiers_from_workflow(workflow: Any) -> list[dict]:
    out: list[dict] = []
    if not isinstance(workflow, list):
        return out

    def walk(nodes: Any) -> None:
        for n in nodes:
            if not isinstance(n, dict):
                continue
            for v in n.get("verifiers") or []:
                if isinstance(v, dict) and "criterion" in v:
                    out.append(v)
                elif isinstance(v, str):
                    out.append({"criterion": v, "status": "failure"})
            walk(n.get("children") or [])

    walk(workflow)
    return out


def _verifiers_from_trajectory_env(trajectory: list[dict]) -> str:
    for step in reversed(trajectory):
        env = step.get("environment") or {}
        flat = _flatten_verifiers_from_workflow(env.get("workflow"))
        if flat:
            return _format_verifiers_export(flat)
    return "(none)"


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
    if "trajectory" in data and isinstance(data["trajectory"], list):
        return [data]
    if "task_units" in data or "session_id" in data:
        return [data]
    return []


def build_context_inputs(sessions_payload: Any) -> list[dict[str, Any]]:
    """Return a list of structured dicts, one per task_unit with execution data.

    Skips sessions with no agent steps in the trajectory (or empty legacy ``agent_trajectory``).

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
        if isinstance(session.get("trajectory"), list):
            traj = session["trajectory"]
            if not any(s.get("actor") == "agent" for s in traj if isinstance(s, dict)):
                continue
            task = _task_from_first_message_action(traj) or session.get("name", "")
            name = session.get("name", "") or ""
            trajectory_text = build_export_trajectory_text(traj)
            verifiers_text = _verifiers_from_trajectory_env(traj)
            inputs.append({
                "task": task,
                "intent": name,
                "verifiers": verifiers_text,
                "trajectory_text": trajectory_text,
                "source": f"session_{si}",
            })
            continue

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
