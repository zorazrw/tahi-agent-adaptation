#!/usr/bin/env python3
"""OPD from export_task_sessions JSON: one learning unit per *initial* agent step.

For every task in a session, this exporter takes the agent's **first**
``agent_trajectory`` (the workflow-generation step or the initial response to
``step k``) and splits it into one learning unit per assistant message. The
later user interactions -- both the chat follow-ups and the file/brain/workflow
edits -- are aggregated into ``followup_actions`` on each unit (cumulative tail
across this task and all subsequent tasks), where ``summarize_followups.py``
can later condense them into a ``summary`` golden answer.

Correction-iteration trajectories (``agent_trajectories[1:]``) are *not*
extracted as units; they are folded into the running ``history`` so subsequent
tasks see the finalized post-correction state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_HUMAN_ACTION_TYPES = (
    "follow_up",
    "file_edit",
    "brain_edit",
    "edit_workflow",
    "edit_verifier",
)


def _normalize_human_action(item: dict) -> dict | None:
    """Coerce a ``human_trajectories`` entry into a compact dict.

    Keeps the type discriminator and the most useful payload fields; passes
    through long bodies (e.g. ``ai``/``edited`` for ``file_edit``) so the
    summarizer can decide what to keep or diff.
    """
    if not isinstance(item, dict):
        return None
    t = item.get("type")
    if t not in _HUMAN_ACTION_TYPES:
        return None
    out: dict = {"type": t, "round_index": item.get("round_index")}
    if t == "follow_up":
        out["prompt"] = item.get("prompt") or ""
    elif t == "file_edit":
        out["path"] = item.get("path") or ""
        if "ai" in item:
            out["ai"] = item.get("ai")
        if "edited" in item:
            out["edited"] = item.get("edited")
    elif t == "brain_edit":
        if "memory" in item:
            out["memory"] = item.get("memory")
        if "skill" in item:
            out["skill"] = item.get("skill")
    elif t == "edit_workflow":
        if "workflow" in item:
            out["workflow"] = item.get("workflow")
    elif t == "edit_verifier":
        for key in ("nodeId", "criterion", "raw"):
            if key in item:
                out[key] = item[key]
    return out


def _is_assistant_msg(msg: object) -> bool:
    return isinstance(msg, dict) and msg.get("role") == "assistant"


def _split_initial_trajectory_into_units(
    msgs: list[dict],
    *,
    base_history: list[dict],
    task_intent: str | None,
    task_index: int,
    followup_actions: list[dict],
    next_index: int,
) -> tuple[list[dict], list[dict]]:
    """Walk a task's initial ``agent_trajectory`` and emit one unit per assistant message.

    The trajectory typically begins with a leading user message that contains
    the task instruction (e.g. the workflow-plan prompt or the per-step
    instruction). That message is carried in the first sub-unit's
    ``user_messages``; subsequent sub-units leave ``user_messages`` empty
    because their trigger (a tool result) is already in ``history``.

    Returns ``(new_units, advanced_history)`` where ``advanced_history``
    contains every message in ``msgs`` (so callers can append the rest of the
    task's trajectories to it).
    """
    new_units: list[dict] = []
    cursor_history = list(base_history)
    pending: list[dict] = []  # messages between the previous emitted assistant and the next one
    initial_user_msg_obj: dict | None = None
    initial_user_msg_text = ""
    sub_index = 0

    for m in msgs:
        if not isinstance(m, dict):
            continue
        if not _is_assistant_msg(m):
            if (
                sub_index == 0
                and initial_user_msg_obj is None
                and m.get("role") == "user"
            ):
                initial_user_msg_obj = m
                initial_user_msg_text = m.get("content") or ""
            pending.append(m)
            continue

        if sub_index == 0 and initial_user_msg_obj is not None:
            # First sub-unit: hoist the leading user message out of history into
            # ``user_messages`` so the teacher demo prepends it explicitly.
            unit_history = list(cursor_history) + [
                p for p in pending if p is not initial_user_msg_obj
            ]
            unit_user_messages = [initial_user_msg_text]
        else:
            unit_history = list(cursor_history) + pending
            unit_user_messages = []

        new_units.append({
            "index": next_index + len(new_units),
            "task_intent": task_intent,
            "task_index": task_index,
            "sub_index": sub_index,
            "is_continuation": sub_index > 0,
            "user_messages": unit_user_messages,
            "response_messages": [m],
            "followup_actions": followup_actions,
            "history": unit_history,
        })

        cursor_history.extend(pending)
        cursor_history.append(m)
        pending = []
        sub_index += 1

    # Trailing non-assistant messages (e.g. a final tool result) are folded into
    # the cursor history so the next task picks up after them.
    cursor_history.extend(pending)
    return new_units, cursor_history


def _normalized_humans_per_task(tasks: list[dict]) -> list[list[dict]]:
    """Per-task normalized human actions, each annotated with ``task_index``."""
    out: list[list[dict]] = []
    for task_idx, task in enumerate(tasks):
        per_task: list[dict] = []
        for h in task.get("human_trajectories") or []:
            n = _normalize_human_action(h)
            if n is None:
                continue
            n["task_index"] = task_idx
            per_task.append(n)
        out.append(per_task)
    return out


def _cumulative_humans_from(per_task_humans: list[list[dict]]) -> list[list[dict]]:
    """``out[i]`` = concatenation of ``per_task_humans[i:]`` (in original order)."""
    out: list[list[dict]] = []
    for i in range(len(per_task_humans)):
        tail: list[dict] = []
        for j in range(i, len(per_task_humans)):
            tail.extend(per_task_humans[j])
        out.append(tail)
    return out


def _session(blob: dict) -> dict | None:
    try:
        meta = {
            "uuid": blob.get("uuid"),
            "name": blob.get("name"),
            "initial_task_instruction": blob.get("initial_task_instruction"),
            "model": blob.get("model"),
            "system_prompt": blob.get("system_prompt"),
            "tool_schemas": blob.get("tool_schemas"),
        }
        tasks = blob.get("task_units") or []

        per_task_humans = _normalized_humans_per_task(tasks)
        cumulative_humans = _cumulative_humans_from(per_task_humans)

        prev_msgs: list[dict] = []
        units: list[dict] = []
        for task_idx, task in enumerate(tasks):
            trajs = task.get("agent_trajectories") or []
            if not trajs:
                continue
            task_intent = task.get("intent")
            first_msgs = trajs[0].get("messages") or []
            tail_humans = cumulative_humans[task_idx]

            # Units for this task come solely from the initial trajectory.
            new_units, _ = _split_initial_trajectory_into_units(
                first_msgs,
                base_history=list(prev_msgs),
                task_intent=task_intent,
                task_index=task_idx,
                followup_actions=tail_humans,
                next_index=len(units),
            )
            units.extend(new_units)

            # Advance running history through every trajectory in this task
            # (initial + corrections), so the *next* task sees the finalized,
            # post-correction conversation state.
            for traj in trajs:
                prev_msgs.extend(traj.get("messages") or [])

        out = {**meta, "learning_units": units}
        return out

    except Exception as e:
        print(f"Error exporting session {blob.get('uuid')}: {e}")
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Export trajectories as OPD learning units.")
    p.add_argument("input", nargs="?", default="-")
    p.add_argument("-o", "--output", default="-")
    args = p.parse_args()
    raw = json.load(sys.stdin) if args.input == "-" else json.loads(Path(args.input).read_text(encoding="utf-8"))
    blobs = raw if isinstance(raw, list) else [raw]
    results = [r for r in (_session(b) for b in blobs) if r is not None]
    payload = results[0] if len(results) == 1 else {"sessions": results}
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output == "-":
        sys.stdout.write(text)
    else:
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
