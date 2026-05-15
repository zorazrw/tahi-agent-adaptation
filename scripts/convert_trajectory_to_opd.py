#!/usr/bin/env python3
"""Convert default trajectory exports into OPD-style learning units.

Input is the default ``export_task_sessions.py`` JSON shape:

    [{"uuid": "...", "name": "...", "trajectory": [...]}]

Output is shaped like the downstream OPD data expected by ``tinker_formatter``:

    {
      "uuid": "...",
      "name": "...",
      "initial_task_instruction": "...",
      "system_prompt": "...",
      "tool_schemas": [...],
      "learning_units": [
        {"index": 0, "user_messages": [...], "human_trajectory": [...], "history": [...]}
      ]
    }

The default trajectory export no longer has raw SDK messages, so this script
reconstructs chat messages from the human-readable ``action`` strings.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from export_task_sessions import (  # noqa: E402
    PI_TOOL_SCHEMAS,
    WORKFLOW_PLAN_INSTRUCTION,
    _pi_system_prompt,
)


TOOL_NAME_MAP = {
    "Read": "read",
    "Write": "write",
    "Edit": "edit",
    "Bash": "bash",
    "Grep": "grep",
    "Find": "find",
    "Ls": "ls",
    "AskUserQuestion": "ask_user_question",
}

TOOL_ARG_KEY_MAP = {
    "read": {"file_path": "path"},
    "write": {"file_path": "path"},
    "edit": {"file_path": "path", "old_string": "oldText", "new_string": "newText"},
    "grep": {"ignore_case": "ignoreCase"},
}

ACTION_RE = re.compile(r"^([A-Za-z_][\w]*)\(")


def _read_json(path: str) -> Any:
    if path == "-":
        return json.load(sys.stdin)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str, payload: Any) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path == "-":
        sys.stdout.write(text)
    else:
        Path(path).write_text(text, encoding="utf-8")


def _blobs(data: Any) -> list[dict]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict) and isinstance(data.get("sessions"), list):
        return [x for x in data["sessions"] if isinstance(x, dict)]
    if isinstance(data, dict) and isinstance(data.get("trajectory"), list):
        return [data]
    return []


def _decode_call_string(action: str, name: str) -> str | None:
    prefix = name + "("
    if not action.startswith(prefix) or not action.endswith(")"):
        return None
    inner = action[len(prefix) : -1].strip()
    if len(inner) >= 2 and inner[0] == '"' == inner[-1]:
        try:
            return str(json.loads(inner))
        except json.JSONDecodeError:
            return None
    return inner


def _action_payload(action: str, name: str) -> Any:
    prefix = name + "("
    if not action.startswith(prefix) or not action.endswith(")"):
        return None
    inner = action[len(prefix) : -1].strip()
    if not inner:
        return {}
    try:
        return json.loads(inner)
    except json.JSONDecodeError:
        return inner


def _first_message_text(steps: list[dict]) -> str:
    for step in steps:
        action = step.get("action")
        if isinstance(action, str):
            text = _decode_call_string(action, "message")
            if text is not None:
                return text
    return ""


def _extract_initial_task_instruction(traj: list[dict]) -> str:
    for step in traj:
        if step.get("actor") != "user":
            continue
        action = step.get("action")
        if not isinstance(action, str):
            continue
        text = _decode_call_string(action, "message")
        if text is not None:
            return text
    return ""


def _extract_cwd(traj: list[dict]) -> str | None:
    for step in traj:
        for key in ("message", "tool_result"):
            value = step.get(key)
            if not isinstance(value, str):
                continue
            match = re.search(r"^Working directory:\s*(.+)$", value, flags=re.MULTILINE)
            if match:
                return match.group(1).strip()
    return None


def _chunk_by_actor(traj: list[dict]) -> list[tuple[str, list[dict]]]:
    chunks: list[tuple[str, list[dict]]] = []
    for step in traj:
        if not isinstance(step, dict):
            continue
        actor = step.get("actor")
        if actor not in ("user", "agent"):
            continue
        if chunks and chunks[-1][0] == actor:
            chunks[-1][1].append(step)
        else:
            chunks.append((actor, [step]))
    return chunks


def _normalize_tool_name(raw_name: str) -> str:
    return TOOL_NAME_MAP.get(raw_name, raw_name)


def _normalize_tool_args(tool_name: str, args: Any) -> Any:
    if not isinstance(args, dict):
        return {"arguments": args}
    args = dict(args)
    key_map = TOOL_ARG_KEY_MAP.get(tool_name, {})
    for old_key, new_key in key_map.items():
        if old_key in args and new_key not in args:
            args[new_key] = args.pop(old_key)
    if tool_name == "edit" and "edits" not in args and ("oldText" in args or "newText" in args):
        args["edits"] = [{"oldText": args.pop("oldText", ""), "newText": args.pop("newText", "")}]
    return args


def _parse_tool_call(action: str) -> tuple[str, dict] | None:
    match = ACTION_RE.match(action)
    if not match:
        return None
    raw_name = match.group(1)
    if raw_name in {"message", "plan"}:
        return None
    args = _action_payload(action, raw_name)
    tool_name = _normalize_tool_name(raw_name)
    return tool_name, _normalize_tool_args(tool_name, args)


def _workflow_tasks_from_environment(step: dict) -> list[dict]:
    env = step.get("environment")
    if not isinstance(env, dict):
        return []
    workflow = env.get("workflow")
    if not isinstance(workflow, list):
        return []
    return _to_llm_native_tree(workflow)


def _to_llm_native_tree(tree: Any) -> list[dict]:
    out: list[dict] = []
    if not isinstance(tree, list):
        return out
    for node in tree:
        if not isinstance(node, dict):
            continue
        verifiers: list[str] = []
        for verifier in node.get("verifiers") or []:
            if isinstance(verifier, dict):
                criterion = verifier.get("criterion")
                if criterion:
                    verifiers.append(str(criterion))
            elif verifier:
                verifiers.append(str(verifier))
        item = {
            "description": str(node.get("description") or ""),
            "outputFiles": [str(x) for x in (node.get("outputFiles") or [])],
            "verifiers": verifiers,
        }
        children = _to_llm_native_tree(node.get("children"))
        if children:
            item["children"] = children
        out.append(item)
    return out


def _steps_to_chat(steps: list[dict]) -> list[dict]:
    chat: list[dict] = []
    tool_seq = 0
    for step in steps:
        action = step.get("action")
        if not isinstance(action, str) or not action:
            continue

        actor = step.get("actor", "agent")
        role = "user" if actor == "user" else "assistant"

        message_text = _decode_call_string(action, "message")
        if message_text is not None:
            chat.append({"role": role, "content": message_text})
            continue

        plan_text = _decode_call_string(action, "plan")
        if plan_text is not None:
            tasks = _workflow_tasks_from_environment(step)
            tool_seq += 1
            call_id = f"call_{tool_seq}"
            chat.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "workflow_plan",
                                "arguments": json.dumps({"tasks": tasks}, ensure_ascii=False),
                            },
                        }
                    ],
                }
            )
            chat.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": "workflow_plan",
                    "content": "Workflow plan registered. Stop now. Do not execute any steps.",
                }
            )
            continue

        parsed = _parse_tool_call(action)
        if parsed is None:
            if role == "user":
                chat.append({"role": "user", "content": action})
            continue

        tool_name, args = parsed
        tool_seq += 1
        call_id = f"call_{tool_seq}"
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": step.get("message") if isinstance(step.get("message"), str) else None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }
            ],
        }
        chat.append(assistant_msg)
        tool_result = step.get("tool_result")
        if tool_result is not None:
            chat.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": tool_result if isinstance(tool_result, str) else json.dumps(tool_result, ensure_ascii=False),
                }
            )
    return chat


def _user_messages_from_steps(steps: list[dict]) -> list[str]:
    out: list[str] = []
    for step in steps:
        action = step.get("action")
        if not isinstance(action, str):
            continue
        text = _decode_call_string(action, "message")
        if text is not None:
            out.append(text)
    return out


def _learning_units(chunks: list[tuple[str, list[dict]]]) -> list[dict]:
    units: list[dict] = []
    for i in range(len(chunks) - 2):
        actor, prompt_steps = chunks[i]
        next_actor, _agent_steps = chunks[i + 1]
        feedback_actor, feedback_steps = chunks[i + 2]
        if actor != "user" or next_actor != "agent" or feedback_actor != "user":
            continue
        user_messages = _user_messages_from_steps(prompt_steps)
        human_trajectory = _steps_to_chat(feedback_steps)
        if not user_messages or not human_trajectory:
            continue
        history_steps: list[dict] = []
        for _chunk_actor, chunk_steps in chunks[:i]:
            history_steps.extend(chunk_steps)
        units.append(
            {
                "index": len(units),
                "user_messages": user_messages,
                "human_trajectory": human_trajectory,
                "history": _steps_to_chat(history_steps),
            }
        )
    return units


def _session(blob: dict, *, model: str, cwd: str | None, include_full_history: bool) -> dict:
    traj = blob.get("trajectory")
    if not isinstance(traj, list):
        return {
            "uuid": blob.get("uuid"),
            "name": blob.get("name"),
            "initial_task_instruction": "",
            "model": model,
            "system_prompt": _pi_system_prompt(cwd),
            "tool_schemas": copy.deepcopy(PI_TOOL_SCHEMAS),
            "learning_units": [],
            "error": "bad_trajectory",
        }

    initial_task_instruction = blob.get("initial_task_instruction")
    if not isinstance(initial_task_instruction, str) or not initial_task_instruction:
        initial_task_instruction = _extract_initial_task_instruction(traj)

    export_cwd = cwd or _extract_cwd(traj)
    chunks = _chunk_by_actor(traj)
    units = _learning_units(chunks)
    if include_full_history and not units:
        user_messages = [x for _actor, steps in chunks for x in _user_messages_from_steps(steps)]
        units = [
            {
                "index": 0,
                "user_messages": user_messages[:1] or [initial_task_instruction],
                "human_trajectory": [{"role": "user", "content": x} for x in user_messages[1:]],
                "history": _steps_to_chat(traj),
            }
        ]

    return {
        "uuid": blob.get("uuid"),
        "name": blob.get("name", ""),
        "initial_task_instruction": initial_task_instruction,
        "model": model,
        "system_prompt": _pi_system_prompt(export_cwd),
        "tool_schemas": copy.deepcopy(PI_TOOL_SCHEMAS),
        "learning_units": units,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert default export_task_sessions trajectory JSON into OPD learning-unit JSON."
    )
    parser.add_argument("input", nargs="?", default="-", help="Input default export JSON, or '-' for stdin")
    parser.add_argument("-o", "--output", default="-", help="Output JSON path, or '-' for stdout")
    parser.add_argument("--model", default="", help="Model name to write into output metadata")
    parser.add_argument("--cwd", default=None, help="Working directory to include in the system prompt")
    parser.add_argument(
        "--include-full-history-if-empty",
        action="store_true",
        help="Emit one fallback unit with the full reconstructed chat if no agent->human feedback units are found.",
    )
    args = parser.parse_args()

    raw = _read_json(args.input)
    sessions = [
        _session(
            blob,
            model=args.model,
            cwd=args.cwd,
            include_full_history=args.include_full_history_if_empty,
        )
        for blob in _blobs(raw)
    ]
    payload: Any = sessions[0] if len(sessions) == 1 else {"sessions": sessions}
    _write_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
