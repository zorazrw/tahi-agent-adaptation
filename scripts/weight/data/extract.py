"""Extract DPO / OPD / REINFORCE learning units from weight-format session JSON.

Weight JSON structure (output of ``export_task_sessions.py --format weight``):

    session.system_prompt           → role description (tool schemas live in
                                       tool_schemas; the renderer injects them
                                       into the system prompt at training time)
    session.tool_schemas            → structured tool JSON schemas
    session.task_units[i]:
      .intent                       → "planning" | node description
      .agent_trajectories[j]:       → round j of agent execution
        .prompt                     → user prompt text for this round
        .messages                   → [user, assistant+tool_calls, tool, assistant, ...]
        .output_files               → (Pi only) artifacts the agent wrote/edited
                                       this round; used for artifact-only
                                       completion construction
      .human_trajectories           → [{type, round_index, prompt}, ...]
      .verifiers                    → [{criterion, status}, ...]

Completion mode: artifact-only. Each round's completion is reconstructed from
``output_files`` as a single ``write`` tool_call per file (assistant message
only; no synthetic tool-result). Prior rounds in the prompt context retain their
full original ``messages`` so the model conditions on the real conversation
history.

Messages are in OpenAI chat format. Tool schemas are NOT embedded in the system
prompt at export time — instead they are injected by the renderer's
``create_conversation_prefix_with_tools`` so the wire-format matches what the
model has been trained on (e.g. Qwen3 ``# Tools / <tools>``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .reward import compute_per_traj_rewards

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _completion_and_mask(
    messages: list[dict],
) -> tuple[list[dict], list[bool]]:
    """Return (completion_messages, is_agent_mask) from a round's messages.

    Skips the first ``user`` message (it belongs to the prompt, not the
    completion) and marks ``assistant`` messages as agent turns.

    NOTE: superseded by :func:`_build_artifact_completion` for current training
    modes; retained for diagnostic / fallback use only.
    """
    completion = messages[1:]
    is_agent = [m.get("role") == "assistant" for m in completion]
    return completion, is_agent


def _is_complete_round(messages: list[dict]) -> bool:
    """Return True if the round ends with a final assistant message.

    Rounds that end on a tool result are incomplete (the agent was mid-execution
    when the session was cut). Such rounds should be excluded from training.
    """
    completion = messages[1:]
    if not completion:
        return False
    return completion[-1].get("role") == "assistant"


def _normalise_tool_schemas(tool_schemas: list[dict] | None) -> list[dict]:
    """Convert exported tool schemas to bare ``ToolSpec`` dicts.

    Accepts both the wrapped OpenAI shape ``{"type": "function", "function": {...}}``
    and bare ``{name, description, parameters}``. Returns the bare form, which is
    what ``Renderer.create_conversation_prefix_with_tools`` expects.
    """
    out: list[dict] = []
    for ts in tool_schemas or []:
        if not isinstance(ts, dict):
            continue
        fn = ts.get("function") if isinstance(ts.get("function"), dict) else ts
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str):
            continue
        out.append({
            "name": name,
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
        })
    return out


def _build_base_prompt(
    system_prompt: str,
    tool_schemas: list[dict] | None,
    first_user_content: str,
    renderer: Any | None = None,
) -> list[dict]:
    """System message(s) + initial user message.

    When ``renderer`` is provided and supports tool-prefix construction, defer
    to its ``create_conversation_prefix_with_tools`` so tool schemas are encoded
    in the model's native chat-template form (e.g. Qwen3 ``# Tools / <tools>``).
    Falls back to a plain system message if the renderer doesn't implement it
    (e.g. RoleColon) or there are no tools to inject.
    """
    msgs: list[dict] = []
    tools = _normalise_tool_schemas(tool_schemas)
    if renderer is not None and tools:
        try:
            prefix = renderer.create_conversation_prefix_with_tools(tools, system_prompt or "")
        except NotImplementedError:
            prefix = None
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "renderer.create_conversation_prefix_with_tools failed (%s); "
                "falling back to plain system prompt", e,
            )
            prefix = None
        if prefix:
            msgs.extend(dict(m) for m in prefix)
        elif system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
    elif system_prompt:
        msgs.append({"role": "system", "content": system_prompt})

    msgs.append({"role": "user", "content": first_user_content})
    return msgs


def _build_artifact_completion(
    output_files: list[dict] | None,
) -> tuple[list[dict], list[bool]]:
    """Construct an artifact-only completion from a round's ``output_files``.

    Emits one ``assistant`` message per output file, containing a single
    ``write`` tool_call with ``{path, content}`` as arguments. No synthetic
    ``tool`` result is appended — only assistant tokens are trained on, so the
    tool result adds context overhead without any gradient signal.

    Returns ``([], [])`` if there are no usable files (round skipped).
    """
    messages: list[dict] = []
    is_agent: list[bool] = []
    files = output_files or []
    for i, f in enumerate(files):
        if not isinstance(f, dict):
            continue
        path = str(f.get("path", "") or "").strip()
        content = f.get("content")
        if not path or not isinstance(content, str):
            continue
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": f"call_artifact_{i}",
                "type": "function",
                "function": {
                    "name": "write",
                    "arguments": json.dumps(
                        {"path": path, "content": content}, ensure_ascii=False,
                    ),
                },
            }],
        })
        is_agent.append(True)
    return messages, is_agent


def _build_conversation_context(
    base_prompt: list[dict],
    rounds: list[dict],
    human_traj: list[dict],
    up_to_round: int,
) -> list[dict]:
    """Build accumulated context up to (but not including) round *k*.

    For k=0 returns ``base_prompt`` alone.
    For k>0 appends each prior round's completion + the follow-up that
    triggered the next round.
    """
    context = list(base_prompt)
    for r in range(up_to_round):
        context.extend(rounds[r]["messages"][1:])
        for h in human_traj:
            if h.get("type") == "follow_up" and h.get("round_index") == r:
                text = h.get("prompt", "")
                if text:
                    context.append({"role": "user", "content": text})
    return context


def _augment_with_feedback(
    prompt: list[dict],
    human_actions: list[dict],
) -> list[dict]:
    """Append privileged human feedback to the last user message (for teacher)."""
    parts: list[str] = []
    for action in human_actions:
        atype = action.get("type", "follow_up")
        if atype == "follow_up":
            text = action.get("prompt", "")
            if text:
                parts.append(f'- Human feedback: "{text}"')
        elif atype == "file_edit":
            path = action.get("path", "")
            parts.append(f"- Human edited file '{path}'")
        elif atype == "edit_workflow":
            parts.append("- Human revised the workflow plan")
        elif atype == "edit_verifier":
            parts.append("- Human edited verification criteria")

    if not parts:
        return prompt

    suffix = (
        "\n\nThe following is feedback from a human collaborator on a "
        "previous attempt. Use this to guide your response:\n"
        + "\n".join(parts)
        + "\n\nNow generate an improved response incorporating the above guidance."
    )
    augmented = list(prompt)
    last = augmented[-1]
    augmented[-1] = {"role": last["role"], "content": last["content"] + suffix}
    return augmented


# ---------------------------------------------------------------------------
# DPO extraction
# ---------------------------------------------------------------------------

def extract_dpo_pairs(
    sessions: list[dict],
    renderer: Any | None = None,
    pair_mode: str = "adjacent",
) -> list[dict[str, Any]]:
    """Extract DPO preference pairs from weight-format sessions.

    Two construction modes (selected via ``pair_mode``):

    - ``"adjacent"`` (default): for each execution task_unit, pair round k
      (rejected) with round k+1 (chosen) for every k in [0, n_rounds-1).
      Prompt for pair k includes system + tools + initial user + prior
      rounds' full messages + follow-ups for rounds 0..k-1. Tacit-preference
      design: the follow-up that triggered chosen R_{k+1} is intentionally
      withheld from the prompt so the model learns the implicit "later
      iteration is better" signal.

    - ``"first_last"``: produce **one** pair per execution unit with rejected
      = R_0's artifact and chosen = R_last's artifact. Prompt is just the
      base prompt (system + tools + initial user) — no history, no
      follow-ups. This is the cleanest sanity-overfit setup: 1 pair per
      unit, zero role conflict, shortest prompts.

    Completions are artifact-only in both modes: each round's chosen /
    rejected is reconstructed from ``round.output_files`` as a single
    ``write`` tool_call per file. Pairs where either side has no artifact
    (e.g. agent only ran ``read`` / ``bash``) are skipped.

    Returns list of::

        {prompt, chosen, rejected, chosen_is_agent, rejected_is_agent}
    """
    if pair_mode not in {"adjacent", "first_last"}:
        raise ValueError(
            f"pair_mode must be 'adjacent' or 'first_last', got {pair_mode!r}"
        )

    pairs: list[dict[str, Any]] = []

    for session in sessions:
        system_prompt = session.get("system_prompt", "")
        tool_schemas = session.get("tool_schemas")

        for unit in session.get("task_units", []):
            if unit.get("intent") == "planning":
                continue
            rounds = unit.get("agent_trajectories", [])
            human_traj = unit.get("human_trajectories", [])
            has_follow_up = any(h.get("type") == "follow_up" for h in human_traj)
            if len(rounds) < 2 or not has_follow_up:
                continue

            first_user = rounds[0]["messages"][0]["content"] if rounds[0].get("messages") else ""
            base_prompt = _build_base_prompt(system_prompt, tool_schemas, first_user, renderer)

            if pair_mode == "first_last":
                rej_msgs, rej_is_agent = _build_artifact_completion(rounds[0].get("output_files"))
                cho_msgs, cho_is_agent = _build_artifact_completion(rounds[-1].get("output_files"))
                if not rej_msgs or not cho_msgs:
                    continue
                pairs.append({
                    "prompt": list(base_prompt),
                    "chosen": cho_msgs,
                    "rejected": rej_msgs,
                    "chosen_is_agent": cho_is_agent,
                    "rejected_is_agent": rej_is_agent,
                })
                continue

            for k in range(len(rounds) - 1):
                rej_msgs, rej_is_agent = _build_artifact_completion(rounds[k].get("output_files"))
                cho_msgs, cho_is_agent = _build_artifact_completion(rounds[k + 1].get("output_files"))

                if not rej_msgs or not cho_msgs:
                    continue

                prompt = _build_conversation_context(
                    base_prompt, rounds, human_traj, up_to_round=k,
                )

                pairs.append({
                    "prompt": prompt,
                    "chosen": cho_msgs,
                    "rejected": rej_msgs,
                    "chosen_is_agent": cho_is_agent,
                    "rejected_is_agent": rej_is_agent,
                })

    return pairs


# ---------------------------------------------------------------------------
# OPD extraction (offline)
# ---------------------------------------------------------------------------

def extract_opd_examples(
    sessions: list[dict],
    renderer: Any | None = None,
) -> list[dict[str, Any]]:
    """Extract offline OPD examples from weight-format sessions.

    For each task_unit with >=2 rounds and human feedback, produces one example
    per round k in [0, N-2]:

    - **student_prompt**: accumulated context the model saw before round k
      (prior rounds' full original messages + their triggering follow-ups).
    - **teacher_prompt**: student_prompt + privileged human feedback from
      round k onwards.
    - **completion**: round k's artifact-only completion, built from
      ``rounds[k].output_files`` as one ``write`` tool_call per file.
    - **is_agent**: per-message mask (True for assistant).

    Rounds without ``output_files`` (no artifact produced) are skipped.

    Returns list of::

        {student_prompt, teacher_prompt, completion, is_agent, round_index}
    """
    examples: list[dict[str, Any]] = []

    for session in sessions:
        system_prompt = session.get("system_prompt", "")
        tool_schemas = session.get("tool_schemas")

        for unit in session.get("task_units", []):
            if unit.get("intent") == "planning":
                continue
            rounds = unit.get("agent_trajectories", [])
            human_traj = unit.get("human_trajectories", [])
            if len(rounds) < 2 or not human_traj:
                continue

            first_user = rounds[0]["messages"][0]["content"] if rounds[0].get("messages") else ""
            base_prompt = _build_base_prompt(system_prompt, tool_schemas, first_user, renderer)

            for k in range(len(rounds) - 1):
                completion, is_agent = _build_artifact_completion(rounds[k].get("output_files"))
                if not completion:
                    continue

                student_prompt = _build_conversation_context(
                    base_prompt, rounds, human_traj, up_to_round=k,
                )

                future_human = [
                    h for h in human_traj
                    if h.get("round_index") is None or h.get("round_index", -1) >= k
                ]
                if not future_human:
                    continue

                teacher_prompt = _augment_with_feedback(student_prompt, future_human)

                examples.append({
                    "student_prompt": student_prompt,
                    "teacher_prompt": teacher_prompt,
                    "completion": completion,
                    "is_agent": is_agent,
                    "round_index": k,
                })

    return examples


# ---------------------------------------------------------------------------
# REINFORCE extraction
# ---------------------------------------------------------------------------

def extract_reinforce_examples(
    sessions: list[dict],
    renderer: Any | None = None,
) -> list[dict[str, Any]]:
    """Extract REINFORCE examples from weight-format sessions.

    For each round k, builds the accumulated context as prompt, round k's
    artifact-only completion (from ``rounds[k].output_files``), and computes
    a scalar reward from verifier pass rate and human intervention count.

    Rounds without ``output_files`` (no artifact produced) are skipped.

    Returns list of::

        {prompt, completion, is_agent, reward}
    """
    examples: list[dict[str, Any]] = []

    for session in sessions:
        system_prompt = session.get("system_prompt", "")
        tool_schemas = session.get("tool_schemas")

        for unit in session.get("task_units", []):
            if unit.get("intent") == "planning":
                continue
            rounds = unit.get("agent_trajectories", [])
            human_traj = unit.get("human_trajectories", [])
            if not rounds:
                continue

            first_user = rounds[0]["messages"][0]["content"] if rounds[0].get("messages") else ""
            base_prompt = _build_base_prompt(system_prompt, tool_schemas, first_user, renderer)
            per_traj_rewards = compute_per_traj_rewards(unit)

            for k, rnd in enumerate(rounds):
                completion, is_agent = _build_artifact_completion(rnd.get("output_files"))
                if not completion:
                    continue

                prompt = _build_conversation_context(
                    base_prompt, rounds, human_traj, up_to_round=k,
                )

                examples.append({
                    "prompt": prompt,
                    "completion": completion,
                    "is_agent": is_agent,
                    "reward": per_traj_rewards[k],
                })

    return examples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _msg_preview(msg: dict) -> str:
    role = msg.get("role", "?")
    content = msg.get("content", "")
    tc = msg.get("tool_calls")
    parts = [f"[{role}]"]
    if tc:
        names = [c.get("function", {}).get("name", "?") for c in tc]
        parts.append(f"tool_calls={names}")
    if content:
        parts.append(content[:80])
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract learning units from weight-format session JSON.",
    )
    parser.add_argument("mode", choices=["dpo", "opd", "reinforce"])
    parser.add_argument("input", help="Path to weight JSON file")
    parser.add_argument("-o", "--output", default=None, help="Write JSON output to file")
    parser.add_argument(
        "--pair-mode",
        choices=["adjacent", "first_last"],
        default="adjacent",
        help="DPO pair construction (ignored for opd/reinforce). "
             "'adjacent' = (R_k, R_{k+1}) with accumulated history (default); "
             "'first_last' = single (R_0, R_last) per unit, prompt = base only.",
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        sessions = json.load(f)
    if not isinstance(sessions, list):
        sessions = [sessions]

    if args.mode == "dpo":
        units = extract_dpo_pairs(sessions, pair_mode=args.pair_mode)
        print(f"Extracted {len(units)} DPO pairs")
        for i, u in enumerate(units):
            print(f"\n── Pair {i} ──")
            print(f"  prompt: {len(u['prompt'])} msgs")
            print(f"  chosen: {len(u['chosen'])} msgs (agent: {sum(u['chosen_is_agent'])})")
            print(f"  rejected: {len(u['rejected'])} msgs (agent: {sum(u['rejected_is_agent'])})")
            for m in u["chosen"][:2]:
                print(f"    cho: {_msg_preview(m)}")

    elif args.mode == "opd":
        units = extract_opd_examples(sessions)
        print(f"Extracted {len(units)} OPD examples")
        for i, u in enumerate(units):
            print(f"\n── Example {i} (round {u['round_index']}) ──")
            print(f"  student_prompt: {len(u['student_prompt'])} msgs")
            print(f"  teacher_prompt: {len(u['teacher_prompt'])} msgs")
            print(f"  completion: {len(u['completion'])} msgs (agent: {sum(u['is_agent'])})")

    else:
        units = extract_reinforce_examples(sessions)
        print(f"Extracted {len(units)} REINFORCE examples")
        for i, u in enumerate(units):
            print(f"\n── Example {i} ──")
            print(f"  prompt: {len(u['prompt'])} msgs")
            print(f"  completion: {len(u['completion'])} msgs, reward={u['reward']:.3f}")

    if args.output:
        text = json.dumps(units, indent=2, ensure_ascii=False) + "\n"
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
