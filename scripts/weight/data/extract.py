"""Extract DPO / OPD / REINFORCE learning units from weight-format session JSON.

Weight JSON structure (output of ``export_task_sessions.py --format weight``):

    session.system_prompt           → full system prompt (tool descriptions included)
    session.tool_schemas            → structured tool JSON schemas
    session.task_units[i]:
      .intent                       → "planning" | node description
      .agent_trajectories[j]:       → round j of agent execution
        .prompt                     → user prompt text for this round
        .messages                   → [user, assistant+tool_calls, tool, assistant, ...]
      .human_trajectories           → [{type, round_index, prompt}, ...]
      .verifiers                    → [{criterion, status}, ...]

Messages are already in OpenAI chat format — no reverse parsing needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .reward import compute_reward


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _completion_and_mask(
    messages: list[dict],
) -> tuple[list[dict], list[bool]]:
    """Return (completion_messages, is_agent_mask) from a round's messages.

    Skips the first ``user`` message (it belongs to the prompt, not the
    completion) and marks ``assistant`` messages as agent turns.
    """
    completion = messages[1:]
    is_agent = [m.get("role") == "assistant" for m in completion]
    return completion, is_agent


def _build_base_prompt(system_prompt: str, first_user_content: str) -> list[dict]:
    """System message + initial user message."""
    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": first_user_content})
    return msgs


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

def extract_dpo_pairs(sessions: list[dict]) -> list[dict[str, Any]]:
    """Extract DPO preference pairs from weight-format sessions.

    For each execution task_unit with >=2 rounds and >=1 follow_up,
    pairs round k (rejected) with round k+1 (chosen).

    Prompt is the initial system + user message (same for all pairs in a
    task_unit).  Each round's completion (messages after the first user
    message) becomes the chosen or rejected trajectory.

    Returns list of::

        {prompt, chosen, rejected, chosen_is_agent, rejected_is_agent}
    """
    pairs: list[dict[str, Any]] = []

    for session in sessions:
        system_prompt = session.get("system_prompt", "")

        for unit in session.get("task_units", []):
            if unit.get("intent") == "planning":
                continue
            rounds = unit.get("agent_trajectories", [])
            human_traj = unit.get("human_trajectories", [])
            has_follow_up = any(h.get("type") == "follow_up" for h in human_traj)
            if len(rounds) < 2 or not has_follow_up:
                continue

            first_user = rounds[0]["messages"][0]["content"] if rounds[0].get("messages") else ""
            base_prompt = _build_base_prompt(system_prompt, first_user)

            for k in range(len(rounds) - 1):
                rej_msgs, rej_is_agent = _completion_and_mask(rounds[k]["messages"])
                cho_msgs, cho_is_agent = _completion_and_mask(rounds[k + 1]["messages"])

                if not rej_msgs or not cho_msgs:
                    continue
                if not any(rej_is_agent) or not any(cho_is_agent):
                    continue

                pairs.append({
                    "prompt": base_prompt,
                    "chosen": cho_msgs,
                    "rejected": rej_msgs,
                    "chosen_is_agent": cho_is_agent,
                    "rejected_is_agent": rej_is_agent,
                })

    return pairs


# ---------------------------------------------------------------------------
# OPD extraction (offline)
# ---------------------------------------------------------------------------

def extract_opd_examples(sessions: list[dict]) -> list[dict[str, Any]]:
    """Extract offline OPD examples from weight-format sessions.

    For each task_unit with >=2 rounds and human feedback, produces one
    example per round k in [0, N-2]:

    - **student_prompt**: accumulated context the model saw before round k.
    - **teacher_prompt**: student_prompt + privileged human feedback from
      round k onwards.
    - **completion**: round k's messages (assistant turns + tool results).
    - **is_agent**: per-message mask (True for assistant).

    Returns list of::

        {student_prompt, teacher_prompt, completion, is_agent, round_index}
    """
    examples: list[dict[str, Any]] = []

    for session in sessions:
        system_prompt = session.get("system_prompt", "")

        for unit in session.get("task_units", []):
            if unit.get("intent") == "planning":
                continue
            rounds = unit.get("agent_trajectories", [])
            human_traj = unit.get("human_trajectories", [])
            if len(rounds) < 2 or not human_traj:
                continue

            first_user = rounds[0]["messages"][0]["content"] if rounds[0].get("messages") else ""
            base_prompt = _build_base_prompt(system_prompt, first_user)

            for k in range(len(rounds) - 1):
                completion, is_agent = _completion_and_mask(rounds[k]["messages"])
                if not completion or not any(is_agent):
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

def extract_reinforce_examples(sessions: list[dict]) -> list[dict[str, Any]]:
    """Extract REINFORCE examples from weight-format sessions.

    For each round k, builds the accumulated context as prompt, round k's
    messages as completion, and computes a scalar reward from verifier pass
    rate and human intervention count.

    Returns list of::

        {prompt, completion, is_agent, reward}
    """
    examples: list[dict[str, Any]] = []

    for session in sessions:
        system_prompt = session.get("system_prompt", "")

        for unit in session.get("task_units", []):
            if unit.get("intent") == "planning":
                continue
            rounds = unit.get("agent_trajectories", [])
            human_traj = unit.get("human_trajectories", [])
            if not rounds:
                continue

            first_user = rounds[0]["messages"][0]["content"] if rounds[0].get("messages") else ""
            base_prompt = _build_base_prompt(system_prompt, first_user)
            reward = compute_reward(unit)

            for k, rnd in enumerate(rounds):
                completion, is_agent = _completion_and_mask(rnd["messages"])
                if not completion or not any(is_agent):
                    continue

                prompt = _build_conversation_context(
                    base_prompt, rounds, human_traj, up_to_round=k,
                )

                examples.append({
                    "prompt": prompt,
                    "completion": completion,
                    "is_agent": is_agent,
                    "reward": reward,
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
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        sessions = json.load(f)
    if not isinstance(sessions, list):
        sessions = [sessions]

    if args.mode == "dpo":
        units = extract_dpo_pairs(sessions)
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
