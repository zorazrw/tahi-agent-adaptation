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
        .reinforce_prompt (optional)→ cached full REINFORCE prompt (cross-unit accumulated
                                       context + this round's opening user); written on first extract
      .human_trajectories           → [{type, round_index, prompt}, ...]
      .verifiers                    → [{criterion, status}, ...]
      .reward (optional)            → list of ``0.0`` / ``1.0``, one per **session rubric**
                                    (same length as the last task_unit's verifier criteria);
                                    when valid, REINFORCE extraction skips LLM regrading.
                                    Training rows still use scalar ``reward`` = mean of this list.

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

from .reward import compute_llm_rubric_file_scores

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


def _session_tools_prefix(
    system_prompt: str,
    tool_schemas: list[dict] | None,
    renderer: Any | None,
) -> list[dict]:
    """System + tool-prefix messages only (no user turn).

    Matches the initial segment of :func:`_build_file_version_index` so REINFORCE
    and DPO share the same cross-unit accumulated-context head.
    """
    tools_prefix: list[dict] = []
    tools = _normalise_tool_schemas(tool_schemas)
    if renderer is not None and tools:
        try:
            prefix = renderer.create_conversation_prefix_with_tools(tools, system_prompt or "")
            tools_prefix.extend(dict(m) for m in prefix)
        except NotImplementedError:
            pass
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "renderer.create_conversation_prefix_with_tools failed (%s); "
                "falling back to plain system prompt", e,
            )
    if not tools_prefix and system_prompt:
        tools_prefix = [{"role": "system", "content": system_prompt}]
    return tools_prefix


# ---------------------------------------------------------------------------
# DPO extraction
# ---------------------------------------------------------------------------

# File extensions that carry no training signal (data files, binaries).
# output_files with these extensions are skipped when building pairs.
_SKIP_EXTENSIONS: frozenset[str] = frozenset({
    ".json", ".jsonl", ".csv", ".tsv", ".xml", ".yaml", ".yml",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf", ".ico",
    ".zip", ".tar", ".gz",
})

def _build_file_version_index(
    session: dict,
    system_prompt: str,
    tool_schemas: list[dict] | None,
    renderer: Any | None,
) -> dict[str, list[dict]]:
    """Scan the entire session and collect, per output filename, all rounds
    that wrote/updated that file, together with the accumulated conversation
    context at the time of each write.

    Crosses task_unit boundaries so the prompt for a file version written in
    a later "polish" unit correctly reflects the full prior conversation
    (including the creation unit) rather than only the current unit's context.

    Tacit-preference design is preserved: the human follow-up that *triggered*
    a given round is included in the prompt (it is the user message that opens
    the round), but the follow-up that will trigger the *next* round is not
    yet visible, exactly as in ``adjacent`` mode.

    Returns::

        {
            "filename.py": [
                {"path": str, "content": str, "prompt": list[dict]},  # first write
                {"path": str, "content": str, "prompt": list[dict]},  # second write
                ...
            ],
            ...
        }

    Only filenames with ≥2 versions are useful for DPO; the caller filters.
    """
    accumulated: list[dict] = list(
        _session_tools_prefix(system_prompt, tool_schemas, renderer),
    )
    file_index: dict[str, list[dict]] = {}

    for unit in session.get("task_units", []):
        if unit.get("intent") == "planning":
            continue
        rounds = unit.get("agent_trajectories", [])
        human_traj = unit.get("human_trajectories", [])

        for round_idx, rnd in enumerate(rounds):
            messages = rnd.get("messages", [])
            if not messages:
                continue

            # prompt = everything accumulated so far + this round's user message
            prompt_for_round = accumulated + [dict(messages[0])]

            for f in rnd.get("output_files") or []:
                if not isinstance(f, dict):
                    continue
                path = str(f.get("path", "") or "").strip()
                content = f.get("content")
                if not path or not isinstance(content, str):
                    continue
                # Key by basename so cross-unit edits to the same file are grouped.
                key = path.split("/")[-1].split("\\")[-1]
                # Skip data files and binaries — not useful as training completions.
                ext = ("." + key.rsplit(".", 1)[-1].lower()) if "." in key else ""
                if ext in _SKIP_EXTENSIONS:
                    continue
                if key not in file_index:
                    file_index[key] = []
                file_index[key].append({
                    "path": path,
                    "content": content,
                    "prompt": prompt_for_round,
                })

            # Advance context: append this round's assistant+tool messages.
            accumulated.extend(messages[1:])

            # Append human follow-ups that arrive after this round.
            for h in human_traj:
                if h.get("type") == "follow_up" and h.get("round_index") == round_idx:
                    text = h.get("prompt", "")
                    if text:
                        accumulated.append({"role": "user", "content": text})

    return file_index


def extract_dpo_pairs(
    sessions: list[dict],
    renderer: Any | None = None,
    pair_mode: str = "adjacent",
) -> list[dict[str, Any]]:
    """Extract DPO preference pairs from weight-format sessions.

    Both modes use a **cross-unit, file-centric** strategy: the entire session
    is scanned across all task_units, rounds are grouped by output filename
    (basename), and pairs are built from the ordered version history of each
    file. This means a file created in unit 1 and polished in unit 3 is handled
    correctly — the prompt for the first version is the natural
    ``system + initial_user`` context, with no "Proceed with: Verify…" noise.

    Modes (``pair_mode``):

    - ``"first_last"`` (default): one pair per file — rejected = first version
      written, chosen = last version written. Cleanest overfit / sanity setup.

    - ``"adjacent"``: one pair per consecutive version step — rejected = version
      k, chosen = version k+1, prompt = context at the time version k was
      written. Produces more pairs and preserves fine-grained preference signal.

    Completions are artifact-only: each version is reconstructed as a single
    ``write`` tool_call. Pairs where either side has no string content are
    skipped. Files with only one version (no revisions) are skipped.

    Returns list of::

        {prompt, chosen, rejected, chosen_is_agent, rejected_is_agent}
    """
    if pair_mode not in {"adjacent", "first_last"}:
        raise ValueError(
            f"pair_mode must be 'adjacent' or 'first_last', got {pair_mode!r}"
        )

    pairs: list[dict[str, Any]] = []

    for session in sessions:
        file_index = _build_file_version_index(
            session,
            session.get("system_prompt", ""),
            session.get("tool_schemas"),
            renderer,
        )
        for versions in file_index.values():
            if len(versions) < 2:
                continue

            if pair_mode == "first_last":
                candidates = [(versions[0], versions[-1])]
            else:  # adjacent
                candidates = [(versions[k], versions[k + 1]) for k in range(len(versions) - 1)]

            for rej, cho in candidates:
                rej_msgs, rej_is_agent = _build_artifact_completion(
                    [{"path": rej["path"], "content": rej["content"]}],
                )
                cho_msgs, cho_is_agent = _build_artifact_completion(
                    [{"path": cho["path"], "content": cho["content"]}],
                )
                if not rej_msgs or not cho_msgs:
                    continue
                pairs.append({
                    "prompt": list(rej["prompt"]),
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

def _reward_cache_is_binary_rubric_row(values: list[Any]) -> bool:
    """Each entry must be exactly 0 or 1 (float or int), not bool subtype quirks."""
    for x in values:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return False
        xf = float(x)
        if xf != 0.0 and xf != 1.0:
            return False
    return True


def _unit_reward_cache_valid(unit: dict[str, Any], n_rubrics: int) -> bool:
    """True if ``unit['reward']`` is a per-rubric 0/1 list matching ``n_rubrics``."""
    r = unit.get("reward")
    if n_rubrics == 0:
        return r is None or (isinstance(r, list) and len(r) == 0)
    if not isinstance(r, list) or len(r) != n_rubrics:
        return False
    return _reward_cache_is_binary_rubric_row(r)


def _reinforce_prompt_cache_valid(value: Any) -> bool:
    """True if ``value`` looks like a frozen OpenAI-style message list."""
    if not isinstance(value, list) or not value:
        return False
    return all(isinstance(m, dict) and isinstance(m.get("role"), str) for m in value)


def _reinforce_exec_unit_sample_indices(n_exec: int) -> set[int]:
    """Indices (0-based) of execution task_units to keep for REINFORCE: first, middle, last.

    With fewer than three execution units, keeps all that exist (one or two).
    """
    if n_exec <= 0:
        return set()
    if n_exec == 1:
        return {0}
    if n_exec == 2:
        return {0, 1}
    return {0, n_exec // 2, n_exec - 1}


def extract_reinforce_examples(
    sessions: list[dict],
    renderer: Any | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Extract REINFORCE examples from weight-format sessions.

    For each round *k*, the **prompt** is built like DPO file-indexing: a single
    session-level ``accumulated`` transcript (system/tools prefix, then all prior
    rounds' ``messages[1:]`` and follow-ups across **all** task_units in session order),
    plus this round's opening ``messages[0]``. That matches
    :func:`_build_file_version_index` so later units see full prior agent traffic,
    not only the current unit's slice.

    The **completion** is still artifact-only from ``rounds[k].output_files``.

    If a task_unit already has a ``reward`` list of length ``len(session rubrics)``
    (the last task_unit's verifier criteria) with only ``0.0`` / ``1.0`` entries, it is
    reused and no LLM is called. Otherwise scores are computed and ``unit['reward']`` is
    set to that per-rubric binary list (``sessions`` is mutated).

    Each round may store ``reinforce_prompt`` (full message list used as the training
    prompt): on first extract it is computed and written onto ``agent_trajectories[k]``;
    on later loads it is reused so the cross-unit accumulated transcript need not be
    recomputed (invalidate by deleting that field if the source ``messages`` change).

    Each extracted training row still carries scalar ``reward`` = mean of the unit's
    rubric 0/1 scores (same for every trajectory index in that unit under LLM grading).

    Only **three** execution task_units per session emit training data (first, middle,
    and last in execution order), **plus** every task_unit whose ``intent`` is
    ``"planning"``. Other execution units still advance ``accumulated`` but do not call
    the rubric LLM, write ``reward`` / ``reinforce_prompt``, or append examples.

    Rounds without ``output_files`` (no artifact produced) are skipped.

    Returns ``(examples, session_dirty)`` where ``session_dirty`` is True if any
    ``reward`` and/or ``reinforce_prompt`` cache was newly written (caller may persist
    ``sessions`` to disk).

    Each example is::

        {prompt, completion, is_agent, reward}
    """
    examples: list[dict[str, Any]] = []
    session_dirty = False

    for session in sessions:
        system_prompt = session.get("system_prompt", "")
        tool_schemas = session.get("tool_schemas")
        # Rubrics: last task_unit's verifiers. Reward LLM runs for planning + sampled execution units.
        rubrics = session.get("task_units", [])[-1].get("verifiers", [])
        rubrics = [v["criterion"] for v in rubrics]  # list[str]

        accumulated: list[dict] = list(
            _session_tools_prefix(system_prompt, tool_schemas, renderer),
        )

        task_units_list = session.get("task_units", [])
        n_exec_units = sum(1 for u in task_units_list if u.get("intent") != "planning")
        exec_unit_keep = _reinforce_exec_unit_sample_indices(n_exec_units)
        exec_unit_idx = -1

        for unit in task_units_list:
            if unit.get("intent") == "planning":
                include_unit = True
            else:
                exec_unit_idx += 1
                include_unit = exec_unit_idx in exec_unit_keep
            rounds = unit.get("agent_trajectories", [])
            human_traj = unit.get("human_trajectories", [])
            if not rounds:
                continue

            n_rubrics = len(rubrics)
            if include_unit:
                if _unit_reward_cache_valid(unit, n_rubrics):
                    rubric_scores = [float(x) for x in unit["reward"]]
                    mean_r = sum(rubric_scores) / len(rubric_scores) if rubric_scores else 1.0
                    per_traj_rewards = [mean_r] * len(rounds)
                    logger.debug("Using cached rubric 0/1 scores (%d criteria)", len(rubric_scores))
                else:
                    mean_r, rubric_scores = compute_llm_rubric_file_scores(
                        unit, rubrics, model="claude-haiku-4-5-20251001",
                    )
                    per_traj_rewards = [mean_r] * len(rounds)
                    unit["reward"] = [float(x) for x in rubric_scores]
                    session_dirty = True
            else:
                per_traj_rewards = []

            for k, rnd in enumerate(rounds):
                messages = rnd.get("messages", [])
                if not messages:
                    continue
                if include_unit:
                    cached = rnd.get("reinforce_prompt")
                    if _reinforce_prompt_cache_valid(cached):
                        prompt = json.loads(json.dumps(cached))
                    else:
                        prompt = accumulated + [dict(messages[0])]
                        rnd["reinforce_prompt"] = json.loads(json.dumps(prompt))
                        session_dirty = True
                    completion, is_agent = _build_artifact_completion(rnd.get("output_files"))
                    if completion:
                        examples.append({
                            "prompt": prompt,
                            "completion": completion,
                            "is_agent": is_agent,
                            "reward": per_traj_rewards[k],
                        })

                accumulated.extend(messages[1:])
                for h in human_traj:
                    if h.get("type") == "follow_up" and h.get("round_index") == k:
                        text = h.get("prompt", "")
                        if text:
                            accumulated.append({"role": "user", "content": text})

    return examples, session_dirty


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
        default="first_last",
        help=(
            "DPO pair construction (ignored for opd/reinforce). "
            "Both modes scan the whole session by file (cross-unit). "
            "'first_last' = one pair per file, first write vs last write (default). "
            "'adjacent' = one pair per consecutive version step per file."
        ),
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
        units, _ = extract_reinforce_examples(sessions)
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
