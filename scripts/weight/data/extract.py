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
        .output_files               → (Pi export) artifacts changed this round; when
                                       absent, derived from write/edit tool_calls in
                                       ``messages`` for completion construction
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
import difflib
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .reward import (
    _merged_files_after_round,
    compute_llm_rubric_file_scores,
    round_output_files_change_only,
    unit_has_meaningful_rubric_files,
)

logger = logging.getLogger(__name__)

# Max chars of the unified diff injected into OPD teacher prompts (avoid blow-ups).
_MAX_FILE_EDIT_DIFF_CHARS = 2000


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# DEAD CODE — superseded by _build_artifact_completion.
# Kept as a fallback for diagnostic use and in case full-trajectory training
# (non-artifact-only) is ever needed again.
def _completion_and_mask(
    messages: list[dict],
) -> tuple[list[dict], list[bool]]:
    """Return (completion_messages, is_agent_mask) from a round's messages."""
    completion = messages[1:]
    is_agent = [m.get("role") == "assistant" for m in completion]
    return completion, is_agent


# DEAD CODE — kept for potential future use if incomplete-round filtering
# is re-introduced for full-trajectory training modes.
def _is_complete_round(messages: list[dict]) -> bool:
    """Return True if the round ends with a final assistant message."""
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
    gt_content: str | None = None,
    student_artifact: dict[str, str] | None = None,
) -> list[dict]:
    """Append privileged context to the last user message for the teacher prompt.

    Captured action types in ``human_actions``:
    - ``follow_up``     → '- Human feedback: "..."'
    - ``file_edit``     → unified diff block
    - ``edit_workflow`` / ``edit_verifier`` — intentionally not injected (commented out).

    Optional privileged blocks:
    - ``student_artifact``: the student artifact at this round (``use_student=True``).
      This gives the teacher explicit before-state context so it can infer *why*
      follow-up instructions were issued.
    - ``gt_content``: final accepted artifact (``use_gt=True``), i.e. SDFT-style
      golden answer reference.
    """
    parts: list[str] = []
    for action in human_actions:
        atype = action.get("type", "follow_up")
        if atype == "follow_up":
            text = action.get("prompt", "")
            if text:
                parts.append(f'- Human feedback: "{text}"')
        elif atype == "file_edit":
            original = action.get("original") or ""
            edited = action.get("edited") or ""
            if edited.strip():
                diff_lines = list(difflib.unified_diff(
                    original.splitlines(keepends=True),
                    edited.splitlines(keepends=True),
                    fromfile="before",
                    tofile="after",
                    lineterm="",
                ))
                if diff_lines:
                    diff_text = "".join(diff_lines)
                    if len(diff_text) > _MAX_FILE_EDIT_DIFF_CHARS:
                        diff_text = diff_text[:_MAX_FILE_EDIT_DIFF_CHARS] + "\n... [truncated]"
                    parts.append("- Human edited file (diff):\n```diff\n" + diff_text + "\n```")
        # elif atype == "edit_workflow":
        #     parts.append("(workflow edit omitted)")
        # elif atype == "edit_verifier":
        #     parts.append("(verifier edit omitted)")

    if not parts and gt_content is None and student_artifact is None:
        return prompt

    suffix = ""
    if student_artifact is not None:
        student_content = student_artifact.get("content", "")
        if student_content:
            suffix += (
                "\n\nThe following context comes from a prior interaction with the "
                "same user on a related artifact. Use it as evidence of the "
                "user's stable preferences, domain expectations, and quality bar. "
                "Do not copy one-off coordinates or incidental implementation "
                "details unless they clearly generalize.\n"
                "Student artifact from that prior attempt:\n```\n"
                + student_content
                + "\n```"
            )
    if parts:
        suffix += (
            "\n\nHuman feedback from that prior interaction:\n"
            + "\n".join(parts)
        )
    if gt_content is not None:
        suffix += (
            "\n\nOptional final accepted artifact from that prior interaction:\n"
            "```\n" + gt_content + "\n```"
        )
    suffix += (
        "\n\nInfer the reusable preferences behind the feedback: visual style, "
        "layout constraints, annotation quality, data-faithfulness, writing tone, "
        "and any domain-specific standards. Apply those reusable preferences "
        "when judging or improving the current response."
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


def _session_initial_context(
    session: dict,
    system_prompt: str,
    tool_schemas: list[dict] | None,
    renderer: Any | None,
) -> list[dict]:
    """System/tools + initial task + completed planning transcript.

    At inference time execution nodes are not launched from only their local
    "Proceed with: ..." prompt. They also see the user's original task
    instruction and the completed planning interaction (including the concrete
    workflow_plan tool call). Training prompts must preserve that same prefix.
    """
    context = list(_session_tools_prefix(system_prompt, tool_schemas, renderer))

    initial = session.get("initial_task_instruction")
    if isinstance(initial, str) and initial.strip():
        context.append({"role": "user", "content": initial})

    for unit in session.get("task_units", []) or []:
        if unit.get("intent") != "planning":
            continue
        for rnd in unit.get("agent_trajectories", []) or []:
            messages = rnd.get("messages", [])
            if messages:
                context.extend(dict(m) for m in messages)

    return context


def _prompt_has_session_initial_context(prompt: Any, session: dict) -> bool:
    """Return True if a cached prompt already includes the current session head."""
    if not _reinforce_prompt_cache_valid(prompt):
        return False

    initial = session.get("initial_task_instruction")
    if isinstance(initial, str) and initial.strip():
        if not any(m.get("role") == "user" and m.get("content") == initial for m in prompt):
            return False

    planning_units = [
        u for u in session.get("task_units", []) or []
        if u.get("intent") == "planning"
    ]
    if not planning_units:
        return True

    for unit in planning_units:
        for rnd in unit.get("agent_trajectories", []) or []:
            for msg in rnd.get("messages", []) or []:
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    planned_tools = [
                        tc.get("function", {}).get("name")
                        for tc in msg.get("tool_calls", [])
                    ]
                    if "workflow_plan" in planned_tools:
                        return any(
                            m.get("role") == "assistant"
                            and any(
                                tc.get("function", {}).get("name") == "workflow_plan"
                                for tc in (m.get("tool_calls") or [])
                            )
                            for m in prompt
                        )
    return True


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


def _artifact_path_skipped_for_training(path: str) -> bool:
    key = path.split("/")[-1].split("\\")[-1]
    ext = ("." + key.rsplit(".", 1)[-1].lower()) if "." in key else ""
    return ext in _SKIP_EXTENSIONS


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
    accumulated: list[dict] = _session_initial_context(
        session, system_prompt, tool_schemas, renderer,
    )
    file_index: dict[str, list[dict]] = {}
    unit_files: dict[str, str] = {}

    for unit_idx, unit in enumerate(session.get("task_units", [])):
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

            for f in round_output_files_change_only(
                rnd, unit_files, skip_path=_artifact_path_skipped_for_training,
            ):
                path = f["path"]
                content = f["content"]
                key = path.split("/")[-1].split("\\")[-1]
                if key not in file_index:
                    file_index[key] = []
                file_index[key].append({
                    "path": path,
                    "content": content,
                    "prompt": prompt_for_round,
                    # Source coordinates — used by OPD to find future feedback items.
                    "unit_idx": unit_idx,
                    "round_idx": round_idx,
                })
            unit_files = _merged_files_after_round(unit_files, rnd)

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


def extract_dpo_accepted_artifacts(
    sessions: list[dict],
    renderer: Any | None = None,
) -> list[dict[str, Any]]:
    """Extract accepted artifact completions for online DPO.

    Online DPO constructs the rejected side from the current policy's sampled
    rollout. The exported weight JSON therefore only needs to provide the
    accepted side: a prompt plus the artifact ``write(path, content)`` that we
    want preferred over the sampled trajectory.
    """
    rows: list[dict[str, Any]] = []

    for session in sessions:
        file_index = _build_file_version_index(
            session,
            session.get("system_prompt", ""),
            session.get("tool_schemas"),
            renderer,
        )
        for versions in file_index.values():
            for version in versions:
                chosen, chosen_is_agent = _build_artifact_completion(
                    [{"path": version["path"], "content": version["content"]}],
                )
                if not chosen:
                    continue
                rows.append({
                    "prompt": list(version["prompt"]),
                    "chosen": chosen,
                    "chosen_is_agent": chosen_is_agent,
                    "expected_path": version["path"],
                })

    return rows


def extract_dpo_final_artifacts(
    sessions: list[dict],
    renderer: Any | None = None,
    min_versions: int = 1,
) -> list[dict[str, Any]]:
    """Extract per-session rollout seeds + final accepted artifacts for agentic DPO.

    Unlike :func:`extract_dpo_accepted_artifacts` (one flat row per file
    version), this groups by session so the online trainer can run a single
    agentic rollout per session and form one DPO pair per artifact:

    * The **chosen** side is the *final* version of each file
      (``versions[-1]`` from :func:`_build_file_version_index`), i.e. the
      accepted artifact after all user follow-ups, scored under that version's
      accumulated-context prompt -- identical formatting to
      :func:`extract_dpo_accepted_artifacts`.
    * The **rollout seed** is the session's initial task (no privileged
      follow-ups), mirroring :func:`extract_reinforce_rollout_seeds`. The
      current policy generates the rejected artifact live at train time.

    Files with fewer than ``min_versions`` versions are skipped (set
    ``min_versions=2`` to restrict to files actually revised after follow-ups).

    Returns one row per session::

        {system_prompt, tool_schemas, prompt_messages, chosen_artifacts, meta}

    where ``chosen_artifacts`` is a list of
    ``{expected_path, basename, chosen, prompt}``.
    """
    rows: list[dict[str, Any]] = []

    for session in sessions:
        system_prompt = session.get("system_prompt", "") or ""
        tool_schemas = session.get("tool_schemas")

        initial_task = session.get("initial_task_instruction")
        if not (isinstance(initial_task, str) and initial_task.strip()):
            initial_task = _first_user_message_text(session)
        if not (isinstance(initial_task, str) and initial_task.strip()):
            # No user anchor to seed the rollout; skip this session.
            continue

        file_index = _build_file_version_index(
            session,
            system_prompt,
            tool_schemas,
            renderer,
        )
        chosen_artifacts: list[dict[str, Any]] = []
        for basename, versions in file_index.items():
            if len(versions) < max(1, min_versions):
                continue
            final = versions[-1]
            chosen, _chosen_is_agent = _build_artifact_completion(
                [{"path": final["path"], "content": final["content"]}],
            )
            if not chosen:
                continue
            # First-written version content (the human's initial draft). Used as
            # an optional offline "first_last" rollout: the first version becomes a
            # synthetic rejected snapshot paired against this final/chosen artifact.
            # Only meaningful when the file was actually revised (>=2 versions).
            first_content = (
                versions[0].get("content") if len(versions) >= 2 else None
            )
            chosen_artifacts.append({
                "expected_path": final["path"],
                "basename": basename,
                "chosen": chosen,
                "prompt": list(final["prompt"]),
                "first_content": first_content,
            })

        if not chosen_artifacts:
            continue

        rows.append({
            "system_prompt": system_prompt,
            "tool_schemas": tool_schemas,
            "prompt_messages": [{"role": "user", "content": initial_task.strip()}],
            "chosen_artifacts": chosen_artifacts,
            "meta": {
                "session_uuid": session.get("uuid"),
                "n_artifacts": len(chosen_artifacts),
            },
        })

    return rows


# ---------------------------------------------------------------------------
# OPD extraction (offline)
# ---------------------------------------------------------------------------

def _extract_opd_examples_legacy(
    sessions: list[dict],
    renderer: Any | None = None,
) -> list[dict[str, Any]]:
    """Legacy unit-local OPD extractor (one example per round within a task_unit).

    Kept for reference and in case full-trajectory / non-artifact-only OPD
    training is needed. The active extractor is :func:`extract_opd_examples`.

    For each task_unit with >=2 rounds and human feedback, produces one example
    per round k in [0, N-2]:
    - student_prompt: accumulated context up to round k (unit-local)
    - teacher_prompt: student_prompt + all future human feedback in this unit
    - completion: round k's artifact-only completion
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


def extract_opd_examples(
    sessions: list[dict],
    renderer: Any | None = None,
    pair_mode: str = "first_last",
    use_gt: bool = False,
    use_student: bool = False,
) -> list[dict[str, Any]]:
    """Extract offline OPD examples using the file-centric cross-unit strategy.

    Mirrors the DPO file-centric design: the full session is scanned via
    :func:`_build_file_version_index`, rounds are grouped by output filename
    (basename), and OPD examples are built from the ordered version history.

    Modes (``pair_mode``):

    - ``"first_last"`` (default): one example per file.
      Student = first version written (v0); teacher has **all** future human
      feedback from v0's round onwards (cross-unit).

    - ``"adjacent"``: one example per consecutive version step.
      Student = version k (v_k); teacher has feedback arriving strictly after
      v_k's source round. More examples, smaller per-example teacher signal.

    When ``use_student=True``, the student artifact for the current example is
    also injected into the teacher prompt as explicit "before-state" context.
    This is stronger than feedback-only prompting and helps the teacher infer
    why follow-ups were issued.

    When ``use_gt=True`` the **last** version's content is appended to every
    teacher prompt as a "final accepted version" reference block (SDFT
    golden-answer conditioning).

    Teacher prompt construction: ALL human action types are included:
    - ``follow_up`` → "Human feedback: \"...\""
    - ``file_edit`` → "Human edited file '...'"
    - ``edit_workflow`` / ``edit_verifier`` — intentionally not injected.

    Cross-unit feedback: for each version v at (unit_idx=U, round_idx=R),
    future feedback = actions with round_index ≥ R in unit U, plus ALL actions
    in any subsequent unit (unit_idx > U).  This is the recommended "envelope"
    so the teacher sees revision context from later task units that polished the
    same file.

    Returns list of::

        {student_prompt, teacher_prompt, completion, is_agent}
    """
    if pair_mode not in {"first_last", "adjacent"}:
        raise ValueError(
            f"pair_mode must be 'first_last' or 'adjacent', got {pair_mode!r}"
        )

    examples: list[dict[str, Any]] = []

    for session in sessions:
        file_index = _build_file_version_index(
            session,
            session.get("system_prompt", ""),
            session.get("tool_schemas"),
            renderer,
        )

        # Flatten all human feedback across the session with position info.
        # round_index == -1 means the action is not tied to a specific round
        # (e.g. edit_workflow that spans the whole unit).
        all_feedback: list[dict] = []
        for unit_idx, unit in enumerate(session.get("task_units", [])):
            if unit.get("intent") == "planning":
                continue
            for h in unit.get("human_trajectories", []) or []:
                all_feedback.append({
                    "unit_idx": unit_idx,
                    "round_index": h.get("round_index", -1),
                    "action": h,
                })

        for versions in file_index.values():
            if len(versions) < 2:
                continue

            gt_content = versions[-1]["content"] if use_gt else None

            # Which versions become the student completion?
            if pair_mode == "first_last":
                student_versions = [versions[0]]
            else:  # adjacent
                student_versions = versions[:-1]

            for v in student_versions:
                v_unit = v["unit_idx"]
                v_round = v["round_idx"]

                # Collect future human feedback (cross-unit envelope).
                future_actions = [
                    item["action"]
                    for item in all_feedback
                    if (item["unit_idx"] > v_unit)
                    or (item["unit_idx"] == v_unit and item["round_index"] >= v_round)
                ]

                if not future_actions and not use_gt:
                    continue  # no teacher signal for this version

                completion, is_agent = _build_artifact_completion(
                    [{"path": v["path"], "content": v["content"]}],
                )
                if not completion:
                    continue

                student_prompt = list(v["prompt"])
                student_artifact = (
                    {"path": v["path"], "content": v["content"]}
                    if use_student else None
                )
                teacher_prompt = _augment_with_feedback(
                    student_prompt,
                    future_actions,
                    gt_content,
                    student_artifact=student_artifact,
                )

                examples.append({
                    "student_prompt": student_prompt,
                    "teacher_prompt": teacher_prompt,
                    "completion": completion,
                    "is_agent": is_agent,
                })

    return examples


# ---------------------------------------------------------------------------
# OPD ver 2
# ---------------------------------------------------------------------------
_HUMAN_ACTION_TYPES = (
    "follow_up",
    "file_edit",
    "brain_edit",
    "edit_workflow",
    "edit_verifier",
)


OPD_REDO_MESSAGE = (
    "The user messages above are follow-ups from a previous session based "
    "on the given chat history. Please think about the reason why the user "
    "asked these specific follow-ups, and reason through the user "
    "preferences reflected by them. Then, provide a response to the "
    "following user request that incorporates the feedback."
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


def _followups_to_golden_chat(human_actions: list[dict]) -> list[dict]:
    """Convert normalized cumulative human actions into a chat-form golden_answer.
    Each item becomes a user-roled message:
    - ``follow_up`` → verbatim user turn with the human's text.
    - ``file_edit`` → synthetic user turn containing a unified diff between
      the assistant's text (``ai``) and the human-edited version (``edited``).
      Truncated at :data:`_MAX_FILE_EDIT_DIFF_CHARS`.
    - ``brain_edit``, ``edit_workflow``, ``edit_verifier`` → terse synthetic
      user notes so the teacher knows non-text user actions occurred.
    Items with no usable payload are skipped. Returns ``[]`` when there are
    no actionable entries.
    """
    out: list[dict] = []
    for action in human_actions:
        atype = action.get("type")
        if atype == "follow_up":
            text = action.get("prompt", "")
            if text:
                out.append({"role": "user", "content": text})
        elif atype == "file_edit":
            path = action.get("path") or "<file>"
            ai_text = action.get("ai") or ""
            edited = action.get("edited") or ""
            if edited.strip():
                diff_lines = list(difflib.unified_diff(
                    ai_text.splitlines(keepends=True),
                    edited.splitlines(keepends=True),
                    fromfile=f"{path} (ai)",
                    tofile=f"{path} (human-edited)",
                    lineterm="",
                ))
                if diff_lines:
                    diff_text = "".join(diff_lines)
                    if len(diff_text) > _MAX_FILE_EDIT_DIFF_CHARS:
                        diff_text = diff_text[:_MAX_FILE_EDIT_DIFF_CHARS] + "\n... [truncated]"
                    out.append({
                        "role": "user",
                        "content": f"I edited '{path}':\n```diff\n{diff_text}\n```",
                    })
        elif atype == "brain_edit":
            parts: list[str] = []
            if action.get("memory"):
                parts.append("memory")
            if action.get("skill"):
                parts.append("skill")
            if parts:
                out.append({
                    "role": "user",
                    "content": f"I edited my {' and '.join(parts)}.",
                })
        elif atype == "edit_workflow":
            out.append({"role": "user", "content": "I edited the workflow plan."})
        elif atype == "edit_verifier":
            crit = action.get("criterion") or ""
            content = "I edited a verifier criterion"
            if crit:
                content += f": {crit}"
            out.append({"role": "user", "content": content + "."})
    return out


def _build_teacher_prompt_chat(
    student_prompt: list[dict],
    golden_chat: list[dict],
    redo_message: str = OPD_REDO_MESSAGE,
) -> list[dict]:
    """Mirror :func:`build_sdft_teacher_prompt` (chat-list branch) but return
    a ``list[dict]`` instead of a ``tinker.ModelInput``.
    Tokenization is left to the downstream dataset (``conversation_to_datum``
    in :mod:`weight.train.formatter`), preserving v1's four-key contract.
    Pattern: ``head + golden_chat + [redo_user] + trailing_user``, where
    ``trailing_user`` is the LAST message peeled off ``student_prompt``.
    For ``sub_index == 0`` sub-units that message is the hoisted user task
    instruction (the intended case). For continuation sub-units it may be
    a tool result or an assistant turn; the teacher will still see it in
    the correct chronological position, with the golden context sitting
    just before it.
    When ``golden_chat`` is empty the student prompt is returned unchanged
    (no privileged signal → no need to splice).
    """
    if not golden_chat:
        return list(student_prompt)
    head = list(student_prompt)
    trailing: list[dict] = []
    if head:
        trailing.insert(0, head.pop())
    return head + list(golden_chat) + [{"role": "user", "content": redo_message}] + trailing


def extract_opd_examples_v2(
    sessions: list[dict],
    renderer: Any | None = None,
    pair_mode: str = "first_last",  # unused; accepted for API parity with v1
    use_gt: bool = False,            # unused; accepted for API parity with v1
    use_student: bool = False,       # unused; accepted for API parity with v1
    *,
    redo_message: str = OPD_REDO_MESSAGE,
    skip_empty_golden: bool = True,  # unused; always True for legacy parity
) -> list[dict[str, Any]]:
    """One example per non-continuation task unit (legacy-parity extraction).

    Matches the data-building semantics of the legacy
    ``export_opd_data._session`` + ``tinker_formatter._opd_build_unit`` pipeline
    exactly: one ``learning_unit`` per task's first assistant turn (continuation
    sub-units within the same task are dropped), with ``golden_answer`` taken
    from the cumulative ``follow_up`` actions (or an LLM-generated ``summary``
    when present). Other human-action types (``file_edit``, ``brain_edit``,
    ``edit_workflow``, ``edit_verifier``) are intentionally excluded from the
    golden chat for legacy parity.

    Args
    ----
    sessions: Weight-format session dicts.
    renderer: Used only by :func:`_session_tools_prefix` to render the
        system + tool-schemas prefix in the model's native chat-template
        form (e.g. Qwen3 ``# Tools / <tools>``). Falls back to a plain
        system message when ``None`` or unsupported.
    redo_message: Wording of the synthetic user message appended after
        ``golden_answer`` (see :data:`OPD_REDO_MESSAGE`).
    skip_empty_golden: Retained for API back-compat; ignored. Empty-golden
        units are always dropped to match legacy ``_opd_build_unit``
        (``if not teacher_demo: return None``).
    """
    examples: list[dict[str, Any]] = []
    for session in sessions:
        system_prompt = session.get("system_prompt", "") or ""
        tool_schemas = session.get("tool_schemas")
        session_uuid = session.get("uuid")
        # Match legacy ``export_opd_data._session``: ``prev_msgs`` starts empty.
        # The renderer-rendered system+tools prefix is prepended per-unit at
        # emit time below (legacy attaches ``[system]`` at unit-build time and
        # ``tool_schemas`` via the downstream renderer; we inline both via
        # ``_session_tools_prefix`` to produce equivalent tokens).
        # The session-level ``initial_task_instruction`` is intentionally NOT
        # injected: the first task's ``agent_trajectories[0].messages[0]`` is
        # the leading user task message, which the splitter hoists into
        # ``user_messages`` of the first sub-unit.
        tools_prefix: list[dict] = list(_session_tools_prefix(
            system_prompt, tool_schemas, renderer,
        ))
        prev_msgs: list[dict] = []
        tasks = session.get("task_units") or []
        per_task_humans = _normalized_humans_per_task(tasks)
        cumulative_humans = _cumulative_humans_from(per_task_humans)
        global_idx = 0
        for task_idx, task in enumerate(tasks):
            trajs = task.get("agent_trajectories") or []
            if not trajs:
                continue
            task_intent = task.get("intent")
            first_msgs = trajs[0].get("messages") or []
            tail_humans = cumulative_humans[task_idx]
            new_units, _ = _split_initial_trajectory_into_units(
                first_msgs,
                base_history=list(prev_msgs),
                task_intent=task_intent,
                task_index=task_idx,
                followup_actions=tail_humans,
                next_index=global_idx,
            )
            global_idx += len(new_units)
            for unit in new_units:
                # --- Legacy ``_opd_build_unit`` semantics (exact parity) ---
                # 1) Drop continuation sub-units: their student prompt would
                #    end with a tool result rather than a user request.
                if unit.get("is_continuation"):
                    continue
                history = unit.get("history") or []
                # 2) Truthiness-filter user_messages.
                user_messages = [m for m in (unit.get("user_messages") or []) if m]
                # 3) Drop if there's no user message to anchor the prompt
                #    AND non-empty history (mid-task units without a user
                #    anchor have no clean prediction position).
                if not user_messages and history:
                    continue
                # 4) Build golden_answer: prefer LLM-generated ``summary``
                #    (assistant turn), else cumulative ``follow_up`` actions
                #    only (each becomes a user turn). Other human-action
                #    types are intentionally excluded for legacy parity.
                summary = unit.get("summary")
                if isinstance(summary, str) and summary.strip():
                    golden_chat: list[dict] = [
                        {"role": "assistant", "content": summary.strip()}
                    ]
                else:
                    followup_actions = unit.get("followup_actions") or []
                    followup_texts = [
                        a["prompt"].strip()
                        for a in followup_actions
                        if isinstance(a, dict)
                        and a.get("type") == "follow_up"
                        and isinstance(a.get("prompt"), str)
                        and a["prompt"].strip()
                    ]
                    golden_chat = [
                        {"role": "user", "content": text}
                        for text in followup_texts
                    ]
                # 5) Drop if no teacher demo signal.
                if not golden_chat:
                    continue
                # 6) Structural sanity: need at least a user anchor, non-empty
                #    history, or an assistant turn in the demo.
                if (
                    not user_messages
                    and not history
                    and not any(m.get("role") == "assistant" for m in golden_chat)
                ):
                    continue
                # 7) Build student_prompt = [system+tools] + history +
                #    [user(um) for um in user_messages]. Matches legacy
                #    ``[system] + history + user_messages`` with tools folded
                #    into the system prefix (legacy attaches them via the
                #    renderer downstream; tokens are equivalent).
                student_prompt: list[dict] = list(tools_prefix)
                student_prompt.extend(history)
                student_prompt.extend(
                    {"role": "user", "content": um} for um in user_messages
                )
                teacher_prompt = _build_teacher_prompt_chat(
                    student_prompt, golden_chat, redo_message=redo_message,
                )
                completion = [dict(m) for m in unit["response_messages"]]
                examples.append({
                    "student_prompt": student_prompt,
                    "teacher_prompt": teacher_prompt,
                    "completion": completion,
                    "is_agent": [True] * len(completion),
                    "golden_answer": golden_chat,
                    "meta": {
                        "session_uuid": session_uuid,
                        "task_intent": task_intent,
                        "task_index": task_idx,
                        "sub_index": unit["sub_index"],
                        "is_continuation": unit["is_continuation"],
                        "global_index": unit["index"],
                        "uses_summary": isinstance(summary, str) and bool(summary.strip()),
                    },
                })
            # Advance running history through every trajectory in this task
            # (initial + corrections) before moving on to the next task.
            for traj in trajs:
                prev_msgs.extend(traj.get("messages") or [])
    return examples


def _first_user_message_text(session: dict) -> str | None:
    """Leading user-turn text across the session's agent trajectories."""
    for task in session.get("task_units") or []:
        for traj in task.get("agent_trajectories") or []:
            for m in traj.get("messages") or []:
                if isinstance(m, dict) and m.get("role") == "user":
                    content = m.get("content")
                    if isinstance(content, str) and content.strip():
                        return content
    return None


def extract_opd_examples_agentic(
    sessions: list[dict],
    renderer: Any | None = None,  # noqa: ARG001 - accepted for API parity
    *,
    redo_message: str = OPD_REDO_MESSAGE,  # noqa: ARG001 - applied downstream
) -> list[dict[str, Any]]:
    """One example per session for agentic on-policy OPD.

    Unlike v1/v2 (which emit a historical completion per task unit), the
    agentic record carries only the *seed* for an on-policy rollout: the
    session's initial task plus its ``system_prompt`` and ``tool_schemas``.
    The student trajectory is generated live at train time (multi-turn, with
    live tool results); the teacher prompt splices the session's cumulative
    user follow-ups (the privileged signal) before the restated initial task,
    mirroring v2's redo paradigm (see :func:`_build_teacher_prompt_chat` and
    :data:`OPD_REDO_MESSAGE`).

    Fields per record:
      - ``prompt_messages``: ``[{"role": "user", "content": initial_task}]``
        (no tools prefix; the rollout/teacher builders inject it from
        ``system_prompt`` + ``tool_schemas`` so advertised schemas match the
        chat template the model was trained under).
      - ``system_prompt`` / ``tool_schemas``: passthrough from the session.
      - ``golden_chat``: cumulative ``follow_up`` user turns for the teacher.
      - ``meta``: ``{session_uuid, n_followups}``.

    ``redo_message`` is accepted for API parity but applied downstream when the
    teacher messages are assembled post-rollout.
    """
    examples: list[dict[str, Any]] = []
    for session in sessions:
        system_prompt = session.get("system_prompt", "") or ""
        tool_schemas = session.get("tool_schemas")
        session_uuid = session.get("uuid")

        initial_task = session.get("initial_task_instruction")
        if not (isinstance(initial_task, str) and initial_task.strip()):
            initial_task = _first_user_message_text(session)
        if not (isinstance(initial_task, str) and initial_task.strip()):
            # No user anchor to seed the rollout; skip this session.
            continue

        tasks = session.get("task_units") or []
        per_task_humans = _normalized_humans_per_task(tasks)
        cumulative_humans = _cumulative_humans_from(per_task_humans)
        all_followups = cumulative_humans[0] if cumulative_humans else []
        followup_texts = [
            a["prompt"].strip()
            for a in all_followups
            if isinstance(a, dict)
            and a.get("type") == "follow_up"
            and isinstance(a.get("prompt"), str)
            and a["prompt"].strip()
        ]
        golden_chat = [{"role": "user", "content": text} for text in followup_texts]
        # No follow-ups => no privileged signal to distil toward (distilling the
        # student toward the frozen base on an identical prompt is a no-op at
        # best). Drop, mirroring v2's empty-golden filtering.
        if not golden_chat:
            continue

        examples.append({
            "prompt_messages": [{"role": "user", "content": initial_task.strip()}],
            "system_prompt": system_prompt,
            "tool_schemas": tool_schemas,
            "golden_chat": golden_chat,
            "meta": {
                "session_uuid": session_uuid,
                "n_followups": len(golden_chat),
            },
        })
    return examples


def extract_reinforce_rollout_seeds(
    sessions: list[dict],
    renderer: Any | None = None,  # noqa: ARG001 - API parity with other extractors
) -> list[dict[str, Any]]:
    """One rollout seed per session for on-policy REINFORCE (no teacher / follow-ups).

    Same live multi-turn agentic rollout as OPD agentic; reward is computed after
    the episode from the sandbox filesystem (see ``compute_llm_rubric_file_blocks``).
    """
    examples: list[dict[str, Any]] = []
    for session in sessions:
        system_prompt = session.get("system_prompt", "") or ""
        tool_schemas = session.get("tool_schemas")

        initial_task = session.get("initial_task_instruction")
        if not (isinstance(initial_task, str) and initial_task.strip()):
            initial_task = _first_user_message_text(session)
        if not (isinstance(initial_task, str) and initial_task.strip()):
            continue

        task_units = session.get("task_units") or []
        verifiers = (task_units[-1].get("verifiers") or []) if task_units else []
        rubrics = [
            str(v["criterion"])
            for v in verifiers
            if isinstance(v, dict) and v.get("criterion")
        ]

        examples.append({
            "prompt_messages": [{"role": "user", "content": initial_task.strip()}],
            "system_prompt": system_prompt,
            "tool_schemas": tool_schemas,
            "rubrics": rubrics,
            "meta": {"session_uuid": session.get("uuid"), "n_rubrics": len(rubrics)},
        })
    return examples


# ---------------------------------------------------------------------------
# REINFORCE extraction
# ---------------------------------------------------------------------------
# TODO (future): migrate to the file-centric approach used by DPO and OPD.
#   Per-round rewards keyed by (unit, round_idx) would better match each training
#   row's artifact delta. For now, REINFORCE uses one LLM rubric grade per included
#   execution unit, grading the cumulative session file snapshot at unit end.

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


def extract_reinforce_examples(
    sessions: list[dict],
    renderer: Any | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Extract REINFORCE examples from weight-format sessions.

    For each round *k*, the **prompt** is built like DPO file-indexing: a single
    session-level ``accumulated`` transcript (system/tools prefix, top-level
    ``initial_task_instruction``, full planning transcript, then all prior
    execution rounds' ``messages[1:]`` and follow-ups across **all** task_units
    in session order), plus this round's opening ``messages[0]``. That matches
    :func:`_build_file_version_index` so later units see full prior agent
    traffic, not only the current unit's slice.

    The **completion** is artifact-only from ``rounds[k].output_files`` when present,
    else from write/edit tool_calls in ``messages`` (see ``round_output_files_change_only``).

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

    Every non-planning execution task_unit is eligible. A unit is included only when it
    changes the session file snapshot (writes or edits under ``prior_files``), so the
    rubric LLM grades non-empty cumulative artifacts. Units that only run bash or read
    files still advance ``accumulated`` but do not call the rubric LLM, write
    ``reward`` / ``reinforce_prompt``, or append examples.

    Rounds without a trainable artifact completion are skipped.

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
        # Rubrics: last task_unit's verifiers.
        rubrics = session.get("task_units", [])[-1].get("verifiers", [])
        rubrics = [v["criterion"] for v in rubrics]  # list[str]

        accumulated: list[dict] = _session_initial_context(
            session, system_prompt, tool_schemas, renderer,
        )

        task_units_list = session.get("task_units", [])
        session_files: dict[str, str] = {}

        for unit in task_units_list:
            if unit.get("intent") == "planning":
                # Planning has already been included in the session head.
                continue
            rounds = unit.get("agent_trajectories", [])
            human_traj = unit.get("human_trajectories", [])
            if not rounds:
                continue

            files_before_unit = dict(session_files)
            include_unit = unit_has_meaningful_rubric_files(files_before_unit, unit)
            n_rubrics = len(rubrics)
            if include_unit:
                if _unit_reward_cache_valid(unit, n_rubrics):
                    rubric_scores = [float(x) for x in unit["reward"]]
                    mean_r = sum(rubric_scores) / len(rubric_scores) if rubric_scores else 1.0
                    per_traj_rewards = [mean_r] * len(rounds)
                    logger.debug("Using cached rubric 0/1 scores (%d criteria)", len(rubric_scores))
                else:
                    mean_r, rubric_scores = compute_llm_rubric_file_scores(
                        unit,
                        rubrics,
                        prior_files=files_before_unit,
                        model="claude-haiku-4-5-20251001",
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
                    if _prompt_has_session_initial_context(cached, session):
                        prompt = json.loads(json.dumps(cached))
                    else:
                        prompt = accumulated + [dict(messages[0])]
                        rnd["reinforce_prompt"] = json.loads(json.dumps(prompt))
                        session_dirty = True
                    round_files = round_output_files_change_only(
                        rnd,
                        session_files,
                        skip_path=_artifact_path_skipped_for_training,
                    )
                    completion, is_agent = _build_artifact_completion(round_files)
                    if completion:
                        examples.append({
                            "prompt": prompt,
                            "completion": completion,
                            "is_agent": is_agent,
                            "reward": per_traj_rewards[k],
                        })

                session_files = _merged_files_after_round(session_files, rnd)
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
            "Pair construction mode for DPO and OPD (ignored for reinforce). "
            "Both modes scan the whole session by file (cross-unit). "
            "'first_last' = one pair per file, first write vs last write (default). "
            "'adjacent' = one pair per consecutive version step per file."
        ),
    )
    parser.add_argument(
        "--use-gt",
        action="store_true",
        help=(
            "OPD only: append the last version of each file to the teacher "
            "prompt as a ground-truth reference (SDFT golden-answer trick). "
            "Ignored for DPO and reinforce."
        ),
    )
    parser.add_argument(
        "--use-student",
        action="store_true",
        help=(
            "OPD only: inject the student artifact itself into the teacher "
            "prompt as explicit before-state context. Can be combined with "
            "--use-gt. Ignored for DPO and reinforce."
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
        units = extract_opd_examples(
            sessions,
            pair_mode=args.pair_mode,
            use_gt=args.use_gt,
            use_student=args.use_student,
        )
        print(
            f"Extracted {len(units)} OPD examples (pair_mode={args.pair_mode}, "
            f"use_gt={args.use_gt}, use_student={args.use_student})"
        )
        for i, u in enumerate(units):
            print(f"\n── Example {i} ──")
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
