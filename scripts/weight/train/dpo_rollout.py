"""On-policy agentic rollout for DPO (isolated from ``run_opd``).

Mirrors the OPD / REINFORCE agentic episode driver but returns the final
sandbox filesystem snapshot (the student's artifacts) instead of grading or
building datums. The DPO trainer matches the snapshot to the chosen file by
basename to construct the rejected artifact, then trains only on that artifact
write (not the intermediate tool calls).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import tinker
from tinker_cookbook import renderers

try:
    from weight.data.extract import _session_tools_prefix
except ModuleNotFoundError:  # pragma: no cover - depends on invocation cwd
    from ..data.extract import _session_tools_prefix

from .run_opd import (
    _agentic_messages_to_log,
    _extract_workflow_plan_tasks,
    _message_text_for_log,
    _plan_step_prompts,
    _tool_calls_for_log,
)
from .tool_rollout_env import (
    FileToolset,
    SandboxAgentToolEnv,
    WorkspaceSandbox,
    zero_reward,
)

logger = logging.getLogger(__name__)


def _raw_trajectory_for_log(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dump a rollout history with FULL (untruncated) content plus char counts.

    Unlike :func:`_agentic_messages_to_log` (which truncates each field for
    compact transcripts), this keeps the raw content so we can inspect exactly
    what is inflating the prompt. ``content_chars`` / ``arguments_chars`` make it
    easy to spot the offending turn/tool output without reading every field.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        text = _message_text_for_log(m.get("content")) or ""
        entry: dict[str, Any] = {"role": role, "content_chars": len(text)}
        if text:
            entry["content"] = text
        tcs = _tool_calls_for_log(m)
        if tcs:
            entry["tool_calls"] = [
                {
                    "name": tc["name"],
                    "arguments_chars": len(tc["arguments"]),
                    "arguments": tc["arguments"],
                }
                for tc in tcs
            ]
        if role == "tool" and m.get("name"):
            entry["name"] = m.get("name")
        out.append(entry)
    return out


async def rollout_one_dpo_episode(
    row: dict[str, Any],
    renderer: renderers.Renderer,
    sampling_client: tinker.SamplingClient,
    *,
    max_tokens: int,
    temperature: float,
    max_turns: int,
    max_turns_per_step: int,
    max_steps: int,
    enable_bash: bool,
    tool_timeout_s: int,
    max_trajectory_tokens: int | None,
    collect_transcript: bool = False,
    collect_raw_trajectory: bool = False,
    log_field_chars: int = 2000,
) -> tuple[dict[str, str], dict[str, float], dict[str, Any] | None]:
    """Run one multi-turn tool-using rollout and return the sandbox snapshot.

    The episode is a sequence of user-turn *segments*: a planning segment (where
    the model calls ``workflow_plan`` on-policy) followed by one segment per
    planned leaf step (a synthesized "Proceed with: ..." user turn). Each
    segment runs the inner agent loop -- sample -> parse -> execute tools -- until
    the model emits an assistant turn with no tool calls or the per-step turn
    cap is hit. All segments share one sandbox.

    Returns ``(snapshot, metrics, optional_transcript_log)`` where ``snapshot``
    maps ``relpath -> content`` for every text file in the final sandbox state.
    """
    system_prompt = row.get("system_prompt", "") or ""
    tool_schemas = row.get("tool_schemas")
    tools_prefix = list(_session_tools_prefix(system_prompt, tool_schemas, renderer))
    prompt_messages = list(row["prompt_messages"])
    initial_messages = tools_prefix + prompt_messages

    sandbox = WorkspaceSandbox(bash_timeout_s=tool_timeout_s)
    metrics_local: dict[str, float] = {}
    try:
        toolset = FileToolset(
            sandbox, enable_bash=enable_bash, bash_timeout_s=tool_timeout_s
        )
        msg_env = SandboxAgentToolEnv(
            tools=toolset.tools(),
            initial_messages=initial_messages,
            max_turns=max_turns,
            reward_fn=zero_reward,
            sandbox=sandbox,
        )
        history = await msg_env.initial_observation()
        stop = renderer.get_stop_sequences()
        n_turns = 0
        n_tool_calls = 0
        parse_failed = False
        overflow = False
        sample_error: str | None = None  # hard sampler error (e.g. context overflow)
        # Token accounting for the rollout (diagnoses context-window overflows).
        max_prompt_tokens = 0  # peak prompt length fed to the sampler
        gen_tokens = 0         # total tokens generated across all turns

        async def run_segment() -> None:
            nonlocal n_turns, n_tool_calls, parse_failed, overflow, history
            nonlocal max_prompt_tokens, gen_tokens, sample_error
            for _ in range(max_turns_per_step):
                if msg_env._turn_count >= max_turns:
                    return
                prompt_input = await asyncio.to_thread(
                    renderer.build_generation_prompt, history
                )
                if prompt_input.length > max_prompt_tokens:
                    max_prompt_tokens = prompt_input.length
                if (
                    max_trajectory_tokens is not None
                    and prompt_input.length >= max_trajectory_tokens
                ):
                    overflow = True
                    return
                try:
                    result = await sampling_client.sample_async(
                        prompt=prompt_input,
                        num_samples=1,
                        sampling_params=tinker.SamplingParams(
                            stop=stop,
                            max_tokens=max_tokens,
                            temperature=temperature,
                        ),
                    )
                except Exception as e:  # noqa: BLE001 - keep the batch alive; record + skip
                    # A hard sampler error (most commonly the prompt exceeding the
                    # model context window) must not abort the whole batch. Record
                    # it so the finished-rollout log captures the offending token
                    # count, then end this rollout gracefully.
                    sample_error = f"{type(e).__name__}: {e}"
                    logger.warning(
                        "dpo agentic rollout sample failed (prompt_tokens=%d): %s",
                        prompt_input.length, sample_error,
                    )
                    return
                tokens = list(result.sequences[0].tokens)
                gen_tokens += len(tokens)
                message, ok = renderer.parse_response(tokens)
                if not ok:
                    parse_failed = True
                    return
                n_turns += 1
                tool_calls = message.get("tool_calls") or []
                n_tool_calls += len(tool_calls)
                if logger.isEnabledFor(logging.DEBUG):
                    preview = _message_text_for_log(
                        message.get("content")
                    )[:200].replace("\n", " ")
                    logger.debug(
                        "dpo agentic turn=%d tools=%s content=%r",
                        n_turns,
                        [b["name"] for b in _tool_calls_for_log(message)],
                        preview,
                    )
                step_result = await msg_env.step(message)
                history = step_result.next_messages
                if not tool_calls:
                    # Final (no-tool) assistant turn => this segment is done.
                    return
                if msg_env._should_stop:
                    return

        # Segment 0: planning. The model registers a workflow_plan on-policy.
        await run_segment()

        # Derive step queries from the model's OWN plan and replay them as
        # subsequent user turns in the same sandbox.
        n_plan_steps = 0
        if not parse_failed and not overflow and not sample_error:
            tasks = _extract_workflow_plan_tasks(history)
            step_prompts = (
                _plan_step_prompts(tasks, sandbox.root, max_steps) if tasks else []
            )
            if not step_prompts:
                metrics_local["agentic/no_plan"] = 1.0
            for step_prompt in step_prompts:
                if msg_env._turn_count >= max_turns:
                    break
                history.append({"role": "user", "content": step_prompt})
                n_plan_steps += 1
                await run_segment()
                if parse_failed or overflow or sample_error:
                    break

        if overflow:
            metrics_local["agentic/context_overflow"] = 1.0

        # Capture the final filesystem state BEFORE the sandbox is torn down.
        snapshot = sandbox.snapshot()

        # Final trajectory length (tokens of the full rollout history). This is
        # the token footprint of the finished data sample; ``max_prompt_tokens``
        # is the peak the sampler actually saw (what trips the context window).
        final_prompt = await asyncio.to_thread(renderer.build_generation_prompt, history)
        final_prompt_tokens = int(final_prompt.length)

        def _episode_log(drop_reason: str | None, *, valid: bool) -> dict[str, Any] | None:
            if not collect_transcript and not collect_raw_trajectory:
                return None
            log: dict[str, Any] = {
                "drop_reason": drop_reason,
                "valid": valid,
                "n_turns": n_turns,
                "n_steps": n_plan_steps,
                "n_tool_calls": n_tool_calls,
                "max_prompt_tokens": int(max_prompt_tokens),
                "final_prompt_tokens": final_prompt_tokens,
                "gen_tokens": int(gen_tokens),
                "sample_error": sample_error,
            }
            if collect_transcript:
                log["messages"] = _agentic_messages_to_log(
                    history, max_field_chars=log_field_chars
                )
            if collect_raw_trajectory:
                # Untruncated per-message dump (+ per-message char counts) so we
                # can pinpoint which turn/tool output is bloating the prompt.
                log["raw_messages"] = _raw_trajectory_for_log(history)
            return log

        metrics_local.update({
            "agentic/turns": float(n_turns),
            "agentic/tool_calls": float(n_tool_calls),
            "agentic/steps": float(n_plan_steps),
            "agentic/parse_failed": 1.0 if parse_failed else 0.0,
            "agentic/sample_error": 1.0 if sample_error else 0.0,
            "agentic/prompt_tokens_max": float(max_prompt_tokens),
            "agentic/prompt_tokens_final": float(final_prompt_tokens),
            "agentic/gen_tokens": float(gen_tokens),
        })

        trajectory_tail = list(history[len(initial_messages):])
        has_assistant = any(
            isinstance(m, dict) and m.get("role") == "assistant" for m in trajectory_tail
        )
        if not has_assistant:
            metrics_local["agentic/empty_trajectory"] = 1.0
            drop_reason = (
                "sample_error" if sample_error
                else "parse_failed" if parse_failed
                else "context_overflow" if overflow
                else "empty_trajectory"
            )
            return snapshot, metrics_local, _episode_log(drop_reason, valid=False)

        return snapshot, metrics_local, _episode_log(None, valid=True)
    finally:
        sandbox.cleanup()
