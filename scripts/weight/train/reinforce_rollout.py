"""On-policy agentic rollout for REINFORCE (isolated from ``run_opd``).

Mirrors the OPD agentic episode driver but grades the sandbox via
:func:`weight.data.reward.grade_sandbox_rubrics` and returns only the student
datum (no teacher / distillation path).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.data import conversation_to_datum

try:
    from weight.data.extract import _session_tools_prefix
    from weight.data.reward import grade_sandbox_rubrics
except ModuleNotFoundError:
    from ..data.extract import _session_tools_prefix
    from ..data.reward import grade_sandbox_rubrics

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


async def rollout_one_reinforce_episode(
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
    max_length: int | None,
    collect_transcript: bool = False,
    log_field_chars: int = 2000,
) -> tuple[tinker.Datum | None, dict[str, float], dict[str, Any] | None]:
    """One agentic episode → (student datum, metrics, optional transcript log)."""
    system_prompt = row.get("system_prompt", "") or ""
    tool_schemas = row.get("tool_schemas")
    tools_prefix = list(_session_tools_prefix(system_prompt, tool_schemas, renderer))
    prompt_messages = list(row["prompt_messages"])
    initial_messages = tools_prefix + prompt_messages
    rubrics = [str(r) for r in (row.get("rubrics") or [])]

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

        async def run_segment() -> None:
            nonlocal n_turns, n_tool_calls, parse_failed, overflow, history
            for _ in range(max_turns_per_step):
                if msg_env._turn_count >= max_turns:
                    return
                prompt_input = await asyncio.to_thread(
                    renderer.build_generation_prompt, history
                )
                if (
                    max_trajectory_tokens is not None
                    and prompt_input.length >= max_trajectory_tokens
                ):
                    overflow = True
                    return
                result = await sampling_client.sample_async(
                    prompt=prompt_input,
                    num_samples=1,
                    sampling_params=tinker.SamplingParams(
                        stop=stop,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    ),
                )
                tokens = list(result.sequences[0].tokens)
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
                        "reinforce agentic turn=%d tools=%s content=%r",
                        n_turns,
                        [b["name"] for b in _tool_calls_for_log(message)],
                        preview,
                    )
                step_result = await msg_env.step(message)
                history = step_result.next_messages
                if not tool_calls:
                    return
                if msg_env._should_stop:
                    return

        await run_segment()

        n_plan_steps = 0
        if not parse_failed and not overflow:
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
                if parse_failed or overflow:
                    break

        if overflow:
            metrics_local["agentic/context_overflow"] = 1.0

        def _episode_log(drop_reason: str | None, *, valid: bool) -> dict[str, Any] | None:
            if not collect_transcript:
                return None
            return {
                "messages": _agentic_messages_to_log(
                    history, max_field_chars=log_field_chars
                ),
                "drop_reason": drop_reason,
                "valid": valid,
                "n_turns": n_turns,
                "n_steps": n_plan_steps,
                "n_tool_calls": n_tool_calls,
            }

        trajectory_tail = list(history[len(initial_messages):])
        if not any(
            isinstance(m, dict) and m.get("role") == "assistant" for m in trajectory_tail
        ):
            metrics_local["agentic/empty_trajectory"] = 1.0
            drop_reason = (
                "parse_failed" if parse_failed
                else "context_overflow" if overflow
                else "empty_trajectory"
            )
            return None, metrics_local, _episode_log(drop_reason, valid=False)

        mean_r = await grade_sandbox_rubrics(sandbox, rubrics)
        metrics_local["reinforce/reward"] = mean_r

        student_datum = conversation_to_datum(
            list(history), renderer, max_length,
            train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        )
        metrics_local.update({
            "agentic/turns": float(n_turns),
            "agentic/tool_calls": float(n_tool_calls),
            "agentic/steps": float(n_plan_steps),
            "agentic/parse_failed": 1.0 if parse_failed else 0.0,
            "agentic/trajectory_tokens": float(student_datum.model_input.length),
        })
        return student_datum, metrics_local, _episode_log(None, valid=True)
    finally:
        sandbox.cleanup()
