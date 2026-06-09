"""REINFORCE training on weight-format session JSON.

Connects to the Tinker cloud API (no local server required).
Set ``TINKER_API_KEY`` in ``scripts/weight/.env`` before running.

Applies offline REINFORCE with a running reward baseline for variance
reduction. Rewards are pre-computed from verifier pass rates and human
intervention counts (see ``weight/data/reward.py``).

Loss: L = -mean_j( (R_j - baseline) * sum_k log pi_theta(a_{j,k} | ctx) )

Usage::

    # Offline (logged trajectories + cached rubric rewards):
    python -m weight.train.run_reinforce \\
        --train-path data/weight.json \\
        --model-name Qwen/Qwen3-4B \\
        --renderer-name qwen3 \\
        --log-path ~/logs/reinforce_run

    # Online agentic (``reinforce_rollout``; grades sandbox via LLM rubric):
    python -m weight.train.run_reinforce \\
        --train-path data/weight.json \\
        --model-name Qwen/Qwen3-4B \\
        --renderer-name qwen3 \\
        --log-path ~/logs/reinforce_online \\
        --reinforce-version online --batch-size 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Literal, cast

import tinker
import torch
import chz

from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.supervised.types import ChatDatasetBuilder, ChatDatasetBuilderCommonConfig
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.utils import ml_log, trace
from tinker_cookbook.utils.format_colorized import format_colorized
from tinker_cookbook.utils.lr_scheduling import LRSchedule, compute_schedule_lr_multiplier
from tinker_cookbook.utils.misc_utils import iteration_dir

from .formatter import WeightReinforceDataBuilder, _hydrate_tool_calls, _load_sessions
from .reinforce_rollout import rollout_one_reinforce_episode
from .run_opd import (
    _completion_expected_path,
    _datum_from_prompt_and_assistant_message,
    _format_agentic_transcript,
    _parse_valid_artifact_write_message,
    _summarize_sample,
    _tool_call_name_and_args,
    _with_artifact_only_instruction,
)

try:
    from weight.data.extract import extract_opd_examples, extract_reinforce_rollout_seeds
    from weight.data.reward import _parse_rubric_results_json
except ModuleNotFoundError:
    from ..data.extract import extract_opd_examples, extract_reinforce_rollout_seeds
    from ..data.reward import _parse_rubric_results_json

logger = logging.getLogger(__name__)
BASELINE_STATE_FILENAME = "reinforce_baseline_state.json"


@chz.chz
class Config:
    """Configuration shared by the offline CLI and online server REINFORCE trainer."""

    log_path: str = chz.field(munger=lambda _, s: str(Path(s).expanduser()))
    model_name: str
    dataset_builder: ChatDatasetBuilder | None = None

    # ``offline``: logged trajectories + cached rubric rewards (default).
    # ``online``: on-policy agentic rollout + live sandbox LLM rubric reward.
    reinforce_version: Literal["offline", "online", "artifact", "grpo-artifact"] = "offline"
    max_length: int | None = None

    load_checkpoint_path: str | None = None
    renderer_name: str | None = None
    lora_rank: int = 32

    learning_rate: float = 1e-5
    lr_schedule: LRSchedule = "linear"
    num_epochs: int = 1

    reward_alpha: float = 0.05
    initial_baseline: float = 0.0

    num_replicas: int = 8
    base_url: str | None = None

    save_every: int = 20
    ttl_seconds: int | None = 604800
    rolling_save_every: int = 0
    rolling_ttl_seconds: int = 7200

    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    wandb_project: str | None = None
    wandb_name: str | None = None
    enable_trace: bool = False
    span_chart_every: int = 0
    max_steps: int | None = None

    # Online agentic rollout (``reinforce_version="online"`` only).
    rollout_max_tokens: int = 4096
    rollout_temperature: float = 1.0
    agentic_max_turns: int = 48
    agentic_max_turns_per_step: int = 8
    agentic_max_steps: int = 6
    agentic_enable_bash: bool = True
    agentic_tool_timeout_s: int = 20
    agentic_max_trajectory_tokens: int | None = None
    log_rollout_samples: bool = True
    rollout_sample_log_chars: int = 4000

    # Online artifact-only rollout (``artifact`` / ``grpo-artifact``).
    rollout_attempts: int = 1
    pair_mode: str = "first_last"
    use_gt: bool = True
    use_student: bool = True
    artifact_only_rollout_instruction: bool = False
    grader_model_name: str | None = None
    grader_renderer_name: str | None = None
    grader_max_tokens: int = 4096
    grader_max_file_chars: int = 14000
    grader_temperature: float = 0.0
    grpo_group_size: int = 4


# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

def _load_env() -> None:
    """Load key=value pairs from scripts/weight/.env into os.environ (no-op if missing)."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


# ---------------------------------------------------------------------------
# Baseline persistence
# ---------------------------------------------------------------------------

def _baseline_state_path(log_path: str) -> Path:
    return Path(log_path) / BASELINE_STATE_FILENAME


def _save_baseline_state(log_path: str, epoch_idx: int, batch_idx: int, baseline: float) -> None:
    path = _baseline_state_path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"epoch": epoch_idx, "batch": batch_idx, "baseline": baseline}),
        encoding="utf-8",
    )


def _load_baseline_state(
    log_path: str, resume_info: checkpoint_utils.CheckpointRecord | None,
) -> float | None:
    if resume_info is None:
        return None
    path = _baseline_state_path(log_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Failed to read REINFORCE baseline state: {exc}")
        return None
    resume_epoch = resume_info.epoch or 0
    resume_batch = resume_info.batch or 0
    if payload.get("epoch") != resume_epoch or payload.get("batch") != resume_batch:
        logger.warning("Ignoring saved baseline: checkpoint position mismatch")
        return None
    baseline = payload.get("baseline")
    if not isinstance(baseline, (int, float)):
        return None
    return float(baseline)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def make_reinforce_loss_fn(advantages: list[float]):
    """Return a REINFORCE loss closure for ``forward_backward_custom``."""

    def loss_fn(
        data: list[tinker.Datum], logprobs_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        weighted_logprobs = []
        for j in range(len(data)):
            weights = torch.tensor(data[j].loss_fn_inputs["weights"].data)
            seq_logprob = torch.dot(logprobs_list[j].float(), weights.float())
            weighted_logprobs.append(seq_logprob)

        logprob_tensor = torch.stack(weighted_logprobs)
        advantage_tensor = torch.tensor(advantages, dtype=torch.float32)
        loss = -(advantage_tensor * logprob_tensor).mean()

        metrics = {
            "reinforce_loss": loss.item(),
            "mean_advantage": advantage_tensor.mean().item(),
            "mean_abs_advantage": advantage_tensor.abs().mean().item(),
            "mean_seq_logprob": logprob_tensor.mean().item(),
        }
        return loss, metrics

    return loss_fn


def print_example(datum: tinker.Datum, tokenizer: Tokenizer, label: str = "") -> None:
    int_tokens = list(datum.model_input.to_ints())
    weights = datum.loss_fn_inputs["weights"].data
    logger.info(f"\n{label}:")
    logger.info(format_colorized(int_tokens, cast(list[float], weights), tokenizer))


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def do_update(
    epoch_idx: int,
    batch_idx: int,
    n_batches: int,
    total_steps: int,
    learning_rate_base: float,
    lr_schedule: LRSchedule,
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
    save_every: int,
    ttl_seconds: int | None,
    rolling_save_every: int,
    rolling_ttl_seconds: int,
    span_chart_every: int,
    training_client: tinker.TrainingClient,
    dataset,
    baseline: float,
    ml_logger: ml_log.Logger,
    log_path: str,
    tokenizer: Tokenizer,
    rolling_mgr: checkpoint_utils.CheckpointManager | None = None,
) -> tuple[dict, float]:
    """Single REINFORCE training step. Returns (metrics, new_baseline)."""
    step = epoch_idx * n_batches + batch_idx
    metrics: dict = {"epoch": epoch_idx}

    with trace.trace_iteration(step=step) as window:
        _save_baseline_state(log_path, epoch_idx, batch_idx, baseline)

        if save_every > 0 and step % save_every == 0 and step > 0:
            with trace.scope_span_sync("save_checkpoint"):
                save_result = checkpoint_utils.save_checkpoint(
                    training_client=training_client,
                    name=f"{step:06d}",
                    log_path=log_path,
                    kind="both",
                    loop_state={"epoch": epoch_idx, "batch": batch_idx},
                    ttl_seconds=ttl_seconds,
                )
            if "state_path" in save_result:
                metrics["state_path"] = save_result["state_path"]

        if rolling_mgr is not None:
            rolling_mgr.maybe_save_rolling(
                step, {"epoch": epoch_idx, "batch": batch_idx},
            )

        learning_rate = learning_rate_base * compute_schedule_lr_multiplier(
            lr_schedule=lr_schedule, step=step, total_steps=total_steps,
        )
        adam_params = tinker.AdamParams(
            learning_rate=learning_rate,
            beta1=adam_beta1,
            beta2=adam_beta2,
            eps=adam_eps,
        )

        with trace.scope_span_sync("get_batch"):
            data = dataset.get_batch(batch_idx)
            rewards = dataset.get_batch_rewards(batch_idx)

        # if step == 0:
        #     for i in range(min(3, len(data))):
        #         int_tokens = list(data[i].model_input.to_ints())
        #         weights = data[i].loss_fn_inputs["weights"].data
        #         logger.info(f"\nExample {i} (reward={rewards[i]:.3f}):")
        #         logger.info(format_colorized(int_tokens, cast(list[float], weights), tokenizer))

        advantages = [r - baseline for r in rewards]
        loss_fn = make_reinforce_loss_fn(advantages)

        with trace.scope_span_sync("step"):
            backward_result = training_client.forward_backward_custom(data, loss_fn).result()
            training_client.optim_step(adam_params).result()

        new_baseline = sum(rewards) / len(rewards) if rewards else baseline
        metrics.update(
            num_trajectories=len(data),
            num_tokens=sum(d.model_input.length for d in data),
            learning_rate=learning_rate,
            progress=step / total_steps,
            baseline=new_baseline,
            mean_reward=sum(rewards) / len(rewards) if rewards else 0.0,
            **backward_result.metrics,
        )

    metrics.update(window.get_timing_metrics())
    window.write_spans_jsonl(Path(log_path) / "timing_spans.jsonl", step=step)
    if span_chart_every > 0 and step % span_chart_every == 0:
        iter_dir = iteration_dir(log_path, step)
        if iter_dir is not None:
            iter_dir.mkdir(parents=True, exist_ok=True)
            trace.save_gantt_chart_html(window, step, iter_dir / "timing_gantt.html")
    ml_logger.log_metrics(metrics=metrics, step=step)
    return metrics, new_baseline


# ---------------------------------------------------------------------------
# Online artifact-only rollout + Tinker grader
# ---------------------------------------------------------------------------

class ArtifactReinforceRolloutDataset:
    """OPD-style artifact prompts for on-policy REINFORCE/GRPO."""

    def __init__(self, rows: list[dict[str, Any]], batch_size: int):
        self._rows = rows
        self._batch_size = batch_size
        self._indices = list(range(len(rows)))

    @classmethod
    def from_weight_json(
        cls,
        path: str,
        renderer: renderers.Renderer,
        max_length: int | None,
        batch_size: int,
        *,
        pair_mode: str = "first_last",
        use_gt: bool = True,
        use_student: bool = True,
        artifact_only_instruction: bool = False,
    ) -> "ArtifactReinforceRolloutDataset":
        examples = extract_opd_examples(
            _load_sessions(path),
            renderer=renderer,
            pair_mode=pair_mode,
            use_gt=use_gt,
            use_student=use_student,
        )
        rows: list[dict[str, Any]] = []
        for ex in examples:
            expected_path = _completion_expected_path(ex.get("completion", []))
            if expected_path is None:
                continue
            student_prompt = ex["student_prompt"]
            if artifact_only_instruction:
                student_prompt = _with_artifact_only_instruction(
                    student_prompt, expected_path,
                )
            student_prompt_messages = _hydrate_tool_calls(student_prompt)
            student_prompt_input = renderer.build_generation_prompt(
                student_prompt_messages
            )
            rubrics = [str(r) for r in (ex.get("rubrics") or []) if str(r).strip()]
            if not rubrics:
                continue
            rows.append({
                "student_prompt_messages": student_prompt_messages,
                "student_prompt_input": student_prompt_input,
                "expected_path": expected_path,
                "rubrics": rubrics,
                "meta": ex.get("meta") or {},
            })
        logger.info(
            "Loaded %d artifact REINFORCE rollout examples from %s "
            "(raw=%d, pair_mode=%s, use_gt=%s, use_student=%s, verifiers=weight-json)",
            len(rows), path, len(examples), pair_mode, use_gt, use_student,
        )
        dataset = cls(rows, batch_size)
        dataset._max_length = max_length
        return dataset

    def __len__(self) -> int:
        if not self._rows:
            return 0
        return (len(self._rows) + self._batch_size - 1) // self._batch_size

    def set_epoch(self, seed: int) -> None:
        rng = random.Random(seed)
        self._indices = list(range(len(self._rows)))
        rng.shuffle(self._indices)

    def get_batch(self, index: int) -> list[dict[str, Any]]:
        start = index * self._batch_size
        end = min(start + self._batch_size, len(self._indices))
        return [self._rows[self._indices[i]] for i in range(start, end)]


def _artifact_args_from_message(message: dict[str, Any]) -> dict[str, str] | None:
    tool_calls = message.get("tool_calls") or []
    if len(tool_calls) != 1:
        return None
    name, args_text = _tool_call_name_and_args(tool_calls[0])
    if name != "write" or not isinstance(args_text, str):
        return None
    try:
        args = json.loads(args_text)
    except json.JSONDecodeError:
        return None
    path = args.get("path")
    content = args.get("content")
    if not isinstance(path, str) or not isinstance(content, str) or not content.strip():
        return None
    return {"path": path, "content": content}


def _truncate_text(text: str, max_len: int) -> str:
    return text if len(text) <= max_len else text[:max_len] + "\n... [truncated]"


def _build_artifact_grader_prompt(
    rubrics: list[str],
    path: str,
    content: str,
    *,
    max_file_chars: int,
) -> str:
    numbered = "\n".join(f"{i}. {r}" for i, r in enumerate(rubrics))
    body = _truncate_text(content, max_file_chars)
    result_template = ",".join('{"pass":false}' for _ in rubrics)
    return "\n".join([
        "Grade a Python script that is intended to generate a final image artifact.",
        "The verifier lines were originally written for the generated image. You only see the script; judge whether running it would likely satisfy each verifier.",
        "Return JSON only. Do not include reasoning, markdown, code fences, labels, or prose.",
        "Your first character must be { and your last character must be }.",
        "Use exactly this schema and exactly one result per verifier, in order:",
        f'{{"results":[{result_template}]}}',
        "",
        "Verifier lines:",
        numbered,
        "",
        f"Script path: {path}",
        "Script content:",
        body,
    ])


_NUMBERED_VERDICT_RE = re.compile(
    r"(?ms)^\s*(\d+)[\.\):]\s+(.*?)(?=^\s*\d+[\.\):]\s+|\Z)"
)


def _parse_pass_fail_from_text(text: str, n: int) -> list[bool | None]:
    """Recover ordered PASS/FAIL judgments from verbose grader text."""
    out: list[bool | None] = [None] * n
    matches = list(_NUMBERED_VERDICT_RE.finditer(text))
    if matches:
        for m in matches:
            idx = int(m.group(1))
            if idx < 0 or idx >= n:
                continue
            verdicts = re.findall(r"\b(PASS|FAIL|TRUE|FALSE)\b", m.group(2), re.IGNORECASE)
            if verdicts:
                last = verdicts[-1].lower()
                out[idx] = last in {"pass", "true"}
        if any(v is not None for v in out):
            return out

    # Last-resort fallback for models that emit a compact unnumbered list.
    verdicts = re.findall(r"\b(PASS|FAIL|TRUE|FALSE)\b", text, re.IGNORECASE)
    for i, verdict in enumerate(verdicts[:n]):
        out[i] = verdict.lower() in {"pass", "true"}
    return out


def _parse_artifact_grader_results(text: str, n: int) -> tuple[list[bool | None], str]:
    try:
        passes, _raw_results = _parse_rubric_results_json(text, n)
        return passes, "json"
    except Exception:
        passes = _parse_pass_fail_from_text(text, n)
        if any(p is not None for p in passes):
            return passes, "text_pass_fail"
        raise


async def _grade_artifact_with_tinker(
    grader_client: tinker.SamplingClient,
    grader_renderer: renderers.Renderer,
    rubrics: list[str],
    artifact: dict[str, str],
    *,
    max_tokens: int,
    max_file_chars: int,
    temperature: float,
) -> tuple[float, list[float], str, str]:
    if not rubrics:
        return 1.0, [], "no_rubrics", "none"
    prompt = _build_artifact_grader_prompt(
        rubrics,
        artifact["path"],
        artifact["content"],
        max_file_chars=max_file_chars,
    )
    prompt_input = grader_renderer.build_generation_prompt([
        {
            "role": "system",
            "content": "You are a strict JSON API. Return only valid JSON and no explanatory text.",
        },
        {"role": "user", "content": prompt},
    ])
    result = await grader_client.sample_async(
        prompt=prompt_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            stop=grader_renderer.get_stop_sequences(),
            max_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    raw = str(grader_renderer.tokenizer.decode(list(result.sequences[0].tokens)))
    try:
        passes, parse_mode = _parse_artifact_grader_results(raw, len(rubrics))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Artifact grader parse failed (%s); reward=0.0", exc)
        return 0.0, [0.0] * len(rubrics), raw, "failed"
    missing_idx = [i for i, p in enumerate(passes) if p is None]
    if missing_idx:
        logger.warning(
            "Artifact grader parse mode=%s recovered %d/%d judgments; missing %s scored as 0.0",
            parse_mode,
            len(rubrics) - len(missing_idx),
            len(rubrics),
            missing_idx[:20],
        )
    scores = [1.0 if p is True else 0.0 for p in passes]
    return sum(scores) / len(rubrics), scores, raw, parse_mode


async def _sample_one_artifact_candidate(
    row_idx: int,
    row: dict[str, Any],
    renderer: renderers.Renderer,
    sampling_client: tinker.SamplingClient,
    grader_client: tinker.SamplingClient,
    grader_renderer: renderers.Renderer,
    *,
    max_tokens: int,
    temperature: float,
    max_length: int | None,
    grader_max_tokens: int,
    grader_max_file_chars: int,
    grader_temperature: float,
    step: int,
    attempt_idx: int,
    sample_log_chars: int,
) -> dict[str, Any]:
    rollout_started_at = time.perf_counter()
    result = await sampling_client.sample_async(
        prompt=row["student_prompt_input"],
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            stop=renderer.get_stop_sequences(),
            max_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    tokens = list(result.sequences[0].tokens)
    expected_path = row["expected_path"]
    ok, reason, canonical_message = _parse_valid_artifact_write_message(
        renderer, tokens, expected_path,
    )
    rec = _summarize_sample(
        renderer,
        tokens,
        expected_path,
        ok,
        reason,
        row_idx,
        attempt_idx,
        step,
        sample_log_chars,
    )
    rec["rollout_elapsed_s"] = time.perf_counter() - rollout_started_at
    rec["session_uuid"] = (row.get("meta") or {}).get("session_uuid")
    rec["artifact_path"] = (row.get("meta") or {}).get("artifact_path")
    if not ok or canonical_message is None:
        return {
            "valid": False,
            "reason": reason,
            "record": rec,
            "rollout_elapsed_s": float(rec["rollout_elapsed_s"]),
        }
    artifact = _artifact_args_from_message(canonical_message)
    if artifact is None:
        return {
            "valid": False,
            "reason": "artifact_args_missing",
            "record": rec,
            "rollout_elapsed_s": float(rec["rollout_elapsed_s"]),
        }
    datum = _datum_from_prompt_and_assistant_message(
        renderer,
        row["student_prompt_messages"],
        canonical_message,
        max_length,
    )
    if datum is None:
        return {
            "valid": False,
            "reason": "too_long_or_empty",
            "record": rec,
            "rollout_elapsed_s": float(rec["rollout_elapsed_s"]),
        }
    reward, rubric_scores, grader_raw, grader_parse_mode = await _grade_artifact_with_tinker(
        grader_client,
        grader_renderer,
        [str(r) for r in row.get("rubrics", [])],
        artifact,
        max_tokens=grader_max_tokens,
        max_file_chars=grader_max_file_chars,
        temperature=grader_temperature,
    )
    rec.update({
        "valid": True,
        "reward": reward,
        "rubric_scores": rubric_scores,
        "grader_parse_mode": grader_parse_mode,
        "grader_raw_preview": grader_raw[:sample_log_chars],
        "graded_path": artifact["path"],
    })
    return {
        "valid": True,
        "reason": reason,
        "datum": datum,
        "reward": float(reward),
        "artifact": artifact,
        "record": rec,
        "rollout_elapsed_s": float(rec["rollout_elapsed_s"]),
    }


async def _sample_artifact_reinforce_async(
    rows: list[dict[str, Any]],
    renderer: renderers.Renderer,
    sampling_client: tinker.SamplingClient,
    grader_client: tinker.SamplingClient,
    grader_renderer: renderers.Renderer,
    config: Config,
    *,
    max_length: int | None,
    step: int,
    sample_log_path: Path | None = None,
) -> tuple[list[tinker.Datum], list[float], dict[str, float]]:
    datums: list[tinker.Datum] = []
    rewards: list[float] = []
    reason_counts: dict[str, int] = {}
    total_attempts = 0
    total_rollout_elapsed_s = 0.0
    records: list[dict[str, Any]] = []

    pending = list(enumerate(rows))
    for attempt_idx in range(max(1, config.rollout_attempts)):
        if not pending:
            break
        total_attempts += len(pending)
        results = await asyncio.gather(*[
            _sample_one_artifact_candidate(
                row_idx,
                row,
                renderer,
                sampling_client,
                grader_client,
                grader_renderer,
                max_tokens=config.rollout_max_tokens,
                temperature=config.rollout_temperature,
                max_length=max_length,
                grader_max_tokens=config.grader_max_tokens,
                grader_max_file_chars=config.grader_max_file_chars,
                grader_temperature=config.grader_temperature,
                step=step,
                attempt_idx=attempt_idx,
                sample_log_chars=config.rollout_sample_log_chars,
            )
            for row_idx, row in pending
        ])
        next_pending: list[tuple[int, dict[str, Any]]] = []
        for (row_idx, row), item in zip(pending, results, strict=True):
            reason = str(item.get("reason") or "unknown")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            total_rollout_elapsed_s += float(item.get("rollout_elapsed_s", 0.0))
            records.append(item["record"])
            if item.get("valid"):
                datums.append(item["datum"])
                rewards.append(float(item["reward"]))
            else:
                next_pending.append((row_idx, row))
        pending = next_pending
    if pending:
        reason_counts["example_filtered"] = reason_counts.get("example_filtered", 0) + len(pending)

    if sample_log_path is not None:
        sample_log_path.parent.mkdir(parents=True, exist_ok=True)
        with sample_log_path.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_rows = float(len(rows))
    n_valid = float(len(datums))
    metrics: dict[str, float] = {
        "artifact_online/batch_examples": n_rows,
        "artifact_online/valid_examples": n_valid,
        "artifact_online/filtered_examples": n_rows - n_valid,
        "artifact_online/filter_rate": (n_rows - n_valid) / max(n_rows, 1.0),
        "artifact_online/attempts": float(total_attempts),
        "artifact_online/rollout_elapsed_s": total_rollout_elapsed_s,
    }
    if total_attempts:
        metrics["artifact_online/mean_rollout_elapsed_s"] = (
            total_rollout_elapsed_s / float(total_attempts)
        )
    if rewards:
        metrics["artifact_online/mean_reward"] = sum(rewards) / len(rewards)
    for reason, count in reason_counts.items():
        safe_reason = reason.replace("/", "_").replace(":", "_")
        metrics[f"artifact_online/filter_reason/{safe_reason}"] = float(count)
    return datums, rewards, metrics


def compute_grpo_group_advantages(rewards: list[float]) -> list[float] | None:
    """Return group-centered advantages, or None when the group has no signal."""
    if len(rewards) < 2:
        return None
    mean_reward = sum(rewards) / len(rewards)
    advantages = [reward - mean_reward for reward in rewards]
    if all(abs(adv) < 1e-8 for adv in advantages):
        return None
    return advantages


async def _sample_artifact_grpo_async(
    rows: list[dict[str, Any]],
    renderer: renderers.Renderer,
    sampling_client: tinker.SamplingClient,
    grader_client: tinker.SamplingClient,
    grader_renderer: renderers.Renderer,
    config: Config,
    *,
    max_length: int | None,
    step: int,
    sample_log_path: Path | None = None,
) -> tuple[list[tinker.Datum], list[float], dict[str, float]]:
    group_size = max(1, int(config.grpo_group_size))
    total_rollout_elapsed_s = 0.0
    indexed = [
        (row_idx, sample_idx, row)
        for row_idx, row in enumerate(rows)
        for sample_idx in range(group_size)
    ]
    results = await asyncio.gather(*[
        _sample_one_artifact_candidate(
            row_idx,
            row,
            renderer,
            sampling_client,
            grader_client,
            grader_renderer,
            max_tokens=config.rollout_max_tokens,
            temperature=config.rollout_temperature,
            max_length=max_length,
            grader_max_tokens=config.grader_max_tokens,
            grader_max_file_chars=config.grader_max_file_chars,
            grader_temperature=config.grader_temperature,
            step=step,
            attempt_idx=sample_idx,
            sample_log_chars=config.rollout_sample_log_chars,
        )
        for row_idx, sample_idx, row in indexed
    ])

    groups: list[list[dict[str, Any]]] = [[] for _ in rows]
    records: list[dict[str, Any]] = []
    valid_episodes = 0
    reward_sum = 0.0
    for (row_idx, sample_idx, _row), item in zip(indexed, results, strict=True):
        rec = item["record"]
        rec["sample_idx"] = sample_idx
        total_rollout_elapsed_s += float(item.get("rollout_elapsed_s", 0.0))
        records.append(rec)
        if item.get("valid"):
            valid_episodes += 1
            reward_sum += float(item["reward"])
            groups[row_idx].append(item)

    datums: list[tinker.Datum] = []
    advantages_out: list[float] = []
    skipped_small = 0
    skipped_constant = 0
    for group in groups:
        if len(group) < 2:
            skipped_small += 1
            continue
        rewards = [float(item["reward"]) for item in group]
        advantages = compute_grpo_group_advantages(rewards)
        if advantages is None:
            skipped_constant += 1
            continue
        for item, advantage in zip(group, advantages, strict=True):
            datums.append(item["datum"])
            advantages_out.append(float(advantage))

    if sample_log_path is not None:
        sample_log_path.parent.mkdir(parents=True, exist_ok=True)
        with sample_log_path.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_episodes = len(rows) * group_size
    metrics: dict[str, float] = {
        "grpo_artifact/batch_prompts": float(len(rows)),
        "grpo_artifact/group_size": float(group_size),
        "grpo_artifact/batch_episodes": float(total_episodes),
        "grpo_artifact/valid_episodes": float(valid_episodes),
        "grpo_artifact/train_datums": float(len(datums)),
        "grpo_artifact/filter_rate": (
            1.0 - valid_episodes / total_episodes if total_episodes else 0.0
        ),
        "grpo_artifact/rollout_elapsed_s": total_rollout_elapsed_s,
        "grpo_artifact/skipped_small_groups": float(skipped_small),
        "grpo_artifact/skipped_constant_groups": float(skipped_constant),
    }
    if total_episodes:
        metrics["grpo_artifact/mean_rollout_elapsed_s"] = (
            total_rollout_elapsed_s / float(total_episodes)
        )
    if valid_episodes:
        metrics["grpo_artifact/mean_reward"] = reward_sum / valid_episodes
    if advantages_out:
        metrics["grpo_artifact/mean_abs_advantage"] = (
            sum(abs(a) for a in advantages_out) / len(advantages_out)
        )
    return datums, advantages_out, metrics


async def train_policy_gradient_batch_async(
    *,
    training_client: tinker.TrainingClient,
    datums: list[tinker.Datum],
    advantages: list[float],
    config: Config,
    step: int,
    total_steps: int,
    extra_metrics: dict[str, float] | None = None,
) -> dict[str, float]:
    loss_fn = make_reinforce_loss_fn(advantages)
    learning_rate = config.learning_rate * compute_schedule_lr_multiplier(
        lr_schedule=config.lr_schedule, step=step, total_steps=total_steps,
    )
    adam_params = tinker.AdamParams(
        learning_rate=learning_rate,
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        eps=config.adam_eps,
    )
    fb_future = await training_client.forward_backward_custom_async(datums, loss_fn)
    backward_result = await fb_future.result_async()
    optim_future = await training_client.optim_step_async(adam_params)
    await optim_future.result_async()
    metrics: dict[str, float] = {
        "learning_rate": learning_rate,
        "progress": step / max(total_steps, 1),
        "num_trajectories": float(len(datums)),
        "num_tokens": float(sum(d.model_input.length for d in datums)),
        **(extra_metrics or {}),
        **backward_result.metrics,
    }
    return metrics


# ---------------------------------------------------------------------------
# Online agentic rollout (reuses OPD episode driver + sandbox rubric reward)
# ---------------------------------------------------------------------------


class OnlineReinforceRolloutDataset:
    """Session seeds for on-policy REINFORCE; rewards come from live rollouts."""

    def __init__(self, rows: list[dict[str, Any]], batch_size: int):
        self._rows = rows
        self._batch_size = batch_size
        self._indices = list(range(len(rows)))

    @classmethod
    def from_weight_json(cls, path: str, batch_size: int) -> "OnlineReinforceRolloutDataset":
        examples = extract_reinforce_rollout_seeds(_load_sessions(path))
        rows = [
            {
                "prompt_messages": _hydrate_tool_calls(ex["prompt_messages"]),
                "system_prompt": ex.get("system_prompt", "") or "",
                "tool_schemas": ex.get("tool_schemas"),
                "rubrics": ex.get("rubrics") or [],
                "meta": ex.get("meta") or {},
            }
            for ex in examples
        ]
        logger.info("Loaded %d online REINFORCE rollout seeds from %s", len(rows), path)
        return cls(rows, batch_size)

    def __len__(self) -> int:
        if not self._rows:
            return 0
        return (len(self._rows) + self._batch_size - 1) // self._batch_size

    def set_epoch(self, seed: int) -> None:
        rng = random.Random(seed)
        self._indices = list(range(len(self._rows)))
        rng.shuffle(self._indices)

    def get_batch(self, index: int) -> list[dict[str, Any]]:
        start = index * self._batch_size
        end = min(start + self._batch_size, len(self._indices))
        return [self._rows[self._indices[i]] for i in range(start, end)]


async def _sample_agentic_reinforce_async(
    rows: list[dict[str, Any]],
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
    step: int,
    sample_log_path: Path | None = None,
    sample_log_chars: int = 4000,
) -> tuple[list[tinker.Datum], list[float], dict[str, float]]:
    results = await asyncio.gather(*[
        rollout_one_reinforce_episode(
            row,
            renderer,
            sampling_client,
            max_tokens=max_tokens,
            temperature=temperature,
            max_turns=max_turns,
            max_turns_per_step=max_turns_per_step,
            max_steps=max_steps,
            enable_bash=enable_bash,
            tool_timeout_s=tool_timeout_s,
            max_trajectory_tokens=max_trajectory_tokens,
            max_length=max_length,
            collect_transcript=sample_log_path is not None,
            log_field_chars=sample_log_chars,
        )
        for row in rows
    ])

    datums: list[tinker.Datum] = []
    rewards: list[float] = []
    n_valid = 0
    agg_reward = 0.0

    sample_log_f = None
    transcript_f = None
    if sample_log_path is not None:
        sample_log_path.parent.mkdir(parents=True, exist_ok=True)
        sample_log_f = sample_log_path.open("a", encoding="utf-8")
        transcript_f = sample_log_path.with_name(
            "reinforce_rollout_transcripts.txt"
        ).open("a", encoding="utf-8")

    try:
        for row, (datum, metrics, episode_log) in zip(rows, results, strict=True):
            reward = metrics.get("reinforce/reward")
            valid = datum is not None and reward is not None
            meta = row.get("meta") or {}
            drop_reason = (episode_log or {}).get("drop_reason")
            if not valid and not drop_reason:
                drop_reason = "dropped"

            logger.info(
                "reinforce rollout step=%d session=%s valid=%s reward=%s drop=%s",
                step,
                meta.get("session_uuid"),
                valid,
                reward,
                drop_reason or "-",
            )

            if sample_log_f is not None:
                rec = {
                    "step": step,
                    "session_uuid": meta.get("session_uuid"),
                    "valid": valid,
                    "reward": reward,
                    "drop_reason": drop_reason,
                    "turns": metrics.get("agentic/turns"),
                    "steps": metrics.get("agentic/steps"),
                    "tool_calls": metrics.get("agentic/tool_calls"),
                    "messages": (episode_log or {}).get("messages") or [],
                }
                sample_log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sample_log_f.flush()
                if transcript_f is not None:
                    transcript_f.write(_format_agentic_transcript(rec) + "\n")
                    transcript_f.flush()

            if not valid:
                continue
            n_valid += 1
            agg_reward += float(reward)
            datums.append(datum)
            rewards.append(float(reward))
    finally:
        if sample_log_f is not None:
            sample_log_f.close()
        if transcript_f is not None:
            transcript_f.close()

    batch = len(rows)
    rollout_metrics: dict[str, float] = {
        "reinforce_online/batch_examples": float(batch),
        "reinforce_online/valid_examples": float(n_valid),
        "reinforce_online/filter_rate": (1.0 - n_valid / batch) if batch else 0.0,
    }
    if n_valid:
        rollout_metrics["reinforce_online/mean_reward"] = agg_reward / n_valid
    return datums, rewards, rollout_metrics


async def train_reinforce_batch_async(
    *,
    training_client: tinker.TrainingClient,
    datums: list[tinker.Datum],
    rewards: list[float],
    baseline: float,
    config: Config,
    step: int,
    total_steps: int,
    extra_metrics: dict[str, float] | None = None,
) -> tuple[dict[str, float], float]:
    """Forward-backward + optim for one REINFORCE batch. Returns (metrics, new_baseline)."""
    advantages = [r - baseline for r in rewards]
    loss_fn = make_reinforce_loss_fn(advantages)
    learning_rate = config.learning_rate * compute_schedule_lr_multiplier(
        lr_schedule=config.lr_schedule, step=step, total_steps=total_steps,
    )
    adam_params = tinker.AdamParams(
        learning_rate=learning_rate,
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        eps=config.adam_eps,
    )
    fb_future = await training_client.forward_backward_custom_async(datums, loss_fn)
    backward_result = await fb_future.result_async()
    optim_future = await training_client.optim_step_async(adam_params)
    await optim_future.result_async()

    new_baseline = sum(rewards) / len(rewards) if rewards else baseline
    metrics: dict[str, float] = {
        "learning_rate": learning_rate,
        "progress": step / max(total_steps, 1),
        "baseline": new_baseline,
        "mean_reward": new_baseline,
        "num_trajectories": float(len(datums)),
        "num_tokens": float(sum(d.model_input.length for d in datums)),
        **(extra_metrics or {}),
        **backward_result.metrics,
    }
    return metrics, new_baseline


def _config_from_cli(
    args: argparse.Namespace,
    *,
    reinforce_version: Literal["offline", "online", "artifact", "grpo-artifact"],
    dataset_builder: ChatDatasetBuilder | None,
) -> Config:
    return Config(
        log_path=str(Path(args.log_path).expanduser()),
        model_name=args.model_name,
        dataset_builder=dataset_builder,
        reinforce_version=reinforce_version,
        renderer_name=args.renderer_name,
        max_length=args.max_length,
        load_checkpoint_path=args.load_checkpoint_path,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        lr_schedule=args.lr_schedule,
        num_epochs=args.num_epochs,
        initial_baseline=args.initial_baseline,
        base_url=args.base_url,
        save_every=args.save_every,
        ttl_seconds=args.ttl_seconds,
        rolling_save_every=args.rolling_save_every,
        rolling_ttl_seconds=args.rolling_ttl_seconds,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        span_chart_every=args.span_chart_every,
        max_steps=args.max_steps,
        rollout_max_tokens=args.rollout_max_tokens,
        rollout_temperature=args.rollout_temperature,
        agentic_max_turns=args.agentic_max_turns,
        agentic_max_turns_per_step=args.agentic_max_turns_per_step,
        agentic_max_steps=args.agentic_max_steps,
        agentic_enable_bash=not args.no_agentic_bash,
        agentic_tool_timeout_s=args.agentic_tool_timeout_s,
        agentic_max_trajectory_tokens=args.agentic_max_trajectory_tokens,
        log_rollout_samples=not args.no_log_rollout_samples,
        rollout_sample_log_chars=args.rollout_sample_log_chars,
        rollout_attempts=args.rollout_attempts,
        pair_mode=args.pair_mode,
        use_gt=not args.no_use_gt,
        use_student=not args.no_use_student,
        artifact_only_rollout_instruction=args.artifact_only_rollout_instruction,
        grader_model_name=args.grader_model_name,
        grader_renderer_name=args.grader_renderer_name,
        grader_max_tokens=args.grader_max_tokens,
        grader_max_file_chars=args.grader_max_file_chars,
        grader_temperature=args.grader_temperature,
        grpo_group_size=args.grpo_group_size,
    )


async def run_artifact_policy_training(
    config: Config,
    dataset: ArtifactReinforceRolloutDataset,
) -> None:
    """Run artifact-only REINFORCE or GRPO with an external Tinker grader."""
    if not dataset._rows:
        raise ValueError("No artifact rollout examples (empty train data or no rubrics).")
    assert config.renderer_name is not None

    renderer = renderers.get_renderer(config.renderer_name, get_tokenizer(config.model_name))
    grader_model_name = config.grader_model_name or config.model_name
    grader_renderer_name = config.grader_renderer_name or config.renderer_name
    grader_renderer = renderers.get_renderer(
        grader_renderer_name,
        get_tokenizer(grader_model_name),
    )
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
        do_configure_logging_module=True,
    )
    model_info.warn_if_renderer_not_recommended(config.model_name, config.renderer_name)
    model_info.warn_if_renderer_not_recommended(grader_model_name, grader_renderer_name)

    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    start_batch = resume_info.batch if resume_info else 0

    user_metadata: dict[str, str] = {}
    if wandb_link := ml_logger.get_logger_url():
        user_metadata["wandb_link"] = wandb_link
    checkpoint_utils.add_renderer_name_to_user_metadata(user_metadata, config.renderer_name)

    service_client = tinker.ServiceClient(base_url=config.base_url)
    if resume_info:
        assert resume_info.state_path is not None
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info.state_path, user_metadata=user_metadata,
        )
    elif config.load_checkpoint_path:
        training_client = service_client.create_training_client_from_state(
            config.load_checkpoint_path, user_metadata=user_metadata,
        )
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank, user_metadata=user_metadata,
        )

    sampling_client = training_client.save_weights_and_get_sampling_client()
    grader_client = service_client.create_sampling_client(base_model=grader_model_name)
    n_batches = len(dataset)
    total_steps = n_batches * config.num_epochs
    if config.max_steps is not None:
        total_steps = min(total_steps, config.max_steps)

    baseline = _load_baseline_state(config.log_path, resume_info)
    if baseline is None:
        baseline = float(config.initial_baseline)

    sample_log_path = None
    if config.log_rollout_samples:
        sample_name = (
            "grpo_artifact_rollout_samples.jsonl"
            if config.reinforce_version == "grpo-artifact"
            else "reinforce_artifact_rollout_samples.jsonl"
        )
        sample_log_path = Path(config.log_path) / sample_name

    logger.info(
        "%s: %d examples, %d batches x %d epochs (max_steps=%s, grader=%s)",
        config.reinforce_version,
        len(dataset._rows),
        n_batches,
        config.num_epochs,
        config.max_steps,
        grader_model_name,
    )

    t0 = time.perf_counter()
    for epoch_idx in range(config.num_epochs):
        dataset.set_epoch(seed=epoch_idx)
        for batch_idx in range(start_batch if epoch_idx == 0 else 0, n_batches):
            step = epoch_idx * n_batches + batch_idx
            if config.max_steps is not None and step >= config.max_steps:
                break

            rows = dataset.get_batch(batch_idx)
            if config.reinforce_version == "grpo-artifact":
                datums, advantages, rollout_metrics = await _sample_artifact_grpo_async(
                    rows,
                    renderer,
                    sampling_client,
                    grader_client,
                    grader_renderer,
                    config,
                    max_length=dataset._max_length,
                    step=step,
                    sample_log_path=sample_log_path,
                )
                no_valid_key = "grpo_artifact/no_valid_batch"
            else:
                _save_baseline_state(config.log_path, epoch_idx, batch_idx, baseline)
                datums, rewards, rollout_metrics = await _sample_artifact_reinforce_async(
                    rows,
                    renderer,
                    sampling_client,
                    grader_client,
                    grader_renderer,
                    config,
                    max_length=dataset._max_length,
                    step=step,
                    sample_log_path=sample_log_path,
                )
                advantages = [r - baseline for r in rewards]
                if rewards:
                    baseline = sum(rewards) / len(rewards)
                rollout_metrics["baseline"] = baseline
                no_valid_key = "artifact_online/no_valid_batch"

            logger.info(
                "%s step=%d valid=%d avg_rollout_s=%.3f",
                config.reinforce_version,
                step,
                len(datums),
                float(rollout_metrics.get(
                    "grpo_artifact/mean_rollout_elapsed_s",
                    rollout_metrics.get("artifact_online/mean_rollout_elapsed_s", 0.0),
                )),
            )

            if not datums:
                ml_logger.log_metrics(
                    metrics={
                        no_valid_key: 1.0,
                        "progress": step / max(total_steps, 1),
                        **rollout_metrics,
                    },
                    step=step,
                )
                logger.warning("Skipping step %d: no trainable artifact samples", step)
                continue

            step_metrics = await train_policy_gradient_batch_async(
                training_client=training_client,
                datums=datums,
                advantages=advantages,
                config=config,
                step=step,
                total_steps=total_steps,
                extra_metrics=rollout_metrics,
            )
            ml_logger.log_metrics(metrics=step_metrics, step=step)

            if config.save_every > 0 and step % config.save_every == 0 and step > 0:
                await checkpoint_utils.save_checkpoint_async(
                    training_client=training_client,
                    name=f"{step:06d}",
                    log_path=config.log_path,
                    kind="both",
                    loop_state={"epoch": epoch_idx, "batch": batch_idx},
                    ttl_seconds=config.ttl_seconds,
                )
            sampling_client = training_client.save_weights_and_get_sampling_client()

    await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"epoch": config.num_epochs, "batch": 0},
        ttl_seconds=None,
    )
    ml_logger.log_metrics(
        metrics={f"{config.reinforce_version}/elapsed_s": time.perf_counter() - t0},
        step=max(total_steps - 1, 0),
    )
    ml_logger.close()
    logger.info("%s training completed", config.reinforce_version)


async def run_online_training(
    config: Config,
    dataset: OnlineReinforceRolloutDataset,
) -> None:
    """CLI/server path for ``reinforce_version=\"online\"``."""
    if not dataset._rows:
        raise ValueError("No REINFORCE online rollout seeds (empty train data).")
    assert config.renderer_name is not None

    renderer = renderers.get_renderer(config.renderer_name, get_tokenizer(config.model_name))
    tokenizer = get_tokenizer(config.model_name)
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
        do_configure_logging_module=True,
    )
    model_info.warn_if_renderer_not_recommended(config.model_name, config.renderer_name)

    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    start_batch = resume_info.batch if resume_info else 0

    user_metadata: dict[str, str] = {}
    if wandb_link := ml_logger.get_logger_url():
        user_metadata["wandb_link"] = wandb_link
    checkpoint_utils.add_renderer_name_to_user_metadata(user_metadata, config.renderer_name)

    service_client = tinker.ServiceClient(base_url=config.base_url)
    if resume_info:
        assert resume_info.state_path is not None
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info.state_path, user_metadata=user_metadata,
        )
    elif config.load_checkpoint_path:
        training_client = service_client.create_training_client_from_state(
            config.load_checkpoint_path, user_metadata=user_metadata,
        )
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank, user_metadata=user_metadata,
        )

    sampling_client = training_client.save_weights_and_get_sampling_client()
    n_batches = len(dataset)
    total_steps = n_batches * config.num_epochs
    if config.max_steps is not None:
        total_steps = min(total_steps, config.max_steps)

    baseline = _load_baseline_state(config.log_path, resume_info)
    if baseline is None:
        baseline = float(config.initial_baseline)

    sample_log_path = (
        Path(config.log_path) / "reinforce_rollout_samples.jsonl"
        if config.log_rollout_samples else None
    )

    logger.info(
        "REINFORCE online: %d seeds, %d batches x %d epochs (max_steps=%s)",
        len(dataset._rows), n_batches, config.num_epochs, config.max_steps,
    )

    for epoch_idx in range(config.num_epochs):
        dataset.set_epoch(seed=epoch_idx)
        for batch_idx in range(start_batch if epoch_idx == 0 else 0, n_batches):
            step = epoch_idx * n_batches + batch_idx
            if config.max_steps is not None and step >= config.max_steps:
                break

            _save_baseline_state(config.log_path, epoch_idx, batch_idx, baseline)

            datums, rewards, rollout_metrics = await _sample_agentic_reinforce_async(
                dataset.get_batch(batch_idx),
                renderer,
                sampling_client,
                max_tokens=config.rollout_max_tokens,
                temperature=config.rollout_temperature,
                max_turns=config.agentic_max_turns,
                max_turns_per_step=config.agentic_max_turns_per_step,
                max_steps=config.agentic_max_steps,
                enable_bash=config.agentic_enable_bash,
                tool_timeout_s=config.agentic_tool_timeout_s,
                max_trajectory_tokens=config.agentic_max_trajectory_tokens,
                max_length=config.max_length,
                step=step,
                sample_log_path=sample_log_path,
                sample_log_chars=config.rollout_sample_log_chars,
            )

            if not datums:
                ml_logger.log_metrics(
                    metrics={
                        "reinforce_online/no_valid_batch": 1.0,
                        "progress": step / max(total_steps, 1),
                        **rollout_metrics,
                    },
                    step=step,
                )
                logger.warning("Skipping step %d: no valid rollout samples", step)
                continue

            # if step == 0:
            #     print_example(datums[0], tokenizer, label="Online rollout example 0")

            step_metrics, baseline = await train_reinforce_batch_async(
                training_client=training_client,
                datums=datums,
                rewards=rewards,
                baseline=baseline,
                config=config,
                step=step,
                total_steps=total_steps,
                extra_metrics=rollout_metrics,
            )
            ml_logger.log_metrics(metrics=step_metrics, step=step)

            if config.save_every > 0 and step % config.save_every == 0 and step > 0:
                await checkpoint_utils.save_checkpoint_async(
                    training_client=training_client,
                    name=f"{step:06d}",
                    log_path=config.log_path,
                    kind="both",
                    loop_state={"epoch": epoch_idx, "batch": batch_idx},
                    ttl_seconds=config.ttl_seconds,
                )
            sampling_client = training_client.save_weights_and_get_sampling_client()

    await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"epoch": config.num_epochs, "batch": 0},
        ttl_seconds=None,
    )
    ml_logger.close()
    logger.info("REINFORCE online training completed")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser(description="REINFORCE training (weight-format, Tinker API)")
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--test-path", default=None)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--renderer-name", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lr-schedule", default="linear")
    parser.add_argument("--initial-baseline", type=float, default=0.0)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--ttl-seconds", type=int, default=604800)
    parser.add_argument("--rolling-save-every", type=int, default=0)
    parser.add_argument("--rolling-ttl-seconds", type=int, default=7200)
    parser.add_argument("--span-chart-every", type=int, default=0)
    parser.add_argument("--load-checkpoint-path", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--base-url", default=None,
        help="Override Tinker API endpoint (leave unset to use production API)",
    )
    parser.add_argument(
        "--reinforce-version",
        choices=("offline", "online", "artifact", "grpo-artifact"),
        default="offline",
        help=(
            'Data path: "offline" (logged trajectories), "online" (agentic rollout), '
            '"artifact" (single write rollout + grader), or "grpo-artifact" '
            "(grouped write rollouts + grader)."
        ),
    )
    parser.add_argument("--rollout-max-tokens", type=int, default=4096)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-attempts", type=int, default=1)
    parser.add_argument("--pair-mode", choices=("first_last", "adjacent"), default="first_last")
    parser.add_argument("--no-use-gt", action="store_true")
    parser.add_argument("--no-use-student", action="store_true")
    parser.add_argument(
        "--artifact-only-rollout-instruction",
        action="store_true",
        help="Append a rollout-only instruction requiring exactly one write(path, content) call.",
    )
    parser.add_argument(
        "--grader-model-name",
        default=None,
        help="External Tinker model used only for artifact reward grading.",
    )
    parser.add_argument(
        "--grader-renderer-name",
        default=None,
        help="Renderer for --grader-model-name; defaults to --renderer-name.",
    )
    parser.add_argument("--grader-max-tokens", type=int, default=4096)
    parser.add_argument("--grader-max-file-chars", type=int, default=14000)
    parser.add_argument("--grader-temperature", type=float, default=0.0)
    parser.add_argument("--grpo-group-size", type=int, default=4)
    parser.add_argument("--agentic-max-turns", type=int, default=48)
    parser.add_argument("--agentic-max-turns-per-step", type=int, default=8)
    parser.add_argument("--agentic-max-steps", type=int, default=6)
    parser.add_argument("--no-agentic-bash", action="store_true")
    parser.add_argument("--agentic-tool-timeout-s", type=int, default=20)
    parser.add_argument(
        "--agentic-max-trajectory-tokens", type=int, default=None,
        help="Skip rollout when the live prompt reaches this many tokens.",
    )
    parser.add_argument(
        "--no-log-rollout-samples", action="store_true",
        help="Disable reinforce_rollout_samples.jsonl / transcript logging.",
    )
    parser.add_argument("--rollout-sample-log-chars", type=int, default=4000)
    args = parser.parse_args()
    log_path = str(Path(args.log_path).expanduser())

    if args.reinforce_version in {"artifact", "grpo-artifact"}:
        renderer = renderers.get_renderer(args.renderer_name, get_tokenizer(args.model_name))
        config = _config_from_cli(
            args,
            reinforce_version=cast(Literal["artifact", "grpo-artifact"], args.reinforce_version),
            dataset_builder=None,
        )
        asyncio.run(run_artifact_policy_training(
            config,
            ArtifactReinforceRolloutDataset.from_weight_json(
                args.train_path,
                renderer,
                args.max_length,
                args.batch_size,
                pair_mode=args.pair_mode,
                use_gt=not args.no_use_gt,
                use_student=not args.no_use_student,
                artifact_only_instruction=args.artifact_only_rollout_instruction,
            ),
        ))
        return

    if args.reinforce_version == "online":
        config = _config_from_cli(args, reinforce_version="online", dataset_builder=None)
        asyncio.run(run_online_training(
            config,
            OnlineReinforceRolloutDataset.from_weight_json(args.train_path, args.batch_size),
        ))
        return

    ml_logger = ml_log.setup_logging(
        log_dir=log_path,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        config=vars(args),
        do_configure_logging_module=True,
    )
    model_info.warn_if_renderer_not_recommended(args.model_name, args.renderer_name)

    dataset_builder = WeightReinforceDataBuilder(
        train_path=args.train_path,
        test_path=args.test_path,
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=args.model_name,
            renderer_name=args.renderer_name,
            max_length=args.max_length,
            batch_size=args.batch_size,
        ),
    )
    dataset, _ = dataset_builder()
    n_batches = len(dataset)
    total_steps = n_batches * args.num_epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    resume_info = checkpoint_utils.get_last_checkpoint(log_path)
    start_epoch = resume_info.epoch or 0 if resume_info else 0
    start_batch = resume_info.batch or 0 if resume_info else 0

    user_metadata: dict[str, str] = {}
    if wandb_link := ml_logger.get_logger_url():
        user_metadata["wandb_link"] = wandb_link
    checkpoint_utils.add_renderer_name_to_user_metadata(user_metadata, args.renderer_name)

    service_client = tinker.ServiceClient(base_url=args.base_url)

    if resume_info:
        assert resume_info.state_path is not None
        checkpoint_utils.check_renderer_name_for_checkpoint(
            service_client, resume_info.state_path, args.renderer_name
        )
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info.state_path, user_metadata=user_metadata,
        )
        logger.info(f"Resumed REINFORCE training from {resume_info.state_path}")
    elif args.load_checkpoint_path:
        checkpoint_utils.check_renderer_name_for_checkpoint(
            service_client, args.load_checkpoint_path, args.renderer_name
        )
        training_client = service_client.create_training_client_from_state(
            args.load_checkpoint_path, user_metadata=user_metadata,
        )
        logger.info(f"Loaded weights from {args.load_checkpoint_path}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=args.model_name, rank=args.lora_rank, user_metadata=user_metadata,
        )

    tokenizer = get_tokenizer(args.model_name)
    rolling_mgr = checkpoint_utils.CheckpointManager(
        training_client=training_client,
        service_client=service_client,
        log_path=log_path,
        save_every=0,
        rolling_save_every=args.rolling_save_every,
        rolling_ttl_seconds=args.rolling_ttl_seconds,
    )

    logger.info(
        "Training for %d batches x %d epochs = %d steps",
        n_batches, args.num_epochs, n_batches * args.num_epochs,
    )

    baseline = _load_baseline_state(log_path, resume_info)
    if baseline is None:
        baseline = args.initial_baseline

    reached_max_steps = False
    for epoch_idx in range(start_epoch, args.num_epochs):
        logger.info(f"Starting epoch {epoch_idx}")
        dataset.set_epoch(seed=epoch_idx)

        for batch_idx in range(start_batch if epoch_idx == start_epoch else 0, n_batches):
            step = epoch_idx * n_batches + batch_idx
            if args.max_steps is not None and step >= args.max_steps:
                reached_max_steps = True
                break
            _, baseline = do_update(
                epoch_idx=epoch_idx,
                batch_idx=batch_idx,
                n_batches=n_batches,
                total_steps=total_steps,
                learning_rate_base=args.learning_rate,
                lr_schedule=args.lr_schedule,
                adam_beta1=args.adam_beta1,
                adam_beta2=args.adam_beta2,
                adam_eps=args.adam_eps,
                save_every=args.save_every,
                ttl_seconds=args.ttl_seconds,
                rolling_save_every=args.rolling_save_every,
                rolling_ttl_seconds=args.rolling_ttl_seconds,
                span_chart_every=args.span_chart_every,
                training_client=training_client,
                dataset=dataset,
                baseline=baseline,
                ml_logger=ml_logger,
                log_path=log_path,
                tokenizer=tokenizer,
                rolling_mgr=rolling_mgr,
            )
        if reached_max_steps:
            break

    did_train = start_epoch < args.num_epochs and (
        args.max_steps is None or start_epoch * n_batches + start_batch < args.max_steps
    )
    if did_train:
        checkpoint_utils.save_checkpoint(
            training_client=training_client,
            name="final",
            log_path=log_path,
            kind="both",
            loop_state={"epoch": args.num_epochs, "batch": 0},
            ttl_seconds=None,
        )
    else:
        logger.info("Training was already complete; nothing to do")
    rolling_mgr.finalize()
    ml_logger.close()
    logger.info("REINFORCE training completed successfully")


if __name__ == "__main__":
    main()
