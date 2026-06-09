"""Offline OPD (On-Policy Distillation) training on weight-format session JSON.

This script operates on **historical trajectories**:

1. Student datums: tokenized conversation (prompt + completion) with weights
   on assistant tokens.
2. Teacher datums: same completion appended to an augmented teacher prompt
   that includes privileged human feedback.
3. For each batch:
   a. Pre-compute teacher logprobs on teacher-forced sequences.
   b. Forward-backward on student datums with a custom loss that uses
      ``advantage = teacher_lp - student_lp`` to push the student towards
      the teacher's distribution.

This is functionally equivalent to our TRL OPSD implementation but uses the
Tinker training API.

Usage::

    python -m scripts.weight.train.run_opd \\
        --train-path data/weight.json \\
        --model-name Qwen/Qwen3-4B \\
        --renderer-name qwen3 \\
        --log-path ~/logs/opd_run
"""

from __future__ import annotations

import asyncio
import ast
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, cast

import chz
import tinker
import torch

from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
from tinker_cookbook.rl.rollouts import do_group_rollout_and_filter_constant_reward
from tinker_cookbook.rl.types import EnvGroupBuilder, StepResult
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.utils import ml_log, trace
from tinker_cookbook.utils.format_colorized import format_colorized
from tinker_cookbook.utils.lr_scheduling import LRSchedule, compute_schedule_lr_multiplier

try:  # Supports both `python -m weight...` from scripts/ and `python -m scripts.weight...`.
    from weight.data.extract import (  # type: ignore[import-not-found]
        OPD_REDO_MESSAGE,
        _session_tools_prefix,
        extract_opd_examples,
        extract_opd_examples_agentic,
        extract_opd_examples_v2,
    )
except ModuleNotFoundError:  # pragma: no cover - depends on invocation cwd
    from ..data.extract import (
        OPD_REDO_MESSAGE,
        _session_tools_prefix,
        extract_opd_examples,
        extract_opd_examples_agentic,
        extract_opd_examples_v2,
    )

try:
    from weight.data.reward import grade_sandbox_rubrics  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - depends on invocation cwd
    from ..data.reward import grade_sandbox_rubrics

from .formatter import OfflineOPDDataset, _hydrate_tool_calls, _load_sessions
from .tool_rollout_env import (
    FileToolset,
    SandboxAgentToolEnv,
    WorkspaceSandbox,
    zero_reward,
)

logger = logging.getLogger(__name__)


def _sequence_logprobs(sequence: Any, n_tokens: int) -> list[float] | None:
    """Best-effort extraction of per-sampled-token logprobs from Tinker output.

    Local copy of the helper in ``reinforce_rollout`` to avoid a circular import
    (``reinforce_rollout`` already imports from this module).
    """
    raw = getattr(sequence, "logprobs", None)
    if raw is None:
        raw = getattr(sequence, "token_logprobs", None)
    if raw is None:
        return None

    values: list[float] = []
    for item in list(raw):
        if item is None:
            values.append(0.0)
        elif isinstance(item, (int, float)):
            values.append(float(item))
        elif isinstance(item, dict):
            val = item.get("logprob", item.get("log_prob"))
            if not isinstance(val, (int, float)):
                return None
            values.append(float(val))
        else:
            val = getattr(item, "logprob", getattr(item, "log_prob", None))
            if not isinstance(val, (int, float)):
                return None
            values.append(float(val))

    if len(values) != n_tokens:
        return None
    return values


def _metrics_dict_from_result(metrics: Any) -> dict[str, Any]:
    if isinstance(metrics, dict):
        return dict(metrics)
    try:
        return dict(metrics.items())  # type: ignore[union-attr]
    except Exception:
        return {}


def _count_topk_supervision_tokens(
    topk_datums: list[tinker.Datum], topk: int | None = None,
) -> int:
    """Count supervised completion positions across a batch.

    Top-K datums store weights as a flattened ``(N, K)`` float tensor. The naive
    "count positives" approach over-counts by a factor of K (each supervised
    position has up to K positive entries). When ``topk`` is known, we reshape
    to ``(N, K)`` and count rows whose sum is non-zero. Fallback heuristics
    handle 1D weights (legacy IS path) and the rare case of unknown K.
    """
    total = 0
    for d in topk_datums:
        w = d.loss_fn_inputs["weights"].data
        t = torch.as_tensor(w, dtype=torch.float32)
        N = d.model_input.length
        if t.dim() == 2:
            total += int((t.abs().sum(dim=-1) > 1e-8).sum().item())
        elif t.dim() == 1 and t.numel() == N * (topk or 0) and topk:
            t2 = t.view(N, topk)
            total += int((t2.abs().sum(dim=-1) > 1e-8).sum().item())
        elif t.dim() == 1 and topk and t.numel() > N and t.numel() % N == 0:
            k_eff = t.numel() // N
            t2 = t.view(N, k_eff)
            total += int((t2.abs().sum(dim=-1) > 1e-8).sum().item())
        else:
            total += int((t > 0).sum().item())
    return max(total, 1)


# ---------------------------------------------------------------------------
# .env loader (shared with run_dpo / run_reinforce)
# ---------------------------------------------------------------------------

def _load_env() -> None:
    """Load key=value pairs from scripts/weight/.env into os.environ.

    No-op if the file is missing. Existing env vars take precedence so an
    explicit ``TINKER_API_KEY=... python -m ...`` invocation overrides .env.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


@chz.chz
class Config:
    """Configuration for offline OPD training."""

    log_path: str = chz.field(munger=lambda _, s: str(Path(s).expanduser()))
    model_name: str
    renderer_name: str | None = None
    lora_rank: int = 32
    base_url: str | None = None

    learning_rate: float = 2e-5
    lr_schedule: LRSchedule = "linear"
    num_epochs: int = 1

    save_every: int = 20
    ttl_seconds: int | None = 604800

    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    load_checkpoint_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    max_steps: int | None = None
    enable_trace: bool = False

    # Backend selection: True → SkyRL-compatible path that pre-computes teacher
    # logprobs via training_client.forward(); False (default) → real Tinker
    # cloud API (live teacher_client.compute_logprobs_async per batch).
    use_skyrl: bool = False

    # Data construction
    pair_mode: str = "first_last"   # "first_last" | "adjacent" (file-centric)
    use_gt: bool = True             # append last-version artifact to teacher prompt
    use_student: bool = True        # append student artifact to teacher prompt
    extract_version: str = "v2"

    # Agentic on-policy rollout (extract_version="agentic").  One continuous
    # multi-turn tool-using episode per session in a live ephemeral sandbox;
    # the whole trajectory is distilled against a teacher whose prompt is
    # augmented with the session's user follow-ups (see tool_rollout_env.py).
    # A session is rolled out as a planning turn + one "Proceed with: <step>"
    # user turn per planned leaf step (steps derived from the model's own plan).
    # ``agentic_max_turns`` is the overall safety ceiling across all segments;
    # ``agentic_max_turns_per_step`` caps the inner agent loop within one step,
    # and ``agentic_max_steps`` caps how many planned steps are replayed.
    agentic_max_turns: int = 48
    agentic_max_turns_per_step: int = 8
    agentic_max_steps: int = 6
    agentic_enable_bash: bool = True
    agentic_tool_timeout_s: int = 20
    agentic_max_trajectory_tokens: int | None = None

    # Top-K distillation (Tinker only; ignored when use_skyrl=True).
    # topk > 0 → forward KL distillation with K teacher vocabulary candidates
    #   per position; loss = -sum_k p_teacher(v_k) * log p_student(v_k).
    # topk = 0 → importance-sampling fallback (advantage = teacher_lp - student_lp).
    # K=20 matches full-vocabulary KL in practice (Shenfeld et al., 2026).
    topk: int = 20

    # Teacher distribution softening (Hinton 2015).  τ=1.0 → no change; τ>1 →
    # flattens teacher probs, raises effective teacher entropy, makes KD less
    # like hard-label SFT.  τ=1.5–2.0 recommended when teacher entropy < 0.5 nat.
    teacher_temperature: float = 1.0

    # Combined GRPO+OPD trainer (server ``mode="grpo_opd"``).  ``grpo_group_size``
    # on-policy episodes are sampled per session and reused for both the GRPO
    # importance-sampling loss and the OPD top-K cross-entropy loss; ``lambda_*``
    # weight the two halves before a single optimizer step (applied by scaling
    # GRPO advantages and OPD top-K weights respectively).
    grpo_group_size: int = 4
    lambda_grpo: float = 1.0
    lambda_opd: float = 1.0

    # Online artifact-only rollout.  When enabled, the historical artifact
    # completion is used only to identify the expected output path; student
    # completions are sampled on-policy from the current model and filtered to
    # exactly one write(path, content) tool call.
    online_rollout: bool = False
    rollout_max_tokens: int = 4096
    rollout_temperature: float = 1.0
    rollout_attempts: int = 1
    log_rollout_samples: bool = True
    rollout_sample_log_chars: int = 4000
    log_teacher_prompts: bool = True
    artifact_only_rollout_instruction: bool = False
    strip_thinking_from_history: bool = False

    # Rollout pipeline selection.  "current" (default) runs
    # ``sampling_client.sample_async`` directly per row and constructs a
    # supervised datum from the canonical assistant message produced by the
    # renderer (parse-filter + canonicalization in the loop).  "legacy"
    # routes through the cookbook ``do_group_rollout_and_filter_constant_reward``
    # + ``assemble_training_data`` path used by the legacy ``tinker_opd``
    # recipe: builds a ``_PromptOnlyEnv`` per row, runs the standard cookbook
    # rollout loop with group_size=1 and zero reward, and trains on the
    # raw sampled tokens (no parse-filter, no canonicalization, no retry).
    # Pick "legacy" to reproduce the original training dynamics.
    rollout_pipeline: str = "current"

    span_chart_every: int = 0

def _extract_completion_info(
    datum: tinker.Datum,
) -> tuple[list[int], list[int]]:
    """Extract completion token positions and completion tokens from a datum.

    Returns (completion_mask_indices, completion_tokens) where:
    - completion_mask_indices: positions in model_input where weight > 0
    - completion_tokens: the actual token IDs at those positions + 1
    """
    weights = datum.loss_fn_inputs["weights"].data
    mask_indices = [i for i, w in enumerate(weights) if w > 0]
    if not mask_indices:
        return [], []

    targets = datum.loss_fn_inputs["target_tokens"].data
    full_tokens = list(datum.model_input.to_ints())
    if targets:
        full_tokens.append(int(targets[-1]))
    completion_tokens = [full_tokens[i + 1] for i in mask_indices if i + 1 < len(full_tokens)]
    return mask_indices, completion_tokens


def _datum_fingerprint(datum: tinker.Datum) -> tuple:
    """Content-based stable key for caching per-datum results across shuffles."""
    mi = tuple(datum.model_input.to_ints())
    tt = tuple(datum.loss_fn_inputs["target_tokens"].data)
    return (mi, tt)


def _forward_teacher_logprobs(
    training_client: tinker.TrainingClient,
    teacher_datums: list[tinker.Datum],
) -> list[torch.Tensor]:
    """Run a gradient-free forward pass on teacher datums via the training client.

    Returns one tensor per datum with per-target-position logprobs (same
    shape as each teacher datum's ``target_tokens``). MUST be called before
    any ``optim_step`` so the policy equals the base / loaded checkpoint
    model — which plays the role of the frozen teacher in offline OPD.

    This replaces ``teacher_client.compute_logprobs_async(seq)``: SkyRL's
    vLLM backend does not yet support ``prompt_logprobs``, so the sampling
    path silently returns ``None``. The training-client forward path is
    the portable way to get per-token logprobs.
    """
    forward_result = training_client.forward(teacher_datums, "cross_entropy").result()
    out: list[torch.Tensor] = []
    for entry in forward_result.loss_fn_outputs:
        lp_data = entry["logprobs"]
        tensor = torch.tensor(lp_data.data, dtype=torch.float32)
        if lp_data.shape is not None:
            tensor = tensor.reshape(lp_data.shape)
        out.append(tensor.detach().cpu())
    return out


def _align_teacher_to_student(
    student_datum: tinker.Datum,
    teacher_datum: tinker.Datum,
    teacher_per_pos_lps: torch.Tensor,
) -> torch.Tensor:
    """Extract teacher logprobs at completion positions, aligned to the student's mask.

    The teacher's target_tokens index the teacher-forced sequence
    (teacher prompt + completion), while the student's weight mask
    indexes the student sequence. Both sides have the same completion
    *length*; we align by taking the first N teacher completion positions
    (those with weight > 0 on the teacher side) to match the N student
    completion positions.
    """
    weights = student_datum.loss_fn_inputs["weights"].data
    mask_indices = [j for j, w in enumerate(weights) if w > 0]
    n_completion = len(mask_indices)

    td_weights = teacher_datum.loss_fn_inputs["weights"].data
    td_mask_indices = [j for j, w in enumerate(td_weights) if w > 0]

    teacher_completion_lps: list[float] = []
    for t in range(min(n_completion, len(td_mask_indices))):
        pos = td_mask_indices[t]
        lp = float(teacher_per_pos_lps[pos]) if pos < len(teacher_per_pos_lps) else 0.0
        teacher_completion_lps.append(lp)
    while len(teacher_completion_lps) < n_completion:
        teacher_completion_lps.append(0.0)
    return torch.tensor(teacher_completion_lps[:n_completion], dtype=torch.float32)


def precompute_teacher_logprob_cache(
    training_client: tinker.TrainingClient,
    dataset: OfflineOPDDataset,
    num_epochs: int,
) -> dict[tuple, torch.Tensor]:
    """Pre-compute aligned teacher logprobs for every student datum we will train on.

    Iterates all epochs × batches and caches per-(student,teacher) aligned
    logprob tensors keyed by ``(student_fp, teacher_fp)``. Dedupes across
    epochs so each unique pair is computed exactly once. Must be called
    before any ``optim_step`` — ``training_client.forward()`` runs on the
    current policy, and we want the initial (teacher) policy.
    """
    cache: dict[tuple, torch.Tensor] = {}
    n_batches = len(dataset)
    for epoch_idx in range(num_epochs):
        dataset.set_epoch(seed=epoch_idx)
        for batch_idx in range(n_batches):
            student_datums, teacher_datums = dataset.get_batch(batch_idx)
            missing_idxs = [
                i for i, (sd, td) in enumerate(zip(student_datums, teacher_datums))
                if (_datum_fingerprint(sd), _datum_fingerprint(td)) not in cache
            ]
            if not missing_idxs:
                continue
            missing_teachers = [teacher_datums[i] for i in missing_idxs]
            per_pos_lps = _forward_teacher_logprobs(training_client, missing_teachers)
            for i, t_lp in zip(missing_idxs, per_pos_lps, strict=True):
                aligned = _align_teacher_to_student(
                    student_datums[i], teacher_datums[i], t_lp,
                )
                cache[(_datum_fingerprint(student_datums[i]), _datum_fingerprint(teacher_datums[i]))] = aligned
    logger.info(
        "Pre-computed teacher logprobs for %d unique (student, teacher) pairs "
        "(across %d epochs x %d batches)",
        len(cache), num_epochs, n_batches,
    )
    return cache


def _lookup_teacher_logprobs(
    student_datums: list[tinker.Datum],
    teacher_datums: list[tinker.Datum],
    cache: dict[tuple, torch.Tensor],
) -> list[torch.Tensor]:
    """Fetch aligned teacher logprobs for the current batch from the cache."""
    return [
        cache[(_datum_fingerprint(sd), _datum_fingerprint(td))]
        for sd, td in zip(student_datums, teacher_datums, strict=True)
    ]


async def _live_teacher_logprobs_async(
    teacher_client: tinker.SamplingClient,
    student_datums: list[tinker.Datum],
    teacher_datums: list[tinker.Datum],
) -> list[torch.Tensor]:
    """Tinker path: fetch teacher logprobs for one batch via a sampling client.

    Builds the full teacher-forced sequence (teacher prompt + completion
    tokens) for each datum and calls ``compute_logprobs_async`` in parallel.
    Returns one tensor per datum, aligned to the student's completion mask
    (so it can be consumed by the same ``do_update`` as the cached path).
    """
    full_sequences: list[tinker.ModelInput] = []
    for td in teacher_datums:
        targets = td.loss_fn_inputs["target_tokens"].data
        if targets:
            full_seq = td.model_input.append_int(int(targets[-1]))
        else:
            full_seq = td.model_input
        full_sequences.append(full_seq)

    raw_all = await asyncio.gather(
        *[teacher_client.compute_logprobs_async(seq) for seq in full_sequences]
    )

    aligned: list[torch.Tensor] = []
    for sd, td, raw in zip(student_datums, teacher_datums, raw_all, strict=True):
        # raw[0] is None; raw[1:][k] = lp(model_input[k+1] | model_input[0..k]).
        per_pos = torch.tensor(
            [lp if lp is not None else 0.0 for lp in raw[1:]], dtype=torch.float32,
        )
        aligned.append(_align_teacher_to_student(sd, td, per_pos))
    return aligned


def _live_teacher_logprobs(
    teacher_client: tinker.SamplingClient,
    student_datums: list[tinker.Datum],
    teacher_datums: list[tinker.Datum],
) -> list[torch.Tensor]:
    """Synchronous wrapper around :func:`_live_teacher_logprobs_async`."""
    return asyncio.run(
        _live_teacher_logprobs_async(teacher_client, student_datums, teacher_datums)
    )


# ---------------------------------------------------------------------------
# Top-K distillation helpers (Tinker only)
# ---------------------------------------------------------------------------

async def _build_offline_topk_datums_async(
    student_datums: list[tinker.Datum],
    teacher_prompt_inputs: list[tinker.ModelInput],
    teacher_client: tinker.SamplingClient,
    topk: int = 20,
    max_context_length: int = 32768,
    vocab_size: int | None = None,
    skip_first_n_tokens: int = 3,
    teacher_temperature: float = 1.0,
) -> tuple[list[tinker.Datum], dict[str, float]]:
    """Build cross_entropy datums with top-K teacher soft targets (offline version).

    Teacher prompts are pre-built (stored in the dataset) and map
    1-to-1 with student datums (no group-index indirection needed).

    For each student datum:
    1. Extract completion tokens from the student datum's mask.
    2. Append them to the teacher prompt to build a teacher-forced sequence.
    3. Call ``teacher_client.sample_async(topk_prompt_logprobs=K)`` in parallel.
    4. For each completion position, read top-K (token_id, log_prob) pairs,
       renormalise over K, and write them into an ``(N, K)`` shaped datum with
       ``cross_entropy`` loss semantics.

    Returns (new_datums, metrics). New datums have ``target_tokens`` shape
    ``(N, K)`` and ``weights`` shape ``(N, K)`` — Tinker's CE loss consumes
    these directly when called via ``forward_backward("cross_entropy")``.
    """
    teacher_forced_seqs: list[tinker.ModelInput] = []
    teacher_prompt_lens: list[int] = []
    completion_lens: list[int] = []
    truncated_count = 0

    for sd, tp in zip(student_datums, teacher_prompt_inputs, strict=True):
        weights = sd.loss_fn_inputs["weights"].data
        mask_indices = [i for i, w in enumerate(weights) if w > 0]
        tp_len = tp.length

        if not mask_indices:
            teacher_forced_seqs.append(tp)
            teacher_prompt_lens.append(tp_len)
            completion_lens.append(0)
            continue

        # Reconstruct full student sequence and extract completion tokens.
        targets = sd.loss_fn_inputs["target_tokens"].data
        full_tokens = list(sd.model_input.to_ints())
        if targets:
            full_tokens.append(int(targets[-1]))
        comp_start = mask_indices[0] + 1
        comp_tokens = full_tokens[comp_start:]

        available = max_context_length - tp_len
        was_truncated = False
        if available <= 0:
            comp_tokens = []
            was_truncated = True
        elif len(comp_tokens) > available:
            comp_tokens = comp_tokens[:available]
            was_truncated = True
        if was_truncated:
            truncated_count += 1

        teacher_forced = tp
        for tok in comp_tokens:
            teacher_forced = teacher_forced.append_int(tok)

        teacher_forced_seqs.append(teacher_forced)
        teacher_prompt_lens.append(tp_len)
        completion_lens.append(len(comp_tokens))

    # Parallel top-K logprob queries.
    topk_responses = await asyncio.gather(
        *[
            teacher_client.sample_async(
                prompt=tf_seq,
                num_samples=1,
                sampling_params=tinker.SamplingParams(max_tokens=1),
                include_prompt_logprobs=True,
                topk_prompt_logprobs=topk,
            )
            for tf_seq in teacher_forced_seqs
        ]
    )

    # Build (N, K) shaped datums.
    total_completion_tokens = 0.0
    total_teacher_entropy = 0.0
    new_datums: list[tinker.Datum] = []

    for i, sd in enumerate(student_datums):
        weights_1d = sd.loss_fn_inputs["weights"].data
        mask_indices = [j for j, w in enumerate(weights_1d) if w > 0]
        N = sd.model_input.length
        comp_len = completion_lens[i]
        tp_len = teacher_prompt_lens[i]

        target_tokens_NK = torch.zeros(N, topk, dtype=torch.long)
        weights_NK = torch.zeros(N, topk, dtype=torch.float32)

        if comp_len > 0 and mask_indices:
            topk_all = topk_responses[i].topk_prompt_logprobs
            n_tokens = min(comp_len, len(mask_indices))

            for t in range(n_tokens):
                if t < skip_first_n_tokens:
                    continue
                teacher_pos = tp_len + t
                student_pos = mask_indices[t]

                if topk_all is None or teacher_pos >= len(topk_all):
                    continue
                topk_entries = topk_all[teacher_pos]
                if not topk_entries:
                    continue

                filtered = [
                    (tok_id, lp)
                    for tok_id, lp in topk_entries[:topk]
                    if vocab_size is None or tok_id < vocab_size
                ]
                if not filtered:
                    continue

                k_actual = len(filtered)
                token_ids = torch.tensor([tid for tid, _ in filtered], dtype=torch.long)
                logprobs = torch.tensor([lp for _, lp in filtered], dtype=torch.float32)
                # Teacher temperature softening: scale log-probs by 1/τ before
                # renormalising → flatter distribution when τ > 1.
                if teacher_temperature != 1.0:
                    logprobs = logprobs / teacher_temperature
                logprobs -= torch.logsumexp(logprobs, dim=0)
                probs = logprobs.exp()

                target_tokens_NK[student_pos, :k_actual] = token_ids
                weights_NK[student_pos, :k_actual] = probs
                total_teacher_entropy += -(probs * logprobs).sum().item()

            total_completion_tokens += n_tokens

        new_datums.append(tinker.Datum(
            model_input=sd.model_input,
            loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_torch(target_tokens_NK),
                "weights": tinker.TensorData.from_torch(weights_NK),
            },
        ))

    metrics: dict[str, float] = {
        "opd/topk_truncated": float(truncated_count),
        "opd/topk_num_datums": float(len(student_datums)),
        "opd/topk_k": float(topk),
    }
    if total_completion_tokens > 0:
        metrics["opd/total_completion_tokens"] = total_completion_tokens
        metrics["opd/mean_teacher_entropy"] = total_teacher_entropy / total_completion_tokens
    return new_datums, metrics


def build_offline_topk_datums(
    student_datums: list[tinker.Datum],
    teacher_prompt_inputs: list[tinker.ModelInput],
    teacher_client: tinker.SamplingClient,
    topk: int = 20,
    max_context_length: int = 32768,
    vocab_size: int | None = None,
    teacher_temperature: float = 1.0,
) -> tuple[list[tinker.Datum], dict[str, float]]:
    """Synchronous wrapper around :func:`_build_offline_topk_datums_async`."""
    return asyncio.run(
        _build_offline_topk_datums_async(
            student_datums, teacher_prompt_inputs, teacher_client,
            topk=topk, max_context_length=max_context_length, vocab_size=vocab_size,
            teacher_temperature=teacher_temperature,
        )
    )


def precompute_all_topk_datums(
    dataset: "OfflineOPDDataset",
    teacher_client: tinker.SamplingClient,
    topk: int = 20,
    max_context_length: int = 32768,
    vocab_size: int | None = None,
    teacher_temperature: float = 1.0,
) -> tuple[list[tinker.Datum], dict[str, float]]:
    """Pre-compute top-K datums for every example in the dataset (offline optimisation).

    Since the teacher is frozen and completions are fixed (offline, not on-policy),
    top-K targets are identical across epochs and can be computed once upfront,
    saving repeated API calls. Returns ``(datums, metrics)`` where ``datums`` is
    aligned 1-to-1 with the dataset's internal un-shuffled order; use
    ``dataset.get_batch_topk_datums(batch_idx)`` to retrieve shuffled batches
    during training. ``metrics`` contains ``opd/mean_teacher_entropy`` etc.
    """
    all_student, _ = zip(
        *[dataset.get_batch(i) for i in range(len(dataset))],
    ) if len(dataset) > 0 else ([], [])
    all_student_flat = [d for batch in all_student for d in batch]
    teacher_prompt_inputs = dataset.get_all_teacher_prompt_inputs()
    if not all_student_flat:
        empty_metrics: dict[str, float] = {
            "opd/topk_truncated": 0.0,
            "opd/topk_num_datums": 0.0,
            "opd/topk_k": float(topk),
        }
        return [], empty_metrics
    logger.info(
        "Pre-computing top-K=%d teacher targets for %d OPD examples",
        topk, len(all_student_flat),
    )
    topk_datums, metrics = build_offline_topk_datums(
        all_student_flat, teacher_prompt_inputs, teacher_client,
        topk=topk, max_context_length=max_context_length, vocab_size=vocab_size,
        teacher_temperature=teacher_temperature,
    )
    logger.info("Top-K pre-computation done: %s", metrics)
    return topk_datums, metrics


def do_update_topk(
    step: int,
    total_steps: int,
    config: "Config",
    training_client: tinker.TrainingClient,
    topk_datums: list[tinker.Datum],
    ml_logger: "ml_log.Logger",
    log_path: str,
    static_teacher_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Single training step for top-K distillation.

    Submits pre-built ``(N, K)`` datums directly to Tinker's built-in
    ``cross_entropy`` loss, which computes
    ``-sum_k p_teacher(v_k|t) * log p_student(v_k|t)`` server-side.
    No custom Python loss needed — the CE loss over (N,K) targets is natively
    supported by the Tinker training API.
    """
    if config.save_every > 0 and step % config.save_every == 0 and step > 0:
        checkpoint_utils.save_checkpoint(
            training_client=training_client,
            name=f"{step:06d}",
            log_path=log_path,
            kind="both",
            loop_state={"batch": step},
            ttl_seconds=config.ttl_seconds,
        )

    learning_rate = config.learning_rate * compute_schedule_lr_multiplier(
        lr_schedule=config.lr_schedule, step=step, total_steps=total_steps,
    )
    adam_params = tinker.AdamParams(
        learning_rate=learning_rate,
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        eps=config.adam_eps,
    )

    result = training_client.forward_backward(topk_datums, "cross_entropy").result()
    training_client.optim_step(adam_params).result()

    rm = _metrics_dict_from_result(result.metrics)
    loss_sum = float(rm.get("loss:sum", rm.get("loss", 0.0)))
    batch_tokens = _count_topk_supervision_tokens(topk_datums, topk=config.topk)
    per_token_ce = loss_sum / float(batch_tokens)

    metrics: dict[str, Any] = {
        # Per-token CE comparable across batches (loss:sum is not).
        "opd_loss": per_token_ce,
        "opd/loss_sum": loss_sum,
        "opd/batch_completion_tokens": float(batch_tokens),
        "opd/per_token_ce": per_token_ce,
        "num_examples": len(topk_datums),
        "learning_rate": learning_rate,
        "progress": step / total_steps,
    }
    metrics.update(rm)
    if static_teacher_metrics:
        metrics.update({k: float(v) for k, v in static_teacher_metrics.items()})
    ml_logger.log_metrics(metrics=metrics, step=step)
    return metrics


# ---------------------------------------------------------------------------
# Online artifact-only rollout
# ---------------------------------------------------------------------------

def _tool_call_name_and_args(tool_call: Any) -> tuple[str | None, str | None]:
    """Extract (function_name, arguments_json) from dict or ToolCall-like objects."""
    if isinstance(tool_call, dict):
        fn = tool_call.get("function", {})
        if isinstance(fn, dict):
            return fn.get("name"), fn.get("arguments")
        name = getattr(fn, "name", None)
        args = getattr(fn, "arguments", None)
        return name, args
    fn = getattr(tool_call, "function", None)
    name = getattr(fn, "name", None)
    args = getattr(fn, "arguments", None)
    return name, args


def _completion_expected_path(completion: list[dict]) -> str | None:
    """Read the expected artifact path from an offline artifact-only completion."""
    if not completion:
        return None
    tool_calls = completion[0].get("tool_calls") or []
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
    return path if isinstance(path, str) and path else None


def _with_artifact_only_instruction(
    prompt: list[dict[str, Any]],
    expected_path: str,
) -> list[dict[str, Any]]:
    """Append an optional rollout-only artifact instruction to the last user turn.

    This is disabled by default because it intentionally changes the online
    rollout prompt relative to inference.  It is useful for A/B smoke runs that
    measure the upper bound when the student is explicitly constrained to write.
    """
    if not prompt:
        return prompt
    out = [dict(m) for m in prompt]
    path_hint = (
        "<a suitable artifact filename>"
        if _is_unknown_artifact_placeholder(expected_path)
        else expected_path
    )
    suffix = (
        "\n\nFor this rollout, respond with exactly one assistant tool call:\n"
        f'write({{"path": "{path_hint}", "content": "<complete final file content>"}})\n'
        "Do not call read, bash, edit, grep, find, ls, workflow_plan, or ask_user_question. "
        "Do not include prose."
    )
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            out[i]["content"] = (out[i].get("content") or "") + suffix
            return out
    out.append({"role": "user", "content": suffix.strip()})
    return out


def _is_unknown_artifact_placeholder(path: str) -> bool:
    return path in {"<inline_script>.py", "<inline_script>"}


def _artifact_path_matches(expected_path: str, actual_path: Any) -> bool:
    # Online OPD trains artifact content, not the output filename.  Keep the
    # expected/sampled paths for logging, but do not filter otherwise valid
    # artifact samples on path mismatches or missing paths.
    return True


_BASH_CAT_WRITE_HEREDOC_RE = re.compile(
    r"""cat\s+>\s+['"]?([^'"\s<>\n]+)['"]?\s+<<\s*['"]?(\w+)['"]?\n(.*?)\n\2(?:\n|$)""",
    re.DOTALL,
)

_BASH_CAT_WRITE_HEREDOC_REVERSED_RE = re.compile(
    r"""cat\s+<<\s*['"]?(\w+)['"]?\s*>\s*['"]?([^'"\s<>\n]+)['"]?\n(.*?)\n\1(?:\n|$)""",
    re.DOTALL,
)

_BASH_TEE_WRITE_HEREDOC_RE = re.compile(
    r"""tee\s+['"]?([^'"\s<>\n]+)['"]?\s+<<\s*['"]?(\w+)['"]?\n(.*?)\n\2(?:\n|$)""",
    re.DOTALL,
)

_BASH_PYTHON_INLINE_HEREDOC_RE = re.compile(
    r"""python3?(?:\s+-)?\s+<<\s*['"]?(\w+)['"]?\n(.*?)\n\1(?:\n|$)""",
    re.DOTALL,
)

_BASH_PYTHON_INLINE_C_RE = re.compile(
    r"""python3?(?:\s+-[\w]+)*\s+-c\s+(?P<quoted>'(?:\\.|[^'])*'|"(?:\\.|[^"])*")""",
    re.DOTALL,
)

_BASH_CAT_WRITE_HEREDOC_REVERSED_LENIENT_RE = re.compile(
    r"""^\s*cat\s+<<\s*['"]?(\w+)['"]?\s*>\s*['"]?([^'"\s<>\n]+)['"]?\n(.*)\s*$""",
    re.DOTALL,
)

_BASH_PYTHON_INLINE_HEREDOC_LENIENT_RE = re.compile(
    r"""^\s*python3?(?:\s+-)?\s+<<\s*['"]?(\w+)['"]?\n(.*)\s*$""",
    re.DOTALL,
)


def _strip_optional_heredoc_delimiter(content: str, delimiter: str) -> str:
    """Trim a trailing heredoc delimiter when a lenient match captured it."""
    lines = content.splitlines()
    if lines and lines[-1].strip() == delimiter:
        return "\n".join(lines[:-1])
    return content


def _python_c_args(command: str) -> dict[str, str] | None:
    """Parse a simple ``python -c "..."`` command as a synthetic source artifact."""
    m = _BASH_PYTHON_INLINE_C_RE.search(command)
    if not m:
        return None
    try:
        content = ast.literal_eval(m.group("quoted"))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(content, str) or not content.strip():
        return None
    return {"path": "<inline_script>.py", "content": content}


def _bash_heredoc_write_args(command: Any) -> dict[str, str] | None:
    """Parse a narrow bash heredoc write into write(path, content) args.

    This intentionally does not execute shell.  It only accepts obvious
    artifact-producing heredocs:
      cat > artifact <<EOF
      cat <<EOF > artifact
      tee artifact <<EOF
      python3 <<EOF
      python3 -c "..."

    Append forms and arbitrary redirections are left filtered.  Inline Python
    scripts are treated as synthetic source artifacts, regardless of the image
    path they may save at runtime.
    """
    if not isinstance(command, str):
        return None
    matches: list[tuple[str, str]] = []
    for rx in (_BASH_CAT_WRITE_HEREDOC_RE, _BASH_TEE_WRITE_HEREDOC_RE):
        for m in rx.finditer(command):
            path = m.group(1).strip()
            content = m.group(3)
            if path and content.strip():
                matches.append((path, content))
    for m in _BASH_CAT_WRITE_HEREDOC_REVERSED_RE.finditer(command):
        path = m.group(2).strip()
        content = m.group(3)
        if path and content.strip():
            matches.append((path, content))
    for m in _BASH_PYTHON_INLINE_HEREDOC_RE.finditer(command):
        content = m.group(2)
        if content.strip():
            matches.append(("<inline_script>.py", content))
    python_c = _python_c_args(command)
    if python_c is not None:
        matches.append((python_c["path"], python_c["content"]))

    # Some tool-rendered bash calls omit the shell heredoc closing delimiter but
    # still contain the source body as the tool argument. Salvage only the two
    # narrow artifact-producing shapes; pure probes such as ls/pip/read remain
    # filtered.
    if not matches:
        m = _BASH_CAT_WRITE_HEREDOC_REVERSED_LENIENT_RE.match(command)
        if m:
            delimiter = m.group(1)
            path = m.group(2).strip()
            content = _strip_optional_heredoc_delimiter(m.group(3), delimiter)
            if path and content.strip():
                matches.append((path, content))
    if not matches:
        m = _BASH_PYTHON_INLINE_HEREDOC_LENIENT_RE.match(command)
        if m:
            delimiter = m.group(1)
            content = _strip_optional_heredoc_delimiter(m.group(2), delimiter)
            if content.strip():
                matches.append(("<inline_script>.py", content))
    if len(matches) != 1:
        return None
    path, content = matches[0]
    return {"path": path, "content": content}


def _parse_valid_artifact_write_message(
    renderer: renderers.Renderer,
    tokens: list[int],
    expected_path: str,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Parse and normalize a valid one-write artifact response.

    The sampled raw text may include thinking or explanatory prose before the
    tool call.  We only use the parsed write call for training so online OPD
    remains artifact-only even when the sampler is chatty.
    """
    try:
        message, parse_success = renderer.parse_response(tokens)
    except Exception as e:  # noqa: BLE001
        return False, f"parse_exception:{type(e).__name__}", None
    if not parse_success:
        return False, "parse_failed", None
    tool_calls = message.get("tool_calls") or []
    if len(tool_calls) != 1:
        return False, f"tool_call_count:{len(tool_calls)}", None
    name, args_text = _tool_call_name_and_args(tool_calls[0])
    if not isinstance(args_text, str):
        return False, "arguments_not_string", None
    try:
        parsed_args = json.loads(args_text)
    except json.JSONDecodeError:
        return False, "arguments_json_error", None
    if not isinstance(parsed_args, dict):
        return False, "arguments_not_object", None

    if name == "write":
        args = parsed_args
        reason = "valid"
    elif name == "bash":
        args = _bash_heredoc_write_args(parsed_args.get("command"))
        if args is None:
            return False, "tool_name:bash", None
        reason = "valid_bash_heredoc"
    else:
        return False, f"tool_name:{name}", None

    content = args.get("content")
    if not isinstance(content, str) or not content.strip():
        return False, "empty_content", None

    canonical_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "type": "function",
            "id": getattr(tool_calls[0], "id", None),
            "function": {
                "name": "write",
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        }],
    }
    return True, reason, canonical_message


def _sample_is_valid_artifact_write(
    renderer: renderers.Renderer,
    tokens: list[int],
    expected_path: str,
) -> tuple[bool, str]:
    """Accept only one parsed write(path, non-empty content) call."""
    ok, reason, _message = _parse_valid_artifact_write_message(
        renderer, tokens, expected_path,
    )
    return ok, reason


def _parse_v2_message(
    renderer: renderers.Renderer,
    tokens: list[int],
) -> tuple[bool, str, dict[str, Any] | None]:
    """v2 (per-assistant-message) on-policy filter.

    Unlike :func:`_parse_valid_artifact_write_message` (v1, artifact-only),
    v2 trains on the verbatim assistant turn — which can be free-form text,
    any tool call (bash/edit/read/grep/...), or multiple tool calls. The
    only contract is that the renderer can successfully parse the sampled
    tokens back into a well-formed assistant message.

    Returns the parsed message with every tool call normalised into the
    plain-dict shape :func:`weight.train.formatter._hydrate_tool_calls`
    expects — ``{"type": "function", "id": ..., "function": {"name":
    ..., "arguments": ...}}`` — because ``renderer.parse_response``
    returns ``tool_calls`` as pydantic ``ToolCall`` objects that the
    hydrator (designed for on-disk dict-form session JSON) does not know
    how to read. No write canonicalisation, no bash heredoc salvage, no
    tool-call count check. An assistant turn that contains neither prose
    nor any tool_calls is rejected because it carries no learning signal.
    """
    try:
        message, parse_success = renderer.parse_response(tokens)
    except Exception as e:  # noqa: BLE001
        return False, f"parse_exception:{type(e).__name__}", None
    if not parse_success:
        return False, "parse_failed", None
    content = message.get("content")
    raw_tool_calls = message.get("tool_calls") or []
    has_text = isinstance(content, str) and content.strip() != ""
    if not has_text and not raw_tool_calls:
        return False, "empty_message", None

    normalized_tool_calls: list[dict[str, Any]] = []
    for tc in raw_tool_calls:
        name, args_text = _tool_call_name_and_args(tc)
        if name is None or not isinstance(args_text, str):
            return False, "tool_call_malformed", None
        tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
        normalized_tool_calls.append({
            "type": "function",
            "id": tc_id,
            "function": {"name": name, "arguments": args_text},
        })

    normalized: dict[str, Any] = {
        "role": message.get("role", "assistant"),
        "content": content if isinstance(content, str) else "",
    }
    if normalized_tool_calls:
        normalized["tool_calls"] = normalized_tool_calls
    return True, "valid", normalized


def _summarize_sample(
    renderer: renderers.Renderer,
    tokens: list[int],
    expected_path: str | None,
    ok: bool,
    reason: str,
    row_index: int,
    attempt_index: int,
    step: int,
    max_chars: int,
    extract_version: str = "v1",
    teacher_prompt_input: tinker.ModelInput | None = None,
) -> dict[str, Any]:
    """Build a JSONL-safe diagnostic record for one online rollout sample.

    Under ``extract_version="v1"`` the returned record is bit-identical to
    the original (no ``extract_version`` key; ``expected_path`` is the
    v1 artifact path string). Under ``"v2"`` the record adds an
    ``extract_version`` field and ``expected_path`` is ``None`` because
    v2 does not derive an artifact path from the historical completion.

    When ``teacher_prompt_input`` is provided, the record additionally
    carries ``teacher_prompt_preview`` (decoded, truncated to
    ``max_chars``) and ``teacher_prompt_token_count`` so the prompt that
    was scored against this student rollout is recoverable for offline
    analysis. The teacher prompt is logged only when the caller opts in
    (default off) because it can run up to the model's full context
    length.
    """
    raw_text = str(renderer.tokenizer.decode(tokens))
    parsed_tool_names: list[str] = []
    parsed_path: str | None = None
    parsed_content_preview: str | None = None
    parse_success = False
    try:
        message, parse_success = renderer.parse_response(tokens)
        for tc in message.get("tool_calls") or []:
            name, args_text = _tool_call_name_and_args(tc)
            parsed_tool_names.append(str(name))
            if isinstance(args_text, str):
                try:
                    args = json.loads(args_text)
                    if name == "bash" and isinstance(args, dict):
                        args = _bash_heredoc_write_args(args.get("command")) or args
                    path = args.get("path") if isinstance(args, dict) else None
                    content = args.get("content") if isinstance(args, dict) else None
                    if isinstance(path, str):
                        parsed_path = path
                    if isinstance(content, str):
                        parsed_content_preview = content[:max_chars]
                except json.JSONDecodeError:
                    pass
    except Exception as e:  # noqa: BLE001
        parsed_tool_names.append(f"parse_exception:{type(e).__name__}")

    record: dict[str, Any] = {
        "step": step,
        "row_index": row_index,
        "attempt_index": attempt_index,
        "expected_path": expected_path,
        "valid": ok,
        "reason": reason,
        "parse_success": parse_success,
        "parsed_tool_names": parsed_tool_names,
        "parsed_path": parsed_path,
        "parsed_content_preview": parsed_content_preview,
        "raw_text_preview": raw_text[:max_chars],
        "raw_token_count": len(tokens),
    }
    if extract_version != "v1":
        record["extract_version"] = extract_version
    if teacher_prompt_input is not None:
        try:
            teacher_tokens = list(teacher_prompt_input.to_ints())
            teacher_text = str(renderer.tokenizer.decode(teacher_tokens))
            record["teacher_prompt_preview"] = teacher_text[:max_chars]
            record["teacher_prompt_token_count"] = len(teacher_tokens)
        except Exception as e:  # noqa: BLE001
            record["teacher_prompt_preview"] = None
            record["teacher_prompt_token_count"] = None
            record["teacher_prompt_error"] = f"{type(e).__name__}: {e}"
    return record


def _datum_from_prompt_and_sample_tokens(
    prompt_input: tinker.ModelInput,
    sampled_tokens: list[int],
    max_length: int | None,
) -> tinker.Datum | None:
    """Build an exact SFT-style datum from prompt tokens + sampled completion tokens."""
    if not sampled_tokens:
        return None
    prompt_tokens = list(prompt_input.to_ints())
    full_tokens = prompt_tokens + list(sampled_tokens)
    if len(full_tokens) < 2:
        return None
    if max_length is not None and len(full_tokens) > max_length:
        return None

    model_input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]

    # Target index j predicts full_tokens[j + 1].  Completion tokens start at
    # full_tokens[prompt_len], so the first supervised target index is
    # prompt_len - 1.
    first_completion_target_idx = max(len(prompt_tokens) - 1, 0)
    weights = [
        1.0 if i >= first_completion_target_idx else 0.0
        for i in range(len(target_tokens))
    ]
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(model_input_tokens),
        loss_fn_inputs={
            "weights": tinker.TensorData(
                data=weights,
                dtype="float32",
                shape=[len(weights)],
            ),
            "target_tokens": tinker.TensorData(
                data=target_tokens,
                dtype="int64",
                shape=[len(target_tokens)],
            ),
        },
    )


def _datum_from_prompt_and_assistant_message(
    renderer: renderers.Renderer,
    prompt_messages: list[dict[str, Any]],
    assistant_message: dict[str, Any],
    max_length: int | None,
) -> tinker.Datum | None:
    """Build a supervised datum from prompt plus normalized assistant write."""
    model_input, weights = renderer.build_supervised_example(
        prompt_messages + _hydrate_tool_calls([assistant_message]),
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    if max_length is not None and model_input.length > max_length:
        return None
    full_tokens = list(model_input.to_ints())
    if len(full_tokens) < 2:
        return None
    target_tokens = full_tokens[1:]
    shifted_weights = weights[1 : len(target_tokens) + 1]
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(full_tokens[:-1]),
        loss_fn_inputs={
            "weights": tinker.TensorData(
                data=[float(w) for w in shifted_weights.tolist()],
                dtype="float32",
                shape=[len(target_tokens)],
            ),
            "target_tokens": tinker.TensorData(
                data=target_tokens,
                dtype="int64",
                shape=[len(target_tokens)],
            ),
        },
    )


class OnlineOPDRolloutDataset:
    """Weight-format OPD examples prepared for on-policy artifact rollout."""

    def __init__(
        self,
        examples: list[dict[str, Any]],
        renderer: renderers.Renderer,
        max_length: int | None,
        batch_size: int,
        artifact_only_instruction: bool = False,
        extract_version: str = "v1",
    ):
        if extract_version == "v2" and artifact_only_instruction:
            logger.warning(
                "artifact_only_instruction=True is v1-only (it instructs the "
                "model to emit exactly one write() call) and is incompatible "
                "with extract_version='v2'. Forcing artifact_only_instruction "
                "to False for this dataset.",
            )
            artifact_only_instruction = False

        rows: list[dict[str, Any]] = []
        if extract_version == "agentic":
            # Agentic rows carry only the rollout seed; the student trajectory
            # is generated live at train time and tokenised post-rollout, so we
            # do NOT pre-tokenise a student/teacher prompt here.
            for ex in examples:
                rows.append({
                    "prompt_messages": _hydrate_tool_calls(ex["prompt_messages"]),
                    "system_prompt": ex.get("system_prompt", "") or "",
                    "tool_schemas": ex.get("tool_schemas"),
                    "golden_chat": _hydrate_tool_calls(ex.get("golden_chat") or []),
                    "rubrics": ex.get("rubrics") or [],
                    "meta": ex.get("meta") or {},
                })
            self._rows = rows
            self._renderer = renderer
            self._max_length = max_length
            self._batch_size = batch_size
            self._extract_version = extract_version
            self._indices = list(range(len(rows)))
            return
        for ex in examples:
            if extract_version == "v2":
                # v2 trains on the verbatim assistant turn; the historical
                # completion is not necessarily an artifact write, so we don't
                # derive an expected path or filter rows on it.
                expected_path = None
            else:
                expected_path = _completion_expected_path(ex.get("completion", []))
                if expected_path is None:
                    continue
            student_prompt = ex["student_prompt"]
            if artifact_only_instruction:
                assert expected_path is not None  # guarded above for v2
                student_prompt = _with_artifact_only_instruction(
                    student_prompt, expected_path,
                )
            student_prompt_messages = _hydrate_tool_calls(student_prompt)
            student_prompt_input = renderer.build_generation_prompt(
                student_prompt_messages
            )
            teacher_prompt_input = renderer.build_generation_prompt(
                _hydrate_tool_calls(ex["teacher_prompt"])
            )
            rows.append({
                "student_prompt_messages": student_prompt_messages,
                "student_prompt_input": student_prompt_input,
                "teacher_prompt_input": teacher_prompt_input,
                "expected_path": expected_path,
            })
        self._rows = rows
        self._renderer = renderer
        self._max_length = max_length
        self._batch_size = batch_size
        self._extract_version = extract_version
        self._indices = list(range(len(rows)))

    @classmethod
    def from_weight_json(
        cls,
        path: str,
        renderer: renderers.Renderer,
        max_length: int | None,
        batch_size: int,
        pair_mode: str = "first_last",
        use_gt: bool = True,
        use_student: bool = True,
        artifact_only_instruction: bool = False,
        extract_version: str = "v2",
    ) -> "OnlineOPDRolloutDataset":
        if extract_version == "agentic":
            examples = extract_opd_examples_agentic(
                _load_sessions(path),
                renderer=renderer,
            )
        else:
            extract_fn = (
                extract_opd_examples_v2 if extract_version == "v2"
                else extract_opd_examples
            )
            examples = extract_fn(
                _load_sessions(path),
                renderer=renderer,
                pair_mode=pair_mode,
                use_gt=use_gt,
                use_student=use_student,
            )
        dataset = cls(
            examples,
            renderer,
            max_length,
            batch_size,
            artifact_only_instruction=artifact_only_instruction,
            extract_version=extract_version,
        )
        logger.info(
            "Loaded %d online OPD rollout examples from %s "
            "(extract=%s, raw=%d, pair_mode=%s, use_gt=%s, use_student=%s)",
            len(dataset._rows), path, extract_version, len(examples),
            pair_mode, use_gt, use_student,
        )
        return dataset

    def __len__(self) -> int:
        if not self._rows:
            return 0
        return (len(self._rows) + self._batch_size - 1) // self._batch_size

    def set_epoch(self, seed: int) -> None:
        import random

        rng = random.Random(seed)
        self._indices = list(range(len(self._rows)))
        rng.shuffle(self._indices)

    def get_batch(self, index: int) -> list[dict[str, Any]]:
        start = index * self._batch_size
        end = min(start + self._batch_size, len(self._indices))
        return [self._rows[self._indices[i]] for i in range(start, end)]


# ---------------------------------------------------------------------------
# Legacy-pipeline rollout adapters
# ---------------------------------------------------------------------------
#
# These adapters route v2 rollout requests through the cookbook's standard
# group-rollout machinery (``do_group_rollout_and_filter_constant_reward`` +
# ``assemble_training_data``).  This is the same pipeline the legacy
# ``tinker_opd`` recipe uses via ``_OPDEnvGroupBuilder`` + ``_PromptOnlyEnv``
# (see ``scripts/tinker_formatter.py`` in the legacy tree).  The point is to
# train on the *raw* sampled tokens (no parse-filter, no canonicalization, no
# retry) so the policy-gradient/distillation signal sees exactly what the
# student emitted, matching legacy training dynamics bit-for-bit.


class _OPDV2PromptOnlyEnv:
    """Minimal single-turn, zero-reward env that emits ``prompt_input``.

    Mirrors the legacy ``_PromptOnlyEnv`` so the cookbook rollout loop can
    drive it without any RL-specific reward / multi-turn logic.
    """

    def __init__(
        self, prompt_input: tinker.ModelInput, stop_condition: list[str] | list[int],
    ):
        self._prompt = prompt_input
        self._stop_condition = stop_condition

    async def initial_observation(self) -> tuple[tinker.ModelInput, list[str] | list[int]]:
        return self._prompt, self._stop_condition

    async def step(self, action: Any, *, extra: Any | None = None) -> StepResult:
        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=self._prompt,  # unused after terminal transition
            next_stop_condition=self._stop_condition,
            metrics={},
            logs={},
        )


class _OPDV2EnvGroupBuilder(EnvGroupBuilder):
    """One-env-per-builder wrapper around a single v2 dataset row.

    We always use ``group_size=1`` because SDFT does not need GRPO-style
    grouped trajectories — each row is an independent supervised example.
    """

    def __init__(
        self,
        prompt_input: tinker.ModelInput,
        stop_condition: list[str] | list[int],
    ):
        self._prompt_input = prompt_input
        self._stop_condition = stop_condition

    async def make_envs(self):
        return [_OPDV2PromptOnlyEnv(self._prompt_input, self._stop_condition)]

    def logging_tags(self) -> list[str]:
        return ["opd_v2"]


def _legacy_datum_to_weights_datum(datum: tinker.Datum) -> tinker.Datum:
    """Translate a cookbook ``assemble_training_data`` datum to the local schema.

    Cookbook's ``trajectory_to_data`` emits
    ``loss_fn_inputs = {target_tokens, logprobs, advantages, mask}``.  The
    downstream OPD code (``_build_offline_topk_datums_async`` and friends)
    reads ``weights`` + ``target_tokens``.  This shim copies ``mask`` into
    ``weights`` so the rest of the pipeline stays untouched.  ``logprobs``
    and ``advantages`` are dropped because top-K CE distillation does not
    consume them.
    """
    mask = datum.loss_fn_inputs["mask"]
    target_tokens = datum.loss_fn_inputs["target_tokens"]
    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={
            "weights": mask,
            "target_tokens": target_tokens,
        },
    )


async def _sample_legacy_pipeline_datums_async(
    rows: list[dict[str, Any]],
    renderer: renderers.Renderer,
    sampling_client: tinker.SamplingClient,
    max_tokens: int,
    temperature: float,
    max_length: int | None,
    step: int,
    sample_log_path: Path | None = None,
    sample_log_chars: int = 4000,
    extract_version: str = "v2",
    log_teacher_prompts: bool = False,
) -> tuple[list[tinker.Datum], list[tinker.ModelInput], dict[str, float]]:
    """Legacy-pipeline rollout: cookbook ``do_group_rollout_*`` + ``assemble_training_data``.

    For each row we build a single-env ``_OPDV2EnvGroupBuilder``, dispatch
    ``do_group_rollout_and_filter_constant_reward(..., do_remove_constant_reward_groups=False)``,
    then convert the resulting trajectory groups to datums via
    ``assemble_training_data``.  No parse-filter, no canonicalization, no
    retry — the trained tokens are the raw sampled tokens.  Each row maps
    1-to-1 to a trajectory group, so ``metadata_D[i]["group_idx"]`` directly
    selects the originating row's teacher prompt.

    The ``opd_online/*`` metric keys are kept identical to the current
    pipeline so dashboards do not break.  ``opd_online/filter_reason/*``
    keys are emitted with ``legacy_pipeline_*`` reasons to make it obvious
    in the logs which branch produced the data.
    """
    stop_condition = renderer.get_stop_sequences()

    valid_rows: list[tuple[int, dict[str, Any]]] = []
    skipped_rows: list[tuple[int, dict[str, Any], str]] = []

    rollout_tasks: list[Any] = []
    indexed_rows = list(enumerate(rows))
    for _row_idx, row in indexed_rows:
        prompt_input = row.get("student_prompt_input")
        if prompt_input is None:
            continue
        builder = _OPDV2EnvGroupBuilder(prompt_input, stop_condition)
        rollout_tasks.append(
            do_group_rollout_and_filter_constant_reward(
                sampling_client=sampling_client,
                env_group_builder=builder,
                max_tokens=max_tokens,
                temperature=temperature,
                do_remove_constant_reward_groups=False,
                enable_logging=False,
            )
        )

    rollout_results = await asyncio.gather(*rollout_tasks, return_exceptions=True)

    trajectory_groups: list[Any] = []
    row_for_group: list[dict[str, Any]] = []
    row_idx_for_group: list[int] = []
    error_counts: dict[str, int] = {}

    for (row_idx, row), result in zip(indexed_rows, rollout_results, strict=True):
        if isinstance(result, BaseException):
            err_name = type(result).__name__
            error_counts[err_name] = error_counts.get(err_name, 0) + 1
            skipped_rows.append((row_idx, row, f"legacy_pipeline_exception:{err_name}"))
            continue
        if result is None:
            skipped_rows.append((row_idx, row, "legacy_pipeline_skipped"))
            continue
        if not result.trajectories_G:
            skipped_rows.append((row_idx, row, "legacy_pipeline_empty_group"))
            continue
        trajectory_groups.append(result)
        row_for_group.append(row)
        row_idx_for_group.append(row_idx)

    advantages_P = compute_advantages(trajectory_groups) if trajectory_groups else []
    data_D, metadata_D = (
        assemble_training_data(trajectory_groups, advantages_P)
        if trajectory_groups else ([], [])
    )

    valid_datums: list[tinker.Datum] = []
    valid_teacher_prompts: list[tinker.ModelInput] = []
    too_long_or_empty = 0

    for datum, meta in zip(data_D, metadata_D, strict=True):
        group_idx = int(meta["group_idx"])
        row = row_for_group[group_idx]
        translated = _legacy_datum_to_weights_datum(datum)
        if max_length is not None and translated.model_input.length + 1 > max_length:
            too_long_or_empty += 1
            continue
        valid_datums.append(translated)
        valid_teacher_prompts.append(row["teacher_prompt_input"])
        valid_rows.append((row_idx_for_group[group_idx], row))

    sample_log_f = None
    if sample_log_path is not None:
        sample_log_path.parent.mkdir(parents=True, exist_ok=True)
        sample_log_f = sample_log_path.open("a", encoding="utf-8")
    try:
        if sample_log_f is not None:
            for group, row, row_idx in zip(
                trajectory_groups, row_for_group, row_idx_for_group, strict=True,
            ):
                traj = group.trajectories_G[0]
                tokens = list(traj.transitions[0].ac.tokens) if traj.transitions else []
                rec = _summarize_sample(
                    renderer,
                    tokens,
                    row.get("expected_path"),
                    True,
                    "legacy_pipeline_ok",
                    row_idx,
                    0,
                    step,
                    sample_log_chars,
                    extract_version=extract_version,
                    teacher_prompt_input=(
                        row.get("teacher_prompt_input")
                        if log_teacher_prompts else None
                    ),
                )
                sample_log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            for row_idx, row, reason in skipped_rows:
                rec = _summarize_sample(
                    renderer,
                    [],
                    row.get("expected_path"),
                    False,
                    reason,
                    row_idx,
                    0,
                    step,
                    sample_log_chars,
                    extract_version=extract_version,
                    teacher_prompt_input=(
                        row.get("teacher_prompt_input")
                        if log_teacher_prompts else None
                    ),
                )
                sample_log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            sample_log_f.flush()
    finally:
        if sample_log_f is not None:
            sample_log_f.close()

    n_rows = float(len(rows))
    n_valid = float(len(valid_datums))
    metrics: dict[str, float] = {
        "opd_online/batch_examples": n_rows,
        "opd_online/valid_examples": n_valid,
        "opd_online/filtered_examples": n_rows - n_valid,
        "opd_online/filter_rate": (n_rows - n_valid) / max(n_rows, 1.0),
        # Each row gets exactly one rollout attempt in the legacy pipeline
        # (no retry loop).  Skipped/exception rows still count as "attempted".
        "opd_online/attempts": n_rows,
    }
    metrics["opd_online/filter_reason/legacy_pipeline_ok"] = float(len(valid_datums))
    if too_long_or_empty:
        metrics["opd_online/filter_reason/legacy_pipeline_too_long"] = float(too_long_or_empty)
    skipped_counts: dict[str, int] = {}
    for _row_idx, _row, reason in skipped_rows:
        skipped_counts[reason] = skipped_counts.get(reason, 0) + 1
    for reason, count in skipped_counts.items():
        safe_reason = reason.replace("/", "_").replace(":", "_")
        metrics[f"opd_online/filter_reason/{safe_reason}"] = float(count)
    return valid_datums, valid_teacher_prompts, metrics


# ---------------------------------------------------------------------------
# Agentic on-policy rollout (extract_version="agentic")
# ---------------------------------------------------------------------------
#
# A session is rolled out as a *sequence of user queries*, mirroring the Pi
# agent loop: a planning turn (the model calls ``workflow_plan`` on-policy)
# followed by one "Proceed with: <step>" user turn per planned step.  The step
# queries are derived from the model's OWN plan (parsed from its workflow_plan
# tool call), so the whole episode stays on-policy.  All segments share one
# ephemeral sandbox (state carries across steps), and the whole trajectory
# (every assistant turn, interleaved with real tool results and the injected
# step user turns) becomes a single student datum (ALL_ASSISTANT_MESSAGES
# mask).  The teacher sees the same trajectory tail but with the session's user
# follow-ups spliced into its prefix (privileged signal), and supplies top-K
# soft targets at every assistant token for forward-KL distillation.


def _node_prompt(
    description: str,
    path_context: str,
    output_files: list[str],
    cwd: str,
) -> str:
    """Synthesize one per-step user instruction.

    Mirrors ``buildPromptForNode`` in ``src/electron/libs/runner.ts`` (sans the
    human-edit note, which has no analogue in an autonomous rollout) so the step
    queries match the chat distribution the model was trained under.
    """
    cwd = (cwd or "").strip()
    if cwd:
        cwd_note = (
            "\n\nWorking directory for this session (the agent runs with this as "
            "cwd). For Read, Write, Edit, and any file paths, use each output file "
            "as a path **relative to this directory** (e.g. the basename `report.md` "
            "means read/write under this folder). Do not place outputs outside this "
            "directory unless the task explicitly requires it.\n"
            f"Working directory: {cwd}"
        )
    else:
        cwd_note = (
            "\n\nUse paths relative to the session working directory for all Read, "
            "Write, and Edit calls."
        )
    has_md = any(str(f).lower().endswith(".md") for f in output_files)
    format_note = (
        "\n\nWhen writing output to .md files, use markdown format so the file "
        "preview renders properly."
        if has_md
        else ""
    )
    files_note = (
        "\n\nRelevant output files for this step:\n"
        + "\n".join(f"- {f}" for f in output_files)
        if output_files
        else ""
    )
    refinement_note = (
        "\n\nWhen refining existing outputs, first read the current on-disk "
        "contents, then edit on top of that version. Do not recreate files from "
        "memory."
    )
    return (
        f"Proceed with: {path_context}\n\nTask: {description}"
        f"{cwd_note}{files_note}{format_note}{refinement_note}"
    )


def _plan_step_prompts(tasks: Any, cwd: str, max_steps: int) -> list[str]:
    """Preorder leaf-node "Proceed with" prompts from a ``workflow_plan`` tree.

    Mirrors the app's per-node execution: each executable (leaf) node becomes
    one user instruction, with ``pathContext`` = the ``" > "``-joined chain of
    ancestor descriptions (matching ``getNodePath`` in the UI).  Parent nodes
    (those with children) are groupings only and do not get their own turn.
    """
    prompts: list[str] = []

    def walk(nodes: Any, ancestors: list[str]) -> None:
        if not isinstance(nodes, list):
            return
        for node in nodes:
            if len(prompts) >= max_steps:
                return
            if not isinstance(node, dict):
                continue
            desc = str(node.get("description") or "").strip()
            path = ancestors + ([desc] if desc else [])
            children = node.get("children")
            if isinstance(children, list) and children:
                walk(children, path)
            else:
                output_files = [str(f) for f in (node.get("outputFiles") or [])]
                path_context = " > ".join(path) if path else desc
                prompts.append(_node_prompt(desc, path_context, output_files, cwd))

    walk(tasks if isinstance(tasks, list) else [], [])
    return prompts[:max_steps]


def _extract_workflow_plan_tasks(messages: list[dict[str, Any]]) -> Any | None:
    """Return the ``tasks`` arg of the last ``workflow_plan`` tool call, if any.

    Reads the model's on-policy plan back out of the rendered trajectory so the
    rollout driver can synthesize step queries from it (option B).
    """
    found: Any | None = None
    for m in messages:
        if not (isinstance(m, dict) and m.get("role") == "assistant"):
            continue
        for tc in (m.get("tool_calls") or []):
            fn = getattr(tc, "function", None)
            if fn is not None:
                name = getattr(fn, "name", None)
                args = getattr(fn, "arguments", None)
            elif isinstance(tc, dict):
                fn_d = tc.get("function") or {}
                name = fn_d.get("name")
                args = fn_d.get("arguments")
            else:
                continue
            if name != "workflow_plan":
                continue
            try:
                parsed = json.loads(args) if isinstance(args, str) else args
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(parsed, dict) and "tasks" in parsed:
                found = parsed["tasks"]
    return found


# ---- rollout transcript logging helpers -----------------------------------
# These render a live agentic rollout (every user turn, assistant turn, tool
# call and tool result) into compact JSON entries and a human-readable block so
# the rollouts are actually inspectable in the logs.


def _message_text_for_log(content: Any) -> str:
    """Best-effort plain-text extraction from a renderer ``Message`` content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            is_dict = isinstance(p, dict)
            ptype = p.get("type") if is_dict else getattr(p, "type", None)
            if ptype == "text":
                parts.append(str(p.get("text", "") if is_dict else getattr(p, "text", "")))
            elif ptype == "thinking":
                think = p.get("thinking", "") if is_dict else getattr(p, "thinking", "")
                parts.append(f"<think>{think}</think>")
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _tool_calls_for_log(message: dict[str, Any]) -> list[dict[str, str]]:
    """Extract ``[{name, arguments}]`` from a message's tool calls (if any)."""
    out: list[dict[str, str]] = []
    for tc in (message.get("tool_calls") or []):
        fn = getattr(tc, "function", None)
        if fn is not None:
            out.append(
                {"name": str(getattr(fn, "name", "")), "arguments": str(getattr(fn, "arguments", ""))}
            )
        elif isinstance(tc, dict):
            f = tc.get("function") or {}
            out.append({"name": str(f.get("name", "")), "arguments": str(f.get("arguments", ""))})
    return out


def _agentic_messages_to_log(
    messages: list[dict[str, Any]], *, max_field_chars: int = 2000
) -> list[dict[str, Any]]:
    """Convert a rollout history into compact, JSON-serializable log entries.

    Each entry keeps the role, truncated text content, any tool calls
    (name + truncated arguments), and the tool name for tool-result turns.
    """
    logged: list[dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        entry: dict[str, Any] = {"role": role}
        text = _message_text_for_log(m.get("content"))
        if text:
            entry["content"] = text[:max_field_chars]
        tcs = _tool_calls_for_log(m)
        if tcs:
            entry["tool_calls"] = [
                {"name": tc["name"], "arguments": tc["arguments"][:max_field_chars]}
                for tc in tcs
            ]
        if role == "tool" and m.get("name"):
            entry["name"] = m.get("name")
        logged.append(entry)
    return logged


def _format_agentic_transcript(record: dict[str, Any]) -> str:
    """Render a logged rollout record as a human-readable transcript block."""
    lines: list[str] = [
        f"================ step {record.get('step')} | session "
        f"{record.get('session_uuid')} ================",
        (
            f"valid={record.get('valid')} drop_reason={record.get('drop_reason')} "
            f"turns={record.get('turns')} steps={record.get('steps')} "
            f"tool_calls={record.get('tool_calls')} "
            f"trajectory_tokens={record.get('trajectory_tokens')} "
            f"n_followups={record.get('n_followups')}"
        ),
    ]
    for m in record.get("messages") or []:
        role = str(m.get("role", "?")).upper()
        name = f" ({m['name']})" if m.get("role") == "tool" and m.get("name") else ""
        lines.append(f"\n----- {role}{name} -----")
        content = m.get("content")
        if content:
            lines.append(content)
        for tc in m.get("tool_calls") or []:
            lines.append(f"  >> tool_call: {tc['name']}({tc['arguments']})")
    lines.append("")
    return "\n".join(lines)


def _build_agentic_teacher_messages(
    row: dict[str, Any],
    trajectory_tail: list[dict[str, Any]],
    tools_prefix: list[dict[str, Any]],
    redo_message: str = OPD_REDO_MESSAGE,
) -> list[dict[str, Any]]:
    """Teacher conversation: tools_prefix + [follow-ups + redo] + task + trajectory.

    The privileged user follow-ups are collapsed into a SINGLE user message
    (one follow-up per line, separated by line breaks) with ``redo_message``
    appended, spliced ONCE before the restated initial task (and thus before
    the first assistant token), so they condition every distilled assistant
    distribution.  ``trajectory_tail`` (the student's assistant/tool messages)
    is byte-identical to the student datum's, so teacher top-K aligns 1-to-1
    onto the student mask.
    """
    golden_chat = row.get("golden_chat") or []
    prompt_messages = row["prompt_messages"]
    followup_texts = [
        str(m.get("content", "")) for m in golden_chat if m.get("content")
    ]
    if followup_texts:
        combined = "\n".join(followup_texts) + "\n\n" + redo_message
    else:
        combined = redo_message
    messages: list[dict[str, Any]] = list(tools_prefix)
    messages.append({"role": "user", "content": combined})
    messages.extend(dict(m) for m in prompt_messages)
    messages.extend(trajectory_tail)
    return messages


async def _rollout_one_agentic_episode(
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
    collect_sampling_trace: bool = False,
    grade_reward: bool = False,
    log_field_chars: int = 2000,
) -> tuple[
    tinker.Datum | None, tinker.Datum | None, dict[str, float], dict[str, Any] | None
]:
    """Run one multi-query tool-using rollout and build (student, teacher) datums.

    The episode is a sequence of user-turn *segments*: a planning segment (where
    the model calls ``workflow_plan`` on-policy) followed by one segment per
    planned leaf step (a synthesized "Proceed with: ..." user turn).  Each
    segment runs the inner agent loop -- sample -> parse -> execute tools -- until
    the model emits an assistant turn with no tool calls (the step's final
    answer) or the per-step turn cap is hit.  All segments share one sandbox.
    """
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
            # Overall safety ceiling across all segments; per-step caps below are
            # the usual limiter.
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
        missing_logprobs = False
        sampling_traces: list[dict[str, Any]] = []

        async def run_segment() -> None:
            """Drive the current user-turn segment to a no-tool assistant turn.

            Sets ``parse_failed`` / ``overflow`` on abort.  Mutates the shared
            ``history`` (== ``msg_env.history``) in place.
            """
            nonlocal n_turns, n_tool_calls, parse_failed, overflow, missing_logprobs, history
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
                sequence = result.sequences[0]
                tokens = list(sequence.tokens)
                if collect_sampling_trace:
                    logprobs = _sequence_logprobs(sequence, len(tokens))
                    if logprobs is None:
                        missing_logprobs = True
                        return
                    sampling_traces.append(
                        {
                            "prompt_input": prompt_input,
                            "tokens": tokens,
                            "logprobs": logprobs,
                        }
                    )
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
                        "agentic turn=%d tools=%s content=%r",
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
                    # A tool explicitly halted the episode (not used by our
                    # toolset, but honor it defensively).
                    return

        # Segment 0: planning. The model registers a workflow_plan on-policy.
        await run_segment()

        # Derive the step queries from the model's OWN plan and replay them as
        # subsequent user turns in the same sandbox.
        n_steps = 0
        if not parse_failed and not overflow and not missing_logprobs:
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
                n_steps += 1
                await run_segment()
                if parse_failed or overflow:
                    break

        if overflow:
            metrics_local["agentic/context_overflow"] = 1.0
        if missing_logprobs:
            metrics_local["agentic/missing_logprobs"] = 1.0

        def _episode_log(drop_reason: str | None, *, valid: bool) -> dict[str, Any] | None:
            if not collect_transcript and not collect_sampling_trace:
                return None
            return {
                "messages": _agentic_messages_to_log(
                    history, max_field_chars=log_field_chars
                ) if collect_transcript else [],
                "drop_reason": drop_reason,
                "valid": valid,
                "n_turns": n_turns,
                "n_steps": n_steps,
                "n_tool_calls": n_tool_calls,
                "sampling_traces": sampling_traces if collect_sampling_trace else [],
            }

        trajectory_tail = list(history[len(initial_messages):])
        if not any(
            isinstance(m, dict) and m.get("role") == "assistant" for m in trajectory_tail
        ):
            # No assistant turn produced (e.g. immediate parse failure): no
            # supervision signal, skip the episode.
            metrics_local["agentic/empty_trajectory"] = 1.0
            drop_reason = (
                "parse_failed" if parse_failed
                else "context_overflow" if overflow
                else "missing_logprobs" if missing_logprobs
                else "empty_trajectory"
            )
            return None, None, metrics_local, _episode_log(drop_reason, valid=False)
        if missing_logprobs:
            return None, None, metrics_local, _episode_log("missing_logprobs", valid=False)

        if grade_reward:
            # Isolate grading: a judge/API/parse failure must not abort the whole
            # training round. On failure leave ``reinforce/reward`` unset so the
            # episode is simply ineligible for GRPO (OPD is unaffected).
            try:
                metrics_local["reinforce/reward"] = await grade_sandbox_rubrics(
                    sandbox, rubrics,
                )
            except Exception as grade_exc:  # noqa: BLE001 - degrade, don't crash
                logger.warning(
                    "grading failed (session=%s): %s",
                    (row.get("meta") or {}).get("session_uuid"),
                    grade_exc,
                )

        # The env-produced messages already carry renderer-native ToolCall
        # objects (from renderer.parse_response), so they must NOT be re-hydrated.
        student_convo = list(history)
        teacher_convo = _build_agentic_teacher_messages(
            row, trajectory_tail, tools_prefix,
        )
        student_datum = conversation_to_datum(
            student_convo, renderer, max_length,
            train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        )
        teacher_datum = conversation_to_datum(
            teacher_convo, renderer, max_length,
            train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        )
        metrics_local.update({
            "agentic/turns": float(n_turns),
            "agentic/tool_calls": float(n_tool_calls),
            "agentic/steps": float(n_steps),
            "agentic/parse_failed": 1.0 if parse_failed else 0.0,
            "agentic/trajectory_tokens": float(student_datum.model_input.length),
        })
        return student_datum, teacher_datum, metrics_local, _episode_log(None, valid=True)
    finally:
        sandbox.cleanup()


async def _sample_agentic_opd_datums_async(
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
) -> tuple[list[tinker.Datum], list[tinker.Datum], dict[str, float]]:
    """Collect agentic student trajectories and follow-up-augmented teacher datums.

    Returns ``(student_datums, teacher_datums, metrics)``.  Each session yields
    at most one (student, teacher) datum pair; sessions that produce no
    assistant turn are dropped.  Student and teacher datums are 1-to-1.
    """
    results = await asyncio.gather(*[
        _rollout_one_agentic_episode(
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

    student_datums: list[tinker.Datum] = []
    teacher_datums: list[tinker.Datum] = []
    n_valid = 0
    agg_turns = 0.0
    agg_tool_calls = 0.0
    agg_steps = 0.0
    agg_parse_failed = 0.0
    agg_overflow = 0.0
    agg_empty = 0.0
    agg_no_plan = 0.0

    sample_log_f = None
    transcript_f = None
    if sample_log_path is not None:
        sample_log_path.parent.mkdir(parents=True, exist_ok=True)
        sample_log_f = sample_log_path.open("a", encoding="utf-8")
        # Sibling human-readable transcript (the actual rollouts, turn by turn).
        transcript_f = sample_log_path.with_name(
            "agentic_rollout_transcripts.txt"
        ).open("a", encoding="utf-8")
    try:
        for (row, (sd, td, m, episode_log)) in zip(rows, results, strict=True):
            agg_parse_failed += m.get("agentic/parse_failed", 0.0)
            agg_overflow += m.get("agentic/context_overflow", 0.0)
            agg_empty += m.get("agentic/empty_trajectory", 0.0)
            agg_no_plan += m.get("agentic/no_plan", 0.0)
            valid = sd is not None and td is not None
            meta = row.get("meta") or {}
            drop_reason = (episode_log or {}).get("drop_reason")
            if not valid and not drop_reason:
                drop_reason = "dropped"

            # Per-session console summary (always on, even without file logging),
            # so the rollouts are visible at a glance during training.
            logger.info(
                "agentic rollout step=%d session=%s valid=%s turns=%d steps=%d "
                "tool_calls=%d tokens=%s drop=%s",
                step,
                meta.get("session_uuid"),
                valid,
                int(m.get("agentic/turns", 0.0)),
                int(m.get("agentic/steps", 0.0)),
                int(m.get("agentic/tool_calls", 0.0)),
                m.get("agentic/trajectory_tokens"),
                drop_reason or "-",
            )

            if sample_log_f is not None:
                rec = {
                    "step": step,
                    "session_uuid": meta.get("session_uuid"),
                    "valid": valid,
                    "drop_reason": drop_reason,
                    "turns": m.get("agentic/turns"),
                    "steps": m.get("agentic/steps"),
                    "tool_calls": m.get("agentic/tool_calls"),
                    "trajectory_tokens": m.get("agentic/trajectory_tokens"),
                    "n_followups": meta.get("n_followups"),
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
            agg_turns += m.get("agentic/turns", 0.0)
            agg_tool_calls += m.get("agentic/tool_calls", 0.0)
            agg_steps += m.get("agentic/steps", 0.0)
            student_datums.append(sd)
            teacher_datums.append(td)
    finally:
        if sample_log_f is not None:
            sample_log_f.close()
        if transcript_f is not None:
            transcript_f.close()

    batch = len(rows)
    metrics: dict[str, float] = {
        "opd_online/batch_examples": float(batch),
        "opd_online/valid_examples": float(n_valid),
        "opd_online/filter_rate": (1.0 - n_valid / batch) if batch else 0.0,
        "agentic/parse_failed_total": agg_parse_failed,
        "agentic/context_overflow_total": agg_overflow,
        "agentic/empty_trajectory_total": agg_empty,
        "agentic/no_plan_total": agg_no_plan,
    }
    if n_valid:
        metrics["agentic/mean_turns"] = agg_turns / n_valid
        metrics["agentic/mean_tool_calls"] = agg_tool_calls / n_valid
        metrics["agentic/mean_steps"] = agg_steps / n_valid
    return student_datums, teacher_datums, metrics


def _scale_topk_weights(
    datums: list[tinker.Datum], scale: float,
) -> list[tinker.Datum]:
    """Return copies of top-K datums with their CE ``weights`` multiplied by ``scale``.

    Scaling the per-token teacher-probability weights scales the cross_entropy
    loss (and therefore its gradient) linearly, which is how the OPD half is
    weighted relative to the GRPO half in the combined trainer.
    """
    if scale == 1.0:
        return datums
    scaled: list[tinker.Datum] = []
    for d in datums:
        w = d.loss_fn_inputs["weights"]
        wt = torch.as_tensor(w.data, dtype=torch.float32) * float(scale)
        new_inputs = dict(d.loss_fn_inputs)
        new_inputs["weights"] = tinker.TensorData.from_torch(wt)
        scaled.append(
            tinker.Datum(model_input=d.model_input, loss_fn_inputs=new_inputs)
        )
    return scaled


async def _sample_combined_grpo_opd_datums_async(
    rows: list[dict[str, Any]],
    renderer: renderers.Renderer,
    sampling_client: tinker.SamplingClient,
    teacher_client: tinker.SamplingClient,
    *,
    group_size: int,
    topk: int,
    max_context_length: int,
    vocab_size: int | None,
    teacher_temperature: float,
    lambda_grpo: float,
    lambda_opd: float,
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
) -> tuple[list[tinker.Datum], list[tinker.Datum], dict[str, float]]:
    """One shared agentic group-rollout feeding both GRPO and OPD.

    Samples ``group_size`` on-policy agentic episodes per session. Each valid
    episode yields (a) per-turn sampling traces + a graded reward for GRPO, and
    (b) a (student, teacher) datum pair (teacher prompt carries the session
    follow-ups) for OPD top-K distillation.

    Returns ``(grpo_is_datums, opd_topk_datums, metrics)``.  GRPO advantages are
    pre-scaled by ``lambda_grpo`` and OPD top-K weights by ``lambda_opd`` so the
    two stock Tinker loss functions (``importance_sampling`` / ``cross_entropy``)
    need no modification.
    """
    # Lazy import avoids a module-load circular import (run_reinforce imports
    # _format_agentic_transcript from this module).
    try:
        from weight.train.run_reinforce import (  # type: ignore[import-not-found]
            build_grpo_importance_sampling_datum,
            compute_grpo_group_advantages,
        )
    except ModuleNotFoundError:  # pragma: no cover - depends on invocation cwd
        from .run_reinforce import (
            build_grpo_importance_sampling_datum,
            compute_grpo_group_advantages,
        )

    group_size = max(1, int(group_size))
    indexed: list[tuple[int, int, dict[str, Any]]] = [
        (row_idx, sample_idx, row)
        for row_idx, row in enumerate(rows)
        for sample_idx in range(group_size)
    ]
    # ``return_exceptions=True`` so one rollout blowing up (e.g. an unhandled
    # sandbox/tool error) degrades to dropping that single episode instead of
    # cancelling every other in-flight rollout and aborting the round.
    results = await asyncio.gather(*[
        _rollout_one_agentic_episode(
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
            collect_sampling_trace=True,
            grade_reward=True,
            log_field_chars=sample_log_chars,
        )
        for _, _, row in indexed
    ], return_exceptions=True)

    # Bucket per-session episodes. Each item carries everything both halves need.
    groups: list[list[dict[str, Any]]] = [[] for _ in rows]
    sample_records: list[dict[str, Any]] = []
    n_rollout_errors = 0
    for (row_idx, sample_idx, row), result in zip(indexed, results, strict=True):
        if isinstance(result, BaseException):
            n_rollout_errors += 1
            logger.warning(
                "combined rollout step=%d session=%s sample=%d/%d failed: %s",
                step,
                (row.get("meta") or {}).get("session_uuid"),
                sample_idx + 1,
                group_size,
                result,
            )
            continue
        sd, td, m, episode_log = result
        reward = m.get("reinforce/reward")
        traces = (episode_log or {}).get("sampling_traces") or []
        meta = row.get("meta") or {}
        drop_reason = (episode_log or {}).get("drop_reason")
        valid = sd is not None and td is not None
        if not valid and not drop_reason:
            drop_reason = "dropped"

        logger.info(
            "combined rollout step=%d session=%s sample=%d/%d valid=%s reward=%s drop=%s",
            step,
            meta.get("session_uuid"),
            sample_idx + 1,
            group_size,
            valid,
            reward,
            drop_reason or "-",
        )

        rec = {
            "step": step,
            "session_uuid": meta.get("session_uuid"),
            "sample_idx": sample_idx,
            "valid": valid,
            "reward": reward,
            "drop_reason": drop_reason,
            "turns": m.get("agentic/turns"),
            "steps": m.get("agentic/steps"),
            "tool_calls": m.get("agentic/tool_calls"),
            "messages": (episode_log or {}).get("messages") or [],
        }
        sample_records.append(rec)
        if valid:
            groups[row_idx].append(
                {
                    "reward": float(reward) if reward is not None else None,
                    "traces": traces,
                    "student_datum": sd,
                    "teacher_datum": td,
                }
            )

    # --- OPD half: collect every valid (student, teacher) pair -------------
    opd_student_datums: list[tinker.Datum] = []
    opd_teacher_datums: list[tinker.Datum] = []
    for group in groups:
        for item in group:
            opd_student_datums.append(item["student_datum"])
            opd_teacher_datums.append(item["teacher_datum"])

    # --- GRPO half: group-centered advantages -> IS datums ----------------
    grpo_datums: list[tinker.Datum] = []
    valid_episodes = 0
    trained_episodes = 0
    skipped_small = 0
    skipped_constant = 0
    reward_sum = 0.0
    abs_adv_sum = 0.0
    adv_count = 0
    for group in groups:
        grpo_eligible = [it for it in group if it["reward"] is not None and it["traces"]]
        valid_episodes += len(grpo_eligible)
        reward_sum += sum(float(it["reward"]) for it in grpo_eligible)
        if len(grpo_eligible) < 2:
            skipped_small += 1
            continue
        rewards = [float(it["reward"]) for it in grpo_eligible]
        advantages = compute_grpo_group_advantages(rewards)
        if advantages is None:
            skipped_constant += 1
            continue
        for item, advantage in zip(grpo_eligible, advantages, strict=True):
            episode_datums_before = len(grpo_datums)
            for trace_rec in item["traces"]:
                datum = build_grpo_importance_sampling_datum(
                    trace_rec["prompt_input"],
                    [int(t) for t in trace_rec["tokens"]],
                    [float(lp) for lp in trace_rec["logprobs"]],
                    advantage * lambda_grpo,
                    max_length,
                )
                if datum is not None:
                    grpo_datums.append(datum)
            if len(grpo_datums) > episode_datums_before:
                trained_episodes += 1
                abs_adv_sum += abs(float(advantage))
                adv_count += 1

    # --- OPD top-K teacher distillation -----------------------------------
    opd_topk_datums: list[tinker.Datum] = []
    topk_metrics: dict[str, float] = {}
    if opd_student_datums:
        opd_topk_datums, topk_metrics = await _build_agentic_topk_datums_async(
            opd_student_datums,
            opd_teacher_datums,
            teacher_client,
            topk=topk,
            max_context_length=max_context_length,
            vocab_size=vocab_size,
            teacher_temperature=teacher_temperature,
        )
        opd_topk_datums = _scale_topk_weights(opd_topk_datums, lambda_opd)

    # --- Optional sample logging ------------------------------------------
    if sample_log_path is not None:
        sample_log_path.parent.mkdir(parents=True, exist_ok=True)
        with sample_log_path.open("a", encoding="utf-8") as sample_log_f:
            transcript_f = sample_log_path.with_name(
                "combined_rollout_transcripts.txt"
            ).open("a", encoding="utf-8")
            try:
                for rec in sample_records:
                    sample_log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    transcript_f.write(_format_agentic_transcript(rec) + "\n")
                sample_log_f.flush()
                transcript_f.flush()
            finally:
                transcript_f.close()

    total_episodes = len(rows) * group_size
    metrics: dict[str, float] = {
        "combined/batch_prompts": float(len(rows)),
        "combined/group_size": float(group_size),
        "combined/batch_episodes": float(total_episodes),
        "combined/rollout_errors": float(n_rollout_errors),
        "grpo_online/valid_episodes": float(valid_episodes),
        "grpo_online/trained_episodes": float(trained_episodes),
        "grpo_online/skipped_small_groups": float(skipped_small),
        "grpo_online/skipped_constant_groups": float(skipped_constant),
        "grpo_online/train_datums": float(len(grpo_datums)),
        "grpo_online/filter_rate": (
            1.0 - valid_episodes / total_episodes if total_episodes else 0.0
        ),
        "opd_online/batch_examples": float(total_episodes),
        "opd_online/valid_examples": float(len(opd_student_datums)),
        "opd_online/train_datums": float(len(opd_topk_datums)),
        "opd_online/filter_rate": (
            1.0 - len(opd_student_datums) / total_episodes if total_episodes else 0.0
        ),
        **topk_metrics,
    }
    if valid_episodes:
        metrics["grpo_online/mean_reward"] = reward_sum / valid_episodes
    if adv_count:
        metrics["grpo_online/mean_abs_advantage"] = abs_adv_sum / adv_count
    return grpo_datums, opd_topk_datums, metrics


async def _build_agentic_topk_datums_async(
    student_datums: list[tinker.Datum],
    teacher_datums: list[tinker.Datum],
    teacher_client: tinker.SamplingClient,
    topk: int = 20,
    max_context_length: int = 32768,
    vocab_size: int | None = None,
    skip_first_n_tokens: int = 0,
    teacher_temperature: float = 1.0,
) -> tuple[list[tinker.Datum], dict[str, float]]:
    """Top-K teacher soft targets for multi-turn (non-contiguous) student masks.

    Generalises :func:`_build_offline_topk_datums_async` to the agentic case:
    the teacher-forced sequence is the *entire* teacher conversation (built by
    ``conversation_to_datum`` with the privileged follow-up prefix), and teacher
    top-K is read at the teacher datum's assistant-mask positions, then mapped
    onto the student datum's assistant-mask positions (1-to-1, in order).
    """
    # Drop (student, teacher) pairs whose teacher-forced sequence -- or student
    # datum -- would exceed the model context window. The teacher top-K prefill
    # below sends the *entire* teacher-forced sequence with ``max_tokens=1``; an
    # oversized sequence 400s server-side and would otherwise kill the whole
    # training round. The teacher sequence is the longer of the two (it carries
    # the privileged follow-up/redo prefix on top of the student trajectory),
    # but we guard the student datum too since it must still fit
    # ``forward_backward`` during training.
    kept_student_datums: list[tinker.Datum] = []
    teacher_forced_seqs: list[tinker.ModelInput] = []
    teacher_mask_indices_list: list[list[int]] = []
    n_dropped_oversized = 0
    for idx, (sd, td) in enumerate(zip(student_datums, teacher_datums)):
        targets = td.loss_fn_inputs["target_tokens"].data
        td_weights = td.loss_fn_inputs["weights"].data
        td_mask = [j for j, w in enumerate(td_weights) if w > 0]
        # Full teacher-forced sequence = model_input (== full[:-1]) + last target.
        teacher_forced = td.model_input
        if len(targets) > 0:
            teacher_forced = teacher_forced.append_int(int(targets[-1]))
        # +1 mirrors the max_tokens=1 prefill request the server validates.
        teacher_prefill_len = teacher_forced.length + 1
        student_len = sd.model_input.length
        if teacher_prefill_len > max_context_length or student_len > max_context_length:
            n_dropped_oversized += 1
            logger.warning(
                "opd agentic topk: dropping oversized pair idx=%d "
                "(teacher_prefill=%d, student=%d, max_context_length=%d)",
                idx,
                teacher_prefill_len,
                student_len,
                max_context_length,
            )
            continue
        kept_student_datums.append(sd)
        teacher_forced_seqs.append(teacher_forced)
        teacher_mask_indices_list.append(td_mask)

    if n_dropped_oversized:
        logger.warning(
            "opd agentic topk: dropped %d/%d pair(s) exceeding context window "
            "(max_context_length=%d); proceeding with %d",
            n_dropped_oversized,
            len(student_datums),
            max_context_length,
            len(kept_student_datums),
        )
    if not kept_student_datums:
        logger.error(
            "opd agentic topk: all %d pair(s) dropped as oversized "
            "(max_context_length=%d); returning empty batch",
            len(student_datums),
            max_context_length,
        )
        return [], {
            "opd/topk_num_datums": 0.0,
            "opd/topk_k": float(topk),
            "opd/topk_dropped_oversized": float(n_dropped_oversized),
        }

    topk_responses = await asyncio.gather(*[
        teacher_client.sample_async(
            prompt=tf_seq,
            num_samples=1,
            sampling_params=tinker.SamplingParams(max_tokens=1),
            include_prompt_logprobs=True,
            topk_prompt_logprobs=topk,
        )
        for tf_seq in teacher_forced_seqs
    ])

    total_completion_tokens = 0.0
    total_teacher_entropy = 0.0
    new_datums: list[tinker.Datum] = []

    for i, sd in enumerate(kept_student_datums):
        weights_1d = sd.loss_fn_inputs["weights"].data
        student_mask = [j for j, w in enumerate(weights_1d) if w > 0]
        td_mask = teacher_mask_indices_list[i]
        N = sd.model_input.length

        target_tokens_NK = torch.zeros(N, topk, dtype=torch.long)
        weights_NK = torch.zeros(N, topk, dtype=torch.float32)

        topk_all = topk_responses[i].topk_prompt_logprobs
        n_tokens = min(len(student_mask), len(td_mask))
        used = 0
        for t in range(n_tokens):
            if t < skip_first_n_tokens:
                continue
            # topk_prompt_logprobs[p] predicts the token AT sequence position p;
            # the t-th assistant token sits at teacher full-seq position
            # td_mask[t] + 1 (target[j] == full[j + 1]).
            teacher_pos = td_mask[t] + 1
            student_pos = student_mask[t]

            if topk_all is None or teacher_pos >= len(topk_all):
                continue
            topk_entries = topk_all[teacher_pos]
            if not topk_entries:
                continue

            filtered = [
                (tok_id, lp)
                for tok_id, lp in topk_entries[:topk]
                if vocab_size is None or tok_id < vocab_size
            ]
            if not filtered:
                continue

            k_actual = len(filtered)
            token_ids = torch.tensor([tid for tid, _ in filtered], dtype=torch.long)
            logprobs = torch.tensor([lp for _, lp in filtered], dtype=torch.float32)
            if teacher_temperature != 1.0:
                logprobs = logprobs / teacher_temperature
            logprobs -= torch.logsumexp(logprobs, dim=0)
            probs = logprobs.exp()

            target_tokens_NK[student_pos, :k_actual] = token_ids
            weights_NK[student_pos, :k_actual] = probs
            total_teacher_entropy += -(probs * logprobs).sum().item()
            used += 1

        total_completion_tokens += used
        new_datums.append(tinker.Datum(
            model_input=sd.model_input,
            loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_torch(target_tokens_NK),
                "weights": tinker.TensorData.from_torch(weights_NK),
            },
        ))

    metrics: dict[str, float] = {
        "opd/topk_num_datums": float(len(kept_student_datums)),
        "opd/topk_k": float(topk),
        "opd/topk_dropped_oversized": float(n_dropped_oversized),
    }
    if total_completion_tokens > 0:
        metrics["opd/total_completion_tokens"] = total_completion_tokens
        metrics["opd/mean_teacher_entropy"] = total_teacher_entropy / total_completion_tokens
    return new_datums, metrics


async def _sample_online_artifact_datums_async(
    rows: list[dict[str, Any]],
    renderer: renderers.Renderer,
    sampling_client: tinker.SamplingClient,
    max_tokens: int,
    temperature: float,
    attempts: int,
    max_length: int | None,
    step: int,
    sample_log_path: Path | None = None,
    sample_log_chars: int = 4000,
    extract_version: str = "v1",
    log_teacher_prompts: bool = False,
    rollout_pipeline: str = "current",
) -> tuple[list[tinker.Datum], list[tinker.ModelInput], dict[str, float]]:
    """Sample student completions and keep only valid responses.

    Filter semantics depend on ``extract_version``:

    * ``"v1"`` (default; legacy artifact-only): the sampled response must
      parse as exactly one ``write(path, non-empty content)`` tool call
      (or a narrow ``bash`` heredoc that we salvage into one). The trained
      message is a canonicalised single-write call.
    * ``"v2"`` (per-assistant-message): the sampled response only needs to
      parse into a well-formed assistant message with non-empty
      content or any tool calls. The trained message is the parsed
      assistant turn, verbatim.

    ``rollout_pipeline`` selects the dispatch path:

    * ``"current"`` (default): in-house ``sample_async`` loop with
      parse-filter + canonicalization + optional retry per row.
    * ``"legacy"``: cookbook ``do_group_rollout_and_filter_constant_reward``
      + ``assemble_training_data`` path used by the legacy ``tinker_opd``
      recipe.  Trains on raw sampled tokens (no parse-filter, no
      canonicalization, no retry).  ``attempts`` is ignored in this mode
      because the legacy pipeline does not retry per row.
    """
    if rollout_pipeline == "legacy":
        return await _sample_legacy_pipeline_datums_async(
            rows=rows,
            renderer=renderer,
            sampling_client=sampling_client,
            max_tokens=max_tokens,
            temperature=temperature,
            max_length=max_length,
            step=step,
            sample_log_path=sample_log_path,
            sample_log_chars=sample_log_chars,
            extract_version=extract_version,
            log_teacher_prompts=log_teacher_prompts,
        )

    valid_datums: list[tinker.Datum] = []
    valid_teacher_prompts: list[tinker.ModelInput] = []
    reason_counts: dict[str, int] = {}
    total_attempts = 0

    sample_log_f = None
    if sample_log_path is not None:
        sample_log_path.parent.mkdir(parents=True, exist_ok=True)
        sample_log_f = sample_log_path.open("a", encoding="utf-8")

    try:
        pending = list(enumerate(rows))
        for attempt_idx in range(max(1, attempts)):
            if not pending:
                break
            total_attempts += len(pending)
            results = await asyncio.gather(*[
                sampling_client.sample_async(
                    prompt=row["student_prompt_input"],
                    num_samples=1,
                    sampling_params=tinker.SamplingParams(
                        stop=renderer.get_stop_sequences(),
                        max_tokens=max_tokens,
                        temperature=temperature,
                    ),
                )
                for _row_idx, row in pending
            ])

            next_pending: list[tuple[int, dict[str, Any]]] = []
            for (row_idx, row), result in zip(pending, results, strict=True):
                expected_path = row["expected_path"]
                tokens = list(result.sequences[0].tokens)
                if extract_version == "v2":
                    ok, reason, canonical_message = _parse_v2_message(
                        renderer, tokens,
                    )
                else:
                    ok, reason, canonical_message = _parse_valid_artifact_write_message(
                        renderer, tokens, expected_path,
                    )
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                if sample_log_f is not None:
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
                        extract_version=extract_version,
                        teacher_prompt_input=(
                            row.get("teacher_prompt_input")
                            if log_teacher_prompts else None
                        ),
                    )
                    sample_log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    sample_log_f.flush()
                if not ok:
                    next_pending.append((row_idx, row))
                    continue
                assert canonical_message is not None
                datum = _datum_from_prompt_and_assistant_message(
                    renderer,
                    row["student_prompt_messages"],
                    canonical_message,
                    max_length,
                )
                if datum is None:
                    reason_counts["too_long_or_empty"] = reason_counts.get("too_long_or_empty", 0) + 1
                    next_pending.append((row_idx, row))
                    continue
                valid_datums.append(datum)
                valid_teacher_prompts.append(row["teacher_prompt_input"])
            pending = next_pending
        if pending:
            reason_counts["example_filtered"] = reason_counts.get("example_filtered", 0) + len(pending)
    finally:
        if sample_log_f is not None:
            sample_log_f.close()

    n_rows = float(len(rows))
    n_valid = float(len(valid_datums))
    metrics: dict[str, float] = {
        "opd_online/batch_examples": n_rows,
        "opd_online/valid_examples": n_valid,
        "opd_online/filtered_examples": n_rows - n_valid,
        "opd_online/filter_rate": (n_rows - n_valid) / max(n_rows, 1.0),
        "opd_online/attempts": float(total_attempts),
    }
    for reason, count in reason_counts.items():
        safe_reason = reason.replace("/", "_").replace(":", "_")
        metrics[f"opd_online/filter_reason/{safe_reason}"] = float(count)
    return valid_datums, valid_teacher_prompts, metrics


def sample_online_artifact_datums(
    rows: list[dict[str, Any]],
    renderer: renderers.Renderer,
    sampling_client: tinker.SamplingClient,
    config: Config,
    max_length: int | None,
    step: int,
) -> tuple[list[tinker.Datum], list[tinker.ModelInput], dict[str, float]]:
    return asyncio.run(_sample_online_artifact_datums_async(
        rows,
        renderer,
        sampling_client,
        max_tokens=config.rollout_max_tokens,
        temperature=config.rollout_temperature,
        attempts=config.rollout_attempts,
        max_length=max_length,
        step=step,
        sample_log_path=(
            Path(config.log_path) / "online_rollout_samples.jsonl"
            if config.log_rollout_samples else None
        ),
        sample_log_chars=config.rollout_sample_log_chars,
        extract_version=config.extract_version,
        log_teacher_prompts=config.log_teacher_prompts,
        rollout_pipeline=config.rollout_pipeline,
    ))


def do_update(
    step: int,
    total_steps: int,
    config: Config,
    training_client: tinker.TrainingClient,
    student_datums: list[tinker.Datum],
    teacher_completion_lps: list[torch.Tensor],
    ml_logger: ml_log.Logger,
    log_path: str,
    tokenizer: Tokenizer,
) -> dict[str, Any]:
    """Single offline OPD training step."""
    metrics: dict[str, Any] = {}

    if config.save_every > 0 and step % config.save_every == 0 and step > 0:
        checkpoint_utils.save_checkpoint(
            training_client=training_client,
            name=f"{step:06d}",
            log_path=log_path,
            kind="both",
            loop_state={"batch": step},
            ttl_seconds=config.ttl_seconds,
        )

    learning_rate = config.learning_rate * compute_schedule_lr_multiplier(
        lr_schedule=config.lr_schedule, step=step, total_steps=total_steps,
    )
    adam_params = tinker.AdamParams(
        learning_rate=learning_rate,
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        eps=config.adam_eps,
    )

    captured_teacher_lps = teacher_completion_lps

    def offline_opd_loss(
        data: list[tinker.Datum],
        logprobs_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total_loss = torch.tensor(0.0)
        total_adv = 0.0
        total_tokens = 0
        global_nll_sum = 0.0

        for i in range(len(data)):
            weights = torch.tensor(data[i].loss_fn_inputs["weights"].data)
            student_lps = logprobs_list[i]
            teacher_lps = captured_teacher_lps[i]

            mask_indices = torch.where(weights > 0)[0]
            n = min(len(mask_indices), len(teacher_lps))
            if n == 0:
                continue

            student_at_comp = torch.stack([student_lps[idx] for idx in mask_indices[:n]])
            teacher_at_comp = teacher_lps[:n]

            advantage = (teacher_at_comp - student_at_comp.detach())
            loss_per_token = -advantage * student_at_comp
            total_loss = total_loss + loss_per_token.sum() / max(n, 1)
            global_nll_sum += float(loss_per_token.sum().item())

            total_adv += advantage.sum().item()
            total_tokens += n

        batch_loss = total_loss / max(len(data), 1)
        per_token_ce = global_nll_sum / max(total_tokens, 1)

        loss_metrics = {
            "opd_loss": per_token_ce,
            "opd/per_token_ce": per_token_ce,
            "opd/batch_completion_tokens": float(total_tokens),
            "mean_advantage": total_adv / max(total_tokens, 1),
            "num_completion_tokens": total_tokens,
        }
        return batch_loss, loss_metrics

    backward_result = training_client.forward_backward_custom(
        student_datums, offline_opd_loss,
    ).result()
    training_client.optim_step(adam_params).result()

    metrics.update(
        num_examples=len(student_datums),
        num_tokens=sum(d.model_input.length for d in student_datums),
        learning_rate=learning_rate,
        progress=step / total_steps,
        **backward_result.metrics,
    )
    ml_logger.log_metrics(metrics=metrics, step=step)
    return metrics


def main(config: Config, dataset: OfflineOPDDataset) -> None:
    """Run the complete offline OPD training loop."""
    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    start_batch = resume_info.batch if resume_info else 0

    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
        do_configure_logging_module=True,
    )

    user_metadata: dict[str, str] = {}
    if wandb_link := ml_logger.get_logger_url():
        user_metadata["wandb_link"] = wandb_link
    checkpoint_utils.add_renderer_name_to_user_metadata(user_metadata, config.renderer_name)
    model_info.warn_if_renderer_not_recommended(config.model_name, config.renderer_name)

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
            base_model=config.model_name, rank=config.lora_rank,
            user_metadata=user_metadata,
        )

    tokenizer = get_tokenizer(config.model_name)

    n_batches = len(dataset)
    total_steps = n_batches * config.num_epochs
    if config.max_steps is not None:
        total_steps = min(total_steps, config.max_steps)

    logger.info(
        f"Offline OPD: {n_batches} batches x {config.num_epochs} epochs "
        f"= {n_batches * config.num_epochs} steps"
    )

    # ------------------------------------------------------------------ #
    # Teacher logprob provider (set up BEFORE any optim_step)             #
    # ------------------------------------------------------------------ #
    use_topk = (not config.use_skyrl) and config.topk > 0
    topk_datums_all: list[tinker.Datum] = []

    if use_topk:
        # Tinker top-K path: pre-compute (N, K) targets ONCE for all examples
        # (teacher and completions are fixed in offline mode → identical across
        # epochs; saves K API calls per training step vs. on-policy SDFT).
        teacher_client = service_client.create_sampling_client(
            base_model=config.model_name,
        )
        logger.info(
            "Top-K mode (K=%d): created frozen teacher SamplingClient for %s. "
            "Pre-computing targets for all %d examples...",
            config.topk, config.model_name, n_batches * dataset._batch_size,
        )
        topk_datums_all, topk_pre_metrics = precompute_all_topk_datums(
            dataset, teacher_client, topk=config.topk,
            teacher_temperature=config.teacher_temperature,
        )
        logger.info(
            "Top-K pre-computation complete (%d datums, τ=%.2f)",
            len(topk_datums_all), config.teacher_temperature,
        )
        get_teacher_lps = None  # not used in top-K path

    elif not config.use_skyrl:
        # Tinker IS path: frozen teacher SamplingClient, per-batch scalar logprobs.
        teacher_client = service_client.create_sampling_client(
            base_model=config.model_name,
        )
        logger.info(
            "Tinker IS mode (topk=0): created frozen teacher SamplingClient for %s",
            config.model_name,
        )

        def get_teacher_lps(
            student_datums: list[tinker.Datum],
            teacher_datums: list[tinker.Datum],
        ) -> list[torch.Tensor]:
            return _live_teacher_logprobs(teacher_client, student_datums, teacher_datums)
    else:
        # SkyRL IS path: pre-compute teacher logprobs via training_client.forward().
        # NOTE: SkyRL does not support topk_prompt_logprobs, so top-K KD is
        # unavailable here. If you need top-K distillation, switch to Tinker
        # (drop --use-skyrl) and set --topk 20.
        teacher_cache = precompute_teacher_logprob_cache(
            training_client, dataset, config.num_epochs,
        )

        def get_teacher_lps(
            student_datums: list[tinker.Datum],
            teacher_datums: list[tinker.Datum],
        ) -> list[torch.Tensor]:
            return _lookup_teacher_logprobs(student_datums, teacher_datums, teacher_cache)

    for epoch_idx in range(config.num_epochs):
        dataset.set_epoch(seed=epoch_idx)
        logger.info(f"Starting epoch {epoch_idx}")

        for batch_idx in range(start_batch if epoch_idx == 0 else 0, n_batches):
            step = epoch_idx * n_batches + batch_idx
            if config.max_steps is not None and step >= config.max_steps:
                break

            student_datums, teacher_datums = dataset.get_batch(batch_idx)

            if step == 0:
                for i in range(min(2, len(student_datums))):
                    int_tokens = list(student_datums[i].model_input.to_ints())
                    weights = student_datums[i].loss_fn_inputs["weights"].data
                    logger.info(f"\nExample {i}:")
                    logger.info(format_colorized(
                        int_tokens, cast(list[float], weights), tokenizer,
                    ))

            if use_topk:
                # Retrieve the pre-computed top-K datums for this batch.
                # topk_datums_all is in original dataset order; batch_idx
                # corresponds to the batch window after set_epoch shuffle.
                start = batch_idx * dataset._batch_size
                end = min(start + dataset._batch_size, len(topk_datums_all))
                batch_topk = [topk_datums_all[dataset._indices[i]]
                              for i in range(start, end)]
                do_update_topk(
                    step=step,
                    total_steps=total_steps,
                    config=config,
                    training_client=training_client,
                    topk_datums=batch_topk,
                    ml_logger=ml_logger,
                    log_path=config.log_path,
                    static_teacher_metrics=(
                        topk_pre_metrics
                        if (epoch_idx == 0 and batch_idx == start_batch)
                        else None
                    ),
                )
            else:
                teacher_lps = get_teacher_lps(student_datums, teacher_datums)
                do_update(
                    step=step,
                    total_steps=total_steps,
                    config=config,
                    training_client=training_client,
                    student_datums=student_datums,
                    teacher_completion_lps=teacher_lps,
                    ml_logger=ml_logger,
                    log_path=config.log_path,
                    tokenizer=tokenizer,
                )

    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": n_batches},
        ttl_seconds=None,
    )
    ml_logger.close()
    logger.info("Offline OPD training completed successfully")


def main_online(
    config: Config,
    dataset: OnlineOPDRolloutDataset,
    renderer: renderers.Renderer,
    tokenizer: Tokenizer,
) -> None:
    """Run online artifact-only OPD with sampled-and-filtered write completions."""
    if config.use_skyrl:
        raise ValueError("--online-rollout currently requires Tinker sampling; do not use --use-skyrl")
    if config.topk <= 0:
        raise ValueError("--online-rollout currently supports top-K mode only; set --topk > 0")

    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    start_batch = resume_info.batch if resume_info else 0

    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
        do_configure_logging_module=True,
    )

    user_metadata: dict[str, str] = {}
    if wandb_link := ml_logger.get_logger_url():
        user_metadata["wandb_link"] = wandb_link
    checkpoint_utils.add_renderer_name_to_user_metadata(user_metadata, config.renderer_name)
    model_info.warn_if_renderer_not_recommended(config.model_name, config.renderer_name)

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
            base_model=config.model_name, rank=config.lora_rank,
            user_metadata=user_metadata,
        )

    teacher_client = service_client.create_sampling_client(base_model=config.model_name)
    sampling_client = training_client.save_weights_and_get_sampling_client()

    n_batches = len(dataset)
    total_steps = n_batches * config.num_epochs
    if config.max_steps is not None:
        total_steps = min(total_steps, config.max_steps)

    logger.info(
        "Online artifact-only OPD: %d batches x %d epochs = %d planned steps "
        "(rollout_max_tokens=%d, rollout_temperature=%.2f, attempts=%d)",
        n_batches,
        config.num_epochs,
        min(n_batches * config.num_epochs, total_steps),
        config.rollout_max_tokens,
        config.rollout_temperature,
        config.rollout_attempts,
    )

    for epoch_idx in range(config.num_epochs):
        dataset.set_epoch(seed=epoch_idx)
        logger.info("Starting online epoch %d", epoch_idx)

        for batch_idx in range(start_batch if epoch_idx == 0 else 0, n_batches):
            step = epoch_idx * n_batches + batch_idx
            if config.max_steps is not None and step >= config.max_steps:
                break

            rows = dataset.get_batch(batch_idx)
            student_datums, teacher_prompt_inputs, rollout_metrics = sample_online_artifact_datums(
                rows,
                renderer,
                sampling_client,
                config,
                max_length=dataset._max_length,
                step=step,
            )

            if step == 0 and student_datums:
                int_tokens = list(student_datums[0].model_input.to_ints())
                weights = student_datums[0].loss_fn_inputs["weights"].data
                logger.info("\nOnline rollout example 0:")
                logger.info(format_colorized(
                    int_tokens, cast(list[float], weights), tokenizer,
                ))

            if not student_datums:
                metrics = {
                    "opd_online/no_valid_batch": 1.0,
                    "learning_rate": config.learning_rate * compute_schedule_lr_multiplier(
                        lr_schedule=config.lr_schedule, step=step, total_steps=total_steps,
                    ),
                    "progress": step / max(total_steps, 1),
                    **rollout_metrics,
                }
                ml_logger.log_metrics(metrics=metrics, step=step)
                logger.warning("Skipping step %d: no valid artifact write samples", step)
                continue

            topk_datums, topk_metrics = build_offline_topk_datums(
                student_datums,
                teacher_prompt_inputs,
                teacher_client,
                topk=config.topk,
                vocab_size=len(tokenizer),
                teacher_temperature=config.teacher_temperature,
            )
            metrics = do_update_topk(
                step=step,
                total_steps=total_steps,
                config=config,
                training_client=training_client,
                topk_datums=topk_datums,
                ml_logger=ml_logger,
                log_path=config.log_path,
                static_teacher_metrics={**rollout_metrics, **topk_metrics},
            )
            logger.info(
                "Online step %d: valid=%d/%d filter_rate=%.3f loss=%.4f",
                step,
                int(rollout_metrics.get("opd_online/valid_examples", 0.0)),
                int(rollout_metrics.get("opd_online/batch_examples", 0.0)),
                rollout_metrics.get("opd_online/filter_rate", 0.0),
                float(metrics.get("opd_loss", 0.0)),
            )

            # Refresh the student sampler so the next rollout is on-policy.
            sampling_client = training_client.save_weights_and_get_sampling_client()

    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": n_batches},
        ttl_seconds=None,
    )
    ml_logger.close()
    logger.info("Online artifact-only OPD training completed successfully")


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path as _Path

    _scripts = _Path(__file__).resolve().parent.parent.parent
    if str(_scripts) not in sys.path:
        sys.path.insert(0, str(_scripts))

    _load_env()

    parser = argparse.ArgumentParser(description="Offline OPD training (weight-format)")
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--renderer-name", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lr-schedule", default="cosine",
                        choices=["linear", "cosine", "constant"],
                        help="LR schedule (default: cosine)")
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--load-checkpoint-path", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--save-every", type=int, default=20,
                        help="Save checkpoint every N steps (0 = only at end)")
    parser.add_argument(
        "--pair-mode", choices=["first_last", "adjacent"], default="first_last",
        help=(
            "Pair construction mode. 'first_last' (default): one example per file, "
            "student = first version, teacher has all future feedback. "
            "'adjacent': one example per consecutive version pair."
        ),
    )
    parser.add_argument(
        "--use-gt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Append the last (ground-truth) version of each file to the teacher "
            "prompt as a reference block. Implements the SDFT golden-answer trick "
            "which sharpens teacher distributions on correct tokens. Default: true; "
            "pass --no-use-gt for ablations."
        ),
    )
    parser.add_argument(
        "--use-student",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Inject student artifact content into the teacher prompt as "
            "before-state context. Default: true; pass --no-use-student for ablations."
        ),
    )
    parser.add_argument(
        "--topk", type=int, default=20,
        help=(
            "Top-K vocabulary candidates for distillation (Tinker only). "
            "K=20 matches full-vocabulary KL in practice. Set to 0 to use "
            "the importance-sampling fallback (advantage = teacher_lp - student_lp)."
        ),
    )
    parser.add_argument(
        "--use-skyrl", action="store_true",
        help=(
            "Use SkyRL-compatible IS path (training_client.forward) instead of "
            "the Tinker teacher SamplingClient. Disables top-K distillation. "
            "NOTE: SkyRL does not support topk_prompt_logprobs so top-K KD "
            "is unavailable on this backend."
        ),
    )
    parser.add_argument(
        "--teacher-temperature", type=float, default=1.0,
        help=(
            "Softening temperature τ for the teacher distribution (Hinton 2015). "
            "τ=1.0 (default) leaves teacher probs unchanged. τ>1 flattens the "
            "distribution (raises entropy), making KD less like hard-label SFT. "
            "Recommended: τ=1.5–2.0 when opd/mean_teacher_entropy < 0.5 nat."
        ),
    )
    parser.add_argument(
        "--online-rollout", action="store_true",
        help=(
            "Enable online artifact-only OPD: sample completions from the current "
            "student prompt, keep only exactly-one write(path, content) tool-call "
            "responses for the expected artifact path, then distill from the "
            "teacher prompt. Does not modify the upstream prompt."
        ),
    )
    parser.add_argument(
        "--rollout-max-tokens", type=int, default=4096,
        help="Maximum tokens for each online student artifact rollout.",
    )
    parser.add_argument(
        "--rollout-temperature", type=float, default=1.0,
        help="Sampling temperature for online student artifact rollout.",
    )
    parser.add_argument(
        "--rollout-attempts", type=int, default=1,
        help="Number of sample/filter attempts per OPD example before filtering it.",
    )
    parser.add_argument(
        "--no-log-rollout-samples", action="store_true",
        help="Disable JSONL logging of online rollout samples and filter reasons.",
    )
    parser.add_argument(
        "--rollout-sample-log-chars", type=int, default=4000,
        help="Max characters of raw/parsed online rollout text to store per sample.",
    )
    parser.add_argument(
        "--log-teacher-prompts", action="store_true",
        help=(
            "Augment online_rollout_samples.jsonl records with the decoded "
            "teacher prompt (truncated to --rollout-sample-log-chars) and its "
            "token count. Off by default because teacher prompts can run up "
            "to the model's full context length."
        ),
    )
    parser.add_argument(
        "--artifact-only-rollout-instruction", action="store_true",
        help=(
            "Append a rollout-only instruction to the last user turn requiring "
            "exactly one write(path, content) tool call. Default is off to keep "
            "the rollout prompt matched to inference."
        ),
    )
    parser.add_argument(
        "--extract-version", choices=["v1", "v2"], default="v2",
        help=(
            "OPD extractor selection. "
        ),
    )
    parser.add_argument(
        "--rollout-pipeline",
        choices=["current", "legacy"],
        default="current",
        help=(
            "Rollout dispatch path. 'current' (default): in-house "
            "sample_async + parse-filter + canonicalization + retry. "
            "'legacy': cookbook do_group_rollout_and_filter_constant_reward "
            "+ assemble_training_data, training on raw sampled tokens (no "
            "parse-filter, no canonicalization, no retry) — matches the "
            "legacy tinker_opd recipe's training dynamics."
        ),
    )
    parser.add_argument(
        "--strip-thinking-from-history",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reasoning renderers (Qwen3/Kimi K2/DeepSeek thinking) strip "
            "<think>...</think> blocks from non-last assistant turns by "
            "default. SDFT teacher prompts always end with a redo user "
            "message, so every assistant turn (including the golden demo) "
            "sits in non-last position. Default --no-strip-thinking-from-history "
            "preserves the golden chain-of-thought so the teacher can attend "
            "to it (matches legacy tinker_opd.Config)."
        ),
    )
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model_name)
    renderer = renderers.get_renderer(args.renderer_name, tokenizer=tokenizer)
    if hasattr(renderer, "strip_thinking_from_history"):
        renderer.strip_thinking_from_history = args.strip_thinking_from_history
        logger.info(
            "Renderer %s: strip_thinking_from_history=%s",
            type(renderer).__name__, args.strip_thinking_from_history,
        )

    cfg = Config(
        log_path=args.log_path,
        model_name=args.model_name,
        renderer_name=args.renderer_name,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        lr_schedule=args.lr_schedule,
        num_epochs=args.num_epochs,
        save_every=args.save_every,
        load_checkpoint_path=args.load_checkpoint_path,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        max_steps=args.max_steps,
        base_url=args.base_url,
        use_skyrl=args.use_skyrl,
        pair_mode=args.pair_mode,
        use_gt=args.use_gt,
        use_student=args.use_student,
        topk=args.topk,
        teacher_temperature=args.teacher_temperature,
        online_rollout=args.online_rollout,
        rollout_max_tokens=args.rollout_max_tokens,
        rollout_temperature=args.rollout_temperature,
        rollout_attempts=args.rollout_attempts,
        log_rollout_samples=not args.no_log_rollout_samples,
        rollout_sample_log_chars=args.rollout_sample_log_chars,
        log_teacher_prompts=args.log_teacher_prompts,
        artifact_only_rollout_instruction=args.artifact_only_rollout_instruction,
        extract_version=args.extract_version,
        strip_thinking_from_history=args.strip_thinking_from_history,
        rollout_pipeline=args.rollout_pipeline,
    )

    if args.online_rollout:
        online_dataset = OnlineOPDRolloutDataset.from_weight_json(
            path=args.train_path,
            renderer=renderer,
            max_length=args.max_length,
            batch_size=args.batch_size,
            pair_mode=args.pair_mode,
            use_gt=args.use_gt,
            use_student=args.use_student,
            artifact_only_instruction=args.artifact_only_rollout_instruction,
            extract_version=args.extract_version,
        )
        main_online(cfg, online_dataset, renderer, tokenizer)
    else:
        dataset = OfflineOPDDataset.from_weight_json(
            path=args.train_path,
            renderer=renderer,
            max_length=args.max_length,
            batch_size=args.batch_size,
            pair_mode=args.pair_mode,
            use_gt=args.use_gt,
            use_student=args.use_student,
            extract_version=args.extract_version,
        )
        main(cfg, dataset)
