"""Offline OPD (On-Policy Distillation) training on weight-format session JSON.

Unlike the on-policy ``tinker_opd.py`` which does student rollout + teacher
scoring, this script operates on **historical trajectories**:

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
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, cast

import chz
import tinker
import torch

from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.utils import ml_log, trace
from tinker_cookbook.utils.format_colorized import format_colorized
from tinker_cookbook.utils.lr_scheduling import LRSchedule, compute_schedule_lr_multiplier

from .formatter import OfflineOPDDataset

logger = logging.getLogger(__name__)


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
    use_gt: bool = False            # append last-version artifact to teacher prompt
    use_student: bool = False       # append student artifact to teacher prompt

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

    Adapted from ``tinker_opd.build_topk_distillation_datums`` for the offline
    setting: teacher prompts are pre-built (stored in the dataset) and map
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
        "--use-gt", action="store_true",
        help=(
            "Append the last (ground-truth) version of each file to the teacher "
            "prompt as a reference block. Implements the SDFT golden-answer trick "
            "which sharpens teacher distributions on correct tokens."
        ),
    )
    parser.add_argument(
        "--use-student", action="store_true",
        help=(
            "Inject student artifact content into the teacher prompt as "
            "before-state context. Can be combined with --use-gt."
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
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model_name)
    renderer = renderers.get_renderer(args.renderer_name, tokenizer=tokenizer)

    dataset = OfflineOPDDataset.from_weight_json(
        path=args.train_path,
        renderer=renderer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        pair_mode=args.pair_mode,
        use_gt=args.use_gt,
        use_student=args.use_student,
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
    )

    main(cfg, dataset)
