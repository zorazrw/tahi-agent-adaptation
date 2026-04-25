"""
Self-Distillation Fine-Tuning (SDFT)
Adapted from tinker-cookbook:
https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/distillation/sdft.py

Implements the SDFT algorithm from
`"Self-Distillation Enables Continual Learning" <https://arxiv.org/abs/2601.19897>`_
(Shenfeld et al., 2026). SDFT is an on-policy distillation method that learns new
skills from demonstrations while preserving prior capabilities.

**How it works:** A teacher model (frozen base weights) sees the question **and** a
golden answer as an in-context demonstration. The student sees only the question and
generates completions on-policy. The teacher's top-K token distribution at each
position is recovered via Tinker's ``topk_prompt_logprobs`` API and used as soft
targets for ``cross_entropy`` loss — approximating the paper's full-vocabulary
forward KL divergence.

Two distillation modes are supported (controlled by :class:`Config` ``.topk``):

- **Top-K distillation** (``topk > 0``, default): Recovers the teacher's top-K
  token distribution and trains with ``cross_entropy``. Validated to match
  full-vocabulary KL on the
  `reference implementation <https://github.com/idanshen/Self-Distillation>`_.

- **Per-token importance sampling** (``topk = 0``): Single-sample approximation
  using ``advantage = teacher_lp - student_lp`` with ``importance_sampling`` loss.

Example usage::

    # SDFT with top-K=20 distillation on tool-use data
    python -m tinker_cookbook.recipes.sdft.train \\
        model_name=Qwen/Qwen3.5-35B-A3B \\
        dataset=toolalpaca \\
        toolalpaca_data_path=~/Self-Distillation/data/tooluse_data/train_data \\
        groups_per_batch=128 \\
        learning_rate=5e-4 \\
        topk=20 \\
        lora_rank=64

See the `recipe README <https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/sdft>`_
for full setup instructions and continual learning results. For background on the
loss functions used, see the `Tinker loss docs <https://tinker-docs.thinkingmachines.ai/tinker/losses>`_.
"""

import asyncio
import logging
import datetime
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import chz
import tinker
import torch
from tinker.types import LossFnType

from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.display import colorize_example
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator, SamplingClientEvaluatorBuilder
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
)
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator, compute_trajectory_metrics
from tinker_cookbook.rl.rollouts import do_group_rollout_and_filter_constant_reward
from tinker_cookbook.rl.train import (
    save_checkpoint_and_get_sampling_client,
    train_step,
)
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    RLDataset,
    TrajectoryGroup,
)
from tinker_cookbook.utils import ml_log, trace

logger = logging.getLogger(__name__)

# DEFAULT_DEMO_TEMPLATE = (
#     "{question}\n\n"
#     "This is an example for a response to the question:\n"
#     "{golden_answer}\n\n"
#     "Now answer with a response of your own, including the thinking process."
# )

OPD_DEMO_TEMPLATE = (
    "{question}\n\n"
    "A previous attempt at this task received the following feedback and "
    "corrections from the user:\n"
    "{golden_answer}\n\n"
    "Now provide an improved response that incorporates the feedback."
)


@runtime_checkable
class SDFTBatchProvider(Protocol):
    """Protocol for SDFT datasets that return builders alongside golden answers."""

    def get_batch(self, index: int) -> tuple[Sequence[EnvGroupBuilder], list[str], list[str]]:
        """Return (env_group_builders, questions, golden_answers) for a batch.

        Each list has the same length (one per problem in the batch).
        """
        ...

    def __len__(self) -> int: ...


def build_sdft_teacher_prompt(
    question: str,
    golden_answer: str,
    renderer: renderers.Renderer,
    system_prompt: str | None = None,
    demo_template: str = OPD_DEMO_TEMPLATE,
) -> tinker.ModelInput:
    """Build teacher ModelInput with golden answer as an in-context demonstration.

    The teacher prompt presents the question alongside the golden answer so the
    model can attend to the demonstration when scoring student completions.

    Returns a ModelInput suitable for appending student completion tokens and
    computing logprobs via a SamplingClient.
    """
    teacher_content = demo_template.format(question=question, golden_answer=golden_answer)
    messages: list[renderers.Message] = []
    if system_prompt:
        msg: renderers.Message = {"role": "system", "content": system_prompt}  # type: ignore[typeddict-item]
        messages.append(msg)
    user_msg: renderers.Message = {"role": "user", "content": teacher_content}  # type: ignore[typeddict-item]
    messages.append(user_msg)
    return renderer.build_generation_prompt(messages)


def _extract_completion_tokens(
    datum: tinker.Datum,
    teacher_prompt: tinker.ModelInput,
    max_context_length: int,
) -> tuple[list[int], int, int, bool]:
    """Extract student completion tokens and compute teacher prompt length.

    Returns (completion_tokens, teacher_prompt_len, completion_start_in_student, was_truncated).
    completion_tokens may be empty if there are no completion tokens or context overflows.
    """
    mask = datum.loss_fn_inputs["mask"].to_torch()
    completion_mask_indices = torch.where(mask > 0)[0]
    teacher_prompt_len = teacher_prompt.length

    if len(completion_mask_indices) == 0:
        return [], teacher_prompt_len, 0, False

    # Reconstruct full student sequence (model_input is left-shifted, missing last target)
    student_full = datum.model_input.append_int(
        cast(int, datum.loss_fn_inputs["target_tokens"].data[-1])
    )
    student_full_tokens = student_full.to_ints()
    # Completion starts at first mask position + 1 (target is left-shifted)
    completion_start = int(completion_mask_indices[0].item()) + 1
    completion_tokens = student_full_tokens[completion_start:]

    available = max_context_length - teacher_prompt_len
    truncated = False
    if available <= 0:
        return [], teacher_prompt_len, completion_start, True
    if len(completion_tokens) > available:
        completion_tokens = completion_tokens[:available]
        truncated = True

    return completion_tokens, teacher_prompt_len, completion_start, truncated


def _build_teacher_forced_sequence(
    teacher_prompt: tinker.ModelInput,
    completion_tokens: list[int],
) -> tinker.ModelInput:
    """Append completion tokens to teacher prompt to form the teacher-forced sequence."""
    teacher_forced = teacher_prompt
    for token in completion_tokens:
        teacher_forced = teacher_forced.append_int(token)
    return teacher_forced


@trace.scope
async def compute_sdft_advantages(
    data_D: list[tinker.Datum],
    metadata_D: list[dict[str, int]],
    teacher_client: tinker.SamplingClient,
    teacher_prompts_P: list[tinker.ModelInput],
    max_context_length: int = 32768,
) -> dict[str, float]:
    """Replace advantages with teacher_lp - student_lp (per-token).

    For each datum, builds the full teacher sequence (teacher_prompt + completion
    tokens), computes teacher logprobs, and sets advantages to the per-token
    difference between teacher and student logprobs.

    Modifies data_D in-place (replaces the ``advantages`` field).

    Args:
        data_D: List of datums from rollout. Must have ``logprobs`` and ``mask``
            fields in ``loss_fn_inputs``.
        metadata_D: Per-datum metadata with ``group_idx`` mapping to teacher_prompts_P.
        teacher_client: SamplingClient for the teacher model.
        teacher_prompts_P: Per-problem teacher prompts (one per group in the batch).
        max_context_length: Maximum context for teacher logprob computation.
            Completion tokens are truncated if teacher_prompt + completion exceeds this.
    """
    teacher_full_sequences_D: list[tinker.ModelInput] = []
    teacher_prompt_lengths_D: list[int] = []
    completion_lengths_D: list[int] = []
    truncated_count = 0

    for i, datum in enumerate(data_D):
        group_idx = metadata_D[i]["group_idx"]
        teacher_prompt = teacher_prompts_P[group_idx]

        completion_tokens, teacher_prompt_len, _, was_truncated = _extract_completion_tokens(
            datum, teacher_prompt, max_context_length
        )
        if was_truncated:
            truncated_count += 1

        if not completion_tokens:
            teacher_full_sequences_D.append(teacher_prompt)
            teacher_prompt_lengths_D.append(teacher_prompt_len)
            completion_lengths_D.append(0)
            continue

        teacher_full = _build_teacher_forced_sequence(teacher_prompt, completion_tokens)
        teacher_full_sequences_D.append(teacher_full)
        teacher_prompt_lengths_D.append(teacher_prompt_len)
        completion_lengths_D.append(len(completion_tokens))

    # Compute teacher logprobs in parallel
    teacher_logprobs_D = await asyncio.gather(
        *[
            teacher_client.compute_logprobs_async(teacher_full)
            for teacher_full in teacher_full_sequences_D
        ]
    )

    # Replace advantages with teacher_lp - student_lp
    sampled_logprobs_D = [datum.loss_fn_inputs["logprobs"].to_torch() for datum in data_D]
    float_masks_D = [datum.loss_fn_inputs["mask"].to_torch().float() for datum in data_D]

    total_advantage_sum = 0.0
    total_mask_sum = 0.0
    total_teacher_lp_sum = 0.0
    total_student_lp_sum = 0.0

    for i, datum in enumerate(data_D):
        mask = float_masks_D[i]
        student_lp = sampled_logprobs_D[i]
        teacher_prompt_len = teacher_prompt_lengths_D[i]
        completion_len = completion_lengths_D[i]

        if completion_len == 0:
            continue

        raw_teacher_lps = teacher_logprobs_D[i]
        teacher_completion_lps = [
            lp if lp is not None else 0.0
            for lp in raw_teacher_lps[teacher_prompt_len : teacher_prompt_len + completion_len]
        ]
        teacher_lp_tensor = torch.tensor(teacher_completion_lps, dtype=torch.float32)

        new_advantages = torch.zeros_like(mask)
        completion_mask_indices = torch.where(mask > 0)[0]

        num_tokens = min(len(teacher_lp_tensor), len(completion_mask_indices))
        for t in range(num_tokens):
            idx = int(completion_mask_indices[t].item())
            new_advantages[idx] = teacher_lp_tensor[t] - student_lp[idx]

        datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(new_advantages)

        masked_advantages = new_advantages * mask
        total_advantage_sum += masked_advantages.sum().item()
        total_mask_sum += mask.sum().item()
        total_teacher_lp_sum += (teacher_lp_tensor[:num_tokens]).sum().item()
        total_student_lp_sum += sum(
            student_lp[int(completion_mask_indices[t].item())].item() for t in range(num_tokens)
        )

    metrics: dict[str, float] = {}
    if total_mask_sum > 0:
        metrics["sdft/mean_advantage"] = total_advantage_sum / total_mask_sum
        metrics["sdft/mean_teacher_lp"] = total_teacher_lp_sum / total_mask_sum
        metrics["sdft/mean_student_lp"] = total_student_lp_sum / total_mask_sum
    metrics["sdft/teacher_truncated_count"] = float(truncated_count)
    metrics["sdft/num_datums"] = float(len(data_D))

    return metrics


@trace.scope
async def build_topk_distillation_datums(
    data_D: list[tinker.Datum],
    metadata_D: list[dict[str, int]],
    teacher_client: tinker.SamplingClient,
    teacher_prompts_P: list[tinker.ModelInput],
    topk: int = 20,
    max_context_length: int = 32768,
    vocab_size: int | None = None,
    skip_first_n_tokens: int = 3,
) -> tuple[list[tinker.Datum], dict[str, float]]:
    """Build cross_entropy datums with top-K teacher soft targets.

    Teacher-forces each student completion through the teacher model to recover
    the teacher's top-K token distribution at each position using Tinker's
    ``topk_prompt_logprobs`` sampling API. Returns new datums with
    ``(N, K)``-shaped ``target_tokens`` and ``weights`` for ``cross_entropy``
    loss.

    This implements forward KL distillation restricted to the top-K vocabulary.
    At each of the T completion token positions, the loss is the cross-entropy
    between the teacher's renormalized top-K distribution and the student::

        L = (1/T) * sum_{t=1}^{T} [ -sum_{k=1}^{K} P_teacher(x_k|t) * log P_student(x_k|t) ]

    This is equivalent to forward KL (up to constant teacher entropy) over the
    top-K tokens that carry most of the probability mass. Validated to match
    full-vocabulary KL on the
    `reference implementation <https://github.com/idanshen/Self-Distillation>`_
    (68.04% vs 68.04% on tooluse with Qwen2.5-7B).

    Args:
        data_D: Datums from rollout (used for model_input and mask alignment).
        metadata_D: Per-datum metadata with ``group_idx`` mapping.
        teacher_client: SamplingClient for the teacher model.
        teacher_prompts_P: Per-problem teacher prompts (built by
            :func:`build_sdft_teacher_prompt`).
        topk: Number of top tokens to distill (K). K=20 is recommended.
        max_context_length: Maximum teacher context length.
        vocab_size: If set, filter out token IDs >= vocab_size (handles
            special tokens from vLLM that exceed the tokenizer's vocabulary).
        skip_first_n_tokens: Skip the first N completion tokens from the
            loss (default 3, matching the reference implementation).

    Returns:
        (new_datums, metrics) where new_datums have ``cross_entropy``
        loss_fn_inputs with ``target_tokens`` shape ``(N, K)`` and
        ``weights`` shape ``(N, K)``.
    """
    # Step 1: Build teacher-forced sequences
    teacher_forced_sequences_D: list[tinker.ModelInput] = []
    teacher_prompt_lengths_D: list[int] = []
    completion_lengths_D: list[int] = []
    truncated_count = 0

    for i, datum in enumerate(data_D):
        group_idx = metadata_D[i]["group_idx"]
        teacher_prompt = teacher_prompts_P[group_idx]

        completion_tokens, teacher_prompt_len, _, was_truncated = _extract_completion_tokens(
            datum, teacher_prompt, max_context_length
        )
        if was_truncated:
            truncated_count += 1

        if not completion_tokens:
            teacher_forced_sequences_D.append(teacher_prompt)
            teacher_prompt_lengths_D.append(teacher_prompt_len)
            completion_lengths_D.append(0)
            continue

        teacher_forced = _build_teacher_forced_sequence(teacher_prompt, completion_tokens)
        teacher_forced_sequences_D.append(teacher_forced)
        teacher_prompt_lengths_D.append(teacher_prompt_len)
        completion_lengths_D.append(len(completion_tokens))

    # Step 2: Get top-K logprobs from teacher in parallel
    topk_responses_D = await asyncio.gather(
        *[
            teacher_client.sample_async(
                prompt=teacher_forced,
                num_samples=1,
                sampling_params=tinker.SamplingParams(max_tokens=1),
                include_prompt_logprobs=True,
                topk_prompt_logprobs=topk,
            )
            for teacher_forced in teacher_forced_sequences_D
        ]
    )

    # Step 3: Build new datums with (N, K) shaped target_tokens and weights.
    # First pass: collect raw weights and count completion tokens per datum.
    raw_datums: list[tuple[torch.Tensor, torch.Tensor, int]] = []  # (targets, weights, n_comp)
    total_completion_tokens = 0.0
    total_teacher_entropy = 0.0

    for i, datum in enumerate(data_D):
        mask = datum.loss_fn_inputs["mask"].to_torch()
        completion_mask_indices = torch.where(mask > 0)[0]
        N = datum.model_input.length
        completion_len = completion_lengths_D[i]
        teacher_prompt_len = teacher_prompt_lengths_D[i]

        target_tokens_NK = torch.zeros(N, topk, dtype=torch.long)
        weights_NK = torch.zeros(N, topk, dtype=torch.float32)
        n_completion_positions = 0

        if completion_len > 0 and len(completion_mask_indices) > 0:
            topk_all = topk_responses_D[i].topk_prompt_logprobs

            num_tokens = min(completion_len, len(completion_mask_indices))
            for t in range(num_tokens):
                # Skip first N completion tokens (reference skips 3)
                if t < skip_first_n_tokens:
                    continue

                teacher_pos = teacher_prompt_len + t
                student_pos = int(completion_mask_indices[t].item())

                if topk_all is None or teacher_pos >= len(topk_all):
                    continue
                topk_entries = topk_all[teacher_pos]
                if topk_entries is None:
                    continue

                # Filter out token IDs that exceed the student's vocab size
                # (teacher sampling may return IDs for special/added tokens)
                filtered = [
                    (tok_id, lp)
                    for tok_id, lp in topk_entries[:topk]
                    if vocab_size is None or tok_id < vocab_size
                ]
                if not filtered:
                    continue

                k_actual = len(filtered)
                token_ids = torch.tensor([tok_id for tok_id, _ in filtered], dtype=torch.long)
                logprobs = torch.tensor([lp for _, lp in filtered], dtype=torch.float32)

                # Renormalize over top-K via logsumexp
                logprobs -= torch.logsumexp(logprobs, dim=0)
                probs = logprobs.exp()

                target_tokens_NK[student_pos, :k_actual] = token_ids
                weights_NK[student_pos, :k_actual] = probs
                n_completion_positions += 1

                # Teacher entropy for monitoring (H = -sum p log p)
                total_teacher_entropy += -(probs * logprobs).sum().item()

            total_completion_tokens += num_tokens

        raw_datums.append((target_tokens_NK, weights_NK, n_completion_positions))

    # No weight normalization — Tinker's CE loss uses raw sum, same convention
    # as the SFT loss. Both produce gradients proportional to num_tokens * lr.
    # Use the same LR range for both.
    new_datums: list[tinker.Datum] = []

    for i, datum in enumerate(data_D):
        target_tokens_NK, weights_NK, n_comp = raw_datums[i]

        new_datum = tinker.Datum(
            model_input=datum.model_input,
            loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_torch(target_tokens_NK),
                "weights": tinker.TensorData.from_torch(weights_NK),
            },
        )
        new_datums.append(new_datum)

    metrics: dict[str, float] = {
        "sdft/teacher_truncated_count": float(truncated_count),
        "sdft/num_datums": float(len(data_D)),
        "sdft/topk": float(topk),
    }
    if total_completion_tokens > 0:
        metrics["sdft/total_completion_tokens"] = total_completion_tokens
        metrics["sdft/mean_teacher_entropy"] = total_teacher_entropy / total_completion_tokens

    return new_datums, metrics


@chz.chz
class Config:
    """Configuration for SDFT training.

    Key parameters:

    - ``topk``: Number of top tokens for distillation (default 20). Set to 0
      for the importance-sampling fallback. K=20 matches full-vocabulary KL
      in practice.
    - ``learning_rate``: For LoRA, use 5e-4 to 1e-3. The top-K CE loss
      produces larger gradients than SFT at the same LR due to more
      completion tokens per step (on-policy generation), so use the lower
      end of the range.
    - ``teacher_sync_every``: Optional periodic hard-sync of student weights
      into the teacher (approximating EMA). ``None`` = static frozen teacher,
      which works comparably to EMA in our experiments.

    See :func:`main` for the training loop.
    """

    # Model
    model_name: str
    renderer_name: str | None = None
    lora_rank: int = 128
    base_url: str | None = None

    # Training
    learning_rate: float = 2e-5
    max_tokens: int = 2048
    temperature: float = 1.0
    loss_fn: LossFnType = "cross_entropy"

    # SDFT-specific
    topk: int = 20
    demo_template: str = OPD_DEMO_TEMPLATE
    system_prompt: str | None = None
    teacher_sync_every: int | None = None
    max_context_length: int = 32768

    # Evaluation
    evaluator_builders: list[SamplingClientEvaluatorBuilder] = chz.field(default_factory=list)
    eval_every: int = 20
    save_every: int = 20

    # Standard infra
    num_substeps: int = 1
    log_path: str = chz.field(munger=lambda _, s: str(Path(s).expanduser()))
    wandb_project: str | None = None
    wandb_name: str | None = None
    load_checkpoint_path: str | None = None
    max_steps: int | None = None
    epochs: int = 1

    enable_trace: bool = False
    span_chart_every: int = 0