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

# TODO: Update this according to the new formatter script

import asyncio
import logging
import datetime
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, cast, runtime_checkable

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

# Used in the list-style teacher prompt (when both question and golden_answer
# are chat messages) to ask the model to redo the response from scratch. The
# default text fits both the "next-iteration trajectory" golden and the
# "LLM-summarized user intent" golden produced by ``summarize_followups.py``.
OPD_REDO_MESSAGE = (
    "The assistant messages above are the user follow-ups from a previous session based on the given chat history. "
    "Please think about the reason why the user asked these specific follow-ups, and reason through the user preferences reflected by them. "
    "Then, provide a response to the following user request that incorporates the feedback. "
)


@runtime_checkable
class SDFTBatchProvider(Protocol):
    """Protocol for SDFT datasets that return builders alongside golden answers.

    Implementations may additionally expose ``set_epoch(seed: int) -> None``
    to permute the row order at epoch boundaries. The OPD trainer calls it
    via ``hasattr`` so the hook stays optional for backwards compatibility.
    """

    def get_batch(
        self, index: int
    ) -> tuple[Sequence[EnvGroupBuilder], list[str | list[renderers.Message]], list[str]]:
        """Return (env_group_builders, questions, golden_answers) for a batch.

        Each list has the same length (one per problem in the batch).
        """
        ...

    def __len__(self) -> int: ...


def build_sdft_teacher_prompt(
    question: str | list[renderers.Message],
    golden_answer: str,
    renderer: renderers.Renderer,
    system_prompt: str | None = None,
    demo_template: str = OPD_DEMO_TEMPLATE,
    chat_redo_message: str = OPD_REDO_MESSAGE,
) -> tinker.ModelInput:
    """Build teacher ModelInput with golden answer as an in-context demonstration.

    The teacher prompt presents the question alongside the golden answer so the
    model can attend to the demonstration when scoring student completions.

    When ``question`` is a chat list and ``golden_answer`` is also a chat list
    (the "next-iteration" or summary form), the prompt is built as
    ``question + golden_answer + [redo]`` where ``redo`` carries the
    ``chat_redo_message`` text. When ``question`` is a plain string, the
    ``demo_template`` is used to build a single templated user turn.

    Returns a ModelInput suitable for appending student completion tokens and
    computing logprobs via a SamplingClient.
    """
    if isinstance(question, list):
        if not isinstance(golden_answer, list):
            raise TypeError(
                f"Expected list[Message] golden_answer when question is a list; got {type(golden_answer)}"
            )
        redo: renderers.Message = {
            "role": "user",
            "content": chat_redo_message,
        }  # type: ignore[typeddict-item]

        # Peel the trailing user turn(s) (the original task request) off the
        # end of `question` and re-attach them AFTER the demo + redo so the
        # task instruction is the last thing the teacher sees before
        # generating its assistant turn.
        head = list(question)
        trailing_user: list[renderers.Message] = []
        if head:
            trailing_user.insert(0, head.pop())

        teacher_messages = head + list(golden_answer) + [redo] + trailing_user
        return renderer.build_generation_prompt(teacher_messages)

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


def _build_student_forced_sequence(datum: tinker.Datum) -> tinker.ModelInput:
    """Reconstruct the full student prompt + sampled completion sequence."""
    target_tokens = datum.loss_fn_inputs["target_tokens"].data
    if not target_tokens:
        return datum.model_input
    return datum.model_input.append_int(cast(int, target_tokens[-1]))


def _renormalized_topk_distribution(
    topk_entries: Sequence[tuple[int, float]],
    topk: int,
    vocab_size: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Return filtered top-k token ids, logprobs, and probs renormalized over top-k."""
    filtered = [
        (tok_id, lp)
        for tok_id, lp in topk_entries[:topk]
        if vocab_size is None or tok_id < vocab_size
    ]
    if not filtered:
        return None

    token_ids = torch.tensor([tok_id for tok_id, _ in filtered], dtype=torch.long)
    logprobs = torch.tensor([lp for _, lp in filtered], dtype=torch.float32)
    logprobs -= torch.logsumexp(logprobs, dim=0)
    probs = logprobs.exp()
    return token_ids, logprobs, probs


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
    student_client: tinker.SamplingClient | None = None,
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
        student_client: Optional SamplingClient for the current student. If
            provided, a second forced top-K query is made on the student rollout
            sequence to compute teacher/student top-K overlap and entropy gap.
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
    student_forced_sequences_D: list[tinker.ModelInput] = []
    teacher_prompt_lengths_D: list[int] = []
    completion_starts_D: list[int] = []
    completion_lengths_D: list[int] = []
    truncated_count = 0

    for i, datum in enumerate(data_D):
        group_idx = metadata_D[i]["group_idx"]
        teacher_prompt = teacher_prompts_P[group_idx]
        student_forced_sequences_D.append(_build_student_forced_sequence(datum))

        completion_tokens, teacher_prompt_len, completion_start, was_truncated = _extract_completion_tokens(
            datum, teacher_prompt, max_context_length
        )
        if was_truncated:
            truncated_count += 1

        if not completion_tokens:
            teacher_forced_sequences_D.append(teacher_prompt)
            teacher_prompt_lengths_D.append(teacher_prompt_len)
            completion_starts_D.append(completion_start)
            completion_lengths_D.append(0)
            continue

        teacher_forced = _build_teacher_forced_sequence(teacher_prompt, completion_tokens)
        teacher_forced_sequences_D.append(teacher_forced)
        teacher_prompt_lengths_D.append(teacher_prompt_len)
        completion_starts_D.append(completion_start)
        completion_lengths_D.append(len(completion_tokens))

    # Step 2: Get top-K logprobs from teacher in parallel
    teacher_topk_task = asyncio.gather(
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
    student_topk_task = None
    if student_client is not None:
        student_topk_task = asyncio.gather(
            *[
                student_client.sample_async(
                    prompt=student_forced,
                    num_samples=1,
                    sampling_params=tinker.SamplingParams(max_tokens=1),
                    include_prompt_logprobs=True,
                    topk_prompt_logprobs=topk,
                )
                for student_forced in student_forced_sequences_D
            ]
        )
    if student_topk_task is None:
        topk_responses_D = await teacher_topk_task
        student_topk_responses_D = None
    else:
        topk_responses_D, student_topk_responses_D = await asyncio.gather(
            teacher_topk_task,
            student_topk_task,
        )

    # Step 3: Build new datums with (N, K) shaped target_tokens and weights.
    # First pass: collect raw weights and count completion tokens per datum.
    raw_datums: list[tuple[torch.Tensor, torch.Tensor, int]] = []  # (targets, weights, n_comp)
    total_completion_tokens = 0.0
    total_teacher_entropy = 0.0
    total_teacher_entropy_positions = 0.0
    total_topk_overlap = 0.0
    total_student_entropy = 0.0
    total_aligned_teacher_entropy = 0.0
    total_topk_metric_positions = 0.0
    # Top-K KL accumulators. For positions where both teacher and student
    # top-K are available, we compute KL on the *union* of top-K token ids
    # using the smallest top-K log-prob as a floor for tokens missing from
    # the other side. Forward = KL(teacher || student) (the loss direction);
    # reverse = KL(student || teacher); sym = mean of the two.
    total_kl_forward = 0.0
    total_kl_reverse = 0.0

    for i, datum in enumerate(data_D):
        mask = datum.loss_fn_inputs["mask"].to_torch()
        completion_mask_indices = torch.where(mask > 0)[0]
        N = datum.model_input.length
        completion_len = completion_lengths_D[i]
        teacher_prompt_len = teacher_prompt_lengths_D[i]
        completion_start = completion_starts_D[i]

        target_tokens_NK = torch.zeros(N, topk, dtype=torch.long)
        weights_NK = torch.zeros(N, topk, dtype=torch.float32)
        n_completion_positions = 0

        if completion_len > 0 and len(completion_mask_indices) > 0:
            topk_all = topk_responses_D[i].topk_prompt_logprobs
            student_topk_all = (
                None
                if student_topk_responses_D is None
                else student_topk_responses_D[i].topk_prompt_logprobs
            )

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

                teacher_distribution = _renormalized_topk_distribution(
                    topk_entries, topk=topk, vocab_size=vocab_size
                )
                if teacher_distribution is None:
                    continue

                token_ids, logprobs, probs = teacher_distribution
                k_actual = len(token_ids)

                target_tokens_NK[student_pos, :k_actual] = token_ids
                weights_NK[student_pos, :k_actual] = probs
                n_completion_positions += 1

                # Teacher entropy for monitoring (H = -sum p log p)
                teacher_entropy = -(probs * logprobs).sum().item()
                total_teacher_entropy += teacher_entropy
                total_teacher_entropy_positions += 1

                if student_topk_all is None:
                    continue

                # ``topk_prompt_logprobs`` indexes the observed token in the
                # forced prompt. ``student_pos`` is the left-shifted loss
                # position, so the generated token itself is one position later.
                student_topk_pos = completion_start + t
                if student_topk_pos >= len(student_topk_all):
                    continue
                student_topk_entries = student_topk_all[student_topk_pos]
                if student_topk_entries is None:
                    continue

                student_distribution = _renormalized_topk_distribution(
                    student_topk_entries, topk=topk, vocab_size=vocab_size
                )
                if student_distribution is None:
                    continue

                student_token_ids, student_logprobs, student_probs = student_distribution
                teacher_id_list = token_ids.tolist()
                student_id_list = student_token_ids.tolist()
                teacher_ids = set(teacher_id_list)
                student_ids = set(student_id_list)
                overlap_denominator = min(topk, len(teacher_ids), len(student_ids))
                if overlap_denominator == 0:
                    continue

                total_topk_overlap += len(teacher_ids & student_ids) / overlap_denominator
                total_student_entropy += -(student_probs * student_logprobs).sum().item()
                total_aligned_teacher_entropy += teacher_entropy

                # Top-K KL on the union of token ids. Tokens that fall outside
                # the other distribution's top-K are floored to its smallest
                # observed log-prob -- this is a biased underestimate when the
                # modes agree and a slight overestimate when they diverge, but
                # gives a stable monitoring signal.
                teacher_lp_by_id = dict(zip(teacher_id_list, logprobs.tolist()))
                teacher_p_by_id = dict(zip(teacher_id_list, probs.tolist()))
                student_lp_by_id = dict(zip(student_id_list, student_logprobs.tolist()))
                student_p_by_id = dict(zip(student_id_list, student_probs.tolist()))

                teacher_floor_lp = float(logprobs.min().item())
                student_floor_lp = float(student_logprobs.min().item())

                kl_fwd = 0.0
                for tid, p_t in teacher_p_by_id.items():
                    log_p_t = teacher_lp_by_id[tid]
                    log_p_s = student_lp_by_id.get(tid, student_floor_lp)
                    kl_fwd += p_t * (log_p_t - log_p_s)

                kl_rev = 0.0
                for sid, p_s in student_p_by_id.items():
                    log_p_s = student_lp_by_id[sid]
                    log_p_t = teacher_lp_by_id.get(sid, teacher_floor_lp)
                    kl_rev += p_s * (log_p_s - log_p_t)

                total_kl_forward += kl_fwd
                total_kl_reverse += kl_rev
                total_topk_metric_positions += 1

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
    if total_teacher_entropy_positions > 0:
        metrics["sdft/mean_teacher_entropy"] = (
            total_teacher_entropy / total_teacher_entropy_positions
        )
    if total_topk_metric_positions > 0:
        mean_teacher_entropy = total_aligned_teacher_entropy / total_topk_metric_positions
        mean_student_entropy = total_student_entropy / total_topk_metric_positions
        metrics["sdft/topk_overlap_ratio"] = total_topk_overlap / total_topk_metric_positions
        metrics["sdft/topk_metric_positions"] = total_topk_metric_positions
        metrics["sdft/mean_student_entropy"] = mean_student_entropy
        metrics["sdft/topk_entropy_gap_teacher_minus_student"] = (
            mean_teacher_entropy - mean_student_entropy
        )
        metrics["sdft/topk_entropy_gap_abs"] = abs(mean_teacher_entropy - mean_student_entropy)
        mean_kl_forward = total_kl_forward / total_topk_metric_positions
        mean_kl_reverse = total_kl_reverse / total_topk_metric_positions
        metrics["sdft/topk_kl_forward"] = mean_kl_forward
        metrics["sdft/topk_kl_reverse"] = mean_kl_reverse
        metrics["sdft/topk_kl_sym"] = 0.5 * (mean_kl_forward + mean_kl_reverse)

    return new_datums, metrics


@trace.scope
async def build_reverse_kl_datums(
    data_D: list[tinker.Datum],
    metadata_D: list[dict[str, int]],
    teacher_client: tinker.SamplingClient,
    teacher_prompts_P: list[tinker.ModelInput],
    student_client: tinker.SamplingClient | None = None,
    topk: int = 0,
    max_context_length: int = 32768,
    vocab_size: int | None = None,
    skip_first_n_tokens: int = 3,
) -> tuple[list[tinker.Datum], dict[str, float]]:
    """Build ``importance_sampling`` datums for reverse-KL distillation.

    Minimizes ``KL(P_student || P_teacher)`` via the standard REINFORCE
    estimator at each student-sampled completion token::

        A_t = log p_teacher(x_t) - log p_student(x_t),  x_t ~ P_student
        loss = -mean_t [ A_t * log p_student(x_t) ]

    This is the mode-seeking dual of the forward-KL top-K cross-entropy used
    by :func:`build_topk_distillation_datums`. The advantage signal is identical
    in form to :func:`compute_sdft_advantages`; the additional structure here
    is Variant B's optional top-K monitoring path that issues teacher *and*
    student top-K queries in parallel with the teacher logprob query so an
    operator can verify reverse top-K KL is actually decreasing.

    Implementation notes:

    * The exact teacher token logprobs are obtained via
      ``compute_logprobs_async`` on the teacher-forced sequence (no top-K
      flooring bias on the gradient itself).
    * When ``topk > 0`` AND ``student_client`` is provided, additional top-K
      queries are issued against both teacher and student to log forward /
      reverse top-K KL, top-K overlap, and entropy gap metrics. Those metrics
      never influence the gradient.
    * The first ``skip_first_n_tokens`` completion positions are excluded
      from both the gradient and the monitoring metrics, matching the
      reference SDFT implementation and the forward-KL builder above.

    Returns:
        ``(new_datums, metrics)`` where each new datum has its ``advantages``
        field populated with ``teacher_lp - student_lp`` over completion-mask
        positions. Pass these to ``train_step`` with
        ``loss_fn="importance_sampling"``.
    """
    # Step 1: build teacher-forced and (optionally) student-forced sequences.
    teacher_full_sequences_D: list[tinker.ModelInput] = []
    student_forced_sequences_D: list[tinker.ModelInput] = []
    teacher_prompt_lengths_D: list[int] = []
    completion_starts_D: list[int] = []
    completion_lengths_D: list[int] = []
    truncated_count = 0

    enable_topk_metrics = topk > 0 and student_client is not None

    for i, datum in enumerate(data_D):
        group_idx = metadata_D[i]["group_idx"]
        teacher_prompt = teacher_prompts_P[group_idx]
        if enable_topk_metrics:
            student_forced_sequences_D.append(_build_student_forced_sequence(datum))

        completion_tokens, teacher_prompt_len, completion_start, was_truncated = _extract_completion_tokens(
            datum, teacher_prompt, max_context_length
        )
        if was_truncated:
            truncated_count += 1

        if not completion_tokens:
            teacher_full_sequences_D.append(teacher_prompt)
            teacher_prompt_lengths_D.append(teacher_prompt_len)
            completion_starts_D.append(completion_start)
            completion_lengths_D.append(0)
            continue

        teacher_full = _build_teacher_forced_sequence(teacher_prompt, completion_tokens)
        teacher_full_sequences_D.append(teacher_full)
        teacher_prompt_lengths_D.append(teacher_prompt_len)
        completion_starts_D.append(completion_start)
        completion_lengths_D.append(len(completion_tokens))

    # Step 2: kick off the teacher logprob query and (optionally) the
    # teacher/student top-K monitoring queries in parallel.
    teacher_logprob_task = asyncio.gather(
        *[teacher_client.compute_logprobs_async(tf) for tf in teacher_full_sequences_D]
    )
    teacher_topk_task = None
    student_topk_task = None
    if enable_topk_metrics:
        teacher_topk_task = asyncio.gather(
            *[
                teacher_client.sample_async(
                    prompt=tf,
                    num_samples=1,
                    sampling_params=tinker.SamplingParams(max_tokens=1),
                    include_prompt_logprobs=True,
                    topk_prompt_logprobs=topk,
                )
                for tf in teacher_full_sequences_D
            ]
        )
        assert student_client is not None
        student_topk_task = asyncio.gather(
            *[
                student_client.sample_async(
                    prompt=sf,
                    num_samples=1,
                    sampling_params=tinker.SamplingParams(max_tokens=1),
                    include_prompt_logprobs=True,
                    topk_prompt_logprobs=topk,
                )
                for sf in student_forced_sequences_D
            ]
        )

    if enable_topk_metrics:
        teacher_logprobs_D, teacher_topk_responses_D, student_topk_responses_D = await asyncio.gather(
            teacher_logprob_task, teacher_topk_task, student_topk_task
        )
    else:
        teacher_logprobs_D = await teacher_logprob_task
        teacher_topk_responses_D = None
        student_topk_responses_D = None

    sampled_logprobs_D = [datum.loss_fn_inputs["logprobs"].to_torch() for datum in data_D]
    float_masks_D = [datum.loss_fn_inputs["mask"].to_torch().float() for datum in data_D]

    # Gradient-signal accumulators.
    total_advantage_sum = 0.0
    total_abs_advantage_sum = 0.0
    total_advantage_positions = 0.0
    total_mask_sum = 0.0
    total_teacher_lp_sum = 0.0
    total_student_lp_sum = 0.0

    # Top-K monitoring accumulators (populated only when enable_topk_metrics).
    total_topk_overlap = 0.0
    total_kl_forward = 0.0
    total_kl_reverse = 0.0
    total_topk_metric_positions = 0.0
    total_teacher_entropy = 0.0
    total_student_entropy = 0.0

    new_datums: list[tinker.Datum] = []

    for i, datum in enumerate(data_D):
        mask = float_masks_D[i]
        student_lp = sampled_logprobs_D[i]
        teacher_prompt_len = teacher_prompt_lengths_D[i]
        completion_len = completion_lengths_D[i]
        completion_start = completion_starts_D[i]

        new_advantages = torch.zeros_like(mask)

        if completion_len > 0:
            raw_teacher_lps = teacher_logprobs_D[i]
            teacher_completion_lps = [
                lp if lp is not None else 0.0
                for lp in raw_teacher_lps[teacher_prompt_len : teacher_prompt_len + completion_len]
            ]
            teacher_lp_tensor = torch.tensor(teacher_completion_lps, dtype=torch.float32)

            completion_mask_indices = torch.where(mask > 0)[0]
            num_tokens = min(len(teacher_lp_tensor), len(completion_mask_indices))
            for t in range(num_tokens):
                if t < skip_first_n_tokens:
                    continue
                idx = int(completion_mask_indices[t].item())
                a_t = float(teacher_lp_tensor[t].item()) - float(student_lp[idx].item())
                new_advantages[idx] = a_t
                total_teacher_lp_sum += float(teacher_lp_tensor[t].item())
                total_student_lp_sum += float(student_lp[idx].item())
                total_advantage_sum += a_t
                total_abs_advantage_sum += abs(a_t)
                total_advantage_positions += 1.0

            total_mask_sum += mask.sum().item()

        # Optional top-K monitoring on the renormalized top-K distribution.
        if (
            enable_topk_metrics
            and completion_len > 0
            and teacher_topk_responses_D is not None
            and student_topk_responses_D is not None
        ):
            teacher_topk_all = teacher_topk_responses_D[i].topk_prompt_logprobs
            student_topk_all = student_topk_responses_D[i].topk_prompt_logprobs
            completion_mask_indices = torch.where(mask > 0)[0]
            num_tokens = min(completion_len, len(completion_mask_indices))
            for t in range(num_tokens):
                if t < skip_first_n_tokens:
                    continue
                teacher_pos = teacher_prompt_len + t
                # ``topk_prompt_logprobs`` indexes the observed token in the
                # forced prompt; the generated token sits one position later
                # than the left-shifted student loss position.
                student_pos = completion_start + t
                if (
                    teacher_topk_all is None
                    or teacher_pos >= len(teacher_topk_all)
                    or student_topk_all is None
                    or student_pos >= len(student_topk_all)
                ):
                    continue
                teacher_entries = teacher_topk_all[teacher_pos]
                student_entries = student_topk_all[student_pos]
                if teacher_entries is None or student_entries is None:
                    continue

                teacher_dist = _renormalized_topk_distribution(
                    teacher_entries, topk=topk, vocab_size=vocab_size
                )
                student_dist = _renormalized_topk_distribution(
                    student_entries, topk=topk, vocab_size=vocab_size
                )
                if teacher_dist is None or student_dist is None:
                    continue

                t_ids, t_lps, t_ps = teacher_dist
                s_ids, s_lps, s_ps = student_dist
                t_id_list = t_ids.tolist()
                s_id_list = s_ids.tolist()
                t_set = set(t_id_list)
                s_set = set(s_id_list)
                overlap_denom = min(topk, len(t_set), len(s_set))
                if overlap_denom == 0:
                    continue

                t_lp_by_id = dict(zip(t_id_list, t_lps.tolist()))
                t_p_by_id = dict(zip(t_id_list, t_ps.tolist()))
                s_lp_by_id = dict(zip(s_id_list, s_lps.tolist()))
                s_p_by_id = dict(zip(s_id_list, s_ps.tolist()))
                t_floor = float(t_lps.min().item())
                s_floor = float(s_lps.min().item())

                kl_fwd = 0.0
                for tid, p_t in t_p_by_id.items():
                    kl_fwd += p_t * (t_lp_by_id[tid] - s_lp_by_id.get(tid, s_floor))
                kl_rev = 0.0
                for sid, p_s in s_p_by_id.items():
                    kl_rev += p_s * (s_lp_by_id[sid] - t_lp_by_id.get(sid, t_floor))

                total_topk_overlap += len(t_set & s_set) / overlap_denom
                total_teacher_entropy += -(t_ps * t_lps).sum().item()
                total_student_entropy += -(s_ps * s_lps).sum().item()
                total_kl_forward += float(kl_fwd)
                total_kl_reverse += float(kl_rev)
                total_topk_metric_positions += 1.0

        # Build a fresh datum with the advantage field overwritten so the
        # train_step's ``importance_sampling`` loss sees the reverse-KL
        # signal. Other ``loss_fn_inputs`` (target_tokens, weights, mask,
        # logprobs) are preserved from the rollout.
        new_loss_fn_inputs = dict(datum.loss_fn_inputs)
        new_loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(new_advantages)
        new_datums.append(
            tinker.Datum(model_input=datum.model_input, loss_fn_inputs=new_loss_fn_inputs)
        )

    metrics: dict[str, float] = {
        "sdft/teacher_truncated_count": float(truncated_count),
        "sdft/num_datums": float(len(data_D)),
        "sdft/topk": float(topk),
        # Numeric flag so W&B/timeseries plots can split runs by direction
        # (string-valued metrics aren't first-class in ml_log).
        "sdft/kl_direction_is_reverse": 1.0,
    }
    if total_advantage_positions > 0:
        mean_advantage = total_advantage_sum / total_advantage_positions
        metrics["sdft/reverse_kl_mean_advantage"] = mean_advantage
        metrics["sdft/reverse_kl_mean_abs_advantage"] = (
            total_abs_advantage_sum / total_advantage_positions
        )
        # Per-token reverse-KL surrogate: E_x~pi[log pi - log p_T] = -mean(A).
        metrics["sdft/reverse_kl_surrogate"] = -mean_advantage
        metrics["sdft/mean_teacher_lp"] = total_teacher_lp_sum / total_advantage_positions
        metrics["sdft/mean_student_lp"] = total_student_lp_sum / total_advantage_positions
    if total_mask_sum > 0:
        metrics["sdft/total_mask_sum"] = total_mask_sum
    if total_topk_metric_positions > 0:
        metrics["sdft/topk_overlap_ratio"] = total_topk_overlap / total_topk_metric_positions
        metrics["sdft/topk_metric_positions"] = total_topk_metric_positions
        mean_teacher_entropy = total_teacher_entropy / total_topk_metric_positions
        mean_student_entropy = total_student_entropy / total_topk_metric_positions
        metrics["sdft/mean_teacher_entropy"] = mean_teacher_entropy
        metrics["sdft/mean_student_entropy"] = mean_student_entropy
        metrics["sdft/topk_entropy_gap_teacher_minus_student"] = (
            mean_teacher_entropy - mean_student_entropy
        )
        metrics["sdft/topk_entropy_gap_abs"] = abs(mean_teacher_entropy - mean_student_entropy)
        mean_kl_forward = total_kl_forward / total_topk_metric_positions
        mean_kl_reverse = total_kl_reverse / total_topk_metric_positions
        metrics["sdft/topk_kl_forward"] = mean_kl_forward
        metrics["sdft/topk_kl_reverse"] = mean_kl_reverse
        metrics["sdft/topk_kl_sym"] = 0.5 * (mean_kl_forward + mean_kl_reverse)

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
    # Direction of the teacher/student KL the trainer minimizes.
    # ``"forward"`` reproduces the validated top-K cross_entropy distillation
    # path (mode-covering, KL(P_teacher || P_student)). ``"reverse"`` switches
    # to a REINFORCE-style estimator with per-token advantage
    # ``log p_teacher(x_t) - log p_student(x_t)`` for x_t sampled from the
    # student (mode-seeking, KL(P_student || P_teacher)).
    kl_direction: Literal["forward", "reverse"] = "forward"
    demo_template: str = OPD_DEMO_TEMPLATE
    chat_redo_message: str = OPD_REDO_MESSAGE
    system_prompt: str | None = None
    teacher_sync_every: int | None = None
    max_context_length: int = 32768

    # Renderer overrides applied after construction. Reasoning models (Qwen3,
    # Kimi K2, DeepSeek V3 thinking, ...) default to stripping ``<think>...
    # </think>`` blocks from non-last assistant messages so that HF-style
    # multi-turn prompts match the served chat template. SDFT teacher prompts
    # always end with the ``redo`` user message, which means *every* assistant
    # turn (including the golden demonstration) sits in history and would
    # otherwise lose its thinking. Default ``False`` here so the teacher can
    # actually attend to the golden chain-of-thought.
    strip_thinking_from_history: bool = False

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

    # Debug: free-form teacher rollout. When ``debug_teacher_rollout`` is True,
    # the trainer asks the teacher to generate from ``teacher_prompts_P`` (the
    # same prompt used for top-K teacher forcing) every
    # ``debug_teacher_rollout_every`` steps and logs the decoded text alongside
    # the student rollout. Costs ~1 extra teacher sample per problem per
    # eligible step, so keep the cadence sparse.
    debug_teacher_rollout: bool = True
    debug_teacher_rollout_every: int = 1