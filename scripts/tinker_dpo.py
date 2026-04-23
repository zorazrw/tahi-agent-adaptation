"""
Direct Preference Optimization (DPO) training
Adpated from tinker-cookbook:
https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/preference/train_dpo.py#L91
"""

import asyncio
import logging
import datetime
from pathlib import Path
from typing import cast

import chz
import tinker
import torch
import torch.nn.functional as F

from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.eval.evaluators import Evaluator, EvaluatorBuilder
from tinker_cookbook.supervised.train import run_evals
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.utils import ml_log, trace
from tinker_cookbook.utils.format_colorized import format_colorized
from tinker_cookbook.utils.lr_scheduling import LRSchedule, compute_schedule_lr_multiplier
from tinker_cookbook.utils.misc_utils import iteration_dir

logger = logging.getLogger(__name__)


@chz.chz
class Config:
    """Configuration for Direct Preference Optimization (DPO) training.

    This is a ``chz`` dataclass that holds all hyperparameters, infrastructure
    settings, and checkpointing options for a DPO training run.

    Attributes:
        log_path (str): Directory for saving checkpoints, metrics, and logs.
        model_name (str): Name of the base model to fine-tune.
        dataset_builder (ChatDatasetBuilder): Builder that produces train (and
            optionally test) datasets of chosen/rejected pairs.
        load_checkpoint_path (str | None): Path to a checkpoint to initialize
            weights from.  ``None`` starts from the base model.
        renderer_name (str | None): Renderer to use for tokenization.  Must
            match the model family (e.g. ``"llama3"``, ``"qwen3"``).
        learning_rate (float): Peak learning rate.  Recommended starting point
            for DPO is ~1e-5.
        lr_schedule (LRSchedule): Learning-rate schedule type (e.g. ``"linear"``).
        num_epochs (int): Number of passes over the dataset.
        dpo_beta (float): KL-penalty coefficient in the DPO loss.  Higher
            values penalize deviations from the reference model more strongly.
        lora_rank (int): LoRA adapter rank.
        num_replicas (int): Number of GPU replicas to use.
        base_url (str | None): Override for the Tinker service URL.
        evaluator_builders (list[EvaluatorBuilder]): Evaluators run every
            ``eval_every`` steps.
        infrequent_evaluator_builders (list[EvaluatorBuilder]): Evaluators run
            every ``infrequent_eval_every`` steps.
        save_every (int): Save a checkpoint every N steps (0 = disabled).
        eval_every (int): Run evaluators every N steps (0 = disabled).
        infrequent_eval_every (int): Run infrequent evaluators every N steps
            (0 = disabled).
        ttl_seconds (int | None): Time-to-live for intermediate checkpoints.
            ``None`` keeps them indefinitely.
        adam_beta1 (float): Adam optimizer beta1.
        adam_beta2 (float): Adam optimizer beta2.
        adam_eps (float): Adam optimizer epsilon.
        wandb_project (str | None): Weights & Biases project name.
        wandb_name (str | None): Weights & Biases run name.
        enable_trace (bool): Whether to record timing traces.
        span_chart_every (int): Save a Gantt timing chart every N steps
            (0 = disabled).
        reference_model_name (str | None): Explicit reference model.  When
            ``None``, the initial training weights are used as the reference.
        max_steps (int | None): Hard cap on training steps.  ``None`` trains
            for the full ``num_epochs * n_batches``.

    Example::

        config = Config(
            log_path="~/logs/dpo_run",
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            dataset_builder=my_dpo_dataset_builder,
            dpo_beta=0.1,
            learning_rate=1e-5,
        )
        main(config)
    """

    # Required parameters
    log_path: str = chz.field(munger=lambda _, s: str(Path(s).expanduser()))
    model_name: str
    dataset_builder: ChatDatasetBuilder
    load_checkpoint_path: str | None = None
    renderer_name: str | None = None
    # dataset_builder optionally returns an evaluator (test set)

    # Training parameters
    learning_rate: float = 1e-5
    lr_schedule: LRSchedule = "linear"
    num_epochs: int = 1
    dpo_beta: float = 0.1

    # Model parameters
    lora_rank: int = 32

    # Infrastructure parameters
    num_replicas: int = 8
    base_url: str | None = None

    # Checkpointing and evaluation (0 = disabled for *_every fields)
    evaluator_builders: list[EvaluatorBuilder] = chz.field(default_factory=list)
    infrequent_evaluator_builders: list[EvaluatorBuilder] = chz.field(default_factory=list)
    save_every: int = 20
    eval_every: int = 10
    infrequent_eval_every: int = 100
    ttl_seconds: int | None = 604800  # 7 days
    # Rolling checkpoint cadence (0 = disabled). Saves training state for resume
    # but skips the sampler-weight export, making it cheaper than periodic checkpoints.
    rolling_save_every: int = 0
    # TTL for rolling checkpoints; short to auto-clean if explicit deletion fails.
    rolling_ttl_seconds: int = 7200  # 2 hours

    # Adam optimizer parameters
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # Logging parameters
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Profiling
    enable_trace: bool = False
    span_chart_every: int = 0

    # DPO-specific parameters
    reference_model_name: str | None = None

    # Maximum number of training steps. If None, train for num_epochs * n_batches.
    max_steps: int | None = None


def compute_dpo_loss(
    chosen_logprobs: list[torch.Tensor],
    rejected_logprobs: list[torch.Tensor],
    chosen_ref_logprobs: list[torch.Tensor],
    rejected_ref_logprobs: list[torch.Tensor],
    dpo_beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the DPO loss and associated training metrics.

    Implements the loss from *Direct Preference Optimization* (Rafailov et al., 2023):
    ``L = -log sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))``.

    Args:
        chosen_logprobs (list[torch.Tensor]): Per-example sum of weighted
            log-probabilities under the policy for chosen responses.
        rejected_logprobs (list[torch.Tensor]): Per-example sum of weighted
            log-probabilities under the policy for rejected responses.
        chosen_ref_logprobs (list[torch.Tensor]): Per-example sum of weighted
            log-probabilities under the reference model for chosen responses.
        rejected_ref_logprobs (list[torch.Tensor]): Per-example sum of weighted
            log-probabilities under the reference model for rejected responses.
        dpo_beta (float): KL-penalty coefficient.  Higher values make the
            loss more sensitive to deviations from the reference model.

    Returns:
        tuple[torch.Tensor, dict[str, float]]: A pair of (scalar loss,
            metrics dict).  The metrics dict contains ``dpo_loss``,
            ``accuracy`` (fraction where chosen is preferred), ``margin``,
            ``chosen_reward``, and ``rejected_reward``.
    """
    # Compute log ratios
    chosen_log_ratio = torch.stack(
        [lp - rlp for lp, rlp in zip(chosen_logprobs, chosen_ref_logprobs, strict=True)]
    )
    rejected_log_ratio = torch.stack(
        [lp - rlp for lp, rlp in zip(rejected_logprobs, rejected_ref_logprobs, strict=True)]
    )

    # Compute DPO loss
    losses = -F.logsigmoid(dpo_beta * (chosen_log_ratio - rejected_log_ratio))
    loss = losses.mean()

    # Compute metrics
    accuracy = (chosen_log_ratio > rejected_log_ratio).float().mean().item()
    chosen_rewards = dpo_beta * chosen_log_ratio
    rejected_rewards = dpo_beta * rejected_log_ratio
    margin = (chosen_rewards - rejected_rewards).mean().item()

    metrics = {
        "dpo_loss": loss.item(),
        "accuracy": accuracy,
        "margin": margin,
        "chosen_reward": chosen_rewards.mean().item(),
        "rejected_reward": rejected_rewards.mean().item(),
    }

    return loss, metrics


def print_example(datum: tinker.Datum, tokenizer: Tokenizer, label: str = ""):
    """Print a colorized, human-readable example from the dataset.

    Decodes the token IDs and displays them with color-coding based on the
    per-token loss weights so that trained-on tokens are visually distinct.

    Args:
        datum (tinker.Datum): A single training datum containing
            ``model_input`` and ``loss_fn_inputs["weights"]``.
        tokenizer (Tokenizer): Tokenizer for decoding token IDs to text.
        label (str): Optional prefix label (e.g. ``"Chosen"`` or
            ``"Rejected"``) printed before the example.
    """
    int_tokens = list(datum.model_input.to_ints())
    weights = datum.loss_fn_inputs["weights"].data
    logger.info(f"\n{label} Example:")
    logger.info(format_colorized(int_tokens, cast(list[float], weights), tokenizer))