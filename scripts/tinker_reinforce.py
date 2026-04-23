"""
Offline REINFORCE training with running baseline.

Applies the REINFORCE policy gradient algorithm to pre-collected agent
trajectories with reward signals, using a running baseline (mean reward from
the previous iteration) for variance reduction.

Loss: L = -mean_j((R_j - b) * sum_k log pi_theta(a_{j,k} | q_j, a_{j,<k}))

Adapted from the DPO training script (tinker_dpo.py) for the training loop
structure and Tinker client management.
"""

import json
import logging
from pathlib import Path
from typing import cast
import datetime
import chz
import tinker
import torch

from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.supervised.types import ChatDatasetBuilder
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.utils import ml_log, trace
from tinker_cookbook.utils.format_colorized import format_colorized
from tinker_cookbook.utils.lr_scheduling import LRSchedule, compute_schedule_lr_multiplier
from tinker_cookbook.utils.misc_utils import iteration_dir

logger = logging.getLogger(__name__)
BASELINE_STATE_FILENAME = "reinforce_baseline_state.json"


@chz.chz
class Config:
    """Configuration for offline REINFORCE training.

    Attributes:
        log_path: Directory for saving checkpoints, metrics, and logs.
        model_name: Name of the base model to fine-tune.
        dataset_builder: Builder that produces train (and optionally test)
            datasets of tokenized datums with reward metadata.
        reward_alpha: Penalty weight for human correction count in the reward
            formula ``reward = verifier - alpha * human``.
        initial_baseline: Starting value for the running reward baseline.
            Updated to the batch mean reward after each training step.
    """

    # Required
    log_path: str = chz.field(munger=lambda _, s: str(Path(s).expanduser()))
    model_name: str
    dataset_builder: ChatDatasetBuilder

    # Checkpoint / model
    load_checkpoint_path: str | None = None
    renderer_name: str | None = None
    lora_rank: int = 32

    # Training
    learning_rate: float = 1e-5
    lr_schedule: LRSchedule = "linear"
    num_epochs: int = 1

    # REINFORCE-specific
    reward_alpha: float = 0.05
    initial_baseline: float = 0.0

    # Infrastructure
    num_replicas: int = 8
    base_url: str | None = None

    # Checkpointing (0 = disabled for *_every fields)
    save_every: int = 20
    ttl_seconds: int | None = 604800  # 7 days
    rolling_save_every: int = 0
    rolling_ttl_seconds: int = 7200  # 2 hours

    # Adam optimizer
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # Logging
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Profiling
    enable_trace: bool = False
    span_chart_every: int = 0

    # Hard cap on training steps; None trains for full num_epochs * n_batches.
    max_steps: int | None = None


def create_training_client(
    config: Config,
    resume_info: checkpoint_utils.CheckpointRecord | None = None,
    user_metadata: dict[str, str] | None = None,
) -> tinker.TrainingClient:
    """Create and configure the training client for REINFORCE.

    Unlike DPO, no reference/sampling client is needed -- the running baseline
    provides sufficient variance reduction without a KL penalty.
    """
    service_client = tinker.ServiceClient(base_url=config.base_url)

    if resume_info:
        assert resume_info.state_path is not None
        checkpoint_utils.check_renderer_name_for_checkpoint(
            service_client, resume_info.state_path, config.renderer_name
        )
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_info.state_path, user_metadata=user_metadata
        )
        logger.info(f"Resumed REINFORCE training from {resume_info.state_path}")
    elif config.load_checkpoint_path:
        checkpoint_utils.check_renderer_name_for_checkpoint(
            service_client, config.load_checkpoint_path, config.renderer_name
        )
        training_client = service_client.create_training_client_from_state(
            config.load_checkpoint_path, user_metadata=user_metadata
        )
        logger.info(f"Loaded weights from {config.load_checkpoint_path}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank, user_metadata=user_metadata
        )
    return training_client


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
    log_path: str, resume_info: checkpoint_utils.CheckpointRecord | None
) -> float | None:
    if resume_info is None:
        return None

    path = _baseline_state_path(log_path)
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Failed to read REINFORCE baseline state from {path}: {exc}")
        return None

    resume_epoch = resume_info.epoch or 0
    resume_batch = resume_info.batch or 0
    if payload.get("epoch") != resume_epoch or payload.get("batch") != resume_batch:
        logger.warning(
            "Ignoring saved REINFORCE baseline because it does not match the latest checkpoint "
            f"(checkpoint=({resume_epoch}, {resume_batch}), "
            f"baseline_state=({payload.get('epoch')}, {payload.get('batch')}))"
        )
        return None

    baseline = payload.get("baseline")
    if not isinstance(baseline, (int, float)):
        logger.warning(f"Ignoring invalid saved REINFORCE baseline: {baseline!r}")
        return None

    return float(baseline)


def make_reinforce_loss_fn(
    advantages: list[float],
):
    """Create a REINFORCE loss closure capturing pre-computed advantages.

    The returned function is compatible with ``forward_backward_custom``::

        loss_fn(data, logprobs_list) -> (loss, metrics)

    For each trajectory j the weighted sequence log-probability is
    ``sum_t(logprob_t * weight_t)`` where weight_t masks out prompt tokens.
    The loss is ``-mean_j(advantage_j * seq_logprob_j)``.
    """

    def reinforce_loss_fn(
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

    return reinforce_loss_fn


def print_example(datum: tinker.Datum, tokenizer: Tokenizer, label: str = ""):
    """Print a colorized, human-readable training example."""
    int_tokens = list(datum.model_input.to_ints())
    weights = datum.loss_fn_inputs["weights"].data
    logger.info(f"\n{label}:")
    logger.info(format_colorized(int_tokens, cast(list[float], weights), tokenizer))
