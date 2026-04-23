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
    # Number of full passes over the dataset (CLI: ``--num-epochs`` or ``--epochs``).
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


def do_update(
    epoch_idx: int,
    batch_idx: int,
    n_batches: int,
    total_steps: int,
    config: Config,
    training_client: tinker.TrainingClient,
    dataset,  # ReinforceDataset
    baseline: float,
    ml_logger: ml_log.Logger,
    log_path: str,
    tokenizer: Tokenizer,
    rolling_mgr: checkpoint_utils.RollingCheckpointManager | None = None,
) -> tuple[dict[str, int | float | str], float]:
    """Perform a single REINFORCE training step.

    Returns ``(metrics, new_baseline)`` where *new_baseline* is the mean
    reward of the current batch (used as the baseline for the next step).
    """
    step = epoch_idx * n_batches + batch_idx
    metrics: dict[str, int | float | str] = {"epoch": epoch_idx}

    with trace.trace_iteration(step=step) as window:
        # Mirror checkpoint resume semantics by storing the baseline for the
        # same loop position that checkpoint_utils records.
        _save_baseline_state(log_path, epoch_idx, batch_idx, baseline)

        # Periodic checkpoint
        if config.save_every > 0 and step % config.save_every == 0 and step > 0:
            with trace.scope_span_sync("save_checkpoint"):
                save_result = checkpoint_utils.save_checkpoint(
                    training_client=training_client,
                    name=f"{step:06d}",
                    log_path=log_path,
                    kind="both",
                    loop_state={"epoch": epoch_idx, "batch": batch_idx},
                    ttl_seconds=config.ttl_seconds,
                )
            if "state_path" in save_result:
                metrics["state_path"] = save_result["state_path"]

        if rolling_mgr is not None:
            rolling_mgr.maybe_save(step=step, loop_state={"epoch": epoch_idx, "batch": batch_idx})

        learning_rate = config.learning_rate * compute_schedule_lr_multiplier(
            lr_schedule=config.lr_schedule, step=step, total_steps=total_steps
        )
        adam_params = tinker.AdamParams(
            learning_rate=learning_rate,
            beta1=config.adam_beta1,
            beta2=config.adam_beta2,
            eps=config.adam_eps,
        )

        # Get batch of datums + associated rewards
        with trace.scope_span_sync("get_batch"):
            data = dataset.get_batch(batch_idx)
            rewards = dataset.get_batch_rewards(batch_idx)

        # Print a few examples on the first step
        if step == 0:
            for i in range(min(3, len(data))):
                print_example(data[i], tokenizer, f"Example {i} (reward={rewards[i]:.3f})")

        # Compute per-trajectory advantages: A_j = R_j - baseline
        advantages = [r - baseline for r in rewards]

        # Forward-backward with REINFORCE loss
        loss_fn = make_reinforce_loss_fn(advantages)

        with trace.scope_span_sync("step"):
            backward_result = training_client.forward_backward_custom(data, loss_fn).result()
            reinforce_metrics = backward_result.metrics
            training_client.optim_step(adam_params).result()

        # Update baseline to the mean reward of this batch
        new_baseline = sum(rewards) / len(rewards) if rewards else baseline

        metrics.update(
            num_trajectories=len(data),
            num_tokens=sum(datum.model_input.length for datum in data),
            learning_rate=learning_rate,
            progress=step / total_steps,
            baseline=new_baseline,
            mean_reward=sum(rewards) / len(rewards) if rewards else 0.0,
            **reinforce_metrics,
        )

    # Timing and logging
    metrics.update(window.get_timing_metrics())
    window.write_spans_jsonl(Path(log_path) / "timing_spans.jsonl", step=step)
    if config.span_chart_every > 0 and step % config.span_chart_every == 0:
        iter_dir = iteration_dir(log_path, step)
        if iter_dir is not None:
            iter_dir.mkdir(parents=True, exist_ok=True)
            trace.save_gantt_chart_html(window, step, iter_dir / "timing_gantt.html")
    ml_logger.log_metrics(metrics=metrics, step=step)

    return metrics, new_baseline


def main(config: Config):
    """Run the complete offline REINFORCE training loop.

    Sets up logging, creates the training client, builds the dataset, and
    iterates through epochs and batches.  The running baseline is initialized
    to ``config.initial_baseline`` and updated to the batch mean reward after
    each step.
    """
    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    start_epoch = resume_info.epoch or 0 if resume_info else 0
    start_batch = resume_info.batch or 0 if resume_info else 0

    # Logging setup
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
        do_configure_logging_module=True,
    )
    if config.enable_trace:
        trace_events_path = str(Path(config.log_path) / "trace_events.jsonl")
        logger.info(f"Tracing enabled. Events saved to {trace_events_path}")
        trace.trace_init(output_file=trace_events_path)

    user_metadata: dict[str, str] = {}
    if wandb_link := ml_logger.get_logger_url():
        user_metadata["wandb_link"] = wandb_link
    checkpoint_utils.add_renderer_name_to_user_metadata(user_metadata, config.renderer_name)
    model_info.warn_if_renderer_not_recommended(config.model_name, config.renderer_name)

    training_client = create_training_client(config, resume_info, user_metadata)
    service_client = tinker.ServiceClient(base_url=config.base_url)
    rolling_mgr = checkpoint_utils.RollingCheckpointManager(
        training_client=training_client,
        service_client=service_client,
        log_path=config.log_path,
        rolling_save_every=config.rolling_save_every,
        save_every=config.save_every,
        rolling_ttl_seconds=config.rolling_ttl_seconds,
    )
    tokenizer = get_tokenizer(config.model_name)

    # Build dataset
    dataset, _ = config.dataset_builder()
    n_batches = len(dataset)
    total_steps = n_batches * config.num_epochs
    if config.max_steps is not None:
        total_steps = min(total_steps, config.max_steps)

    logger.info(
        f"Training for {n_batches} batches x {config.num_epochs} epochs = "
        f"{n_batches * config.num_epochs} steps"
    )

    # Initialize running baseline
    baseline = _load_baseline_state(config.log_path, resume_info)
    if baseline is None:
        baseline = config.initial_baseline

    # Training loop
    reached_max_steps = False
    for epoch_idx in range(start_epoch, config.num_epochs):
        logger.info(f"Starting epoch {epoch_idx}")
        dataset.set_epoch(seed=epoch_idx)

        for batch_idx in range(start_batch if epoch_idx == start_epoch else 0, n_batches):
            step = epoch_idx * n_batches + batch_idx
            if config.max_steps is not None and step >= config.max_steps:
                reached_max_steps = True
                break
            _, baseline = do_update(
                epoch_idx=epoch_idx,
                batch_idx=batch_idx,
                n_batches=n_batches,
                total_steps=total_steps,
                config=config,
                training_client=training_client,
                dataset=dataset,
                baseline=baseline,
                ml_logger=ml_logger,
                log_path=config.log_path,
                tokenizer=tokenizer,
                rolling_mgr=rolling_mgr,
            )
        if reached_max_steps:
            break

    # Final checkpoint
    did_train = start_epoch < config.num_epochs and (
        config.max_steps is None or start_epoch * n_batches + start_batch < config.max_steps
    )
    if did_train:
        checkpoint_utils.save_checkpoint(
            training_client=training_client,
            name="final",
            log_path=config.log_path,
            kind="both",
            loop_state={"epoch": config.num_epochs, "batch": 0},
            ttl_seconds=None,
        )
    else:
        logger.info("Training was already complete; nothing to do")
    rolling_mgr.finalize()

    ml_logger.close()
    logger.info("REINFORCE training completed successfully")


def print_example(datum: tinker.Datum, tokenizer: Tokenizer, label: str = ""):
    """Print a colorized, human-readable training example."""
    int_tokens = list(datum.model_input.to_ints())
    weights = datum.loss_fn_inputs["weights"].data
    logger.info(f"\n{label}:")
    logger.info(format_colorized(int_tokens, cast(list[float], weights), tokenizer))


if __name__ == "__main__":
    import argparse

    from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

    from tinker_formatter import ReinforceDataBuilder

    parser = argparse.ArgumentParser(
        description="Offline REINFORCE training on agent trajectories"
    )
    parser.add_argument("--train-path", required=True, help="Path to reinforce training JSON")
    parser.add_argument("--test-path", default=None, help="Path to reinforce test JSON")
    parser.add_argument(
        "--model-name",
        required=True,
        help="Base model (e.g. meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--renderer-name",
        required=True,
        help="Renderer matching model family (e.g. llama3, qwen3)",
    )
    parser.add_argument("--log-path", required=True, help="Directory for checkpoints and logs")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--reward-alpha", type=float, default=0.05)
    parser.add_argument("--initial-baseline", type=float, default=0.0)
    parser.add_argument(
        "--num-epochs",
        "--epochs",
        type=int,
        default=1,
        dest="num_epochs",
        help="Number of full passes over the training set",
    )
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--load-checkpoint-path", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    dataset_builder = ReinforceDataBuilder(
        train_path=args.train_path,
        test_path=args.test_path,
        reward_alpha=args.reward_alpha,
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=args.model_name,
            renderer_name=args.renderer_name,
            max_length=args.max_length,
            batch_size=args.batch_size,
        ),
    )

    config = Config(
        log_path=args.log_path,
        model_name=args.model_name,
        dataset_builder=dataset_builder,
        renderer_name=args.renderer_name,
        learning_rate=args.learning_rate,
        reward_alpha=args.reward_alpha,
        initial_baseline=args.initial_baseline,
        num_epochs=args.num_epochs,
        lora_rank=args.lora_rank,
        load_checkpoint_path=args.load_checkpoint_path,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        max_steps=args.max_steps,
    )

    main(config)
