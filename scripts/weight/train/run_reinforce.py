"""REINFORCE training on weight-format session JSON.

Connects to the Tinker cloud API (no local server required).
Set ``TINKER_API_KEY`` in ``scripts/weight/.env`` before running.

Applies offline REINFORCE with a running reward baseline for variance
reduction. Rewards are pre-computed from verifier pass rates and human
intervention counts (see ``weight/data/reward.py``).

Loss: L = -mean_j( (R_j - baseline) * sum_k log pi_theta(a_{j,k} | ctx) )

Usage::

    python -m weight.train.run_reinforce \\
        --train-path data/weight.json \\
        --model-name Qwen/Qwen3-4B \\
        --renderer-name qwen3 \\
        --log-path ~/logs/reinforce_run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import cast

import tinker
import torch

from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.utils import ml_log, trace
from tinker_cookbook.utils.format_colorized import format_colorized
from tinker_cookbook.utils.lr_scheduling import LRSchedule, compute_schedule_lr_multiplier
from tinker_cookbook.utils.misc_utils import iteration_dir

from .formatter import WeightReinforceDataBuilder

logger = logging.getLogger(__name__)
BASELINE_STATE_FILENAME = "reinforce_baseline_state.json"


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
    rolling_mgr: checkpoint_utils.RollingCheckpointManager | None = None,
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
            rolling_mgr.maybe_save(step=step, loop_state={"epoch": epoch_idx, "batch": batch_idx})

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

        if step == 0:
            for i in range(min(3, len(data))):
                int_tokens = list(data[i].model_input.to_ints())
                weights = data[i].loss_fn_inputs["weights"].data
                logger.info(f"\nExample {i} (reward={rewards[i]:.3f}):")
                logger.info(format_colorized(int_tokens, cast(list[float], weights), tokenizer))

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
    args = parser.parse_args()

    log_path = str(Path(args.log_path).expanduser())

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
    rolling_mgr = checkpoint_utils.RollingCheckpointManager(
        training_client=training_client,
        service_client=service_client,
        log_path=log_path,
        rolling_save_every=args.rolling_save_every,
        save_every=args.save_every,
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
