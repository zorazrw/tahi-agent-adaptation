"""DPO training on weight-format session JSON.

Usage::

    python -m weight.train.run_dpo \\
        --train-path data/weight.json \\
        --model-name Qwen/Qwen3-4B \\
        --renderer-name qwen3 \\
        --log-path ~/logs/dpo_run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any, cast

import tinker
import torch
import torch.nn.functional as F

from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.supervised.train import run_evals
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.utils import ml_log, trace
from tinker_cookbook.utils.format_colorized import format_colorized
from tinker_cookbook.utils.lr_scheduling import LRSchedule, compute_schedule_lr_multiplier
from tinker_cookbook.utils.misc_utils import iteration_dir

from tinker_cookbook.supervised.types import (
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
)

from .formatter import WeightDPODataBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reference logprob helpers (work around SkyRL's missing prompt_logprobs)
# ---------------------------------------------------------------------------

def _datum_fingerprint(datum: tinker.Datum) -> tuple:
    """Content-based stable key for caching reference logprobs across shuffles.

    SupervisedDatasetFromHFDataset re-instantiates Datum objects on every
    ``get_batch`` call, so ``id(datum)`` is not stable across epochs.
    Hashing by (model_input tokens, target_tokens) is stable and unique.
    """
    mi = tuple(datum.model_input.to_ints())
    tt = tuple(datum.loss_fn_inputs["target_tokens"].data)
    return (mi, tt)


def _forward_logprobs(
    training_client: tinker.TrainingClient,
    datums: list[tinker.Datum],
) -> list[torch.Tensor]:
    """Gradient-free forward pass returning per-target-token logprobs.

    Uses ``training_client.forward(data, "cross_entropy")`` which returns
    ``loss_fn_outputs[i]["logprobs"]`` — one logprob per target position.

    MUST be called before any ``optim_step`` when the caller needs base /
    reference logprobs, because ``forward()`` uses the current weights.
    """
    forward_result = training_client.forward(datums, "cross_entropy").result()
    out: list[torch.Tensor] = []
    for entry in forward_result.loss_fn_outputs:
        lp_data = entry["logprobs"]
        tensor = torch.tensor(lp_data.data, dtype=torch.float32)
        if lp_data.shape is not None:
            tensor = tensor.reshape(lp_data.shape)
        out.append(tensor.detach().cpu())
    return out


def precompute_ref_logprob_cache(
    training_client: tinker.TrainingClient,
    dataset: SupervisedDataset,
    num_epochs: int,
) -> dict[tuple, torch.Tensor]:
    """Pre-compute reference logprobs for every datum the training loop will see.

    Iterates all epochs × batches and caches per-target-token logprobs keyed
    by :func:`_datum_fingerprint`. Deduplicates across epochs so each unique
    datum is forwarded exactly once. Must be called before any ``optim_step``.

    SkyRL's vLLM backend does not yet support prompt_logprobs via the sampling
    API, so we route through ``training_client.forward()`` instead of the
    usual ``reference_client.compute_logprobs_async()`` path.
    """
    cache: dict[tuple, torch.Tensor] = {}
    n_batches = len(dataset)
    for epoch_idx in range(num_epochs):
        dataset.set_epoch(seed=epoch_idx)
        for batch_idx in range(n_batches):
            data = dataset.get_batch(batch_idx)
            missing = [d for d in data if _datum_fingerprint(d) not in cache]
            if not missing:
                continue
            logprobs = _forward_logprobs(training_client, missing)
            for datum, lp in zip(missing, logprobs, strict=True):
                cache[_datum_fingerprint(datum)] = lp
    logger.info(
        "Pre-computed reference logprobs for %d unique datums "
        "(across %d epochs x %d batches)",
        len(cache), num_epochs, n_batches,
    )
    return cache


# ---------------------------------------------------------------------------
# DPO loss
# ---------------------------------------------------------------------------

def compute_dpo_loss(
    chosen_logprobs: list[torch.Tensor],
    rejected_logprobs: list[torch.Tensor],
    chosen_ref_logprobs: list[torch.Tensor],
    rejected_ref_logprobs: list[torch.Tensor],
    dpo_beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """DPO loss from Rafailov et al. (2023).

    L = -log sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
    where log_ratio = log pi_policy - log pi_ref.
    """
    chosen_log_ratio = torch.stack(
        [lp - rlp for lp, rlp in zip(chosen_logprobs, chosen_ref_logprobs, strict=True)]
    )
    rejected_log_ratio = torch.stack(
        [lp - rlp for lp, rlp in zip(rejected_logprobs, rejected_ref_logprobs, strict=True)]
    )
    losses = -F.logsigmoid(dpo_beta * (chosen_log_ratio - rejected_log_ratio))
    loss = losses.mean()

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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _print_example(datum: tinker.Datum, tokenizer: Tokenizer, label: str = "") -> None:
    int_tokens = list(datum.model_input.to_ints())
    weights = datum.loss_fn_inputs["weights"].data
    logger.info(f"\n{label} Example:")
    logger.info(format_colorized(int_tokens, cast(list[float], weights), tokenizer))


def do_update(
    epoch_idx: int,
    batch_idx: int,
    n_batches: int,
    total_steps: int,
    dpo_beta: float,
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
    eval_every: int,
    infrequent_eval_every: int,
    training_client: tinker.TrainingClient,
    ref_logprob_cache: dict[tuple, torch.Tensor],
    evaluators: list[Any],
    infrequent_evaluators: list[Any],
    dataset: SupervisedDataset,
    ml_logger: ml_log.Logger,
    log_path: str,
    tokenizer: Tokenizer,
    rolling_mgr: checkpoint_utils.RollingCheckpointManager | None = None,
) -> None:
    step = epoch_idx * n_batches + batch_idx
    metrics: dict[str, int | float | str] = {"epoch": epoch_idx}

    with trace.trace_iteration(step=step) as window:
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
            rolling_mgr.maybe_save(
                step=step, loop_state={"epoch": epoch_idx, "batch": batch_idx}
            )

        learning_rate = learning_rate_base * compute_schedule_lr_multiplier(
            lr_schedule=lr_schedule, step=step, total_steps=total_steps
        )
        adam_params = tinker.AdamParams(
            learning_rate=learning_rate,
            beta1=adam_beta1,
            beta2=adam_beta2,
            eps=adam_eps,
        )

        if eval_every > 0 and step % eval_every == 0:
            with trace.scope_span_sync("evals"):
                eval_metrics = asyncio.run(run_evals(evaluators, training_client, step))
            metrics.update(eval_metrics)

        if infrequent_eval_every > 0 and step % infrequent_eval_every == 0:
            with trace.scope_span_sync("infrequent_evals"):
                eval_metrics = asyncio.run(
                    run_evals(infrequent_evaluators, training_client, step)
                )
            metrics.update(eval_metrics)

        with trace.scope_span_sync("get_batch"):
            data = dataset.get_batch(batch_idx)

        chosen_data = [datum for i, datum in enumerate(data) if i % 2 == 0]
        rejected_data = [datum for i, datum in enumerate(data) if i % 2 == 1]

        if step == 0:
            for i in range(min(10, len(chosen_data))):
                _print_example(chosen_data[i], tokenizer, "Chosen")
                _print_example(rejected_data[i], tokenizer, "Rejected")

        with trace.scope_span_sync("get_ref_logprobs"):
            all_ref_logprob_seqs = [ref_logprob_cache[_datum_fingerprint(d)] for d in data]
            chosen_ref_logprob_seqs = [all_ref_logprob_seqs[i] for i in range(0, len(data), 2)]
            rejected_ref_logprob_seqs = [all_ref_logprob_seqs[i] for i in range(1, len(data), 2)]

        def dpo_loss_fn(
            data: list[tinker.Datum], logprobs_list: list[torch.Tensor]
        ) -> tuple[torch.Tensor, dict[str, float]]:
            chosen_logprob_seqs = [logprobs_list[i] for i in range(0, len(data), 2)]
            rejected_logprob_seqs = [logprobs_list[i] for i in range(1, len(data), 2)]

            chosen_logprobs: list[torch.Tensor] = []
            chosen_ref_logprobs: list[torch.Tensor] = []
            rejected_logprobs: list[torch.Tensor] = []
            rejected_ref_logprobs: list[torch.Tensor] = []

            for i in range(len(chosen_data)):
                chosen_weights = torch.tensor(chosen_data[i].loss_fn_inputs["weights"].data)
                chosen_logprobs.append(
                    torch.dot(chosen_logprob_seqs[i].float(), chosen_weights.float())
                )
                chosen_ref_logprobs.append(
                    torch.dot(chosen_ref_logprob_seqs[i].float(), chosen_weights.float())
                )
                rejected_weights = torch.tensor(rejected_data[i].loss_fn_inputs["weights"].data)
                rejected_logprobs.append(
                    torch.dot(rejected_logprob_seqs[i].float(), rejected_weights.float())
                )
                rejected_ref_logprobs.append(
                    torch.dot(rejected_ref_logprob_seqs[i].float(), rejected_weights.float())
                )

            return compute_dpo_loss(
                chosen_logprobs=chosen_logprobs,
                rejected_logprobs=rejected_logprobs,
                chosen_ref_logprobs=chosen_ref_logprobs,
                rejected_ref_logprobs=rejected_ref_logprobs,
                dpo_beta=dpo_beta,
            )

        with trace.scope_span_sync("step"):
            backward_result = training_client.forward_backward_custom(data, dpo_loss_fn).result()
            training_client.optim_step(adam_params).result()

        metrics.update(
            num_pairs=len(chosen_data),
            num_tokens=sum(d.model_input.length for d in data),
            learning_rate=learning_rate,
            progress=step / total_steps,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO training (weight-format)")
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--test-path", default=None)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--renderer-name", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lr-schedule", default="linear")
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--ttl-seconds", type=int, default=604800)
    parser.add_argument("--rolling-save-every", type=int, default=0)
    parser.add_argument("--rolling-ttl-seconds", type=int, default=7200)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--infrequent-eval-every", type=int, default=100)
    parser.add_argument("--span-chart-every", type=int, default=0)
    parser.add_argument("--load-checkpoint-path", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args()

    log_path = str(Path(args.log_path).expanduser())

    # ------------------------------------------------------------------ #
    # Logging                                                              #
    # ------------------------------------------------------------------ #
    ml_logger = ml_log.setup_logging(
        log_dir=log_path,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        config=vars(args),
        do_configure_logging_module=True,
    )
    model_info.warn_if_renderer_not_recommended(args.model_name, args.renderer_name)

    # ------------------------------------------------------------------ #
    # Dataset                                                              #
    # ------------------------------------------------------------------ #
    dataset_builder = WeightDPODataBuilder(
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

    # ------------------------------------------------------------------ #
    # Resume                                                               #
    # ------------------------------------------------------------------ #
    resume_info = checkpoint_utils.get_last_checkpoint(log_path)
    start_epoch = resume_info.epoch or 0 if resume_info else 0
    start_batch = resume_info.batch if resume_info else 0

    # ------------------------------------------------------------------ #
    # Training client                                                      #
    # ------------------------------------------------------------------ #
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
            resume_info.state_path, user_metadata=user_metadata
        )
        logger.info(f"Resumed DPO training from {resume_info.state_path}")
    elif args.load_checkpoint_path:
        checkpoint_utils.check_renderer_name_for_checkpoint(
            service_client, args.load_checkpoint_path, args.renderer_name
        )
        training_client = service_client.create_training_client_from_state(
            args.load_checkpoint_path, user_metadata=user_metadata
        )
        logger.info(f"Loaded weights from {args.load_checkpoint_path}")
    else:
        training_client = service_client.create_lora_training_client(
            base_model=args.model_name,
            rank=args.lora_rank,
            user_metadata=user_metadata,
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

    # ------------------------------------------------------------------ #
    # Pre-compute reference logprobs (before any optim_step)              #
    # ------------------------------------------------------------------ #
    ref_logprob_cache = precompute_ref_logprob_cache(
        training_client, dataset, args.num_epochs,
    )

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #
    reached_max_steps = False
    for epoch_idx in range(start_epoch, args.num_epochs):
        logger.info("Starting epoch %d", epoch_idx)
        dataset.set_epoch(seed=epoch_idx)

        for batch_idx in range(start_batch if epoch_idx == start_epoch else 0, n_batches):
            step = epoch_idx * n_batches + batch_idx
            if args.max_steps is not None and step >= args.max_steps:
                reached_max_steps = True
                break
            do_update(
                epoch_idx=epoch_idx,
                batch_idx=batch_idx,
                n_batches=n_batches,
                total_steps=total_steps,
                dpo_beta=args.dpo_beta,
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
                eval_every=args.eval_every,
                infrequent_eval_every=args.infrequent_eval_every,
                training_client=training_client,
                ref_logprob_cache=ref_logprob_cache,
                evaluators=[],
                infrequent_evaluators=[],
                dataset=dataset,
                ml_logger=ml_logger,
                log_path=log_path,
                tokenizer=tokenizer,
                rolling_mgr=rolling_mgr,
            )
        if reached_max_steps:
            break

    # ------------------------------------------------------------------ #
    # Final checkpoint                                                     #
    # ------------------------------------------------------------------ #
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
    logger.info("DPO training completed successfully")


if __name__ == "__main__":
    main()
