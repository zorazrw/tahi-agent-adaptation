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
        --model-name Qwen/Qwen3.5-4B \\
        --renderer-name qwen3 \\
        --log-path ~/logs/opd_run
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, cast

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


async def _compute_teacher_logprobs(
    student_datums: list[tinker.Datum],
    teacher_datums: list[tinker.Datum],
    teacher_client: tinker.SamplingClient,
) -> list[torch.Tensor]:
    """Compute per-completion-position teacher logprobs.

    For each example, builds the teacher-forced sequence (teacher datum
    model_input + last target token) and extracts logprobs at the
    completion positions.

    Returns a list of tensors, one per datum, aligned with the student
    datum's weight mask (only positions where weight > 0).
    """
    teacher_full_seqs = []
    for td in teacher_datums:
        targets = td.loss_fn_inputs["target_tokens"].data
        if targets:
            seq = td.model_input.append_int(int(targets[-1]))
        else:
            seq = td.model_input
        teacher_full_seqs.append(seq)

    raw_lps = await asyncio.gather(
        *[teacher_client.compute_logprobs_async(seq) for seq in teacher_full_seqs]
    )

    result: list[torch.Tensor] = []
    for i, sd in enumerate(student_datums):
        weights = sd.loss_fn_inputs["weights"].data
        mask_indices = [j for j, w in enumerate(weights) if w > 0]
        n_completion = len(mask_indices)

        td = teacher_datums[i]
        td_weights = td.loss_fn_inputs["weights"].data
        td_mask_indices = [j for j, w in enumerate(td_weights) if w > 0]

        teacher_lp_raw = raw_lps[i]
        teacher_completion_lps = []
        for t in range(min(n_completion, len(td_mask_indices))):
            pos = td_mask_indices[t]
            lp = teacher_lp_raw[pos] if pos < len(teacher_lp_raw) else 0.0
            teacher_completion_lps.append(lp if lp is not None else 0.0)

        while len(teacher_completion_lps) < n_completion:
            teacher_completion_lps.append(0.0)

        result.append(torch.tensor(teacher_completion_lps[:n_completion], dtype=torch.float32))

    return result


def do_update(
    step: int,
    total_steps: int,
    config: Config,
    training_client: tinker.TrainingClient,
    teacher_client: tinker.SamplingClient,
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

            total_adv += advantage.sum().item()
            total_tokens += n

        batch_loss = total_loss / max(len(data), 1)

        loss_metrics = {
            "opd_loss": batch_loss.item(),
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

    teacher_client = service_client.create_sampling_client(base_model=config.model_name)
    tokenizer = get_tokenizer(config.model_name)

    n_batches = len(dataset)
    total_steps = n_batches * config.num_epochs
    if config.max_steps is not None:
        total_steps = min(total_steps, config.max_steps)

    logger.info(
        f"Offline OPD: {n_batches} batches x {config.num_epochs} epochs "
        f"= {n_batches * config.num_epochs} steps"
    )

    for epoch_idx in range(config.num_epochs):
        dataset.set_epoch(seed=epoch_idx)
        logger.info(f"Starting epoch {epoch_idx}")

        for batch_idx in range(start_batch if epoch_idx == 0 else 0, n_batches):
            step = epoch_idx * n_batches + batch_idx
            if config.max_steps is not None and step >= config.max_steps:
                break

            student_datums, teacher_datums = dataset.get_batch(batch_idx)

            teacher_lps = asyncio.run(
                _compute_teacher_logprobs(student_datums, teacher_datums, teacher_client)
            )

            if step == 0:
                for i in range(min(2, len(student_datums))):
                    int_tokens = list(student_datums[i].model_input.to_ints())
                    weights = student_datums[i].loss_fn_inputs["weights"].data
                    logger.info(f"\nExample {i}:")
                    logger.info(format_colorized(
                        int_tokens, cast(list[float], weights), tokenizer,
                    ))

            do_update(
                step=step,
                total_steps=total_steps,
                config=config,
                training_client=training_client,
                teacher_client=teacher_client,
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

    parser = argparse.ArgumentParser(description="Offline OPD training (weight-format)")
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--renderer-name", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--load-checkpoint-path", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model_name)
    renderer = renderers.get_renderer(args.renderer_name, tokenizer=tokenizer)

    dataset = OfflineOPDDataset.from_weight_json(
        path=args.train_path,
        renderer=renderer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    cfg = Config(
        log_path=args.log_path,
        model_name=args.model_name,
        renderer_name=args.renderer_name,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        load_checkpoint_path=args.load_checkpoint_path,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        max_steps=args.max_steps,
    )

    main(cfg, dataset)
