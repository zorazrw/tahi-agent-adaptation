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
    get_teacher_lps: Callable[
        [list[tinker.Datum], list[tinker.Datum]], list[torch.Tensor]
    ]
    if not config.use_skyrl:
        # Tinker cloud path: a frozen base-model SamplingClient acts as the
        # teacher.  Per-batch compute_logprobs_async is cheap and avoids
        # materialising a full cache up front.
        teacher_client = service_client.create_sampling_client(
            base_model=config.model_name,
        )
        logger.info(
            "Tinker mode: created frozen teacher SamplingClient for %s",
            config.model_name,
        )

        def get_teacher_lps(
            student_datums: list[tinker.Datum],
            teacher_datums: list[tinker.Datum],
        ) -> list[torch.Tensor]:
            return _live_teacher_logprobs(teacher_client, student_datums, teacher_datums)
    else:
        # SkyRL path: pre-compute teacher logprobs for every (student, teacher)
        # pair via training_client.forward() at the initial weights.
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
            teacher_lps = get_teacher_lps(student_datums, teacher_datums)

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
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--load-checkpoint-path", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--use-skyrl", action="store_true",
        help=(
            "Use the SkyRL-compatible path that pre-computes teacher "
            "logprobs via training_client.forward(). Default (off) uses the "
            "Tinker cloud path: query a frozen teacher SamplingClient per "
            "batch via compute_logprobs_async."
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
        base_url=args.base_url,
        use_skyrl=args.use_skyrl,
    )

    main(cfg, dataset)
