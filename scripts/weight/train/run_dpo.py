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
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Callable, cast

import tinker
import torch
import torch.nn.functional as F
import chz

from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.eval.evaluators import EvaluatorBuilder
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.train import run_evals
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.utils import ml_log, trace
from tinker_cookbook.utils.format_colorized import format_colorized
from tinker_cookbook.utils.lr_scheduling import LRSchedule, compute_schedule_lr_multiplier
from tinker_cookbook.utils.misc_utils import iteration_dir

from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
)

from .formatter import WeightDPODataBuilder, _hydrate_tool_calls, _load_sessions
from .run_opd import (
    _parse_valid_artifact_write_message,
    _summarize_sample,
    _with_artifact_only_instruction,
)

try:  # Supports both `python -m weight...` from scripts/ and `python -m scripts.weight...`.
    from weight.data.extract import (  # type: ignore[import-not-found]
        extract_dpo_accepted_artifacts,
        extract_dpo_pairs,
    )
except ModuleNotFoundError:  # pragma: no cover - depends on invocation cwd
    from ..data.extract import extract_dpo_accepted_artifacts, extract_dpo_pairs

logger = logging.getLogger(__name__)


@chz.chz
class Config:
    """Configuration shared by the offline CLI and online server DPO trainer."""

    log_path: str = chz.field(munger=lambda _, s: str(Path(s).expanduser()))
    model_name: str
    dataset_builder: ChatDatasetBuilder | None = None
    load_checkpoint_path: str | None = None
    renderer_name: str | None = None

    learning_rate: float = 1e-5
    lr_schedule: LRSchedule = "linear"
    num_epochs: int = 1
    dpo_beta: float = 0.1
    rpo_alpha: float = 0.0
    use_ipo: bool = False

    # Online artifact DPO. When enabled, sampled current-policy artifact
    # rollouts become rejected responses, while accepted artifact writes from
    # the weight JSON become chosen responses.
    online_rollout: bool = False
    rollout_max_tokens: int = 4096
    rollout_temperature: float = 1.0
    rollout_attempts: int = 1
    log_rollout_samples: bool = True
    rollout_sample_log_chars: int = 4000
    artifact_only_rollout_instruction: bool = False

    lora_rank: int = 32
    num_replicas: int = 8
    base_url: str | None = None

    evaluator_builders: list[EvaluatorBuilder] = chz.field(default_factory=list)
    infrequent_evaluator_builders: list[EvaluatorBuilder] = chz.field(default_factory=list)
    save_every: int = 20
    eval_every: int = 10
    infrequent_eval_every: int = 100
    ttl_seconds: int | None = 604800
    rolling_save_every: int = 0
    rolling_ttl_seconds: int = 7200

    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    wandb_project: str | None = None
    wandb_name: str | None = None
    enable_trace: bool = False
    span_chart_every: int = 0
    max_steps: int | None = None


# ---------------------------------------------------------------------------
# .env loader (shared with run_opd / run_reinforce)
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


# ---------------------------------------------------------------------------
# Reference logprob helpers
# ---------------------------------------------------------------------------
#
# Two paths, selected by ``--use-skyrl``:
#
# 1. SkyRL path (``--use-skyrl``): SkyRL's vLLM backend does not yet expose
#    ``prompt_logprobs`` via the sampling API, so ``compute_logprobs_async``
#    silently returns None.  We pre-compute every datum's reference logprobs
#    once via ``training_client.forward()`` BEFORE any ``optim_step`` and
#    cache them by content fingerprint.
#
# 2. Tinker path (default): on the real Tinker cloud API we snapshot
#    the initial weights into a frozen ``SamplingClient`` once, then call
#    ``reference_client.compute_logprobs_async`` per batch.  This is the
#    canonical DPO recipe.
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


def _live_ref_logprobs(
    reference_client: tinker.SamplingClient,
    data: list[tinker.Datum],
) -> list[torch.Tensor]:
    """Tinker path: fetch reference logprobs for one batch via the sampling client.

    Builds the full sequence (model_input + last target token) for each datum
    and calls ``compute_logprobs_async`` in parallel.  Returns one tensor per
    datum aligned with ``target_tokens`` (length = model_input.length).
    """
    full_sequences = []
    for datum in data:
        targets = datum.loss_fn_inputs["target_tokens"].data
        if targets:
            full_seq = datum.model_input.append_int(int(targets[-1]))
        else:
            full_seq = datum.model_input
        full_sequences.append(full_seq)

    async def _gather():
        return await asyncio.gather(
            *[reference_client.compute_logprobs_async(seq) for seq in full_sequences]
        )

    raw = asyncio.run(_gather())
    # raw[i][0] is None (no prior context); raw[i][1:] are the per-target logprobs.
    return [
        torch.tensor(
            [lp if lp is not None else 0.0 for lp in r[1:]], dtype=torch.float32,
        )
        for r in raw
    ]


# ---------------------------------------------------------------------------
# DPO loss
# ---------------------------------------------------------------------------

def compute_dpo_loss(
    chosen_logprobs: list[torch.Tensor],
    rejected_logprobs: list[torch.Tensor],
    chosen_ref_logprobs: list[torch.Tensor],
    rejected_ref_logprobs: list[torch.Tensor],
    dpo_beta: float,
    rpo_alpha: float = 0.0,
    use_ipo: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Preference loss (DPO / IPO), optionally with RPO SFT anchor.

    **Vanilla DPO** (Rafailov et al. 2023)::

        L = -log sigmoid(beta * h)      h = log_ratio_chosen - log_ratio_rejected

    where ``log_ratio = log pi_policy - log pi_ref``.

    **IPO** (Azar et al. 2023, ``use_ipo=True``)::

        L = (h - 1 / (2 * beta)) ** 2

    Replaces logsigmoid with a squared loss.  The gradient is proportional to
    ``(h - 1/(2β))``, so it decays to zero when the gap reaches the target
    value ``1/(2β)``.  This prevents reward hacking (the gap cannot grow
    to ±∞) while still driving the model to prefer chosen.

    **RPO anchor** (Pang et al. 2024, ``rpo_alpha > 0``)::

        L = base_loss + rpo_alpha * (-mean(chosen_logprob))

    Applicable on top of either DPO or IPO.  The SFT term anchors
    ``log pi_policy(chosen)`` absolutely upward so the optimizer cannot
    satisfy the pairwise objective by driving both completions down
    (likelihood displacement).  Typical values: 0.1 (mild) – 0.5 (strong).

    The two mechanisms are orthogonal:
    - IPO controls the *shape* of the reward gap (bounded vs unbounded).
    - RPO controls the *absolute* position of chosen logp.
    """
    chosen_log_ratio = torch.stack(
        [lp - rlp for lp, rlp in zip(chosen_logprobs, chosen_ref_logprobs, strict=True)]
    )
    rejected_log_ratio = torch.stack(
        [lp - rlp for lp, rlp in zip(rejected_logprobs, rejected_ref_logprobs, strict=True)]
    )

    h = chosen_log_ratio - rejected_log_ratio   # gap, shape (batch,)

    if use_ipo:
        # IPO: squared loss targeting h = 1/(2β)
        # Gradient ∝ (h - target), naturally zeros at optimum.
        target = 1.0 / (2.0 * dpo_beta)
        losses = (h - target) ** 2
    else:
        # DPO: -log σ(β·h)
        losses = -F.logsigmoid(dpo_beta * h)

    base_loss = losses.mean()

    accuracy = (chosen_log_ratio > rejected_log_ratio).float().mean().item()
    chosen_rewards = dpo_beta * chosen_log_ratio
    rejected_rewards = dpo_beta * rejected_log_ratio
    margin = (chosen_rewards - rejected_rewards).mean().item()

    metrics = {
        "dpo_loss": base_loss.item(),   # base pairwise loss (DPO or IPO term)
        "accuracy": accuracy,
        "margin": margin,
        "chosen_reward": chosen_rewards.mean().item(),
        "rejected_reward": rejected_rewards.mean().item(),
    }

    if use_ipo:
        target_val = 1.0 / (2.0 * dpo_beta)
        metrics["ipo_target"] = target_val
        metrics["ipo_gap_error"] = (h.mean() - target_val).item()

    if rpo_alpha > 0.0:
        # SFT anchor on chosen: -mean log pi_policy(chosen).
        # Grows positive when policy assigns low prob to chosen.
        sft_anchor = -torch.stack(chosen_logprobs).mean()
        loss = base_loss + rpo_alpha * sft_anchor
        metrics["sft_anchor"] = sft_anchor.item()
        metrics["rpo_alpha"] = rpo_alpha
    else:
        loss = base_loss

    metrics["loss"] = loss.item()
    return loss, metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _print_example(datum: tinker.Datum, tokenizer: Tokenizer, label: str = "") -> None:
    int_tokens = list(datum.model_input.to_ints())
    weights = datum.loss_fn_inputs["weights"].data
    logger.info(f"\n{label} Example:")
    logger.info(format_colorized(int_tokens, cast(list[float], weights), tokenizer))


def _artifact_write_path(messages: list[dict[str, Any]]) -> str | None:
    """Return the artifact path from an artifact-only completion."""
    for msg in messages:
        for tc in (msg.get("tool_calls") or []):
            fn = tc.get("function") if isinstance(tc, dict) else None
            if not isinstance(fn, dict) or fn.get("name") != "write":
                continue
            args_text = fn.get("arguments")
            if not isinstance(args_text, str):
                continue
            try:
                args = json.loads(args_text)
            except json.JSONDecodeError:
                continue
            path = args.get("path")
            if isinstance(path, str) and path.strip():
                return path
    return None


class OnlineDPOPairDataset:
    """Offline DPO pairs plus aligned online-rollout seeds.

    Each row keeps:
    - the regular offline chosen/rejected pair built from ``pair_mode``
    - an aligned chosen side plus prompt for sampling an additional on-policy
      rejected artifact write from the current policy
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        renderer: renderers.Renderer,
        max_length: int | None,
        batch_size: int,
    ):
        self._rows = rows
        self._renderer = renderer
        self._max_length = max_length
        self._batch_size = batch_size
        self._indices = list(range(len(rows)))

    @classmethod
    def from_weight_json(
        cls,
        path: str,
        renderer: renderers.Renderer,
        max_length: int | None,
        batch_size: int,
        pair_mode: str = "first_last",
        pair_min_gap: int = 1,
        artifact_only_instruction: bool = False,
    ) -> "OnlineDPOPairDataset":
        pair_rows = extract_dpo_pairs(
            _load_sessions(path),
            renderer=renderer,
            pair_mode=pair_mode,
            pair_min_gap=pair_min_gap,
        )
        rows: list[dict[str, Any]] = []
        for row in pair_rows:
            expected_path = _artifact_write_path(row["chosen"])
            if expected_path is None:
                continue

            offline_prompt_messages = _hydrate_tool_calls(row["prompt"])
            chosen_messages = _hydrate_tool_calls(row["chosen"])
            rejected_messages = _hydrate_tool_calls(row["rejected"])

            offline_chosen_datum = conversation_to_datum(
                offline_prompt_messages + chosen_messages,
                renderer,
                max_length,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )
            offline_rejected_datum = conversation_to_datum(
                offline_prompt_messages + rejected_messages,
                renderer,
                max_length,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )

            online_prompt = row["prompt"]
            if artifact_only_instruction:
                online_prompt = _with_artifact_only_instruction(online_prompt, expected_path)
            online_prompt_messages = _hydrate_tool_calls(online_prompt)
            online_chosen_datum = conversation_to_datum(
                online_prompt_messages + chosen_messages,
                renderer,
                max_length,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )

            rows.append({
                "offline_chosen_datum": offline_chosen_datum,
                "offline_rejected_datum": offline_rejected_datum,
                "prompt_messages": online_prompt_messages,
                "prompt_input": renderer.build_generation_prompt(online_prompt_messages),
                "chosen_datum": online_chosen_datum,
                "expected_path": expected_path,
            })
        dataset = cls(rows, renderer, max_length, batch_size)
        logger.info(
            "Loaded %d DPO pair rows with online-rollout seeds from %s "
            "(pair_mode=%s, pair_min_gap=%d, raw_pairs=%d)",
            len(dataset._rows), path, pair_mode, pair_min_gap, len(pair_rows),
        )
        return dataset

    def __len__(self) -> int:
        if not self._rows:
            return 0
        return (len(self._rows) + self._batch_size - 1) // self._batch_size

    def set_epoch(self, seed: int) -> None:
        rng = random.Random(seed)
        self._indices = list(range(len(self._rows)))
        rng.shuffle(self._indices)

    def get_batch(self, index: int) -> list[dict[str, Any]]:
        start = index * self._batch_size
        end = min(start + self._batch_size, len(self._indices))
        data: list[tinker.Datum] = []
        for i in range(start, end):
            row = self._rows[self._indices[i]]
            data.extend([row["offline_chosen_datum"], row["offline_rejected_datum"]])
        return data

    def get_online_batch(self, index: int) -> list[dict[str, Any]]:
        start = index * self._batch_size
        end = min(start + self._batch_size, len(self._indices))
        return [self._rows[self._indices[i]] for i in range(start, end)]


class OnlineDPOAcceptedDataset:
    """Accepted artifact rows for online-only DPO rollout."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        renderer: renderers.Renderer,
        max_length: int | None,
        batch_size: int,
    ):
        self._rows = rows
        self._renderer = renderer
        self._max_length = max_length
        self._batch_size = batch_size
        self._indices = list(range(len(rows)))

    @classmethod
    def from_weight_json(
        cls,
        path: str,
        renderer: renderers.Renderer,
        max_length: int | None,
        batch_size: int,
        artifact_only_instruction: bool = False,
    ) -> "OnlineDPOAcceptedDataset":
        accepted_rows = extract_dpo_accepted_artifacts(
            _load_sessions(path),
            renderer=renderer,
        )
        rows: list[dict[str, Any]] = []
        for row in accepted_rows:
            prompt = row["prompt"]
            expected_path = row["expected_path"]
            if artifact_only_instruction:
                prompt = _with_artifact_only_instruction(prompt, expected_path)
            prompt_messages = _hydrate_tool_calls(prompt)
            chosen_messages = _hydrate_tool_calls(row["chosen"])
            chosen_datum = conversation_to_datum(
                prompt_messages + chosen_messages,
                renderer,
                max_length,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )
            rows.append({
                "prompt_messages": prompt_messages,
                "prompt_input": renderer.build_generation_prompt(prompt_messages),
                "chosen_datum": chosen_datum,
                "expected_path": expected_path,
            })
        dataset = cls(rows, renderer, max_length, batch_size)
        logger.info(
            "Loaded %d online-only DPO accepted artifact rows from %s (raw=%d)",
            len(dataset._rows), path, len(accepted_rows),
        )
        return dataset


async def _sample_online_dpo_pairs_async(
    rows: list[dict[str, Any]],
    renderer: renderers.Renderer,
    sampling_client: tinker.SamplingClient,
    max_tokens: int,
    temperature: float,
    attempts: int,
    max_length: int | None,
    step: int,
    sample_log_path: Path | None = None,
    sample_log_chars: int = 4000,
) -> tuple[list[tinker.Datum], dict[str, float]]:
    """Sample rejected artifact writes and interleave them with accepted chosen datums."""
    data: list[tinker.Datum] = []
    reason_counts: dict[str, int] = {}
    total_attempts = 0

    sample_log_f = None
    if sample_log_path is not None:
        sample_log_path.parent.mkdir(parents=True, exist_ok=True)
        sample_log_f = sample_log_path.open("a", encoding="utf-8")

    try:
        pending = list(enumerate(rows))
        for attempt_idx in range(max(1, attempts)):
            if not pending:
                break
            total_attempts += len(pending)
            results = await asyncio.gather(*[
                sampling_client.sample_async(
                    prompt=row["prompt_input"],
                    num_samples=1,
                    sampling_params=tinker.SamplingParams(
                        stop=renderer.get_stop_sequences(),
                        max_tokens=max_tokens,
                        temperature=temperature,
                    ),
                )
                for _row_idx, row in pending
            ])

            next_pending: list[tuple[int, dict[str, Any]]] = []
            for (row_idx, row), result in zip(pending, results, strict=True):
                expected_path = row["expected_path"]
                tokens = list(result.sequences[0].tokens)
                ok, reason, rejected_message = _parse_valid_artifact_write_message(
                    renderer, tokens, expected_path,
                )
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                if sample_log_f is not None:
                    rec = _summarize_sample(
                        renderer,
                        tokens,
                        expected_path,
                        ok,
                        reason,
                        row_idx,
                        attempt_idx,
                        step,
                        sample_log_chars,
                    )
                    rec["mode"] = "dpo_online"
                    sample_log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    sample_log_f.flush()
                if not ok:
                    next_pending.append((row_idx, row))
                    continue
                assert rejected_message is not None
                rejected_datum = conversation_to_datum(
                    row["prompt_messages"] + _hydrate_tool_calls([rejected_message]),
                    renderer,
                    max_length,
                    train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                )
                data.extend([row["chosen_datum"], rejected_datum])
            pending = next_pending
        if pending:
            reason_counts["example_filtered"] = reason_counts.get("example_filtered", 0) + len(pending)
    finally:
        if sample_log_f is not None:
            sample_log_f.close()

    n_rows = float(len(rows))
    n_valid = float(len(data) // 2)
    metrics: dict[str, float] = {
        "dpo_online/batch_examples": n_rows,
        "dpo_online/valid_pairs": n_valid,
        "dpo_online/filtered_examples": n_rows - n_valid,
        "dpo_online/filter_rate": (n_rows - n_valid) / max(n_rows, 1.0),
        "dpo_online/attempts": float(total_attempts),
    }
    for reason, count in reason_counts.items():
        safe_reason = reason.replace("/", "_").replace(":", "_")
        metrics[f"dpo_online/filter_reason/{safe_reason}"] = float(count)
    return data, metrics


def do_update(
    epoch_idx: int,
    batch_idx: int,
    n_batches: int,
    total_steps: int,
    dpo_beta: float,
    rpo_alpha: float,
    use_ipo: bool,
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
    get_ref_logprobs: Callable[[list[tinker.Datum]], list[torch.Tensor]],
    evaluators: list[Any],
    infrequent_evaluators: list[Any],
    dataset: SupervisedDataset,
    ml_logger: ml_log.Logger,
    log_path: str,
    tokenizer: Tokenizer,
    rolling_mgr: checkpoint_utils.CheckpointManager | None = None,
    data_override: list[tinker.Datum] | None = None,
    extra_metrics: dict[str, float] | None = None,
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
            rolling_mgr.maybe_save_rolling(
                step, {"epoch": epoch_idx, "batch": batch_idx},
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
            data = data_override if data_override is not None else dataset.get_batch(batch_idx)
        if extra_metrics:
            metrics.update(extra_metrics)

        chosen_data = [datum for i, datum in enumerate(data) if i % 2 == 0]
        rejected_data = [datum for i, datum in enumerate(data) if i % 2 == 1]

        if step == 0:
            for i in range(min(10, len(chosen_data))):
                _print_example(chosen_data[i], tokenizer, "Chosen")
                _print_example(rejected_data[i], tokenizer, "Rejected")

        with trace.scope_span_sync("get_ref_logprobs"):
            all_ref_logprob_seqs = get_ref_logprobs(data)
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
                rpo_alpha=rpo_alpha,
                use_ipo=use_ipo,
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
    _load_env()

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
    parser.add_argument(
        "--use-skyrl", action="store_true",
        help=(
            "Use the SkyRL-compatible path that pre-computes ref logprobs "
            "via training_client.forward(). Default (off) uses the Tinker "
            "cloud path: snapshot a frozen reference SamplingClient via "
            "save_weights_and_get_sampling_client and fetch ref logprobs "
            "per batch with compute_logprobs_async."
        ),
    )
    parser.add_argument(
        "--pair-mode",
        choices=["adjacent", "first_last", "all_pairs", "min_gap_pairs"],
        default="first_last",
        help=(
            "DPO preference pair construction. All modes scan the whole session "
            "by output filename (cross-unit). "
            "'first_last' (default): one pair per file, first write vs last write. "
            "'adjacent': one pair per consecutive version step per file. "
            "'all_pairs': every ordered earlier/later version pair. "
            "'min_gap_pairs': every earlier/later pair with version gap >= --pair-min-gap."
        ),
    )
    parser.add_argument(
        "--pair-min-gap",
        type=int,
        default=1,
        help=(
            "Minimum version-index gap used by --pair-mode min_gap_pairs. "
            "Ignored by other pair modes."
        ),
    )
    parser.add_argument(
        "--rpo-alpha",
        type=float,
        default=0.0,
        help=(
            "If > 0, add an RPO-style SFT anchor on the chosen completion: "
            "loss = base_loss + rpo_alpha * (-mean(chosen_logprob)). Directly "
            "mitigates likelihood displacement (chosen reward drifting "
            "negative). Typical values: 0.1 (mild) to 0.5 (strong). 0 "
            "disables. Compatible with both DPO (default) and --ipo."
        ),
    )
    parser.add_argument(
        "--ipo",
        action="store_true",
        help=(
            "Use IPO loss (Azar et al. 2023) instead of DPO. Replaces "
            "-log sigmoid(β·h) with (h - 1/(2β))². The squared loss targets "
            "a finite reward gap 1/(2β), preventing reward hacking and "
            "keeping the gap from growing to ±∞. Can be combined with "
            "--rpo-alpha for an anchored IPO variant."
        ),
    )
    parser.add_argument(
        "--online-rollout", action="store_true",
        help=(
            "Enable online artifact DPO: sample current-policy write(path, content) "
            "rollouts as rejected responses, and use accepted artifact writes from "
            "--train-path as chosen responses."
        ),
    )
    parser.add_argument("--rollout-max-tokens", type=int, default=4096)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-attempts", type=int, default=1)
    parser.add_argument(
        "--no-log-rollout-samples", action="store_true",
        help="Disable JSONL logging of online DPO rollout samples and filter reasons.",
    )
    parser.add_argument("--rollout-sample-log-chars", type=int, default=4000)
    parser.add_argument(
        "--artifact-only-rollout-instruction", action="store_true",
        help=(
            "Append a rollout-only instruction requiring exactly one write(path, content) "
            "tool call. Default is off to keep rollout prompts matched to inference."
        ),
    )
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
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=args.model_name,
        renderer_name=args.renderer_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    online_renderer = None
    if args.online_rollout:
        if args.use_skyrl:
            raise ValueError("--online-rollout requires Tinker sampling; do not use --use-skyrl")
        online_renderer = renderers.get_renderer(
            args.renderer_name,
            tokenizer=get_tokenizer(args.model_name),
        )
        dataset_builder = None
        dataset = OnlineDPOPairDataset.from_weight_json(
            path=args.train_path,
            renderer=online_renderer,
            max_length=args.max_length,
            batch_size=args.batch_size,
            pair_mode=args.pair_mode,
            pair_min_gap=args.pair_min_gap,
            artifact_only_instruction=args.artifact_only_rollout_instruction,
        )
    else:
        dataset_builder = WeightDPODataBuilder(
            train_path=args.train_path,
            test_path=args.test_path,
            pair_mode=args.pair_mode,
            pair_min_gap=args.pair_min_gap,
            common_config=common_config,
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
    rolling_mgr = checkpoint_utils.CheckpointManager(
        training_client=training_client,
        service_client=service_client,
        log_path=log_path,
        save_every=0,
        rolling_save_every=args.rolling_save_every,
        rolling_ttl_seconds=args.rolling_ttl_seconds,
    )

    logger.info(
        "Training for %d batches x %d epochs = %d steps",
        n_batches, args.num_epochs, n_batches * args.num_epochs,
    )

    # ------------------------------------------------------------------ #
    # Reference logprob provider (set up BEFORE any optim_step)           #
    # ------------------------------------------------------------------ #
    if not args.use_skyrl:
        # Tinker cloud path: snapshot the initial weights once.
        # The returned SamplingClient is frozen and cheap to query per batch.
        reference_client = training_client.save_weights_and_get_sampling_client()
        sampling_client = reference_client
        logger.info("Tinker mode: created frozen reference SamplingClient")

        def get_ref_logprobs(data: list[tinker.Datum]) -> list[torch.Tensor]:
            return _live_ref_logprobs(reference_client, data)
    else:
        # SkyRL path: pre-compute ref logprobs for every datum we will see,
        # using training_client.forward() at the initial weights.
        ref_logprob_cache = precompute_ref_logprob_cache(
            training_client, dataset, args.num_epochs,
        )

        def get_ref_logprobs(data: list[tinker.Datum]) -> list[torch.Tensor]:
            return [ref_logprob_cache[_datum_fingerprint(d)] for d in data]

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
            data_override = None
            extra_metrics = None
            if args.online_rollout:
                assert online_renderer is not None
                offline_data = dataset.get_batch(batch_idx)
                rows = dataset.get_online_batch(batch_idx)
                online_data, extra_metrics = asyncio.run(
                    _sample_online_dpo_pairs_async(
                        rows,
                        online_renderer,
                        sampling_client,
                        max_tokens=args.rollout_max_tokens,
                        temperature=args.rollout_temperature,
                        attempts=args.rollout_attempts,
                        max_length=args.max_length,
                        step=step,
                        sample_log_path=(
                            Path(log_path) / "dpo_online_rollout_samples.jsonl"
                            if not args.no_log_rollout_samples else None
                        ),
                        sample_log_chars=args.rollout_sample_log_chars,
                    )
                )
                data_override = offline_data + online_data
                if not online_data:
                    extra_metrics = {
                        **(extra_metrics or {}),
                        "dpo_online/no_valid_batch": 1.0,
                    }
            do_update(
                epoch_idx=epoch_idx,
                batch_idx=batch_idx,
                n_batches=n_batches,
                total_steps=total_steps,
                dpo_beta=args.dpo_beta,
                rpo_alpha=args.rpo_alpha,
                use_ipo=args.ipo,
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
                get_ref_logprobs=get_ref_logprobs,
                evaluators=[],
                infrequent_evaluators=[],
                dataset=dataset,
                ml_logger=ml_logger,
                log_path=log_path,
                tokenizer=tokenizer,
                rolling_mgr=rolling_mgr,
                data_override=data_override,
                extra_metrics=extra_metrics,
            )
            if args.online_rollout:
                sampling_client = training_client.save_weights_and_get_sampling_client()
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
