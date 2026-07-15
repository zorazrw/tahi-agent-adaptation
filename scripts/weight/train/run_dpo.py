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
import difflib
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

from . import logprob_viz
from .dpo_rollout import rollout_one_dpo_episode
from .formatter import WeightDPODataBuilder, _hydrate_tool_calls, _load_sessions
from .run_opd import (
    _parse_valid_artifact_write_message,
    _summarize_sample,
    _with_artifact_only_instruction,
)

try:  # Supports both `python -m weight...` from scripts/ and `python -m scripts.weight...`.
    from weight.data.extract import (  # type: ignore[import-not-found]
        extract_dpo_accepted_artifacts,
        extract_dpo_final_artifacts,
    )
except ModuleNotFoundError:  # pragma: no cover - depends on invocation cwd
    from ..data.extract import (
        extract_dpo_accepted_artifacts,
        extract_dpo_final_artifacts,
    )

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

    # Agentic online DPO. When enabled, the rejected side is produced by a full
    # multi-turn tool-using rollout in a sandbox; the student's final artifact
    # (matched to the chosen file by basename) becomes the rejected response,
    # while the session's final accepted artifact remains the chosen response.
    # Only the artifact write is trained on, not the intermediate tool calls.
    agentic_rollout: bool = False
    # "on_policy": refresh the sampler after every optim step so rejected
    # rollouts track the current policy. "off_policy": keep the frozen initial
    # snapshot (== reference model) for the whole run, using the rollouts purely
    # as a one-shot negative-augmentation source.
    agentic_policy_mode: str = "off_policy"
    agentic_num_rollouts: int = 8
    # batch_size is the target #preference-pairs per batch; each batch draws
    # batch_size // agentic_pairs_per_session sessions, each contributing exactly
    # agentic_pairs_per_session pairs (subsampled if it yields more, oversampled
    # by cycling if fewer). agentic_session_reserve fetches extra sessions per
    # batch so zero-yield sessions can be replaced.
    agentic_pairs_per_session: int = 1
    agentic_session_reserve: int = 0
    # Cap on rollouts running concurrently (0 = unlimited). Bounds peak memory
    # since each in-flight rollout holds a sandbox + growing prompt.
    agentic_max_concurrent_rollouts: int = 0
    # Additive offline rollout source: also inject each session's first-written
    # artifact version as a synthetic rejected snapshot (the "first_last" pair),
    # alongside the on-policy model rollouts. Default off.
    agentic_include_first_last: bool = False
    # Min content-similarity (0..1) for matching a student file to a chosen
    # artifact by CONTENT/type rather than filename. Lower => more pairs (looser
    # matches); higher => fewer but tighter. Single same-extension candidates and
    # single-artifact/single-file rollouts bypass this floor.
    agentic_match_min_similarity: float = 0.05
    agentic_max_turns: int = 48
    agentic_max_turns_per_step: int = 8
    agentic_max_steps: int = 6
    agentic_enable_bash: bool = True
    agentic_tool_timeout_s: int = 20
    agentic_max_trajectory_tokens: int | None = None
    min_artifact_versions: int = 1

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


class OnlineDPOAcceptedDataset:
    """Accepted artifact rows for online DPO rollout.

    Each row supplies the chosen side from weight JSON. At train time the
    current policy samples a fresh artifact write from the same prompt; that
    sampled write becomes the rejected side.
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
            "Loaded %d online DPO accepted artifact rows from %s (raw=%d)",
            len(dataset._rows), path, len(accepted_rows),
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
        return [self._rows[self._indices[i]] for i in range(start, end)]


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


# ---------------------------------------------------------------------------
# Agentic online DPO (full multi-turn tool-using rollout for the rejected side)
# ---------------------------------------------------------------------------


def _artifact_write_content(messages: list[dict[str, Any]]) -> str | None:
    """Read the ``write()`` content from an artifact-only completion (dict form)."""
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
            content = args.get("content")
            if isinstance(content, str):
                return content
    return None


class OnlineDPOAgenticDataset:
    """Per-session rollout seeds + final accepted artifacts for agentic DPO.

    Each row carries the agentic rollout seed (initial task + system_prompt +
    tool_schemas) and one or more chosen artifacts (the session's final accepted
    file versions). At train time the current policy runs a full multi-turn
    rollout; its final sandbox files become the rejected side, matched to each
    chosen artifact by basename. Both sides are trained as artifact-only single
    ``write`` messages under the chosen version's prompt (LAST_ASSISTANT_MESSAGE),
    so intermediate tool calls drive the sandbox but are never trained on.
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        renderer: renderers.Renderer,
        max_length: int | None,
        batch_size: int,
        pairs_per_session: int = 1,
        session_reserve: int = 0,
    ):
        self._rows = rows
        self._renderer = renderer
        self._max_length = max_length
        # ``batch_size`` is the target number of preference pairs (training
        # samples) per batch. Each batch draws ``sessions_per_batch`` sessions,
        # each contributing exactly ``pairs_per_session`` pairs, so
        # sessions_per_batch * pairs_per_session == batch_size.
        self._batch_size = batch_size
        self._pairs_per_session = max(1, pairs_per_session)
        self._sessions_per_batch = max(1, batch_size // self._pairs_per_session)
        # Extra sessions fetched per batch so the sampler can replace sessions
        # whose rollouts yield zero valid pairs.
        self._session_reserve = max(0, session_reserve)
        if batch_size % self._pairs_per_session != 0:
            logger.warning(
                "batch_size=%d is not a multiple of pairs_per_session=%d; "
                "realized batch will be %d pairs (sessions_per_batch=%d)",
                batch_size, self._pairs_per_session,
                self._sessions_per_batch * self._pairs_per_session,
                self._sessions_per_batch,
            )
        self._indices = list(range(len(rows)))

    @property
    def sessions_per_batch(self) -> int:
        return self._sessions_per_batch

    @property
    def pairs_per_session(self) -> int:
        return self._pairs_per_session

    @classmethod
    def from_weight_json(
        cls,
        path: str,
        renderer: renderers.Renderer,
        max_length: int | None,
        batch_size: int,
        pairs_per_session: int = 1,
        session_reserve: int = 0,
        min_versions: int = 1,
        artifact_only_instruction: bool = False,
    ) -> "OnlineDPOAgenticDataset":
        seed_rows = extract_dpo_final_artifacts(
            _load_sessions(path),
            renderer=renderer,
            min_versions=min_versions,
        )
        rows: list[dict[str, Any]] = []
        for seed in seed_rows:
            chosen_artifacts: list[dict[str, Any]] = []
            for art in seed["chosen_artifacts"]:
                prompt = art["prompt"]
                expected_path = art["expected_path"]
                if artifact_only_instruction:
                    prompt = _with_artifact_only_instruction(prompt, expected_path)
                prompt_messages = _hydrate_tool_calls(prompt)
                chosen_messages = _hydrate_tool_calls(art["chosen"])
                chosen_datum = conversation_to_datum(
                    prompt_messages + chosen_messages,
                    renderer,
                    max_length,
                    train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                )
                chosen_artifacts.append({
                    "expected_path": expected_path,
                    "basename": art["basename"],
                    "prompt_messages": prompt_messages,
                    "chosen_datum": chosen_datum,
                    "chosen_content": _artifact_write_content(art["chosen"]),
                    "first_content": art.get("first_content"),
                })
            if not chosen_artifacts:
                continue
            rows.append({
                "system_prompt": seed["system_prompt"],
                "tool_schemas": seed["tool_schemas"],
                "prompt_messages": _hydrate_tool_calls(seed["prompt_messages"]),
                "chosen_artifacts": chosen_artifacts,
                "meta": seed.get("meta") or {},
            })
        dataset = cls(
            rows, renderer, max_length, batch_size,
            pairs_per_session=pairs_per_session,
            session_reserve=session_reserve,
        )
        logger.info(
            "Loaded %d agentic DPO session rows from %s (raw_sessions=%d, "
            "artifacts=%d, sessions_per_batch=%d, pairs_per_session=%d, reserve=%d)",
            len(dataset._rows), path, len(seed_rows),
            sum(len(r["chosen_artifacts"]) for r in rows),
            dataset._sessions_per_batch, dataset._pairs_per_session,
            dataset._session_reserve,
        )
        return dataset

    def __len__(self) -> int:
        if not self._rows:
            return 0
        return (len(self._rows) + self._sessions_per_batch - 1) // self._sessions_per_batch

    def set_epoch(self, seed: int) -> None:
        rng = random.Random(seed)
        self._indices = list(range(len(self._rows)))
        rng.shuffle(self._indices)

    def get_batch(self, index: int) -> list[dict[str, Any]]:
        # Return a pool of ``sessions_per_batch + session_reserve`` sessions.
        # The stride is ``sessions_per_batch`` (the reserve overlaps into the
        # next batch's primary sessions, which is fine for online sampling).
        # The pool is capped to the number of *distinct* sessions so the same
        # session is never rolled out more than once per batch (to get multiple
        # rollouts per session, use ``num_rollouts``, not this pool). Indices
        # wrap around so the pool is always full even at epoch end.
        n = len(self._indices)
        if n == 0:
            return []
        pool = min(self._sessions_per_batch + self._session_reserve, n)
        start = index * self._sessions_per_batch
        return [self._rows[self._indices[(start + off) % n]] for off in range(pool)]


# Char cap for similarity scoring (bounds difflib cost on large artifacts).
_SIM_CAP = 20_000
# Tie-break bonus when basenames happen to match: a hint, never a hard key.
_NAME_BONUS = 0.25


def _content_similarity(a: str, b: str) -> float:
    """Cheap 0..1 content-similarity ratio (capped for cost)."""
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a[:_SIM_CAP], b[:_SIM_CAP]).quick_ratio()


def _match_student_artifacts(
    chosen_artifacts: list[dict[str, Any]],
    snapshot: dict[str, str],
    *,
    min_similarity: float = 0.05,
) -> dict[int, tuple[str, str, float]]:
    """Map ``chosen_idx -> (student_content, matched_by, similarity)`` by CONTENT.

    The rollout starts from an empty sandbox, so the student picks its own
    filename/path; matching the chosen artifact by basename misses most of the
    time. Instead we correspond by content/type (the rejected write is re-framed
    under the chosen artifact's canonical path anyway, so only the *content* of
    the student's attempt matters):

    1. **Positional fallback** -- one chosen artifact + one produced file => pair
       them directly regardless of name/type (the common single-file task).
    2. **Extension gating** -- candidates are restricted to files sharing the
       chosen artifact's extension when any exist (else all text files).
    3. **Content similarity** -- each (chosen, candidate) is scored; pairs are
       assigned greedily best-first, each student file used at most once.
    4. **Name as a tie-break bonus only** -- a matching basename nudges the score
       but is never required.

    A sole same-extension candidate (``extension_unique``) and the positional
    fallback bypass ``min_similarity`` (strong structural signals); everything
    else must clear the floor to avoid pairing unrelated files.
    """
    files = list(snapshot.items())  # [(relpath, content)]
    if not files or not chosen_artifacts:
        return {}

    if len(chosen_artifacts) == 1 and len(files) == 1:
        return {0: (files[0][1], "positional", 1.0)}

    # (adj_score, raw_similarity, matched_by, chosen_idx, file_idx)
    scored: list[tuple[float, float, str, int, int]] = []
    for ci, art in enumerate(chosen_artifacts):
        cext = os.path.splitext(art.get("expected_path", ""))[1].lower()
        cbase = art.get("basename")
        cc = art.get("chosen_content") or ""
        same_ext = [
            (fi, p, c)
            for fi, (p, c) in enumerate(files)
            if os.path.splitext(p)[1].lower() == cext
        ]
        gated = bool(same_ext)
        pool = same_ext if gated else list(
            (fi, p, c) for fi, (p, c) in enumerate(files)
        )
        unique_in_pool = len(pool) == 1
        for fi, p, c in pool:
            sim = _content_similarity(cc, c)
            adj = sim
            matched_by = "extension" if gated else "similarity"
            if os.path.basename(p) == cbase:
                adj += _NAME_BONUS
                matched_by = "name"
            if gated and unique_in_pool:
                matched_by = "extension_unique"
            scored.append((adj, sim, matched_by, ci, fi))

    scored.sort(key=lambda t: t[0], reverse=True)
    used_c: set[int] = set()
    used_f: set[int] = set()
    out: dict[int, tuple[str, str, float]] = {}
    for _adj, sim, matched_by, ci, fi in scored:
        if ci in used_c or fi in used_f:
            continue
        if matched_by != "extension_unique" and sim < min_similarity:
            continue
        out[ci] = (files[fi][1], matched_by, sim)
        used_c.add(ci)
        used_f.add(fi)
    return out


async def _sample_agentic_dpo_pairs_async(
    rows: list[dict[str, Any]],
    renderer: renderers.Renderer,
    sampling_client: tinker.SamplingClient,
    *,
    num_rollouts: int = 1,
    pairs_per_session: int = 1,
    target_sessions: int = 1,
    max_concurrent_rollouts: int = 0,
    match_min_similarity: float = 0.05,
    include_first_last: bool = False,
    max_tokens: int,
    temperature: float,
    max_turns: int,
    max_turns_per_step: int,
    max_steps: int,
    enable_bash: bool,
    tool_timeout_s: int,
    max_trajectory_tokens: int | None,
    max_length: int | None,
    step: int,
    sample_log_path: Path | None = None,
    sample_log_chars: int = 4000,
    finish_log_path: Path | None = None,
    raw_trajectory_log_path: Path | None = None,
) -> tuple[list[tinker.Datum], dict[str, float]]:
    """Run agentic rollouts and assemble a fixed-size, session-balanced batch.

    ``rows`` is a *pool* of session seeds (``target_sessions`` plus a reserve).
    The current policy runs ``num_rollouts`` independent multi-turn tool-using
    rollouts per pooled session, all dispatched in parallel. Each rollout's final
    sandbox files are matched to each chosen artifact by CONTENT/type (see
    ``_match_student_artifacts``, ``match_min_similarity``) rather than filename;
    every matched, non-empty, non-identical student file becomes a rejected
    ``write`` paired with that artifact's chosen ``write`` (both trained
    artifact-only under the chosen version's prompt).

    The batch is then assembled to a fixed size: up to ``target_sessions``
    sessions that produced >=1 valid pair are selected, and EACH contributes
    exactly ``pairs_per_session`` pairs -- subsampled if it produced more,
    oversampled (cycled) if it produced fewer. Pairs are interleaved round-robin
    so adjacent samples come from different sessions. The realized batch therefore
    holds ``selected_sessions * pairs_per_session`` pairs, which equals the target
    ``target_sessions * pairs_per_session`` whenever enough pooled sessions yield
    at least one valid pair (the reserve covers zero-yield sessions).
    """
    num_rollouts = max(1, num_rollouts)
    pairs_per_session = max(1, pairs_per_session)
    target_sessions = max(1, target_sessions)

    # A finished-rollout record (turns + token footprint) is written to a
    # SEPARATE log file the moment each rollout completes -- i.e. before the
    # training step and independently of whether sibling rollouts or the forward/
    # backward later fail. The file is opened up front and writes are serialized.
    finish_log_f = None
    if finish_log_path is not None:
        finish_log_path.parent.mkdir(parents=True, exist_ok=True)
        finish_log_f = finish_log_path.open("a", encoding="utf-8")
    finish_lock = asyncio.Lock()

    # Raw (untruncated) trajectory dump, also written the instant each rollout
    # finishes, to diagnose what inflates the prompt to context-overflow sizes.
    raw_traj_f = None
    if raw_trajectory_log_path is not None:
        raw_trajectory_log_path.parent.mkdir(parents=True, exist_ok=True)
        raw_traj_f = raw_trajectory_log_path.open("a", encoding="utf-8")
    raw_traj_lock = asyncio.Lock()
    collect_raw = raw_trajectory_log_path is not None

    # Cap how many rollouts run concurrently (0/negative = unlimited). Each
    # in-flight rollout holds a sandbox + growing prompt in memory, so this is the
    # main lever for bounding peak RSS when a batch dispatches many rollouts.
    rollout_sem = (
        asyncio.Semaphore(max_concurrent_rollouts)
        if max_concurrent_rollouts and max_concurrent_rollouts > 0
        else None
    )

    def _drop_reason_from_metrics(m: dict[str, float]) -> str | None:
        if m.get("agentic/rollout_error"):
            return "rollout_error"
        if m.get("agentic/sample_error"):
            return "sample_error"
        if m.get("agentic/parse_failed"):
            return "parse_failed"
        if m.get("agentic/context_overflow"):
            return "context_overflow"
        if m.get("agentic/empty_trajectory"):
            return "empty_trajectory"
        return None

    async def _run_episode(row: dict[str, Any]) -> Any:
        if rollout_sem is None:
            return await rollout_one_dpo_episode(
                row, renderer, sampling_client,
                max_tokens=max_tokens, temperature=temperature, max_turns=max_turns,
                max_turns_per_step=max_turns_per_step, max_steps=max_steps,
                enable_bash=enable_bash, tool_timeout_s=tool_timeout_s,
                max_trajectory_tokens=max_trajectory_tokens,
                collect_transcript=sample_log_path is not None,
                collect_raw_trajectory=collect_raw, log_field_chars=sample_log_chars,
            )
        async with rollout_sem:
            return await rollout_one_dpo_episode(
                row, renderer, sampling_client,
                max_tokens=max_tokens, temperature=temperature, max_turns=max_turns,
                max_turns_per_step=max_turns_per_step, max_steps=max_steps,
                enable_bash=enable_bash, tool_timeout_s=tool_timeout_s,
                max_trajectory_tokens=max_trajectory_tokens,
                collect_transcript=sample_log_path is not None,
                collect_raw_trajectory=collect_raw, log_field_chars=sample_log_chars,
            )

    async def _run_and_log(row: dict[str, Any], rollout_idx: int) -> Any:
        meta = row.get("meta") or {}
        try:
            res = await _run_episode(row)
            snapshot, m, episode_log = res
        except Exception as e:  # noqa: BLE001 - keep batch alive, still log finish
            logger.warning(
                "dpo agentic rollout crashed step=%d session=%s rollout=%d: %s: %s",
                step, meta.get("session_uuid"), rollout_idx, type(e).__name__, e,
            )
            snapshot, m, episode_log = {}, {"agentic/rollout_error": 1.0}, None

        turns = int(m.get("agentic/turns", 0.0))
        steps = int(m.get("agentic/steps", 0.0))
        tool_calls = int(m.get("agentic/tool_calls", 0.0))
        prompt_tokens_max = int(m.get("agentic/prompt_tokens_max", 0.0))
        prompt_tokens_final = int(m.get("agentic/prompt_tokens_final", 0.0))
        gen_tokens = int(m.get("agentic/gen_tokens", 0.0))
        drop_reason = _drop_reason_from_metrics(m)
        logger.info(
            "dpo agentic rollout done step=%d session=%s rollout=%d turns=%d "
            "steps=%d tool_calls=%d prompt_tokens_max=%d prompt_tokens_final=%d "
            "gen_tokens=%d%s",
            step, meta.get("session_uuid"), rollout_idx, turns, steps, tool_calls,
            prompt_tokens_max, prompt_tokens_final, gen_tokens,
            f" drop={drop_reason}" if drop_reason else "",
        )
        if finish_log_f is not None:
            rec = {
                "step": step,
                "mode": "dpo_agentic",
                "session_uuid": meta.get("session_uuid"),
                "rollout_idx": rollout_idx,
                "trajectory_valid": drop_reason is None,
                "drop_reason": drop_reason,
                "turns": turns,
                "steps": steps,
                "tool_calls": tool_calls,
                "prompt_tokens_max": prompt_tokens_max,
                "prompt_tokens_final": prompt_tokens_final,
                "gen_tokens": gen_tokens,
            }
            async with finish_lock:
                finish_log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                finish_log_f.flush()
        if raw_traj_f is not None:
            # Pop the raw messages so the (potentially huge) untruncated content is
            # not retained in ``results_by_row`` for the whole batch after we've
            # written it -- keeping it around was a major memory amplifier.
            raw_messages = (episode_log or {}).pop("raw_messages", None) or []
            raw_rec = {
                "step": step,
                "mode": "dpo_agentic",
                "session_uuid": meta.get("session_uuid"),
                "rollout_idx": rollout_idx,
                "drop_reason": drop_reason,
                "turns": turns,
                "prompt_tokens_max": prompt_tokens_max,
                "prompt_tokens_final": prompt_tokens_final,
                "n_messages": len(raw_messages),
                "raw_messages": raw_messages,
            }
            async with raw_traj_lock:
                raw_traj_f.write(json.dumps(raw_rec, ensure_ascii=False) + "\n")
                raw_traj_f.flush()
            del raw_messages, raw_rec
        return snapshot, m, episode_log

    # Flatten (row, rollout) into one parallel dispatch, then regroup by row.
    flat_tasks = []
    flat_row_idx: list[int] = []
    try:
        for row_idx, row in enumerate(rows):
            for r_idx in range(num_rollouts):
                flat_tasks.append(_run_and_log(row, r_idx))
                flat_row_idx.append(row_idx)
        flat_results = await asyncio.gather(*flat_tasks)
    finally:
        if finish_log_f is not None:
            finish_log_f.close()
        if raw_traj_f is not None:
            raw_traj_f.close()
    results_by_row: list[list[Any]] = [[] for _ in rows]
    for row_idx, res in zip(flat_row_idx, flat_results, strict=True):
        results_by_row[row_idx].append(res)

    # Optional offline "first_last" rollout: frame each session's first-written
    # artifact version as a synthetic rollout snapshot (the rejected side),
    # additive to the on-policy model rollouts above. It flows through the exact
    # same _match_student_artifacts + identical/empty filters and per-session
    # balancing as a real rollout. The metrics are zeroed (turns/tokens=0) and
    # tagged so it does not pollute rollout-cost aggregates or logs.
    n_first_last_injected = 0
    if include_first_last:
        for row_idx, row in enumerate(rows):
            snapshot: dict[str, str] = {}
            for art in row.get("chosen_artifacts") or []:
                first_content = art.get("first_content")
                if not isinstance(first_content, str) or not first_content.strip():
                    continue
                # Skip when the first draft already equals the chosen artifact;
                # the downstream "identical" filter would drop it anyway.
                if first_content == art.get("chosen_content"):
                    continue
                snapshot[art["expected_path"]] = first_content
            if not snapshot:
                continue
            synthetic_metrics = {"agentic/offline_first_last": 1.0}
            synthetic_log = (
                {"valid": True, "drop_reason": None, "source": "first_last"}
                if sample_log_path is not None
                else None
            )
            results_by_row[row_idx].append((snapshot, synthetic_metrics, synthetic_log))
            n_first_last_injected += 1

    data: list[tinker.Datum] = []
    # Valid [chosen, rejected] pairs grouped by session (aligned with ``rows``),
    # so the batch can be balanced/interleaved across sessions afterwards.
    pairs_by_session: list[list[tuple[tinker.Datum, tinker.Datum]]] = []
    reason_counts: dict[str, int] = {}
    n_potential = 0
    n_rollouts_total = 0
    n_first_last_pairs = 0  # valid pairs sourced from the offline first_last rollout
    agg_turns = 0.0
    agg_tool_calls = 0.0
    agg_steps = 0.0
    agg_parse_failed = 0.0
    agg_overflow = 0.0
    agg_empty = 0.0
    agg_sample_error = 0.0
    agg_rollout_error = 0.0
    agg_prompt_tokens_max = 0.0   # peak across all rollouts
    agg_prompt_tokens_final = 0.0  # summed (for mean)
    agg_gen_tokens = 0.0           # summed (for mean)

    sample_log_f = None
    if sample_log_path is not None:
        sample_log_path.parent.mkdir(parents=True, exist_ok=True)
        sample_log_f = sample_log_path.open("a", encoding="utf-8")

    try:
        for row, rollout_results in zip(rows, results_by_row, strict=True):
            chosen_artifacts = row.get("chosen_artifacts") or []
            meta = row.get("meta") or {}
            # Each rollout can contribute one rejected per chosen artifact.
            n_potential += len(chosen_artifacts) * len(rollout_results)
            this_session_pairs: list[tuple[tinker.Datum, tinker.Datum]] = []

            for rollout_idx, (snapshot, m, episode_log) in enumerate(rollout_results):
                # The synthetic offline first_last "rollout" carries no real
                # trajectory, so it is excluded from the on-policy rollout-cost
                # aggregates (turns/tokens/error counts) to keep their means and
                # the rollout count reflective of actual model sampling.
                is_first_last = bool(m.get("agentic/offline_first_last"))
                if not is_first_last:
                    n_rollouts_total += 1
                    agg_turns += m.get("agentic/turns", 0.0)
                    agg_tool_calls += m.get("agentic/tool_calls", 0.0)
                    agg_steps += m.get("agentic/steps", 0.0)
                    agg_parse_failed += m.get("agentic/parse_failed", 0.0)
                    agg_overflow += m.get("agentic/context_overflow", 0.0)
                    agg_empty += m.get("agentic/empty_trajectory", 0.0)
                    agg_sample_error += m.get("agentic/sample_error", 0.0)
                    agg_rollout_error += m.get("agentic/rollout_error", 0.0)
                    agg_prompt_tokens_max = max(
                        agg_prompt_tokens_max, m.get("agentic/prompt_tokens_max", 0.0)
                    )
                    agg_prompt_tokens_final += m.get("agentic/prompt_tokens_final", 0.0)
                    agg_gen_tokens += m.get("agentic/gen_tokens", 0.0)

                # Correspond the student's produced files to each chosen artifact
                # by CONTENT/type (not filename); see ``_match_student_artifacts``.
                matches = _match_student_artifacts(
                    chosen_artifacts, snapshot, min_similarity=match_min_similarity
                )

                matched_pairs: list[dict[str, Any]] = []
                for ci, art in enumerate(chosen_artifacts):
                    match = matches.get(ci)
                    if match is None:
                        reason_counts["no_match"] = reason_counts.get("no_match", 0) + 1
                        continue
                    student_content, matched_by, match_score = match
                    if not student_content.strip():
                        reason_counts["empty_content"] = reason_counts.get("empty_content", 0) + 1
                        continue
                    if art.get("chosen_content") is not None and student_content == art["chosen_content"]:
                        reason_counts["identical"] = reason_counts.get("identical", 0) + 1
                        continue
                    rejected_message = {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "call_artifact_rejected",
                            "type": "function",
                            "function": {
                                "name": "write",
                                "arguments": json.dumps(
                                    {"path": art["expected_path"], "content": student_content},
                                    ensure_ascii=False,
                                ),
                            },
                        }],
                    }
                    rejected_datum = conversation_to_datum(
                        art["prompt_messages"] + _hydrate_tool_calls([rejected_message]),
                        renderer,
                        max_length,
                        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                    )
                    this_session_pairs.append((art["chosen_datum"], rejected_datum))
                    reason_counts["valid"] = reason_counts.get("valid", 0) + 1
                    mb_key = f"matched_by_{matched_by}"
                    reason_counts[mb_key] = reason_counts.get(mb_key, 0) + 1
                    matched_pairs.append({
                        "basename": art["basename"],
                        "expected_path": art["expected_path"],
                        "matched_by": matched_by,
                        "match_score": round(float(match_score), 4),
                        "rejected_content_preview": student_content[:sample_log_chars],
                    })

                # Finished-rollout turns/tokens are logged per-rollout in
                # ``_run_and_log`` (separate file, before the training step). Here
                # we only persist the richer per-rollout transcript + matched pairs.
                if is_first_last:
                    n_first_last_pairs += len(matched_pairs)

                drop_reason = (episode_log or {}).get("drop_reason")
                turns = int(m.get("agentic/turns", 0.0))
                prompt_tokens_max = int(m.get("agentic/prompt_tokens_max", 0.0))
                prompt_tokens_final = int(m.get("agentic/prompt_tokens_final", 0.0))
                gen_tokens = int(m.get("agentic/gen_tokens", 0.0))
                if sample_log_f is not None:
                    rec = {
                        "step": step,
                        "mode": "dpo_agentic",
                        "source": "first_last" if is_first_last else "rollout",
                        "session_uuid": meta.get("session_uuid"),
                        "rollout_idx": -1 if is_first_last else rollout_idx,
                        "valid": bool(matched_pairs),
                        "drop_reason": drop_reason,
                        "n_pairs": len(matched_pairs),
                        "turns": turns,
                        "steps": m.get("agentic/steps"),
                        "tool_calls": m.get("agentic/tool_calls"),
                        "prompt_tokens_max": prompt_tokens_max,
                        "prompt_tokens_final": prompt_tokens_final,
                        "gen_tokens": gen_tokens,
                        "matched_pairs": matched_pairs,
                        "messages": (episode_log or {}).get("messages") or [],
                    }
                    sample_log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    sample_log_f.flush()

            pairs_by_session.append(this_session_pairs)
            logger.info(
                "dpo agentic step=%d session=%s rollouts=%d pairs=%d/%d",
                step,
                meta.get("session_uuid"),
                len(rollout_results),
                len(this_session_pairs),
                len(chosen_artifacts) * len(rollout_results),
            )
    finally:
        if sample_log_f is not None:
            sample_log_f.close()

    # Assemble a fixed-size batch with an equal number of pairs per session.
    # Shuffle within each session (so subsampling isn't biased toward the first
    # rollout/artifact) and shuffle the session order (so which sessions get
    # picked / dropped is unbiased). Select up to ``target_sessions`` sessions
    # that produced >=1 valid pair; the pooled reserve covers zero-yield ones.
    # Give each selected session EXACTLY ``pairs_per_session`` pairs, cycling
    # (oversampling) when it produced fewer, then interleave round-robin.
    rng = random.Random(step)
    for pairs in pairs_by_session:
        rng.shuffle(pairs)
    n_valid_total = sum(len(pairs) for pairs in pairs_by_session)
    n_zero_yield = sum(1 for pairs in pairs_by_session if not pairs)
    qualifying = [pairs for pairs in pairs_by_session if pairs]
    rng.shuffle(qualifying)
    selected = qualifying[:target_sessions]

    n_oversampled = 0
    per_session_selected: list[list[tuple[tinker.Datum, tinker.Datum]]] = []
    for pairs in selected:
        if len(pairs) >= pairs_per_session:
            chosen_pairs = pairs[:pairs_per_session]
        else:
            chosen_pairs = [pairs[j % len(pairs)] for j in range(pairs_per_session)]
            n_oversampled += pairs_per_session - len(pairs)
        per_session_selected.append(chosen_pairs)

    for j in range(pairs_per_session):
        for pairs in per_session_selected:
            chosen_datum, rejected_datum = pairs[j]
            data.extend([chosen_datum, rejected_datum])

    n_selected = len(selected)
    n_used_distinct = sum(min(len(pairs), pairs_per_session) for pairs in selected)
    n_dropped = n_valid_total - n_used_distinct
    under_target = max(0, target_sessions - n_selected)

    n_rows = float(len(rows))
    n_valid_pairs = float(len(data) // 2)
    n_pot = float(n_potential)
    metrics: dict[str, float] = {
        "dpo_online/batch_examples": n_pot,
        "dpo_online/valid_pairs": n_valid_pairs,
        "dpo_online/filtered_examples": n_pot - float(n_valid_total),
        "dpo_online/filter_rate": (n_pot - float(n_valid_total)) / max(n_pot, 1.0),
        "dpo_online/sessions": n_rows,
        "dpo_online/rollouts": float(n_rollouts_total),
        "dpo_online/num_rollouts_per_session": float(num_rollouts),
        "dpo_online/target_pairs": float(target_sessions * pairs_per_session),
        "dpo_online/target_sessions": float(target_sessions),
        "dpo_online/selected_sessions": float(n_selected),
        "dpo_online/under_target_sessions": float(under_target),
        "dpo_online/zero_yield_sessions": float(n_zero_yield),
        "dpo_online/pairs_per_session": float(pairs_per_session),
        "dpo_online/valid_pairs_prebalance": float(n_valid_total),
        "dpo_online/oversampled_pairs": float(n_oversampled),
        "dpo_online/dropped_pairs": float(n_dropped),
        "dpo_online/first_last_injected": float(n_first_last_injected),
        "dpo_online/first_last_pairs": float(n_first_last_pairs),
        "agentic/turns": agg_turns / max(float(n_rollouts_total), 1.0),
        "agentic/tool_calls": agg_tool_calls / max(float(n_rollouts_total), 1.0),
        "agentic/steps": agg_steps / max(float(n_rollouts_total), 1.0),
        "agentic/parse_failed": agg_parse_failed,
        "agentic/context_overflow": agg_overflow,
        "agentic/empty_trajectory": agg_empty,
        "agentic/sample_error": agg_sample_error,
        "agentic/rollout_error": agg_rollout_error,
        "agentic/prompt_tokens_max": agg_prompt_tokens_max,
        "agentic/prompt_tokens_final_mean": agg_prompt_tokens_final / max(float(n_rollouts_total), 1.0),
        "agentic/gen_tokens_mean": agg_gen_tokens / max(float(n_rollouts_total), 1.0),
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
    logprob_viz_enabled: bool = False,
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

        # When per-token visualization is requested for this epoch's first
        # batch, capture the policy per-token logprobs computed inside the loss
        # closure. This is a pure read of tensors that already exist -- no extra
        # inference is triggered.
        capture_viz = logprob_viz_enabled and batch_idx == 0
        viz_capture: dict[str, list[torch.Tensor]] = {}

        def dpo_loss_fn(
            data: list[tinker.Datum], logprobs_list: list[torch.Tensor]
        ) -> tuple[torch.Tensor, dict[str, float]]:
            chosen_logprob_seqs = [logprobs_list[i] for i in range(0, len(data), 2)]
            rejected_logprob_seqs = [logprobs_list[i] for i in range(1, len(data), 2)]

            if capture_viz:
                viz_capture["chosen"] = [
                    s.detach().float().cpu() for s in chosen_logprob_seqs
                ]
                viz_capture["rejected"] = [
                    s.detach().float().cpu() for s in rejected_logprob_seqs
                ]

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

        if capture_viz and "rejected" in viz_capture:
            try:
                viz_dir = Path(log_path) / "logprob_viz"
                html_path = logprob_viz.render_epoch(
                    chosen_data=chosen_data,
                    rejected_data=rejected_data,
                    chosen_lp_seqs=viz_capture.get("chosen", []),
                    rejected_lp_seqs=viz_capture.get("rejected", []),
                    tokenizer=tokenizer,
                    epoch=epoch_idx,
                    out_dir=viz_dir,
                )
                logger.info("Wrote token log-prob visualization: %s", html_path)
            except Exception:
                logger.warning(
                    "Failed to render token log-prob visualization for epoch %d",
                    epoch_idx,
                    exc_info=True,
                )

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
    parser.add_argument(
        "--logprob-viz", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "Write a per-epoch HTML visualization coloring every token of each "
            "chosen and rejected sample in the epoch's first batch by its policy "
            "log probability. Reuses logprobs already computed in the DPO loss "
            "(no extra inference). Use --no-logprob-viz to disable."
        ),
    )
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
        choices=["adjacent", "first_last"],
        default="first_last",
        help=(
            "DPO preference pair construction. Both modes scan the whole session "
            "by output filename (cross-unit). "
            "'first_last' (default): one pair per file, first write vs last write. "
            "'adjacent': one pair per consecutive version step per file."
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
    parser.add_argument(
        "--agentic-rollout", action="store_true",
        help=(
            "Enable agentic online DPO: the rejected side is produced by a full "
            "multi-turn tool-using rollout in a sandbox (read/write/edit/grep/find/"
            "ls/bash). The student's final artifact (matched to the chosen file by "
            "basename) becomes the rejected write; the session's final accepted "
            "artifact remains chosen. Only the artifact write is trained on. "
            "Requires Tinker sampling (not compatible with --use-skyrl)."
        ),
    )
    parser.add_argument(
        "--agentic-policy-mode",
        choices=["on_policy", "off_policy"],
        default="on_policy",
        help=(
            "Rejected-rollout policy for agentic DPO. 'on_policy' (default) "
            "refreshes the sampler after every step so negatives track the "
            "current policy. 'off_policy' keeps the frozen initial snapshot "
            "(== reference model) for the whole run, using rollouts as a "
            "one-shot negative-augmentation source (cheaper, fully cacheable)."
        ),
    )
    parser.add_argument(
        "--agentic-num-rollouts", type=int, default=1,
        help=(
            "Number of independent rejected rollouts to sample per session, all "
            "dispatched in parallel. Each chosen artifact pairs with every "
            "matching rollout, yielding up to K*num_rollouts pairs per session."
        ),
    )
    parser.add_argument(
        "--agentic-pairs-per-session", type=int, default=1,
        help=(
            "Preference pairs each selected session contributes to a batch. "
            "--batch-size is the target #pairs, so sessions-per-batch = "
            "batch_size // pairs_per_session. A session producing more pairs is "
            "subsampled; one producing fewer is oversampled (cycled)."
        ),
    )
    parser.add_argument(
        "--agentic-session-reserve", type=int, default=0,
        help=(
            "Extra sessions fetched per batch (beyond sessions-per-batch) so the "
            "sampler can replace sessions whose rollouts yield zero valid pairs."
        ),
    )
    parser.add_argument(
        "--agentic-max-concurrent-rollouts", type=int, default=0,
        help=(
            "Max rollouts running at once (0 = unlimited). Each in-flight rollout "
            "holds a sandbox + growing prompt in memory, so lower this to cap RSS."
        ),
    )
    parser.add_argument(
        "--agentic-match-min-similarity", type=float, default=0.05,
        help=(
            "Min content-similarity (0..1) to match a student file to a chosen "
            "artifact by CONTENT/type instead of filename. Lower => more (looser) "
            "pairs; higher => fewer but tighter. Single same-extension candidates "
            "and single-artifact/single-file rollouts bypass this floor."
        ),
    )
    parser.add_argument(
        "--agentic-include-first-last", action="store_true",
        help=(
            "Additionally inject each session's first-written artifact version "
            "as a synthetic rejected snapshot (the offline first_last pair), "
            "alongside the on-policy model rollouts."
        ),
    )
    parser.add_argument("--rollout-max-turns", type=int, default=48,
                        help="Overall safety ceiling on assistant turns per agentic episode.")
    parser.add_argument("--max-turns-per-step", type=int, default=8,
                        help="Inner agent-loop cap within a single planned step.")
    parser.add_argument("--max-steps-per-episode", type=int, default=6,
                        help="Max planned (leaf) steps replayed as 'Proceed with' turns.")
    parser.add_argument(
        "--enable-bash", action=argparse.BooleanOptionalAction, default=True,
        help="Allow the bash tool during agentic rollout (runs on the host).",
    )
    parser.add_argument("--tool-timeout-s", type=int, default=20,
                        help="Per-bash-command timeout during agentic rollout.")
    parser.add_argument("--max-trajectory-tokens", type=int, default=None,
                        help="Stop an agentic rollout near this many prompt tokens (null = no cap).")
    parser.add_argument(
        "--min-artifact-versions", type=int, default=1,
        help=(
            "Only include files with at least this many versions as chosen artifacts. "
            "Set to 2 to restrict to files actually revised after user follow-ups."
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
    if args.agentic_rollout:
        if args.use_skyrl:
            raise ValueError("--agentic-rollout requires Tinker sampling; do not use --use-skyrl")
        online_renderer = renderers.get_renderer(
            args.renderer_name,
            tokenizer=get_tokenizer(args.model_name),
        )
        dataset_builder = None
        dataset = OnlineDPOAgenticDataset.from_weight_json(
            path=args.train_path,
            renderer=online_renderer,
            max_length=args.max_length,
            batch_size=args.batch_size,
            pairs_per_session=args.agentic_pairs_per_session,
            session_reserve=args.agentic_session_reserve,
            min_versions=args.min_artifact_versions,
            artifact_only_instruction=args.artifact_only_rollout_instruction,
        )
    elif args.online_rollout:
        if args.use_skyrl:
            raise ValueError("--online-rollout requires Tinker sampling; do not use --use-skyrl")
        online_renderer = renderers.get_renderer(
            args.renderer_name,
            tokenizer=get_tokenizer(args.model_name),
        )
        dataset_builder = None
        dataset = OnlineDPOAcceptedDataset.from_weight_json(
            path=args.train_path,
            renderer=online_renderer,
            max_length=args.max_length,
            batch_size=args.batch_size,
            artifact_only_instruction=args.artifact_only_rollout_instruction,
        )
    else:
        dataset_builder = WeightDPODataBuilder(
            train_path=args.train_path,
            test_path=args.test_path,
            pair_mode=args.pair_mode,
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
            if args.agentic_rollout:
                assert online_renderer is not None
                rows = dataset.get_batch(batch_idx)
                data_override, extra_metrics = asyncio.run(
                    _sample_agentic_dpo_pairs_async(
                        rows,
                        online_renderer,
                        sampling_client,
                        num_rollouts=args.agentic_num_rollouts,
                        pairs_per_session=max(1, args.agentic_pairs_per_session),
                        target_sessions=max(
                            1, args.batch_size // max(1, args.agentic_pairs_per_session)
                        ),
                        max_concurrent_rollouts=args.agentic_max_concurrent_rollouts,
                        match_min_similarity=args.agentic_match_min_similarity,
                        include_first_last=args.agentic_include_first_last,
                        max_tokens=args.rollout_max_tokens,
                        temperature=args.rollout_temperature,
                        max_turns=args.rollout_max_turns,
                        max_turns_per_step=args.max_turns_per_step,
                        max_steps=args.max_steps_per_episode,
                        enable_bash=args.enable_bash,
                        tool_timeout_s=args.tool_timeout_s,
                        max_trajectory_tokens=args.max_trajectory_tokens,
                        max_length=args.max_length,
                        step=step,
                        sample_log_path=(
                            Path(log_path) / "dpo_agentic_rollout_samples.jsonl"
                            if not args.no_log_rollout_samples else None
                        ),
                        sample_log_chars=args.rollout_sample_log_chars,
                        finish_log_path=Path(log_path) / "dpo_agentic_rollout_finished.jsonl",
                        raw_trajectory_log_path=(
                            Path(log_path) / "dpo_agentic_rollout_trajectories.jsonl"
                            if not args.no_log_rollout_samples else None
                        ),
                    )
                )
                if not data_override:
                    metrics = {
                        "epoch": epoch_idx,
                        "dpo_online/no_valid_batch": 1.0,
                        "progress": step / max(total_steps, 1),
                        **(extra_metrics or {}),
                    }
                    ml_logger.log_metrics(metrics=metrics, step=step)
                    logger.warning(
                        "Skipping DPO step %d: no valid agentic artifact pairs (filter_rate=%.3f)",
                        step,
                        (extra_metrics or {}).get("dpo_online/filter_rate", 0.0),
                    )
                    continue
            elif args.online_rollout:
                assert online_renderer is not None
                rows = dataset.get_batch(batch_idx)
                data_override, extra_metrics = asyncio.run(
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
                if not data_override:
                    metrics = {
                        "epoch": epoch_idx,
                        "dpo_online/no_valid_batch": 1.0,
                        "progress": step / max(total_steps, 1),
                        **(extra_metrics or {}),
                    }
                    ml_logger.log_metrics(metrics=metrics, step=step)
                    logger.warning(
                        "Skipping DPO step %d: no valid online artifact pairs (filter_rate=%.3f)",
                        step,
                        (extra_metrics or {}).get("dpo_online/filter_rate", 0.0),
                    )
                    continue
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
                logprob_viz_enabled=args.logprob_viz,
            )
            # Refresh the on-policy sampler after each step. Off-policy agentic
            # DPO intentionally keeps the frozen initial snapshot for the whole
            # run (rollouts as a one-shot negative-augmentation source).
            refresh_sampler = args.online_rollout or (
                args.agentic_rollout and args.agentic_policy_mode != "off_policy"
            )
            if refresh_sampler:
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
