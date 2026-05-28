"""Tinker DatasetBuilders for weight-format session JSON.

Reads weight-format JSON directly; no reverse parsing via ``traj_to_chat()``
needed.

Three builders:
- ``WeightDPODataBuilder``       → chosen/rejected Datum pairs for DPO
- ``WeightReinforceDataBuilder`` → Datums with per-trajectory rewards
- ``OfflineOPDDataset``          → student/teacher Datum pairs for offline OPD
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import chz
import datasets
import tinker

from tinker_cookbook import renderers
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.renderers.base import ToolCall
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset

try:  # Supports both `python -m weight...` from scripts/ and `python -m scripts.weight...`.
    from weight.data.extract import (  # type: ignore[import-not-found]  # noqa: E402
        extract_dpo_pairs,
        extract_opd_examples,
        extract_opd_examples_v2,
        extract_reinforce_examples,
    )
except ModuleNotFoundError:  # pragma: no cover - depends on invocation cwd
    from ..data.extract import (  # noqa: E402
        extract_dpo_pairs,
        extract_opd_examples,
        extract_opd_examples_v2,
        extract_reinforce_examples,
    )

logger = logging.getLogger(__name__)

# When True, assistant top-level ``thinking`` is not merged into content; it is
# dropped so training matches non-thinking / instruct-style prompts.
OMIT_ASSISTANT_THINKING = True


# ---------------------------------------------------------------------------
# Tool-call hydration (shared)
# ---------------------------------------------------------------------------

def _hydrate_tool_calls(conversation: list[dict]) -> list[dict]:
    """Convert plain-dict tool_calls to ToolCall pydantic objects for the renderer.

    Also:
    - Normalises ``content: null`` → ``content: ""`` for messages without thinking.
    - Converts the top-level ``thinking`` field on assistant messages into the
      list-of-parts format that Qwen3Renderer understands::

          content: [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "..."}]

      This ensures the renderer encodes ``<think>...</think>`` tokens for the
      *current* turn while still stripping thinking from historical turns per the
      default ``strip_thinking_from_history=True`` behaviour.

      If ``thinking`` is absent or empty (e.g. Qwen3-Instruct non-thinking mode)
      the message is left unchanged so the code gracefully supports both modes.

      If module flag ``OMIT_ASSISTANT_THINKING`` is True, top-level ``thinking`` is
      ignored (same as absent): only ``content`` is kept for the assistant turn.
    """
    out = []
    for msg in conversation:
        thinking_text = msg.get("thinking") if msg.get("role") == "assistant" else None

        if thinking_text and not OMIT_ASSISTANT_THINKING:
            # Build list-of-parts content so Qwen3Renderer sees <think>...</think>
            parts: list[dict] = [{"type": "thinking", "thinking": thinking_text}]
            text = msg.get("content") or ""
            if text:
                parts.append({"type": "text", "text": text})
            # Rebuild msg without the original content/thinking keys
            msg = {k: v for k, v in msg.items() if k not in ("content", "thinking")}
            msg = {"content": parts, **msg}
        else:
            # Normalise null content (common when a message only has tool_calls)
            if msg.get("content") is None:
                msg = {**msg, "content": ""}
            # Drop empty/residual thinking key so downstream code doesn't see it
            if "thinking" in msg:
                msg = {k: v for k, v in msg.items() if k != "thinking"}

        if msg.get("tool_calls"):
            msg = {
                **msg,
                "tool_calls": [
                    ToolCall(
                        id=tc.get("id"),
                        function=ToolCall.FunctionBody(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        ),
                    )
                    for tc in msg["tool_calls"]
                ],
            }
        elif "tool_calls" in msg:
            msg = {k: v for k, v in msg.items() if k != "tool_calls"}
        out.append(msg)
    return out


def _load_json_root(path: str) -> Any:
    """Load the full JSON document from disk (list of sessions, or wrapper dict)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sessions_from_json_root(root: Any) -> list[dict]:
    """Same session list resolution as ``_load_sessions``, without reading a file."""
    if isinstance(root, list):
        return root
    if isinstance(root, dict) and "sessions" in root:
        return root["sessions"]
    return [root]


def _load_sessions(path: str) -> list[dict]:
    return _sessions_from_json_root(_load_json_root(path))


def _save_sessions_json(path: str, root: Any) -> None:
    """Write weight JSON back; preserves top-level shape (array vs ``{"sessions":...}``)."""
    text = json.dumps(root, indent=2, ensure_ascii=False) + "\n"
    Path(path).write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------

@chz.chz
class WeightDPODataBuilder(ChatDatasetBuilder):
    """Build interleaved chosen/rejected Datum pairs from weight-format JSON."""

    train_path: str
    test_path: str | None = None
    pair_mode: str = "adjacent"

    def _load(self, path: str) -> list[dict]:
        sessions = _sessions_from_json_root(_load_json_root(path))
        # Pass the renderer so tool schemas are injected into the system prefix
        # via ``Renderer.create_conversation_prefix_with_tools`` rather than
        # hand-rolled into the system prompt text. Falls back to plain system
        # prompt when the renderer doesn't support tools (e.g. RoleColon).
        return extract_dpo_pairs(
            sessions, renderer=self.renderer, pair_mode=self.pair_mode,
        )

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        train_rows = self._load(self.train_path)
        logger.info(
            "Loaded %d DPO pairs from %s (pair_mode=%s)",
            len(train_rows), self.train_path, self.pair_mode,
        )
        train_ds = datasets.Dataset.from_list(train_rows)

        def flatmap_fn(row: dict) -> list:
            chosen_convo = _hydrate_tool_calls(row["prompt"] + row["chosen"])
            rejected_convo = _hydrate_tool_calls(row["prompt"] + row["rejected"])
            # LAST_ASSISTANT_MESSAGE: only the final artifact write is masked
            # for the loss. Historical assistant turns in the prompt context
            # cancel exactly in the DPO log-ratio (chosen / rejected share the
            # prompt token-for-token), so masking them out is mathematically
            # equivalent to ALL_ASSISTANT_MESSAGES while satisfying the
            # qwen3_5 renderer's extension property and matching Tinker default.
            chosen_datum = conversation_to_datum(
                chosen_convo, self.renderer, self.common_config.max_length,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )
            rejected_datum = conversation_to_datum(
                rejected_convo, self.renderer, self.common_config.max_length,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )
            return [chosen_datum, rejected_datum]

        train_dataset = SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.common_config.batch_size, flatmap_fn=flatmap_fn,
        )

        test_dataset = None
        if self.test_path is not None:
            test_rows = self._load(self.test_path)
            test_ds = datasets.Dataset.from_list(test_rows)
            test_dataset = SupervisedDatasetFromHFDataset(
                test_ds, batch_size=len(test_ds), flatmap_fn=flatmap_fn,
            )

        return train_dataset, test_dataset


# ---------------------------------------------------------------------------
# REINFORCE
# ---------------------------------------------------------------------------

class _ReinforceDataset:
    """Supervised dataset that keeps per-trajectory reward metadata aligned."""

    def __init__(self, datums: list, rewards: list[float], batch_size: int):
        assert len(datums) == len(rewards)
        self._datums = datums
        self._rewards = rewards
        self._batch_size = batch_size
        self._indices = list(range(len(datums)))

    def __len__(self) -> int:
        if not self._datums:
            return 0
        return (len(self._datums) + self._batch_size - 1) // self._batch_size

    def set_epoch(self, seed: int) -> None:
        rng = random.Random(seed)
        self._indices = list(range(len(self._datums)))
        rng.shuffle(self._indices)

    def get_batch(self, index: int) -> list:
        start = index * self._batch_size
        end = min(start + self._batch_size, len(self._indices))
        return [self._datums[self._indices[i]] for i in range(start, end)]

    def get_batch_rewards(self, index: int) -> list[float]:
        start = index * self._batch_size
        end = min(start + self._batch_size, len(self._indices))
        return [self._rewards[self._indices[i]] for i in range(start, end)]


@chz.chz
class WeightReinforceDataBuilder(ChatDatasetBuilder):
    """Build Datums with per-trajectory rewards from weight-format JSON."""

    train_path: str
    test_path: str | None = None

    def _build_dataset(self, path: str, batch_size: int) -> _ReinforceDataset:
        root = _load_json_root(path)
        sessions = _sessions_from_json_root(root)
        examples, session_dirty = extract_reinforce_examples(sessions, renderer=self.renderer)
        logger.info("Loaded %d REINFORCE examples from %s", len(examples), path)
        if session_dirty:
            try:
                _save_sessions_json(path, root)
                logger.info("Wrote REINFORCE cache (reward and/or reinforce_prompt) to %s", path)
            except OSError as e:
                logger.warning("Could not persist REINFORCE session cache to %s: %s", path, e)

        datums: list[tinker.Datum] = []
        rewards: list[float] = []
        for ex in examples:
            conversation = _hydrate_tool_calls(ex["prompt"] + ex["completion"])
            # LAST_ASSISTANT_MESSAGE: train only on the final artifact write
            # (the action whose reward we computed). Historical assistant turns
            # in the prompt come from prior rounds and were not sampled under
            # the current reward signal, so they shouldn't contribute to the
            # REINFORCE gradient.
            datum = conversation_to_datum(
                conversation, self.renderer, self.common_config.max_length,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )
            datums.append(datum)
            rewards.append(ex["reward"])
        return _ReinforceDataset(datums=datums, rewards=rewards, batch_size=batch_size)

    def __call__(self) -> tuple[_ReinforceDataset, _ReinforceDataset | None]:
        train = self._build_dataset(self.train_path, self.common_config.batch_size)
        test = None
        if self.test_path is not None:
            test = self._build_dataset(self.test_path, self.common_config.batch_size)
        return train, test


# ---------------------------------------------------------------------------
# Offline OPD
# ---------------------------------------------------------------------------

class OfflineOPDDataset:
    """Dataset for offline OPD: pre-built student/teacher Datum pairs.

    Each batch returns ``(student_datums, teacher_datums)`` where the teacher
    datums share the same completion tokens but have an augmented prompt prefix
    containing privileged human feedback (and optionally the ground-truth
    artifact when ``use_gt=True``).

    Additionally stores ``teacher_prompt_inputs`` — tokenized teacher prompts
    WITHOUT the completion appended — for the top-K KD path in
    :mod:`run_opd`. These are built by calling
    ``renderer.build_generation_prompt`` on just the teacher prompt messages.
    """

    def __init__(
        self,
        student_datums: list[tinker.Datum],
        teacher_datums: list[tinker.Datum],
        teacher_prompt_inputs: list[tinker.ModelInput],
        batch_size: int,
    ):
        assert len(student_datums) == len(teacher_datums) == len(teacher_prompt_inputs)
        self._student = student_datums
        self._teacher = teacher_datums
        self._teacher_prompt_inputs = teacher_prompt_inputs
        self._batch_size = batch_size
        self._indices = list(range(len(student_datums)))

    @classmethod
    def from_weight_json(
        cls,
        path: str,
        renderer: renderers.Renderer,
        max_length: int | None,
        batch_size: int,
        pair_mode: str = "first_last",
        use_gt: bool = True,
        use_student: bool = True,
        extract_version: str = "v2",
    ) -> "OfflineOPDDataset":
        sessions = _load_sessions(path)
        extract_fn = (
            extract_opd_examples_v2 if extract_version == "v2"
            else extract_opd_examples
        )
        examples = extract_fn(
            sessions,
            renderer=renderer,
            pair_mode=pair_mode,
            use_gt=use_gt,
            use_student=use_student,
        )
        logger.info(
            "Loaded %d OPD examples from %s "
            "(extract=%s, pair_mode=%s, use_gt=%s, use_student=%s)",
            len(examples), path, extract_version, pair_mode, use_gt, use_student,
        )

        student_datums: list[tinker.Datum] = []
        teacher_datums: list[tinker.Datum] = []
        teacher_prompt_inputs: list[tinker.ModelInput] = []

        for ex in examples:
            student_convo = _hydrate_tool_calls(ex["student_prompt"] + ex["completion"])
            teacher_convo = _hydrate_tool_calls(ex["teacher_prompt"] + ex["completion"])

            # LAST_ASSISTANT_MESSAGE: critical here because student vs teacher
            # prompts differ (teacher has privileged human feedback appended),
            # so historical assistant turns render to *different* token prefixes
            # in the two datums. With ALL_ASSISTANT_MESSAGES the prior-turn
            # logprobs would contaminate the OPD KL on tokens that aren't
            # the actual student/teacher action. LAST scopes the loss to the
            # one artifact write, which is what we actually want to align.
            student_datum = conversation_to_datum(
                student_convo, renderer, max_length,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )
            teacher_datum = conversation_to_datum(
                teacher_convo, renderer, max_length,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )
            # Build teacher prompt ModelInput (no completion) for top-K KD.
            # build_generation_prompt produces a left-pad-ready ModelInput from
            # the teacher prompt messages only; completion tokens are appended
            # per-datum by the top-K pre-computation step in run_opd.
            teacher_prompt_input = renderer.build_generation_prompt(
                _hydrate_tool_calls(ex["teacher_prompt"])
            )

            student_datums.append(student_datum)
            teacher_datums.append(teacher_datum)
            teacher_prompt_inputs.append(teacher_prompt_input)

        return cls(student_datums, teacher_datums, teacher_prompt_inputs, batch_size)

    def __len__(self) -> int:
        if not self._student:
            return 0
        return (len(self._student) + self._batch_size - 1) // self._batch_size

    def set_epoch(self, seed: int) -> None:
        rng = random.Random(seed)
        self._indices = list(range(len(self._student)))
        rng.shuffle(self._indices)

    def get_batch(
        self, index: int,
    ) -> tuple[list[tinker.Datum], list[tinker.Datum]]:
        start = index * self._batch_size
        end = min(start + self._batch_size, len(self._indices))
        s = [self._student[self._indices[i]] for i in range(start, end)]
        t = [self._teacher[self._indices[i]] for i in range(start, end)]
        return s, t

    def get_all_teacher_prompt_inputs(self) -> list[tinker.ModelInput]:
        """Return all teacher prompt ModelInputs in dataset order (not shuffled).

        Used by the top-K pre-computation step in run_opd to build teacher-forced
        sequences. Note: these are in original (un-shuffled) order; the mapping
        back to student datums uses self._indices set by set_epoch.
        """
        return list(self._teacher_prompt_inputs)

    def get_batch_teacher_prompt_inputs(
        self, index: int,
    ) -> list[tinker.ModelInput]:
        """Return teacher prompt inputs for the current batch (respects set_epoch shuffle)."""
        start = index * self._batch_size
        end = min(start + self._batch_size, len(self._indices))
        return [self._teacher_prompt_inputs[self._indices[i]] for i in range(start, end)]
