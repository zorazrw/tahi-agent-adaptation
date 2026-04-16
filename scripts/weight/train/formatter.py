"""Tinker DatasetBuilders for weight-format session JSON.

Replaces ``tinker_formatter.py`` — reads weight-format JSON directly,
no reverse parsing via ``traj_to_chat()`` needed.

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

_weight_data = Path(__file__).resolve().parent.parent / "data"
if str(_weight_data) not in sys.path:
    sys.path.insert(0, str(_weight_data))

from extract import extract_dpo_pairs, extract_opd_examples, extract_reinforce_examples  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool-call hydration (shared)
# ---------------------------------------------------------------------------

def _hydrate_tool_calls(conversation: list[dict]) -> list[dict]:
    """Convert plain-dict tool_calls to ToolCall pydantic objects for the renderer."""
    out = []
    for msg in conversation:
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


def _load_sessions(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "sessions" in raw:
        return raw["sessions"]
    return [raw]


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------

@chz.chz
class WeightDPODataBuilder(ChatDatasetBuilder):
    """Build interleaved chosen/rejected Datum pairs from weight-format JSON."""

    train_path: str
    test_path: str | None = None

    def _load(self, path: str) -> list[dict]:
        sessions = _load_sessions(path)
        return extract_dpo_pairs(sessions)

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        train_rows = self._load(self.train_path)
        logger.info("Loaded %d DPO pairs from %s", len(train_rows), self.train_path)
        train_ds = datasets.Dataset.from_list(train_rows)

        def flatmap_fn(row: dict) -> list:
            chosen_convo = _hydrate_tool_calls(row["prompt"] + row["chosen"])
            rejected_convo = _hydrate_tool_calls(row["prompt"] + row["rejected"])
            chosen_datum = conversation_to_datum(
                chosen_convo, self.renderer, self.common_config.max_length,
                train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            )
            rejected_datum = conversation_to_datum(
                rejected_convo, self.renderer, self.common_config.max_length,
                train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
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
        sessions = _load_sessions(path)
        examples = extract_reinforce_examples(sessions)
        logger.info("Loaded %d REINFORCE examples from %s", len(examples), path)

        datums: list[tinker.Datum] = []
        rewards: list[float] = []
        for ex in examples:
            conversation = _hydrate_tool_calls(ex["prompt"] + ex["completion"])
            datum = conversation_to_datum(
                conversation, self.renderer, self.common_config.max_length,
                train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
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

    Each batch returns ``(student_datums, teacher_model_inputs)`` where
    the teacher model inputs share completion tokens with the student but
    have a different (augmented) prompt prefix.
    """

    def __init__(
        self,
        student_datums: list[tinker.Datum],
        teacher_datums: list[tinker.Datum],
        batch_size: int,
    ):
        assert len(student_datums) == len(teacher_datums)
        self._student = student_datums
        self._teacher = teacher_datums
        self._batch_size = batch_size
        self._indices = list(range(len(student_datums)))

    @classmethod
    def from_weight_json(
        cls,
        path: str,
        renderer: renderers.Renderer,
        max_length: int | None,
        batch_size: int,
    ) -> "OfflineOPDDataset":
        sessions = _load_sessions(path)
        examples = extract_opd_examples(sessions)
        logger.info("Loaded %d OPD examples from %s", len(examples), path)

        student_datums: list[tinker.Datum] = []
        teacher_datums: list[tinker.Datum] = []

        for ex in examples:
            student_convo = _hydrate_tool_calls(ex["student_prompt"] + ex["completion"])
            teacher_convo = _hydrate_tool_calls(ex["teacher_prompt"] + ex["completion"])

            student_datum = conversation_to_datum(
                student_convo, renderer, max_length,
                train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            )
            teacher_datum = conversation_to_datum(
                teacher_convo, renderer, max_length,
                train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            )
            student_datums.append(student_datum)
            teacher_datums.append(teacher_datum)

        return cls(student_datums, teacher_datums, batch_size)

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
