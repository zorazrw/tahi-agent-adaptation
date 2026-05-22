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
from tinker_cookbook.rl.types import EnvGroupBuilder, StepResult, Sequence
from tinker_cookbook.third_party.openai_compat import openai_tools_to_tinker

# from weight.data.extract import (  # noqa: E402
#     extract_dpo_pairs,
#     extract_reinforce_examples,
# )
# TODO: Update this
extract_dpo_pairs = None
extract_reinforce_examples = None

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
    pair_mode: str = "adjacent"

    def _load(self, path: str) -> list[dict]:
        sessions = _load_sessions(path)
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
        sessions = _load_sessions(path)
        examples = extract_reinforce_examples(sessions, renderer=self.renderer)
        logger.info("Loaded %d REINFORCE examples from %s", len(examples), path)

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
# OPD
# ---------------------------------------------------------------------------
def _prepare_messages_with_tools(
    renderer: renderers.Renderer,
    messages: list[dict],
    tool_schemas: list[dict] | None,
) -> list[renderers.Message]:
    """Inject OpenAI-format tool schemas using the renderer-native tool prefix."""
    normalized: list[renderers.Message] = _hydrate_tool_calls([dict(msg) for msg in messages])  # type: ignore[list-item]
    if not tool_schemas:
        return normalized

    tool_specs = openai_tools_to_tinker(tool_schemas)
    system_prompt = ""
    remaining = normalized
    if normalized and normalized[0].get("role") == "system":
        content = normalized[0].get("content") or ""
        system_prompt = content if isinstance(content, str) else ""
        remaining = normalized[1:]

    return renderer.create_conversation_prefix_with_tools(tool_specs, system_prompt) + remaining


def _opd_build_unit(
    unit: dict, system_prompt: str | None, tool_schemas: list[dict] | None
) -> dict | None:
    """Convert a learning unit into ``{student_prompt, golden_answer, tool_schemas}``.

    Prefers an LLM-generated ``summary`` (string) as the golden response, which
    represents the user's final intent across all remaining follow-up actions.
    Falls back to the original ``response_messages`` (one or more
    assistant/tool turns produced after the trigger) when no ``summary`` is
    present.

    The leading ``user_messages`` (typically the task instruction for
    ``sub_index=0`` units) are appended to ``student_prompt`` so the student
    rollout actually sees what it's being asked to do. ``golden_answer``
    therefore contains only the assistant turn(s); the teacher prompt assembled
    by :func:`build_sdft_teacher_prompt` is ``student_prompt + golden_answer +
    [redo]``, which still ends up as ``system + history + user + golden +
    redo`` -- structurally identical to the previous formulation, but now the
    student is conditioned on the same user message the teacher sees.

    Continuation sub-units (one assistant turn after a tool result) are
    skipped: their student prompt would end with a tool result rather than a
    user request, the teacher prompt's user-request relocation has nothing to
    relocate, and the gradient signal at those positions is dominated by
    template tokens that the student already matches.
    """
    if unit.get("is_continuation"):
        return None
    history = unit.get("history") or []
    user_messages = [m for m in (unit.get("user_messages") or []) if m]
    if not user_messages and history:
        return None
    summary = unit.get("summary")

    if isinstance(summary, str) and summary.strip():
        teacher_demo: list[dict] = [
            {"role": "assistant", "content": summary.strip()}
        ]
    else:
        followup_actions = unit.get("followup_actions") or []
        followup_texts = [
            a["prompt"].strip()
            for a in followup_actions
            if isinstance(a, dict)
            and a.get("type") == "follow_up"
            and isinstance(a.get("prompt"), str)
            and a["prompt"].strip()
        ]
        teacher_demo: list[dict] = [
            {"role": "user", "content": text} for text in followup_texts
        ]

    if not teacher_demo:
        return None

    # A unit must have at least *some* signal beyond the system prompt: either
    # a user message that anchors the prompt, an assistant response in the
    # demo, or a non-empty history that pins the prediction position.
    if not user_messages and not history and not any(
        m.get("role") == "assistant" for m in teacher_demo
    ):
        return None

    student_prompt: list[dict] = []
    if system_prompt:
        student_prompt.append({"role": "system", "content": system_prompt})
    student_prompt.extend(history)
    student_prompt.extend({"role": "user", "content": um} for um in user_messages)

    return {
        "student_prompt": student_prompt,
        "golden_answer": teacher_demo,   # used by build_sdft_teacher_prompt
        "tool_schemas": tool_schemas,
        "uses_summary": isinstance(summary, str) and bool(summary.strip()),
    }
    

class _OPDEnvGroupBuilder(EnvGroupBuilder):
    """Minimal prompt-only ``EnvGroupBuilder`` for SDFT.

    Fallback used when ``tinker_cookbook.recipes.sdft.datasets`` is not
    installed.  Creates single-turn, zero-reward environments that
    present the student prompt for on-policy generation.
    """

    def __init__(
        self,
        prompt: object,
        stop_condition: object,
        group_size: int = 1,
    ):
        self._prompt = prompt
        self._stop_condition = stop_condition
        self._group_size = group_size

    async def make_envs(self) -> Sequence:
        envs = []
        for _ in range(self._group_size):
            envs.append(_PromptOnlyEnv(self._prompt, self._stop_condition))
        return envs

    def logging_tags(self) -> list[str]:
        return ["opd"]


class _PromptOnlyEnv:
    """Minimal single-turn environment that yields zero reward.

    Provides an initial ``ModelInput`` prompt and immediately terminates
    the episode on the first step.  Compatible with the ``Env`` protocol
    from ``tinker_cookbook.rl.types``.
    """

    def __init__(self, prompt: object, stop_condition: object):
        self._prompt = prompt
        self._stop_condition = stop_condition

    async def initial_observation(self) -> tuple[object, object]:
        return self._prompt, self._stop_condition

    async def step(self, action: object, *, extra: object | None = None) -> StepResult:
        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=self._prompt,  # Unused after terminal transition.
            next_stop_condition=self._stop_condition,
            metrics={},
            logs={},
        )
        
        
@chz.chz
class OPDDataBuilder:

    train_path: str
    test_path: str | None = None

    def _load_units(self, raw: object) -> list[dict]:
        """Support both single-session and multi-session OPD export shapes."""
        if not isinstance(raw, dict):
            return []
        if isinstance(raw.get("learning_units"), list):
            return [u for u in raw["learning_units"] if isinstance(u, dict)]
        if isinstance(raw.get("sessions"), list):
            units: list[dict] = []
            for sess in raw["sessions"]:
                if not isinstance(sess, dict):
                    continue
                lu = sess.get("learning_units")
                if isinstance(lu, list):
                    units.extend(u for u in lu if isinstance(u, dict))
            return units
        return []

    @staticmethod
    def _build_unit(unit: dict, system_prompt: str | None, tool_schemas: list[dict] | None) -> dict | None:
        return _opd_build_unit(unit, system_prompt, tool_schemas)

    def _load(self, path: str) -> list[dict]:
        with open(path, "r") as f:
            raw = json.load(f)
        rows: list[dict] = []
        if isinstance(raw, dict) and isinstance(raw.get("sessions"), list):
            for session in raw["sessions"]:
                if not isinstance(session, dict):
                    continue
                system_prompt = session.get("system_prompt")
                tool_schemas = session.get("tool_schemas")
                for unit in self._load_units(session):
                    row = self._build_unit(unit, system_prompt, tool_schemas)
                    if row is not None:
                        rows.append(row)
            return rows

        system_prompt = raw.get("system_prompt") if isinstance(raw, dict) else None
        tool_schemas = raw.get("tool_schemas") if isinstance(raw, dict) else None
        for unit in self._load_units(raw):
            row = self._build_unit(unit, system_prompt, tool_schemas)
            if row is not None:
                rows.append(row)

        return rows

    def __call__(
        self,
        renderer: renderers.Renderer,
        batch_size: int = 4,
        group_size: int = 1,
    ) -> "OPDSDFTDataset":
        rows = self._load(self.train_path)
        questions = [r["student_prompt"] for r in rows]
        golden_answers = [r["golden_answer"] for r in rows]
        tool_schemas_by_question = [r.get("tool_schemas") for r in rows]
        return OPDSDFTDataset(
            questions=questions,
            golden_answers=golden_answers,
            renderer=renderer,
            tool_schemas_by_question=tool_schemas_by_question,
            batch_size=batch_size,
            group_size=group_size,
        )


class OPDSDFTDataset:
    """SDFT batch provider for on-policy distillation using OPD data.

    Implements the ``SDFTBatchProvider`` protocol expected by
    :func:`tinker_opd.main`.  The *question* is the accumulated user
    messages and the *golden_answer* is the serialised human-correction
    trajectory.
    """

    def __init__(
        self,
        questions: list[list[dict]],  # list of OpenAI-style chat messages
        golden_answers: list[str],
        renderer: renderers.Renderer,
        tool_schemas: list[dict] | None = None,
        tool_schemas_by_question: list[list[dict] | None] | None = None,
        batch_size: int = 4,
        group_size: int = 1,
    ):
        if len(questions) != len(golden_answers):
            raise ValueError(
                f"questions ({len(questions)}) and golden_answers "
                f"({len(golden_answers)}) must have the same length"
            )
        self._questions = questions
        self._golden_answers = golden_answers
        self._renderer = renderer
        if tool_schemas_by_question is not None:
            if len(tool_schemas_by_question) != len(questions):
                raise ValueError(
                    "tool_schemas_by_question must have one entry per question"
                )
            self._tool_schemas_by_question = tool_schemas_by_question
        else:
            self._tool_schemas_by_question = [tool_schemas for _ in questions]
        self._batch_size = batch_size
        self._group_size = group_size
        # Permutation over rows, used by ``get_batch`` so ``set_epoch`` can
        # reshuffle the inter-session order without touching the underlying
        # row arrays. Defaults to identity (insertion order) so behavior is
        # unchanged for callers that never invoke ``set_epoch``.
        self._indices: list[int] = list(range(len(self._questions)))

    @classmethod
    def _load_units(cls, raw: object) -> list[dict]:
        """Support both single-session and multi-session OPD export shapes."""
        if not isinstance(raw, dict):
            return []
        if isinstance(raw.get("learning_units"), list):
            return [u for u in raw["learning_units"] if isinstance(u, dict)]
        if isinstance(raw.get("sessions"), list):
            units: list[dict] = []
            for sess in raw["sessions"]:
                if not isinstance(sess, dict):
                    continue
                lu = sess.get("learning_units")
                if isinstance(lu, list):
                    units.extend(u for u in lu if isinstance(u, dict))
            return units
        return []
    
    @staticmethod
    def _build_unit(
        unit: dict, system_prompt: str | None, tool_schemas: list[dict] | None
    ) -> dict | None:
        return _opd_build_unit(unit, system_prompt, tool_schemas)

    @classmethod
    def from_json(
        cls,
        data_path: str,
        renderer: renderers.Renderer,
        batch_size: int = 4,
        group_size: int = 1,
    ) -> "OPDSDFTDataset":
        """Load OPD data from a JSON file."""
        with open(data_path, "r") as f:
            raw = json.load(f)

        questions: list[list[dict]] = []
        golden_answers: list[list[dict]] = []
        tool_schemas_by_question: list[list[dict] | None] = []

        def _consume(unit: dict, system_prompt: str | None, tool_schemas: list[dict] | None) -> None:
            row = cls._build_unit(unit, system_prompt, tool_schemas)
            if row is None:
                return
            questions.append(row["student_prompt"])
            golden_answers.append(row["golden_answer"])
            tool_schemas_by_question.append(row.get("tool_schemas"))

        if isinstance(raw, dict) and isinstance(raw.get("sessions"), list):
            for session in raw["sessions"]:
                if not isinstance(session, dict):
                    continue
                system_prompt = session.get("system_prompt")
                tool_schemas = session.get("tool_schemas")
                for unit in cls._load_units(session):
                    _consume(unit, system_prompt, tool_schemas)
        else:
            system_prompt = raw.get("system_prompt") if isinstance(raw, dict) else None
            tool_schemas = raw.get("tool_schemas") if isinstance(raw, dict) else None
            for unit in cls._load_units(raw):
                _consume(unit, system_prompt, tool_schemas)

        return cls(
            questions=questions,
            golden_answers=golden_answers,
            renderer=renderer,
            tool_schemas_by_question=tool_schemas_by_question,
            batch_size=batch_size,
            group_size=group_size,
        )

    def _num_batch_slots(self) -> int:
        """Number of distinct batches covering ``_questions`` (at least 1 for empty data)."""
        return max(1, (len(self._questions) + self._batch_size - 1) // self._batch_size)

    def __len__(self) -> int:
        return self._num_batch_slots()

    def set_epoch(self, seed: int) -> None:
        """Reshuffle the row permutation used by ``get_batch``.

        Mirrors the convention used by :class:`ReinforceDataset` and
        ``SupervisedDataset`` implementations in tinker-cookbook. Callers
        (trainers) should invoke this once per epoch with a round+epoch
        derived seed so that:

        * batches within a round mix examples across sessions instead of
          slicing the row array in insertion order, and
        * repeated epochs over the same round don't replay an identical
          curriculum.

        The seed is fed into a local ``random.Random`` so global RNG state
        is untouched.
        """
        import random

        rng = random.Random(seed)
        self._indices = list(range(len(self._questions)))
        rng.shuffle(self._indices)

    def get_batch(
        self, index: int
    ) -> tuple[Sequence[EnvGroupBuilder], list[list[renderers.Message]], list[str]]:
        # Wrap so training can run multiple epochs (``tinker_opd`` uses ``index`` up to
        # ``len(dataset) * epochs - 1``) without empty batches past the first epoch.
        idx = index % self._num_batch_slots()
        start = idx * self._batch_size
        end = min(start + self._batch_size, len(self._indices))

        picks = self._indices[start:end]
        raw_batch_questions = [self._questions[i] for i in picks]
        batch_tool_schemas = [self._tool_schemas_by_question[i] for i in picks]
        batch_questions = [
            self._prepare_question(q, tool_schemas)
            for q, tool_schemas in zip(raw_batch_questions, batch_tool_schemas)
        ]
        batch_golden = [self._golden_answers[i] for i in picks]
        builders: list[EnvGroupBuilder] = [
            self._make_builder(q) for q in batch_questions
        ]
        return builders, batch_questions, batch_golden

    def _prepare_question(
        self, question: list[dict], tool_schemas: list[dict] | None
    ) -> list[renderers.Message]:
        return _prepare_messages_with_tools(self._renderer, question, tool_schemas)

    def _make_builder(self, question: list[renderers.Message]) -> EnvGroupBuilder:
        """Create an EnvGroupBuilder for a single student prompt.

        Uses a local builder because the cookbook's ``PromptOnlyEnv`` expects a
        string question, while OPD passes an already-rendered ``ModelInput``.
        """
        prompt = self._renderer.build_generation_prompt(question)

        return _OPDEnvGroupBuilder(
            prompt=prompt,
            stop_condition=self._renderer.get_stop_sequences(),
            group_size=self._group_size,
        )