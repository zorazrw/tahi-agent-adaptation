import json
import re
from collections.abc import Sequence

import chz
import datasets

from tinker_cookbook import renderers
from tinker_cookbook.renderers.base import ToolCall
from tinker_cookbook.rl.types import EnvGroupBuilder, StepResult
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset, conversation_to_datum
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset
from tinker_cookbook.third_party.openai_compat import openai_tools_to_tinker


_TOOL_NAME_RE = re.compile(r"^([A-Za-z_][\w]*)\(")


def _decode_message_like(action: str, prefix: str) -> str | None:
    """Parse ``message("...")`` / ``plan("...")`` bodies via JSON string rules."""
    head = prefix + "("
    if not (action.startswith(head) and action.endswith(")")):
        return None
    inner = action[len(head) : -1].strip()
    if len(inner) >= 2 and inner[0] == '"' == inner[-1]:
        try:
            return str(json.loads(inner))
        except json.JSONDecodeError:
            return None
    return inner or None


def _parse_tool_call(action: str) -> tuple[str, str] | None:
    """
    Extract OpenAI ``function.name`` and ``function.arguments`` from actions like
    ``Write({"file_path": "...", "content": "..."})`` or ``verify("uuid")``.
    Tool name comes from regex; arguments are the JSON object string (or a wrapped
    JSON object for scalar JSON literals).
    """
    m = _TOOL_NAME_RE.match(action)
    if not m:
        return None
    name = m.group(1)
    if name in ("message", "plan"):
        return None

    rest = action[m.end() :].lstrip()

    if rest.startswith("{"):
        try:
            _obj, end = json.JSONDecoder().raw_decode(rest)
        except json.JSONDecodeError:
            return None
        if rest[end:].strip() != ")":
            return None
        return name, rest[:end]

    if not rest.endswith(")"):
        return None
    inner = rest[:-1].strip()
    if not inner:
        return None
    try:
        val = json.loads(inner)
    except json.JSONDecodeError:
        return None
    if isinstance(val, dict):
        return name, json.dumps(val, ensure_ascii=False)
    return name, json.dumps({"arguments": val}, ensure_ascii=False)


def traj_to_chat(trajectory: list[dict]) -> list[dict]:
    """
    Convert a trajectory to OpenAI-compatible chat messages.

    - ``message("...")`` / ``plan("...")`` → ``user`` or ``assistant`` ``content``.
    - ``ToolName({...})`` → ``assistant`` with ``tool_calls``, optional following
        ``tool`` message when ``tool_result`` is present on the same step.
    """
    chat: list[dict] = []
    tool_seq = 0

    for event in trajectory:
        if not isinstance(event, dict):
            continue
        action = event.get("action")
        if not isinstance(action, str) or not action:
            continue

        actor = event.get("actor", "agent")
        role = "user" if actor == "user" else "assistant"

        for prefix in ("message", "plan"):
            body = _decode_message_like(action, prefix)
            if body is not None:
                chat.append({"role": role, "content": body})
                break
        else:
            parsed = _parse_tool_call(action)
            if parsed is not None:
                t_name, arguments = parsed
                tool_seq += 1
                call_id = f"call_{tool_seq}"
                chat.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": t_name, "arguments": arguments},
                            }
                        ],
                    }
                )
                tr = event.get("tool_result")
                if tr is not None:
                    chat.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": tr
                            if isinstance(tr, str)
                            else json.dumps(tr, ensure_ascii=False),
                        }
                    )
            else:
                chat.append({"role": role, "content": action})

    return chat


def chat_to_text(messages: list[dict]) -> str:
    """Serialize OpenAI-style chat messages to a human-readable text format.

    Handles text content, tool calls, and tool results.  Used to convert
    human correction trajectories into golden-answer strings for SDFT.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "") or ""
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                content += f"\n{fn.get('name', '')}({fn.get('arguments', '')})"
        if content:
            parts.append(f"[{role}]: {content}")
    return "\n".join(parts)


def _fold_thinking_into_content(msg: dict) -> dict:
    """Move a top-level ``thinking`` field into the renderer's structured content.

    The session/OPD JSON stores assistant chain-of-thought as a sibling key
    (``msg["thinking"]``), but ``tinker_cookbook.renderers.Message`` only
    recognizes thinking when it appears as a ``ThinkingPart`` inside
    ``content``. Without this conversion the renderer silently drops the
    reasoning, so neither the in-context history nor the golden demonstration
    ever expose the chain-of-thought to the teacher / student.

    Returns a new dict with the top-level ``thinking`` key removed; ``content``
    becomes ``[ThinkingPart, TextPart]`` (or just ``[ThinkingPart]`` when the
    visible content is empty). When ``content`` is already a structured list,
    a ThinkingPart is only prepended if none is present.
    """
    if msg.get("role") != "assistant":
        return {k: v for k, v in msg.items() if k != "thinking"}

    thinking = msg.get("thinking")
    if not isinstance(thinking, str) or not thinking.strip():
        return {k: v for k, v in msg.items() if k != "thinking"}

    new_msg = {k: v for k, v in msg.items() if k != "thinking"}
    content = new_msg.get("content")
    thinking_part = {"type": "thinking", "thinking": thinking}

    if isinstance(content, list):
        if not any(
            isinstance(p, dict) and p.get("type") == "thinking" for p in content
        ):
            new_msg["content"] = [thinking_part, *content]
        return new_msg

    text = content if isinstance(content, str) else ""
    if text:
        new_msg["content"] = [thinking_part, {"type": "text", "text": text}]
    else:
        new_msg["content"] = [thinking_part]
    return new_msg


def _hydrate_tool_calls(conversation: list[dict]) -> list[dict]:
    """Convert plain-dict tool_calls to ToolCall pydantic objects for the renderer.

    Also strips ``tool_calls`` keys that Arrow set to ``None`` so the
    renderer never encounters them, and folds any top-level ``thinking``
    field into the structured content list (see
    :func:`_fold_thinking_into_content`).
    """
    out = []
    for msg in conversation:
        msg = _fold_thinking_into_content(dict(msg))
        if msg.get("content") is None:
            msg["content"] = ""
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


@chz.chz
class DPODataBuilder(ChatDatasetBuilder):
    """Build interleaved chosen/rejected Datum pairs for DPO training.

    Reads a JSON file produced by ``export_dpo_data.py`` and converts each
    learning unit into a ``[chosen_datum, rejected_datum]`` pair suitable for
    :func:`tinker_dpo.main`.
    """

    train_path: str
    test_path: str | None = None

    def _load_units(self, raw: object) -> list[dict]:
        """Support both single-session and multi-session DPO export shapes."""
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

    def _load(self, path: str) -> list[dict]:
        with open(path, "r") as f:
            raw = json.load(f)
        rows = []
        for unit in self._load_units(raw):
            prompt = [{"role": "user", "content": msg} for msg in unit["user_messages"]]
            rows.append(
                {
                    "prompt": prompt,
                    "chosen": traj_to_chat(unit["chosen_trajectory"]),
                    "rejected": traj_to_chat(unit["rejected_trajectory"]),
                }
            )
        return rows

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        train_rows = self._load(self.train_path)
        train_ds = datasets.Dataset.from_list(train_rows)

        def flatmap_fn(row: dict) -> list:
            chosen_convo = _hydrate_tool_calls(row["prompt"] + row["chosen"])
            rejected_convo = _hydrate_tool_calls(row["prompt"] + row["rejected"])
            chosen_datum = conversation_to_datum(
                chosen_convo, self.renderer, self.common_config.max_length
            )
            rejected_datum = conversation_to_datum(
                rejected_convo, self.renderer, self.common_config.max_length
            )
            return [chosen_datum, rejected_datum]

        train_dataset = SupervisedDatasetFromHFDataset(
            train_ds, batch_size=self.common_config.batch_size, flatmap_fn=flatmap_fn
        )

        test_dataset = None
        if self.test_path is not None:
            test_rows = self._load(self.test_path)
            test_ds = datasets.Dataset.from_list(test_rows)
            test_dataset = SupervisedDatasetFromHFDataset(
                test_ds, batch_size=len(test_ds), flatmap_fn=flatmap_fn
            )

        return train_dataset, test_dataset


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

    def get_batch(
        self, index: int
    ) -> tuple[Sequence[EnvGroupBuilder], list[list[renderers.Message]], list[str]]:
        # Wrap so training can run multiple epochs (``tinker_opd`` uses ``index`` up to
        # ``len(dataset) * epochs - 1``) without empty batches past the first epoch.
        idx = index % self._num_batch_slots()
        start = idx * self._batch_size
        end = min(start + self._batch_size, len(self._questions))

        raw_batch_questions = self._questions[start:end]
        batch_tool_schemas = self._tool_schemas_by_question[start:end]
        batch_questions = [
            self._prepare_question(q, tool_schemas)
            for q, tool_schemas in zip(raw_batch_questions, batch_tool_schemas)
        ]
        batch_golden = self._golden_answers[start:end]
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


class ReinforceDataset:
    """Supervised dataset that keeps per-trajectory reward metadata aligned with datums.

    Implements the ``SupervisedDataset`` interface (``get_batch``, ``set_epoch``,
    ``__len__``) and additionally exposes ``get_batch_rewards`` so the REINFORCE
    training loop can retrieve rewards that stay aligned with datums through
    epoch shuffling.
    """

    def __init__(self, datums: list, rewards: list[float], batch_size: int):
        if len(datums) != len(rewards):
            raise ValueError(
                f"datums ({len(datums)}) and rewards ({len(rewards)}) must have the same length"
            )
        self._datums = datums
        self._rewards = rewards
        self._batch_size = batch_size
        self._indices = list(range(len(datums)))

    def __len__(self) -> int:
        if not self._datums:
            return 0
        return (len(self._datums) + self._batch_size - 1) // self._batch_size

    def set_epoch(self, seed: int) -> None:
        import random

        rng = random.Random(seed)
        self._indices = list(range(len(self._datums)))
        rng.shuffle(self._indices)

    def get_batch(self, index: int) -> list:
        start = index * self._batch_size
        end = min(start + self._batch_size, len(self._indices))
        return [self._datums[self._indices[i]] for i in range(start, end)]

    def get_batch_rewards(self, index: int) -> list[float]:
        """Return the scalar rewards for the same trajectories as ``get_batch``."""
        start = index * self._batch_size
        end = min(start + self._batch_size, len(self._indices))
        return [self._rewards[self._indices[i]] for i in range(start, end)]


@chz.chz
class ReinforceDataBuilder(ChatDatasetBuilder):
    """Build tokenized Datums with per-trajectory rewards for REINFORCE training.

    Reads a JSON file containing agent interaction data with reward signals
    and converts each learning unit into a tokenized Datum paired with a scalar
    reward.

    Reward formula: ``reward = verifier - alpha * human`` where ``verifier`` is
    the 0-1 success ratio and ``human`` penalizes trajectories requiring more
    human corrections.

    Supports both single-session format (top-level ``learning_units``) and
    multi-session format (top-level ``sessions`` array, each containing
    ``learning_units``).
    """

    train_path: str
    test_path: str | None = None
    reward_alpha: float = 0.05

    def _load_units(self, path: str) -> list[dict]:
        with open(path, "r") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return []
        if isinstance(raw.get("learning_units"), list):
            return [u for u in raw["learning_units"] if isinstance(u, dict)]
        if isinstance(raw.get("sessions"), list):
            units: list[dict] = []
            for session in raw["sessions"]:
                if not isinstance(session, dict):
                    continue
                lu = session.get("learning_units")
                if isinstance(lu, list):
                    units.extend(u for u in lu if isinstance(u, dict))
            return units
        return []

    def _build_dataset(self, path: str, batch_size: int) -> ReinforceDataset:
        units = self._load_units(path)
        datums = []
        rewards: list[float] = []
        for unit in units:
            prompt = [{"role": "user", "content": msg} for msg in unit["user_messages"]]
            response = traj_to_chat(unit["agent_trajectory"])
            conversation = _hydrate_tool_calls(prompt + response)
            datum = conversation_to_datum(
                conversation, self.renderer, self.common_config.max_length
            )
            reward_data = unit["reward"]
            reward = reward_data["verifier"] - self.reward_alpha * reward_data["human"]
            datums.append(datum)
            rewards.append(reward)
        return ReinforceDataset(datums=datums, rewards=rewards, batch_size=batch_size)

    def __call__(self) -> tuple[ReinforceDataset, ReinforceDataset | None]:
        train_dataset = self._build_dataset(self.train_path, self.common_config.batch_size)
        test_dataset = None
        if self.test_path is not None:
            test_dataset = self._build_dataset(self.test_path, self.common_config.batch_size)
        return train_dataset, test_dataset


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