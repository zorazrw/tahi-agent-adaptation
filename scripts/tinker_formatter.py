import json
import re
from collections.abc import Sequence

import chz
import datasets

from tinker_cookbook import renderers
from tinker_cookbook.renderers.base import ToolCall
from tinker_cookbook.rl.types import EnvGroupBuilder
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset, conversation_to_datum
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset

try:
    from functools import partial

    from tinker_cookbook.distillation.datasets import PromptOnlyEnv
    from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
except ImportError:
    ProblemGroupBuilder = None  # type: ignore[assignment,misc]
    PromptOnlyEnv = None  # type: ignore[assignment,misc]


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
        content = msg.get("content", "")
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                content += f"\n{fn.get('name', '')}({fn.get('arguments', '')})"
        if content:
            parts.append(f"[{role}]: {content}")
    return "\n".join(parts)


def _hydrate_tool_calls(conversation: list[dict]) -> list[dict]:
    """Convert plain-dict tool_calls to ToolCall pydantic objects for the renderer.

    Also strips ``tool_calls`` keys that Arrow set to ``None`` so the
    renderer never encounters them.
    """
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

    def _load(self, path: str) -> list[dict]:
        with open(path, "r") as f:
            raw = json.load(f)
        rows = []
        # student_prompt = [
        #     {
        #         "role": "user",
        #         "content": raw["initial_message"]["steps"][0]["action"].strip("message(").strip(")"),
        #     }
        # ]
        units = self._load_units(raw)
        for unit in units[1:]:
            question = "\n".join(unit["user_messages"])
            golden_answer = chat_to_text(traj_to_chat(unit["human_trajectory"]))
            student_prompt = [{"role": "user", "content": msg} for msg in unit["user_messages"]]
            teacher_prompt = student_prompt + [
                {
                    "role": "user",
                    "content": (
                        "Here is the user's response after the original agent's "
                        "response was executed. Please use this to improve your "
                        "response:\n" + golden_answer
                    ),
                }
            ]
            rows.append(
                {
                    "student_prompt": student_prompt,
                    "teacher_prompt": teacher_prompt,
                    "question": question,
                    "golden_answer": golden_answer,
                }
            )
        return rows

    def __call__(
        self,
        renderer: renderers.Renderer,
        batch_size: int = 4,
        group_size: int = 1,
    ) -> "OPDSDFTDataset":
        rows = self._load(self.train_path)
        questions = [r["question"] for r in rows]
        golden_answers = [r["golden_answer"] for r in rows]
        return OPDSDFTDataset(
            questions=questions,
            golden_answers=golden_answers,
            renderer=renderer,
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
        questions: list[str],
        golden_answers: list[str],
        renderer: renderers.Renderer,
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

    @classmethod
    def from_json(
        cls,
        data_path: str,
        renderer: renderers.Renderer,
        batch_size: int = 4,
        group_size: int = 1,
    ) -> "OPDSDFTDataset":
        """Load OPD data from a JSON file.

        Skips ``learning_units[0]`` (the initial attempt with no prior
        human correction to distil from).
        """
        with open(data_path, "r") as f:
            raw = json.load(f)

        questions: list[str] = []
        golden_answers: list[str] = []

        units = cls._load_units(raw)
        for unit in units[1:]:
            questions.append("\n".join(unit["user_messages"]))
            golden_answers.append(
                chat_to_text(traj_to_chat(unit["human_trajectory"]))
            )

        return cls(
            questions=questions,
            golden_answers=golden_answers,
            renderer=renderer,
            batch_size=batch_size,
            group_size=group_size,
        )

    def __len__(self) -> int:
        return max(1, (len(self._questions) + self._batch_size - 1) // self._batch_size)

    def get_batch(
        self, index: int
    ) -> tuple[Sequence[EnvGroupBuilder], list[str], list[str]]:
        start = index * self._batch_size
        end = min(start + self._batch_size, len(self._questions))

        batch_questions = self._questions[start:end]
        batch_golden = self._golden_answers[start:end]
        builders: list[EnvGroupBuilder] = [
            self._make_builder(q) for q in batch_questions
        ]
        return builders, batch_questions, batch_golden

    def _make_builder(self, question: str) -> EnvGroupBuilder:
        """Create an EnvGroupBuilder for a single student prompt.

        Prefers the cookbook's ``ProblemGroupBuilder`` / ``PromptOnlyEnv``
        when available, otherwise falls back to a local minimal builder.
        """
        if ProblemGroupBuilder is not None and PromptOnlyEnv is not None:
            return ProblemGroupBuilder(
                env_thunk=partial(PromptOnlyEnv, question, self._renderer),
                num_envs=self._group_size,
                dataset_name="opd",
            )
        return _OPDEnvGroupBuilder(
            question=question,
            renderer=self._renderer,
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
        question: str,
        renderer: renderers.Renderer,
        group_size: int = 1,
    ):
        self._question = question
        self._renderer = renderer
        self._group_size = group_size

    async def make_envs(self) -> Sequence:
        messages: list[renderers.Message] = [
            {"role": "user", "content": self._question}  # type: ignore[typeddict-item]
        ]
        prompt = self._renderer.build_generation_prompt(messages)
        envs = []
        for _ in range(self._group_size):
            envs.append(_PromptOnlyEnv(prompt))
        return envs

    def logging_tags(self) -> list[str]:
        return ["opd"]


class _PromptOnlyEnv:
    """Minimal single-turn environment that yields zero reward.

    Provides an initial ``ModelInput`` prompt and immediately terminates
    the episode on the first step.  Compatible with the ``Env`` protocol
    from ``tinker_cookbook.rl.types``.
    """

    def __init__(self, prompt: object):
        self._prompt = prompt
        self._done = False

    def get_initial_observation(self) -> object:
        return self._prompt

    async def step(self, action: object) -> tuple[object | None, float, bool, dict]:
        self._done = True
        return None, 0.0, True, {}

    async def reset(self) -> object:
        self._done = False
        return self._prompt