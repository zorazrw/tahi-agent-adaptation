import json
import re

import chz
import datasets

from tinker_cookbook.renderers.base import ToolCall
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset, conversation_to_datum
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset


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

    def _load(self, path: str) -> list[dict]:
        with open(path, "r") as f:
            raw = json.load(f)
        rows = []
        for unit in raw["learning_units"]:
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