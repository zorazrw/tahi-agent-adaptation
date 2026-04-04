import json
import re
import chz
import datasets

from tinker_cookbook.preference.preference_datasets import ComparisonDatasetBuilder
from tinker_cookbook.preference.types import (
    Comparison,
    ComparisonRenderer,
    ComparisonRendererFromChatRenderer,
    LabeledComparison,
)


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
                chat.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
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
                            "content": tr
                            if isinstance(tr, str)
                            else json.dumps(tr, ensure_ascii=False),
                        }
                    )
            else:
                chat.append({"role": role, "content": action})

    return chat


@chz.chz
class DPODataBuilder(ComparisonDatasetBuilder):
    train_path: str
    test_path: str | None = None
    
    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        import json
        
        train_data = []
        with open(self.train_path, "r") as f:
            for datum in json.load(f)["learning_units"]:
                datum = {
                    "completion_A": traj_to_chat(datum["chosen_trajectory"]),
                    "completion_B": traj_to_chat(datum["rejected_trajectory"]),
                    "user_messages": datum["user_messages"],
                }
                train_data.append(datum)
        train_dataset = datasets.Dataset.from_list(train_data)
        
        test_dataset = None
        if self.test_path is not None:
            test_data = []
            with open(self.test_path, "r") as f:
                for datum in json.load(f)["learning_units"]:
                    datum = {
                        "completion_A": traj_to_chat(datum["chosen_trajectory"]),
                        "completion_B": traj_to_chat(datum["rejected_trajectory"]),
                        "user_messages": datum["user_messages"],
                    }
                    test_data.append(datum)
            test_dataset = datasets.Dataset.from_list(test_data)
            
        return train_dataset, test_dataset
    
    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        prompt_conversion = [
            {"role": "user", "content": msg} for msg in example["user_messages"]
        ]
        comparison = Comparison(
            prompt_conversion=prompt_conversion,
            completion_A=example["completion_A"],
            completion_B=example["completion_B"],
        )
        return LabeledComparison(comparison=comparison, label="A")
    

if __name__ == "__main__":
    dpo_data_builder = DPODataBuilder(train_path="scripts/dpo.json")
    train_dataset, test_dataset = dpo_data_builder.get_train_and_test_datasets()
    for example in train_dataset:
        print(example)
        break