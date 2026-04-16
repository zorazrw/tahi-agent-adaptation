def parse_qwen3_instruct(chunk_dict: dict, state: dict) -> dict | None:
    """Rewrite split tool calls from Tinker's Qwen3-Instruct streaming format.

    Tinker may split a single tool call into two chunks with different IDs:
      1. id="functions.XXX:N", name="XXX", arguments=""  (name-only)
      2. id="call:N",          name="",   arguments="{...}" (args-only)

    This merges them into a single tool call with the correct id and name.

    Args:
        chunk_dict: A chat completion chunk as a mutable dict.
        state: Mutable dict persisted across chunks within one stream.
               Pass a fresh ``{}`` for the first chunk of each request.

    Returns:
        The rewritten chunk dict to forward, or ``None`` to suppress it.
    """
    pending = state.setdefault("pending_tool_calls", {})
    suppress = False

    for choice in chunk_dict.get("choices") or []:
        delta = choice.get("delta") or {}
        tool_calls = delta.get("tool_calls")
        if not tool_calls:
            continue

        for tc in tool_calls:
            idx = tc.get("index", 0)
            fn = tc.get("function") or {}
            tc_id = tc.get("id")
            name = fn.get("name")

            if name and tc_id and tc_id.startswith("functions."):
                pending[idx] = (tc_id, name)
                suppress = True
            elif idx in pending:
                real_id, real_name = pending[idx]
                tc["id"] = real_id
                if not fn.get("name"):
                    fn["name"] = real_name
                    tc["function"] = fn

    return None if suppress else chunk_dict


func_mapping = {
    "qwen3_instruct": parse_qwen3_instruct,
}