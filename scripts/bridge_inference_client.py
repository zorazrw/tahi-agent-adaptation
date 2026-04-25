from __future__ import annotations

import json
import importlib
import sys
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Callable


class DictModel:
    """Small response wrapper with the subset of OpenAI SDK methods we use."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def model_dump(self) -> dict[str, Any]:
        return self._data

    def model_dump_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self._data, indent=indent)


class BridgeInferenceClient:
    """AsyncOpenAI-shaped inference client backed by tinker_bridge.run_request."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        resolve_renderer: Callable[[str], tuple[str, str]],
    ) -> None:
        self.base_url = _service_base_url(base_url)
        self.api_key = api_key
        self.resolve_renderer = resolve_renderer
        self.chat = _BridgeChat(self)


class _BridgeChat:
    def __init__(self, parent: BridgeInferenceClient) -> None:
        self.completions = _BridgeCompletions(parent)


class _BridgeCompletions:
    def __init__(self, parent: BridgeInferenceClient) -> None:
        self._parent = parent

    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        stop: list[str] | str | None = None,
        n: int | None = None,
        **kwargs: Any,
    ) -> DictModel | AsyncIterator[DictModel]:
        if n not in (None, 1):
            raise ValueError("Tinker bridge inference only supports n=1")

        run_request = _load_bridge_run_request()
        base_model, renderer_name = self._parent.resolve_renderer(model)
        token_limit = max_tokens if max_tokens is not None else max_completion_tokens

        result = await run_request(
            {
                "provider": {
                    "base_url": self._parent.base_url,
                    "api_key": self._parent.api_key,
                },
                "model": {
                    "id": model,
                    "base_model": base_model,
                    "model_path": model if model.startswith("tinker://") else "",
                    "renderer_name": renderer_name,
                },
                "options": {
                    "max_tokens": token_limit or 4096,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "stop": stop,
                    "reasoning": kwargs.get("reasoning"),
                },
                "context": {
                    "system_prompt": _first_system_text(messages),
                    "messages": messages,
                    "tools": tools or [],
                },
            }
        )
        if not result.get("ok"):
            raise RuntimeError(str(result.get("error") or "Tinker bridge request failed"))

        completion = _bridge_result_to_completion(result, model)
        if stream:
            return _stream_completion(completion.model_dump())
        return completion


def _load_bridge_run_request() -> Callable[[dict[str, Any]], Any]:
    bridge_project = Path(__file__).resolve().parents[1] / "tinker-bridge"
    bridge_path = str(bridge_project)
    if bridge_path not in sys.path:
        sys.path.insert(0, bridge_path)

    module = importlib.import_module("tinker_bridge.bridge")
    return module.run_request


def _service_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    for suffix in ("/oai/api/v1", "/api/v1"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def _first_system_text(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") != "system":
            continue
        return _content_to_text(message.get("content"))
    return ""


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content)


def _bridge_message_to_openai(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    text_parts: list[str] = []
    thinking_parts: list[str] = []

    if isinstance(content, str):
        text_parts.append(content)
    elif isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and part.get("text"):
                text_parts.append(str(part["text"]))
            elif part.get("type") == "thinking" and part.get("thinking"):
                thinking_parts.append(str(part["thinking"]))

    openai_message: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(text_parts),
    }
    if thinking_parts:
        openai_message["reasoning_content"] = "\n".join(thinking_parts)
    if message.get("tool_calls"):
        openai_message["tool_calls"] = message["tool_calls"]
    return openai_message


def _bridge_result_to_completion(result: dict[str, Any], model: str) -> DictModel:
    message = _bridge_message_to_openai(result.get("message") or {})
    tool_calls = message.get("tool_calls") or []
    finish_reason = "tool_calls" if tool_calls else "stop"
    if result.get("parse_success") is False and not tool_calls:
        finish_reason = "length"

    usage = result.get("usage") or {}
    data = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("input", 0),
            "completion_tokens": usage.get("output", 0),
            "total_tokens": usage.get("totalTokens", 0),
        },
    }
    return DictModel(data)


async def _stream_completion(completion: dict[str, Any]) -> AsyncIterator[DictModel]:
    choice = (completion.get("choices") or [{}])[0] or {}
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason")

    def chunk(delta: dict[str, Any], finish: str | None = None) -> DictModel:
        return DictModel(
            {
                "id": completion.get("id"),
                "object": "chat.completion.chunk",
                "created": completion.get("created"),
                "model": completion.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": finish,
                        "logprobs": None,
                    }
                ],
            }
        )

    yield chunk({"role": "assistant", "content": None})

    reasoning = message.get("reasoning_content")
    if reasoning:
        yield chunk({"reasoning_content": reasoning})

    content = message.get("content")
    if content:
        yield chunk({"content": content})

    for idx, tool_call in enumerate(message.get("tool_calls") or []):
        fn = tool_call.get("function") or {}
        yield chunk(
            {
                "tool_calls": [
                    {
                        "index": idx,
                        "id": tool_call.get("id"),
                        "type": tool_call.get("type", "function"),
                        "function": {
                            "name": fn.get("name", ""),
                            "arguments": fn.get("arguments", ""),
                        },
                    }
                ]
            }
        )

    yield chunk({}, finish_reason)
