from __future__ import annotations

import asyncio
from typing import Any

import tinker

from tinker_cookbook import renderers
from tinker_cookbook.model_info import (
    get_recommended_renderer_name,
    get_recommended_renderer_names,
)
from tinker_cookbook.renderers.base import message_to_jsonable
from tinker_cookbook.third_party.openai_compat import (
    openai_messages_to_tinker,
    openai_tools_to_tinker,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer


def _pick_renderer_name(base_model: str, requested: str | None, reasoning: str | None) -> str:
    if requested:
        return requested

    candidates = get_recommended_renderer_names(base_model)
    if reasoning == "off":
        disabled = next((name for name in candidates if "disable_thinking" in name), None)
        if disabled:
            return disabled

    return get_recommended_renderer_name(base_model)


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        entry: dict[str, Any] = {
            "role": role,
            "content": content if content is not None else "",
        }

        if "tool_calls" in message:
            entry["tool_calls"] = message["tool_calls"]
        if "tool_call_id" in message:
            entry["tool_call_id"] = message["tool_call_id"]
        if "name" in message:
            entry["name"] = message["name"]
        normalized.append(entry)
    return normalized


def _build_service_client(provider: dict[str, Any]) -> tinker.ServiceClient:
    service_kwargs: dict[str, Any] = {}
    base_url = str(provider.get("base_url") or "").strip()
    api_key = str(provider.get("api_key") or "").strip()
    if base_url:
        service_kwargs["base_url"] = base_url
    if api_key:
        service_kwargs["api_key"] = api_key
    return tinker.ServiceClient(**service_kwargs)


async def resolve_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    provider = payload.get("provider") or {}
    tinker_path = str(payload.get("tinker_path") or "").strip()
    if not tinker_path:
        raise ValueError("tinker_path is required")

    service_client = _build_service_client(provider)
    rest_client = service_client.create_rest_client()
    info = await rest_client.get_weights_info_by_tinker_path(tinker_path)
    base_model = str(getattr(info, "base_model", "") or "").strip()
    if not base_model:
        raise ValueError("No base model found for checkpoint")

    return {
        "ok": True,
        "base_model": base_model,
    }


async def run_request(payload: dict[str, Any]) -> dict[str, Any]:
    provider = payload.get("provider") or {}
    model = payload.get("model") or {}
    context = payload.get("context") or {}
    options = payload.get("options") or {}

    base_model = str(model.get("base_model") or "").strip()
    if not base_model:
        raise ValueError("model.base_model is required")

    model_name = str(model.get("id") or "").strip() or "tinker"
    model_path_raw = model.get("model_path")
    renderer_name_raw = model.get("renderer_name")
    reasoning = options.get("reasoning")
    tool_defs = context.get("tools") or []

    service_client = _build_service_client(provider)

    sampling_kwargs: dict[str, Any] = {"base_model": base_model}
    model_path = str(model_path_raw or "").strip()
    if model_path:
        sampling_kwargs["model_path"] = model_path
    sampling_client = service_client.create_sampling_client(**sampling_kwargs)

    tokenizer = get_tokenizer(base_model)
    renderer_name = _pick_renderer_name(
        base_model,
        str(renderer_name_raw).strip() if renderer_name_raw else None,
        str(reasoning).strip() if reasoning else None,
    )
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    openai_messages = _normalize_messages(context.get("messages") or [])
    tinker_messages = openai_messages_to_tinker(openai_messages)

    if tool_defs:
        system_prompt = str(context.get("system_prompt") or "")
        prefix = renderer.create_conversation_prefix_with_tools(
            openai_tools_to_tinker(tool_defs),
            system_prompt,
        )
        if openai_messages and openai_messages[0].get("role") == "system":
            tinker_messages = prefix + tinker_messages[1:]
        else:
            tinker_messages = prefix + tinker_messages

    model_input = renderer.build_generation_prompt(tinker_messages)
    prompt_token_ids = model_input.to_ints()

    stop = options.get("stop")
    if not stop:
        stop = renderer.get_stop_sequences()

    sample_response = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            temperature=float(options.get("temperature", 0.0)),
            max_tokens=int(options.get("max_tokens", 4096)),
            top_p=float(options.get("top_p", 1.0)),
            top_k=int(options.get("top_k", -1)),
            stop=stop,
        ),
    )

    sequence = sample_response.sequences[0]
    completion_token_ids = sequence.tokens
    parsed_message, parse_success = renderer.parse_response(completion_token_ids)

    return {
        "ok": True,
        "renderer_name": renderer_name,
        "parse_success": parse_success,
        "message": message_to_jsonable(parsed_message),
        "usage": {
            "input": len(prompt_token_ids),
            "output": len(completion_token_ids),
            "cacheRead": 0,
            "cacheWrite": 0,
            "totalTokens": len(prompt_token_ids) + len(completion_token_ids),
        },
    }


def run_request_sync(payload: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(run_request(payload))


def resolve_checkpoint_sync(payload: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(resolve_checkpoint(payload))
