"""
Resolve LLM credentials from the Agent Cowork Pi runtime (same as in-app task solving).

Reads ``<userData>/pi-agent/settings.json`` for default provider/model, then loads
provider-specific config from ``auth.json``, ``tinker-provider.json``, and ``models.json``.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TINKER_PROVIDER = "tinker"
OPENAI_COMPATIBLE_PROVIDER = "openai-compatible"


class PiLlmConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResolvedRuntimeLlm:
    provider: str
    model: str
    api_key: str
    base_url: str | None = None
    base_model: str | None = None
    model_path: str | None = None
    renderer_name: str | None = None
    reasoning: str | None = None
    max_tokens: int = 16384


def _user_data_roots() -> list[Path]:
    if env := os.environ.get("AGENT_COWORK_USER_DATA"):
        return [Path(env).expanduser()]
    home = Path.home()
    if sys.platform == "darwin":
        base = home / "Library/Application Support"
        return [base / "agent-cowork", base / "Agent Cowork"]
    if sys.platform == "win32":
        ad = Path(os.environ.get("APPDATA", home / "AppData/Roaming"))
        return [ad / "agent-cowork", ad / "Agent Cowork"]
    xdg = Path(os.environ.get("XDG_CONFIG_HOME", str(home / ".config")))
    return [xdg / "agent-cowork", xdg / "Agent Cowork"]


def default_agent_cowork_user_data() -> Path:
    for root in _user_data_roots():
        if (root / "pi-agent" / "settings.json").is_file():
            return root
    return _user_data_roots()[0]


def pi_agent_dir(user_data: Path | None = None) -> Path:
    root = (user_data or default_agent_cowork_user_data()).expanduser().resolve()
    return root / "pi-agent"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_auth_key(agent_dir: Path, provider: str) -> str:
    raw = _read_json(agent_dir / "auth.json")
    if isinstance(raw, dict):
        entry = raw.get(provider)
        if isinstance(entry, dict) and entry.get("type") == "api_key":
            key = str(entry.get("key") or "").strip()
            if key:
                return key
    env_map = {
        TINKER_PROVIDER: "TINKER_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        OPENAI_COMPATIBLE_PROVIDER: "OPENAI_COMPATIBLE_API_KEY",
    }
    env_name = env_map.get(provider)
    if env_name:
        return (os.environ.get(env_name) or "").strip()
    if provider == "anthropic":
        return (os.environ.get("ANTHROPIC_AUTH_TOKEN") or "").strip()
    return ""


def _thinking_to_reasoning(thinking_level: str | None) -> str | None:
    if (thinking_level or "").strip().lower() in ("", "off"):
        return "off"
    return None


def _read_tinker_config(agent_dir: Path) -> dict[str, Any]:
    raw = _read_json(agent_dir / "tinker-provider.json")
    if not isinstance(raw, dict) or raw.get("provider") != TINKER_PROVIDER:
        raise PiLlmConfigError(f"Missing or invalid {agent_dir / 'tinker-provider.json'}")
    model = raw.get("model")
    if not isinstance(model, dict):
        raise PiLlmConfigError("tinker-provider.json: model block required")
    base_model = str(model.get("baseModel") or "").strip()
    renderer_name = str(model.get("rendererName") or "").strip()
    slug = str(model.get("id") or "").strip()
    if not base_model or not renderer_name or not slug:
        raise PiLlmConfigError(
            "tinker-provider.json requires model.id, model.baseModel, and model.rendererName"
        )
    model_path = str(model.get("modelPath") or "").strip()
    return {
        "base_url": (str(raw.get("baseUrl") or "").strip() or None),
        "id": slug,
        "base_model": base_model,
        "model_path": model_path or None,
        "renderer_name": renderer_name,
        "max_tokens": int(model.get("maxTokens") or 16384),
    }


def _read_openai_compatible(agent_dir: Path) -> tuple[str, str, str]:
    raw = _read_json(agent_dir / "models.json")
    if not isinstance(raw, dict):
        raise PiLlmConfigError(f"Missing {agent_dir / 'models.json'} for openai-compatible provider")
    providers = raw.get("providers")
    if not isinstance(providers, dict):
        raise PiLlmConfigError("models.json: providers object required")
    cfg = providers.get(OPENAI_COMPATIBLE_PROVIDER)
    if not isinstance(cfg, dict):
        raise PiLlmConfigError("openai-compatible provider not configured in models.json")
    base_url = str(cfg.get("baseUrl") or "").strip().rstrip("/")
    api_format = str(cfg.get("api") or "openai-completions").strip()
    model_id = ""
    models = cfg.get("models")
    if isinstance(models, list):
        for item in models:
            if isinstance(item, dict) and str(item.get("id") or "").strip():
                model_id = str(item["id"]).strip()
                break
    if not base_url or not model_id:
        raise PiLlmConfigError("openai-compatible baseUrl and model id are required in models.json")
    return base_url, model_id, api_format


def _anthropic_base_url(agent_dir: Path) -> str | None:
    raw = _read_json(agent_dir / "models.json")
    if not isinstance(raw, dict):
        return None
    providers = raw.get("providers")
    if not isinstance(providers, dict):
        return None
    anthropic = providers.get("anthropic")
    if not isinstance(anthropic, dict):
        return None
    base = str(anthropic.get("baseUrl") or "").strip().rstrip("/")
    return base or None


def resolve_runtime_llm(
    *,
    user_data: Path | None = None,
    model_override: str | None = None,
) -> ResolvedRuntimeLlm:
    """Load the same default provider/model the Electron app uses for Pi sessions."""
    agent_dir = pi_agent_dir(user_data)
    settings = _read_json(agent_dir / "settings.json")
    if not isinstance(settings, dict):
        settings = {}
    provider = str(settings.get("defaultProvider") or "").strip()
    model = (model_override or "").strip() or str(settings.get("defaultModel") or "").strip()
    thinking = str(settings.get("defaultThinkingLevel") or "").strip() or None

    if not provider or not model:
        raise PiLlmConfigError(
            f"No Pi runtime defaults in {agent_dir / 'settings.json'}. "
            "Save provider + model in the app Settings → Runtime Configuration."
        )

    if provider == TINKER_PROVIDER:
        tinker = _read_tinker_config(agent_dir)
        if model != tinker["id"]:
            raise PiLlmConfigError(
                f"--model {model!r} does not match tinker-provider.json id {tinker['id']!r}"
            )
        api_key = _read_auth_key(agent_dir, TINKER_PROVIDER)
        if not api_key:
            raise PiLlmConfigError("Tinker API key missing in pi-agent/auth.json or TINKER_API_KEY")
        model_path = tinker["model_path"]
        if model_path and not model_path.startswith("tinker://"):
            model_path = None
        return ResolvedRuntimeLlm(
            provider=TINKER_PROVIDER,
            model=tinker["id"],
            api_key=api_key,
            base_url=tinker["base_url"],
            base_model=tinker["base_model"],
            model_path=model_path,
            renderer_name=tinker["renderer_name"],
            reasoning=_thinking_to_reasoning(thinking),
            max_tokens=max(256, int(tinker["max_tokens"])),
        )

    if provider == "anthropic":
        api_key = _read_auth_key(agent_dir, "anthropic")
        if not api_key:
            raise PiLlmConfigError("Anthropic API key missing in pi-agent/auth.json or ANTHROPIC_API_KEY")
        base = _anthropic_base_url(agent_dir) or (os.environ.get("ANTHROPIC_BASE_URL") or "").strip() or None
        return ResolvedRuntimeLlm(provider="anthropic", model=model, api_key=api_key, base_url=base)

    if provider == "openai":
        api_key = _read_auth_key(agent_dir, "openai")
        if not api_key:
            raise PiLlmConfigError("OpenAI API key missing in pi-agent/auth.json or OPENAI_API_KEY")
        return ResolvedRuntimeLlm(provider="openai", model=model, api_key=api_key)

    if provider == OPENAI_COMPATIBLE_PROVIDER:
        base_url, model_id, api_format = _read_openai_compatible(agent_dir)
        if api_format == "openai-responses":
            raise PiLlmConfigError("openai-responses is not supported for induce; use openai-completions.")
        api_key = _read_auth_key(agent_dir, OPENAI_COMPATIBLE_PROVIDER)
        return ResolvedRuntimeLlm(
            provider=OPENAI_COMPATIBLE_PROVIDER,
            model=model_id,
            api_key=api_key or "local",
            base_url=base_url,
        )

    raise PiLlmConfigError(f"Unsupported Pi provider: {provider!r}")


def _text_from_bridge_message(message: dict[str, Any]) -> str:
    content = message.get("content")
    parts: list[str] = []
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                parts.append(str(block["text"]))
    return "\n".join(parts).strip()


def _run_tinker_bridge(cfg: ResolvedRuntimeLlm, messages: list[dict[str, Any]], max_tokens: int) -> str:
    bridge_project = Path(__file__).resolve().parents[1] / "tinker-bridge"
    if str(bridge_project) not in sys.path:
        sys.path.insert(0, str(bridge_project))
    run_request_sync = importlib.import_module("tinker_bridge.bridge").run_request_sync

    model_path = cfg.model_path or ""
    result = run_request_sync(
        {
            "provider": {"base_url": cfg.base_url or "", "api_key": cfg.api_key},
            "model": {
                "id": cfg.model,
                "base_model": cfg.base_model,
                "model_path": model_path,
                "renderer_name": cfg.renderer_name,
            },
            "options": {
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "reasoning": cfg.reasoning,
            },
            "context": {
                "system_prompt": "",
                "messages": messages,
                "tools": [],
            },
        }
    )
    if not result.get("ok"):
        raise RuntimeError(str(result.get("error") or "Tinker bridge request failed"))
    text = _text_from_bridge_message(result.get("message") or {})
    if not text:
        raise RuntimeError("Tinker bridge returned empty text")
    return text


def _anthropic_text(cfg: ResolvedRuntimeLlm, system: str | None, user: str, max_tokens: int) -> str:
    import anthropic

    kw: dict[str, str] = {"api_key": cfg.api_key}
    if cfg.base_url:
        kw["base_url"] = cfg.base_url
    client = anthropic.Anthropic(**kw)
    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": user}],
    }
    if system:
        kwargs["system"] = system
    msg = client.messages.create(**kwargs)
    return "".join(
        str(b.text)
        for b in getattr(msg, "content", None) or []
        if getattr(b, "type", None) == "text" and getattr(b, "text", None)
    )


def _openai_chat_text(cfg: ResolvedRuntimeLlm, system: str | None, user: str, max_tokens: int) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url if cfg.provider == OPENAI_COMPATIBLE_PROVIDER else None,
    )
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    choice = (resp.choices or [None])[0]
    if choice is None or choice.message is None:
        return ""
    content = choice.message.content
    return content if isinstance(content, str) else str(content or "")


def runtime_llm_text(
    cfg: ResolvedRuntimeLlm,
    system: str | None,
    user: str,
    *,
    max_tokens: int = 1024,
) -> str:
    """One-shot text generation using the resolved Pi runtime backend."""
    if cfg.provider == TINKER_PROVIDER:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return _run_tinker_bridge(cfg, messages, min(max_tokens, cfg.max_tokens))
    if cfg.provider == "anthropic":
        return _anthropic_text(cfg, system, user, max_tokens)
    if cfg.provider in ("openai", OPENAI_COMPATIBLE_PROVIDER):
        return _openai_chat_text(cfg, system, user, max_tokens)
    raise PiLlmConfigError(f"Unsupported provider: {cfg.provider!r}")
