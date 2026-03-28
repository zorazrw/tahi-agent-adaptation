"""
Resolve Anthropic API settings the same way as the Electron app (see
`claude-settings.ts` / `buildEnvForConfig` / `runner.ts`).

Priority:
  1. api-config.json under Electron userData (app settings UI)
  2. ~/.claude/settings.json  env.ANTHROPIC_AUTH_TOKEN, ANTHROPIC_BASE_URL, ANTHROPIC_MODEL
     (same predicates as getCurrentApiConfig — all three must be present)
  3. Environment: ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN, optional ANTHROPIC_BASE_URL,
     optional ANTHROPIC_MODEL
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL = "claude-sonnet-4-5-20250929"


class AnthropicConfigError(RuntimeError):
    pass


@dataclass
class ResolvedAnthropic:
    api_key: str
    # When base_url is None, the Python SDK defaults to https://api.anthropic.com
    base_url: str | None
    model: str
    source: str


def _api_config_json_candidates() -> list[Path]:
    home = Path.home()
    out: list[Path] = []
    if sys.platform == "darwin":
        base = home / "Library" / "Application Support"
        out.extend(
            [
                base / "agent-cowork" / "api-config.json",
                base / "Agent Cowork" / "api-config.json",
            ]
        )
    elif sys.platform == "win32":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            ad = Path(appdata)
            out.extend(
                [
                    ad / "agent-cowork" / "api-config.json",
                    ad / "Agent Cowork" / "api-config.json",
                ]
            )
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME", str(home / ".config"))
        xdg_path = Path(xdg)
        out.extend(
            [
                xdg_path / "agent-cowork" / "api-config.json",
                xdg_path / "Agent Cowork" / "api-config.json",
            ]
        )
    return out


def _load_api_config_json() -> ResolvedAnthropic | None:
    for path in _api_config_json_candidates():
        try:
            if not path.is_file():
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        key = raw.get("apiKey")
        base = (raw.get("baseURL") or "").strip()
        model = (raw.get("model") or "").strip()
        if key and base and model:
            return ResolvedAnthropic(
                api_key=str(key),
                base_url=base.rstrip("/") or None,
                model=model,
                source=f"api-config.json ({path})",
            )
    return None


def _load_claude_settings_json() -> ResolvedAnthropic | None:
    path = Path.home() / ".claude" / "settings.json"
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    env = parsed.get("env") or {}
    if not isinstance(env, dict):
        return None
    auth = env.get("ANTHROPIC_AUTH_TOKEN")
    base = env.get("ANTHROPIC_BASE_URL")
    model = env.get("ANTHROPIC_MODEL")
    if auth and base and model:
        return ResolvedAnthropic(
            api_key=str(auth),
            base_url=str(base).strip().rstrip("/") or None,
            model=str(model).strip(),
            source=f"~/.claude/settings.json ({path})",
        )
    return None


def _load_from_environment() -> ResolvedAnthropic | None:
    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if not key:
        return None
    base = (os.environ.get("ANTHROPIC_BASE_URL") or "").strip()
    model = (os.environ.get("ANTHROPIC_MODEL") or "").strip() or DEFAULT_MODEL
    return ResolvedAnthropic(
        api_key=key.strip(),
        base_url=base.rstrip("/") or None,
        model=model,
        source="environment variables",
    )


def resolve_anthropic_config() -> ResolvedAnthropic:
    """
    Match runner.ts: same config files and env var names as buildEnvForConfig
    (ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, ANTHROPIC_MODEL).
    """
    found = _load_api_config_json()
    if found:
        return found
    found = _load_claude_settings_json()
    if found:
        return found
    found = _load_from_environment()
    if found:
        return found

    paths = "\n  ".join(str(p) for p in _api_config_json_candidates())
    raise AnthropicConfigError(
        "No Anthropic API configuration found. The app uses the same sources as Settings:\n"
        f"  1. api-config.json (Electron userData), tried:\n  {paths}\n"
        "  2. ~/.claude/settings.json with env.ANTHROPIC_AUTH_TOKEN, ANTHROPIC_BASE_URL, "
        "ANTHROPIC_MODEL (all three)\n"
        "  3. Environment: ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN, optional "
        "ANTHROPIC_BASE_URL and ANTHROPIC_MODEL"
    )


def make_anthropic_client(resolved: ResolvedAnthropic):
    """Build anthropic.Anthropic with api_key and base_url like runner's merged env."""
    import anthropic

    kwargs: dict = {"api_key": resolved.api_key}
    if resolved.base_url:
        kwargs["base_url"] = resolved.base_url
    return anthropic.Anthropic(**kwargs)
