"""
Extract memories and skills from session JSON (e.g. ``out.json``).

Accepts export shape ``{ uuid, name, trajectory }``, a JSON array of those objects, or legacy ``{ sessions: [...] }``.
Outputs: ``<output>/memories/<slug>.md`` and ``skills/<slug>.md``.

Requires: anthropic, python-dotenv. API key resolution matches the Electron app (see below).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_MODEL = "claude-sonnet-4-5-20250929"


class AnthropicConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResolvedAnthropicConfig:
    api_key: str
    base_url: str | None
    model: str


def _api_config_paths() -> list[Path]:
    home = Path.home()
    if sys.platform == "darwin":
        b = home / "Library/Application Support"
        return [b / "agent-cowork/api-config.json", b / "Agent Cowork/api-config.json"]
    if sys.platform == "win32":
        ad = os.environ.get("APPDATA")
        if not ad:
            return []
        root = Path(ad)
        return [root / "agent-cowork/api-config.json", root / "Agent Cowork/api-config.json"]
    xdg = Path(os.environ.get("XDG_CONFIG_HOME", str(home / ".config")))
    return [xdg / "agent-cowork/api-config.json", xdg / "Agent Cowork/api-config.json"]


def _resolved_from_api_config(path: Path) -> ResolvedAnthropicConfig | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    key = raw.get("apiKey")
    base = (raw.get("baseURL") or "").strip()
    model = (raw.get("model") or "").strip()
    if key and base and model:
        return ResolvedAnthropicConfig(str(key), base.rstrip("/") or None, model)
    return None


def _resolved_from_claude_settings() -> ResolvedAnthropicConfig | None:
    try:
        parsed = json.loads((Path.home() / ".claude/settings.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    env = parsed.get("env") or {}
    if not isinstance(env, dict):
        return None
    auth, base, model = env.get("ANTHROPIC_AUTH_TOKEN"), env.get("ANTHROPIC_BASE_URL"), env.get("ANTHROPIC_MODEL")
    if auth and base and model:
        return ResolvedAnthropicConfig(
            str(auth),
            str(base).strip().rstrip("/") or None,
            str(model).strip(),
        )
    return None


def _resolved_from_env() -> ResolvedAnthropicConfig | None:
    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if not key:
        return None
    base = (os.environ.get("ANTHROPIC_BASE_URL") or "").strip().rstrip("/") or None
    model = (os.environ.get("ANTHROPIC_MODEL") or "").strip() or DEFAULT_MODEL
    return ResolvedAnthropicConfig(key.strip(), base, model)


def resolve_anthropic_config() -> ResolvedAnthropicConfig:
    """Same order as the app: userData ``api-config.json``, ``~/.claude/settings.json``, then env."""
    for path in _api_config_paths():
        r = _resolved_from_api_config(path)
        if r:
            return r
    r = _resolved_from_claude_settings()
    if r:
        return r
    r = _resolved_from_env()
    if r:
        return r
    tried = ", ".join(str(p) for p in _api_config_paths())
    raise AnthropicConfigError(
        "No Anthropic credentials. Use app Settings (writes api-config.json), or "
        "~/.claude/settings.json with ANTHROPIC_AUTH_TOKEN+ANTHROPIC_BASE_URL+ANTHROPIC_MODEL, or "
        f"env ANTHROPIC_API_KEY (optional ANTHROPIC_BASE_URL, ANTHROPIC_MODEL). Tried: {tried}"
    )


def make_anthropic_client(cfg: ResolvedAnthropicConfig):
    import anthropic

    kw: dict[str, str] = {"api_key": cfg.api_key}
    if cfg.base_url:
        kw["base_url"] = cfg.base_url
    return anthropic.Anthropic(**kw)


def _session_blobs(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if not isinstance(data, dict):
        return []
    if isinstance(data.get("sessions"), list):
        return [x for x in data["sessions"] if isinstance(x, dict)]
    if "trajectory" in data and isinstance(data["trajectory"], list):
        return [data]
    if "task_units" in data or "session_id" in data:
        return [data]
    return []


def build_context_inputs(data: Any) -> list[dict[str, Any]]:
    """Rows: ``name``, ``actions`` (trajectory action strings), ``source`` (uuid or session_i)."""
    rows: list[dict[str, Any]] = []
    for i, blob in enumerate(_session_blobs(data)):
        raw_traj = blob.get("trajectory")
        if not isinstance(raw_traj, list):
            continue
        if not any(isinstance(s, dict) and s.get("actor") == "agent" for s in raw_traj):
            continue
        nm = blob.get("name")
        name_str = nm if isinstance(nm, str) else ""
        actions = [
            entry["action"]
            for entry in raw_traj
            if isinstance(entry, dict) and isinstance(entry.get("action"), str)
        ]
        sid = blob.get("uuid")
        source = sid.strip() if isinstance(sid, str) and sid.strip() else f"session_{i}"
        rows.append({"name": name_str, "actions": actions, "source": source})
    return rows

MEMORY_SYSTEM = """From the task and numbered action log, write up to 6 lines the assistant should remember later.
Each line: Fact: ... or Preference: ... One sentence; no long paths or raw dumps. If nothing fits: NONE"""

SKILL_SYSTEM = """From the task and numbered log, describe the workflow the agent used: ordered steps, generalized (no long paths).
Reply with:
Title: <short name>
1. <step>
2. <step>
...
If nothing fits: NONE"""


def _llm_text(client, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


def extract_memories(client, model: str, task: str, log: str) -> list[str]:
    user = f"Task:\n{task}\n\nLog:\n{log or '(empty)'}\n"
    try:
        raw = _llm_text(client, model, MEMORY_SYSTEM, user)
    except Exception:
        logger.exception("Memory LLM failed")
        return []
    out: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.upper() == "NONE":
            continue
        low = s.lower()
        if low.startswith("fact:"):
            s = s[5:].strip()
        elif low.startswith("preference:"):
            s = s[11:].strip()
        s = " ".join(s.split())
        if s and not s.startswith("##"):
            out.append(s)
    return out


def extract_skill(client, model: str, task: str, log: str) -> tuple[str, list[str]] | None:
    user = (
        f"Task:\n{task}\n\nLog:\n{log or '(empty)'}\n\n"
        "Use Title: plus numbered steps only.\n"
    )
    try:
        raw = _llm_text(client, model, SKILL_SYSTEM, user, max_tokens=2048)
    except Exception:
        logger.exception("Skill LLM failed")
        return None
    blob = raw.strip()
    if not blob or blob.upper() == "NONE":
        return None
    title, steps = "", []
    step_re = re.compile(r"^\d+[\.\)]\s*(.+)$")
    for line in blob.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("title:"):
            title = line.split(":", 1)[1].strip()
            continue
        m = step_re.match(line)
        if m:
            steps.append(m.group(1).strip())
    if title and steps:
        return (title, steps)
    return None


def _slug(name: str, fallback: str) -> str:
    s = (name or "").strip()
    if len(s) >= 2 and s[0] in "\"'" and s[0] == s[-1]:
        try:
            s = json.loads(s)
        except json.JSONDecodeError:
            s = s[1:-1]
    s = re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")
    return (s or re.sub(r"[^a-z0-9]+", "-", fallback.lower()).strip("-") or "session")[:100]


def default_agent_cowork_user_data() -> Path:
    if env := os.environ.get("AGENT_COWORK_USER_DATA"):
        return Path(env).expanduser()
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library/Application Support/agent-cowork"
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", home / "AppData/Roaming")) / "agent-cowork"
    return home / ".config/agent-cowork"


def main() -> None:
    p = argparse.ArgumentParser(description="Extract memories & skills from session JSON")
    p.add_argument("--data_path", required=True, help="Path to session JSON")
    p.add_argument("--output_dir", default=None, help="Output root (default: app userData)")
    p.add_argument("--model", default=None, help="Anthropic model id")
    args = p.parse_args()
    load_dotenv()

    with open(args.data_path, encoding="utf-8") as f:
        raw = json.load(f)
    inputs = build_context_inputs(raw)
    if not inputs:
        logger.warning("Nothing to extract.")
        return

    try:
        cfg = resolve_anthropic_config()
    except AnthropicConfigError as e:
        logger.error("%s", e)
        raise SystemExit(1) from e
    client = make_anthropic_client(cfg)
    model = args.model or cfg.model
    logger.info("Model %s", model)

    out = (Path(args.output_dir) if args.output_dir else default_agent_cowork_user_data()).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    mem_dir, sk_dir = out / "memories", out / "skills"
    mem_dir.mkdir(parents=True, exist_ok=True)
    sk_dir.mkdir(parents=True, exist_ok=True)
    seen: dict[str, int] = {}
    nm, ns = 0, 0

    for row in inputs:
        name = row["name"]
        src = row["source"]
        actions = row.get("actions") or []
        log = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(actions))
        base = _slug(name, src)
        n = seen.get(base, 0)
        seen[base] = n + 1
        stem = base if n == 0 else f"{base}-{''.join(c for c in src if c.isalnum())[:8] or n}"

        memories = extract_memories(client, model, name, log)
        (mem_dir / f"{stem}.md").write_text(
            ("\n\n".join(memories) + "\n") if memories else "",
            encoding="utf-8",
        )

        skill = extract_skill(client, model, name, log)
        if skill:
            t, steps = skill
            body = t + "\n" + "\n".join(f"{i + 1}. {st}" for i, st in enumerate(steps)) + "\n"
            (sk_dir / f"{stem}.md").write_text(body, encoding="utf-8")
            ns += 1
        else:
            (sk_dir / f"{stem}.md").write_text("", encoding="utf-8")

        nm += len(memories)
        logger.info("%s → %s.md", src, stem)

    logger.info("Done: %d memory lines, %d skills → %s", nm, ns, out)


if __name__ == "__main__":
    main()
