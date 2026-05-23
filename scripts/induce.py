"""
Extract memories and skills from session JSON (e.g. ``out.json``).

Accepts export shape ``{ uuid, name, trajectory }``, weight-based ``{ uuid, name, task_units, ... }``
(a JSON array of those objects), or legacy ``{ sessions: [...] }``.
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


def resolve_anthropic_config(
    *,
    skip_api_config: bool = False,
    skip_claude_settings: bool = False,
) -> ResolvedAnthropicConfig:
    """Same order as the app: userData ``api-config.json``, ``~/.claude/settings.json``, then env."""
    if not skip_api_config:
        for path in _api_config_paths():
            r = _resolved_from_api_config(path)
            if r:
                return r
    if not skip_claude_settings:
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


def anthropic_user_text(
    client: Any,
    model: str,
    user: str,
    *,
    system: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    """
    One Messages API call; concatenate text blocks. Same pattern as memory/skill extraction.
    Omit ``system`` when the full instruction lives in ``user`` (e.g. verifier-labeler-style prompts).
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": user}],
    }
    if system:
        kwargs["system"] = system
    msg = client.messages.create(**kwargs)
    parts: list[str] = []
    for b in getattr(msg, "content", None) or []:
        btype = getattr(b, "type", None)
        if btype == "text":
            t = getattr(b, "text", None)
            if t:
                parts.append(str(t))
        # Extended-thinking models may emit thinking blocks; ignore those for extraction.
    return "".join(parts)


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


def _normalized_trajectory(blob: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Flat trajectory with per-step ``actor`` for induce.

    Legacy exports use a top-level ``trajectory`` list. Weight-based exports use ``task_units``;
    each unit has ``actor`` and a ``trajectory`` of steps (without per-step actor).
    """
    raw = blob.get("trajectory")
    if isinstance(raw, list) and raw:
        return [x for x in raw if isinstance(x, dict)]

    units = blob.get("task_units")
    if not isinstance(units, list) or not units:
        return []

    merged: list[dict[str, Any]] = []
    for u in units:
        if not isinstance(u, dict):
            continue
        actor = str(u.get("actor") or "user")
        traj = u.get("trajectory")
        if not isinstance(traj, list):
            continue
        for step in traj:
            if not isinstance(step, dict):
                continue
            row = dict(step)
            row.setdefault("actor", actor)
            merged.append(row)
    return merged


def build_context_inputs(data: Any) -> list[dict[str, Any]]:
    """Rows: ``name``, ``task`` (long instruction when present), ``actions``, ``source``."""
    rows: list[dict[str, Any]] = []
    for i, blob in enumerate(_session_blobs(data)):
        if not isinstance(blob, dict):
            continue
        raw_traj = _normalized_trajectory(blob)
        if not raw_traj:
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
        task_blob = blob.get("task")
        task_str = task_blob.strip() if isinstance(task_blob, str) else ""
        rows.append({"name": name_str, "task": task_str, "actions": actions, "source": source})
    return rows


MEMORY_SYSTEM = """From the task description and the numbered action log, write up to 6 short facts or user preferences worth remembering later.

Output rules:
- One fact per line. Plain text only (no markdown headers like # or ##).
- Each line is a single sentence (no numbered lists in the sense of "1." as list markers—use plain sentences).
- Optional prefixes "Fact:" or "Preference:" on a line are OK.
- If nothing is worth saving, output exactly the single word NONE (nothing else)."""

SKILL_SYSTEM = """From the task and numbered log, describe the workflow the agent used: ordered steps, generalized (no long paths).
Reply with:
Title: <short task name>
1. <step>
2. <step>
...
If nothing fits: NONE"""


def _llm_text(client, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
    return anthropic_user_text(client, model, user, system=system, max_tokens=max_tokens, temperature=0.0)


def _strip_outer_fences(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t
    first_nl = t.find("\n")
    if first_nl != -1:
        t = t[first_nl + 1 :]
    if t.rstrip().endswith("```"):
        t = t.rstrip()[:-3].rstrip()
    return t


_NUM_BULLET_RE = re.compile(r"^\s*(?:[-*+•]|\d+[\.)])\s+")


def _normalize_memory_line(line: str) -> str | None:
    s = line.strip()
    if not s:
        return None
    if s.upper().rstrip(".") in ("NONE", "N/A", "NA"):
        return None
    if s.startswith("##") or re.match(r"^#\s+\S", s):
        return None
    low = s.lower()
    if (
        low.startswith(("here are the", "below are the", "the following ", "summary:", "memories:", "facts:"))
        and len(s) < 140
    ):
        return None
    while _NUM_BULLET_RE.match(s):
        s = _NUM_BULLET_RE.sub("", s, count=1).strip()
    low = s.lower()
    if low.startswith("fact:"):
        s = s[5:].strip()
    elif low.startswith("preference:"):
        s = s[11:].strip()
    s = " ".join(s.split())
    if not s or s.upper().rstrip(".") == "NONE":
        return None
    if s.startswith("##"):
        return None
    return s


def extract_memories(client, model: str, task: str, log: str) -> list[str]:
    task_block = (task or "").strip() or "(no title)"
    user = f"Task / session title:\n{task_block}\n\nLog:\n{log or '(empty)'}\n"
    try:
        raw = _llm_text(client, model, MEMORY_SYSTEM, user)
    except Exception:
        logger.exception("Memory LLM failed")
        return []
    raw = _strip_outer_fences(raw)
    out: list[str] = []
    for line in raw.replace("\r\n", "\n").split("\n"):
        s = _normalize_memory_line(line)
        if s:
            out.append(s)
    if not out and raw.strip() and raw.strip().upper() not in ("NONE",):
        logger.warning(
            "Memory extraction produced 0 lines after parsing (model returned non-empty text). Preview: %s",
            raw.strip()[:500],
        )
    return out[:6]


def extract_skill(client, model: str, task: str, log: str) -> tuple[str, list[str]] | None:
    task_block = (task or "").strip() or "(no title)"
    user = (
        f"Task:\n{task_block}\n\nLog:\n{log or '(empty)'}\n\n"
        "Use Title: plus numbered steps only.\n"
    )
    try:
        raw = _llm_text(client, model, SKILL_SYSTEM, user, max_tokens=2048)
    except Exception:
        logger.exception("Skill LLM failed")
        return None
    blob = _strip_outer_fences(raw.strip())
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
        task_for_llm = (row.get("task") or "").strip() if isinstance(row.get("task"), str) else ""
        if not task_for_llm:
            task_for_llm = (name or "").strip()
        actions = row.get("actions") or []
        log = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(actions))
        base = _slug(name, src)
        n = seen.get(base, 0)
        seen[base] = n + 1
        stem = base if n == 0 else f"{base}-{''.join(c for c in src if c.isalnum())[:8] or n}"

        memories = extract_memories(client, model, task_for_llm, log)
        (mem_dir / f"{stem}.md").write_text(
            ("\n\n".join(memories) + "\n") if memories else "",
            encoding="utf-8",
        )

        skill = extract_skill(client, model, task_for_llm, log)
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
