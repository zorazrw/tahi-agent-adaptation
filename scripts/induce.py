"""
Extract memories and skills from session JSON (e.g. ``out.json``).

Accepts export shape ``{ uuid, name, trajectory }``, weight-based ``{ uuid, name, task_units, ... }``
(a JSON array of those objects), or legacy ``{ sessions: [...] }``.
Outputs: ``<output>/memories/<slug>.md`` and ``skills/<slug>.md``.

When export JSON includes ``expertise_task`` (e.g. ``data-viz-html``), writes to that stem
instead of slugifying the session title, and includes existing memory/skill file content in the LM prompt.

Requires: python-dotenv; ``anthropic`` for Claude models; Pi runtime deps (e.g. tinker bridge) otherwise.

Model selection: ``--model`` or Pi ``defaultModel``. If that name starts with ``claude``, use
Anthropic credentials from ``pi-agent/auth.json`` (same as in-app Settings), then env.
Otherwise use the Pi agent runtime (typically Tinker).
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

from pi_llm import (
    PiLlmConfigError,
    ResolvedRuntimeLlm,
    default_agent_cowork_user_data,
    resolve_runtime_llm,
    runtime_llm_text,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
_TASK_STEM_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,99}$")
_MAX_EXISTING_FILE_CHARS = 6000


class AnthropicConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResolvedAnthropicConfig:
    api_key: str
    base_url: str | None
    model: str


def _resolved_from_auth_json() -> ResolvedAnthropicConfig | None:
    auth_path = default_agent_cowork_user_data() / "pi-agent" / "auth.json"
    try:
        raw = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    entry = raw.get("anthropic")
    if not isinstance(entry, dict) or entry.get("type") != "api_key":
        return None
    key = str(entry.get("key") or "").strip()
    if not key:
        return None
    agent_dir = auth_path.parent
    base: str | None = None
    try:
        models = json.loads((agent_dir / "models.json").read_text(encoding="utf-8"))
        if isinstance(models, dict):
            providers = models.get("providers")
            if isinstance(providers, dict):
                anthropic = providers.get("anthropic")
                if isinstance(anthropic, dict):
                    base = str(anthropic.get("baseUrl") or "").strip().rstrip("/") or None
    except (OSError, json.JSONDecodeError):
        pass
    if not base:
        base = (os.environ.get("ANTHROPIC_BASE_URL") or "").strip().rstrip("/") or None
    return ResolvedAnthropicConfig(key, base, _anthropic_model())


def _anthropic_model() -> str:
    env_model = (os.environ.get("ANTHROPIC_MODEL") or "").strip()
    if env_model:
        return env_model
    settings_model = _pi_settings_model()
    if settings_model.lower().startswith("claude"):
        return settings_model
    return DEFAULT_MODEL


def _resolved_from_env() -> ResolvedAnthropicConfig | None:
    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if not key:
        return None
    base = (os.environ.get("ANTHROPIC_BASE_URL") or "").strip().rstrip("/") or None
    return ResolvedAnthropicConfig(key.strip(), base, _anthropic_model())


def resolve_anthropic_config(
    *,
    skip_api_config: bool = False,
    skip_claude_settings: bool = False,
) -> ResolvedAnthropicConfig:
    """Anthropic creds from ``pi-agent/auth.json`` (in-app Settings), then env."""
    _ = skip_claude_settings
    if not skip_api_config:
        r = _resolved_from_auth_json()
        if r:
            return r
    r = _resolved_from_env()
    if r:
        return r
    auth_path = default_agent_cowork_user_data() / "pi-agent" / "auth.json"
    raise AnthropicConfigError(
        f"No Anthropic credentials. Save Anthropic API key in app Settings (writes {auth_path}), "
        "or set ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN "
        "(optional ANTHROPIC_BASE_URL, ANTHROPIC_MODEL)."
    )


def _pi_settings_model() -> str:
    path = default_agent_cowork_user_data() / "pi-agent" / "settings.json"
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(settings, dict):
        return ""
    return str(settings.get("defaultModel") or "").strip()


def resolve_induce_llm(*, model_override: str | None = None) -> ResolvedRuntimeLlm:
    """Claude models use ``auth.json``; others use the Pi runtime (e.g. Tinker)."""
    model = (model_override or "").strip() or _pi_settings_model()
    if model.lower().startswith("claude"):
        cfg = resolve_anthropic_config()
        return ResolvedRuntimeLlm(
            provider="anthropic",
            model=model,
            api_key=cfg.api_key,
            base_url=cfg.base_url,
        )
    return resolve_runtime_llm(model_override=model_override)


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


def _is_user_message_action(action: str) -> bool:
    return action.strip().startswith("message(")


def _format_action_entry(entry: dict[str, str], *, include_tool_result: bool = False) -> str:
    action = entry["action"]
    if not include_tool_result:
        return action
    tool_result = entry.get("tool_result")
    if isinstance(tool_result, str) and tool_result.strip():
        prefix = "[USER EDIT — infer preferences from these changes]\n" if not _is_user_message_action(action) else ""
        return f"{action}\n{prefix}tool_result: {tool_result.strip()}"
    return action


def _build_action_log(
    entries: list[dict[str, str]],
    *,
    actors: set[str] | None = None,
    msg_only: bool = False,
    include_tool_results: bool = False,
) -> str:
    actions: list[str] = []
    for entry in entries:
        actor = entry.get("actor") or "user"
        if actors is not None and actor not in actors:
            continue
        action = entry["action"]
        if msg_only and actor == "user" and not _is_user_message_action(action):
            continue
        actions.append(_format_action_entry(entry, include_tool_result=include_tool_results))
    return "\n".join(f"{i + 1}. {a}" for i, a in enumerate(actions))


def build_context_inputs(data: Any, *, msg_only: bool = False) -> list[dict[str, Any]]:
    """Rows: ``name``, ``task``, ``action_entries`` (``actor``, ``action``, optional ``tool_result``), ``source``."""
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
        action_entries: list[dict[str, str]] = []
        for entry in raw_traj:
            if not isinstance(entry, dict) or not isinstance(entry.get("action"), str):
                continue
            row_entry: dict[str, str] = {
                "actor": str(entry.get("actor") or "user"),
                "action": entry["action"],
            }
            tool_result = entry.get("tool_result")
            if isinstance(tool_result, str) and tool_result.strip():
                row_entry["tool_result"] = tool_result
            action_entries.append(row_entry)
        if not action_entries:
            continue
        sid = blob.get("uuid")
        source = sid.strip() if isinstance(sid, str) and sid.strip() else f"session_{i}"
        task_blob = blob.get("task")
        task_str = task_blob.strip() if isinstance(task_blob, str) else ""
        expertise_task = blob.get("expertise_task")
        expertise_str = expertise_task.strip() if isinstance(expertise_task, str) else ""
        rows.append(
            {
                "name": name_str,
                "task": task_str,
                "action_entries": action_entries,
                "source": source,
                "expertise_task": expertise_str,
            }
        )
    return rows


MEMORY_SYSTEM = """From the task description and the numbered action log, write up to 8 short facts or user preferences worth remembering later.

Your primary job is to extract NEW information from the current session Log (user messages, edits, styling choices, corrections, tools used, preferences). The existing memory file—if any—is background only.

Two kinds of user evidence — weigh both; do not ignore direct edits:
1. message("...") actions: stated preferences in the user's own words.
2. Non-message actions (edit(...), brain_edit(), etc.) with tool_result: the user changed files directly. Read tool_result carefully — diffs, removed/added lines, CSS/property changes, deleted memory/skill text — and infer preferences from what they changed, not only what they typed.
When tool_result shows concrete changes (e.g. font-size 18px → 22px, grid removed, colors changed), turn those into Preference: lines even if no message says it explicitly.

Example:
  action: edit("chart.html")
  tool_result: font-size: 18px → font-size: 22px; grid: { display: true } → { display: false }
→ Preference: User prefers larger chart text and no gridlines.

When an existing memory file is provided:
- Mine the facts/preferences not already captured (mandatory).
- Edit originals when this session refines or contradicts them.
Output the updated memory entries (not repeating current content).

Output rules:
- One fact per line. Plain text only (NO markdown headers like # or ##).
- Each line is a single sentence (no numbered lists in the sense of "1." as list markers—use plain sentences).
- Add prefixes "Fact:" or "Preference:" to each line.
- If nothing is worth saving, output exactly the single word NONE (nothing else).
- Do not include reasoning, analysis, or a thinking process. Start directly with Title: (or NONE).

Reply with:
Title: <short task name>
- Fact: <item>
- Preference: <item>
...
"""

MEMORY_CONSOLIDATE_SYSTEM = """Merge new induction entries into the existing memory file and produce a refined, cross-session version.

You receive (i) the original memory file and (ii) new entries derived from a new session.
Merge (ii) into (i): keep durable prior preferences, add genuinely new ones, and lightly deduplicate.
Output line count should stay about the same as (i) or slightly more, but definitely substantially less than (i) and (ii) combineds.

Keep:
- Specific user preferences that apply to certain contexts
- Recurring styling or workflow habits (e.g. larger fonts, no gridlines, compact layout) across tasks
- General facts about how the user works

Remove or merge only when necessary:
- Nonsensical or contradictory entries
- Highly task-specific details unlikely to help elsewhere (e.g. a color for one named column/bar)
- Duplicate lines that say the same thing in different words.
- Make each line concise but do not over-compress away useful detail.

Output rules:
- One entry per line. Plain text only (NO markdown headers like # or ##).
- Prefix each line with "Fact:" or "Preference:".
- Do not include reasoning or a thinking process.

Reply with:
Title: <short topic name>
- Fact: <item>
- Preference: <item>
...
"""

SKILL_SYSTEM = """From the task and numbered log, describe the workflow the agent used: ordered steps, generalized (no long paths).

Your primary job is to capture what THIS session did—especially techniques, fixes, and steps visible in the Log. The existing skill file—if any—is background only.

When an existing skill file is provided:
- FIRST update the workflow using concrete steps from this session's Log (mandatory when the log is non-empty).
- Add or revise steps for anything new in this session (e.g. chart tweaks, file edits, verification, user-requested changes).
- Adopt the useful parts from original steps only when still accurate; merge duplicates.
- Do not return the existing skill unchanged if the log shows new agent work.
Output the full updated skill (not a diff). Generalize: no long paths, file paths, or raw code.

Reply with:
Title: <short task name>
1. <step>
2. <step>
...
If nothing fits: NONE"""


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
_MEMORY_ANSWER_LINE_RE = re.compile(
    r"^\s*(?:[-*+•]|\d+[\.)]\s+)?(?:\*{1,2})?(?:title|fact|preference)\s*:",
    re.I,
)
_LABEL_PREFIX_RE = re.compile(
    r"^(?:\*{1,2})?(?:preference|fact)(?:\*{1,2}:|:\*{0,2}|\s*:)\s*",
    re.I,
)


def _strip_memory_preamble(text: str) -> str:
    """Drop leading thinking/reasoning before the structured memory answer."""
    t = text.strip()
    if not t:
        return t
    lines = t.replace("\r\n", "\n").split("\n")
    if not re.match(r"^thinking\s+process\s*:", lines[0].strip(), re.I):
        return t
    for i, line in enumerate(lines):
        if _MEMORY_ANSWER_LINE_RE.match(line):
            return "\n".join(lines[i:]).strip()
    return ""


def _normalize_memory_line(line: str) -> str | None:
    s = line.strip()
    if not s:
        return None
    while _NUM_BULLET_RE.match(s):
        s = _NUM_BULLET_RE.sub("", s, count=1).strip()
    if not _LABEL_PREFIX_RE.match(s):
        return None
    return s or None


def _parse_memory_lines(raw: str) -> list[str]:
    blob = _strip_memory_preamble(_strip_outer_fences(raw))
    if not blob or blob.strip().upper() == "NONE":
        return []
    out: list[str] = []
    for line in blob.replace("\r\n", "\n").split("\n"):
        s = _normalize_memory_line(line)
        if s:
            out.append(s)
    return out


def _clip_existing(text: str) -> str:
    s = text.strip()
    if len(s) <= _MAX_EXISTING_FILE_CHARS:
        return s
    return s[:_MAX_EXISTING_FILE_CHARS] + "\n...[truncated]"


def _existing_file_block(label: str, content: str) -> str:
    body = _clip_existing(content)
    if not body:
        return ""
    return (
        f"{label} (background only—do not copy back unchanged; prioritize new learnings from the Log below):\n"
        f"{body}\n\n"
    )


def _merge_tail_instruction(has_existing: bool, kind: str) -> str:
    if not has_existing:
        return ""
    if kind == "memory":
        return (
            "\nInstruction: Output ONLY new memory facts/preferences from this session's Log. "
            "Use the existing file to skip duplicates—do not repeat or revise prior lines.\n"
        )
    return (
        f"\nMerge instruction: The Log above is from a NEW session. "
        f"Extract fresh workflow steps from it. Keep prior file content only when still useful; "
        f"edit or replace stale lines. Do not return the prior file unchanged if the log has agent actions.\n"
    )


def extract_memories(
    runtime: ResolvedRuntimeLlm, task: str, log: str, *, existing_memory: str = ""
) -> list[str]:
    task_block = (task or "").strip() or "(no title)"
    has_existing = bool(existing_memory.strip())
    user = (
        f"Task / session title:\n{task_block}\n\n"
        f"{_existing_file_block('Existing memory file', existing_memory)}"
        "User action log (messages + direct edits; for edit/brain_edit steps, "
        "mine tool_result diffs for preferences):\n"
        f"{log or '(empty)'}\n"
        f"{_merge_tail_instruction(has_existing, 'memory')}"
    )
    try:
        raw = runtime_llm_text(runtime, MEMORY_SYSTEM, user, max_tokens=2048)
    except Exception:
        logger.exception("Memory LLM failed")
        return []
    out = _parse_memory_lines(raw)
    if not out and raw.strip() and raw.strip().upper() not in ("NONE",):
        logger.warning(
            "Memory extraction produced 0 lines after parsing (model returned non-empty text). Preview: %s",
            raw.strip()[:500],
        )
    return out[:8 if has_existing else 6]


def _log_memory_consolidation_diff(
    stem: str, before_lines: list[str], after_lines: list[str]
) -> None:
    if before_lines == after_lines:
        logger.info(
            "Memory consolidation (%s): no line changes (%d lines)", stem, len(after_lines)
        )
        return
    before_set = set(before_lines)
    after_set = set(after_lines)
    removed = [ln for ln in before_lines if ln not in after_set]
    added = [ln for ln in after_lines if ln not in before_set]
    unchanged = len(before_set & after_set)
    parts = [
        f"Memory consolidation ({stem}): {len(before_lines)} → {len(after_lines)} lines "
        f"({unchanged} unchanged)",
        "─" * 72,
    ]
    if removed:
        parts.append("  removed:")
        parts.extend(f"    − {ln}" for ln in removed)
    if added:
        parts.append("  added:")
        parts.extend(f"    + {ln}" for ln in added)
    if not removed and not added:
        parts.append("  (lines reordered only)")
    parts.append("─" * 72)
    logger.info("\n%s", "\n".join(parts))


def consolidate_memory_file(
    runtime: ResolvedRuntimeLlm,
    path: Path,
    *,
    original_content: str,
    new_entries: list[str],
) -> list[str]:
    """Second LM pass: merge new entries into the pre-induction memory file."""
    original = original_content.strip()
    new_block = "\n\n".join(new_entries).strip()
    if not original and not new_block:
        return []
    before_lines = _parse_memory_lines(original) + new_entries
    user = (
        f"(i) Original memory file (before this induction round):\n"
        f"{original or '(empty)'}\n\n"
        f"(ii) New entries from this induction round:\n"
        f"{new_block or '(none)'}\n"
    )
    try:
        raw = runtime_llm_text(runtime, MEMORY_CONSOLIDATE_SYSTEM, user, max_tokens=2048)
    except Exception:
        logger.exception("Memory consolidation LLM failed")
        return []
    out = _parse_memory_lines(raw)
    if not out:
        if raw.strip().upper() == "NONE":
            _log_memory_consolidation_diff(path.stem, before_lines, [])
            path.write_text("", encoding="utf-8")
        elif raw.strip():
            logger.warning(
                "Memory consolidation produced 0 lines after parsing; keeping file. Preview: %s",
                raw.strip()[:500],
            )
        return []
    _log_memory_consolidation_diff(path.stem, before_lines, out)
    path.write_text("\n\n".join(out) + "\n", encoding="utf-8")
    return out


def extract_skill(
    runtime: ResolvedRuntimeLlm, task: str, log: str, *, existing_skill: str = ""
) -> tuple[str, list[str]] | None:
    task_block = (task or "").strip() or "(no title)"
    has_existing = bool(existing_skill.strip())
    user = (
        f"Task:\n{task_block}\n\n"
        f"{_existing_file_block('Existing skill file', existing_skill)}"
        f"Log (primary source—capture what this session did):\n{log or '(empty)'}\n\n"
        f"{_merge_tail_instruction(has_existing, 'skill')}"
        "Use Title: plus numbered steps only.\n"
    )
    try:
        raw = runtime_llm_text(runtime, SKILL_SYSTEM, user, max_tokens=2048)
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


def _normalize_task_stem(value: str) -> str | None:
    s = (value or "").strip().lower()
    if s and _TASK_STEM_RE.match(s):
        return s
    return None


def _output_stem(row: dict[str, Any]) -> str:
    expertise = row.get("expertise_task")
    if isinstance(expertise, str):
        stem = _normalize_task_stem(expertise)
        if stem:
            return stem
    name = row.get("name")
    name_str = name if isinstance(name, str) else ""
    source = row.get("source")
    source_str = source if isinstance(source, str) else "session"
    return _slug(name_str, source_str)


def _read_existing_outputs(out: Path, stem: str) -> tuple[str, str]:
    mem_path = out / "memories" / f"{stem}.md"
    skill_path = out / "skills" / f"{stem}.md"
    memory = mem_path.read_text(encoding="utf-8") if mem_path.is_file() else ""
    skill = skill_path.read_text(encoding="utf-8") if skill_path.is_file() else ""
    if memory.strip().startswith("## Auto memory (fallback)"):
        memory = ""
    if skill.strip().startswith("Auto-induction fallback skill"):
        skill = ""
    return memory, skill


def _append_memory_file(path: Path, memories: list[str]) -> None:
    if not memories:
        return
    new_block = "\n\n".join(memories) + "\n"
    prefix = ""
    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        if existing.strip():
            prefix = "\n\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(prefix + new_block)


def _slug(name: str, fallback: str) -> str:
    s = (name or "").strip()
    if len(s) >= 2 and s[0] in "\"'" and s[0] == s[-1]:
        try:
            s = json.loads(s)
        except json.JSONDecodeError:
            s = s[1:-1]
    s = re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")
    return (s or re.sub(r"[^a-z0-9]+", "-", fallback.lower()).strip("-") or "session")[:100]


def main() -> None:
    p = argparse.ArgumentParser(description="Extract memories & skills from session JSON")
    p.add_argument("--data_path", required=True, help="Path to session JSON")
    p.add_argument("--output_dir", default=None, help="Output root (default: app userData)")
    p.add_argument(
        "--model",
        default=None,
        help="Override model; claude* uses Anthropic, otherwise Pi/Tinker runtime",
    )
    p.add_argument(
        "--msg_only",
        action="store_true",
        help="Only include user message(...) actions; drop other user actions (edit, brain_edit, etc.)",
    )
    p.add_argument(
        "--memory_only",
        action="store_true",
        help="Only induce memory files; skip skill extraction",
    )
    args = p.parse_args()
    load_dotenv()

    with open(args.data_path, encoding="utf-8") as f:
        raw = json.load(f)
    inputs = build_context_inputs(raw, msg_only=args.msg_only)
    if not inputs:
        logger.warning("Nothing to extract.")
        return

    try:
        runtime = resolve_induce_llm(model_override=args.model)
    except (AnthropicConfigError, PiLlmConfigError) as e:
        logger.error("%s", e)
        raise SystemExit(1) from e
    logger.info(
        "Induce LLM: provider=%s model=%s msg_only=%s memory_only=%s",
        runtime.provider,
        runtime.model,
        args.msg_only,
        args.memory_only,
    )

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
            task_for_llm = (name or "").strip() if isinstance(name, str) else ""
        entries = row.get("action_entries") or []
        memory_log = _build_action_log(
            entries, actors={"user"}, msg_only=args.msg_only, include_tool_results=True
        )
        base = _output_stem(row)
        expertise_stem = _normalize_task_stem(
            row.get("expertise_task") if isinstance(row.get("expertise_task"), str) else ""
        )
        if expertise_stem:
            stem = expertise_stem
        else:
            n = seen.get(base, 0)
            seen[base] = n + 1
            stem = base if n == 0 else f"{base}-{''.join(c for c in src if c.isalnum())[:8] or n}"

        existing_memory, existing_skill = _read_existing_outputs(out, stem)
        memories = extract_memories(
            runtime, task_for_llm, memory_log, existing_memory=existing_memory
        )
        mem_path = mem_dir / f"{stem}.md"
        _append_memory_file(mem_path, memories)
        if memories:
            consolidate_memory_file(
                runtime,
                mem_path,
                original_content=existing_memory,
                new_entries=memories,
            )

        if not args.memory_only:
            skill_log = _build_action_log(entries, msg_only=args.msg_only)
            skill = extract_skill(
                runtime, task_for_llm, skill_log, existing_skill=existing_skill
            )
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
