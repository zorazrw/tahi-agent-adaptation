"""
Extract memories and skills from session JSON (e.g. ``out.json``).

Accepts export shape ``{ uuid, name, trajectory }``, weight-based ``{ uuid, name, task_units, ... }``
(a JSON array of those objects), or legacy ``{ sessions: [...] }``.
Outputs: ``<output>/memories/<slug>.md`` and ``skills/<slug>.md``.

Per memory file: (1) extract new entries per session, (2) merge after each session,
(3) final cross-session polish after all sessions complete.

Use ``--finalize_memory_only`` to run only pass (3) on existing ``memories/*.md`` files
(e.g. after a prior full induction) without re-reading session JSON.

When export JSON includes ``expertise_task`` (e.g. ``data-viz-html``), writes to that stem
instead of slugifying the session title, and includes existing memory/skill file content in the LM prompt.

Requires: python-dotenv and ``anthropic``.

Model: defaults to ``claude-sonnet-4-5`` (same as verifier-generator), overridable via ``--model``.
Always uses Anthropic credentials from ``pi-agent/auth.json`` (in-app Settings), then env —
independent of the task-solving model in Settings.
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
    ResolvedRuntimeLlm,
    default_agent_cowork_user_data,
    runtime_llm_text,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_MODEL = "claude-sonnet-4-5"
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
    """Anthropic Sonnet + ``auth.json``, like verifier-generator — not the task-solving model."""
    model = (model_override or "").strip() or DEFAULT_MODEL
    cfg = resolve_anthropic_config()
    return ResolvedRuntimeLlm(
        provider="anthropic",
        model=model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
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


def _is_user_message_action(action: str) -> bool:
    return action.strip().startswith("message(")


_PLAN_EDIT_PREFIXES = ("edit_workflow(", "edit_plan(")


def _tool_result_prefix(action: str) -> str:
    a = action.strip()
    if _is_user_message_action(a):
        return ""
    if a.startswith("edit_verifier("):
        return "[USER EDIT — verifier rubric diff (Verifiers region)]\n"
    if any(a.startswith(p) for p in _PLAN_EDIT_PREFIXES):
        return "[USER EDIT — workflow plan diff (Progress region)]\n"
    return "[USER EDIT — infer preferences from these changes]\n"


# formatQuotedSelectionMessage forms: "Quote (from path):…" or "Quote:…"
# (often under "Human comments on text files:"). Require the trailing ":" so casual
# phrases like "Quote (author, year) style" are not treated as file-quote actions.
_QUOTE_COMMENT_RE = re.compile(r"(?:^|[\n\"])Quote\s*(?:\([^)\n]*\))?:")


def _is_quote_comment_action(action: str) -> bool:
    """True for message(...) rows that carry inline file-quote comments, not chat prefs."""
    a = action.strip()
    if not _is_user_message_action(a):
        return False
    if "Human comments on text files:" in a:
        return True
    return _QUOTE_COMMENT_RE.search(a) is not None


def _is_task_setup_message(action: str) -> bool:
    """True for the initial task prompt that pastes paper Title + Introduction.

    Usually the first user message(...) in a writing session (sometimes repeated);
    not a preference signal under --msg_only.
    """
    a = action.strip()
    if not _is_user_message_action(a):
        return False
    # Canonical writing dumps: "...given the title and introduction... Title: ... Introduction: ..."
    if "Title:" in a and "Introduction:" in a:
        return True
    return "Write an abstract of the paper" in a and "Introduction:" in a


def _is_msg_only_user_action(action: str) -> bool:
    """True for preference-bearing message(...) actions kept under --msg_only.

    Drops every other action type (edit_workflow/edit_plan/edit_verifier, file edits,
    Quote / file-comment messages, initial Title/Introduction task dumps, etc.).
    """
    a = action.strip()
    if not _is_user_message_action(a):
        return False
    if _is_quote_comment_action(a) or _is_task_setup_message(a):
        return False
    return True


def _format_action_entry(entry: dict[str, str], *, include_tool_result: bool = False) -> str:
    action = entry["action"]
    if not include_tool_result:
        return action
    tool_result = entry.get("tool_result")
    if isinstance(tool_result, str) and tool_result.strip():
        prefix = _tool_result_prefix(action)
        return f"{action}\n{prefix}tool_result: {tool_result.strip()}"
    return action


def _build_action_log(
    entries: list[dict[str, str]],
    *,
    actors: set[str] | None = None,
    include_tool_results: bool = False,
) -> str:
    actions: list[str] = []
    for entry in entries:
        actor = entry.get("actor") or "user"
        if actors is not None and actor not in actors:
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
            if msg_only and not _is_msg_only_user_action(row_entry["action"]):
                continue
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
2. Non-message actions with tool_result: the user changed artifacts directly. Read tool_result carefully:
   - edit(...), brain_edit(), etc.: file diffs (styling, content, deleted memory/skill text).
   - edit_workflow() / edit_plan(): workflow plan diff (Progress region — added/removed/reordered steps).
   - edit_verifier(): rubric diff (Verifiers region — criterion text and pass/fail status changes).
Infer preferences from what they changed, not only what they typed.
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
Output line count should stay about the same as the original memory (i). Do not add excessive new entries.

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

MEMORY_FINAL_CONSOLIDATE_SYSTEM = """Refine the complete memory file produced by multiple induction sessions into a concise, high-quality cross-session reference.

You receive the full memory file accumulated across sessions. Produce a polished final version suitable for loading into future agent context.

Keep:
- Durable user preferences (layout, typography, styling habits) that recur across chart types
- General facts about how the user works (e.g. iterative refinement, verifier editing habits)
- Context-scoped rules when the scope is clear (e.g. "in bar charts" vs "in curve plots")

Remove or merge aggressively:
- Duplicate or near-duplicate lines (merge into one clearer line)
- Highly task-specific facts (dataset names, axis tick values for one chart, specific model names or scores)
- Implementation/library details (Chart.js APIs: afterDraw, beginPath, getPixelForValue, plugin methods)
- Overly granular pixel-level rules that repeat a general theme already captured elsewhere

Target the smallest line set that preserves all distinct preferences. Prefer fewer, broader lines over many narrow ones.

Output rules:
- One entry per line. Plain text only (NO markdown headers like # or ##).
- Prefix each line with "Fact:" or "Preference:".
- Do not include reasoning or a thinking process.

Reply with:
Title: <short topic name>
- Fact: <item>
- Preference: <item>
...
If nothing is worth keeping: NONE"""

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
        "User action log (messages + direct edits; for edit/brain_edit, edit_workflow/edit_plan, "
        "and edit_verifier steps, mine tool_result diffs for preferences):\n"
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


def finalize_memory_file(runtime: ResolvedRuntimeLlm, path: Path) -> list[str]:
    """Third LM pass: polish the complete memory file after all sessions are processed."""
    if not path.is_file():
        return []
    original = path.read_text(encoding="utf-8").strip()
    if not original:
        return []
    before_lines = _parse_memory_lines(original)
    if len(before_lines) < 2:
        return before_lines
    user = f"Complete memory file (all sessions merged):\n{original}\n"
    try:
        raw = runtime_llm_text(
            runtime, MEMORY_FINAL_CONSOLIDATE_SYSTEM, user, max_tokens=4096
        )
    except Exception:
        logger.exception("Final memory consolidation LLM failed")
        return before_lines
    out = _parse_memory_lines(raw)
    if not out:
        if raw.strip().upper() == "NONE":
            _log_memory_consolidation_diff(f"{path.stem} (final)", before_lines, [])
            path.write_text("", encoding="utf-8")
        elif raw.strip():
            logger.warning(
                "Final memory consolidation produced 0 lines after parsing; keeping file. Preview: %s",
                raw.strip()[:500],
            )
        return before_lines
    _log_memory_consolidation_diff(f"{path.stem} (final)", before_lines, out)
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


def _memory_paths_to_finalize(mem_dir: Path, stems: list[str] | None) -> list[Path]:
    if not mem_dir.is_dir():
        return []
    if stems:
        paths = [mem_dir / f"{stem}.md" for stem in stems]
        return [p for p in paths if p.is_file()]
    return sorted(mem_dir.glob("*.md"))


def run_finalize_memory_only(
    runtime: ResolvedRuntimeLlm,
    out: Path,
    *,
    stems: list[str] | None = None,
) -> int:
    """Run pass (3) only: refine existing memory files without session JSON."""
    mem_dir = out / "memories"
    paths = _memory_paths_to_finalize(mem_dir, stems)
    if not paths:
        if stems:
            logger.warning("No memory files found for stems %s under %s", stems, mem_dir)
        else:
            logger.warning("No memory files found under %s", mem_dir)
        return 0

    refined = 0
    for path in paths:
        content = path.read_text(encoding="utf-8").strip()
        if not content or content.startswith("## Auto memory (fallback)"):
            logger.info("Skipping %s (empty or fallback)", path.name)
            continue
        before = len(_parse_memory_lines(content))
        after_lines = finalize_memory_file(runtime, path)
        if after_lines:
            refined += 1
            logger.info("Finalized %s (%d → %d lines)", path.name, before, len(after_lines))
    return refined


def main() -> None:
    p = argparse.ArgumentParser(description="Extract memories & skills from session JSON")
    p.add_argument(
        "--data_path",
        help="Path to session JSON (not required with --finalize_memory_only)",
    )
    p.add_argument("--output_dir", default=None, help="Output root (default: app userData)")
    p.add_argument(
        "--model",
        default=None,
        help=f"Override induce model (default: {DEFAULT_MODEL}; always uses Anthropic auth)",
    )
    p.add_argument(
        "--msg_only",
        action="store_true",
        help="Only keep message(...) actions; drop everything else "
        "(edit_workflow, edit_plan, edit_verifier, file edits, Quote / file-comment messages, "
        "initial Title/Introduction task dumps, etc.)",
    )
    p.add_argument(
        "--memory_only",
        action="store_true",
        help="Only induce memory files; skip skill extraction",
    )
    p.add_argument(
        "--finalize_memory_only",
        action="store_true",
        help="Only run final memory refinement (pass 3) on existing memories/*.md; "
        "does not read session JSON or re-induce",
    )
    p.add_argument(
        "--memory_stem",
        nargs="+",
        default=None,
        metavar="STEM",
        help="With --finalize_memory_only: refine only these memory file stem(s) "
        "(default: all *.md under memories/)",
    )
    args = p.parse_args()
    load_dotenv()

    if args.finalize_memory_only and args.data_path:
        logger.warning("--data_path is ignored with --finalize_memory_only")
    if not args.finalize_memory_only and not args.data_path:
        p.error("--data_path is required unless --finalize_memory_only is set")

    try:
        runtime = resolve_induce_llm(model_override=args.model)
    except AnthropicConfigError as e:
        logger.error("%s", e)
        raise SystemExit(1) from e

    out = (Path(args.output_dir) if args.output_dir else default_agent_cowork_user_data()).expanduser().resolve()

    if args.finalize_memory_only:
        logger.info(
            "Finalize memory only: provider=%s model=%s output=%s stems=%s",
            runtime.provider,
            runtime.model,
            out,
            args.memory_stem or "all",
        )
        out.mkdir(parents=True, exist_ok=True)
        (out / "memories").mkdir(parents=True, exist_ok=True)
        n = run_finalize_memory_only(runtime, out, stems=args.memory_stem)
        logger.info("Done: finalized %d memory file(s) → %s", n, out)
        return

    with open(args.data_path, encoding="utf-8") as f:
        raw = json.load(f)
    inputs = build_context_inputs(raw, msg_only=args.msg_only)
    if not inputs:
        logger.warning("Nothing to extract.")
        return

    logger.info(
        "Induce LLM: provider=%s model=%s msg_only=%s memory_only=%s",
        runtime.provider,
        runtime.model,
        args.msg_only,
        args.memory_only,
    )

    out.mkdir(parents=True, exist_ok=True)
    mem_dir, sk_dir = out / "memories", out / "skills"
    mem_dir.mkdir(parents=True, exist_ok=True)
    sk_dir.mkdir(parents=True, exist_ok=True)
    seen: dict[str, int] = {}
    touched_memory_stems: set[str] = set()
    nm, ns = 0, 0

    for row in inputs:
        name = row["name"]
        src = row["source"]
        task_for_llm = (row.get("task") or "").strip() if isinstance(row.get("task"), str) else ""
        if not task_for_llm:
            task_for_llm = (name or "").strip() if isinstance(name, str) else ""
        entries = row.get("action_entries") or []
        memory_log = _build_action_log(
            entries, actors={"user"}, include_tool_results=True
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
            skill_log = _build_action_log(entries)
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
        touched_memory_stems.add(stem)
        logger.info("%s → %s.md", src, stem)

    for stem in sorted(touched_memory_stems):
        finalize_memory_file(runtime, mem_dir / f"{stem}.md")

    logger.info("Done: %d memory lines, %d skills → %s", nm, ns, out)


if __name__ == "__main__":
    main()
