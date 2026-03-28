"""
LLM-based knowledge inducer: extract memories and skills from human-agent
collaboration trajectories.

Two separate prompts (memory / skill) following the ReasoningBank extractor
pattern: one class, two extraction methods, structured text output parsed
with field markers.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any

from context.store import MemoryEntry, SkillEntry

logger = logging.getLogger(__name__)

# ── Prompt templates ────────────────────────────────────────────────

MEMORY_SYSTEM = """\
You are an expert at analyzing human-agent collaboration to extract declarative knowledge — user preferences, domain facts, and quality constraints.

You will receive a task description and the interleaved trajectory of agent actions, environment observations, and human feedback. The human is a domain expert whose interventions reveal tacit knowledge that the agent should internalize.

## Guidelines
Extract preferences, facts, or constraints that would help the agent on future similar tasks.

## Important notes
- The human's actions are the primary signal — analyze what the human corrected, added, or rejected.
- Focus on generalizable insights, not task-specific details.
- Do not mention specific file paths, URLs, or content verbatim. Abstract to patterns.
- Extract at most 5 memory items. Only extract what is clearly supported by evidence.
- If no meaningful memory can be extracted, output nothing."""

SKILL_SYSTEM = """\
You are an expert at analyzing human-agent collaboration to extract procedural knowledge — reusable multi-step workflows and strategies.

You will receive a task description and the interleaved trajectory of agent actions, environment observations, and human feedback. The human is a domain expert whose interventions reveal effective procedures the agent should learn.

## Guidelines
Extract reusable procedures or strategies that describe HOW to accomplish similar tasks.

## Important notes
- Focus on the process, not the content. Abstract specific actions to general patterns.
- If the agent improved between rounds (after human feedback), extract what it did differently.
- Do not mention specific file paths, URLs, or content verbatim.
- Extract at most 3 skills. Only extract what is clearly supported by evidence.
- If no meaningful skill can be extracted, output nothing."""

MEMORY_USER_TEMPLATE = """\
Task: {task}
Current step: {intent}
Verifiers:
{verifiers}

Trajectory:
{trajectory}

Extract memories in this format:

MEMORY 1:
TYPE: <preference | fact | constraint>
CONTENT: <one sentence describing the preference, fact, or constraint>
EVIDENCE: <which human action or signal led to this memory>

MEMORY 2:
...

(Extract at most 5 memories. If none, output: NO MEMORIES.)"""

SKILL_USER_TEMPLATE = """\
Task: {task}
Current step: {intent}
Verifiers:
{verifiers}

Trajectory:
{trajectory}

Extract skills in this format:

SKILL 1:
TITLE: <short name for the skill>
STEPS:
1. <first step>
2. <second step>
...
EVIDENCE: <which actions or feedback led to this skill>

SKILL 2:
...

(Extract at most 3 skills. If none, output: NO SKILLS.)"""


# ── Inducer class ───────────────────────────────────────────────────

class KnowledgeInducer:
    """Extract memory and skill entries via LLM calls."""

    def __init__(self, client, model: str = "claude-sonnet-4-5-20250929"):
        self.client = client
        self.model = model

    def _call_llm(self, system: str, user: str) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts: list[str] = []
        for block in msg.content:
            if block.type == "text":
                parts.append(block.text)
        return "".join(parts)

    # ── Memory extraction ───────────────────────────────────────────

    def extract_memories(
        self, task: str, intent: str, verifiers: str, trajectory: str, source: str,
    ) -> list[MemoryEntry]:
        user_msg = MEMORY_USER_TEMPLATE.format(
            task=task, intent=intent, verifiers=verifiers, trajectory=trajectory,
        )
        try:
            raw = self._call_llm(MEMORY_SYSTEM, user_msg)
        except Exception:
            logger.exception("LLM call failed for memory extraction (source=%s)", source)
            return []
        return self._parse_memories(raw, source)

    @staticmethod
    def _parse_memories(text: str, source: str) -> list[MemoryEntry]:
        if _explicit_no_memories(text):
            return []
        entries: list[MemoryEntry] = []
        parts = re.split(r"(?i)MEMORY\s+\d+\s*:?", text)
        for part in parts[1:]:
            content = _extract_field(part, "CONTENT:")
            mem_type = _extract_field(part, "TYPE:") or "preference"
            evidence = _extract_field(part, "EVIDENCE:")
            if not content:
                continue
            mem_type = mem_type.strip().lower()
            if mem_type not in ("preference", "fact", "constraint"):
                mem_type = "preference"
            entries.append(MemoryEntry(
                content=content.strip(),
                type=mem_type,
                evidence=evidence.strip(),
                source=source,
            ))
        return entries

    # ── Skill extraction ────────────────────────────────────────────

    def extract_skills(
        self, task: str, intent: str, verifiers: str, trajectory: str, source: str,
    ) -> list[SkillEntry]:
        user_msg = SKILL_USER_TEMPLATE.format(
            task=task, intent=intent, verifiers=verifiers, trajectory=trajectory,
        )
        try:
            raw = self._call_llm(SKILL_SYSTEM, user_msg)
        except Exception:
            logger.exception("LLM call failed for skill extraction (source=%s)", source)
            return []
        return self._parse_skills(raw, source)

    @staticmethod
    def _parse_skills(text: str, source: str) -> list[SkillEntry]:
        if _explicit_no_skills(text):
            return []
        entries: list[SkillEntry] = []
        parts = re.split(r"(?i)SKILL\s+\d+\s*:?", text)
        for part in parts[1:]:
            title = _extract_field(part, "TITLE:")
            evidence = _extract_field(part, "EVIDENCE:")
            steps = _extract_steps(part)
            if not title or not steps:
                continue
            entries.append(SkillEntry(
                title=title.strip(),
                steps=steps,
                evidence=evidence.strip(),
                source=source,
            ))
        return entries


# ── Parsing helpers ─────────────────────────────────────────────────

# Next structured field after current value (case-insensitive); stops multi-line CONTENT/STEPS correctly.
_NEXT_FIELD_RE = re.compile(
    r"(?is)(?:^|[\n\r])\s*(?:"
    r"TYPE\s*:|CONTENT\s*:|EVIDENCE\s*:|TITLE\s*:|STEPS\s*:|"
    r"MEMORY\s+\d+\s*:?|SKILL\s+\d+\s*:?"
    r")"
)

_NO_MEMORIES_LINE = re.compile(r"(?im)^\s*NO MEMORIES\.?\s*$")
_NO_SKILLS_LINE = re.compile(r"(?im)^\s*NO SKILLS\.?\s*$")


def _explicit_no_memories(text: str) -> bool:
    """True only for explicit opt-out, not phrases like 'no memories to extract'."""
    t = text.strip()
    if re.fullmatch(r"(?i)NO MEMORIES\.?", t):
        return True
    return bool(_NO_MEMORIES_LINE.search(text))


def _explicit_no_skills(text: str) -> bool:
    t = text.strip()
    if re.fullmatch(r"(?i)NO SKILLS\.?", t):
        return True
    return bool(_NO_SKILLS_LINE.search(text))


def _extract_field(text: str, field_name: str) -> str:
    """Slice text after ``FIELD:`` until the next known field marker (any case)."""
    key = field_name.rstrip(":").strip()
    start_m = re.search(rf"(?i){re.escape(key)}\s*:\s*", text)
    if not start_m:
        return ""
    start = start_m.end()
    rest = text[start:]
    next_m = _NEXT_FIELD_RE.search(rest)
    if next_m:
        return rest[: next_m.start()].strip()
    return rest.strip()


def _extract_steps(text: str) -> list[str]:
    """Extract numbered steps from the STEPS: block."""
    raw = _extract_field(text, "STEPS:")
    if not raw:
        return []
    steps: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
        if cleaned:
            steps.append(cleaned)
    return steps
