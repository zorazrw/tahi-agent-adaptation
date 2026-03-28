"""
Persistent knowledge store for memory and skill entries.

Modelled after ReasoningBank's ``memory.py``: dataclass entries serialised to
flat JSON arrays on disk, with ``to_dict`` / ``from_dict`` round-tripping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class MemoryEntry:
    content: str
    type: str  # "preference" | "fact" | "constraint"
    evidence: str
    source: str  # e.g. "session_0/unit_0"
    created_at: str = ""
    embedding: Optional[list[float]] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> MemoryEntry:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SkillEntry:
    title: str
    steps: list[str]
    evidence: str
    source: str
    created_at: str = ""
    embedding: Optional[list[float]] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    @property
    def description(self) -> str:
        return self.steps[0] if self.steps else self.title

    def to_dict(self) -> dict:
        d = asdict(self)
        d["description"] = self.description
        return d

    @classmethod
    def from_dict(cls, data: dict) -> SkillEntry:
        data.pop("description", None)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def format_memories_markdown(
    memories: list[MemoryEntry],
    heading: str = "Memories",
) -> str:
    lines = [f"# {heading}", ""]
    for i, m in enumerate(memories, 1):
        lines.extend([
            f"## {i}. {m.type}",
            "",
            m.content,
            "",
            f"- **Evidence:** {m.evidence}",
            f"- **Source:** {m.source}",
            f"- **Created at:** {m.created_at}",
            "",
            "---",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def format_skills_markdown(
    skills: list[SkillEntry],
    heading: str = "Skills",
) -> str:
    lines = [f"# {heading}", ""]
    for i, s in enumerate(skills, 1):
        lines.extend([f"## {i}. {s.title}", ""])
        for j, step in enumerate(s.steps, 1):
            lines.append(f"{j}. {step}")
        lines.extend([
            "",
            f"- **Evidence:** {s.evidence}",
            f"- **Source:** {s.source}",
            f"- **Created at:** {s.created_at}",
            "",
            "---",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


class KnowledgeStore:
    """JSON-backed store for memories and skills."""

    def __init__(
        self,
        memory_path: str | Path = "memory.json",
        skills_path: str | Path = "skills.json",
    ):
        self.memory_path = Path(memory_path)
        self.skills_path = Path(skills_path)
        self.memories: list[MemoryEntry] = []
        self.skills: list[SkillEntry] = []
        self.load()

    def _ensure_parents(self):
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.skills_path.parent.mkdir(parents=True, exist_ok=True)

    def add_memories(self, entries: list[MemoryEntry]):
        for e in entries:
            if not self._memory_exists(e.content):
                self.memories.append(e)
        self.save()

    def add_skills(self, entries: list[SkillEntry]):
        for e in entries:
            if not self._skill_exists(e.title):
                self.skills.append(e)
        self.save()

    def get_all_memories(self) -> list[MemoryEntry]:
        return list(self.memories)

    def get_all_skills(self) -> list[SkillEntry]:
        return list(self.skills)

    def save(self):
        self._ensure_parents()
        if self.memory_path.suffix.lower() == ".md":
            self.memory_path.write_text(
                format_memories_markdown(self.memories),
                encoding="utf-8",
            )
        else:
            with open(self.memory_path, "w") as f:
                json.dump(
                    [m.to_dict() for m in self.memories],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        if self.skills_path.suffix.lower() == ".md":
            self.skills_path.write_text(
                format_skills_markdown(self.skills),
                encoding="utf-8",
            )
        else:
            with open(self.skills_path, "w") as f:
                json.dump(
                    [s.to_dict() for s in self.skills],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

    def load(self):
        if self.memory_path.suffix.lower() == ".md":
            self.memories = []
        else:
            self.memories = self._load_list(self.memory_path, MemoryEntry)
        if self.skills_path.suffix.lower() == ".md":
            self.skills = []
        else:
            self.skills = self._load_list(self.skills_path, SkillEntry)

    @staticmethod
    def _load_list(path: Path, cls):
        try:
            if path.exists():
                with open(path) as f:
                    return [cls.from_dict(d) for d in json.load(f)]
        except (json.JSONDecodeError, TypeError):
            pass
        return []

    def _memory_exists(self, content: str) -> bool:
        return any(m.content == content for m in self.memories)

    def _skill_exists(self, title: str) -> bool:
        return any(s.title == title for s in self.skills)

    def clear(self):
        self.memories = []
        self.skills = []
        self.save()

    def __repr__(self) -> str:
        return f"KnowledgeStore(memories={len(self.memories)}, skills={len(self.skills)})"
