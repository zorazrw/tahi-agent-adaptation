"""
Entry-point for context-based adaptation: extract memories and skills
from human-agent collaboration logs.

Usage
-----
    python extract_context.py \
        --data_path ./out.json \
        --output_dir outputs/context/ \
        --model claude-sonnet-4-5-20250929

Requires: pip install anthropic python-dotenv

Authenticates like the desktop app (runner.ts): api-config.json in Electron
userData, or ~/.claude/settings.json (ANTHROPIC_AUTH_TOKEN + BASE_URL + MODEL),
or ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN in the environment.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from anthropic_config import AnthropicConfigError, make_anthropic_client, resolve_anthropic_config
from data.build_context_inputs import build_context_inputs
from context.inducer import KnowledgeInducer
from context.store import KnowledgeStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def default_agent_cowork_user_data() -> Path:
    """Match Electron userData: macOS Application Support, etc. Prefer existing dirs."""
    home = Path.home()
    if sys.platform == "darwin":
        base = home / "Library" / "Application Support"
        env = os.environ.get("AGENT_COWORK_USER_DATA")
        if env:
            return Path(env).expanduser()
        for name in ("Agent Cowork", "agent-cowork"):
            p = base / name
            if p.is_dir():
                return p
        return base / "agent-cowork"
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", str(home / "AppData" / "Roaming"))
        root = Path(appdata)
        env = os.environ.get("AGENT_COWORK_USER_DATA")
        if env:
            return Path(env).expanduser()
        for name in ("Agent Cowork", "agent-cowork"):
            p = root / name
            if p.is_dir():
                return p
        return root / "agent-cowork"
    env = os.environ.get("AGENT_COWORK_USER_DATA")
    if env:
        return Path(env).expanduser()
    for name in ("Agent Cowork", "agent-cowork"):
        p = home / ".config" / name
        if p.is_dir():
            return p
    return home / ".config" / "agent-cowork"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CoSkill Context-Based Adaptation — Knowledge Extraction")
    p.add_argument("--data_path", type=str, required=True,
                    help="Path to raw session JSON (out.json)")
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Root userData dir for memories/memory.md and skills/skill.md (default: Application Support folder)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Anthropic model id (default: from app API config, else claude-sonnet-4-5-20250929)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    load_dotenv()

    with open(args.data_path, encoding="utf-8") as f:
        raw_json = json.load(f)

    inputs = build_context_inputs(raw_json)
    logger.info("Processing %d task units", len(inputs))

    if not inputs:
        logger.warning("No task units with execution data found.")
        return

    try:
        resolved = resolve_anthropic_config()
    except AnthropicConfigError as e:
        logger.error("%s", e)
        raise SystemExit(1) from e

    logger.info("Anthropic API: %s (model from config unless overridden)", resolved.source)
    client = make_anthropic_client(resolved)
    model = args.model if args.model is not None else resolved.model
    inducer = KnowledgeInducer(client, model=model)

    out_dir = (Path(args.output_dir) if args.output_dir else default_agent_cowork_user_data()).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    store = KnowledgeStore(
        memory_path=out_dir / "memories" / "memory.md",
        skills_path=out_dir / "skills" / "skill.md",
    )

    extraction_log: list[dict] = []

    for item in inputs:
        source = item["source"]
        logger.info("Extracting from %s  (intent: %s)", source, item["intent"][:60])

        memories = inducer.extract_memories(
            task=item["task"],
            intent=item["intent"],
            verifiers=item["verifiers"],
            trajectory=item["trajectory_text"],
            source=source,
        )
        skills = inducer.extract_skills(
            task=item["task"],
            intent=item["intent"],
            verifiers=item["verifiers"],
            trajectory=item["trajectory_text"],
            source=source,
        )

        store.add_memories(memories)
        store.add_skills(skills)

        log_entry = {
            "source": source,
            "intent": item["intent"],
            "memories_extracted": len(memories),
            "skills_extracted": len(skills),
            "memories": [m.to_dict() for m in memories],
            "skills": [s.to_dict() for s in skills],
        }
        extraction_log.append(log_entry)

        for m in memories:
            logger.info("  MEMORY [%s]: %s", m.type, m.content[:80])
        for s in skills:
            logger.info("  SKILL: %s (%d steps)", s.title, len(s.steps))

    log_path = out_dir / "extraction_log.json"
    with open(log_path, "w") as f:
        json.dump(extraction_log, f, indent=2, ensure_ascii=False)

    logger.info(
        "Done. Total: %d memories, %d skills. Saved to %s (memories/memory.md, skills/skill.md)",
        len(store.memories), len(store.skills), out_dir,
    )


if __name__ == "__main__":
    main()
