"""Shared verifier/rubric LM labeling prompt fragments and results validation."""

from __future__ import annotations

from typing import Any


def format_numbered_lines(lines: list[str]) -> str:
    """1-based numbered list for model-facing criteria/rubric lines."""
    return "\n".join(f"{i}. {line}" for i, line in enumerate(lines, start=1))


def results_array_instructions(*, count: int, item_word: str = "criterion") -> str:
    """How long the JSON ``results`` array must be and how it maps to numbered lines."""
    if count <= 0:
        return ""
    plural = f"{item_word}s"
    return (
        f"There are {count} {plural} below. The results array must contain exactly {count} objects "
        f"(no more, no fewer).\n"
        f"results[0] is the verdict for {item_word} 1, results[1] for {item_word} 2, "
        f"..., results[{count - 1}] for {item_word} {count}."
    )


def validate_results_length(results: list[Any], expected: int, *, label: str = "results") -> None:
    if len(results) != expected:
        raise ValueError(f"Expected exactly {expected} entries in {label}, got {len(results)}")


def results_length_retry_hint(expected: int) -> str:
    word = "criterion" if expected == 1 else "criteria"
    return (
        f"Your previous reply had the wrong number of results entries. "
        f"Return exactly {expected} objects in results — one per numbered line "
        f"(1 through {expected}) in the {word} list above."
    )
