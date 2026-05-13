"""Reward computation for weight-based task units.

R_total = w_success * verifier_pass_rate + w_efficiency * exp(-alpha * n_follow_ups)

Per-trajectory variant uses each trajectory's own verifiers (finer-grained signal)
and positional decay via follow-ups-after-round-k instead of a flat efficiency term.

LLM rubric variant (``compute_llm_rubric_file_scores`` / ``compute_llm_rubric_file_reward``):
grades file text merged from ``agent_trajectories[*].output_files`` (``path`` + ``content``),
in round order (later rounds win per path). Uses Anthropic (``scripts/induce.py``); requires ``anthropic``.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _verifier_pass_rate(verifiers: list[dict]) -> float | None:
    """Return pass rate in [0, 1] for a verifier list, or None if empty."""
    if not verifiers:
        return None
    n_pass = sum(1 for v in verifiers if v.get("status") is True)
    return n_pass / len(verifiers)


def compute_reward(
    unit: dict,
    w_success: float = 0.6,
    w_efficiency: float = 0.4,
    alpha: float = 0.5,
) -> float:
    """Compute a bounded scalar reward for one task_unit.

    Uses unit-level verifiers (final state) and total follow-up count.
    All rounds within the unit share this single reward value.
    """
    rate = _verifier_pass_rate(unit.get("verifiers", []))
    r_success = rate if rate is not None else 1.0

    human_traj = unit.get("human_trajectories", [])
    n_follow_ups = sum(1 for h in human_traj if h.get("type") == "follow_up")
    r_efficiency = math.exp(-alpha * n_follow_ups)

    return w_success * r_success + w_efficiency * r_efficiency


def _interpolate_traj_rates(trajs: list[dict], unit_verifiers: list[dict]) -> list[float]:
    """Resolve verifier pass rate for every trajectory, filling gaps by interpolation.

    A trajectory with an empty ``verifiers`` list has no direct evidence of
    quality.  Falling back to unit-level verifiers (the final outcome) is
    over-optimistic: it ignores the fact that the user was still unsatisfied
    enough to keep iterating.  Instead we interpolate between the nearest
    *preceding* and *following* trajectories that *do* have verifier evidence:

    * Both neighbours present → linear midpoint: ``(prev + next) / 2``
    * Only a preceding neighbour → use its rate (can't be better; the user
      followed up, so conservatively assume no improvement)
    * Only a following neighbour → use its rate
    * No traj has verifiers at all → fall back to unit-level (last resort)

    This guarantees that a gap trajectory's ``r_success`` sits strictly between
    its neighbours' rates (when both exist) and never exceeds the unit-level
    optimistic upper bound.
    """
    n = len(trajs)
    raw: list[float | None] = [_verifier_pass_rate(t.get("verifiers", [])) for t in trajs]

    # If every traj is None, fall back to unit-level for all.
    unit_rate = _verifier_pass_rate(unit_verifiers)
    if all(r is None for r in raw):
        fallback = unit_rate if unit_rate is not None else 1.0
        return [fallback] * n

    resolved: list[float] = []
    for k in range(n):
        if raw[k] is not None:
            resolved.append(raw[k])  # type: ignore[arg-type]
            continue

        # Find nearest non-None predecessor and successor.
        prev_rate: float | None = None
        for j in range(k - 1, -1, -1):
            if raw[j] is not None:
                prev_rate = raw[j]
                break

        next_rate: float | None = None
        for j in range(k + 1, n):
            if raw[j] is not None:
                next_rate = raw[j]
                break

        if prev_rate is not None and next_rate is not None:
            resolved.append((prev_rate + next_rate) / 2.0)
        elif prev_rate is not None:
            resolved.append(prev_rate)
        else:
            # next_rate is not None here (we handled the all-None case above).
            resolved.append(next_rate)  # type: ignore[arg-type]

    return resolved

def compute_per_traj_rewards(
    unit: dict,
    w_success: float = 0.6,
    w_efficiency: float = 0.4,
    alpha: float = 0.5,
) -> list[float]:
    """Compute per-trajectory rewards using fine-grained verifier and decay signals.

    For each trajectory k the reward combines:

    * ``r_success``: verifier pass rate resolved via ``_interpolate_traj_rates``.
      Trajectories with no verifiers get a rate interpolated between their
      nearest neighbours — never the over-optimistic unit-level fallback unless
      no trajectory has verifiers at all.

    * ``r_efficiency``: exponential decay based on *follow-ups that occur after
      round k*.  The more human interventions were still needed after a round,
      the lower its efficiency score.

    Variable-length verifier arrays across trajectories are handled naturally:
    each trajectory's pass rate is ``sum(pass) / len(its own verifiers)``.

    Returns a list of floats, one per trajectory (same length as
    ``unit["agent_trajectories"]``).
    """
    trajs = unit.get("agent_trajectories", [])
    if not trajs:
        return []

    human_traj = unit.get("human_trajectories", [])
    traj_rates = _interpolate_traj_rates(trajs, unit.get("verifiers", []))

    rewards: list[float] = []
    for k, r_success in enumerate(traj_rates):
        # A follow-up whose round_index == k means the user was unsatisfied
        # with round k.  Follow-ups with no round_index are treated as global.
        n_after = sum(
            1 for h in human_traj
            if h.get("type") == "follow_up"
            and h.get("round_index", k) >= k
        )
        r_efficiency = math.exp(-alpha * n_after)
        rewards.append(w_success * r_success + w_efficiency * r_efficiency)

    return rewards


def compute_per_round_rewards(
    unit: dict,
    strategy: str = "shared",
    base_fraction: float = 0.3,
    **kwargs: Any,
) -> list[float]:
    """Compute per-round reward list.

    strategy="shared": all rounds get the same reward (unit-level).
    strategy="progressive": linearly increasing, last round gets full reward.
    strategy="per_traj": use ``compute_per_traj_rewards`` (recommended).
    """
    n = len(unit.get("agent_trajectories", []))
    if n == 0:
        return []
    if strategy == "per_traj":
        return compute_per_traj_rewards(unit, **kwargs)
    total = compute_reward(unit, **kwargs)
    if strategy == "progressive" and n > 1:
        base = total * base_fraction
        return [base + (i / (n - 1)) * (total - base) for i in range(n)]
    return [total] * n


# ---------------------------------------------------------------------------
# LLM rubric grading (unit file text vs string rubrics)
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent


def _ensure_scripts_on_path() -> None:
    s = str(_SCRIPTS_DIR)
    if s not in sys.path:
        sys.path.insert(0, s)


def _unit_files_to_blocks(unit: dict[str, Any]) -> list[tuple[str, str]]:
    """Build ``(path, content)`` blocks for rubric grading from ``output_files`` only.

    Each round's ``output_files`` entries use weight shape ``{"path": "...", "content": "..."}``.
    Rounds are walked in order; the same ``path`` in a later round overwrites earlier
    snapshots. Returns ``[]`` when there are no usable files (grader sees no file body).
    """
    merged: dict[str, str] = {}
    for rnd in unit.get("agent_trajectories", []) or []:
        if not isinstance(rnd, dict):
            continue
        for f in rnd.get("output_files") or []:
            if not isinstance(f, dict):
                continue
            path = str(f.get("path", "") or "").strip()
            content = f.get("content")
            if not path or not isinstance(content, str):
                continue
            merged[path] = content

    return sorted(merged.items(), key=lambda kv: kv[0])


def _truncate_file_text(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n... [truncated]"


def _build_rubric_grader_prompt(
    rubrics: list[str],
    file_blocks: list[tuple[str, str]],
    *,
    max_file_chars: int = 14_000,
) -> str:
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(rubrics))
    rendered: list[str] = []
    for rel, text in file_blocks:
        body = _truncate_file_text(text, max_file_chars) if text else "(file missing or empty)"
        rendered.append(f"### {rel}\n\n{body}")
    files_joined = "\n\n---\n\n".join(rendered) if rendered else "(no output files)"

    return "\n".join(
        [
            "You are an automated checker for a coding task output.",
            "Given rubric lines and the current output files (below), decide whether each rubric is satisfied.",
            'Reply with ONLY a JSON object of this exact shape: {"results":[{"pass":true},{"pass":false},...]}',
            "The results array must have exactly one object per rubric line, in the same order (indices 0 .. n-1).",
            "pass: true means satisfied; false means not satisfied.",
            "",
            "Rubric lines (in order):",
            numbered,
            "",
            "Output files and contents:",
            files_joined,
        ]
    )


def _pass_from_result_item(item: Any) -> bool | None:
    """Interpret one ``results`` element: ``bool``, ``{"pass":...}``, or common aliases."""
    if isinstance(item, bool):
        return item
    if isinstance(item, (int, float)) and item in (0, 1):
        return bool(item)
    if not isinstance(item, dict):
        return None
    # Case-insensitive key match for pass-like fields (models vary).
    lower_map = {str(k).lower(): v for k, v in item.items()}
    for alias in ("pass", "passed", "satisfied", "ok", "success"):
        if alias in lower_map:
            v = lower_map[alias]
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, str):
                s = v.strip().lower()
                if s in ("true", "yes", "1", "pass", "ok"):
                    return True
                if s in ("false", "no", "0", "fail"):
                    return False
    return None


def _parse_rubric_results_json(text: str, n: int) -> tuple[list[bool | None], list[Any]]:
    """Extract pass/fail per rubric from model text.

    Returns ``(parsed, raw_results)`` where ``parsed[i]`` is ``True``/``False``/``None``
    (``None`` = model gave no usable score for rubric *i*). ``raw_results`` is the
    decoded ``results`` array for error messages.
    """
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    raw = (fence.group(1) if fence else text).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object in model response")
    parsed_json = json.loads(raw[start : end + 1])
    results = parsed_json.get("results")
    if not isinstance(results, list):
        raise ValueError("Missing results array")
    out: list[bool | None] = [None] * n
    for i in range(n):
        if i >= len(results):
            break
        out[i] = _pass_from_result_item(results[i])
    return out, results


def compute_llm_rubric_file_scores(
    unit: dict[str, Any],
    rubrics: list[str],
    *,
    client: Any | None = None,
    model: str | None = None,
    max_tokens: int = 1024,
    max_file_chars: int = 14_000,
) -> tuple[float, list[float]]:
    """Grade unit file text (see ``_unit_files_to_blocks``) against ``rubrics`` with one LLM call.

    Returns ``(mean_pass, rubric_scores)`` where ``rubric_scores`` has length
    ``len(rubrics)``, each entry ``0.0`` or ``1.0`` (fail/pass per criterion).
    ``mean_pass`` is the arithmetic mean of those scores in ``[0, 1]``.

    Empty ``rubrics`` returns ``(1.0, [])`` (no criteria to violate).

    Uses Anthropic Messages API via ``induce.anthropic_user_text`` when ``client`` /
    ``model`` are omitted (resolves credentials like ``scripts/induce.py``).
    """
    if not rubrics:
        return 1.0, []

    file_blocks = _unit_files_to_blocks(unit)
    user_prompt = _build_rubric_grader_prompt(rubrics, file_blocks, max_file_chars=max_file_chars)

    _ensure_scripts_on_path()
    import induce  # noqa: PLC0415 — optional until this function runs

    if client is None or model is None:
        cfg = induce.resolve_anthropic_config()
        client = induce.make_anthropic_client(cfg)
        model = model or cfg.model

    raw = induce.anthropic_user_text(
        client,
        model,
        user_prompt,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    passes, results_list = _parse_rubric_results_json(raw, len(rubrics))
    missing_idx = [i for i, p in enumerate(passes) if p is None]
    if missing_idx:
        logger.warning(
            "LLM rubric parse: %d/%d usable (missing indices %s); len(results)=%d. "
            "Missing slots scored as 0.0.",
            len(rubrics) - len(missing_idx),
            len(rubrics),
            missing_idx[:20],
            len(results_list),
        )
    scored = [1.0 if p is True else 0.0 for p in passes]
    mean = sum(scored) / len(rubrics)
    return mean, scored


def compute_llm_rubric_file_reward(
    unit: dict[str, Any],
    rubrics: list[str],
    *,
    client: Any | None = None,
    model: str | None = None,
    max_tokens: int = 1024,
    max_file_chars: int = 14_000,
) -> float:
    """Same grading as ``compute_llm_rubric_file_scores``; returns only the mean pass rate."""
    mean, _ = compute_llm_rubric_file_scores(
        unit,
        rubrics,
        client=client,
        model=model,
        max_tokens=max_tokens,
        max_file_chars=max_file_chars,
    )
    return mean


def compute_per_traj_rewards_llm_rubrics(
    unit: dict[str, Any],
    rubrics: list[str],
    **kwargs: Any,
) -> list[float]:
    """Same length as ``unit['agent_trajectories']``; each entry is the shared LLM rubric mean.

    ``rubrics`` is caller-defined (e.g. session-final verifiers from the last task_unit only);
    it need not match ``unit['verifiers']``.

    Drop-in alternative to ``compute_per_traj_rewards`` when rewards should come from
    grading merged ``output_files`` text against string rubrics instead of stored verifier metadata.
    """
    trajs = unit.get("agent_trajectories", [])
    if not trajs:
        return []
    mean, _ = compute_llm_rubric_file_scores(unit, rubrics, **kwargs)
    return [mean] * len(trajs)
