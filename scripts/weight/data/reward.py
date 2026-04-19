"""Reward computation for weight-based task units.

R_total = w_success * verifier_pass_rate + w_efficiency * exp(-alpha * n_follow_ups)

Per-trajectory variant uses each trajectory's own verifiers (finer-grained signal)
and positional decay via follow-ups-after-round-k instead of a flat efficiency term.
"""

from __future__ import annotations

import math
from typing import Any


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
