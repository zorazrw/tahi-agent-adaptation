"""Reward computation for weight-based task units.

R_total = w_success * verifier_pass_rate + w_efficiency * exp(-alpha * n_follow_ups)
"""

from __future__ import annotations

import math
from typing import Any


def compute_reward(
    unit: dict,
    w_success: float = 0.6,
    w_efficiency: float = 0.4,
    alpha: float = 0.5,
) -> float:
    """Compute a bounded scalar reward for one task_unit."""
    verifiers = unit.get("verifiers", [])
    if verifiers:
        n_pass = sum(1 for v in verifiers if v.get("status") is True)
        r_success = n_pass / len(verifiers)
    else:
        r_success = 1.0

    human_traj = unit.get("human_trajectories", [])
    n_follow_ups = sum(1 for h in human_traj if h.get("type") == "follow_up")
    r_efficiency = math.exp(-alpha * n_follow_ups)

    return w_success * r_success + w_efficiency * r_efficiency


def compute_per_round_rewards(
    unit: dict,
    strategy: str = "shared",
    base_fraction: float = 0.3,
    **kwargs: Any,
) -> list[float]:
    """Compute per-round reward list.

    strategy="shared": all rounds get the same reward.
    strategy="progressive": linearly increasing, last round gets full reward.
    """
    total = compute_reward(unit, **kwargs)
    n = len(unit.get("agent_trajectories", []))
    if n == 0:
        return []
    if strategy == "progressive" and n > 1:
        base = total * base_fraction
        return [base + (i / (n - 1)) * (total - base) for i in range(n)]
    return [total] * n
