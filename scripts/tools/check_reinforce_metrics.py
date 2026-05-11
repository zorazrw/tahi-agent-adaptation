#!/usr/bin/env python3
"""Auto-flag instability patterns in REINFORCE metrics.jsonl logs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _rolling_median(vals: list[float], idx: int, window: int) -> float:
    start = max(0, idx - window + 1)
    return float(median(vals[start : idx + 1]))


def _load_rows(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s:
            continue
        obj = json.loads(s)
        if isinstance(obj, dict):
            out.append(obj)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Flag unstable REINFORCE training metrics.")
    p.add_argument("metrics_jsonl", type=Path)
    p.add_argument("--window", type=int, default=10, help="Rolling window size")
    p.add_argument("--adv-spike-mult", type=float, default=2.0, help="Advantage spike multiplier")
    p.add_argument("--loss-spike-mult", type=float, default=3.0, help="Loss spike multiplier")
    p.add_argument("--baseline-gap-mult", type=float, default=2.0, help="Baseline gap multiplier")
    p.add_argument("--time-spike-mult", type=float, default=2.0, help="Step-time spike multiplier")
    args = p.parse_args()

    rows = _load_rows(args.metrics_jsonl)
    if not rows:
        print("No metrics rows found.")
        return 1

    steps = [int(r.get("step", i)) for i, r in enumerate(rows)]
    adv_vals = [float(r.get("mean_abs_advantage", 0.0)) for r in rows if _is_num(r.get("mean_abs_advantage"))]
    loss_vals = [abs(float(r.get("reinforce_loss", 0.0))) for r in rows if _is_num(r.get("reinforce_loss"))]
    time_vals = [float(r.get("time/step", 0.0)) for r in rows if _is_num(r.get("time/step"))]
    gap_vals = [
        abs(float(r["mean_reward"]) - float(r["baseline_prev"]))
        for r in rows
        if _is_num(r.get("mean_reward")) and _is_num(r.get("baseline_prev"))
    ]

    print(f"rows={len(rows)} step_range={steps[0]}..{steps[-1]}")
    issues: list[str] = []

    for r in rows:
        st = r.get("step", "?")
        for k, v in r.items():
            if _is_num(v) and (math.isnan(float(v)) or math.isinf(float(v))):
                issues.append(f"step {st}: {k} is NaN/Inf")

    for i, r in enumerate(rows):
        st = r.get("step", i)

        if _is_num(r.get("mean_abs_advantage")) and adv_vals:
            cur = float(r["mean_abs_advantage"])
            med = _rolling_median(adv_vals, min(i, len(adv_vals) - 1), args.window)
            if med > 0 and cur > args.adv_spike_mult * med:
                issues.append(
                    f"step {st}: mean_abs_advantage spike {cur:.4f} > {args.adv_spike_mult:.2f}x rolling_median {med:.4f}"
                )

        if _is_num(r.get("reinforce_loss")) and loss_vals:
            cur = abs(float(r["reinforce_loss"]))
            med = _rolling_median(loss_vals, min(i, len(loss_vals) - 1), args.window)
            if med > 0 and cur > args.loss_spike_mult * med:
                issues.append(
                    f"step {st}: |reinforce_loss| spike {cur:.4f} > {args.loss_spike_mult:.2f}x rolling_median {med:.4f}"
                )

        if _is_num(r.get("time/step")) and time_vals:
            cur = float(r["time/step"])
            med = _rolling_median(time_vals, min(i, len(time_vals) - 1), args.window)
            if med > 0 and cur > args.time_spike_mult * med:
                issues.append(
                    f"step {st}: time/step spike {cur:.3f}s > {args.time_spike_mult:.2f}x rolling_median {med:.3f}s"
                )

        if _is_num(r.get("mean_reward")) and _is_num(r.get("baseline_prev")) and gap_vals:
            cur = abs(float(r["mean_reward"]) - float(r["baseline_prev"]))
            med = _rolling_median(gap_vals, min(i, len(gap_vals) - 1), args.window)
            if med > 0 and cur > args.baseline_gap_mult * med:
                issues.append(
                    f"step {st}: baseline lag |mean_reward-baseline_prev|={cur:.4f} > "
                    f"{args.baseline_gap_mult:.2f}x rolling_median {med:.4f}"
                )

    if issues:
        print("status=warnings")
        for x in issues:
            print(f"- {x}")
    else:
        print("status=ok")
        print("No instability warnings triggered.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

