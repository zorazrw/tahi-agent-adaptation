#!/usr/bin/env python3
"""Plot selected SDFT metrics from a server.log file.

Usage:
  python scripts/plot_sdft_metrics.py
  python scripts/plot_sdft_metrics.py --log server.log --output sdft_metrics.png
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


METRICS = [
    "sdft/mean_student_entropy",
    "sdft/mean_teacher_entropy",
    "sdft/topk_entropy_gap_teacher_minus_student",
    "sdft/topk_overlap_ratio",
    "sdft/total_completion_tokens",
]


def parse_metric_rows(log_path: Path) -> dict[str, list[float]]:
    """Extract metric values from Rich-style table rows in the log."""
    values: dict[str, list[float]] = {metric: [] for metric in METRICS}
    row_pattern = re.compile(r"│\s*(sdft/[^│]+?)\s*│\s*([-+]?\d+(?:\.\d+)?)\s*│")

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = row_pattern.search(line)
            if not match:
                continue

            metric = match.group(1).strip()
            if metric in values:
                values[metric].append(float(match.group(2)))

    return values


def plot_metrics(values: dict[str, list[float]], output_path: Path) -> None:
    """Write one stacked line plot per metric with matplotlib."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep matplotlib cache files out of user-level locations in sandboxed runs.
    mpl_config_dir = output_path.parent / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError("matplotlib is required. Install with: pip install matplotlib") from exc

    non_empty = [(metric, series) for metric, series in values.items() if series]
    if not non_empty:
        raise ValueError("No requested SDFT metrics found in the log.")

    fig_height = max(2.2 * len(non_empty), 6.0)
    fig, axes = plt.subplots(len(non_empty), 1, figsize=(12, fig_height), sharex=True)
    if len(non_empty) == 1:
        axes = [axes]

    for ax, (metric, series) in zip(axes, non_empty):
        x = list(range(1, len(series) + 1))
        ax.plot(x, series, marker="o", linewidth=1.8, markersize=4)
        ax.set_title(metric, loc="left", fontsize=10)
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("metric occurrence in log")
    fig.suptitle("SDFT Metrics from server.log", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default="server.log", type=Path, help="Path to server.log")
    parser.add_argument(
        "--output",
        default="sdft_metrics.png",
        type=Path,
        help="Path to write the plot image",
    )
    args = parser.parse_args()

    if not args.log.exists():
        parser.error(f"log file not found: {args.log}")

    values = parse_metric_rows(args.log)
    plot_metrics(values, args.output)

    counts = ", ".join(f"{metric}={len(series)}" for metric, series in values.items())
    print(f"Wrote {args.output} ({counts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
