from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def read_last_row(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows[-1]


def main():
    parser = argparse.ArgumentParser(description="Summarize final-round metrics across run directories.")
    parser.add_argument("runs", nargs="+", help="Run directories containing round_metrics.csv")
    args = parser.parse_args()

    rows = [read_last_row(Path(run) / "round_metrics.csv") for run in args.runs]
    numeric_keys = [
        "mean_val_accuracy",
        "mean_test_accuracy",
        "queries",
        "forward_queries",
        "backward_calls",
        "message_count",
        "transmitted_bytes",
        "aggregation_seconds",
        "round_seconds",
    ]
    for key in numeric_keys:
        values = [float(row[key]) for row in rows]
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"{key}: mean={mean:.10g}, sample_sd={sd:.10g}, n={len(values)}")


if __name__ == "__main__":
    main()
