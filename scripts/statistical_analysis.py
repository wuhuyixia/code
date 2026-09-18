from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy import stats


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p_values[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    w_pos = float(ranks[nonzero > 0].sum())
    w_neg = float(ranks[nonzero < 0].sum())
    denom = w_pos + w_neg
    return 0.0 if denom == 0 else (w_pos - w_neg) / denom


def paired_summary(reference: np.ndarray, comparator: np.ndarray, confidence: float) -> dict:
    if reference.shape != comparator.shape:
        raise ValueError("Paired samples must have identical shapes.")
    if reference.size < 2:
        raise ValueError("At least two paired observations are required.")

    diff = reference - comparator
    n = int(diff.size)
    mean = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))

    alpha = 1.0 - confidence
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    half_width = critical * sd / math.sqrt(n)

    if np.allclose(diff, 0.0):
        p_value = 1.0
    else:
        wilcoxon = stats.wilcoxon(
            diff,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
        p_value = float(wilcoxon.pvalue)

    return {
        "n": n,
        "mean_difference": mean,
        "sample_sd_difference": sd,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "wilcoxon_p": p_value,
        "rank_biserial": rank_biserial(diff),
    }


def read_wide_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not rows:
            raise ValueError(f"No rows found in {path}")
        if reader.fieldnames is None:
            raise ValueError("Input CSV must include a header row.")
        return rows, list(reader.fieldnames)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Paired statistical analysis for manuscript seed-level results. "
            "The input CSV must contain one row per matched seed and one column "
            "per method."
        )
    )
    parser.add_argument("--input", required=True, help="Wide CSV containing matched seed results")
    parser.add_argument("--reference", required=True, help="Reference method column, e.g. Algorithm1")
    parser.add_argument(
        "--comparators",
        nargs="+",
        required=True,
        help="Comparator method columns, e.g. DFedAvg Krum TrimmedMean",
    )
    parser.add_argument("--seed-column", default="seed")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output", default=None, help="Optional output CSV")
    args = parser.parse_args()

    if not (0.0 < args.confidence < 1.0):
        raise ValueError("--confidence must lie in (0, 1).")

    rows, fields = read_wide_csv(Path(args.input))
    required = [args.seed_column, args.reference, *args.comparators]
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    seeds = [row[args.seed_column] for row in rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Each seed must appear exactly once in the input CSV.")

    reference = np.asarray([float(row[args.reference]) for row in rows], dtype=float)
    summaries = []
    raw_p = []

    for comparator in args.comparators:
        values = np.asarray([float(row[comparator]) for row in rows], dtype=float)
        result = paired_summary(reference, values, args.confidence)
        result["comparison"] = f"{args.reference} vs {comparator}"
        summaries.append(result)
        raw_p.append(result["wilcoxon_p"])

    adjusted = holm_adjust(raw_p)
    for result, p_adj in zip(summaries, adjusted):
        result["holm_p"] = p_adj

    columns = [
        "comparison",
        "n",
        "mean_difference",
        "sample_sd_difference",
        "ci_low",
        "ci_high",
        "wilcoxon_p",
        "holm_p",
        "rank_biserial",
    ]

    writer = csv.DictWriter(
        open(args.output, "w", newline="", encoding="utf-8") if args.output else __import__("sys").stdout,
        fieldnames=columns,
    )
    writer.writeheader()
    for result in summaries:
        writer.writerow({key: result[key] for key in columns})


if __name__ == "__main__":
    main()
