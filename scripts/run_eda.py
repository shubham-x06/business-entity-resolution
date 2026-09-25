#!/usr/bin/env python
"""
run_eda.py  –  Exploratory Data Analysis over the full entity-resolution dataset.

Generates ``experiments/eda_report.md`` with:
  1. Per-file row counts (all 6 source files + ground truth)
  2. Country distribution per file
  3. Match-count distribution & singleton rate from ground truth
  4. Empty-business_address rate in S2/S3 (train & test)
  5. Match split per S1 entity between S2 vs S3

Usage
-----
    python scripts/run_eda.py --data-dir dataset/
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# ── Helpers ─────────────────────────────────────────────────────────

def _load(path: Path) -> pd.DataFrame:
    """Load a TSV with all strings, no NA coercion."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)


def _pct(n: int, total: int) -> str:
    """Format a percentage to two decimals."""
    if total == 0:
        return "0.00%"
    return f"{100.0 * n / total:.2f}%"


def _comma(n: int) -> str:
    """Format an integer with comma separators."""
    return f"{n:,}"


# ── Analysis functions ──────────────────────────────────────────────

def row_counts(data_dir: Path) -> List[Tuple[str, int]]:
    """Return (relative_path, row_count) for every TSV."""
    files = [
        "train/train_source1.tsv",
        "train/train_source2.tsv",
        "train/train_source3.tsv",
        "train/train_ground_truth.tsv",
        "test/test_source1.tsv",
        "test/test_source2.tsv",
        "test/test_source3.tsv",
    ]
    results = []
    for rel in files:
        p = data_dir / rel
        if p.is_file():
            df = _load(p)
            results.append((rel, len(df)))
        else:
            results.append((rel, -1))
    return results


def country_distribution(data_dir: Path) -> Dict[str, Dict[str, int]]:
    """Return {file_label: {country: count}}."""
    files = {
        "train_source1": "train/train_source1.tsv",
        "train_source2": "train/train_source2.tsv",
        "train_source3": "train/train_source3.tsv",
        "test_source1": "test/test_source1.tsv",
        "test_source2": "test/test_source2.tsv",
        "test_source3": "test/test_source3.tsv",
    }
    result: Dict[str, Dict[str, int]] = {}
    for label, rel in files.items():
        p = data_dir / rel
        if p.is_file():
            df = _load(p)
            result[label] = dict(df["country"].value_counts().items())
    return result


def match_count_distribution(
    data_dir: Path,
) -> Tuple[pd.Series, int, int, float, float, int]:
    """Analyse the ground truth.

    Returns
    -------
    match_counts : pd.Series
        Number of matched IDs per S1 entity (0 for singletons).
    n_total : int
    n_singletons : int
    singleton_rate : float  (percentage)
    mean_matches : float    (over ALL entities, including singletons)
    max_matches : int
    """
    gt_path = data_dir / "train" / "train_ground_truth.tsv"
    gt = _load(gt_path)

    def _count_ids(s: str) -> int:
        if s == "":
            return 0
        return len(s.split(","))

    match_counts = gt["matched_entity_ids"].apply(_count_ids)
    n_total = len(gt)
    n_singletons = int((match_counts == 0).sum())
    singleton_rate = 100.0 * n_singletons / n_total if n_total else 0.0
    mean_matches = float(match_counts.mean()) if n_total else 0.0
    max_matches = int(match_counts.max()) if n_total else 0

    return match_counts, n_total, n_singletons, singleton_rate, mean_matches, max_matches


def s2_s3_split(data_dir: Path) -> Tuple[float, float, float, float]:
    """Mean S2 and S3 matches per S1 entity (all entities and non-singletons).

    Returns
    -------
    mean_s2_all : float
        Mean S2 matches across all S1 entities (including singletons) ≈ 1.68.
    mean_s3_all : float
        Mean S3 matches across all S1 entities (including singletons) ≈ 1.79.
    mean_s2_ns : float
        Mean S2 matches across non-singleton S1 entities ≈ 1.77.
    mean_s3_ns : float
        Mean S3 matches across non-singleton S1 entities ≈ 1.89.
    """
    gt_path = data_dir / "train" / "train_ground_truth.tsv"
    gt = _load(gt_path)

    s2_counts = []
    s3_counts = []
    for ids_str in gt["matched_entity_ids"]:
        if ids_str == "":
            continue
        ids = ids_str.split(",")
        s2 = sum(1 for i in ids if i.startswith("S2-"))
        s3 = sum(1 for i in ids if i.startswith("S3-"))
        s2_counts.append(s2)
        s3_counts.append(s3)

    n_all = len(gt)
    n_ns = len(s2_counts)
    mean_s2_all = sum(s2_counts) / n_all if n_all else 0.0
    mean_s3_all = sum(s3_counts) / n_all if n_all else 0.0
    mean_s2_ns = sum(s2_counts) / n_ns if n_ns else 0.0
    mean_s3_ns = sum(s3_counts) / n_ns if n_ns else 0.0
    return mean_s2_all, mean_s3_all, mean_s2_ns, mean_s3_ns


def empty_address_rate(data_dir: Path) -> Dict[str, Tuple[int, int, float]]:
    """Empty-business_address counts for S2/S3 train & test.

    Returns
    -------
    dict
        ``{label: (n_empty, n_total, pct)}``
    """
    files = {
        "train_source2": "train/train_source2.tsv",
        "train_source3": "train/train_source3.tsv",
        "test_source2": "test/test_source2.tsv",
        "test_source3": "test/test_source3.tsv",
    }
    result: Dict[str, Tuple[int, int, float]] = {}
    for label, rel in files.items():
        p = data_dir / rel
        if p.is_file():
            df = _load(p)
            n_empty = int((df["business_address"] == "").sum())
            n_total = len(df)
            pct = 100.0 * n_empty / n_total if n_total else 0.0
            result[label] = (n_empty, n_total, pct)
    return result


def match_histogram_table(match_counts: pd.Series) -> str:
    """Build a markdown table of match-count frequencies."""
    freq = match_counts.value_counts().sort_index()
    lines = ["| Matches | Count | % |", "|------:|------:|-----:|"]
    total = len(match_counts)
    for n_matches, count in freq.items():
        lines.append(
            f"| {n_matches} | {_comma(count)} | {_pct(count, total)} |"
        )
    return "\n".join(lines)


# ── Report generator ────────────────────────────────────────────────

def generate_report(data_dir: Path) -> str:
    """Build the full EDA report as a markdown string."""
    sections: List[str] = []
    sections.append("# Entity Resolution — EDA Report\n")
    sections.append(f"**Data directory:** `{data_dir}`\n")

    # 1. Row counts
    sections.append("## 1. Row Counts\n")
    rc = row_counts(data_dir)
    lines = ["| File | Rows |", "|------|-----:|"]
    for rel, count in rc:
        lines.append(f"| `{rel}` | {_comma(count) if count >= 0 else 'MISSING'} |")
    sections.append("\n".join(lines) + "\n")

    # 2. Country distribution
    sections.append("## 2. Country Distribution\n")
    cd = country_distribution(data_dir)
    for label in sorted(cd):
        sections.append(f"### {label}\n")
        tbl_lines = ["| Country | Count |", "|---------|------:|"]
        for country, cnt in sorted(cd[label].items(), key=lambda x: -x[1]):
            tbl_lines.append(f"| {country} | {_comma(cnt)} |")
        sections.append("\n".join(tbl_lines) + "\n")

    # 3. Match-count distribution
    sections.append("## 3. Match-Count Distribution (Ground Truth)\n")
    match_counts, n_total, n_singletons, singleton_rate, mean_matches, max_matches = (
        match_count_distribution(data_dir)
    )
    sections.append(f"- **Total S1 entities:** {_comma(n_total)}")
    sections.append(f"- **Singletons (0 matches):** {_comma(n_singletons)} ({singleton_rate:.2f}%)")
    sections.append(f"- **Mean matches/entity:** {mean_matches:.2f}")
    sections.append(f"- **Max matches/entity:** {max_matches}")
    sections.append("")
    sections.append(match_histogram_table(match_counts) + "\n")

    # 4. S2 vs S3 split
    sections.append("## 4. Match Split: S2 vs S3\n")
    mean_s2_all, mean_s3_all, mean_s2_ns, mean_s3_ns = s2_s3_split(data_dir)
    sections.append(f"- **All S1 entities (including singletons):**")
    sections.append(f"  - Mean S2 matches per entity: {mean_s2_all:.2f}")
    sections.append(f"  - Mean S3 matches per entity: {mean_s3_all:.2f}")
    sections.append(f"  - Combined mean matches: {mean_s2_all + mean_s3_all:.2f}")
    sections.append(f"- **Non-singleton S1 entities (at least 1 match):**")
    sections.append(f"  - Mean S2 matches per entity: {mean_s2_ns:.2f}")
    sections.append(f"  - Mean S3 matches per entity: {mean_s3_ns:.2f}")
    sections.append(f"  - Combined mean matches: {mean_s2_ns + mean_s3_ns:.2f}\n")

    # 5. Empty-address rate
    sections.append("## 5. Empty-Address Rate (S2 / S3)\n")
    ea = empty_address_rate(data_dir)
    ea_lines = ["| File | Empty | Total | Rate |", "|------|------:|------:|-----:|"]
    # Group train and test
    train_empty = 0
    train_total = 0
    test_empty = 0
    test_total = 0
    for label in sorted(ea):
        n_empty, n_total_file, pct = ea[label]
        ea_lines.append(f"| {label} | {_comma(n_empty)} | {_comma(n_total_file)} | {pct:.2f}% |")
        if "train" in label:
            train_empty += n_empty
            train_total += n_total_file
        else:
            test_empty += n_empty
            test_total += n_total_file

    # Aggregate rows
    if train_total:
        train_pct = 100.0 * train_empty / train_total
        ea_lines.append(f"| **train S2+S3** | **{_comma(train_empty)}** | **{_comma(train_total)}** | **{train_pct:.2f}%** |")
    if test_total:
        test_pct = 100.0 * test_empty / test_total
        ea_lines.append(f"| **test S2+S3** | **{_comma(test_empty)}** | **{_comma(test_total)}** | **{test_pct:.2f}%** |")
    sections.append("\n".join(ea_lines) + "\n")

    return "\n".join(sections)


# ── CLI ─────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA and generate report.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to the dataset/ directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the report (default: experiments/eda_report.md).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_dir = args.data_dir.resolve()

    # Default output: <repo_root>/experiments/eda_report.md
    if args.output is None:
        repo_root = Path(__file__).resolve().parents[1]
        out_path = repo_root / "experiments" / "eda_report.md"
    else:
        out_path = args.output.resolve()

    print(f"[EDA] Data directory: {data_dir}")
    print(f"[EDA] Output: {out_path}")

    report = generate_report(data_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"[EDA] Report written to {out_path}")

    # Print a summary to stdout as well
    print("\n" + report)


if __name__ == "__main__":
    main()
