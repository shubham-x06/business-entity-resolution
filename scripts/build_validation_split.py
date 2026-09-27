#!/usr/bin/env python
"""
scripts/build_validation_split.py  –  Stratified validation entity split builder.

Builds a reproducible, stratified hold-out split of Source 1 entities for validation.
Stratifies S1 entities by (country, singleton/has-match) into 4 strata:
  - US-singleton
  - US-has-match
  - India-singleton
  - India-has-match

Outputs the held-out entity_id list to experiments/validation_entities.txt
(one entity_id per line) and reports exact counts and percentages per stratum.

Usage
-----
    python scripts/build_validation_split.py [--val-ratio 0.10] [--seed 42]
                                            [--output experiments/validation_entities.txt]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_validation_split")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Build stratified validation split for Source 1 entities.",
    )
    parser.add_argument(
        "--source1",
        type=Path,
        default=repo_root / "dataset" / "train" / "train_source1.tsv",
        help="Path to train_source1.tsv",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=repo_root / "dataset" / "train" / "train_ground_truth.tsv",
        help="Path to train_ground_truth.tsv",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Validation sampling fraction per stratum (default: 0.10, ~10-15%% recommended).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible stratification (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "experiments" / "validation_entities.txt",
        help="Destination path for held-out validation entity IDs (default: experiments/validation_entities.txt).",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=repo_root / "experiments" / "validation_split_summary.json",
        help="Optional path to write stratum counts summary JSON.",
    )
    return parser.parse_args(argv)


def load_entity_strata(
    source1_path: Path,
    ground_truth_path: Path,
) -> pd.DataFrame:
    """Load Source 1 entities and ground truth, classifying each into its stratum.

    Strata definition:
        ``{country}-{'singleton' if matched_entity_ids is empty else 'has-match'}``

    Parameters
    ----------
    source1_path : Path
        Path to ``train_source1.tsv``.
    ground_truth_path : Path
        Path to ``train_ground_truth.tsv``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``entity_id``, ``country``, ``matched_entity_ids``,
        ``is_singleton``, ``stratum``.
    """
    logger.info("Loading Source 1 records from %s", source1_path)
    s1_df = pd.read_csv(
        source1_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["entity_id", "country"],
    )

    logger.info("Loading ground truth records from %s", ground_truth_path)
    gt_df = pd.read_csv(
        ground_truth_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["source1_entity_id", "matched_entity_ids"],
    )

    merged = s1_df.merge(
        gt_df,
        left_on="entity_id",
        right_on="source1_entity_id",
        how="left",
    )

    if len(merged) != len(s1_df):
        logger.warning(
            "Row count mismatch after merge: S1 has %d rows, merged has %d rows",
            len(s1_df),
            len(merged),
        )

    # Missing from GT or empty string in matched_entity_ids means singleton
    matched_col = merged["matched_entity_ids"].fillna("").astype(str).str.strip()
    is_singleton = matched_col == ""
    merged["is_singleton"] = is_singleton

    # Build stratum column: e.g. "US-singleton", "US-has-match", "India-singleton", "India-has-match"
    status_label = merged["is_singleton"].map({True: "singleton", False: "has-match"})
    merged["stratum"] = merged["country"] + "-" + status_label

    return merged[["entity_id", "country", "matched_entity_ids", "is_singleton", "stratum"]]


def build_stratified_validation_split(
    df_strata: pd.DataFrame,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Sample held-out validation entities stratified by (country, singleton/has-match).

    Parameters
    ----------
    df_strata : pd.DataFrame
        DataFrame containing ``entity_id`` and ``stratum``.
    val_ratio : float, default 0.10
        Fraction of entities to sample from each stratum (e.g. 0.10 = 10%).
    seed : int, default 42
        Reproducibility seed.

    Returns
    -------
    train_df : pd.DataFrame
        Entities assigned to the training split.
    val_df : pd.DataFrame
        Entities assigned to the validation split.
    summary : dict
        Stratum counts and breakdown metrics.
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be between 0.0 and 1.0, got {val_ratio}")

    logger.info(
        "Sampling validation split with val_ratio=%.4f (seed=%d)...",
        val_ratio,
        seed,
    )

    # Sample uniformly per stratum
    val_df = (
        df_strata.groupby("stratum", group_keys=False)
        .sample(frac=val_ratio, random_state=seed)
        .copy()
    )

    val_entity_set = set(val_df["entity_id"])
    train_df = df_strata[~df_strata["entity_id"].isin(val_entity_set)].copy()

    # Calculate exact stratum counts
    strata_counts: Dict[str, Dict[str, Any]] = {}
    all_strata = sorted(df_strata["stratum"].unique())

    for st in all_strata:
        total_st = int((df_strata["stratum"] == st).sum())
        val_st = int((val_df["stratum"] == st).sum())
        train_st = int((train_df["stratum"] == st).sum())
        pct = (val_st / total_st * 100.0) if total_st > 0 else 0.0

        strata_counts[st] = {
            "total": total_st,
            "train": train_st,
            "val": val_st,
            "val_percentage": round(pct, 3),
        }

    total_entities = len(df_strata)
    total_val = len(val_df)
    total_train = len(train_df)

    summary = {
        "val_ratio_requested": val_ratio,
        "seed": seed,
        "total_s1_entities": total_entities,
        "total_train_entities": total_train,
        "total_val_entities": total_val,
        "overall_val_percentage": round(total_val / total_entities * 100.0, 3),
        "strata": strata_counts,
    }

    return train_df, val_df, summary


def save_validation_entities(
    val_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save validation entity IDs to disk (one per line).

    Parameters
    ----------
    val_df : pd.DataFrame
        DataFrame with an ``entity_id`` column.
    output_path : Path
        Output destination file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_ids = sorted(val_df["entity_id"].astype(str).tolist())
    logger.info("Writing %d validation entity IDs to %s", len(sorted_ids), output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        for eid in sorted_ids:
            f.write(f"{eid}\n")


def print_report(summary: Dict[str, Any]) -> None:
    """Print a clean ASCII summary table of the stratified split."""
    print("\n" + "=" * 78)
    print("STRATIFIED VALIDATION ENTITY SPLIT REPORT")
    print("=" * 78)
    print(f"Sampling Ratio: {summary['val_ratio_requested']:.2%}  |  Random Seed: {summary['seed']}")
    print(f"Total Source 1 Entities: {summary['total_s1_entities']:,}")
    print(f"Train Entities:          {summary['total_train_entities']:,} ({summary['total_train_entities']/summary['total_s1_entities']*100:.2f}%)")
    print(f"Validation Entities:     {summary['total_val_entities']:,} ({summary['overall_val_percentage']:.2f}%)")
    print("-" * 78)
    print(f"{'Stratum':<22} | {'Total':>12} | {'Train':>12} | {'Validation':>12} | {'Val %':>8}")
    print("-" * 78)

    for st, counts in summary["strata"].items():
        print(
            f"{st:<22} | {counts['total']:>12,} | {counts['train']:>12,} | "
            f"{counts['val']:>12,} | {counts['val_percentage']:>7.2f}%"
        )

    print("=" * 78 + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    args = parse_args(argv)

    if not args.source1.is_file():
        logger.error("Source 1 file not found: %s", args.source1)
        return 1
    if not args.ground_truth.is_file():
        logger.error("Ground truth file not found: %s", args.ground_truth)
        return 1

    df_strata = load_entity_strata(args.source1, args.ground_truth)
    train_df, val_df, summary = build_stratified_validation_split(
        df_strata=df_strata,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    save_validation_entities(val_df, args.output)

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved summary JSON to %s", args.summary_json)

    print_report(summary)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
