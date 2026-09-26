#!/usr/bin/env python
"""
run_train.py  –  CLI entry-point for model training & evaluation.

Usage
-----
    python scripts/run_train.py [--stage {all,blocking,features,train,eval}]
                                [--root ROOT] [--sample N]

Stages
------
1. blocking : Generate candidate pairs from S1 × (S2 ∪ S3) using country-
              partitioned multi-signal inverted index.
              Writes output/candidate_pairs.tsv and computes candidate-set
              reduction ratio + recall@K on train_ground_truth.tsv.
2. features : Compute pairwise feature matrix from candidate pairs.
3. train    : Train LightGBM ranker/classifier on features.
4. eval     : Evaluate on validation set and log metrics.
5. all      : Run stages 1 → 4 sequentially (default).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# ── Logging setup ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list[str] | None
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Train the business-entity resolution model.",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "blocking", "features", "train", "eval"],
        default="all",
        help="Which pipeline stage to execute. Default: 'all'.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Repository root (student_resource/).  Default: auto-detected.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Subsample N rows from Source 1 for fast prototyping.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional identifier for this experiment run.",
    )
    parser.add_argument(
        "--country",
        type=str,
        default=None,
        help="Optional country filter ('India', 'US'). Restricts feature extraction to S1 entities from that country.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Maximum candidate pairs to process in feature extraction (for sampling/benchmarking).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Chunk size (number of candidate pairs) per feature extraction batch. Default: 100,000.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output destination path (.parquet or .tsv) for extracted features.",
    )
    parser.add_argument(
        "--candidate-pairs-path",
        type=Path,
        default=None,
        help=(
            "Override path to the candidate pairs TSV file. "
            "Default: output/candidate_pairs.tsv (combined). "
            "Use checkpoints/candidates_india.tsv or candidates_us.tsv "
            "to run a single partition without the full 5.7GB combined file."
        ),
    )
    return parser.parse_args(argv)


# ── Stage runners ───────────────────────────────────────────────────

def run_blocking_stage(
    root: Path,
    sample: int | None = None,
    run_id: str | None = None,
) -> Dict[str, Any]:
    """Execute Stage 1: Candidate blocking on real dataset."""
    from business_entity_resolution.blocking import generate_candidate_pairs
    from business_entity_resolution.io_utils import (
        load_config,
        load_ground_truth,
        load_source,
        write_candidate_pairs,
    )

    cfg_dir = root / "code" / "business_entity_resolution" / "configs"
    paths_cfg = load_config(cfg_dir / "paths.yaml")
    blocking_cfg = load_config(cfg_dir / "blocking.yaml")

    # ── Load sources ────────────────────────────────────────────────
    data_cfg = paths_cfg["data"]["train"]
    s1 = load_source(root / data_cfg["source1"], expected_source=1)
    if sample is not None and sample > 0:
        s1 = s1.head(sample).copy()
        print(f"[blocking] Sampled Source 1 to {len(s1):,} rows")
    s2 = load_source(root / data_cfg["source2"], expected_source=2)
    s3 = load_source(root / data_cfg["source3"], expected_source=3)

    print(f"[blocking] S1: {len(s1):,} rows | S2: {len(s2):,} rows | S3: {len(s3):,} rows")

    # ── Generate candidates ─────────────────────────────────────────
    t0 = time.time()
    out_dir = root / "output"
    out_cand_file = out_dir / "candidate_pairs.tsv"
    ckpt_dir = (root / "checkpoints") if (sample is None or sample <= 0) else None

    candidates = generate_candidate_pairs(
        s1, s2, s3, blocking_cfg,
        checkpoint_dir=ckpt_dir,
        output_path=out_cand_file,
    )
    elapsed = time.time() - t0

    # Ensure final candidate file exists
    if not out_cand_file.is_file() or out_cand_file.stat().st_size == 0:
        write_candidate_pairs(candidates, out_cand_file)

    # ── Compute metrics ─────────────────────────────────────────────
    n_pairs = sum(len(v) for v in candidates.values())
    n_with = sum(1 for v in candidates.values() if v)
    cands_per_s1 = [len(v) for v in candidates.values()]
    mean_cands = n_pairs / len(candidates) if candidates else 0

    import statistics
    median_cands = statistics.median(cands_per_s1) if cands_per_s1 else 0

    total_ref = len(s2) + len(s3)
    max_possible = len(s1) * total_ref
    reduction_ratio = 1.0 - (n_pairs / max_possible) if max_possible > 0 else 0.0

    # ── Evaluate recall against ground truth ────────────────────────
    gt_path = root / data_cfg["ground_truth"]
    recall = None
    if gt_path.is_file():
        gt = load_ground_truth(gt_path)
        total_true_pairs = 0
        max_k = blocking_cfg.get("candidate_ranking", {}).get("max_candidates_per_entity", 200)
        k_eval_list = [k for k in [50, 100, 150, 200, 300, 500] if k <= max_k]
        if max_k not in k_eval_list:
            k_eval_list.append(max_k)
        k_eval_list.sort()
        found_true_pairs_k = {k: 0 for k in k_eval_list}
        
        for _, row in gt.iterrows():
            s1_id = row["source1_entity_id"]
            if s1_id not in candidates:
                continue
            matched_ids_str = row["matched_entity_ids"]
            if not matched_ids_str:
                continue
            true_matches = set(matched_ids_str.split(","))
            total_true_pairs += len(true_matches)
            
            cand_list = candidates.get(s1_id, [])
            for k in found_true_pairs_k.keys():
                cand_set_k = set(cand_list[:k])
                found_true_pairs_k[k] += len(true_matches & cand_set_k)

        print("\n" + "=" * 60)
        print("BLOCKING EVALUATION REPORT")
        print("=" * 60)
        print(f"S1 Entities Evaluated:      {len(candidates):,}")
        print(f"Total True Pairs Evaluated: {total_true_pairs:,}")
        print(f"Total Candidate Pairs:      {n_pairs:,}")
        print(f"Mean Candidates / S1:       {mean_cands:.2f}")
        print(f"Median Candidates / S1:     {median_cands:.1f}")
        print(f"Reduction Ratio:            {reduction_ratio * 100:.4f}%")
        print(f"Elapsed Time:               {elapsed:.1f}s ({len(s1)/elapsed:.1f} queries/s)")
        print("-" * 60)
        for k in k_eval_list:
            rec_k = found_true_pairs_k[k] / total_true_pairs if total_true_pairs > 0 else 0.0
            print(f"  Recall@{k:<4}: {rec_k * 100:.2f}% ({found_true_pairs_k[k]:,}/{total_true_pairs:,} true pairs)")
        print("=" * 60 + "\n")
        recall = {f"recall@{k}": found_true_pairs_k[k] / total_true_pairs for k in k_eval_list} if total_true_pairs > 0 else {}

    return {
        "n_s1": len(s1),
        "n_pairs": n_pairs,
        "mean_candidates": mean_cands,
        "median_candidates": median_cands,
        "reduction_ratio": reduction_ratio,
        "elapsed_seconds": elapsed,
        "recall": recall,
    }


def run_features_stage(
    root: Path,
    country: str | None = None,
    max_pairs: int | None = None,
    chunk_size: int = 100000,
    output_path: Path | None = None,
    candidate_pairs_path: Path | None = None,
    run_id: str | None = None,
) -> Dict[str, Any]:
    """Execute Stage 2: Vectorized chunked pairwise feature extraction."""
    from business_entity_resolution.features import (
        load_and_normalize_entity_cache,
        extract_features_streaming,
    )

    # Allow callers to point at a country-specific checkpoint (e.g. checkpoints/candidates_india.tsv)
    # instead of the full combined output/candidate_pairs.tsv (5.7 GB).
    cand_path = candidate_pairs_path or (root / "output" / "candidate_pairs.tsv")
    c_tag = country.lower() if country else "all"
    out = output_path or (root / "output" / f"features_{c_tag}.parquet")

    print(f"\n[features] Loading and normalizing entity cache (country={country or 'ALL'})...")
    entity_cache, vectorizer = load_and_normalize_entity_cache(
        root=root,
        country_filter=country,
    )

    print(f"[features] Streaming candidate pairs: {cand_path} -> {out}")
    if candidate_pairs_path:
        print(f"[features] NOTE: using explicit --candidate-pairs-path override (not the combined file).")
    print(f"[features] Chunk size: {chunk_size:,} | Max pairs: {max_pairs or 'ALL'}")

    stats = extract_features_streaming(
        candidate_pairs_path=cand_path,
        entity_cache=entity_cache,
        output_path=out,
        country_filter=country,
        vectorizer=vectorizer,
        chunk_size=chunk_size,
        max_pairs=max_pairs,
    )

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION STAGE REPORT")
    print("=" * 60)
    print(f"Country Filter:        {country or 'ALL'}")
    print(f"Total Pairs Processed: {stats['total_pairs']:,}")
    print(f"S1 Entities Processed: {stats['total_s1_entities']:,}")
    print(f"Elapsed Time:          {stats['elapsed_seconds']:.2f} s")
    print(f"Processing Speed:      {stats['pairs_per_sec']:,.0f} pairs/sec")
    print(f"Peak Memory Usage:     {stats['peak_memory_mb']:.1f} MB")
    print(f"Output File:           {stats['output_path']}")

    total_440m = 440091505
    if stats["pairs_per_sec"] > 0:
        est_full_sec = total_440m / stats["pairs_per_sec"]
        print(f"Extrapolated Full 440M Runtime: {est_full_sec / 60.0:.1f} minutes ({est_full_sec / 3600.0:.2f} hours)")
    print("=" * 60 + "\n")

    return stats


def main(argv: list[str] | None = None) -> None:
    """Entry-point for the training pipeline."""
    args = parse_args(argv)
    print(f"[run_train] root = {args.root}")
    print(f"[run_train] stage = {args.stage}")
    if args.country:
        print(f"[run_train] country = {args.country}")
    if args.sample:
        print(f"[run_train] sample = {args.sample}")
    if args.max_pairs:
        print(f"[run_train] max_pairs = {args.max_pairs}")
    if args.run_id:
        print(f"[run_train] run_id = {args.run_id}")

    # Add src to sys.path so package imports resolve cleanly
    src_dir = args.root / "code" / "business_entity_resolution" / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    if args.stage in ("blocking", "all"):
        run_blocking_stage(args.root, sample=args.sample, run_id=args.run_id)
        if args.stage == "blocking":
            return

    if args.stage in ("features", "all"):
        run_features_stage(
            root=args.root,
            country=args.country,
            max_pairs=args.max_pairs,
            chunk_size=args.chunk_size,
            output_path=args.output,
            candidate_pairs_path=args.candidate_pairs_path,
            run_id=args.run_id,
        )
        if args.stage == "features":
            return

    if args.stage in ("train", "all"):
        print("[run_train] Stage 'train' not yet implemented (Milestone 7).")
        if args.stage == "train":
            return

    if args.stage in ("eval", "all"):
        print("[run_train] Stage 'eval' not yet implemented (Milestone 8).")
        if args.stage == "eval":
            return


if __name__ == "__main__":
    main()
