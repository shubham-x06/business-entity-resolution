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
        choices=["all", "blocking", "features", "train", "model", "eval", "threshold-tuning"],
        default="all",
        help="Which pipeline stage to execute. 'all' runs the full orchestrated pipeline. Default: 'all'.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of S1 entities to hold out for stratified validation. Default: 0.10.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.98,
        help="Classification probability threshold for validation evaluation. Default: 0.98.",
    )
    parser.add_argument(
        "--force-mine",
        action="store_true",
        help="Force re-execution of streaming hard-negative mining even if checkpoint exists.",
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


<<<<<<< HEAD
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



def run_model_stage(
    root: Path,
    sample: int | None = None,
    max_pairs: int | None = None,
    run_id: str | None = None,
    val_ratio: float = 0.1,
    threshold: float = 0.98,
    force_mine: bool = False,
) -> Dict[str, Any]:
    """Execute Stage 3: LightGBM training with hard-negative mining and validation evaluation."""
    import gc
    import json
    from business_entity_resolution.io_utils import load_config, load_ground_truth, load_source
    from business_entity_resolution.model import (
        DEFAULT_FEATURE_COLUMNS,
        build_classifier,
        evaluate_validation,
        get_peak_memory_mb,
        load_model,
        mine_hard_negatives_and_prepare_datasets,
        save_model,
        stratified_entity_split,
        train,
    )
    from business_entity_resolution.scoring import load_matches_dict
    import numpy as np
    import pyarrow.parquet as pq

    t_stage_start = time.time()
    cfg_dir = root / "code" / "business_entity_resolution" / "configs"
    model_cfg = load_config(cfg_dir / "model.yaml")["lgbm"]
    paths_cfg = load_config(cfg_dir / "paths.yaml")

    feat_india = root / "output" / "features_india_full.parquet"
    feat_us = root / "output" / "features_us_full.parquet"
    if not feat_india.is_file() or not feat_us.is_file():
        raise FileNotFoundError(
            f"Required feature parquet files not found! Checked: {feat_india} and {feat_us}"
        )

    print("\n" + "=" * 60)
    print("STAGE 3: LIGHTGBM MODEL TRAINING & VALIDATION")
    print("=" * 60)
    print(f"Features India: {feat_india} ({feat_india.stat().st_size / (1024**3):.2f} GB)")
    print(f"Features US:    {feat_us} ({feat_us.stat().st_size / (1024**3):.2f} GB)")
    print(f"Validation Ratio: {val_ratio * 100:.1f}% (stratified by country + singleton)")
    print(f"Decision Threshold: {threshold}")
    print(f"Peak RAM at startup: {get_peak_memory_mb():.1f} MB")
    print("-" * 60)

    # ── Load ground truth & S1 metadata ─────────────────────────────
    gt_path = root / paths_cfg["data"]["train"]["ground_truth"]
    s1_path = root / paths_cfg["data"]["train"]["source1"]
    gt_dict = load_matches_dict(gt_path)
    s1 = load_source(s1_path, expected_source=1)
    s1_country_map = dict(zip(s1["entity_id"], s1["country"]))
    s1_ids = list(s1["entity_id"])

    if sample is not None and sample > 0:
        s1_ids = s1_ids[:sample]
        s1_country_map = {k: s1_country_map[k] for k in s1_ids}
        gt_dict = {k: gt_dict.get(k, set()) for k in s1_ids}
        print(f"[model] Sampled to first {len(s1_ids):,} Source 1 entities")

    # ── Stratified entity split ──────────────────────────────────────
    ckpt_dir = root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    val_entity_file = ckpt_dir / "val_entity_ids.json"

    if val_entity_file.is_file() and not force_mine and sample is None:
        val_entities = set(json.loads(val_entity_file.read_text(encoding="utf-8")))
        train_entities = set(s1_ids) - val_entities
        print(f"[model] Loaded {len(val_entities):,} held-out validation entities from {val_entity_file.name}")
    else:
        train_entities, val_entities = stratified_entity_split(
            s1_ids=s1_ids,
            s1_country_map=s1_country_map,
            gt_dict=gt_dict,
            val_ratio=val_ratio,
            random_state=42,
        )
        if sample is None:
            val_entity_file.write_text(json.dumps(sorted(val_entities)), encoding="utf-8")
        print(f"[model] Created stratified split: {len(train_entities):,} train entities, {len(val_entities):,} val entities")

    train_parquet = ckpt_dir / "train_mined_cap40.parquet" if (ckpt_dir / "train_mined_cap40.parquet").is_file() else (ckpt_dir / "train_mined.parquet")
    val_candidates_parquet = ckpt_dir / "val_candidates.parquet"
    val_mined_parquet = ckpt_dir / "val_mined_cap40.parquet" if (ckpt_dir / "val_mined_cap40.parquet").is_file() else (ckpt_dir / "val_mined.parquet")

    # ── Hard-negative mining / dataset preparation ──────────────────
    if (
        train_parquet.is_file()
        and val_candidates_parquet.is_file()
        and not force_mine
        and sample is None
        and max_pairs is None
    ):
        print(
            f"[model] Using existing mined datasets:\n"
            f"  Train: {train_parquet} ({train_parquet.stat().st_size / (1024**2):.1f} MB)\n"
            f"  Val:   {val_candidates_parquet} ({val_candidates_parquet.stat().st_size / (1024**2):.1f} MB)"
        )
    else:
        print("[model] Streaming 440M candidate pairs and mining hard negatives...")
        mine_stats = mine_hard_negatives_and_prepare_datasets(
            feature_paths=[feat_india, feat_us],
            gt_dict=gt_dict,
            train_entities=train_entities,
            val_entities=val_entities,
            output_train_parquet=train_parquet,
            output_val_candidates_parquet=val_candidates_parquet,
            output_val_mined_parquet=val_mined_parquet,
            feature_columns=DEFAULT_FEATURE_COLUMNS,
            max_pairs=max_pairs,
        )
        print(
            f"[model] Mining complete: {mine_stats['total_train_rows']:,} train rows "
            f"({mine_stats['total_train_pos']:,} pos, {mine_stats['total_train_neg']:,} neg, ratio={mine_stats['train_imbalance_ratio']:.2f}) | "
            f"Val candidates: {mine_stats['total_val_candidates']:,} | "
            f"Elapsed: {mine_stats['elapsed_seconds']:.1f}s | Peak RAM: {mine_stats['peak_memory_mb']:.1f} MB"
        )

    # ── Load training data into memory ──────────────────────────────
    print(f"\n[model] Loading training set into memory (Peak RAM before load: {get_peak_memory_mb():.1f} MB)...")
    train_tbl = pq.read_table(train_parquet, columns=["label"] + DEFAULT_FEATURE_COLUMNS)
    y_train = train_tbl["label"].to_numpy(zero_copy_only=False).astype(np.int8)
    X_train = np.column_stack([
        train_tbl[col].to_numpy(zero_copy_only=False) for col in DEFAULT_FEATURE_COLUMNS
    ]).astype(np.float32)
    del train_tbl
    gc.collect()

    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    scale_pos_weight = 1.0
    print(
        f"[model] Loaded X_train: {X_train.shape} ({X_train.nbytes / (1024**2):.1f} MB) | "
        f"Positives: {n_pos:,} | Negatives: {n_neg:,} | scale_pos_weight: {scale_pos_weight:.2f} (unweighted) | "
        f"Peak RAM: {get_peak_memory_mb():.1f} MB"
    )

    X_val_eval, y_val_eval = None, None
    if val_mined_parquet.is_file():
        val_mined_tbl = pq.read_table(val_mined_parquet, columns=["label"] + DEFAULT_FEATURE_COLUMNS)
        y_val_eval = val_mined_tbl["label"].to_numpy(zero_copy_only=False).astype(np.int8)
        X_val_eval = np.column_stack([
            val_mined_tbl[col].to_numpy(zero_copy_only=False) for col in DEFAULT_FEATURE_COLUMNS
        ]).astype(np.float32)
        del val_mined_tbl
        gc.collect()
        print(f"[model] Loaded validation monitoring set: {X_val_eval.shape} ({X_val_eval.nbytes / (1024**2):.1f} MB)")

    # ── Train LightGBM model ─────────────────────────────────────────
    print(f"\n[model] Training LightGBM classifier ({model_cfg.get('n_estimators', 500)} trees)...")
    clf = build_classifier(model_cfg, scale_pos_weight=scale_pos_weight)
    t_train_start = time.time()
    clf = train(
        clf,
        X_train,
        y_train,
        X_val=X_val_eval,
        y_val=y_val_eval,
        early_stopping_rounds=50,
        verbose_eval=50,
    )
    train_elapsed = time.time() - t_train_start
    best_iter = getattr(clf, "best_iteration_", model_cfg.get("n_estimators", 500))
    print(
        f"[model] Training finished in {train_elapsed:.2f}s ({train_elapsed / 60.0:.2f}m) | "
        f"Best iteration: {best_iter} | Peak RAM: {get_peak_memory_mb():.1f} MB"
    )

    # ── Persist model ────────────────────────────────────────────────
    ckpt_model_txt = ckpt_dir / "lgbm_model.txt"
    ckpt_model_pkl = ckpt_dir / "lgbm_model.pkl"
    save_model(clf, ckpt_model_txt)
    save_model(clf, ckpt_model_pkl)

    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    save_model(clf, models_dir / "lgbm_entity_model.txt")
    print(f"[model] Model saved to {ckpt_model_txt} and {models_dir / 'lgbm_entity_model.txt'}")

    # Free training matrices before validation evaluation
    del X_train, y_train, X_val_eval, y_val_eval
    gc.collect()

    # ── Validation evaluation ────────────────────────────────────────
    val_gt = {eid: gt_dict[eid] for eid in val_entities}
    print(
        f"\n[model] Scoring held-out validation set ({len(val_entities):,} entities) "
        f"at naive baseline threshold = {threshold}..."
    )
    eval_metrics = evaluate_validation(
        clf=clf,
        val_candidates_path=val_candidates_parquet,
        val_gt=val_gt,
        s1_country_map=s1_country_map,
        feature_columns=DEFAULT_FEATURE_COLUMNS,
        threshold=threshold,
    )

    stage_elapsed = time.time() - t_stage_start
    peak_ram_final = get_peak_memory_mb()

    # ── Experiments logging ──────────────────────────────────────────
    exp_id = run_id or f"run_{int(time.time())}"
    exp_dir = root / "experiments" / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = exp_dir / "metrics.json"

    report = {
        "run_id": exp_id,
        "stage": "model",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training": {
            "n_train_rows": int(n_pos + n_neg),
            "n_train_pos": n_pos,
            "n_train_neg": n_neg,
            "scale_pos_weight": float(scale_pos_weight),
            "best_iteration": int(best_iter) if best_iter is not None else None,
            "training_elapsed_seconds": train_elapsed,
            "total_stage_seconds": stage_elapsed,
            "peak_memory_mb": peak_ram_final,
        },
        "validation": eval_metrics,
    }
    metrics_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[model] Experiment metrics logged to {metrics_file}")

    # ── Print comprehensive evaluation table ─────────────────────────
    print("\n" + "=" * 60)
    print("LIGHTGBM MODEL EVALUATION REPORT (NAIVE THRESHOLD 0.5)")
    print("=" * 60)
    print(f"Overall Macro F_0.5:   {eval_metrics['macro_f_beta']:.4f}")
    print(f"Singleton Mean:        {eval_metrics['singleton_mean']:.4f} ({eval_metrics['n_singletons']:,} entities)")
    print(f"Has-Match Mean:        {eval_metrics['has_match_mean']:.4f} ({eval_metrics['n_has_match']:,} entities)")
    print(f"Total Val Entities:    {eval_metrics['n_entities']:,}")
    print(f"Val Pairs Scored:      {eval_metrics['val_pairs_scored']:,}")
    if "india" in eval_metrics:
        print("-" * 60)
        print(f"India Macro F_0.5:     {eval_metrics['india']['macro_f_beta']:.4f}")
        print(f"  India Singleton Mean: {eval_metrics['india']['singleton_mean']:.4f}")
        print(f"  India Has-Match Mean: {eval_metrics['india']['has_match_mean']:.4f}")
    if "us" in eval_metrics:
        print(f"US Macro F_0.5:        {eval_metrics['us']['macro_f_beta']:.4f}")
        print(f"  US Singleton Mean:    {eval_metrics['us']['singleton_mean']:.4f}")
        print(f"  US Has-Match Mean:    {eval_metrics['us']['has_match_mean']:.4f}")
    print("-" * 60)
    print(f"Training Time:         {train_elapsed:.2f} s ({train_elapsed / 60.0:.2f} min)")
    print(f"Total Stage Time:      {stage_elapsed:.2f} s ({stage_elapsed / 60.0:.2f} min)")
    print(f"Peak Working Set RAM:  {peak_ram_final:.1f} MB")
    print("=" * 60 + "\n")

    if eval_metrics["macro_f_beta"] < 0.5:
        logger.error(
            f"Validation macro F_0.5 ({eval_metrics['macro_f_beta']:.4f}) is surprisingly low (< 0.50)! "
            f"Stopping for investigation before proceeding."
        )

    return report


def run_threshold_tuning_stage(
    root: Path,
    sample: int | None = None,
    run_id: str | None = None,
    thresholds: list[float] | None = None,
) -> Dict[str, Any]:
    """Execute Stage 4: F_0.5-optimal threshold tuning & many-to-many grouping."""
    import gc
    import json
    from collections import defaultdict
    import numpy as np
    import pyarrow.parquet as pq
    from business_entity_resolution.grouping import (
        DEFAULT_THRESHOLDS,
        group_by_threshold,
        sweep_thresholds,
        sweep_thresholds_per_country,
    )
    from business_entity_resolution.io_utils import load_config, load_source
    from business_entity_resolution.model import (
        DEFAULT_FEATURE_COLUMNS,
        get_peak_memory_mb,
        load_model,
    )
    from business_entity_resolution.scoring import load_matches_dict

    t_stage_start = time.time()
    cfg_dir = root / "code" / "business_entity_resolution" / "configs"
    paths_cfg = load_config(cfg_dir / "paths.yaml")

    ckpt_dir = root / "checkpoints"
    # Locate model artifact (prefer cap40 model, then lgbm_model.txt, then models/)
    model_path = ckpt_dir / "lgbm_model_cap40.txt"
    if not model_path.is_file():
        model_path = ckpt_dir / "lgbm_model.txt"
    if not model_path.is_file():
        model_path = root / "models" / "lgbm_entity_model.txt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found in {ckpt_dir} or {root / 'models'}!")

    val_cand_path = ckpt_dir / "val_candidates.parquet"
    if not val_cand_path.is_file():
        val_cand_path = ckpt_dir / "val_candidates_cap40.parquet"
    if not val_cand_path.is_file():
        raise FileNotFoundError(f"Validation candidates parquet not found in {ckpt_dir}!")

    val_entity_file = ckpt_dir / "val_entity_ids.json"
    if not val_entity_file.is_file():
        raise FileNotFoundError(f"Validation entity file not found: {val_entity_file}!")

    gt_path = root / paths_cfg["data"]["train"]["ground_truth"]
    s1_path = root / paths_cfg["data"]["train"]["source1"]

    print("\n" + "=" * 60)
    print("STAGE 4: F_0.5-OPTIMAL THRESHOLD TUNING & GROUPING")
    print("=" * 60)
    print(f"Model Artifact:        {model_path} ({model_path.stat().st_size / (1024**2):.2f} MB)")
    print(f"Val Candidates:        {val_cand_path} ({val_cand_path.stat().st_size / (1024**2):.2f} MB)")
    print(f"Peak RAM at startup:   {get_peak_memory_mb():.1f} MB")
    print("-" * 60)

    # 1. Load model
    print(f"[threshold-tuning] Loading LightGBM model from {model_path.name}...")
    clf = load_model(model_path)

    # 2. Load ground truth and entity metadata
    gt_dict = load_matches_dict(gt_path)
    s1 = load_source(s1_path, expected_source=1)
    s1_country_map = dict(zip(s1["entity_id"], s1["country"]))

    val_entities = json.loads(val_entity_file.read_text(encoding="utf-8"))
    if sample is not None and sample > 0:
        val_entities = val_entities[:sample]
        print(f"[threshold-tuning] Sampled to first {len(val_entities):,} validation entities")

    val_gt = {eid: gt_dict.get(eid, set()) for eid in val_entities}
    entity_countries = {eid: s1_country_map.get(eid, "_unknown") for eid in val_entities}
    val_set = set(val_entities)

    n_sing = sum(1 for true_ids in val_gt.values() if len(true_ids) == 0)
    n_match = len(val_gt) - n_sing
    print(
        f"[threshold-tuning] Universe: {len(val_entities):,} validation entities "
        f"({n_sing:,} singletons, {n_match:,} matched)"
    )

    # Threshold grid to evaluate
    sweep_grid = thresholds or [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]
    min_thresh = min(sweep_grid)

    # 3. Stream validation candidates, predicting and pre-filtering to score >= min_thresh
    print(
        f"\n[threshold-tuning] Streaming 44M candidate pairs with memory-safe pre-filtering (min_thresh={min_thresh})..."
    )
    t_pred_start = time.time()
    pf = pq.ParquetFile(val_cand_path)
    cols_to_read = ["source1_entity_id", "candidate_entity_id"] + DEFAULT_FEATURE_COLUMNS

    scored_pairs: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    # Ensure every validation entity has an entry in scored_pairs (even if empty)
    for eid in val_entities:
        scored_pairs[eid] = []

    pairs_scanned = 0
    pairs_retained = 0
    batch_size = 500000

    for batch in pf.iter_batches(batch_size=batch_size, columns=cols_to_read):
        s1_ids = batch["source1_entity_id"].to_pylist()
        cand_ids = batch["candidate_entity_id"].to_pylist()

        X_batch = np.column_stack([
            batch[col].to_numpy(zero_copy_only=False) for col in DEFAULT_FEATURE_COLUMNS
        ]).astype(np.float32)

        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_batch)[:, 1]
        elif hasattr(clf, "predict"):
            probs = clf.predict(X_batch)
        else:
            raise TypeError(f"Unknown classifier type: {type(clf)}")

        keep_idx = np.where(probs >= min_thresh)[0]
        for idx in keep_idx:
            eid = s1_ids[idx]
            if eid in val_set:
                scored_pairs[eid].append((cand_ids[idx], float(probs[idx])))
                pairs_retained += 1

        pairs_scanned += len(s1_ids)

    pred_elapsed = time.time() - t_pred_start
    peak_ram_pred = get_peak_memory_mb()
    print(
        f"[threshold-tuning] Scored {pairs_scanned:,} pairs in {pred_elapsed:.2f}s "
        f"({pairs_scanned / pred_elapsed:,.0f} pairs/s) | "
        f"Retained pairs (P >= {min_thresh}): {pairs_retained:,} | "
        f"Peak RAM: {peak_ram_pred:.1f} MB"
    )

    # 4. Global threshold sweep using grouping.sweep_thresholds
    print(f"\n[threshold-tuning] Running global threshold sweep over {sweep_grid}...")
    t_sweep_start = time.time()
    sweep_res = sweep_thresholds(
        scored_pairs=scored_pairs,
        ground_truth=val_gt,
        thresholds=sweep_grid,
        beta=0.5,
    )
    sweep_elapsed = time.time() - t_sweep_start

    best_thresh = sweep_res["best_threshold"]
    best_metrics = sweep_res["best_metrics"]

    # 5. Per-country threshold sweep using grouping.sweep_thresholds_per_country
    print(f"[threshold-tuning] Running per-country threshold sweep...")
    country_res = sweep_thresholds_per_country(
        scored_pairs=scored_pairs,
        ground_truth=val_gt,
        entity_countries=entity_countries,
        thresholds=sweep_grid,
        beta=0.5,
    )

    # 6. Evaluate predictions at chosen threshold and calculate many-to-many sanity stats
    best_preds = group_by_threshold(scored_pairs, threshold=best_thresh)

    # Many-to-many sanity check
    s1_multi_s2 = 0
    s1_multi_s3 = 0
    s1_multi_either = 0
    s1_multi_both = 0
    total_predicted_matches = 0
    entities_with_any_pred = 0

    for eid, cands in best_preds.items():
        if not cands:
            continue
        entities_with_any_pred += 1
        total_predicted_matches += len(cands)
        n_s2 = sum(1 for cid in cands if cid.startswith("S2-"))
        n_s3 = sum(1 for cid in cands if cid.startswith("S3-"))
        if n_s2 > 1:
            s1_multi_s2 += 1
        if n_s3 > 1:
            s1_multi_s3 += 1
        if n_s2 > 1 or n_s3 > 1:
            s1_multi_either += 1
        if n_s2 > 1 and n_s3 > 1:
            s1_multi_both += 1

    stage_elapsed = time.time() - t_stage_start
    peak_ram_final = get_peak_memory_mb()

    # 7. Experiment logging
    exp_run_id = run_id or f"threshold_tuning_{int(time.time())}"
    exp_dir = root / "experiments" / exp_run_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = exp_dir / "metrics.json"

    # Separate cohort metrics at best threshold
    india_metrics = country_res.get("India", {}).get("sweep", {}).get(best_thresh, {})
    us_metrics = country_res.get("US", {}).get("sweep", {}).get(best_thresh, {})

    report = {
        "run_id": exp_run_id,
        "stage": "threshold_tuning",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_artifact": str(model_path.name),
        "chosen_threshold": best_thresh,
        "overall_macro_f_beta": best_metrics["macro_f_beta"],
        "singleton_mean": best_metrics["singleton_mean"],
        "has_match_mean": best_metrics["has_match_mean"],
        "n_entities": best_metrics["n_entities"],
        "n_singletons": best_metrics["n_singletons"],
        "n_has_match": best_metrics["n_has_match"],
        "india_metrics_at_chosen_threshold": india_metrics,
        "us_metrics_at_chosen_threshold": us_metrics,
        "country_sweep_independent_optima": {
            c: {
                "best_threshold": r["best_threshold"],
                "macro_f_beta": r["best_metrics"]["macro_f_beta"],
            }
            for c, r in country_res.items()
        },
        "many_to_many_stats": {
            "entities_with_any_prediction": entities_with_any_pred,
            "total_predicted_matches": total_predicted_matches,
            "avg_matches_per_matched_entity": (total_predicted_matches / entities_with_any_pred) if entities_with_any_pred > 0 else 0.0,
            "entities_with_gt1_s2_match": s1_multi_s2,
            "entities_with_gt1_s3_match": s1_multi_s3,
            "entities_with_gt1_match_either_source": s1_multi_either,
            "entities_with_gt1_match_both_sources": s1_multi_both,
        },
        "full_sweep_table": {
            f"{th:.2f}": m for th, m in sweep_res["sweep"].items()
        },
        "timing": {
            "prediction_elapsed_seconds": pred_elapsed,
            "sweep_elapsed_seconds": sweep_elapsed,
            "total_stage_seconds": stage_elapsed,
            "peak_memory_mb": peak_ram_final,
        },
    }
    metrics_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[threshold-tuning] Metrics successfully logged to {metrics_file}")

    # 8. Print comprehensive evaluation tables
    print("\n" + "=" * 90)
    print(f"THRESHOLD TUNING SWEEP RESULTS (MODEL: {model_path.name})")
    print("=" * 90)
    print(f"{'Thresh':<8} {'Overall F0.5':<14} {'Singleton':<12} {'Has-Match':<12} {'India F0.5':<12} {'US F0.5':<10} {'Status'}")
    print("-" * 90)
    for th in sweep_grid:
        ov = sweep_res["sweep"].get(th, {})
        ind = country_res.get("India", {}).get("sweep", {}).get(th, {})
        us = country_res.get("US", {}).get("sweep", {}).get(th, {})
        is_best = "* CHOSEN" if th == best_thresh else ""
        print(
            f"{th:<8.2f} {ov.get('macro_f_beta', 0.0):<14.4f} {ov.get('singleton_mean', 0.0):<12.4f} "
            f"{ov.get('has_match_mean', 0.0):<12.4f} {ind.get('macro_f_beta', 0.0):<12.4f} {us.get('macro_f_beta', 0.0):<10.4f} {is_best}"
        )
    print("=" * 90)

    print("\n" + "=" * 60)
    print("CHOSEN THRESHOLD PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Operating Threshold:      {best_thresh:.2f}")
    print(f"Overall Macro F_0.5:      {best_metrics['macro_f_beta']:.4f}")
    print(f"  Singleton Subset F_0.5: {best_metrics['singleton_mean']:.4f} ({best_metrics['n_singletons']:,} entities)")
    print(f"  Has-Match Subset F_0.5: {best_metrics['has_match_mean']:.4f} ({best_metrics['n_has_match']:,} entities)")
    print("-" * 60)
    print(f"India Macro F_0.5:        {india_metrics.get('macro_f_beta', 0.0):.4f} ({india_metrics.get('n_entities', 0):,} entities)")
    print(f"  India Singleton Mean:   {india_metrics.get('singleton_mean', 0.0):.4f}")
    print(f"  India Has-Match Mean:   {india_metrics.get('has_match_mean', 0.0):.4f}")
    print(f"US Macro F_0.5:           {us_metrics.get('macro_f_beta', 0.0):.4f} ({us_metrics.get('n_entities', 0):,} entities)")
    print(f"  US Singleton Mean:      {us_metrics.get('singleton_mean', 0.0):.4f}")
    print(f"  US Has-Match Mean:      {us_metrics.get('has_match_mean', 0.0):.4f}")
    print("-" * 60)
    print("MANY-TO-MANY SANITY CHECK COUNTS:")
    print(f"  Entities with >1 S2 match:          {s1_multi_s2:,}")
    print(f"  Entities with >1 S3 match:          {s1_multi_s3:,}")
    print(f"  Entities with >1 match (either):    {s1_multi_either:,}")
    print(f"  Entities with >1 match (both S2&S3):{s1_multi_both:,}")
    print(f"  Total Predicted Match Pairs:        {total_predicted_matches:,}")
    if entities_with_any_pred > 0:
        print(f"  Avg Matches per Matched Entity:     {(total_predicted_matches / entities_with_any_pred):.2f}")
    print("-" * 60)
    print(f"Prediction & Filter Time: {pred_elapsed:.2f} s")
    print(f"Threshold Sweep Time:     {sweep_elapsed:.2f} s")
    print(f"Total Stage Time:         {stage_elapsed:.2f} s ({stage_elapsed / 60.0:.2f} min)")
    print(f"Peak Working Set RAM:     {peak_ram_final:.1f} MB")
    print("=" * 60 + "\n")

    return report


def run_full_pipeline(root: Path, sample: int | None = None) -> Dict[str, Any]:
    """Execute the full orchestrated training pipeline.

    Loads all configs and calls ``pipeline.run_train_pipeline``.
    """
    from business_entity_resolution.io_utils import load_config
    from business_entity_resolution.pipeline import run_train_pipeline

    cfg_dir = root / "code" / "business_entity_resolution" / "configs"
    paths_cfg = load_config(cfg_dir / "paths.yaml")
    blocking_cfg = load_config(cfg_dir / "blocking.yaml")
    model_cfg = load_config(cfg_dir / "model.yaml")

    config = {
        "root": str(root),
        "paths": paths_cfg,
        "blocking": blocking_cfg,
        "model": model_cfg,
        "sample": sample,
    }

    result = run_train_pipeline(config)

    logger.info("Pipeline summary:")
    logger.info("  Completed stages: %s", result["completed"])
    logger.info("  Skipped stages:   %s", result["skipped"])
    logger.info("  Total elapsed:    %.2fs", result["elapsed_total_seconds"])

    return result


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

    if args.stage in ("train", "model", "all"):
        run_model_stage(
            root=args.root,
            sample=args.sample,
            max_pairs=args.max_pairs,
            run_id=args.run_id,
            val_ratio=args.val_ratio,
            threshold=args.threshold,
            force_mine=args.force_mine,
        )
        if args.stage in ("train", "model"):
            return

    if args.stage in ("threshold-tuning", "eval", "all"):
        run_threshold_tuning_stage(
            root=args.root,
            sample=args.sample,
            run_id=args.run_id,
        )
        if args.stage in ("threshold-tuning", "eval"):
            return


if __name__ == "__main__":
    main()

