#!/usr/bin/env python
"""
remediate_branch_b.py  –  Milestone 7 Remediation (Branch B):
Re-run hard-negative mining with expanded negative caps:
  - max_negatives_has_match: 15 -> 40
  - singleton_negatives: 10 -> 35
  - neg_pos_ratio: 5 -> 10 (expanded ratio per match)
  - min_negatives_has_match: 5 -> 10

Then retrain LightGBM with scale_pos_weight=1.0 and evaluate on held-out validation set.
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq

# Add src to sys.path
root = Path(__file__).resolve().parents[1]
src_dir = root / "code" / "business_entity_resolution" / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from business_entity_resolution.io_utils import load_config, load_source
from business_entity_resolution.model import (
    DEFAULT_FEATURE_COLUMNS,
    build_classifier,
    evaluate_validation,
    get_peak_memory_mb,
    mine_hard_negatives_and_prepare_datasets,
    save_model,
    train,
)
from business_entity_resolution.scoring import load_matches_dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("remediate_branch_b")


def main() -> None:
    t_stage_start = time.time()
    cfg_dir = root / "code" / "business_entity_resolution" / "configs"
    model_cfg = load_config(cfg_dir / "model.yaml")["lgbm"]
    paths_cfg = load_config(cfg_dir / "paths.yaml")

    ckpt_dir = root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Feature inputs (full 440M candidate pairs)
    feat_india = root / "output" / "features_india_full.parquet"
    feat_us = root / "output" / "features_us_full.parquet"
    if not feat_india.is_file() or not feat_us.is_file():
        raise FileNotFoundError(
            f"Required feature parquet files not found! Checked: {feat_india} and {feat_us}"
        )

    # Output paths for expanded mining (do NOT overwrite original spw=4.01 checkpoints)
    train_parquet = ckpt_dir / "train_mined_cap40.parquet"
    val_candidates_parquet = ckpt_dir / "val_candidates_cap40.parquet"
    val_mined_parquet = ckpt_dir / "val_mined_cap40.parquet"
    val_entity_file = ckpt_dir / "val_entity_ids.json"

    print("\n" + "=" * 60)
    print("BRANCH B: EXPANDED NEGATIVE MINING & RETRAINING (spw=1.0)")
    print("=" * 60)
    print(f"Features India:        {feat_india} ({feat_india.stat().st_size / (1024**3):.2f} GB)")
    print(f"Features US:           {feat_us} ({feat_us.stat().st_size / (1024**3):.2f} GB)")
    print(f"Negative Mining Caps:  Matched=40 (was 15) | Singletons=35 (was 10)")
    print(f"Sampling Ratio:        neg_pos_ratio=10 | min_neg=10")
    print(f"scale_pos_weight:      1.0 (unweighted)")
    print(f"Decision Threshold:    0.5")
    print(f"Output Train:          {train_parquet.name}")
    print(f"Peak RAM at startup:   {get_peak_memory_mb():.1f} MB")
    print("-" * 60)

    # ── Load ground truth & validation entity set ────────────────────
    gt_path = root / paths_cfg["data"]["train"]["ground_truth"]
    s1_path = root / paths_cfg["data"]["train"]["source1"]
    gt_dict = load_matches_dict(gt_path)
    s1 = load_source(s1_path, expected_source=1)
    s1_country_map = dict(zip(s1["entity_id"], s1["country"]))
    s1_ids = list(s1["entity_id"])

    if not val_entity_file.is_file():
        raise FileNotFoundError(f"Validation entity file not found: {val_entity_file}")
    val_entities = set(json.loads(val_entity_file.read_text(encoding="utf-8")))
    train_entities = set(s1_ids) - val_entities
    val_gt = {eid: gt_dict[eid] for eid in val_entities}
    print(
        f"[remediate_b] Loaded {len(val_entities):,} held-out validation entities and "
        f"{len(train_entities):,} training entities"
    )

    # ── STEP 1: Re-run mining stage over 440M candidate pairs ────────
    print("\n[remediate_b] STEP 1: Re-scanning 440M candidate pairs with expanded negative mining...")
    t_mine_start = time.time()
    mine_stats = mine_hard_negatives_and_prepare_datasets(
        feature_paths=[feat_india, feat_us],
        gt_dict=gt_dict,
        train_entities=train_entities,
        val_entities=val_entities,
        output_train_parquet=train_parquet,
        output_val_candidates_parquet=val_candidates_parquet,
        output_val_mined_parquet=val_mined_parquet,
        feature_columns=DEFAULT_FEATURE_COLUMNS,
        neg_pos_ratio=10,
        max_negatives_has_match=40,
        min_negatives_has_match=10,
        singleton_negatives=35,
        batch_row_groups=5,
    )
    mine_elapsed = time.time() - t_mine_start
    print(
        f"[remediate_b] Mining complete in {mine_elapsed:.1f}s ({mine_elapsed / 60.0:.2f}m) | "
        f"Train rows: {mine_stats['total_train_rows']:,} (pos={mine_stats['total_train_pos']:,}, "
        f"neg={mine_stats['total_train_neg']:,}, neg:pos ratio={mine_stats['train_imbalance_ratio']:.2f}:1) | "
        f"Val candidates: {mine_stats['total_val_candidates']:,} | "
        f"Peak RAM: {get_peak_memory_mb():.1f} MB"
    )

    # ── STEP 2: Load newly mined training data into memory ───────────
    print(f"\n[remediate_b] STEP 2: Loading newly mined training set into memory...")
    train_tbl = pq.read_table(train_parquet, columns=["label"] + DEFAULT_FEATURE_COLUMNS)
    y_train = train_tbl["label"].to_numpy(zero_copy_only=False).astype(np.int8)
    X_train = np.column_stack([
        train_tbl[col].to_numpy(zero_copy_only=False) for col in DEFAULT_FEATURE_COLUMNS
    ]).astype(np.float32)
    del train_tbl
    gc.collect()

    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    print(
        f"[remediate_b] Loaded X_train: {X_train.shape} ({X_train.nbytes / (1024**2):.1f} MB) | "
        f"Positives: {n_pos:,} | Negatives: {n_neg:,} | scale_pos_weight: 1.00 | "
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
        print(f"[remediate_b] Loaded validation monitoring set: {X_val_eval.shape} ({X_val_eval.nbytes / (1024**2):.1f} MB)")

    # ── STEP 3: Train LightGBM classifier with scale_pos_weight=1.0 ──
    print(f"\n[remediate_b] STEP 3: Training LightGBM classifier with scale_pos_weight=1.0 ({model_cfg.get('n_estimators', 500)} trees)...")
    clf = build_classifier(model_cfg, scale_pos_weight=1.0)
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
        f"[remediate_b] Training finished in {train_elapsed:.2f}s ({train_elapsed / 60.0:.2f}m) | "
        f"Best iteration: {best_iter} | Peak RAM: {get_peak_memory_mb():.1f} MB"
    )

    # ── Persist model ────────────────────────────────────────────────
    ckpt_model_txt = ckpt_dir / "lgbm_model_cap40.txt"
    ckpt_model_pkl = ckpt_dir / "lgbm_model_cap40.pkl"
    save_model(clf, ckpt_model_txt)
    save_model(clf, ckpt_model_pkl)

    # Also update main model artifacts
    save_model(clf, ckpt_dir / "lgbm_model.txt")
    save_model(clf, ckpt_dir / "lgbm_model.pkl")
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    save_model(clf, models_dir / "lgbm_entity_model.txt")
    save_model(clf, models_dir / "lgbm_entity_model.pkl")
    print(f"[remediate_b] Model saved to {ckpt_model_txt} and {models_dir / 'lgbm_entity_model.txt'}")

    del X_train, y_train, X_val_eval, y_val_eval
    gc.collect()

    # ── STEP 4: Validation evaluation at threshold 0.5 ───────────────
    val_eval_path = val_candidates_parquet if val_candidates_parquet.is_file() else (ckpt_dir / "val_candidates.parquet")
    print(
        f"\n[remediate_b] STEP 4: Scoring held-out validation set ({len(val_entities):,} entities) "
        f"from {val_eval_path.name} at naive baseline threshold = 0.5..."
    )
    t_eval_start = time.time()
    eval_metrics = evaluate_validation(
        clf=clf,
        val_candidates_path=val_eval_path,
        val_gt=val_gt,
        s1_country_map=s1_country_map,
        feature_columns=DEFAULT_FEATURE_COLUMNS,
        threshold=0.5,
    )
    eval_elapsed = time.time() - t_eval_start

    stage_elapsed = time.time() - t_stage_start
    peak_ram_final = get_peak_memory_mb()

    # ── Experiment logging ───────────────────────────────────────────
    exp_id = f"remediate_b_cap40_{int(time.time())}"
    exp_dir = root / "experiments" / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = exp_dir / "metrics.json"

    report = {
        "run_id": exp_id,
        "stage": "model_remediation_branch_b_cap40",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mining": {
            "max_negatives_has_match": 40,
            "singleton_negatives": 35,
            "neg_pos_ratio": 10,
            "min_negatives_has_match": 10,
            "mining_elapsed_seconds": mine_elapsed,
            **mine_stats,
        },
        "training": {
            "n_train_rows": int(n_pos + n_neg),
            "n_train_pos": n_pos,
            "n_train_neg": n_neg,
            "scale_pos_weight": 1.0,
            "best_iteration": int(best_iter) if best_iter is not None else None,
            "training_elapsed_seconds": train_elapsed,
            "eval_elapsed_seconds": eval_elapsed,
            "total_stage_seconds": stage_elapsed,
            "peak_memory_mb": peak_ram_final,
        },
        "validation": eval_metrics,
    }
    metrics_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[remediate_b] Experiment metrics logged to {metrics_file}")

    # ── Print comprehensive evaluation table ─────────────────────────
    print("\n" + "=" * 60)
    print("LIGHTGBM MODEL EVALUATION REPORT (EXPANDED MINING + SPW=1.0, THRESHOLD 0.5)")
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
    print(f"Mining Time:           {mine_elapsed:.2f} s ({mine_elapsed / 60.0:.2f} min)")
    print(f"Training Time:         {train_elapsed:.2f} s ({train_elapsed / 60.0:.2f} min)")
    print(f"Validation Eval Time:  {eval_elapsed:.2f} s ({eval_elapsed / 60.0:.2f} min)")
    print(f"Total Stage Time:      {stage_elapsed:.2f} s ({stage_elapsed / 60.0:.2f} min)")
    print(f"Peak Working Set RAM:  {peak_ram_final:.1f} MB")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
