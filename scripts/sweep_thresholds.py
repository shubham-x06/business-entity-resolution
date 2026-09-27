#!/usr/bin/env python
"""
sweep_thresholds.py  –  Milestone 7: Threshold calibration on winning model (cap40, spw=1.0).

Streams checkpoints/val_candidates.parquet once, computes probabilities using
checkpoints/lgbm_model_cap40.txt, and evaluates macro F_0.5 across thresholds:
[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99].
"""

from __future__ import annotations

from collections import defaultdict
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Sequence, Set, Tuple

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

# Add src to sys.path
root = Path(__file__).resolve().parents[1]
src_dir = root / "code" / "business_entity_resolution" / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from business_entity_resolution.io_utils import load_config, load_source
from business_entity_resolution.model import DEFAULT_FEATURE_COLUMNS, get_peak_memory_mb, load_model
from business_entity_resolution.scoring import compute_f_beta, load_matches_dict, score_entity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sweep_thresholds")

THRESHOLDS: List[float] = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]


def evaluate_cohort(
    predictions: Dict[str, Set[str]],
    val_gt: Dict[str, Set[str]],
    entity_subset: Sequence[str],
) -> Tuple[float, float, float]:
    """Compute (macro_f_beta, singleton_mean, has_match_mean) for a given entity subset."""
    all_scores: List[float] = []
    singleton_scores: List[float] = []
    has_match_scores: List[float] = []

    for eid in entity_subset:
        true_ids = val_gt.get(eid, set())
        pred_ids = predictions.get(eid, set())
        s = score_entity(pred_ids, true_ids, beta=0.5)
        all_scores.append(s)
        if len(true_ids) == 0:
            singleton_scores.append(s)
        else:
            has_match_scores.append(s)

    macro_f = float(np.mean(all_scores)) if all_scores else 0.0
    singleton_m = float(np.mean(singleton_scores)) if singleton_scores else 0.0
    has_match_m = float(np.mean(has_match_scores)) if has_match_scores else 0.0
    return macro_f, singleton_m, has_match_m


def main() -> None:
    t_start = time.time()
    cfg_dir = root / "code" / "business_entity_resolution" / "configs"
    paths_cfg = load_config(cfg_dir / "paths.yaml")

    ckpt_dir = root / "checkpoints"
    model_path = ckpt_dir / "lgbm_model_cap40.txt"
    if not model_path.is_file():
        model_path = ckpt_dir / "lgbm_model_cap40.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found in {ckpt_dir}!")

    val_cand_path = ckpt_dir / "val_candidates.parquet"
    if not val_cand_path.is_file():
        val_cand_path = ckpt_dir / "val_candidates_cap40.parquet"
    if not val_cand_path.is_file():
        raise FileNotFoundError(f"Validation candidates parquet not found in {ckpt_dir}!")

    val_entity_file = ckpt_dir / "val_entity_ids.json"
    gt_path = root / paths_cfg["data"]["train"]["ground_truth"]
    s1_path = root / paths_cfg["data"]["train"]["source1"]

    print("\n" + "=" * 60)
    print("STAGE 3 REMEDIATION: THRESHOLD CALIBRATION SWEEP")
    print("=" * 60)
    print(f"Model Artifact:        {model_path} ({model_path.stat().st_size / (1024**2):.2f} MB)")
    print(f"Val Candidates:        {val_cand_path} ({val_cand_path.stat().st_size / (1024**2):.2f} MB)")
    print(f"Sweep Thresholds:      {THRESHOLDS}")
    print(f"Peak RAM at startup:   {get_peak_memory_mb():.1f} MB")
    print("-" * 60)

    # ── Load model ───────────────────────────────────────────────────
    print(f"[sweep] Loading LightGBM model from {model_path.name}...")
    clf = load_model(model_path)

    # ── Load Ground Truth and Entity Metadata ────────────────────────
    gt_dict = load_matches_dict(gt_path)
    s1 = load_source(s1_path, expected_source=1)
    s1_country_map = dict(zip(s1["entity_id"], s1["country"]))

    val_entities = json.loads(val_entity_file.read_text(encoding="utf-8"))
    val_gt = {eid: gt_dict[eid] for eid in val_entities}

    val_india = [eid for eid in val_entities if s1_country_map.get(eid) == "India"]
    val_us = [eid for eid in val_entities if s1_country_map.get(eid) == "US"]
    print(
        f"[sweep] Total validation entities: {len(val_entities):,} "
        f"(India: {len(val_india):,}, US: {len(val_us):,})"
    )

    # ── Predict all pairs once, storing pairs with P >= min(THRESHOLDS) ─
    min_thresh = min(THRESHOLDS)
    print(f"\n[sweep] Streaming candidate pairs and predicting probabilities (min_thresh={min_thresh})...")
    t_pred_start = time.time()

    pf = pq.ParquetFile(val_cand_path)
    total_val_pairs = pf.metadata.num_rows
    cols_to_read = ["source1_entity_id", "candidate_entity_id"] + DEFAULT_FEATURE_COLUMNS

    # Map: eid -> list of (cand_id, prob) for prob >= min_thresh
    candidate_predictions: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    total_retained_pairs = 0
    pairs_scored = 0

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
            candidate_predictions[s1_ids[idx]].append((cand_ids[idx], float(probs[idx])))

        pairs_scored += len(s1_ids)
        total_retained_pairs += len(keep_idx)

    pred_elapsed = time.time() - t_pred_start
    print(
        f"[sweep] Prediction pass complete in {pred_elapsed:.2f}s "
        f"({pairs_scored / pred_elapsed:,.0f} pairs/s) | "
        f"Pairs with prob >= {min_thresh}: {total_retained_pairs:,} | "
        f"Peak RAM: {get_peak_memory_mb():.1f} MB"
    )

    # ── Sweep each threshold ─────────────────────────────────────────
    print("\n[sweep] Evaluating validation metrics across thresholds...")
    results: List[Dict[str, Any]] = []

    best_thresh = None
    best_macro_f = -1.0

    for th in THRESHOLDS:
        # Build predictions dict for this threshold
        pred_dict: Dict[str, Set[str]] = {}
        for eid, cand_probs in candidate_predictions.items():
            matched = {cid for cid, p in cand_probs if p >= th}
            if matched:
                pred_dict[eid] = matched

        overall_f, overall_sing, overall_match = evaluate_cohort(pred_dict, val_gt, val_entities)
        india_f, india_sing, india_match = evaluate_cohort(pred_dict, val_gt, val_india)
        us_f, us_sing, us_match = evaluate_cohort(pred_dict, val_gt, val_us)

        res = {
            "threshold": th,
            "overall": {
                "macro_f_beta": overall_f,
                "singleton_mean": overall_sing,
                "has_match_mean": overall_match,
            },
            "india": {
                "macro_f_beta": india_f,
                "singleton_mean": india_sing,
                "has_match_mean": india_match,
            },
            "us": {
                "macro_f_beta": us_f,
                "singleton_mean": us_sing,
                "has_match_mean": us_match,
            },
        }
        results.append(res)

        if overall_f > best_macro_f:
            best_macro_f = overall_f
            best_thresh = th

        print(
            f"  Threshold {th:4.2f} -> Overall F_0.5: {overall_f:.4f} "
            f"(Sing: {overall_sing:.4f}, Match: {overall_match:.4f}) | "
            f"India: {india_f:.4f} | US: {us_f:.4f}"
        )

    total_elapsed = time.time() - t_start

    # ── Experiment logging ───────────────────────────────────────────
    exp_id = f"threshold_sweep_{int(time.time())}"
    exp_dir = root / "experiments" / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    sweep_file = exp_dir / "metrics.json"

    sweep_report = {
        "run_id": exp_id,
        "stage": "threshold_calibration_sweep",
        "model_artifact": str(model_path.name),
        "best_threshold": best_thresh,
        "best_macro_f_beta": best_macro_f,
        "sweep_results": results,
        "timing": {
            "prediction_elapsed_seconds": pred_elapsed,
            "total_elapsed_seconds": total_elapsed,
        },
    }
    sweep_file.write_text(json.dumps(sweep_report, indent=2), encoding="utf-8")
    print(f"\n[sweep] Full threshold sweep metrics logged to {sweep_file}")

    # ── Print comprehensive evaluation table ─────────────────────────
    print("\n" + "=" * 90)
    print("THRESHOLD CALIBRATION SWEEP RESULTS (MODEL: cap40, spw=1.0)")
    print("=" * 90)
    print(f"{'Thresh':<8} {'Overall F0.5':<14} {'Singleton':<12} {'Has-Match':<12} {'India F0.5':<12} {'US F0.5':<10} {'Status'}")
    print("-" * 90)
    for r in results:
        th = r["threshold"]
        ov = r["overall"]
        ind = r["india"]
        us = r["us"]
        is_best = "* BEST" if th == best_thresh else ""
        print(
            f"{th:<8.2f} {ov['macro_f_beta']:<14.4f} {ov['singleton_mean']:<12.4f} "
            f"{ov['has_match_mean']:<12.4f} {ind['macro_f_beta']:<12.4f} {us['macro_f_beta']:<10.4f} {is_best}"
        )
    print("=" * 90)
    print(f"\nRECOMMENDED OPTIMAL THRESHOLD: {best_thresh:.2f} (Macro F_0.5 = {best_macro_f:.4f})")
    print(f"Total Sweep Execution Time:    {total_elapsed:.2f}s ({total_elapsed / 60.0:.2f}m)")
    print(f"Peak Working Set RAM:          {get_peak_memory_mb():.1f} MB\n")


if __name__ == "__main__":
    main()
