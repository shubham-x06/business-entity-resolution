#!/usr/bin/env python
"""
run_train.py  –  CLI entry-point for the training pipeline.

Usage
-----
    python scripts/run_train.py [--root ROOT] [--stage STAGE]

Loads configs, runs normalisation → blocking → features → model training,
and saves the trained model artifact.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ── Ensure src/ is importable ───────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_CODE_DIR = _SCRIPT_DIR.parent  # code/business_entity_resolution/
_SRC_DIR = _CODE_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


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
        description="Train the entity-resolution model.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Repository root (student_resource/).  Default: auto-detected.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=["blocking", "features", "model", "all"],
        help="Run a single pipeline stage.  Default: all.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Experiment run ID (for output directory naming).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample N rows from Source 1 for fast pipeline validation.",
    )
    return parser.parse_args(argv)


def run_blocking_stage(
    root: Path,
    run_id: str | None = None,
    sample: int | None = None,
) -> dict:
    """Run the blocking stage: generate candidate pairs and evaluate recall.

    Parameters
    ----------
    root : Path
        Repository root.
    run_id : str | None
        Experiment run ID.
    sample : int | None
        Optional sample limit on Source 1.
    """
    from business_entity_resolution.io_utils import (
        load_config,
        load_ground_truth,
        load_source,
        write_candidate_pairs,
    )
    from business_entity_resolution.blocking import generate_candidate_pairs

    # ── Load configs ────────────────────────────────────────────────
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
    candidates = generate_candidate_pairs(s1, s2, s3, blocking_cfg)
    elapsed = time.time() - t0

    # ── Write candidate_pairs.tsv ───────────────────────────────────
    out_dir = root / "output"
    write_candidate_pairs(candidates, out_dir / "candidate_pairs.tsv")

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
                
        for k in sorted(found_true_pairs_k.keys()):
            r_k = found_true_pairs_k[k] / total_true_pairs if total_true_pairs > 0 else 0.0
            k_lens = [min(len(cl), k) for cl in candidates.values()]
            mean_k = sum(k_lens) / len(k_lens) if k_lens else 0.0
            sorted_k_lens = sorted(k_lens)
            median_k = sorted_k_lens[len(sorted_k_lens) // 2] if sorted_k_lens else 0.0
            print(f"[blocking] Recall@{k}: {r_k:.4f} ({found_true_pairs_k[k]:,}/{total_true_pairs:,}) | Mean cands: {mean_k:.1f} | Median cands: {median_k:.1f}")
            
        recall = found_true_pairs_k[max_k] / total_true_pairs if total_true_pairs > 0 else 0.0
    else:
        print(f"[blocking] Ground truth not found at {gt_path} — skipping recall evaluation")

    # ── Save metrics ────────────────────────────────────────────────
    rid = run_id or time.strftime("%Y%m%d_%H%M%S")
    exp_dir = root / "experiments" / rid
    exp_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "stage": "blocking",
        "run_id": rid,
        "s1_entities": len(candidates),
        "total_candidate_pairs": n_pairs,
        "mean_candidates_per_s1": round(mean_cands, 2),
        "median_candidates_per_s1": round(median_cands, 2),
        "s1_with_candidates": n_with,
        "reduction_ratio": round(reduction_ratio, 6),
        "wall_clock_seconds": round(elapsed, 1),
    }
    if recall is not None:
        metrics["recall"] = round(recall, 4)

    metrics_path = exp_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[blocking] Metrics saved to {metrics_path}")
    print(f"[blocking] Mean candidates/S1: {mean_cands:.1f}")
    print(f"[blocking] Median candidates/S1: {median_cands:.1f}")
    print(f"[blocking] Reduction ratio: {reduction_ratio:.6f}")
    print(f"[blocking] Wall-clock: {elapsed:.0f}s")

    return metrics


def main(argv: list[str] | None = None) -> None:
    """Entry-point for the training pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    args = parse_args(argv)
    print(f"[run_train] root = {args.root}")

    if args.stage == "blocking":
        run_blocking_stage(args.root, run_id=args.run_id, sample=args.sample)
    elif args.stage is None or args.stage == "all":
        print("[run_train] Full training pipeline not yet implemented.")
        print("[run_train] Use --stage blocking to run the blocking stage.")
    else:
        print(f"[run_train] Stage '{args.stage}' not yet implemented.")


if __name__ == "__main__":
    main()
