#!/usr/bin/env python
"""
run_infer.py  –  CLI entry-point for the end-to-end inference pipeline.

Usage
-----
    python scripts/run_infer.py --test-dir dataset/test --out-dir output/
    python scripts/run_infer.py [--test-dir DIR] [--out-dir DIR] [--threshold 0.98]

Loads test sources, trained LightGBM model, runs country-partitioned blocking,
vectorized feature extraction, model scoring at calibrated threshold, and writes:
  - output/matching_results.tsv (every S1 entity has exactly one row)
  - output/candidate_pairs.tsv (every S1 entity has exactly one row)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

# ── Logging setup ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_infer")


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
        description="Run end-to-end inference for entity resolution.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("dataset/test"),
        help="Folder containing test_source1.tsv, test_source2.tsv, test_source3.tsv (default: dataset/test).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output"),
        help="Output folder to write matching_results.tsv and candidate_pairs.tsv (default: output).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.98,
        help="Decision threshold for match inclusion (default: 0.98 frozen from Milestone 8).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root (student_resource/). Default: auto-detected.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Subsample N rows from Source 1 for fast prototyping.",
    )
    parser.add_argument(
        "--country",
        type=str,
        default=None,
        help="Optional country filter ('France', 'India', 'US') for partition testing.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250000,
        help="Vectorized feature computation chunk size (default: 250,000).",
    )
    return parser.parse_args(argv)


def run_full_infer_pipeline(
    root: Path,
    test_dir: Path,
    out_dir: Path,
    threshold: float = 0.98,
    sample: int | None = None,
    country: str | None = None,
    chunk_size: int = 250000,
) -> Dict[str, Any]:
    """Execute the full orchestrated inference pipeline.

    Loads all configs and calls ``pipeline.run_infer_pipeline``.
    """
    from business_entity_resolution.io_utils import load_config
    from business_entity_resolution.pipeline import run_infer_pipeline

    cfg_dir = root / "code" / "business_entity_resolution" / "configs"
    paths_cfg = load_config(cfg_dir / "paths.yaml") if (cfg_dir / "paths.yaml").is_file() else {}
    blocking_cfg = load_config(cfg_dir / "blocking.yaml") if (cfg_dir / "blocking.yaml").is_file() else {}
    model_cfg = load_config(cfg_dir / "model.yaml") if (cfg_dir / "model.yaml").is_file() else {}

    config: Dict[str, Any] = {
        "root": str(root),
        "test_dir": str(test_dir.resolve() if test_dir.is_absolute() else (root / test_dir).resolve()),
        "out_dir": str(out_dir.resolve() if out_dir.is_absolute() else (root / out_dir).resolve()),
        "paths": paths_cfg,
        "blocking": blocking_cfg,
        "model": model_cfg,
        "sample": sample,
        "country": country,
        "threshold": threshold,
        "chunk_size": chunk_size,
    }

    t0 = time.time()
    result = run_infer_pipeline(config)
    total_elapsed = time.time() - t0

    logger.info("=" * 60)
    logger.info("INFERENCE PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("  Completed stages: %s", result["completed"])
    logger.info("  Skipped stages:   %s", result["skipped"])
    logger.info("  Threshold used:   %.3f", result.get("threshold_used", threshold))
    logger.info("  Total elapsed:    %.2fs (%.2f min)", total_elapsed, total_elapsed / 60.0)
    if "output" in result.get("metrics", {}):
        m = result["metrics"]["output"]
        logger.info("  Total S1 Entities:     %s", f"{m.get('total_s1', 0):,}")
        logger.info("  Matched Entities:      %s", f"{m.get('total_matched', 0):,}")
        logger.info("  Singleton Entities:    %s", f"{m.get('total_singletons', 0):,}")
        logger.info("  Candidate Pairs:       %s", f"{m.get('total_candidates', 0):,}")
        logger.info("  Peak RAM:              %.1f MB", m.get("peak_ram_mb", 0.0))
        logger.info("  Matching results:      %s", m.get("matching_results_path"))
        logger.info("  Candidate pairs:       %s", m.get("candidate_pairs_path"))
    logger.info("=" * 60)

    return result


def main(argv: list[str] | None = None) -> None:
    """Entry-point for the inference pipeline."""
    args = parse_args(argv)

    repo_root = args.root or Path(__file__).resolve().parents[3]
    test_dir = args.test_dir
    if not test_dir.is_absolute():
        test_dir = repo_root / test_dir
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir

    logger.info("repo_root = %s", repo_root)
    logger.info("test_dir  = %s", test_dir)
    logger.info("out_dir   = %s", out_dir)
    logger.info("threshold = %.2f", args.threshold)
    if args.country:
        logger.info("country   = %s", args.country)
    if args.sample:
        logger.info("sample    = %d", args.sample)

    src_dir = repo_root / "code" / "business_entity_resolution" / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    run_full_infer_pipeline(
        root=repo_root,
        test_dir=test_dir,
        out_dir=out_dir,
        threshold=args.threshold,
        sample=args.sample,
        country=args.country,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
