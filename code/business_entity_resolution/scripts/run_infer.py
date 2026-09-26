#!/usr/bin/env python
"""
run_infer.py  –  CLI entry-point for the inference pipeline.

Usage
-----
    python scripts/run_infer.py [--root ROOT] [--sample N] [--threshold T]

Loads configs and a trained model, runs blocking → features → prediction →
grouping, and writes output/matching_results.tsv and output/candidate_pairs.tsv.
"""

from __future__ import annotations

import argparse
import logging
import sys
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
        description="Run inference with the entity-resolution model.",
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
        "--threshold",
        type=float,
        default=None,
        help=(
            "Match probability threshold.  Overrides model.yaml's "
            "threshold.match value."
        ),
    )
    return parser.parse_args(argv)


def run_full_infer_pipeline(
    root: Path,
    sample: int | None = None,
    threshold: float | None = None,
) -> Dict[str, Any]:
    """Execute the full orchestrated inference pipeline.

    Loads all configs and calls ``pipeline.run_infer_pipeline``.
    """
    from business_entity_resolution.io_utils import load_config
    from business_entity_resolution.pipeline import run_infer_pipeline

    cfg_dir = root / "code" / "business_entity_resolution" / "configs"
    paths_cfg = load_config(cfg_dir / "paths.yaml")
    blocking_cfg = load_config(cfg_dir / "blocking.yaml")
    model_cfg = load_config(cfg_dir / "model.yaml")

    config: Dict[str, Any] = {
        "root": str(root),
        "paths": paths_cfg,
        "blocking": blocking_cfg,
        "model": model_cfg,
        "sample": sample,
    }
    if threshold is not None:
        config["threshold"] = threshold

    result = run_infer_pipeline(config)

    logger.info("Pipeline summary:")
    logger.info("  Completed stages: %s", result["completed"])
    logger.info("  Skipped stages:   %s", result["skipped"])
    logger.info("  Threshold used:   %.3f", result.get("threshold_used", 0.5))
    logger.info("  Total elapsed:    %.2fs", result["elapsed_total_seconds"])

    return result


def main(argv: list[str] | None = None) -> None:
    """Entry-point for the inference pipeline."""
    args = parse_args(argv)
    logger.info("root = %s", args.root)
    if args.sample:
        logger.info("sample = %d", args.sample)
    if args.threshold is not None:
        logger.info("threshold = %.3f", args.threshold)

    # Add src to sys.path so package imports resolve cleanly
    src_dir = args.root / "code" / "business_entity_resolution" / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    run_full_infer_pipeline(
        args.root, sample=args.sample, threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
