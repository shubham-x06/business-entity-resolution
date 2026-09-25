#!/usr/bin/env python
"""
run_infer.py  –  CLI entry-point for the inference pipeline.

Usage
-----
    python scripts/run_infer.py [--root ROOT]

Loads configs and a trained model, runs blocking → features → prediction →
grouping, and writes output/matching_results.tsv and output/candidate_pairs.tsv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry-point for the inference pipeline."""
    args = parse_args(argv)
    print(f"[run_infer] root = {args.root}")
    print("[run_infer] Inference pipeline not yet implemented (Milestone 2).")


if __name__ == "__main__":
    main()
