#!/usr/bin/env python
"""
run_train.py  –  CLI entry-point for the training pipeline.

Usage
-----
    python scripts/run_train.py [--root ROOT]

Loads configs, runs normalisation → blocking → features → model training,
and saves the trained model artifact.
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
        description="Train the entity-resolution model.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Repository root (student_resource/).  Default: auto-detected.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry-point for the training pipeline."""
    args = parse_args(argv)
    print(f"[run_train] root = {args.root}")
    print("[run_train] Training pipeline not yet implemented (Milestone 2).")


if __name__ == "__main__":
    main()
