"""
pipeline.py  –  End-to-end orchestration for training and inference.

Provides top-level ``run_train()`` and ``run_infer()`` functions that
chain together normalisation → blocking → features → model → grouping → I/O.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def run_train(cfg: Dict[str, Any], root: Path) -> None:
    """Execute the full training pipeline.

    Steps
    -----
    1. Load & normalise training sources.
    2. Load ground-truth labels.
    3. Generate candidate pairs (blocking).
    4. Build feature matrix + label vector.
    5. Train LightGBM classifier.
    6. Evaluate on validation split.
    7. Save model artifact.

    Parameters
    ----------
    cfg : dict
        Merged configuration (paths + blocking + model).
    root : Path
        Repository root (``student_resource/``).
    """
    raise NotImplementedError("Training pipeline – implement in Milestone 2")


def run_infer(cfg: Dict[str, Any], root: Path) -> None:
    """Execute the full inference pipeline.

    Steps
    -----
    1. Load & normalise test sources.
    2. Generate candidate pairs (blocking).
    3. Build feature matrix.
    4. Load trained model and predict.
    5. Apply threshold & group results.
    6. Write ``matching_results.tsv`` and ``candidate_pairs.tsv``.

    Parameters
    ----------
    cfg : dict
        Merged configuration (paths + blocking + model).
    root : Path
        Repository root (``student_resource/``).
    """
    raise NotImplementedError("Inference pipeline – implement in Milestone 2")
