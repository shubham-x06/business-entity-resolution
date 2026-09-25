"""
model.py  –  LightGBM wrapper for entity-resolution match classification.

Wraps ``lightgbm.LGBMClassifier`` with helpers for training, prediction,
threshold tuning, and model persistence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def build_classifier(model_cfg: Dict[str, Any]) -> Any:
    """Instantiate an ``LGBMClassifier`` from config values.

    Parameters
    ----------
    model_cfg : dict
        Parsed ``configs/model.yaml["lgbm"]``.

    Returns
    -------
    lightgbm.LGBMClassifier
    """
    raise NotImplementedError("Build classifier – implement in Milestone 2")


def train(
    clf: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> Any:
    """Fit the classifier on training data.

    Parameters
    ----------
    clf : LGBMClassifier
    X_train : np.ndarray
    y_train : np.ndarray
    X_val : np.ndarray | None
    y_val : np.ndarray | None

    Returns
    -------
    LGBMClassifier
        The fitted classifier.
    """
    raise NotImplementedError("Train – implement in Milestone 2")


def predict_proba(clf: Any, X: np.ndarray) -> np.ndarray:
    """Return match probabilities.

    Parameters
    ----------
    clf : LGBMClassifier
    X : np.ndarray

    Returns
    -------
    np.ndarray
        1-D array of P(match).
    """
    raise NotImplementedError("Predict – implement in Milestone 2")


def save_model(clf: Any, path: Path) -> None:
    """Persist a trained model to disk.

    Parameters
    ----------
    clf : LGBMClassifier
    path : Path
    """
    raise NotImplementedError("Save model – implement in Milestone 2")


def load_model(path: Path) -> Any:
    """Load a previously saved model from disk.

    Parameters
    ----------
    path : Path

    Returns
    -------
    LGBMClassifier
    """
    raise NotImplementedError("Load model – implement in Milestone 2")
