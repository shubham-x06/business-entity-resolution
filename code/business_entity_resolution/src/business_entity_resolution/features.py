"""
features.py  –  Pairwise feature engineering for entity resolution.

Computes string-similarity features (via ``rapidfuzz``) and token-overlap
metrics for each candidate pair produced by the blocking stage.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def compute_name_features(name_a: str, name_b: str) -> Dict[str, float]:
    """Compute name-similarity features for a single pair.

    Features include ``rapidfuzz.fuzz.ratio``, ``token_sort_ratio``,
    ``token_set_ratio``, and ``partial_ratio``.

    Parameters
    ----------
    name_a : str
        Normalised name from Source 1.
    name_b : str
        Normalised name from Source 2/3.

    Returns
    -------
    dict[str, float]
    """
    raise NotImplementedError("Name features – implement in Milestone 2")


def compute_address_features(addr_a: str, addr_b: str) -> Dict[str, float]:
    """Compute address-similarity features for a single pair.

    Parameters
    ----------
    addr_a : str
    addr_b : str

    Returns
    -------
    dict[str, float]
    """
    raise NotImplementedError("Address features – implement in Milestone 2")


def compute_country_features(country_a: str, country_b: str) -> Dict[str, float]:
    """Compute country-match feature.

    Parameters
    ----------
    country_a : str
    country_b : str

    Returns
    -------
    dict[str, float]
        ``{"country_match": 1.0}`` if same country, else ``0.0``.
    """
    return {"country_match": 1.0 if country_a == country_b else 0.0}


def build_feature_matrix(
    pairs: List[Tuple[str, str]],
    source1: pd.DataFrame,
    candidates: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Build a feature matrix for a list of candidate pairs.

    Parameters
    ----------
    pairs : list[tuple[str, str]]
        ``(source1_id, candidate_id)`` pairs.
    source1 : pd.DataFrame
    candidates : pd.DataFrame
        Combined Source 2 + Source 3 records.

    Returns
    -------
    pair_df : pd.DataFrame
        DataFrame with ``s1_id``, ``cand_id`` columns and all features.
    X : np.ndarray
        Numeric feature matrix ready for the model.
    """
    raise NotImplementedError("Feature matrix – implement in Milestone 2")
