"""
scoring.py  –  Official Amazon ML Challenge 2026 evaluation metric scorer.

Implements the exact challenge macro F_0.5 evaluation metric:
1. Per-entity F_beta scoring (default beta=0.5, precision weighted 2x over recall).
2. Singleton handling:
   - A Source1 entity with zero true matches is a "singleton".
   - If ground_truth is empty:
       - Correctly predicted empty set -> score = 1.0
       - Predicted any match (false positive) -> score = 0.0
3. Non-singleton handling:
   - precision = |predicted & true| / |predicted| if predicted else 0.0
   - recall = |predicted & true| / |true|
   - F_0.5 = (1 + 0.5^2) * precision * recall / (0.5^2 * precision + recall)
4. Macro-averaging:
   - Macro average across ALL Source1 entities in ground_truth.
   - Missing prediction in predictions dict is treated as empty set (set()).
   - Reports overall macro F_0.5, singleton mean, has-match mean, and entity counts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set, Union

import pandas as pd

from business_entity_resolution.io_utils import load_tsv

logger = logging.getLogger(__name__)


def compute_f_beta(
    precision: float,
    recall: float,
    beta: float = 0.5,
) -> float:
    """Compute the F_beta score given precision, recall, and beta.

    Standard formula:
        F_beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)

    Parameters
    ----------
    precision : float
        Precision score in [0.0, 1.0].
    recall : float
        Recall score in [0.0, 1.0].
    beta : float, default=0.5
        Weight of precision vs recall (beta=0.5 weights precision 2x over recall).

    Returns
    -------
    float
        Computed F_beta score in [0.0, 1.0]. Guarded against division by zero.
    """
    if precision <= 0.0 or recall <= 0.0:
        return 0.0

    beta_sq = beta * beta
    denom = (beta_sq * precision) + recall
    if denom <= 0.0:
        return 0.0

    numerator = (1.0 + beta_sq) * precision * recall
    return numerator / denom


def score_entity(
    predicted_ids: Set[str],
    true_ids: Set[str],
    beta: float = 0.5,
) -> float:
    """Score exactly ONE Source1 entity's predictions against its ground truth.

    Parameters
    ----------
    predicted_ids : set[str]
        Set of predicted matched entity IDs.
    true_ids : set[str]
        Set of true matched entity IDs from ground truth.
    beta : float, default=0.5
        F-beta parameter (0.5 for challenge specification).

    Returns
    -------
    float
        Score in [0.0, 1.0].
    """
    # ── Singleton special case ──────────────────────────────────────
    if not true_ids:
        # True match set is empty: correct empty prediction = 1.0, any match = 0.0
        return 1.0 if not predicted_ids else 0.0

    # ── Non-singleton case ──────────────────────────────────────────
    if not predicted_ids:
        return 0.0

    tp = len(predicted_ids & true_ids)
    if tp == 0:
        return 0.0

    precision = tp / len(predicted_ids)
    recall = tp / len(true_ids)
    return compute_f_beta(precision, recall, beta=beta)


def macro_f_beta(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    beta: float = 0.5,
) -> Dict[str, Any]:
    """Compute macro-averaged F_beta score across all Source1 entities.

    The ``ground_truth`` dictionary defines the authoritative universe of
    entities to score.  Any entity in ``ground_truth`` that is missing from
    ``predictions`` is evaluated with an empty predicted set (``set()``).

    Parameters
    ----------
    predictions : dict[str, set[str]]
        Mapping ``{source1_id: set(predicted_ids)}``.
    ground_truth : dict[str, set[str]]
        Mapping ``{source1_id: set(true_ids)}``.
    beta : float, default=0.5
        F-beta parameter (0.5 for challenge specification).

    Returns
    -------
    dict
        - "macro_f_beta": float, arithmetic mean across all entities
        - "n_entities": int, total entities evaluated
        - "singleton_mean": float | None, mean score of singleton entities
        - "has_match_mean": float | None, mean score of non-singleton entities
        - "n_singletons": int, count of singleton entities
        - "n_has_match": int, count of non-singleton entities
    """
    all_scores = []
    singleton_scores = []
    has_match_scores = []

    for entity_id, true_ids in ground_truth.items():
        predicted_ids = predictions.get(entity_id, set())
        s = score_entity(predicted_ids, true_ids, beta=beta)
        all_scores.append(s)

        if not true_ids:
            singleton_scores.append(s)
        else:
            has_match_scores.append(s)

    n_entities = len(all_scores)
    macro_score = sum(all_scores) / n_entities if n_entities > 0 else 0.0

    singleton_mean = (
        sum(singleton_scores) / len(singleton_scores)
        if singleton_scores
        else None
    )
    has_match_mean = (
        sum(has_match_scores) / len(has_match_scores)
        if has_match_scores
        else None
    )

    return {
        "macro_f_beta": macro_score,
        "n_entities": n_entities,
        "singleton_mean": singleton_mean,
        "has_match_mean": has_match_mean,
        "n_singletons": len(singleton_scores),
        "n_has_match": len(has_match_scores),
    }


def load_matches_dict(
    source: Union[Path, str, pd.DataFrame],
) -> Dict[str, Set[str]]:
    """Load matching results or ground truth into a ``{entity_id: set(ids)}`` dictionary.

    Compatible with both ``matching_results.tsv`` and ``train_ground_truth.tsv``.
    Empty string cells in the matched IDs column represent singletons (empty set).

    Parameters
    ----------
    source : Path | str | pd.DataFrame
        Path to TSV file, or an already-loaded DataFrame.

    Returns
    -------
    dict[str, set[str]]
        Mapping ``{source1_entity_id: set(matched_entity_ids)}``.
    """
    if isinstance(source, pd.DataFrame):
        df = source
    else:
        p = Path(source)
        if not p.is_file() or p.stat().st_size == 0:
            return {}
        try:
            df = load_tsv(p)
        except pd.errors.EmptyDataError:
            return {}

    if df.empty:
        return {}

    id_col = "source1_entity_id" if "source1_entity_id" in df.columns else df.columns[0]
    match_col = "matched_entity_ids" if "matched_entity_ids" in df.columns else df.columns[1]

    ids = df[id_col].tolist()
    matches = df[match_col].tolist()

    result: Dict[str, Set[str]] = {}
    for eid, m_str in zip(ids, matches):
        s_eid = str(eid)
        if m_str and m_str != "nan":
            result[s_eid] = {mid.strip() for mid in str(m_str).split(",") if mid.strip()}
        else:
            result[s_eid] = set()

    return result
