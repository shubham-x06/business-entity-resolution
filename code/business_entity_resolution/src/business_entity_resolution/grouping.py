"""
grouping.py  –  Post-model grouping and transitive-closure logic.

After the classifier produces per-pair match probabilities, this module
aggregates results per Source 1 entity and (optionally) applies
transitive-closure to propagate matches.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


def threshold_pairs(
    pair_df: pd.DataFrame,
    proba: list[float],
    threshold: float = 0.5,
) -> List[Tuple[str, str]]:
    """Keep only pairs whose match probability ≥ *threshold*.

    Parameters
    ----------
    pair_df : pd.DataFrame
        Must contain ``s1_id`` and ``cand_id`` columns.
    proba : list[float]
        Match probabilities aligned with *pair_df* rows.
    threshold : float

    Returns
    -------
    list[tuple[str, str]]
        Accepted ``(s1_id, cand_id)`` pairs.
    """
    raise NotImplementedError("Threshold filtering – implement in Milestone 2")


def group_matches(
    accepted_pairs: List[Tuple[str, str]],
    all_s1_ids: List[str],
) -> Dict[str, List[str]]:
    """Group accepted pairs by Source 1 entity.

    Parameters
    ----------
    accepted_pairs : list[tuple[str, str]]
    all_s1_ids : list[str]
        Every Source 1 entity_id (ensures singletons get an empty list).

    Returns
    -------
    dict[str, list[str]]
        ``{s1_id: [matched_ids]}``
    """
    raise NotImplementedError("Grouping – implement in Milestone 2")
