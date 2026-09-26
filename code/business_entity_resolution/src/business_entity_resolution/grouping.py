"""
grouping.py  –  Threshold sweep + many-to-many match grouping logic.

After the classifier produces per-pair match probabilities, this module:
1. Groups candidates per Source1 entity at a given threshold (many-to-many).
2. Sweeps thresholds to find the one maximising macro F_0.5.
3. Optionally sweeps per-country thresholds for heterogeneous datasets.

Key design decisions:
- An entity can have MULTIPLE true matches from the same source — this is
  normal, not a bug.  group_by_threshold includes ALL candidates >= threshold,
  not just the top-scoring one.
- An entity with zero candidates above threshold gets an empty set (predicted
  singleton).  This is correct behavior and is not special-cased away.
- The sweep uses scoring.py's macro_f_beta, which already handles singleton
  scoring (correct empty prediction = 1.0, false positive on singleton = 0.0).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from business_entity_resolution.scoring import macro_f_beta

logger = logging.getLogger(__name__)

# ── Default threshold grid ──────────────────────────────────────────────────

DEFAULT_THRESHOLDS: List[float] = [
    round(t / 10.0, 1) for t in range(1, 10)
]  # [0.1, 0.2, ..., 0.9]


# ── Core grouping ───────────────────────────────────────────────────────────

def group_by_threshold(
    scored_pairs: Dict[str, List[Tuple[str, float]]],
    threshold: float,
) -> Dict[str, Set[str]]:
    """Group scored candidate pairs into match sets at a given threshold.

    For each Source1 entity, include ALL candidate_ids with score >= threshold
    (many-to-many: not just the single highest-scoring one).

    An entity with zero candidates above threshold produces an empty set
    (predicted singleton), which is correct behavior.

    Parameters
    ----------
    scored_pairs : dict[str, list[tuple[str, float]]]
        Mapping ``{s1_entity_id: [(candidate_id, score), ...]}``.
        All candidates that passed through the model, regardless of score.
    threshold : float
        Minimum score to include a candidate in the match set.

    Returns
    -------
    dict[str, set[str]]
        ``{s1_entity_id: set(matched_candidate_ids)}``.
    """
    predictions: Dict[str, Set[str]] = {}
    for entity_id, candidates in scored_pairs.items():
        matched = {
            cand_id
            for cand_id, score in candidates
            if score >= threshold
        }
        predictions[entity_id] = matched
    return predictions


# ── Threshold sweep ─────────────────────────────────────────────────────────

def sweep_thresholds(
    scored_pairs: Dict[str, List[Tuple[str, float]]],
    ground_truth: Dict[str, Set[str]],
    thresholds: Optional[List[float]] = None,
    beta: float = 0.5,
) -> Dict[str, Any]:
    """Sweep candidate thresholds and find the one maximising macro F_beta.

    For each threshold in ``thresholds``:
      1. ``group_by_threshold(scored_pairs, threshold)`` → predictions
      2. ``macro_f_beta(predictions, ground_truth)`` → metrics at that threshold

    Parameters
    ----------
    scored_pairs : dict[str, list[tuple[str, float]]]
        Mapping ``{s1_entity_id: [(candidate_id, score), ...]}``.
    ground_truth : dict[str, set[str]]
        Mapping ``{s1_entity_id: set(true_ids)}``.
    thresholds : list[float] | None
        Thresholds to evaluate.  Defaults to ``[0.1, 0.2, ..., 0.9]``.
    beta : float, default=0.5
        F-beta parameter.

    Returns
    -------
    dict
        - "sweep": dict mapping threshold (float) → full metrics dict from
          ``macro_f_beta`` (includes macro_f_beta, singleton_mean,
          has_match_mean, n_entities, n_singletons, n_has_match).
        - "best_threshold": float, threshold with highest macro_f_beta.
        - "best_metrics": dict, full metrics at the best threshold.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    sweep_results: Dict[float, Dict[str, Any]] = {}
    best_threshold: float = thresholds[0]
    best_score: float = -1.0
    best_metrics: Dict[str, Any] = {}

    for t in thresholds:
        predictions = group_by_threshold(scored_pairs, t)
        metrics = macro_f_beta(predictions, ground_truth, beta=beta)
        sweep_results[t] = metrics

        logger.debug(
            "threshold=%.2f  macro_f_beta=%.4f  singleton_mean=%s  "
            "has_match_mean=%s",
            t,
            metrics["macro_f_beta"],
            metrics.get("singleton_mean"),
            metrics.get("has_match_mean"),
        )

        if metrics["macro_f_beta"] > best_score:
            best_score = metrics["macro_f_beta"]
            best_threshold = t
            best_metrics = metrics

    logger.info(
        "Sweep complete: best threshold=%.2f  macro_f_beta=%.4f",
        best_threshold,
        best_score,
    )

    return {
        "sweep": sweep_results,
        "best_threshold": best_threshold,
        "best_metrics": best_metrics,
    }


# ── Per-country threshold sweep ────────────────────────────────────────────

def sweep_thresholds_per_country(
    scored_pairs: Dict[str, List[Tuple[str, float]]],
    ground_truth: Dict[str, Set[str]],
    entity_countries: Dict[str, str],
    thresholds: Optional[List[float]] = None,
    beta: float = 0.5,
) -> Dict[str, Dict[str, Any]]:
    """Sweep thresholds independently per country.

    Useful when a global threshold underperforms for specific countries
    (e.g. a country with zero training precedent).

    Parameters
    ----------
    scored_pairs : dict[str, list[tuple[str, float]]]
        Mapping ``{s1_entity_id: [(candidate_id, score), ...]}``.
    ground_truth : dict[str, set[str]]
        Mapping ``{s1_entity_id: set(true_ids)}``.
    entity_countries : dict[str, str]
        Mapping ``{s1_entity_id: country_code}``.  Entities missing from
        this mapping are grouped under the key ``"_unknown"``.
    thresholds : list[float] | None
        Thresholds to evaluate.  Defaults to ``[0.1, 0.2, ..., 0.9]``.
    beta : float, default=0.5
        F-beta parameter.

    Returns
    -------
    dict[str, dict]
        Mapping ``{country_code: sweep_result}`` where each ``sweep_result``
        has the same shape as the return value of ``sweep_thresholds``.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    # ── Partition scored_pairs and ground_truth by country ───────────
    country_scored: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
    country_gt: Dict[str, Dict[str, Set[str]]] = {}

    # Partition ground_truth first (it defines the entity universe)
    for entity_id, true_ids in ground_truth.items():
        country = entity_countries.get(entity_id, "_unknown")
        country_gt.setdefault(country, {})[entity_id] = true_ids

    # Partition scored_pairs
    for entity_id, candidates in scored_pairs.items():
        country = entity_countries.get(entity_id, "_unknown")
        country_scored.setdefault(country, {})[entity_id] = candidates

    # ── Sweep per country ───────────────────────────────────────────
    results: Dict[str, Dict[str, Any]] = {}
    all_countries = set(country_gt.keys()) | set(country_scored.keys())

    for country in sorted(all_countries):
        c_scored = country_scored.get(country, {})
        c_gt = country_gt.get(country, {})

        if not c_gt:
            logger.warning("Country %s has no ground truth entities", country)
            continue

        country_result = sweep_thresholds(
            c_scored, c_gt, thresholds=thresholds, beta=beta,
        )
        results[country] = country_result

        logger.info(
            "Country %s: best threshold=%.2f  macro_f_beta=%.4f  "
            "(%d entities)",
            country,
            country_result["best_threshold"],
            country_result["best_metrics"]["macro_f_beta"],
            country_result["best_metrics"]["n_entities"],
        )

    return results
