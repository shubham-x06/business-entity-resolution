"""
test_scoring.py  –  Unit tests for the official F_0.5 macro evaluation scorer.

Validates:
- README worked example (precision=2/3, recall=1.0, F_0.5 ≈ 0.7142857)
- Singleton handling (correct empty prediction = 1.0, false positive = 0.0)
- Perfect non-empty match (score = 1.0)
- Complete miss (disjoint non-empty sets = 0.0)
- Precision vs Recall asymmetry (precision loss penalized more heavily than recall loss)
- Macro F_beta averaging with singletons, non-singletons, and missing predictions
- Boundary guards (zero division, empty ground truth, None means)
- TSV loader functionality (matching_results and ground_truth format parsing)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "code" / "business_entity_resolution" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.scoring import (
    compute_f_beta,
    load_matches_dict,
    macro_f_beta,
    score_entity,
)


# ── 1. compute_f_beta formula and edge cases ─────────────────────────

class TestComputeFBeta:
    """Tests for raw compute_f_beta function."""

    def test_zero_precision_and_recall(self) -> None:
        assert compute_f_beta(0.0, 0.0, beta=0.5) == 0.0

    def test_zero_precision(self) -> None:
        assert compute_f_beta(0.0, 1.0, beta=0.5) == 0.0

    def test_zero_recall(self) -> None:
        assert compute_f_beta(1.0, 0.0, beta=0.5) == 0.0

    def test_perfect_score(self) -> None:
        assert compute_f_beta(1.0, 1.0, beta=0.5) == 1.0

    def test_readme_worked_example_math(self) -> None:
        # precision = 2/3, recall = 1.0, beta = 0.5
        # F_0.5 = 1.25 * (2/3 * 1.0) / (0.25 * 2/3 + 1.0) = (5/6) / (7/6) = 5/7 ≈ 0.7142857
        result = compute_f_beta(2.0 / 3.0, 1.0, beta=0.5)
        expected = 5.0 / 7.0
        assert abs(result - expected) < 1e-6
        assert abs(result - 0.7142857142857143) < 0.001


# ── 2. score_entity per-entity tests ─────────────────────────────────

class TestScoreEntity:
    """Tests for single-entity scoring per README specifications."""

    def test_readme_worked_example(self) -> None:
        """README worked example: precision = 2/3, recall = 1.0, F_0.5 ≈ 0.714."""
        predicted_ids = {"S2-00047", "S2-00193", "S3-00812"}
        true_ids = {"S2-00047", "S3-00812"}
        result = score_entity(predicted_ids, true_ids, beta=0.5)
        assert abs(result - 0.7142857142857143) < 0.001

    def test_singleton_correctly_predicted_empty(self) -> None:
        """Singleton correctly predicted empty scores 1.0."""
        assert score_entity(set(), set(), beta=0.5) == 1.0

    def test_singleton_false_positive(self) -> None:
        """Singleton with predicted match (false positive) scores 0.0."""
        assert score_entity({"S2-00001"}, set(), beta=0.5) == 0.0

    def test_perfect_non_empty_match(self) -> None:
        """Exact non-empty match scores 1.0."""
        matched = {"S2-00005", "S3-00010"}
        assert score_entity(matched, matched, beta=0.5) == 1.0

    def test_total_miss_disjoint(self) -> None:
        """Total miss (disjoint non-empty sets) scores 0.0."""
        predicted = {"S2-00099"}
        true_ids = {"S2-00001"}
        assert score_entity(predicted, true_ids, beta=0.5) == 0.0

    def test_empty_prediction_for_non_empty_truth(self) -> None:
        """Predicting empty for a non-singleton entity scores 0.0."""
        assert score_entity(set(), {"S2-00001"}, beta=0.5) == 0.0

    def test_precision_loss_penalized_more_than_recall_loss(self) -> None:
        """F_0.5 must penalize precision loss more heavily than recall loss.

        Precision loss: precision = 1/3, recall = 1.0
        Recall loss:    precision = 1.0, recall = 1/3
        """
        # Precision loss (2 false positives)
        score_prec_loss = score_entity(
            predicted_ids={"S2-A", "S2-B", "S2-C"},
            true_ids={"S2-A"},
            beta=0.5,
        )
        # Recall loss (2 false negatives)
        score_rec_loss = score_entity(
            predicted_ids={"S2-A"},
            true_ids={"S2-A", "S2-B", "S2-C"},
            beta=0.5,
        )

        # In F_0.5:
        # prec_loss = 5/13 ≈ 0.3846
        # rec_loss  = 5/7  ≈ 0.7143
        assert score_prec_loss < score_rec_loss
        assert abs(score_prec_loss - (5.0 / 13.0)) < 1e-6
        assert abs(score_rec_loss - (5.0 / 7.0)) < 1e-6


# ── 3. macro_f_beta macro evaluation tests ───────────────────────────

class TestMacroFBeta:
    """Tests for multi-entity macro evaluation."""

    def test_macro_f_beta_mix(self) -> None:
        """Macro evaluation with 2 singletons, 3 has-match, and 1 missing prediction.

        Entity breakdown:
        1. "S1-01" (singleton): pred={}, true={} -> score = 1.0
        2. "S1-02" (singleton): pred={"S2-X"}, true={} -> score = 0.0
        3. "S1-03" (has-match): pred={"S2-A", "S2-B"}, true={"S2-A", "S2-B"} -> score = 1.0
        4. "S1-04" (has-match): pred={"S2-C", "S2-D", "S2-E"}, true={"S2-C"}
           -> prec=1/3, rec=1.0, score = 5/13 ≈ 0.384615
        5. "S1-05" (has-match): MISSING from predictions dict
           -> treated as pred={}, true={"S2-F"} -> score = 0.0
        """
        ground_truth = {
            "S1-01": set(),
            "S1-02": set(),
            "S1-03": {"S2-A", "S2-B"},
            "S1-04": {"S2-C"},
            "S1-05": {"S2-F"},
        }
        predictions = {
            "S1-01": set(),
            "S1-02": {"S2-X"},
            "S1-03": {"S2-A", "S2-B"},
            "S1-04": {"S2-C", "S2-D", "S2-E"},
            # S1-05 is deliberately omitted
        }

        metrics = macro_f_beta(predictions, ground_truth, beta=0.5)

        assert metrics["n_entities"] == 5
        assert metrics["n_singletons"] == 2
        assert metrics["n_has_match"] == 3

        # Singleton mean: (1.0 + 0.0) / 2 = 0.5
        assert abs(metrics["singleton_mean"] - 0.5) < 1e-6

        # Has-match mean: (1.0 + 5/13 + 0.0) / 3 = (18/13) / 3 = 6/13 ≈ 0.461538
        expected_has_match = (1.0 + (5.0 / 13.0) + 0.0) / 3.0
        assert abs(metrics["has_match_mean"] - expected_has_match) < 1e-6

        # Macro mean: (1.0 + 0.0 + 1.0 + 5/13 + 0.0) / 5 = (31/13) / 5 = 31/65 ≈ 0.476923
        expected_macro = (1.0 + 0.0 + 1.0 + (5.0 / 13.0) + 0.0) / 5.0
        assert abs(metrics["macro_f_beta"] - expected_macro) < 1e-6

    def test_macro_f_beta_empty_ground_truth(self) -> None:
        """Empty ground truth returns 0.0 macro score and None means."""
        metrics = macro_f_beta({}, {})
        assert metrics["macro_f_beta"] == 0.0
        assert metrics["n_entities"] == 0
        assert metrics["singleton_mean"] is None
        assert metrics["has_match_mean"] is None
        assert metrics["n_singletons"] == 0
        assert metrics["n_has_match"] == 0

    def test_macro_f_beta_only_singletons(self) -> None:
        """When all entities are singletons, has_match_mean is None."""
        gt = {"S1-01": set(), "S1-02": set()}
        pred = {"S1-01": set(), "S1-02": {"S2-A"}}
        metrics = macro_f_beta(pred, gt)
        assert metrics["n_entities"] == 2
        assert metrics["n_singletons"] == 2
        assert metrics["n_has_match"] == 0
        assert metrics["singleton_mean"] == 0.5
        assert metrics["has_match_mean"] is None
        assert metrics["macro_f_beta"] == 0.5

    def test_macro_f_beta_only_has_match(self) -> None:
        """When no entities are singletons, singleton_mean is None."""
        gt = {"S1-01": {"S2-A"}}
        pred = {"S1-01": {"S2-A"}}
        metrics = macro_f_beta(pred, gt)
        assert metrics["n_entities"] == 1
        assert metrics["n_singletons"] == 0
        assert metrics["n_has_match"] == 1
        assert metrics["singleton_mean"] is None
        assert metrics["has_match_mean"] == 1.0
        assert metrics["macro_f_beta"] == 1.0


# ── 4. load_matches_dict loader tests ────────────────────────────────

class TestLoadMatchesDict:
    """Tests for TSV / DataFrame match dictionary loader."""

    def test_load_from_dataframe(self) -> None:
        df = pd.DataFrame({
            "source1_entity_id": ["S1-001", "S1-002", "S1-003"],
            "matched_entity_ids": ["S2-001,S3-002", "", "S2-003"],
        })
        result = load_matches_dict(df)
        assert result == {
            "S1-001": {"S2-001", "S3-002"},
            "S1-002": set(),
            "S1-003": {"S2-003"},
        }

    def test_load_from_tsv_file(self, tmp_path: Path) -> None:
        tsv_file = tmp_path / "test_matches.tsv"
        tsv_file.write_text(
            "source1_entity_id\tmatched_entity_ids\n"
            "S1-001\tS2-100,S3-200\n"
            "S1-002\t\n"
            "S1-003\tS2-300\n",
            encoding="utf-8",
        )
        result = load_matches_dict(tsv_file)
        assert result["S1-001"] == {"S2-100", "S3-200"}
        assert result["S1-002"] == set()
        assert result["S1-003"] == {"S2-300"}

    def test_load_empty_file(self, tmp_path: Path) -> None:
        empty_file = tmp_path / "empty.tsv"
        empty_file.write_text("", encoding="utf-8")
        result = load_matches_dict(empty_file)
        assert result == {}
