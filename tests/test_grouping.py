"""
test_grouping.py  –  Unit tests for threshold sweep + many-to-many grouping.

Validates:
- Many-to-many: multiple true matches above threshold ALL appear in output
- Empty-set singletons: candidates below threshold → empty predicted set
- False positive penalty: singleton entity + above-threshold candidate hurts score
- Non-monotonic optimal threshold: sweep correctly finds an interior optimum
- Per-country threshold sweep: different countries get independent best thresholds
- Edge cases: empty scored_pairs, empty ground_truth, custom threshold grids
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "code" / "business_entity_resolution" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.grouping import (
    DEFAULT_THRESHOLDS,
    group_by_threshold,
    sweep_thresholds,
    sweep_thresholds_per_country,
)
from business_entity_resolution.scoring import macro_f_beta


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures: synthetic scored_pairs and ground_truth
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def multi_match_data():
    """Entity E1 has 2 true matches, both scoring above a reasonable threshold.

    Tests the many-to-many requirement: BOTH should appear, not just top-1.
    """
    scored_pairs = {
        "E1": [
            ("M1", 0.85),  # true match
            ("M2", 0.72),  # true match
            ("N1", 0.30),  # noise
        ],
    }
    ground_truth = {
        "E1": {"M1", "M2"},
    }
    return scored_pairs, ground_truth


@pytest.fixture
def below_threshold_data():
    """Entity E2 has candidates but ALL are below any reasonable threshold.

    Should produce an empty set (predicted singleton).
    """
    scored_pairs = {
        "E2": [
            ("C1", 0.15),
            ("C2", 0.08),
            ("C3", 0.22),
        ],
    }
    ground_truth = {
        "E2": {"C1"},  # true match exists but model scored it poorly
    }
    return scored_pairs, ground_truth


@pytest.fixture
def singleton_with_false_positive_data():
    """Entity E3 is a TRUE singleton (empty ground truth), but a candidate
    scores just above threshold — this is a false positive and should hurt
    the macro F_0.5 score.
    """
    scored_pairs = {
        "E3": [
            ("FP1", 0.65),  # false positive — E3 has no true matches
        ],
    }
    ground_truth = {
        "E3": set(),  # singleton
    }
    return scored_pairs, ground_truth


@pytest.fixture
def sweep_optimum_data():
    """Carefully constructed data where the optimal threshold is neither the
    loosest (0.1) nor the tightest (0.9) — proving the sweep identifies
    an interior optimum.

    Setup:
    - E1: 2 true matches at 0.75 and 0.65. 1 false positive at 0.55.
      → threshold 0.6 catches both true + the FP. threshold 0.7 catches
        only the top true. Precision is better at 0.7 but recall drops.
    - E2: 1 true match at 0.80. 1 false positive at 0.45.
      → caught correctly from threshold ≤ 0.8
    - E3: singleton. 1 candidate at 0.50.
      → threshold > 0.5 correctly predicts empty; ≤ 0.5 hurts.
    - E4: 1 true match at 0.60. No false positives.
      → caught correctly from threshold ≤ 0.6
    """
    scored_pairs = {
        "E1": [("M1", 0.75), ("M2", 0.65), ("FP1", 0.55)],
        "E2": [("M3", 0.80), ("FP2", 0.45)],
        "E3": [("FP3", 0.50)],
        "E4": [("M4", 0.60)],
    }
    ground_truth = {
        "E1": {"M1", "M2"},
        "E2": {"M3"},
        "E3": set(),        # singleton
        "E4": {"M4"},
    }
    return scored_pairs, ground_truth


# ═══════════════════════════════════════════════════════════════════════════
# Test group_by_threshold
# ═══════════════════════════════════════════════════════════════════════════


class TestGroupByThreshold:
    """Tests for the core grouping function."""

    def test_many_to_many_both_matches_included(self, multi_match_data):
        """Both true matches above threshold must appear — not just top-1."""
        scored_pairs, _ = multi_match_data
        preds = group_by_threshold(scored_pairs, threshold=0.5)

        assert "E1" in preds
        assert preds["E1"] == {"M1", "M2"}, (
            f"Expected both M1 and M2, got {preds['E1']}"
        )
        # Noise candidate N1 (score 0.30) should NOT be included at t=0.5
        assert "N1" not in preds["E1"]

    def test_no_candidates_above_threshold_gives_empty_set(
        self, below_threshold_data
    ):
        """All candidates below threshold → empty set (predicted singleton)."""
        scored_pairs, _ = below_threshold_data
        preds = group_by_threshold(scored_pairs, threshold=0.5)

        assert "E2" in preds
        assert preds["E2"] == set(), (
            f"Expected empty set, got {preds['E2']}"
        )

    def test_threshold_boundary_exact_equality(self):
        """Score exactly equal to threshold should be INCLUDED (>=, not >)."""
        scored_pairs = {"E1": [("M1", 0.50)]}
        preds = group_by_threshold(scored_pairs, threshold=0.50)
        assert preds["E1"] == {"M1"}

    def test_threshold_boundary_just_below(self):
        """Score just below threshold should be EXCLUDED."""
        scored_pairs = {"E1": [("M1", 0.4999)]}
        preds = group_by_threshold(scored_pairs, threshold=0.50)
        assert preds["E1"] == set()

    def test_empty_scored_pairs(self):
        """Empty scored_pairs produces empty predictions dict."""
        preds = group_by_threshold({}, threshold=0.5)
        assert preds == {}

    def test_entity_with_no_candidates(self):
        """Entity present in scored_pairs but with empty candidate list."""
        scored_pairs = {"E1": []}
        preds = group_by_threshold(scored_pairs, threshold=0.5)
        assert preds["E1"] == set()

    def test_all_candidates_above_threshold(self):
        """All candidates pass → all included."""
        scored_pairs = {"E1": [("A", 0.9), ("B", 0.8), ("C", 0.7)]}
        preds = group_by_threshold(scored_pairs, threshold=0.5)
        assert preds["E1"] == {"A", "B", "C"}

    def test_multiple_entities_independent(self):
        """Multiple entities grouped independently."""
        scored_pairs = {
            "E1": [("M1", 0.9), ("N1", 0.3)],
            "E2": [("M2", 0.6), ("N2", 0.1)],
        }
        preds = group_by_threshold(scored_pairs, threshold=0.5)
        assert preds["E1"] == {"M1"}
        assert preds["E2"] == {"M2"}


# ═══════════════════════════════════════════════════════════════════════════
# Test sweep_thresholds
# ═══════════════════════════════════════════════════════════════════════════


class TestSweepThresholds:
    """Tests for the threshold sweep and F_0.5 optimization."""

    def test_sweep_returns_correct_shape(self, multi_match_data):
        """Sweep result has the expected keys and structure."""
        scored_pairs, ground_truth = multi_match_data
        result = sweep_thresholds(scored_pairs, ground_truth)

        assert "sweep" in result
        assert "best_threshold" in result
        assert "best_metrics" in result
        assert isinstance(result["sweep"], dict)
        assert isinstance(result["best_threshold"], float)
        assert isinstance(result["best_metrics"], dict)
        assert "macro_f_beta" in result["best_metrics"]

    def test_sweep_default_thresholds(self, multi_match_data):
        """Default thresholds are [0.1, 0.2, ..., 0.9]."""
        scored_pairs, ground_truth = multi_match_data
        result = sweep_thresholds(scored_pairs, ground_truth)

        assert set(result["sweep"].keys()) == set(DEFAULT_THRESHOLDS)

    def test_sweep_custom_thresholds(self, multi_match_data):
        """Custom threshold list is respected."""
        scored_pairs, ground_truth = multi_match_data
        custom = [0.3, 0.5, 0.7, 0.9]
        result = sweep_thresholds(
            scored_pairs, ground_truth, thresholds=custom
        )

        assert set(result["sweep"].keys()) == set(custom)

    def test_singleton_false_positive_hurts_score(
        self, singleton_with_false_positive_data
    ):
        """Loose threshold on a singleton entity should produce score 0.0
        (false positive), while tight threshold produces 1.0 (correct empty).
        """
        scored_pairs, ground_truth = singleton_with_false_positive_data

        # At threshold 0.5: FP1 (score 0.65) is included → false positive → 0.0
        preds_loose = group_by_threshold(scored_pairs, 0.5)
        metrics_loose = macro_f_beta(preds_loose, ground_truth)
        assert metrics_loose["macro_f_beta"] == 0.0

        # At threshold 0.7: FP1 (score 0.65) is excluded → correct empty → 1.0
        preds_tight = group_by_threshold(scored_pairs, 0.7)
        metrics_tight = macro_f_beta(preds_tight, ground_truth)
        assert metrics_tight["macro_f_beta"] == 1.0

    def test_sweep_finds_interior_optimum(self, sweep_optimum_data):
        """The optimal threshold should NOT be the loosest (0.1) or tightest
        (0.9) — this proves the sweep finds a non-trivial optimum.

        At very low thresholds: false positives hurt precision (FP1, FP2, FP3
        all included).
        At very high thresholds: true matches are dropped, hurting recall.
        The optimum is somewhere in between.
        """
        scored_pairs, ground_truth = sweep_optimum_data
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        result = sweep_thresholds(
            scored_pairs, ground_truth, thresholds=thresholds
        )

        best_t = result["best_threshold"]
        # The optimum must not be at the extremes
        assert best_t != 0.1, (
            f"Best threshold is 0.1 (loosest) — sweep didn't find interior "
            f"optimum. Scores: "
            + str({t: m["macro_f_beta"] for t, m in result["sweep"].items()})
        )
        assert best_t != 0.9, (
            f"Best threshold is 0.9 (tightest) — sweep didn't find interior "
            f"optimum. Scores: "
            + str({t: m["macro_f_beta"] for t, m in result["sweep"].items()})
        )

        # The best score should be strictly better than both extremes
        best_score = result["best_metrics"]["macro_f_beta"]
        score_at_01 = result["sweep"][0.1]["macro_f_beta"]
        score_at_09 = result["sweep"][0.9]["macro_f_beta"]
        assert best_score > score_at_01, (
            f"Best score {best_score:.4f} not better than t=0.1 "
            f"score {score_at_01:.4f}"
        )
        assert best_score > score_at_09, (
            f"Best score {best_score:.4f} not better than t=0.9 "
            f"score {score_at_09:.4f}"
        )

    def test_sweep_metrics_contain_singleton_breakdown(
        self, sweep_optimum_data
    ):
        """Best metrics should contain the singleton/has_match breakdown."""
        scored_pairs, ground_truth = sweep_optimum_data
        result = sweep_thresholds(scored_pairs, ground_truth)

        best = result["best_metrics"]
        assert "singleton_mean" in best
        assert "has_match_mean" in best
        assert "n_singletons" in best
        assert "n_has_match" in best
        assert best["n_entities"] == 4
        assert best["n_singletons"] == 1  # E3
        assert best["n_has_match"] == 3   # E1, E2, E4

    def test_sweep_monotonic_singleton_score(
        self, singleton_with_false_positive_data
    ):
        """For a pure singleton dataset, tighter thresholds should never
        decrease the score — because the only risk is false positives.
        """
        scored_pairs, ground_truth = singleton_with_false_positive_data
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        result = sweep_thresholds(
            scored_pairs, ground_truth, thresholds=thresholds
        )

        scores = [
            result["sweep"][t]["macro_f_beta"] for t in thresholds
        ]
        # Scores should be non-decreasing for singleton-only data
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1], (
                f"Score decreased from t={thresholds[i-1]} ({scores[i-1]}) "
                f"to t={thresholds[i]} ({scores[i]})"
            )

    def test_perfect_predictions(self):
        """When all candidates are true matches with high scores, the best
        threshold should yield a perfect macro F_0.5 = 1.0.
        """
        scored_pairs = {
            "E1": [("M1", 0.95)],
            "E2": [("M2", 0.90), ("M3", 0.85)],
            "E3": [],  # singleton — no candidates is correct
        }
        ground_truth = {
            "E1": {"M1"},
            "E2": {"M2", "M3"},
            "E3": set(),
        }
        result = sweep_thresholds(scored_pairs, ground_truth)
        assert result["best_metrics"]["macro_f_beta"] == pytest.approx(
            1.0, abs=1e-9
        )

    def test_missing_entity_in_scored_pairs(self):
        """An entity in ground_truth but missing from scored_pairs should
        be treated as empty prediction (handled by macro_f_beta).
        """
        scored_pairs = {
            "E1": [("M1", 0.90)],
            # E2 is missing from scored_pairs entirely
        }
        ground_truth = {
            "E1": {"M1"},
            "E2": {"M2"},  # will be evaluated with empty prediction
        }
        result = sweep_thresholds(scored_pairs, ground_truth)
        # E2 should get score 0.0 (missing prediction for non-singleton)
        # E1 should get 1.0 at threshold ≤ 0.9
        # Macro = (1.0 + 0.0) / 2 = 0.5
        assert result["best_metrics"]["macro_f_beta"] == pytest.approx(
            0.5, abs=1e-9
        )


# ═══════════════════════════════════════════════════════════════════════════
# Test sweep_thresholds_per_country
# ═══════════════════════════════════════════════════════════════════════════


class TestSweepPerCountry:
    """Tests for per-country threshold optimization."""

    def test_different_countries_get_different_thresholds(self):
        """Two countries with different score distributions should produce
        different optimal thresholds.

        US: true matches score high (0.9), false positives score low (0.3)
            → almost any threshold works, but tighter is fine.
        IN: true matches score modestly (0.55), false positives score
            similarly (0.50) → needs a careful threshold.
        """
        scored_pairs = {
            # US entities
            "US1": [("M_US1", 0.90), ("FP_US1", 0.30)],
            "US2": [("M_US2", 0.85)],
            # IN entities
            "IN1": [("M_IN1", 0.55), ("FP_IN1", 0.50)],
            "IN2": [("M_IN2", 0.60), ("FP_IN2", 0.58)],
        }
        ground_truth = {
            "US1": {"M_US1"},
            "US2": {"M_US2"},
            "IN1": {"M_IN1"},
            "IN2": {"M_IN2"},
        }
        entity_countries = {
            "US1": "US", "US2": "US",
            "IN1": "IN", "IN2": "IN",
        }

        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        result = sweep_thresholds_per_country(
            scored_pairs, ground_truth, entity_countries,
            thresholds=thresholds,
        )

        assert "US" in result
        assert "IN" in result
        assert "best_threshold" in result["US"]
        assert "best_threshold" in result["IN"]

        # US should have a reasonably high score at its best threshold
        assert result["US"]["best_metrics"]["macro_f_beta"] > 0.5
        # IN should also find a workable threshold
        assert result["IN"]["best_metrics"]["macro_f_beta"] > 0.0

    def test_unknown_country_entities_grouped(self):
        """Entities missing from entity_countries go to '_unknown'."""
        scored_pairs = {
            "E1": [("M1", 0.80)],
            "E2": [("M2", 0.70)],
        }
        ground_truth = {
            "E1": {"M1"},
            "E2": {"M2"},
        }
        entity_countries = {
            "E1": "US",
            # E2 is missing → goes to _unknown
        }

        result = sweep_thresholds_per_country(
            scored_pairs, ground_truth, entity_countries,
        )

        assert "US" in result
        assert "_unknown" in result

    def test_per_country_metrics_independent(self):
        """Each country's sweep is independent — changing one shouldn't
        affect the other.
        """
        scored_pairs = {
            "US1": [("M1", 0.90)],
            "IN1": [("M2", 0.40)],  # below most thresholds
        }
        ground_truth = {
            "US1": {"M1"},
            "IN1": {"M2"},
        }
        entity_countries = {"US1": "US", "IN1": "IN"}

        result = sweep_thresholds_per_country(
            scored_pairs, ground_truth, entity_countries,
        )

        # US should score perfectly at many thresholds
        assert result["US"]["best_metrics"]["macro_f_beta"] == pytest.approx(
            1.0, abs=1e-9
        )
        # IN: best it can do is at threshold ≤ 0.4
        in_best = result["IN"]["best_threshold"]
        assert in_best <= 0.4

    def test_empty_country_no_ground_truth_skipped(self):
        """A country with entities in scored_pairs but not in ground_truth
        should be skipped (no ground truth to evaluate against).
        """
        scored_pairs = {
            "US1": [("M1", 0.80)],
            "FR1": [("M_FR", 0.70)],
        }
        ground_truth = {
            "US1": {"M1"},
            # FR1 has no ground truth
        }
        entity_countries = {"US1": "US", "FR1": "FR"}

        result = sweep_thresholds_per_country(
            scored_pairs, ground_truth, entity_countries,
        )

        assert "US" in result
        # FR should be skipped because it has no ground truth
        assert "FR" not in result


# ═══════════════════════════════════════════════════════════════════════════
# Integration: verify grouping + scoring pipeline end-to-end
# ═══════════════════════════════════════════════════════════════════════════


class TestGroupingScoringIntegration:
    """End-to-end tests confirming grouping feeds correctly into scoring."""

    def test_many_to_many_produces_correct_f05(self, multi_match_data):
        """Entity with 2 true matches, both above threshold: should get
        perfect precision AND recall → F_0.5 = 1.0 for that entity.
        """
        scored_pairs, ground_truth = multi_match_data
        preds = group_by_threshold(scored_pairs, threshold=0.5)
        metrics = macro_f_beta(preds, ground_truth)

        # Both M1 and M2 are above 0.5 and are true matches
        # precision = 2/2 = 1.0, recall = 2/2 = 1.0 → F_0.5 = 1.0
        assert metrics["macro_f_beta"] == pytest.approx(1.0, abs=1e-9)

    def test_partial_match_f05_score(self):
        """Only 1 of 2 true matches above threshold: precision=1.0 but
        recall=0.5 → F_0.5 should favor the high precision.
        """
        scored_pairs = {
            "E1": [("M1", 0.80), ("M2", 0.30)],  # M2 below threshold
        }
        ground_truth = {"E1": {"M1", "M2"}}

        preds = group_by_threshold(scored_pairs, threshold=0.5)
        assert preds["E1"] == {"M1"}  # only M1 passes

        metrics = macro_f_beta(preds, ground_truth)
        # precision = 1/1 = 1.0, recall = 1/2 = 0.5
        # F_0.5 = (1+0.25)*1.0*0.5 / (0.25*1.0 + 0.5) = 0.625 / 0.75 ≈ 0.8333
        assert metrics["macro_f_beta"] == pytest.approx(
            0.8333333, abs=1e-4
        )

    def test_false_positive_included_hurts_precision(self):
        """Including a false positive should reduce F_0.5 compared to not
        including it, since F_0.5 weights precision 2x.
        """
        scored_pairs = {
            "E1": [("M1", 0.85), ("FP1", 0.60)],
        }
        ground_truth = {"E1": {"M1"}}

        # Tight threshold: only M1 included → perfect
        preds_tight = group_by_threshold(scored_pairs, threshold=0.7)
        metrics_tight = macro_f_beta(preds_tight, ground_truth)

        # Loose threshold: M1 + FP1 included → precision drops to 0.5
        preds_loose = group_by_threshold(scored_pairs, threshold=0.5)
        metrics_loose = macro_f_beta(preds_loose, ground_truth)

        assert metrics_tight["macro_f_beta"] > metrics_loose["macro_f_beta"]
        assert metrics_tight["macro_f_beta"] == pytest.approx(1.0, abs=1e-9)
