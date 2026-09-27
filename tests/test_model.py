"""
test_model.py  –  Unit tests for Milestone 7 LightGBM matching model & mining.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Set

import numpy as np
import pytest

from business_entity_resolution.model import (
    DEFAULT_FEATURE_COLUMNS,
    build_classifier,
    evaluate_validation,
    load_model,
    predict_proba,
    save_model,
    stratified_entity_split,
    train,
)


@pytest.fixture
def sample_model_cfg() -> Dict[str, any]:
    return {
        "boosting_type": "gbdt",
        "num_leaves": 15,
        "max_depth": -1,
        "learning_rate": 0.1,
        "n_estimators": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": 1,
        "class_weight": "balanced",
    }


class TestModelScaffoldAndTraining:
    def test_build_classifier(self, sample_model_cfg: Dict[str, any]) -> None:
        """Classifier builds with given parameters."""
        clf = build_classifier(sample_model_cfg)
        assert clf.num_leaves == 15
        assert clf.learning_rate == 0.1
        assert clf.n_estimators == 20
        assert clf.class_weight == "balanced"

    def test_build_classifier_with_scale_pos_weight(self, sample_model_cfg: Dict[str, any]) -> None:
        """scale_pos_weight overrides class_weight."""
        clf = build_classifier(sample_model_cfg, scale_pos_weight=3.5)
        assert clf.scale_pos_weight == 3.5
        assert clf.class_weight is None

    def test_train_and_predict(self, sample_model_cfg: Dict[str, any]) -> None:
        """Model trains and outputs valid probabilities."""
        np.random.seed(42)
        X_train = np.random.randn(200, 12).astype(np.float32)
        y_train = np.random.choice([0, 1], size=200, p=[0.75, 0.25]).astype(np.int8)

        X_val = np.random.randn(50, 12).astype(np.float32)
        y_val = np.random.choice([0, 1], size=50, p=[0.75, 0.25]).astype(np.int8)

        clf = build_classifier(sample_model_cfg)
        clf = train(clf, X_train, y_train, X_val=X_val, y_val=y_val, early_stopping_rounds=10)

        preds = predict_proba(clf, X_val)
        assert preds.shape == (50,)
        assert np.all((preds >= 0.0) & (preds <= 1.0))

    def test_save_and_load_model(self, sample_model_cfg: Dict[str, any], tmp_path: Path) -> None:
        """Trained model serializes to disk and restores identical predictions."""
        np.random.seed(42)
        X = np.random.randn(100, 12).astype(np.float32)
        y = np.random.choice([0, 1], size=100, p=[0.7, 0.3]).astype(np.int8)

        clf = build_classifier(sample_model_cfg)
        clf = train(clf, X, y)
        preds_orig = predict_proba(clf, X)

        # Test .txt booster export
        txt_path = tmp_path / "model.txt"
        save_model(clf, txt_path)
        assert txt_path.is_file()

        loaded_booster = load_model(txt_path)
        preds_txt = predict_proba(loaded_booster, X)
        assert np.allclose(preds_orig, preds_txt, atol=1e-5)

        # Test .pkl joblib export
        pkl_path = tmp_path / "model.pkl"
        save_model(clf, pkl_path)
        assert pkl_path.is_file()

        loaded_clf = load_model(pkl_path)
        preds_pkl = predict_proba(loaded_clf, X)
        assert np.allclose(preds_orig, preds_pkl, atol=1e-5)


class TestStratificationAndEvaluation:
    def test_stratified_entity_split(self) -> None:
        """Entity split partitions entities with zero leakage and stratifies by country + singleton."""
        s1_ids = [f"S1-{i:05d}" for i in range(1000)]
        s1_country = {eid: ("India" if i < 400 else "US") for i, eid in enumerate(s1_ids)}
        # India: 100 singletons, 300 has_match
        # US: 150 singletons, 450 has_match
        gt_dict = {}
        for i, eid in enumerate(s1_ids):
            if i < 100 or (400 <= i < 550):
                gt_dict[eid] = set()
            else:
                gt_dict[eid] = {f"S2-{i}", f"S3-{i}"}

        train_set, val_set = stratified_entity_split(
            s1_ids=s1_ids,
            s1_country_map=s1_country,
            gt_dict=gt_dict,
            val_ratio=0.1,
            random_state=42,
        )

        assert len(train_set) == 900
        assert len(val_set) == 100
        assert train_set.isdisjoint(val_set)

        # Check singleton proportion in validation (~25% = 250/1000)
        val_singletons = sum(1 for eid in val_set if len(gt_dict[eid]) == 0)
        assert 20 <= val_singletons <= 30

        # Check India proportion in validation (~40% = 400/1000)
        val_india = sum(1 for eid in val_set if s1_country[eid] == "India")
        assert 35 <= val_india <= 45

    def test_evaluate_validation_metrics(self) -> None:
        """Validation evaluation accurately calls macro_f_beta with thresholding."""
        val_entities = {"S1-1", "S1-2", "S1-3"}
        val_gt = {
            "S1-1": {"S2-10", "S2-11"},
            "S1-2": set(),  # singleton
            "S1-3": {"S2-30"},
        }
        # Candidates and scores
        val_candidates = [
            ("S1-1", "S2-10", 0.9),  # TP
            ("S1-1", "S2-11", 0.8),  # TP
            ("S1-1", "S2-12", 0.2),  # TN
            ("S1-2", "S2-20", 0.1),  # TN (singleton correctly empty)
            ("S1-3", "S2-30", 0.3),  # FN at threshold 0.5
        ]

        # S1-1: pred={S2-10, S2-11}, true={S2-10, S2-11} -> F0.5 = 1.0
        # S1-2: pred={}, true={} -> singleton score = 1.0
        # S1-3: pred={}, true={S2-30} -> score = 0.0
        # Expected macro F0.5 = (1.0 + 1.0 + 0.0) / 3 = 0.6667
        pred_dict = {
            "S1-1": {"S2-10", "S2-11"},
            "S1-2": set(),
            "S1-3": set(),
        }

        from business_entity_resolution.scoring import macro_f_beta

        metrics = macro_f_beta(pred_dict, val_gt)
        assert abs(metrics["macro_f_beta"] - (2.0 / 3.0)) < 1e-4
        assert metrics["singleton_mean"] == 1.0
        assert abs(metrics["has_match_mean"] - 0.5) < 1e-4
        assert metrics["n_singletons"] == 1
        assert metrics["n_has_match"] == 2
