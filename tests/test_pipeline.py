"""
test_pipeline.py  –  Unit tests for pipeline orchestration skeleton.

Validates:
- run_train_pipeline and run_infer_pipeline execute without raising
- Returned summary dict correctly reports which stages ran vs. were skipped
- Stages that depend on prior stages cascade-skip correctly
- Logging messages are emitted for each stage transition
- Config loading works with real config files
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "code" / "business_entity_resolution" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.io_utils import load_config
from business_entity_resolution.pipeline import (
    StageResult,
    _run_stage,
    run_infer_pipeline,
    run_train_pipeline,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures: synthetic dataset + configs
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Create a tiny synthetic dataset in the expected directory structure.

    Creates train/ and test/ directories with source1/2/3.tsv files
    and a ground_truth.tsv file.  Returns the root path.
    """
    # ── Create directory structure ──────────────────────────────────
    train_dir = tmp_path / "dataset" / "train"
    train_dir.mkdir(parents=True)
    test_dir = tmp_path / "dataset" / "test"
    test_dir.mkdir(parents=True)
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)

    # ── Source 1 (S1-* entity_ids) ──────────────────────────────────
    s1_data = pd.DataFrame({
        "entity_id": ["S1-001", "S1-002", "S1-003"],
        "business_name": ["Acme Corp", "Beta LLC", "Gamma Inc"],
        "business_address": ["123 Main St, NY", "456 Oak Ave, CA", "789 Pine Rd, TX"],
        "country": ["US", "US", "US"],
    })

    # ── Source 2 (S2-* entity_ids) ──────────────────────────────────
    s2_data = pd.DataFrame({
        "entity_id": ["S2-001", "S2-002", "S2-003"],
        "business_name": ["Acme Corporation", "Delta Corp", "Beta Limited"],
        "business_address": ["123 Main Street, NY", "111 Elm St, FL", "456 Oak Avenue, CA"],
        "country": ["US", "US", "US"],
    })

    # ── Source 3 (S3-* entity_ids) ──────────────────────────────────
    s3_data = pd.DataFrame({
        "entity_id": ["S3-001", "S3-002"],
        "business_name": ["Gamma Incorporated", "Epsilon Ltd"],
        "business_address": ["789 Pine Road, TX", "222 Birch Ln, WA"],
        "country": ["US", "US"],
    })

    # ── Ground truth ────────────────────────────────────────────────
    gt_data = pd.DataFrame({
        "source1_entity_id": ["S1-001", "S1-002", "S1-003"],
        "matched_entity_ids": ["S2-001", "S2-003", "S3-001"],
    })

    # ── Write all files ─────────────────────────────────────────────
    for name, df in [
        ("train_source1.tsv", s1_data),
        ("train_source2.tsv", s2_data),
        ("train_source3.tsv", s3_data),
        ("train_ground_truth.tsv", gt_data),
    ]:
        df.to_csv(train_dir / name, sep="\t", index=False)

    for name, df in [
        ("test_source1.tsv", s1_data),
        ("test_source2.tsv", s2_data),
        ("test_source3.tsv", s3_data),
    ]:
        df.to_csv(test_dir / name, sep="\t", index=False)

    # ── Copy real config files (use existing schemas) ───────────────
    cfg_dir = tmp_path / "code" / "business_entity_resolution" / "configs"
    cfg_dir.mkdir(parents=True)

    real_cfg_dir = REPO_ROOT / "code" / "business_entity_resolution" / "configs"
    for cfg_name in ("paths.yaml", "blocking.yaml", "model.yaml"):
        real_path = real_cfg_dir / cfg_name
        if real_path.is_file():
            import shutil
            shutil.copy2(real_path, cfg_dir / cfg_name)
        else:
            # Fallback minimal config
            (cfg_dir / cfg_name).write_text("{}\n")

    return tmp_path


@pytest.fixture
def train_config(synthetic_dataset):
    """Build a config dict suitable for run_train_pipeline."""
    root = synthetic_dataset
    cfg_dir = root / "code" / "business_entity_resolution" / "configs"

    paths_cfg = load_config(cfg_dir / "paths.yaml")
    blocking_cfg = load_config(cfg_dir / "blocking.yaml")
    model_cfg = load_config(cfg_dir / "model.yaml")

    return {
        "root": str(root),
        "paths": paths_cfg,
        "blocking": blocking_cfg,
        "model": model_cfg,
        "sample": None,
    }


@pytest.fixture
def infer_config(synthetic_dataset):
    """Build a config dict suitable for run_infer_pipeline."""
    root = synthetic_dataset
    cfg_dir = root / "code" / "business_entity_resolution" / "configs"

    paths_cfg = load_config(cfg_dir / "paths.yaml")
    blocking_cfg = load_config(cfg_dir / "blocking.yaml")
    model_cfg = load_config(cfg_dir / "model.yaml")

    return {
        "root": str(root),
        "paths": paths_cfg,
        "blocking": blocking_cfg,
        "model": model_cfg,
        "sample": None,
        "threshold": 0.5,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Test _run_stage wrapper
# ═══════════════════════════════════════════════════════════════════════════


class TestRunStage:
    """Tests for the stage execution wrapper."""

    def test_completed_stage(self):
        """Stage that succeeds returns status='completed'."""
        sr = _run_stage("test_ok", lambda: 42)
        assert sr.status == "completed"
        assert sr.result == 42
        assert sr.elapsed_seconds >= 0
        assert sr.error_msg is None

    def test_not_implemented_stage(self):
        """Stage raising NotImplementedError returns status='skipped'."""
        def stub():
            raise NotImplementedError("not done yet")

        sr = _run_stage("test_stub", stub)
        assert sr.status == "skipped"
        assert sr.result is None
        assert "NotImplementedError" in sr.error_msg

    def test_attribute_error_stage(self):
        """Stage raising AttributeError returns status='skipped'."""
        def bad_attr():
            raise AttributeError("module has no attribute 'foo'")

        sr = _run_stage("test_attr", bad_attr)
        assert sr.status == "skipped"
        assert "AttributeError" in sr.error_msg

    def test_other_exception_propagates(self):
        """Stage raising an unexpected exception should propagate."""
        def kaboom():
            raise ValueError("unexpected")

        with pytest.raises(ValueError, match="unexpected"):
            _run_stage("test_boom", kaboom)

    def test_to_dict_completed(self):
        """StageResult.to_dict for completed stage."""
        sr = StageResult("x", "completed", 1.234, result="ok")
        d = sr.to_dict()
        assert d["status"] == "completed"
        assert d["elapsed_seconds"] == 1.234
        assert "error" not in d

    def test_to_dict_skipped(self):
        """StageResult.to_dict for skipped stage includes error."""
        sr = StageResult("x", "skipped", 0.001, error_msg="not impl")
        d = sr.to_dict()
        assert d["status"] == "skipped"
        assert d["error"] == "not impl"


# ═══════════════════════════════════════════════════════════════════════════
# Test run_train_pipeline
# ═══════════════════════════════════════════════════════════════════════════


class TestRunTrainPipeline:
    """Tests for the training pipeline orchestration."""

    def test_does_not_raise(self, train_config):
        """Pipeline runs end-to-end without crashing, even though blocking/
        features/model stages are not yet implemented.
        """
        result = run_train_pipeline(train_config)
        assert isinstance(result, dict)

    def test_returns_correct_shape(self, train_config):
        """Result dict has the expected top-level keys."""
        result = run_train_pipeline(train_config)
        assert "stages" in result
        assert "metrics" in result
        assert "completed" in result
        assert "skipped" in result
        assert "elapsed_total_seconds" in result

    def test_load_data_and_normalize_complete(self, train_config):
        """Load data and normalize stages should complete (they are
        implemented).
        """
        result = run_train_pipeline(train_config)
        assert "load_data" in result["stages"]
        assert result["stages"]["load_data"]["status"] == "completed"
        assert "normalize" in result["stages"]
        assert result["stages"]["normalize"]["status"] == "completed"

    def test_blocking_stage_present(self, train_config):
        """Blocking stage should be attempted (and may complete or skip
        depending on blocking.py's implementation status).
        """
        result = run_train_pipeline(train_config)
        assert "blocking" in result["stages"]
        # It either completed or was skipped — it should not be missing
        assert result["stages"]["blocking"]["status"] in (
            "completed", "skipped",
        )

    def test_downstream_stages_cascade_skip(self, train_config):
        """If blocking is skipped, featurize/train_model/predict should also
        be skipped (cascade dependency).
        """
        result = run_train_pipeline(train_config)

        blocking_status = result["stages"]["blocking"]["status"]
        if blocking_status == "skipped":
            assert result["stages"]["featurize"]["status"] == "skipped"
            assert result["stages"]["train_model"]["status"] == "skipped"
            assert result["stages"]["predict"]["status"] == "skipped"
            assert result["stages"]["threshold_sweep"]["status"] == "skipped"

    def test_completed_and_skipped_lists_partition(self, train_config):
        """Every stage appears in exactly one of completed/skipped."""
        result = run_train_pipeline(train_config)
        all_stages = set(result["stages"].keys())
        completed_set = set(result["completed"])
        skipped_set = set(result["skipped"])

        assert completed_set | skipped_set == all_stages
        assert completed_set & skipped_set == set()

    def test_with_sample(self, train_config):
        """Pipeline respects sample parameter."""
        train_config["sample"] = 2
        result = run_train_pipeline(train_config)
        assert result["stages"]["load_data"]["status"] == "completed"

    def test_elapsed_time_positive(self, train_config):
        """Total elapsed time should be positive."""
        result = run_train_pipeline(train_config)
        assert result["elapsed_total_seconds"] > 0

    def test_logging_output(self, train_config, caplog):
        """Pipeline emits INFO-level log messages for each stage."""
        with caplog.at_level(logging.INFO):
            run_train_pipeline(train_config)

        log_text = caplog.text
        assert "STARTED" in log_text
        # At least load_data and normalize should complete
        assert "COMPLETED" in log_text


# ═══════════════════════════════════════════════════════════════════════════
# Test run_infer_pipeline
# ═══════════════════════════════════════════════════════════════════════════


class TestRunInferPipeline:
    """Tests for the inference pipeline orchestration."""

    def test_does_not_raise(self, infer_config):
        """Pipeline runs end-to-end without crashing."""
        result = run_infer_pipeline(infer_config)
        assert isinstance(result, dict)

    def test_returns_correct_shape(self, infer_config):
        """Result dict has the expected top-level keys."""
        result = run_infer_pipeline(infer_config)
        assert "stages" in result
        assert "metrics" in result
        assert "completed" in result
        assert "skipped" in result
        assert "elapsed_total_seconds" in result
        assert "threshold_used" in result

    def test_threshold_from_config(self, infer_config):
        """Threshold used should match what was passed in config."""
        infer_config["threshold"] = 0.65
        result = run_infer_pipeline(infer_config)
        assert result["threshold_used"] == 0.65

    def test_threshold_default_from_model_yaml(self, infer_config):
        """Without explicit threshold, should use model.yaml default."""
        del infer_config["threshold"]
        result = run_infer_pipeline(infer_config)
        # model.yaml has threshold.match = 0.5
        assert result["threshold_used"] == 0.5

    def test_load_data_and_normalize_complete(self, infer_config):
        """Load data and normalize should complete."""
        result = run_infer_pipeline(infer_config)
        assert result["stages"]["load_data"]["status"] == "completed"
        assert result["stages"]["normalize"]["status"] == "completed"

    def test_downstream_cascade_on_skip(self, infer_config):
        """If blocking skips, downstream stages should cascade-skip."""
        result = run_infer_pipeline(infer_config)

        if result["stages"]["blocking"]["status"] == "skipped":
            assert result["stages"]["featurize"]["status"] == "skipped"
            assert result["stages"]["predict"]["status"] == "skipped"
            assert result["stages"]["group_and_write"]["status"] == "skipped"


# ═══════════════════════════════════════════════════════════════════════════
# Test CLI entry-points
# ═══════════════════════════════════════════════════════════════════════════


class TestCLIEntryPoints:
    """Verify that CLI scripts can parse the new flags."""

    def test_run_train_help(self):
        """run_train.py --help should succeed."""
        import subprocess

        script = (
            REPO_ROOT / "code" / "business_entity_resolution"
            / "scripts" / "run_train.py"
        )
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "orchestrated pipeline" in result.stdout

    def test_run_infer_help(self):
        """run_infer.py --help should succeed and mention '--threshold'."""
        import subprocess

        script = (
            REPO_ROOT / "code" / "business_entity_resolution"
            / "scripts" / "run_infer.py"
        )
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "--threshold" in result.stdout
