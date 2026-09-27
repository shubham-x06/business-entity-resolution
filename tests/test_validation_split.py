"""
tests/test_validation_split.py  –  Unit tests for stratified validation split script.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_validation_split import (
    build_stratified_validation_split,
    load_entity_strata,
    save_validation_entities,
)


@pytest.fixture
def sample_strata_df() -> pd.DataFrame:
    """Create a synthetic DataFrame covering all 4 strata."""
    records = []
    # US-has-match: 100
    for i in range(100):
        records.append({
            "entity_id": f"S1-US-M-{i:03d}",
            "country": "US",
            "matched_entity_ids": f"S2-{i},S3-{i}",
            "is_singleton": False,
            "stratum": "US-has-match",
        })
    # US-singleton: 20
    for i in range(20):
        records.append({
            "entity_id": f"S1-US-S-{i:03d}",
            "country": "US",
            "matched_entity_ids": "",
            "is_singleton": True,
            "stratum": "US-singleton",
        })
    # India-has-match: 80
    for i in range(80):
        records.append({
            "entity_id": f"S1-IN-M-{i:03d}",
            "country": "India",
            "matched_entity_ids": f"S2-{i}",
            "is_singleton": False,
            "stratum": "India-has-match",
        })
    # India-singleton: 10
    for i in range(10):
        records.append({
            "entity_id": f"S1-IN-S-{i:03d}",
            "country": "India",
            "matched_entity_ids": "",
            "is_singleton": True,
            "stratum": "India-singleton",
        })
    return pd.DataFrame(records)


def test_build_stratified_validation_split_proportions(sample_strata_df: pd.DataFrame):
    """Verify that sampling samples exactly the specified fraction from each stratum."""
    val_ratio = 0.10
    train_df, val_df, summary = build_stratified_validation_split(
        sample_strata_df, val_ratio=val_ratio, seed=42
    )

    # 10% of 100 is 10, 10% of 20 is 2, 10% of 80 is 8, 10% of 10 is 1
    assert summary["strata"]["US-has-match"]["val"] == 10
    assert summary["strata"]["US-singleton"]["val"] == 2
    assert summary["strata"]["India-has-match"]["val"] == 8
    assert summary["strata"]["India-singleton"]["val"] == 1

    assert len(val_df) == 21
    assert len(train_df) == 189
    assert len(val_df) + len(train_df) == 210


def test_no_leakage_between_train_and_val(sample_strata_df: pd.DataFrame):
    """Ensure training and validation sets have zero overlapping entity IDs."""
    train_df, val_df, _ = build_stratified_validation_split(
        sample_strata_df, val_ratio=0.15, seed=123
    )
    train_ids = set(train_df["entity_id"])
    val_ids = set(val_df["entity_id"])

    assert len(train_ids & val_ids) == 0
    assert train_ids | val_ids == set(sample_strata_df["entity_id"])


def test_split_determinism(sample_strata_df: pd.DataFrame):
    """Ensure identical seed produces identical entity IDs in validation split."""
    _, val_df_1, _ = build_stratified_validation_split(sample_strata_df, val_ratio=0.10, seed=99)
    _, val_df_2, _ = build_stratified_validation_split(sample_strata_df, val_ratio=0.10, seed=99)

    assert sorted(val_df_1["entity_id"]) == sorted(val_df_2["entity_id"])


def test_invalid_val_ratio(sample_strata_df: pd.DataFrame):
    """Ensure invalid ratios raise ValueError."""
    with pytest.raises(ValueError):
        build_stratified_validation_split(sample_strata_df, val_ratio=0.0)
    with pytest.raises(ValueError):
        build_stratified_validation_split(sample_strata_df, val_ratio=1.0)
    with pytest.raises(ValueError):
        build_stratified_validation_split(sample_strata_df, val_ratio=-0.1)


def test_save_validation_entities_roundtrip(tmp_path: Path):
    """Ensure save_validation_entities produces newline-delimited, sorted entity IDs."""
    val_df = pd.DataFrame({"entity_id": ["S1-003", "S1-001", "S1-002"]})
    out_file = tmp_path / "val_ids.txt"
    save_validation_entities(val_df, out_file)

    assert out_file.is_file()
    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert lines == ["S1-001", "S1-002", "S1-003"]


def test_load_entity_strata(tmp_path: Path):
    """Verify loading and stratum assignment from synthetic TSV files."""
    s1_path = tmp_path / "s1.tsv"
    gt_path = tmp_path / "gt.tsv"

    s1_content = (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-1\tName 1\tAddr 1\tIndia\n"
        "S1-2\tName 2\tAddr 2\tIndia\n"
        "S1-3\tName 3\tAddr 3\tUS\n"
        "S1-4\tName 4\tAddr 4\tUS\n"
    )
    s1_path.write_text(s1_content, encoding="utf-8")

    gt_content = (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-100\n"
        "S1-2\t\n"
        "S1-3\tS3-300,S2-301\n"
        "S1-4\t \n"
    )
    gt_path.write_text(gt_content, encoding="utf-8")

    df = load_entity_strata(s1_path, gt_path)
    assert len(df) == 4

    strata_map = dict(zip(df["entity_id"], df["stratum"]))
    assert strata_map["S1-1"] == "India-has-match"
    assert strata_map["S1-2"] == "India-singleton"
    assert strata_map["S1-3"] == "US-has-match"
    assert strata_map["S1-4"] == "US-singleton"


def test_cli_help():
    """Verify CLI help flag works."""
    cmd = [sys.executable, "scripts/build_validation_split.py", "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "Build stratified validation split" in res.stdout
