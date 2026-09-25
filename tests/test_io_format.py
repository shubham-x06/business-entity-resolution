"""
test_io_format.py  –  Validate TSV loaders and writers against spec.

Covers:
  - load_source: column validation, prefix validation, missing-file warning
  - load_ground_truth: column validation, prefix validation, missing-file warning
  - load_source on the real train_source1.tsv (2,206,821 rows)
  - load_ground_truth on the real train_ground_truth.tsv
  - write_matching_results + write_candidate_pairs byte-exact round-trip
  - Writer edge cases: singletons (empty match list), multiple matches
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

# ── Repo root & import path ────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "code" / "business_entity_resolution" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.io_utils import (  # noqa: E402
    load_ground_truth,
    load_source,
    load_tsv,
    write_candidate_pairs,
    write_matching_results,
)

# ── Paths to real data ─────────────────────────────────────────────
TRAIN_DIR = REPO_ROOT / "dataset" / "train"
SOURCE1_PATH = TRAIN_DIR / "train_source1.tsv"
SOURCE2_PATH = TRAIN_DIR / "train_source2.tsv"
SOURCE3_PATH = TRAIN_DIR / "train_source3.tsv"
GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"


# ===================================================================
# 1.  load_source – schema & prefix validation
# ===================================================================

class TestLoadSource:
    """Tests for load_source()."""

    def test_loads_real_source1(self) -> None:
        """load_source must parse the full train_source1.tsv (2,206,821 rows)."""
        if not SOURCE1_PATH.is_file():
            pytest.skip("train_source1.tsv not present")
        df = load_source(SOURCE1_PATH, expected_source=1)
        assert len(df) == 2_206_821
        assert list(df.columns) == [
            "entity_id", "business_name", "business_address", "country",
        ]
        # All entity_ids start with S1-
        assert df["entity_id"].str.startswith("S1-").all()

    def test_validates_columns(self, tmp_path: Path) -> None:
        """Missing required columns should raise ValueError."""
        bad = tmp_path / "bad.tsv"
        bad.write_text("entity_id\tname\n" "S1-001\tFoo\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing required columns"):
            load_source(bad)

    def test_validates_prefix(self, tmp_path: Path) -> None:
        """Wrong entity_id prefix should raise ValueError."""
        f = tmp_path / "src.tsv"
        f.write_text(
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S2-001\tFoo\t123 Main\tUS\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="do not start with"):
            load_source(f, expected_source=1)

    def test_prefix_validation_passes_correct_source(self, tmp_path: Path) -> None:
        """Correct prefix should pass without error."""
        f = tmp_path / "src.tsv"
        f.write_text(
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S2-001\tFoo\t123 Main\tUS\n"
            "S2-002\tBar\t456 Oak\tUS\n",
            encoding="utf-8",
        )
        df = load_source(f, expected_source=2)
        assert len(df) == 2

    def test_missing_file_warns(self, tmp_path: Path) -> None:
        """Missing file should warn and return empty DataFrame."""
        missing = tmp_path / "nope.tsv"
        with pytest.warns(UserWarning, match="not found"):
            df = load_source(missing)
        assert len(df) == 0
        assert list(df.columns) == [
            "entity_id", "business_name", "business_address", "country",
        ]

    def test_no_na_coercion(self, tmp_path: Path) -> None:
        """Values like 'NA', 'null', '' must stay as literal strings."""
        f = tmp_path / "src.tsv"
        f.write_text(
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S1-001\tNA Corp\t\tUS\n"
            "S1-002\tnull LLC\tNull Street\tUS\n",
            encoding="utf-8",
        )
        df = load_source(f, expected_source=1)
        assert df.loc[0, "business_name"] == "NA Corp"
        assert df.loc[0, "business_address"] == ""       # empty, NOT NaN
        assert df.loc[1, "business_name"] == "null LLC"


# ===================================================================
# 2.  load_ground_truth
# ===================================================================

class TestLoadGroundTruth:
    """Tests for load_ground_truth()."""

    def test_loads_real_ground_truth(self) -> None:
        """load_ground_truth must parse the full file (2,206,821 rows)."""
        if not GT_PATH.is_file():
            pytest.skip("train_ground_truth.tsv not present")
        df = load_ground_truth(GT_PATH)
        assert len(df) == 2_206_821
        assert list(df.columns) == [
            "source1_entity_id", "matched_entity_ids",
        ]
        assert df["source1_entity_id"].str.startswith("S1-").all()
        # matched_entity_ids should be str (possibly empty), never NaN
        assert df["matched_entity_ids"].dtype == object  # str dtype
        assert not df["matched_entity_ids"].isna().any()

    def test_validates_columns(self, tmp_path: Path) -> None:
        """Missing columns should raise ValueError."""
        bad = tmp_path / "bad_gt.tsv"
        bad.write_text("id\tmatches\n" "S1-001\tS2-002\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing required columns"):
            load_ground_truth(bad)

    def test_validates_s1_prefix(self, tmp_path: Path) -> None:
        """Non-S1 prefix in source1_entity_id should raise."""
        f = tmp_path / "gt.tsv"
        f.write_text(
            "source1_entity_id\tmatched_entity_ids\n"
            "S2-001\tS3-002\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="lack the 'S1-' prefix"):
            load_ground_truth(f)

    def test_missing_file_warns(self, tmp_path: Path) -> None:
        """Missing file should warn and return empty DataFrame."""
        with pytest.warns(UserWarning, match="not found"):
            df = load_ground_truth(tmp_path / "gone.tsv")
        assert len(df) == 0


# ===================================================================
# 3.  Writer round-trip tests
# ===================================================================

class TestWriteMatchingResults:
    """Tests for write_matching_results() byte-exact format."""

    def test_roundtrip_byte_exact(self, tmp_path: Path) -> None:
        """write → read must reproduce the exact format from the README."""
        matches = {
            "S1-00001": ["S2-00047", "S2-00193", "S3-00812"],
            "S1-00002": ["S3-00004"],
            "S1-00003": [],  # singleton
        }
        out = tmp_path / "matching_results.tsv"
        write_matching_results(matches, out)

        expected = (
            "source1_entity_id\tmatched_entity_ids\n"
            "S1-00001\tS2-00047,S2-00193,S3-00812\n"
            "S1-00002\tS3-00004\n"
            "S1-00003\t\n"
        )
        actual = out.read_text(encoding="utf-8")
        assert actual == expected, (
            f"Byte mismatch.\nExpected:\n{expected!r}\nActual:\n{actual!r}"
        )

    def test_roundtrip_load_back(self, tmp_path: Path) -> None:
        """The written file must be loadable with load_tsv and retain values."""
        matches = {
            "S1-001": ["S2-010", "S3-020"],
            "S1-002": [],
        }
        out = tmp_path / "mr.tsv"
        write_matching_results(matches, out)

        df = load_tsv(out)
        assert list(df.columns) == [
            "source1_entity_id", "matched_entity_ids",
        ]
        row0 = df[df["source1_entity_id"] == "S1-001"].iloc[0]
        assert row0["matched_entity_ids"] == "S2-010,S3-020"

        row1 = df[df["source1_entity_id"] == "S1-002"].iloc[0]
        assert row1["matched_entity_ids"] == ""  # empty, not NaN

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Writer should create intermediate directories."""
        deep = tmp_path / "a" / "b" / "c" / "out.tsv"
        write_matching_results({"S1-001": []}, deep)
        assert deep.is_file()


class TestWriteCandidatePairs:
    """Tests for write_candidate_pairs() byte-exact format."""

    def test_roundtrip_byte_exact(self, tmp_path: Path) -> None:
        """write → read must reproduce the exact README format."""
        candidates = {
            "S1-00001": ["S2-00047", "S2-00193", "S3-00812", "S3-00999"],
            "S1-00002": ["S3-00004"],
            "S1-00003": [],
        }
        out = tmp_path / "candidate_pairs.tsv"
        write_candidate_pairs(candidates, out)

        expected = (
            "source1_entity_id\tcandidate_entity_ids\n"
            "S1-00001\tS2-00047,S2-00193,S3-00812,S3-00999\n"
            "S1-00002\tS3-00004\n"
            "S1-00003\t\n"
        )
        actual = out.read_text(encoding="utf-8")
        assert actual == expected

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Writer should create intermediate directories."""
        deep = tmp_path / "x" / "y" / "z" / "cp.tsv"
        write_candidate_pairs({"S1-001": ["S2-002"]}, deep)
        assert deep.is_file()


# ===================================================================
# 4.  load_tsv low-level
# ===================================================================

class TestLoadTsv:
    """Tests for load_tsv()."""

    def test_missing_file_warns(self, tmp_path: Path) -> None:
        """Missing file should warn and return empty DataFrame."""
        with pytest.warns(UserWarning, match="not found"):
            df = load_tsv(tmp_path / "missing.tsv")
        assert df.empty

    def test_reads_tab_separated(self, tmp_path: Path) -> None:
        """Tab-separated data should be parsed into correct columns."""
        f = tmp_path / "data.tsv"
        f.write_text("a\tb\tc\n1\t2\t3\n", encoding="utf-8")
        df = load_tsv(f)
        assert list(df.columns) == ["a", "b", "c"]
        assert df.iloc[0]["a"] == "1"
