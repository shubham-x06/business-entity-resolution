"""
test_features.py  –  Unit and integration tests for pairwise feature engineering.

Verifies:
1. Scalar feature helpers (compute_name_features, compute_address_features, compute_country_features).
2. Explicit sentinel handling for empty-address pairs (-1.0 sentinel, explicit empty flags).
3. Vectorized chunk computation (no NaNs, correct ranges in [0, 1], float32/int16 types).
4. Country filter functionality (restricting to India or US partition).
5. Integration test on a real slice of candidate_pairs.tsv (first ~10,000 rows).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "code" / "business_entity_resolution" / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.features import (
    FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    SENTINEL_EMPTY_ADDR,
    build_feature_matrix,
    compute_address_features,
    compute_country_features,
    compute_name_features,
    compute_pairwise_features_chunk,
    extract_features_streaming,
    fit_name_tfidf_vectorizer,
    load_and_normalize_entity_cache,
)


# ── 1. Scalar Feature Tests ──────────────────────────────────────────

class TestScalarFeatures:
    """Verify scalar helper functions."""

    def test_name_features_identical(self) -> None:
        feats = compute_name_features("acme corp", "acme corp")
        assert feats["name_jaro_winkler"] == pytest.approx(1.0, abs=1e-5)
        assert feats["name_levenshtein_ratio"] == pytest.approx(1.0, abs=1e-5)
        assert feats["name_token_jaccard"] == pytest.approx(1.0, abs=1e-5)
        assert feats["name_len_diff"] == pytest.approx(0.0, abs=1e-5)

    def test_name_features_completely_different(self) -> None:
        feats = compute_name_features("apple technology", "zebra wildlife sanctuary")
        assert 0.0 <= feats["name_jaro_winkler"] <= 1.0
        assert 0.0 <= feats["name_levenshtein_ratio"] <= 1.0
        assert feats["name_token_jaccard"] == 0.0
        assert 0.0 <= feats["name_len_diff"] <= 1.0

    def test_name_features_with_tfidf(self) -> None:
        corpus = ["apple inc", "google llc", "microsoft corp", "tata consultancy services"]
        vec = fit_name_tfidf_vectorizer(corpus, max_features=100)
        feats = compute_name_features("apple inc", "apple corp", vectorizer=vec)
        assert 0.0 < feats["name_tfidf_cosine"] <= 1.0

    def test_address_features_well_formed(self) -> None:
        feats = compute_address_features("123 main street new york", "123 main st new york")
        assert 0.0 <= feats["addr_token_overlap"] <= 1.0
        assert 0.0 <= feats["addr_edit_sim"] <= 1.0
        assert 0.0 <= feats["addr_len_diff"] <= 1.0
        assert feats["s1_empty_addr"] == 0.0
        assert feats["cand_empty_addr"] == 0.0

    def test_address_features_empty_s1(self) -> None:
        feats = compute_address_features("", "123 main street")
        assert feats["addr_token_overlap"] == SENTINEL_EMPTY_ADDR
        assert feats["addr_edit_sim"] == SENTINEL_EMPTY_ADDR
        assert feats["addr_len_diff"] == SENTINEL_EMPTY_ADDR
        assert feats["s1_empty_addr"] == 1.0
        assert feats["cand_empty_addr"] == 0.0

    def test_address_features_empty_candidate(self) -> None:
        feats = compute_address_features("123 main street", "")
        assert feats["addr_token_overlap"] == SENTINEL_EMPTY_ADDR
        assert feats["addr_edit_sim"] == SENTINEL_EMPTY_ADDR
        assert feats["addr_len_diff"] == SENTINEL_EMPTY_ADDR
        assert feats["s1_empty_addr"] == 0.0
        assert feats["cand_empty_addr"] == 1.0

    def test_address_features_both_empty(self) -> None:
        feats = compute_address_features("", "   ")
        assert feats["addr_token_overlap"] == SENTINEL_EMPTY_ADDR
        assert feats["addr_edit_sim"] == SENTINEL_EMPTY_ADDR
        assert feats["addr_len_diff"] == SENTINEL_EMPTY_ADDR
        assert feats["s1_empty_addr"] == 1.0
        assert feats["cand_empty_addr"] == 1.0

    def test_country_features(self) -> None:
        assert compute_country_features("India", "India")["country_match"] == 1.0
        assert compute_country_features("india", "INDIA")["country_match"] == 1.0
        assert compute_country_features("US", "India")["country_match"] == 0.0
        assert compute_country_features("", "US")["country_match"] == 0.0


# ── 2. Vectorized Chunk Computation Tests ────────────────────────────

class TestVectorizedChunkFeatures:
    """Verify batch/chunk vectorized feature calculations."""

    @pytest.fixture
    def sample_chunk_inputs(self) -> Dict[str, Any]:
        s1_ids = ["S1-1", "S1-2", "S1-3", "S1-4"]
        cand_ids = ["S2-10", "S2-20", "S3-30", "S3-40"]
        ranks = np.array([0, 1, 0, 2], dtype=np.int16)
        s1_names = ["tata consultancy services", "apple inc", "general electric co", "shree ram enterprise"]
        cand_names = ["tata consultancy services ltd", "apple corporation", "general motors co", "shree ram textiles"]
        s1_addrs = ["mumbai maharashtra 400001", "1 infinite loop cupertino ca", "", "delhi 110001"]
        cand_addrs = ["mumbai maharashtra", "", "300 renaissance center detroit", "delhi near metro 110001"]
        s1_countries = ["India", "US", "US", "India"]
        cand_countries = ["India", "US", "US", "India"]
        return {
            "s1_ids": s1_ids,
            "cand_ids": cand_ids,
            "ranks": ranks,
            "s1_names": s1_names,
            "cand_names": cand_names,
            "s1_addrs": s1_addrs,
            "cand_addrs": cand_addrs,
            "s1_countries": s1_countries,
            "cand_countries": cand_countries,
        }

    def test_chunk_shape_and_columns(self, sample_chunk_inputs: Dict[str, Any]) -> None:
        vec = fit_name_tfidf_vectorizer(sample_chunk_inputs["s1_names"] + sample_chunk_inputs["cand_names"])
        df = compute_pairwise_features_chunk(**sample_chunk_inputs, vectorizer=vec)

        assert len(df) == 4
        assert list(df.columns) == OUTPUT_COLUMNS
        assert df["source1_entity_id"].tolist() == ["S1-1", "S1-2", "S1-3", "S1-4"]

    def test_no_nans_and_valid_ranges(self, sample_chunk_inputs: Dict[str, Any]) -> None:
        vec = fit_name_tfidf_vectorizer(sample_chunk_inputs["s1_names"])
        df = compute_pairwise_features_chunk(**sample_chunk_inputs, vectorizer=vec)

        # Absolutely NO NaNs anywhere in the output
        assert df.isna().sum().sum() == 0, f"Found unexpected NaNs: {df.isna().sum()}"

        # Similarities must be in [0.0, 1.0]
        for col in ["name_jaro_winkler", "name_levenshtein_ratio", "name_token_jaccard", "name_tfidf_cosine", "name_len_diff"]:
            assert (df[col] >= 0.0).all(), f"{col} has negative values"
            assert (df[col] <= 1.0).all(), f"{col} has values > 1.0"

        # Country match must be 0.0 or 1.0
        assert set(df["country_match"].unique()).issubset({0.0, 1.0})

        # Empty address flags must be 0.0 or 1.0
        assert set(df["s1_empty_addr"].unique()).issubset({0.0, 1.0})
        assert set(df["cand_empty_addr"].unique()).issubset({0.0, 1.0})

    def test_empty_address_sentinel_in_chunk(self, sample_chunk_inputs: Dict[str, Any]) -> None:
        df = compute_pairwise_features_chunk(**sample_chunk_inputs)

        # Row 0: both addresses present -> features >= 0.0, flags == 0.0
        assert df.loc[0, "s1_empty_addr"] == 0.0
        assert df.loc[0, "cand_empty_addr"] == 0.0
        assert 0.0 <= df.loc[0, "addr_token_overlap"] <= 1.0
        assert 0.0 <= df.loc[0, "addr_edit_sim"] <= 1.0
        assert 0.0 <= df.loc[0, "addr_len_diff"] <= 1.0

        # Row 1: cand address empty -> sentinel -1.0, cand_empty_addr == 1.0
        assert df.loc[1, "s1_empty_addr"] == 0.0
        assert df.loc[1, "cand_empty_addr"] == 1.0
        assert df.loc[1, "addr_token_overlap"] == SENTINEL_EMPTY_ADDR
        assert df.loc[1, "addr_edit_sim"] == SENTINEL_EMPTY_ADDR
        assert df.loc[1, "addr_len_diff"] == SENTINEL_EMPTY_ADDR

        # Row 2: S1 address empty -> sentinel -1.0, s1_empty_addr == 1.0
        assert df.loc[2, "s1_empty_addr"] == 1.0
        assert df.loc[2, "cand_empty_addr"] == 0.0
        assert df.loc[2, "addr_token_overlap"] == SENTINEL_EMPTY_ADDR
        assert df.loc[2, "addr_edit_sim"] == SENTINEL_EMPTY_ADDR
        assert df.loc[2, "addr_len_diff"] == SENTINEL_EMPTY_ADDR


# ── 3. Compatibility Matrix API Tests ────────────────────────────────

class TestBuildFeatureMatrix:
    def test_build_feature_matrix_basic(self) -> None:
        s1_df = pd.DataFrame({
            "entity_id": ["S1-1", "S1-2"],
            "business_name": ["Acme Corp", "Beta LLC"],
            "business_address": ["123 Main St", ""],
            "country": ["US", "US"],
        })
        cand_df = pd.DataFrame({
            "entity_id": ["S2-1", "S3-2"],
            "business_name": ["Acme Corporation", "Beta Limited"],
            "business_address": ["123 Main Street", "456 Oak Rd"],
            "country": ["US", "US"],
        })
        pairs = [("S1-1", "S2-1"), ("S1-2", "S3-2")]

        pair_df, X = build_feature_matrix(pairs, s1_df, cand_df)
        assert len(pair_df) == 2
        assert X.shape == (2, len(FEATURE_COLUMNS))
        assert not np.isnan(X).any()
        assert X.dtype == np.float32


# ── 4. Country Filter Sanity Check ───────────────────────────────────

class TestCountryFilterSanity:
    """Verify that country filter restricts feature extraction to matching S1 entities."""

    def test_country_filter_india(self, tmp_path: Path) -> None:
        # Create mock candidate pairs file with both India and US S1 entities
        mock_cand_file = tmp_path / "mock_candidate_pairs.tsv"
        with open(mock_cand_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            f.write("S1-IN-1\tS2-IN-10,S3-IN-20\n")
            f.write("S1-US-1\tS2-US-10,S3-US-20\n")
            f.write("S1-IN-2\tS2-IN-30\n")

        # Mock entity cache
        entity_cache = {
            "S1-IN-1": ("tata motors", "mumbai", "India"),
            "S1-IN-2": ("reliance retail", "delhi", "India"),
            "S1-US-1": ("apple corp", "cupertino", "US"),
            "S2-IN-10": ("tata motors ltd", "mumbai maharashtra", "India"),
            "S3-IN-20": ("tata power", "mumbai", "India"),
            "S2-IN-30": ("reliance industries", "delhi", "India"),
            "S2-US-10": ("apple inc", "cupertino ca", "US"),
            "S3-US-20": ("alphabet google", "mountain view", "US"),
        }

        out_file = tmp_path / "features_india.parquet"
        stats = extract_features_streaming(
            candidate_pairs_path=mock_cand_file,
            entity_cache=entity_cache,
            output_path=out_file,
            country_filter="India",
            chunk_size=10,
        )

        assert stats["total_s1_entities"] == 2  # Only S1-IN-1 and S1-IN-2
        assert stats["total_pairs"] == 3  # (S1-IN-1: 2 cands) + (S1-IN-2: 1 cand)

        df_out = pd.read_parquet(out_file)
        assert len(df_out) == 3
        # Confirm ALL processed entities belong to India
        assert set(df_out["source1_entity_id"]).issubset({"S1-IN-1", "S1-IN-2"})
        assert (df_out["country_match"] == 1.0).all()


# ── 5. Integration Test on Real Slice (First ~10,000 Rows) ────────────

class TestRealDataSlice:
    """Run feature extraction on a small real slice of candidate_pairs.tsv."""

    @pytest.mark.skipif(
        not (REPO_ROOT / "output" / "candidate_pairs.tsv").is_file(),
        reason="candidate_pairs.tsv not present",
    )
    def test_first_10000_pairs_real_slice(self, tmp_path: Path) -> None:
        cand_path = REPO_ROOT / "output" / "candidate_pairs.tsv"
        out_parquet = tmp_path / "slice_features.parquet"

        # Load entity cache for India (first partition)
        entity_cache, vec = load_and_normalize_entity_cache(
            root=REPO_ROOT,
            country_filter="India",
        )

        stats = extract_features_streaming(
            candidate_pairs_path=cand_path,
            entity_cache=entity_cache,
            output_path=out_parquet,
            country_filter="India",
            chunk_size=5000,
            max_pairs=10000,
        )

        assert stats["total_pairs"] >= 10000
        assert stats["pairs_per_sec"] > 10000  # High throughput verified

        df = pd.read_parquet(out_parquet)
        assert len(df) == stats["total_pairs"]
        assert list(df.columns) == OUTPUT_COLUMNS

        # Verify no NaNs in any feature column
        assert df.isna().sum().sum() == 0, f"Found NaNs: {df.isna().sum()}"

        # Verify all similarities are in [0, 1]
        for col in ["name_jaro_winkler", "name_levenshtein_ratio", "name_token_jaccard", "name_tfidf_cosine"]:
            assert (df[col] >= 0.0).all(), f"Negative values in {col}"
            assert (df[col] <= 1.0).all(), f"Values > 1.0 in {col}"

        # Verify sentinel handling for empty addresses
        empty_mask = (df["s1_empty_addr"] == 1.0) | (df["cand_empty_addr"] == 1.0)
        if empty_mask.any():
            assert (df.loc[empty_mask, "addr_token_overlap"] == SENTINEL_EMPTY_ADDR).all()
            assert (df.loc[empty_mask, "addr_edit_sim"] == SENTINEL_EMPTY_ADDR).all()
            assert (df.loc[empty_mask, "addr_len_diff"] == SENTINEL_EMPTY_ADDR).all()

        non_empty_mask = ~empty_mask
        if non_empty_mask.any():
            assert (df.loc[non_empty_mask, "addr_token_overlap"] >= 0.0).all()
            assert (df.loc[non_empty_mask, "addr_edit_sim"] >= 0.0).all()
            assert (df.loc[non_empty_mask, "addr_len_diff"] >= 0.0).all()
