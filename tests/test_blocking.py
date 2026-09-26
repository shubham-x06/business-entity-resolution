"""
test_blocking.py  –  Unit tests for MinHash LSH blocking and candidate generation.

Covers:
- MinHash construction and LSH index build/query
- Character n-gram shingling
- Address token inverted index
- Country-partitioned blocking
- Empty-address fallback (name-only path)
- Open-set country handling
- Config-driven thresholds
- Integration with normalize.py
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

from business_entity_resolution.blocking import (
    _build_minhash,
    _char_ngrams,
    _extract_significant_tokens,
    _normalize_source,
    build_address_token_index,
    build_minhash_lsh_index,
    generate_candidate_pairs,
    query_address_candidates,
    query_candidates,
)


# ── 1. Shingling ────────────────────────────────────────────────────

class TestCharNgrams:
    """Test character n-gram extraction."""

    def test_basic_trigrams(self) -> None:
        result = _char_ngrams("abcde", n=3)
        assert result == ["abc", "bcd", "cde"]

    def test_short_string(self) -> None:
        result = _char_ngrams("ab", n=3)
        assert result == ["ab"]

    def test_empty_string(self) -> None:
        assert _char_ngrams("", n=3) == []

    def test_exact_n_length(self) -> None:
        assert _char_ngrams("abc", n=3) == ["abc"]

    def test_devanagari_trigrams(self) -> None:
        """Non-Latin scripts produce valid n-grams without crashing."""
        text = "राम मार्केटिंग"
        result = _char_ngrams(text, n=3)
        assert len(result) > 0
        # All n-grams should be length 3
        assert all(len(ng) == 3 for ng in result)


# ── 2. MinHash construction ─────────────────────────────────────────

class TestMinHashBuild:
    """Test MinHash creation."""

    def test_identical_texts_high_similarity(self) -> None:
        m1 = _build_minhash("acme corp", num_perm=128, ngram_size=3)
        m2 = _build_minhash("acme corp", num_perm=128, ngram_size=3)
        assert m1.jaccard(m2) == 1.0

    def test_similar_texts_positive_similarity(self) -> None:
        m1 = _build_minhash("acme corporation", num_perm=128, ngram_size=3)
        m2 = _build_minhash("acme corp", num_perm=128, ngram_size=3)
        sim = m1.jaccard(m2)
        assert sim > 0.3, f"Expected > 0.3, got {sim}"

    def test_different_texts_low_similarity(self) -> None:
        m1 = _build_minhash("apple inc", num_perm=128, ngram_size=3)
        m2 = _build_minhash("microsoft corp", num_perm=128, ngram_size=3)
        sim = m1.jaccard(m2)
        assert sim < 0.3, f"Expected < 0.3, got {sim}"

    def test_empty_text(self) -> None:
        m = _build_minhash("", num_perm=128, ngram_size=3)
        assert m is not None


# ── 3. LSH index build and query ────────────────────────────────────

class TestLSHIndexing:
    """Test building and querying the LSH index."""

    @pytest.fixture
    def sample_ref_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "entity_id": ["S2-001", "S2-002", "S2-003", "S3-001"],
            "name_clean": ["acme corp", "acme corporation", "totally different", "acme company"],
        })

    def test_build_and_query_finds_similar(self, sample_ref_df: pd.DataFrame) -> None:
        lsh, minhashes = build_minhash_lsh_index(
            sample_ref_df,
            text_col="name_clean",
            id_col="entity_id",
            num_perm=128,
            threshold=0.3,
            ngram_size=3,
        )
        assert len(minhashes) == 4

        query_df = pd.DataFrame({
            "entity_id": ["S1-001"],
            "name_clean": ["acme corp"],
        })
        results = query_candidates(
            lsh, query_df,
            text_col="name_clean",
            id_col="entity_id",
            num_perm=128,
            ngram_size=3,
        )
        assert "S1-001" in results
        candidates = results["S1-001"]
        # Should find the similar "acme" entries
        assert "S2-001" in candidates  # exact match
        assert "S2-002" in candidates  # very similar

    def test_dissimilar_not_returned(self, sample_ref_df: pd.DataFrame) -> None:
        lsh, _ = build_minhash_lsh_index(
            sample_ref_df,
            text_col="name_clean",
            id_col="entity_id",
            num_perm=128,
            threshold=0.5,  # higher threshold
            ngram_size=3,
        )
        query_df = pd.DataFrame({
            "entity_id": ["S1-099"],
            "name_clean": ["xyz plumbing services"],
        })
        results = query_candidates(
            lsh, query_df,
            text_col="name_clean",
            id_col="entity_id",
            num_perm=128,
            ngram_size=3,
        )
        # Shouldn't match any of the "acme" entries
        assert len(results["S1-099"]) == 0


# ── 4. Address token blocking ──────────────────────────────────────

class TestAddressTokenBlocking:
    """Test address token extraction and inverted index."""

    def test_extract_significant_tokens(self) -> None:
        tokens = _extract_significant_tokens("123 main rd tahlequah ok", min_len=3)
        assert "main" in tokens
        assert "tahlequah" in tokens
        # "ok" is too short (len 2)
        assert "ok" not in tokens
        # "123" is all digits
        assert "123" not in tokens

    def test_extract_from_empty(self) -> None:
        assert _extract_significant_tokens("", min_len=3) == []
        assert _extract_significant_tokens("  ", min_len=3) == []

    def test_build_and_query_address_index(self) -> None:
        ref_df = pd.DataFrame({
            "entity_id": ["S2-001", "S2-002", "S3-001"],
            "address_clean": [
                "123 main rd tahlequah ok",
                "456 elm st charlotte nc",
                "789 main ave tahlequah ok",
            ],
        })
        idx = build_address_token_index(ref_df, addr_col="address_clean", id_col="entity_id")
        assert "tahlequah" in idx
        assert "S2-001" in idx["tahlequah"]
        assert "S3-001" in idx["tahlequah"]

        query_df = pd.DataFrame({
            "entity_id": ["S1-001"],
            "address_clean": ["17560 ellis rd tahlequah ok"],
        })
        results = query_address_candidates(idx, query_df, addr_col="address_clean", id_col="entity_id")
        # Should find S2-001 and S3-001 via "tahlequah"
        assert "S2-001" in results["S1-001"]
        assert "S3-001" in results["S1-001"]

    def test_high_freq_token_pruning(self) -> None:
        """Tokens above max_token_freq are pruned from the index."""
        records = pd.DataFrame({
            "entity_id": [f"S2-{i}" for i in range(100)],
            "address_clean": ["new delhi west delhi delhi"] * 100,
        })
        idx = build_address_token_index(
            records, addr_col="address_clean", id_col="entity_id",
            max_token_freq=50,
        )
        # "delhi" appears 100 times -> pruned
        assert "delhi" not in idx


# ── 5. Empty address (name-only fallback) ───────────────────────────

class TestEmptyAddressFallback:
    """Rows with empty business_address must not be dropped and should
    still be reachable via name-based blocking."""

    def test_empty_addr_row_in_lsh_candidates(self) -> None:
        ref_df = pd.DataFrame({
            "entity_id": ["S2-100", "S2-200"],
            "name_clean": ["ram marketing pvt ltd", "some other co"],
            "address_clean": ["", "456 elm st"],  # S2-100 has empty address
        })
        lsh, _ = build_minhash_lsh_index(
            ref_df, text_col="name_clean", id_col="entity_id",
            num_perm=128, threshold=0.3, ngram_size=3,
        )
        query_df = pd.DataFrame({
            "entity_id": ["S1-001"],
            "name_clean": ["ram marketing pvt ltd"],
        })
        results = query_candidates(lsh, query_df, text_col="name_clean",
                                   id_col="entity_id", num_perm=128, ngram_size=3)
        assert "S2-100" in results["S1-001"]

    def test_generate_candidates_with_empty_addr(self) -> None:
        """Full pipeline handles empty addresses without dropping rows."""
        s1 = pd.DataFrame({
            "entity_id": ["S1-001"],
            "business_name": ["Ram Marketing Pvt Ltd"],
            "business_address": ["New Delhi, West Delhi, Delhi"],
            "country": ["India"],
        })
        s2 = pd.DataFrame({
            "entity_id": ["S2-100", "S2-200"],
            "business_name": ["Ram Marketing Private Limited", "Other Co"],
            "business_address": ["", "Some Address, Delhi"],  # S2-100 has empty addr
            "country": ["India", "India"],
        })
        s3 = pd.DataFrame({
            "entity_id": ["S3-100"],
            "business_name": ["Ram Mkt Pvt Ltd"],
            "business_address": ["Delhi"],
            "country": ["India"],
        })
        cfg = {
            "minhash_lsh": {"num_perm": 128, "threshold": 0.3, "ngram_size": 3},
            "name_blocking": {"enabled": True, "prefix_len": 3},
            "country_blocking": {"enabled": True},
            "address_blocking": {"enabled": True, "min_token_len": 3,
                                "max_token_freq": 50000, "min_shared_tokens": 1},
        }
        result = generate_candidate_pairs(s1, s2, s3, cfg)
        assert "S1-001" in result
        # S2-100 should be found via name LSH despite empty address
        assert "S2-100" in result["S1-001"]


# ── 6. Country-partitioned blocking ─────────────────────────────────

class TestCountryPartitioning:
    """Verify records from different countries are never paired."""

    def test_cross_country_not_paired(self) -> None:
        s1 = pd.DataFrame({
            "entity_id": ["S1-001"],
            "business_name": ["Acme Corp"],
            "business_address": ["123 Main Rd"],
            "country": ["US"],
        })
        s2 = pd.DataFrame({
            "entity_id": ["S2-001"],
            "business_name": ["Acme Corp"],  # same name!
            "business_address": ["123 Main Rd"],
            "country": ["India"],  # different country
        })
        s3 = pd.DataFrame({
            "entity_id": ["S3-001"],
            "business_name": ["Acme Company"],
            "business_address": ["456 Elm St"],
            "country": ["US"],
        })
        cfg = {
            "minhash_lsh": {"num_perm": 128, "threshold": 0.3, "ngram_size": 3},
            "name_blocking": {"enabled": False},
            "country_blocking": {"enabled": True},
            "address_blocking": {"enabled": False},
        }
        result = generate_candidate_pairs(s1, s2, s3, cfg)
        # S2-001 is India, S1-001 is US — should NOT be paired
        assert "S2-001" not in result["S1-001"]
        # S3-001 is US — could be paired if similar enough
        # (may or may not be found depending on LSH; just verify S2-001 excluded)

    def test_same_country_paired(self) -> None:
        s1 = pd.DataFrame({
            "entity_id": ["S1-001"],
            "business_name": ["Acme Corp"],
            "business_address": ["123 Main Rd"],
            "country": ["US"],
        })
        s2 = pd.DataFrame({
            "entity_id": ["S2-001"],
            "business_name": ["Acme Corporation"],
            "business_address": ["123 Main Road"],
            "country": ["US"],
        })
        s3 = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
        cfg = {
            "minhash_lsh": {"num_perm": 128, "threshold": 0.3, "ngram_size": 3},
            "name_blocking": {"enabled": True, "prefix_len": 4},
            "country_blocking": {"enabled": True},
            "address_blocking": {"enabled": True, "min_token_len": 3,
                                "max_token_freq": 50000, "min_shared_tokens": 1},
        }
        result = generate_candidate_pairs(s1, s2, s3, cfg)
        assert "S2-001" in result["S1-001"]


# ── 7. Open-set country handling ────────────────────────────────────

class TestOpenSetCountryBlocking:
    """Unseen country strings (France, etc.) should work without error."""

    def test_france_country_blocking(self) -> None:
        s1 = pd.DataFrame({
            "entity_id": ["S1-FR1"],
            "business_name": ["Société Générale S.A."],
            "business_address": ["12 Rue de la Paix, 75002 Paris"],
            "country": ["France"],
        })
        s2 = pd.DataFrame({
            "entity_id": ["S2-FR1"],
            "business_name": ["Societe Generale SA"],
            "business_address": ["12 Rue de la Paix, Paris"],
            "country": ["France"],
        })
        s3 = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
        cfg = {
            "minhash_lsh": {"num_perm": 128, "threshold": 0.3, "ngram_size": 3},
            "name_blocking": {"enabled": True, "prefix_len": 4},
            "country_blocking": {"enabled": True},
            "address_blocking": {"enabled": True, "min_token_len": 3,
                                "max_token_freq": 50000, "min_shared_tokens": 1},
        }
        result = generate_candidate_pairs(s1, s2, s3, cfg)
        # Should not crash, and the French entities should be paired
        assert "S1-FR1" in result
        assert "S2-FR1" in result["S1-FR1"]

    def test_country_none_does_not_crash(self) -> None:
        s1 = pd.DataFrame({
            "entity_id": ["S1-001"],
            "business_name": ["Test Corp"],
            "business_address": ["123 Main Rd"],
            "country": ["ZZ_UNKNOWN"],
        })
        s2 = pd.DataFrame({
            "entity_id": ["S2-001"],
            "business_name": ["Test Corporation"],
            "business_address": ["123 Main Road"],
            "country": ["ZZ_UNKNOWN"],
        })
        s3 = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
        cfg = {
            "minhash_lsh": {"num_perm": 128, "threshold": 0.3, "ngram_size": 3},
            "name_blocking": {"enabled": True, "prefix_len": 4},
            "country_blocking": {"enabled": True},
            "address_blocking": {"enabled": False},
        }
        result = generate_candidate_pairs(s1, s2, s3, cfg)
        assert "S1-001" in result


# ── 8. All S1 entities get an entry ─────────────────────────────────

class TestAllS1Covered:
    """Every S1 entity must have an entry in the result dict."""

    def test_singleton_s1_gets_empty_list(self) -> None:
        s1 = pd.DataFrame({
            "entity_id": ["S1-001", "S1-002"],
            "business_name": ["Acme Corp", "Totally Unique XYZ"],
            "business_address": ["123 Main Rd", ""],
            "country": ["US", "US"],
        })
        s2 = pd.DataFrame({
            "entity_id": ["S2-001"],
            "business_name": ["Acme Corporation"],
            "business_address": ["123 Main Road"],
            "country": ["US"],
        })
        s3 = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
        cfg = {
            "minhash_lsh": {"num_perm": 128, "threshold": 0.5, "ngram_size": 3},
            "name_blocking": {"enabled": False},
            "country_blocking": {"enabled": True},
            "address_blocking": {"enabled": False},
        }
        result = generate_candidate_pairs(s1, s2, s3, cfg)
        # Both S1 entities must be in result
        assert "S1-001" in result
        assert "S1-002" in result


# ── 9. Normalisation integration ────────────────────────────────────

class TestNormalisationIntegration:
    """Verify _normalize_source correctly applies normalize_name/normalize_address."""

    def test_normalize_source_adds_columns(self) -> None:
        df = pd.DataFrame({
            "entity_id": ["S1-001"],
            "business_name": ["Acme Corporation"],
            "business_address": ["123 Main Road"],
            "country": ["US"],
        })
        result = _normalize_source(df)
        assert "name_clean" in result.columns
        assert "address_clean" in result.columns
        assert result["name_clean"].iloc[0] == "acme corp"
        assert result["address_clean"].iloc[0] == "123 main rd"

    def test_normalize_source_devanagari(self) -> None:
        df = pd.DataFrame({
            "entity_id": ["S2-001"],
            "business_name": ["राम मार्केटिंग प्राइवेट लिमिटेड"],
            "business_address": ["KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi"],
            "country": ["India"],
        })
        result = _normalize_source(df)
        assert "राम" in result["name_clean"].iloc[0]
        assert "pvt" in result["name_clean"].iloc[0]
        assert "ltd" in result["name_clean"].iloc[0]
