"""
test_normalize.py  –  Unit tests for Unicode-aware text normalization.

Covers:
- Legal-suffix canonicalization (Corp/Corporation, Pvt/Private, Ltd/Limited, etc.)
- Street abbreviation canonicalization (Rd/Road, St/Street, Ave/Avenue, etc.)
- Ampersand canonicalization (& vs and)
- Real dataset examples from train_source1.tsv and train_source2.tsv
- Non-Latin script preservation (Devanagari Hindi records with combining marks)
- Open-set country handling (France, Germany, None, unseen strings)
- Tokenization and edge cases (None, NaN, empty strings)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "code" / "business_entity_resolution" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.normalize import (
    canonicalize_legal_suffix,
    canonicalize_street_suffix,
    normalize_address,
    normalize_business_name,
    normalize_dataframe,
    normalize_name,
    normalize_text,
    tokenize,
)


# ── 1. Legal-suffix canonicalization ────────────────────────────────

class TestLegalSuffixes:
    """Test Corp/Corporation, Pvt/Private, Ltd/Limited, etc."""

    def test_corp_corporation_canonicalization(self) -> None:
        name_full = normalize_name("Acme Corporation")
        name_short = normalize_name("Acme Corp")
        assert name_full == name_short == "acme corp"

    def test_corp_in_real_business_name(self) -> None:
        s1 = normalize_name("Pushpam Institute Corp")
        s2 = normalize_name("Pushpam Institute Corporation")
        assert s1 == s2 == "pushpam institute corp"

    def test_pvt_private_canonicalization(self) -> None:
        name_full = normalize_name("Apex Private Limited")
        name_short = normalize_name("Apex Pvt Ltd")
        assert name_full == name_short == "apex pvt ltd"

    def test_ltd_limited_canonicalization(self) -> None:
        name_full = normalize_name("Global Technologies Limited")
        name_short = normalize_name("Global Technologies Ltd")
        assert name_full == name_short == "global technologies ltd"

    def test_inc_incorporated_canonicalization(self) -> None:
        name_full = normalize_name("Alpha Solutions Incorporated")
        name_short = normalize_name("Alpha Solutions Inc")
        assert name_full == name_short == "alpha solutions inc"

    def test_acronym_dots_collapse(self) -> None:
        assert normalize_name("Acme L.L.C.") == "acme llc"
        assert normalize_name("Apex P.L.C.") == "apex plc"
        assert normalize_name("Omega L.L.P.") == "omega llp"


# ── 2. Street abbreviation canonicalization ─────────────────────────

class TestStreetAbbreviations:
    """Test Rd/Road, St/Street, Ave/Avenue, etc."""

    def test_rd_road_canonicalization(self) -> None:
        addr_full = normalize_address("17560 Ellis Road, Tahlequah, OK")
        addr_short = normalize_address("17560 Ellis Rd, Tahlequah, OK")
        assert addr_full == addr_short
        assert "ellis rd" in addr_full

    def test_st_street_canonicalization(self) -> None:
        addr_full = normalize_address("833 Reliance Street, Charlotte, NC")
        addr_short = normalize_address("833 Reliance St, Charlotte, NC")
        assert addr_full == addr_short
        assert "reliance st" in addr_full

    def test_ave_avenue_canonicalization(self) -> None:
        addr_full = normalize_address("1216 Preston Avenue, Charlottesville, VA")
        addr_short = normalize_address("1216 Preston Ave, Charlottesville, VA")
        assert addr_full == addr_short
        assert "preston ave" in addr_full

    def test_blvd_dr_ln_canonicalization(self) -> None:
        assert normalize_address("100 Sunset Boulevard") == normalize_address("100 Sunset Blvd")
        assert normalize_address("200 Ocean Drive") == normalize_address("200 Ocean Dr")
        assert normalize_address("300 Penny Lane") == normalize_address("300 Penny Ln")


# ── 3. Ampersand canonicalization ───────────────────────────────────

class TestAmpersand:
    """Test & vs and equivalence."""

    def test_ampersand_vs_and_in_name(self) -> None:
        n1 = normalize_name("Callicoat & Dailey Inc")
        n2 = normalize_name("Callicoat and Dailey Inc")
        assert n1 == n2 == "callicoat and dailey inc"

    def test_ampersand_in_real_data(self) -> None:
        n1 = normalize_name("Foot & Ankle Allied Center LLC")
        n2 = normalize_name("Foot and Ankle Allied Center LLC")
        assert n1 == n2 == "foot and ankle allied center llc"

    def test_ampersand_without_spaces(self) -> None:
        n1 = normalize_name("A&B Corp")
        n2 = normalize_name("A and B Corp")
        assert n1 == n2 == "a and b corp"


# ── 4. Real examples directly from train_source1 and train_source2 ──

class TestRealDatasetExamples:
    """Test real records pulled directly from train_source1.tsv and train_source2.tsv."""

    def test_real_example_source1_pushpam_corp(self) -> None:
        # Source1 Entity: S1-223600564
        raw_name = "Pushpam Institute Corp"
        raw_addr = (
            "1204, Block-C, Stratum @ Venus Ground, Nr. Jhansi Ki Rani Statue, "
            "Nehrunagar, Ahmadabad City, Ahmedabad, Gujarat"
        )
        norm_name = normalize_name(raw_name, country="India")
        norm_addr = normalize_address(raw_addr, country="India")

        assert norm_name == "pushpam institute corp"
        assert "stratumevenus" not in norm_addr  # '@' handled properly
        assert "1204 block c stratum" in norm_addr
        assert "ahmedabad gujarat" in norm_addr

    def test_real_example_source1_ellis_road(self) -> None:
        # Source1 Entity: S1-773889195
        raw_name = "Prime Money"
        raw_addr = "17560 Ellis Road, Tahlequah, OK"
        norm_name = normalize_name(raw_name, country="US")
        norm_addr = normalize_address(raw_addr, country="US")

        assert norm_name == "prime money"
        assert norm_addr == "17560 ellis rd tahlequah ok"

    def test_real_example_source1_callicoat_street(self) -> None:
        # Source1 Entity: S1-309349399
        raw_name = "Callicoat & Dailey Inc"
        raw_addr = "Charlotte, NC, 833 Reliance Street"
        norm_name = normalize_name(raw_name, country="US")
        norm_addr = normalize_address(raw_addr, country="US")

        assert norm_name == "callicoat and dailey inc"
        assert norm_addr == "charlotte nc 833 reliance st"

    def test_real_example_source2_devanagari_ram_marketing(self) -> None:
        # Source2 Entity: S2-166376419 (Devanagari Hindi script)
        raw_name = "राम मार्केटिंग प्राइवेट लिमिटेड"
        raw_addr = "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi"
        norm_name = normalize_name(raw_name, country="India")
        norm_addr = normalize_address(raw_addr, country="India")

        # Must not crash, must not mangle Hindi script into disconnected letters
        assert "राम" in norm_name
        assert "मार्केटिंग" in norm_name
        # Combining vowel marks (matras) and viramas must stay intact
        assert "र म" not in norm_name
        # Address normalisation handles hyphens and slashes
        assert norm_addr == "kh no 570 13 new delhi west delhi delhi"

    def test_real_example_source2_devanagari_aditya(self) -> None:
        # Source2 Entity: S2-639257739
        raw_name = "आदित्य प्रॉपर्टीज एलएलपी"
        raw_addr = "G-3/571, GULMOHAR COLONY, BHOPAL, Madhya Pradesh"
        norm_name = normalize_name(raw_name, country="India")
        norm_addr = normalize_address(raw_addr, country="India")

        assert "आदित्य" in norm_name
        assert "प्रॉपर्टीज" in norm_name
        assert "g 3 571 gulmohar colony bhopal madhya pradesh" == norm_addr


# ── 5. Open-set Country Support (France & unseen countries) ──────────

class TestOpenSetCountry:
    """Verify normalisation accepts country=None, France, or any unseen string without error."""

    def test_france_country_address(self) -> None:
        raw_addr = "12 Rue de la Paix, 75002 Paris"
        norm_addr = normalize_address(raw_addr, country="France")
        assert norm_addr == "12 rue de la paix 75002 paris"

    def test_france_country_name(self) -> None:
        raw_name = "Société Générale S.A."
        norm_name = normalize_name(raw_name, country="France")
        assert "société générale" in norm_name

    @pytest.mark.parametrize("country", [None, "US", "India", "France", "Germany", "Japan", "ZZ_UNKNOWN", ""])
    def test_unseen_country_strings_do_not_crash(self, country: str | None) -> None:
        name = normalize_name("Acme Corporation", country=country)
        addr = normalize_address("123 Main Road", country=country)
        assert name == "acme corp"
        assert addr == "123 main rd"


# ── 6. Tokenization & Edge Cases ────────────────────────────────────

class TestTokenizationAndEdgeCases:
    """Test tokenize helper, return_tokens option, and missing-value handling."""

    def test_tokenize_helper(self) -> None:
        tokens = tokenize("Acme Corporation, Inc. & Co.")
        assert tokens == ["acme", "corporation", "inc", "and", "co"]

    def test_tokenize_devanagari(self) -> None:
        tokens = tokenize("राम मार्केटिंग")
        assert tokens == ["राम", "मार्केटिंग"]

    def test_normalize_name_return_tokens(self) -> None:
        tokens = normalize_name("Apex Private Limited", return_tokens=True)
        assert tokens == ["apex", "pvt", "ltd"]

    def test_normalize_address_return_tokens(self) -> None:
        tokens = normalize_address("123 Main Road", return_tokens=True)
        assert tokens == ["123", "main", "rd"]

    def test_none_and_nan_handling(self) -> None:
        assert normalize_name(None) == ""
        assert normalize_name(float("nan")) == ""
        assert normalize_address(None) == ""
        assert normalize_address(float("nan")) == ""
        assert tokenize(None) == []
        assert tokenize("") == []

    def test_normalize_business_name_alias(self) -> None:
        assert normalize_business_name("Acme Corporation") == normalize_name("Acme Corporation")

    def test_dataframe_normalization(self) -> None:
        df = pd.DataFrame({
            "business_name": ["Acme Corp", "Baid Retail Corporation", None],
            "business_address": ["123 Main Road", "456 High Street", float("nan")],
            "country": ["US", "India", "France"],
        })
        cleaned_df = normalize_dataframe(df)
        assert "name_clean" in cleaned_df.columns
        assert "address_clean" in cleaned_df.columns
        assert cleaned_df["name_clean"].tolist() == ["acme corp", "baid retail corp", ""]
        assert cleaned_df["address_clean"].tolist() == ["123 main rd", "456 high st", ""]
