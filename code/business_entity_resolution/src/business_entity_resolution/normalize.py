"""
normalize.py  –  Unicode-aware text normalisation for business names and addresses.

Provides deterministic cleaning steps (NFKC normalisation, lowercasing, punctuation
removal, legal-suffix canonicalisation, street-abbreviation expansion, whitespace
collapsing, and tokenisation) so that downstream blocking and feature-engineering
operate on comparable strings.

Key properties:
- Unicode-aware: preserves non-Latin scripts (Devanagari, Cyrillic, etc.) and their
  combining vowel marks (matras, viramas) without crashing or mangling.
- Open-set country support: accepts country=None, "US", "India", "France", or any
  unseen country string without a restrictive hardcoded branch.
- No network calls or external gazetteers.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Union

import pandas as pd

# ── Precomputed Unicode punctuation & symbol translation table ──────
# Maps all Unicode punctuation (P*) and symbol (S*) code points in the BMP
# to a space ' ', leaving letters (L*), combining marks (M*), and numbers (N*)
# completely intact.
_PUNCT_SYMBOLS_TABLE: dict[int, str] = {
    i: " "
    for i in range(0x10000)
    if unicodedata.category(chr(i)).startswith(("P", "S"))
}

# Regex to collapse dots in acronyms: e.g. "L.L.C." -> "LLC", "U.S.A." -> "USA"
_ACRONYM_DOTS_RE = re.compile(r"(?<=\b[A-Za-z])\.(?=[A-Za-z](\.|\b))")

# Regex to strip apostrophes in contractions/possessives: "Orelee's" -> "Orelees"
_APOSTROPHE_RE = re.compile(r"(?<=\w)['’](?=\w)")


# ── Canonical dictionaries ──────────────────────────────────────────

_LEGAL_SUFFIX_MAP: dict[str, str] = {
    # Corporation
    "corporation": "corp",
    "corp": "corp",
    # Incorporated
    "incorporated": "inc",
    "inc": "inc",
    # Limited
    "limited": "ltd",
    "ltd": "ltd",
    # Private
    "private": "pvt",
    "pvt": "pvt",
    # Company
    "company": "co",
    "co": "co",
    # LLC / LLP / PLC
    "llc": "llc",
    "llp": "llp",
    "plc": "plc",
    # Public
    "public": "pub",
    # Devanagari transliterated legal suffixes
    "प्राइवेट": "pvt",
    "लिमिटेड": "ltd",
    "कंपनी": "co",
    "कॉर्पोरेशन": "corp",
    "कॉर्प": "corp",
    "एलएलपी": "llp",
}

_STREET_ABBREVIATIONS: dict[str, str] = {
    # Road
    "road": "rd",
    "rd": "rd",
    # Street
    "street": "st",
    "st": "st",
    "str": "st",
    # Avenue
    "avenue": "ave",
    "ave": "ave",
    "av": "ave",
    # Boulevard
    "boulevard": "blvd",
    "blvd": "blvd",
    "bvd": "blvd",
    # Drive
    "drive": "dr",
    "dr": "dr",
    # Lane
    "lane": "ln",
    "ln": "ln",
    # Highway
    "highway": "hwy",
    "hwy": "hwy",
    # Court
    "court": "ct",
    "ct": "ct",
    # Circle
    "circle": "cir",
    "cir": "cir",
    # Place
    "place": "pl",
    "pl": "pl",
    # Parkway
    "parkway": "pkwy",
    "pkwy": "pkwy",
    # Square
    "square": "sq",
    "sq": "sq",
    # Suite
    "suite": "ste",
    "ste": "ste",
    # Apartment
    "apartment": "apt",
    "apt": "apt",
    # Building
    "building": "bldg",
    "bldg": "bldg",
    # Floor
    "floor": "fl",
    "fl": "fl",
}


# ── Core text normalisation ─────────────────────────────────────────

def normalize_text(text: Optional[str]) -> str:
    """Clean and normalise raw text with full Unicode support.

    Applies NFKC normalisation, handles acronym dots, maps '&' to ' and ',
    removes punctuation/symbols while preserving non-Latin scripts and
    combining marks, lowercases, and collapses whitespace.

    Parameters
    ----------
    text : str | None
        Raw input string (may be ``None`` / ``NaN``).

    Returns
    -------
    str
        Cleaned string; empty string when input is missing or empty.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text_str = str(text).strip()
    if not text_str:
        return ""

    # 1. Unicode NFKC canonical decomposition & composition
    text_str = unicodedata.normalize("NFKC", text_str)

    # 2. Collapse dots in acronyms (e.g., "L.L.C." -> "LLC", "P.L.C." -> "PLC")
    text_str = _ACRONYM_DOTS_RE.sub("", text_str)

    # 3. Strip internal apostrophes (e.g., "Orelee's" -> "Orelees")
    text_str = _APOSTROPHE_RE.sub("", text_str)

    # 4. Canonicalise '&' to ' and ' before symbol stripping
    text_str = text_str.replace("&", " and ")

    # 5. Lowercase
    text_str = text_str.lower()

    # 6. Translate Unicode punctuation and symbols to whitespace
    #    (Preserves letters L*, numbers N*, and combining marks M*)
    text_str = text_str.translate(_PUNCT_SYMBOLS_TABLE)

    # 7. Collapse whitespace
    return " ".join(text_str.split())


def canonicalize_legal_suffix(name: str) -> str:
    """Replace common legal suffixes with canonical short forms.

    Parameters
    ----------
    name : str
        A pre-normalised business name.

    Returns
    -------
    str
        Name with canonicalised legal suffixes.
    """
    tokens = name.split()
    tokens = [_LEGAL_SUFFIX_MAP.get(t, t) for t in tokens]
    return " ".join(tokens)


def canonicalize_street_suffix(address: str) -> str:
    """Replace common street and address abbreviations with canonical forms.

    Parameters
    ----------
    address : str
        A pre-normalised address string.

    Returns
    -------
    str
        Address with canonicalised street terms.
    """
    tokens = address.split()
    tokens = [_STREET_ABBREVIATIONS.get(t, t) for t in tokens]
    return " ".join(tokens)


def tokenize(text: Optional[str]) -> List[str]:
    """Tokenize a text string into normalized word tokens.

    Parameters
    ----------
    text : str | None
        Input text.

    Returns
    -------
    list[str]
        List of non-empty cleaned tokens.
    """
    cleaned = normalize_text(text)
    return cleaned.split() if cleaned else []


# ── High-level public APIs ──────────────────────────────────────────

def normalize_name(
    name: Optional[str],
    country: Optional[str] = None,
    return_tokens: bool = False,
) -> Union[str, List[str]]:
    """Full normalisation pipeline for a business name.

    Parameters
    ----------
    name : str | None
        Raw business name.
    country : str | None, optional
        Country identifier (e.g. 'US', 'India', 'France', or None).
        Any country string is accepted gracefully.
    return_tokens : bool, default False
        If True, returns a list of tokens; otherwise returns a single string.

    Returns
    -------
    str | list[str]
        Normalised business name or token list.
    """
    cleaned = normalize_text(name)
    cleaned = canonicalize_legal_suffix(cleaned)
    tokens = cleaned.split()
    return tokens if return_tokens else " ".join(tokens)


def normalize_address(
    address: Optional[str],
    country: Optional[str] = None,
    return_tokens: bool = False,
) -> Union[str, List[str]]:
    """Full normalisation pipeline for a business address.

    Parameters
    ----------
    address : str | None
        Raw business address.
    country : str | None, optional
        Country identifier (e.g. 'US', 'India', 'France', or None).
        Any country string is accepted gracefully.
    return_tokens : bool, default False
        If True, returns a list of tokens; otherwise returns a single string.

    Returns
    -------
    str | list[str]
        Normalised business address or token list.
    """
    cleaned = normalize_text(address)
    cleaned = canonicalize_street_suffix(cleaned)
    tokens = cleaned.split()
    return tokens if return_tokens else " ".join(tokens)


# Alias for backward compatibility with Milestone 1
normalize_business_name = normalize_name


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply normalisation to all text columns of a source DataFrame.

    Adds ``name_clean`` and ``address_clean`` columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``business_name`` and ``business_address`` columns.
        May optionally contain ``country``.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with added normalised columns.
    """
    df = df.copy()
    if "country" in df.columns:
        df["name_clean"] = [
            normalize_name(name, country=c)
            for name, c in zip(df["business_name"], df["country"])
        ]
        df["address_clean"] = [
            normalize_address(addr, country=c)
            for addr, c in zip(df["business_address"], df["country"])
        ]
    else:
        df["name_clean"] = df["business_name"].apply(normalize_name)
        df["address_clean"] = df["business_address"].apply(normalize_address)
    return df
