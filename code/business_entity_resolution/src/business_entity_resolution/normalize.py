"""
normalize.py  –  Text normalisation for business names and addresses.

Provides deterministic cleaning steps (lowercasing, punctuation removal,
legal-suffix canonicalization, whitespace collapsing) so that downstream
blocking and feature-engineering operate on comparable strings.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd


# ── Legal-suffix canonical forms ────────────────────────────────────
_LEGAL_SUFFIX_MAP: dict[str, str] = {
    "corporation": "corp",
    "incorporated": "inc",
    "limited": "ltd",
    "private": "pvt",
    "company": "co",
}


def normalize_text(text: Optional[str]) -> str:
    """Lowercase, strip punctuation, and collapse whitespace.

    Parameters
    ----------
    text : str | None
        Raw input string (may be ``None`` / ``NaN``).

    Returns
    -------
    str
        Cleaned string; empty string when input is missing.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonicalize_legal_suffix(name: str) -> str:
    """Replace common legal suffixes with canonical short forms.

    Parameters
    ----------
    name : str
        A *pre-normalised* business name.

    Returns
    -------
    str
        Name with legal suffixes replaced.
    """
    tokens = name.split()
    tokens = [_LEGAL_SUFFIX_MAP.get(t, t) for t in tokens]
    return " ".join(tokens)


def normalize_business_name(name: Optional[str]) -> str:
    """Full normalisation pipeline for a business name.

    Parameters
    ----------
    name : str | None

    Returns
    -------
    str
    """
    name = normalize_text(name)
    name = name.replace("&", " and ")
    name = re.sub(r"\s+", " ", name).strip()
    name = canonicalize_legal_suffix(name)
    return name


def normalize_address(address: Optional[str]) -> str:
    """Full normalisation pipeline for a business address.

    Parameters
    ----------
    address : str | None

    Returns
    -------
    str
    """
    return normalize_text(address)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply normalisation to all text columns of a source DataFrame.

    Adds ``name_clean`` and ``address_clean`` columns in-place.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``business_name`` and ``business_address`` columns.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with added normalised columns.
    """
    df = df.copy()
    df["name_clean"] = df["business_name"].apply(normalize_business_name)
    df["address_clean"] = df["business_address"].apply(normalize_address)
    return df
