"""
io_utils.py  –  I/O helpers for TSV data, YAML configs, and result writing.

Public API
----------
Config helpers:
    load_config, merge_configs

Source / ground-truth loaders (schema-validated):
    load_source, load_ground_truth

Low-level TSV reader (no validation):
    load_tsv

Result writers (README-exact format):
    write_matching_results, write_candidate_pairs
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ── Expected schemas ────────────────────────────────────────────────
_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
_GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

_PREFIX_FOR_SOURCE: Dict[int, str] = {1: "S1-", 2: "S2-", 3: "S3-"}


# ── Config helpers (unchanged from Milestone 1) ────────────────────
def load_config(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path : Path

    Returns
    -------
    dict
    """
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def merge_configs(*paths: Path) -> Dict[str, Any]:
    """Load and merge multiple YAML config files.

    Later files override earlier ones at the top-level key level.

    Parameters
    ----------
    *paths : Path

    Returns
    -------
    dict
    """
    merged: Dict[str, Any] = {}
    for p in paths:
        merged.update(load_config(p))
    return merged


# ── Low-level TSV reader ────────────────────────────────────────────
def load_tsv(path: Path) -> pd.DataFrame:
    """Read a tab-separated file into a DataFrame.

    All columns are read as ``str``.  Default NA detection is
    **disabled** so that genuinely empty ``matched_entity_ids`` cells
    come back as ``""`` rather than ``NaN``.

    Parameters
    ----------
    path : Path

    Returns
    -------
    pd.DataFrame

    Warns
    -----
    UserWarning
        If *path* does not exist (returns an empty DataFrame).
    """
    path = Path(path)
    if not path.is_file():
        warnings.warn(
            f"TSV file not found, returning empty DataFrame: {path}",
            UserWarning,
            stacklevel=2,
        )
        return pd.DataFrame()

    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,  # preserve empty strings
        na_filter=False,        # never coerce to NaN
    )


# ── Schema-validated source loader ──────────────────────────────────
def load_source(
    path: Path,
    *,
    expected_source: Optional[int] = None,
) -> pd.DataFrame:
    """Load a source TSV and validate its schema.

    Parameters
    ----------
    path : Path
        Path to a ``*_source{1,2,3}.tsv`` file.
    expected_source : int | None
        If provided (1, 2, or 3), every ``entity_id`` is checked to
        start with the corresponding prefix (``S1-``, ``S2-``, ``S3-``).

    Returns
    -------
    pd.DataFrame
        Columns: ``entity_id``, ``business_name``, ``business_address``,
        ``country``.  All values are ``str``.

    Raises
    ------
    ValueError
        If required columns are missing or entity-ID prefixes don't match.

    Warns
    -----
    UserWarning
        If *path* does not exist (returns an empty DataFrame with the
        expected columns).
    """
    path = Path(path)
    if not path.is_file():
        warnings.warn(
            f"Source file not found, returning empty DataFrame: {path}",
            UserWarning,
            stacklevel=2,
        )
        return pd.DataFrame(columns=_SOURCE_COLUMNS)

    df = load_tsv(path)

    # ── Column validation ───────────────────────────────────────────
    missing = set(_SOURCE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Source file {path.name} is missing required columns: "
            f"{sorted(missing)}.  Got: {df.columns.tolist()}"
        )

    # ── Prefix validation ───────────────────────────────────────────
    if expected_source is not None:
        prefix = _PREFIX_FOR_SOURCE.get(expected_source)
        if prefix is None:
            raise ValueError(
                f"expected_source must be 1, 2, or 3; got {expected_source}"
            )
        bad_mask = ~df["entity_id"].str.startswith(prefix)
        n_bad = bad_mask.sum()
        if n_bad > 0:
            sample = df.loc[bad_mask, "entity_id"].head(5).tolist()
            raise ValueError(
                f"{n_bad:,} entity_id(s) in {path.name} do not start with "
                f"'{prefix}'. Sample: {sample}"
            )

    logger.info("Loaded %s: %s rows", path.name, f"{len(df):,}")
    return df


# ── Ground-truth loader ────────────────────────────────────────────
def load_ground_truth(path: Path) -> pd.DataFrame:
    """Load the ground-truth matching file and validate its schema.

    Parameters
    ----------
    path : Path
        Path to ``train_ground_truth.tsv``.

    Returns
    -------
    pd.DataFrame
        Columns: ``source1_entity_id`` (str),
        ``matched_entity_ids`` (str — comma-separated, possibly empty).

    Raises
    ------
    ValueError
        If required columns are missing or ``source1_entity_id`` values
        don't carry the ``S1-`` prefix.

    Warns
    -----
    UserWarning
        If *path* does not exist (returns an empty DataFrame).
    """
    path = Path(path)
    if not path.is_file():
        warnings.warn(
            f"Ground-truth file not found, returning empty DataFrame: {path}",
            UserWarning,
            stacklevel=2,
        )
        return pd.DataFrame(columns=_GROUND_TRUTH_COLUMNS)

    df = load_tsv(path)

    # ── Column validation ───────────────────────────────────────────
    missing = set(_GROUND_TRUTH_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Ground-truth file {path.name} is missing required columns: "
            f"{sorted(missing)}.  Got: {df.columns.tolist()}"
        )

    # ── S1-prefix validation ────────────────────────────────────────
    bad_mask = ~df["source1_entity_id"].str.startswith("S1-")
    n_bad = bad_mask.sum()
    if n_bad > 0:
        sample = df.loc[bad_mask, "source1_entity_id"].head(5).tolist()
        raise ValueError(
            f"{n_bad:,} source1_entity_id(s) in {path.name} lack the "
            f"'S1-' prefix. Sample: {sample}"
        )

    logger.info("Loaded %s: %s rows", path.name, f"{len(df):,}")
    return df


# ── Result writers (README-exact format) ────────────────────────────
def write_matching_results(
    matches: Dict[str, List[str]],
    path: Path,
) -> None:
    """Write ``matching_results.tsv`` in the exact README format.

    Format
    ------
    * Tab-separated, two columns: ``source1_entity_id``,
      ``matched_entity_ids``.
    * ``matched_entity_ids`` is a comma-joined list (no spaces, no quotes).
    * Singletons get an empty string (not NaN, not "NaN").
    * Trailing newline after the last row.

    Parameters
    ----------
    matches : dict[str, list[str]]
        ``{source1_entity_id: [matched_entity_ids]}``.
    path : Path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = ["source1_entity_id\tmatched_entity_ids"]
    for s1_id in sorted(matches):
        matched = ",".join(matches[s1_id])
        lines.append(f"{s1_id}\t{matched}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s (%d entries)", path, len(matches))


def write_candidate_pairs(
    candidates: Dict[str, List[str]],
    path: Path,
) -> None:
    """Write ``candidate_pairs.tsv`` in the exact README format.

    Format
    ------
    * Tab-separated, two columns: ``source1_entity_id``,
      ``candidate_entity_ids``.
    * ``candidate_entity_ids`` is a comma-joined list.
    * Empty string when blocking found no candidates.
    * Trailing newline after the last row.

    Parameters
    ----------
    candidates : dict[str, list[str]]
        ``{source1_entity_id: [candidate_entity_ids]}``.
    path : Path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = ["source1_entity_id\tcandidate_entity_ids"]
    for s1_id in sorted(candidates):
        cands = ",".join(candidates[s1_id])
        lines.append(f"{s1_id}\t{cands}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s (%d entries)", path, len(candidates))
