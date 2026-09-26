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
    write_matching_results, write_candidate_pairs, append_candidate_pairs, load_candidate_pairs
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
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge two config dictionaries.

    *override* takes precedence over *base*.
    """
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = merge_configs(merged[k], v)
        else:
            merged[k] = v
    return merged


# ── Low-level TSV loader (no schema enforcement) ───────────────────
def load_tsv(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Read a TSV into a DataFrame without type coercions or column assumptions.

    * All columns are read as ``str`` (never float / int / bool).
    * Missing/empty fields stay as empty string ``""`` (never ``NaN``).
    * Leading / trailing whitespace in string values is NOT stripped.
    * ``#`` is not treated as a comment character.
    * No quoting tricks: every line is split purely on ``\\t``.

    Warns (UserWarning) and returns an empty DataFrame if *path* does not exist.
    """
    path = Path(path)
    if not path.is_file():
        warnings.warn(
            f"File not found, returning empty DataFrame: {path}",
            UserWarning,
            stacklevel=2,
        )
        return pd.DataFrame()

    defaults = {
        "sep": "\t",
        "dtype": str,
        "keep_default_na": False,
        "na_values": [],
        "comment": None,
        "encoding": "utf-8",
        "on_bad_lines": "error",
    }
    defaults.update(kwargs)
    return pd.read_csv(path, **defaults)


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

    # Keep only the canonical columns in canonical order
    df = df[_SOURCE_COLUMNS].copy()

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
                f"{n_bad:,} entity_id(s) in {path.name} lack the '{prefix}' "
                f"prefix. Sample: {sample}"
            )

    logger.info("Loaded %s: %s rows", path.name, f"{len(df):,}")
    return df


# ── Schema-validated ground-truth loader ────────────────────────────
def load_ground_truth(path: Path) -> pd.DataFrame:
    """Load ``train_ground_truth.tsv`` and validate its schema.

    Parameters
    ----------
    path : Path
        Path to the ground-truth file.

    Returns
    -------
    pd.DataFrame
        Columns: ``source1_entity_id``, ``matched_entity_ids``.
        All values are ``str``.

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

    with open(path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in sorted(matches):
            matched = ",".join(matches[s1_id])
            f.write(f"{s1_id}\t{matched}\n")

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

    with open(path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in sorted(candidates):
            cands = ",".join(candidates[s1_id])
            f.write(f"{s1_id}\t{cands}\n")

    logger.info("Wrote %s (%d entries)", path, len(candidates))


def append_candidate_pairs(
    candidates: Dict[str, List[str]],
    path: Path,
    *,
    write_header: bool = False,
) -> None:
    """Append candidate pairs to a TSV file in exact README format.

    Parameters
    ----------
    candidates : dict[str, list[str]]
        ``{source1_entity_id: [candidate_entity_ids]}``.
    path : Path
        Target TSV path.
    write_header : bool
        If True, writes the header line before records (creates or truncates file).
        If False, appends without header.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mode = "w" if write_header else "a"
    with open(path, mode, encoding="utf-8") as f:
        if write_header:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in sorted(candidates):
            cands = ",".join(candidates[s1_id])
            f.write(f"{s1_id}\t{cands}\n")

    logger.info(
        "%s %s (%d entries)",
        "Wrote" if write_header else "Appended to",
        path,
        len(candidates),
    )


def load_candidate_pairs(path: Path) -> Dict[str, List[str]]:
    """Load a ``candidate_pairs.tsv`` or checkpoint file into memory.

    Parameters
    ----------
    path : Path
        Path to ``candidate_pairs.tsv`` or a partition checkpoint TSV.

    Returns
    -------
    dict[str, list[str]]
        Mapping of ``source1_entity_id`` to list of candidate IDs.
    """
    path = Path(path)
    if not path.is_file():
        warnings.warn(
            f"Candidate pairs file not found, returning empty dict: {path}",
            UserWarning,
            stacklevel=2,
        )
        return {}

    candidates: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            s1_id = parts[0]
            cand_str = parts[1] if len(parts) > 1 else ""
            candidates[s1_id] = cand_str.split(",") if cand_str else []

    logger.info("Loaded %s (%d entries)", path.name, len(candidates))
    return candidates
