"""
io_utils.py  –  I/O helpers for TSV data, YAML configs, and result writing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


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


def load_tsv(path: Path) -> pd.DataFrame:
    """Read a tab-separated file into a DataFrame.

    Parameters
    ----------
    path : Path

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(path, sep="\t", dtype=str)


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


def write_matching_results(
    matches: Dict[str, List[str]],
    path: Path,
) -> None:
    """Write ``matching_results.tsv``.

    Parameters
    ----------
    matches : dict[str, list[str]]
        ``{source1_entity_id: [matched_entity_ids]}``.
    path : Path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for s1_id in sorted(matches):
        matched = ",".join(matches[s1_id])
        rows.append(f"{s1_id}\t{matched}")
    header = "source1_entity_id\tmatched_entity_ids"
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def write_candidate_pairs(
    candidates: Dict[str, List[str]],
    path: Path,
) -> None:
    """Write ``candidate_pairs.tsv``.

    Parameters
    ----------
    candidates : dict[str, list[str]]
        ``{source1_entity_id: [candidate_entity_ids]}``.
    path : Path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for s1_id in sorted(candidates):
        cands = ",".join(candidates[s1_id])
        rows.append(f"{s1_id}\t{cands}")
    header = "source1_entity_id\tcandidate_entity_ids"
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
