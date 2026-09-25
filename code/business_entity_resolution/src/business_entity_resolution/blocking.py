"""
blocking.py  –  Candidate-pair generation via MinHash LSH and rule-based keys.

Uses `datasketch.MinHash` / `datasketch.MinHashLSH` for approximate
nearest-neighbour blocking, optionally combined with exact-match keys
(country, name prefix).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd


def build_minhash_lsh_index(
    records: pd.DataFrame,
    text_col: str = "name_clean",
    num_perm: int = 128,
    threshold: float = 0.3,
    ngram_size: int = 3,
) -> Any:
    """Build a MinHashLSH index over *records*.

    Parameters
    ----------
    records : pd.DataFrame
        Must contain ``entity_id`` and *text_col*.
    text_col : str
        Column to shingle.
    num_perm : int
        Number of permutation functions for MinHash.
    threshold : float
        Jaccard threshold for the LSH index.
    ngram_size : int
        Character n-gram size for shingling.

    Returns
    -------
    Any
        A ``datasketch.MinHashLSH`` index (to be implemented).
    """
    raise NotImplementedError("MinHash LSH blocking – implement in Milestone 2")


def query_candidates(
    lsh_index: Any,
    query_records: pd.DataFrame,
    text_col: str = "name_clean",
    num_perm: int = 128,
    ngram_size: int = 3,
) -> Dict[str, List[str]]:
    """Query the LSH index for candidate matches.

    Parameters
    ----------
    lsh_index : MinHashLSH
    query_records : pd.DataFrame
    text_col : str
    num_perm : int
    ngram_size : int

    Returns
    -------
    dict[str, list[str]]
        Mapping from query entity_id to list of candidate entity_ids.
    """
    raise NotImplementedError("LSH querying – implement in Milestone 2")


def generate_candidate_pairs(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    blocking_cfg: Dict[str, Any],
) -> List[Tuple[str, str]]:
    """Generate all (source1_id, candidate_id) pairs via blocking.

    Parameters
    ----------
    source1 : pd.DataFrame
    source2 : pd.DataFrame
    source3 : pd.DataFrame
    blocking_cfg : dict
        Parsed ``configs/blocking.yaml``.

    Returns
    -------
    list[tuple[str, str]]
        Candidate pairs ``(s1_id, sx_id)``.
    """
    raise NotImplementedError("Full blocking pipeline – implement in Milestone 2")
