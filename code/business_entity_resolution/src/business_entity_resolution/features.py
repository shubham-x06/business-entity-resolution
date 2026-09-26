"""
features.py  –  Vectorized chunked pairwise feature engineering for entity resolution.

Computes string-similarity features (via RapidFuzz), token-set metrics, TF-IDF cosine
similarity, length differences, and explicit empty-address flags for candidate pairs
produced by the blocking stage.

Design Principles & Scale Constraints:
1. Streaming / Chunked Processing: Candidate pairs are read from disk in batches (e.g.
   100,000 pairs/chunk) and features are streamed directly to disk (Parquet or TSV append)
   without accumulating millions of rows in RAM.
2. Low-Memory Reference Entity Cache: S1, S2, and S3 are loaded once and normalized using
   ``normalize.py`` (cached to parquet checkpoints). Compact tuples (name, address, country)
   held in an in-memory dictionary consume < 2 GB RAM for 5M entities.
3. Fully Vectorized Compute: Within each chunk, string similarities (RapidFuzz C-extension),
   token set operations, length differences, and TF-IDF row-wise dot products execute
   over batches without per-pair Python object instantiation.
4. Country Partitioning Filter: Supports ``--country India`` or ``--country US`` to restrict
   feature extraction to only candidate pairs where Source 1 belongs to that country.
5. Explicit Empty-Address Handling: ~3% of real Source 2/3 entities lack an address.
   When either or both addresses in a pair are empty:
   - ``addr_token_overlap`` = -1.0 (sentinel)
   - ``addr_edit_sim`` = -1.0 (sentinel)
   - ``addr_len_diff`` = -1.0 (sentinel)
   - ``s1_empty_addr`` = 1.0 if S1 address is empty else 0.0
   - ``cand_empty_addr`` = 1.0 if candidate address is empty else 0.0
   This prevents arbitrary similarities between empty strings and enables tree models
   to split on missingness.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Logging setup ───────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Feature Schema Constants ─────────────────────────────────────────
SENTINEL_EMPTY_ADDR: float = -1.0

FEATURE_COLUMNS: List[str] = [
    "blocking_rank",
    "name_jaro_winkler",
    "name_levenshtein_ratio",
    "name_token_jaccard",
    "name_tfidf_cosine",
    "name_len_diff",
    "addr_token_overlap",
    "addr_edit_sim",
    "addr_len_diff",
    "country_match",
    "s1_empty_addr",
    "cand_empty_addr",
]

OUTPUT_COLUMNS: List[str] = [
    "source1_entity_id",
    "candidate_entity_id",
] + FEATURE_COLUMNS


# ── Peak Memory Monitoring Helper ───────────────────────────────────

class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def get_peak_memory_mb() -> float:
    """Return peak process memory (working set) in MB.

    Cross-platform: uses Windows PSAPI on Windows; falls back to
    tracemalloc or resource on POSIX.
    """
    if sys.platform == "win32":
        try:
            pmc = _PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            handle = k32.GetCurrentProcess()
            psapi = ctypes.windll.psapi
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
                wintypes.DWORD,
            ]
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                return float(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0)
        except Exception:
            pass

    # POSIX / fallback
    try:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        # On Linux rusage.ru_maxrss is in KiB; on Darwin it's in bytes
        if sys.platform == "darwin":
            return float(rusage.ru_maxrss) / (1024.0 * 1024.0)
        return float(rusage.ru_maxrss) / 1024.0
    except ImportError:
        pass

    try:
        import tracemalloc
        if tracemalloc.is_tracing():
            _, peak = tracemalloc.get_traced_memory()
            return float(peak) / (1024.0 * 1024.0)
    except Exception:
        pass

    return 0.0


def get_current_rss_mb() -> float:
    """Return current process resident working set in MB."""
    if sys.platform == "win32":
        try:
            pmc = _PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            handle = k32.GetCurrentProcess()
            psapi = ctypes.windll.psapi
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
                wintypes.DWORD,
            ]
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                return float(pmc.WorkingSetSize) / (1024.0 * 1024.0)
        except Exception:
            pass
    return 0.0


# ── Scalar Feature Helpers (Backward Compatibility / Unit Testing) ───

def compute_name_features(
    name_a: str,
    name_b: str,
    vectorizer: Optional[TfidfVectorizer] = None,
) -> Dict[str, float]:
    """Compute name-similarity features for a single pair of normalized names.

    Parameters
    ----------
    name_a : str
        Normalised name from Source 1.
    name_b : str
        Normalised name from Source 2/3.
    vectorizer : TfidfVectorizer, optional
        Pre-fitted vectorizer for TF-IDF cosine similarity.

    Returns
    -------
    dict[str, float]
        Dictionary with keys: ``name_jaro_winkler``, ``name_levenshtein_ratio``,
        ``name_token_jaccard``, ``name_tfidf_cosine``, ``name_len_diff``.
    """
    na = name_a if name_a is not None else ""
    nb = name_b if name_b is not None else ""

    jw = float(JaroWinkler.similarity(na, nb))
    lev = float(Levenshtein.normalized_similarity(na, nb))

    toks_a = set(na.split())
    toks_b = set(nb.split())
    union_toks = toks_a | toks_b
    jaccard = float(len(toks_a & toks_b) / len(union_toks)) if union_toks else 1.0

    max_len = max(len(na), len(nb), 1)
    len_diff = float(abs(len(na) - len(nb)) / max_len)

    tfidf_cos = 0.0
    if vectorizer is not None and (na or nb):
        try:
            vecs = vectorizer.transform([na, nb])
            tfidf_cos = float(vecs[0].multiply(vecs[1]).sum())
        except Exception:
            tfidf_cos = 0.0

    return {
        "name_jaro_winkler": jw,
        "name_levenshtein_ratio": lev,
        "name_token_jaccard": jaccard,
        "name_tfidf_cosine": tfidf_cos,
        "name_len_diff": len_diff,
    }


def compute_address_features(addr_a: str, addr_b: str) -> Dict[str, float]:
    """Compute address-similarity features for a single pair.

    Explicitly handles empty addresses: when either address is empty,
    similarity features receive sentinel value -1.0 and empty flags are set.

    Parameters
    ----------
    addr_a : str
        Normalised address from Source 1.
    addr_b : str
        Normalised address from Source 2/3.

    Returns
    -------
    dict[str, float]
        Dictionary with keys: ``addr_token_overlap``, ``addr_edit_sim``,
        ``addr_len_diff``, ``s1_empty_addr``, ``cand_empty_addr``.
    """
    aa = addr_a if addr_a is not None else ""
    ab = addr_b if addr_b is not None else ""

    s1_empty = 1.0 if not aa.strip() else 0.0
    cand_empty = 1.0 if not ab.strip() else 0.0

    if s1_empty > 0.5 or cand_empty > 0.5:
        return {
            "addr_token_overlap": SENTINEL_EMPTY_ADDR,
            "addr_edit_sim": SENTINEL_EMPTY_ADDR,
            "addr_len_diff": SENTINEL_EMPTY_ADDR,
            "s1_empty_addr": s1_empty,
            "cand_empty_addr": cand_empty,
        }

    toks_a = set(aa.split())
    toks_b = set(ab.split())
    min_len = min(len(toks_a), len(toks_b))
    overlap = float(len(toks_a & toks_b) / min_len) if min_len > 0 else 0.0

    edit_sim = float(Levenshtein.normalized_similarity(aa, ab))

    max_l = max(len(aa), len(ab), 1)
    len_diff = float(abs(len(aa) - len(ab)) / max_l)

    return {
        "addr_token_overlap": overlap,
        "addr_edit_sim": edit_sim,
        "addr_len_diff": len_diff,
        "s1_empty_addr": 0.0,
        "cand_empty_addr": 0.0,
    }


def compute_country_features(country_a: str, country_b: str) -> Dict[str, float]:
    """Compute country-match feature.

    Parameters
    ----------
    country_a : str
    country_b : str

    Returns
    -------
    dict[str, float]
        ``{"country_match": 1.0}`` if same country, else ``0.0``.
    """
    ca = str(country_a).strip().lower() if country_a else ""
    cb = str(country_b).strip().lower() if country_b else ""
    match = 1.0 if (ca and cb and ca == cb) else 0.0
    return {"country_match": match}


# ── TF-IDF Vectorizer Fitting ────────────────────────────────────────

def fit_name_tfidf_vectorizer(
    corpus: Sequence[str],
    max_features: int = 25000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> TfidfVectorizer:
    """Fit a compact, memory-efficient TF-IDF vectorizer on business names.

    Decided and documented: Vectorizer is fit per country partition (or on
    the full reference corpus) once globally, not per-pair. Sublinear TF
    scaling is applied to dampen high-frequency tokens (e.g. 'inc', 'pvt').

    Parameters
    ----------
    corpus : sequence of str
        Normalised business names from reference sources.
    max_features : int, default 25000
        Maximum vocabulary size to guarantee bounded sparse matrix size.
    ngram_range : tuple of (int, int), default (1, 2)
        Unigram + bigram word features.

    Returns
    -------
    TfidfVectorizer
    """
    logger.info("Fitting TF-IDF vectorizer on %d names (max_features=%d)...",
                len(corpus), max_features)
    vec = TfidfVectorizer(
        max_features=max_features,
        analyzer="word",
        ngram_range=ngram_range,
        sublinear_tf=True,
        dtype=np.float32,
    )
    vec.fit(corpus)
    logger.info("TF-IDF vectorizer fit complete. Vocabulary size: %d", len(vec.vocabulary_))
    return vec


# ── Vectorized Batch / Chunk Feature Computation ─────────────────────

def compute_pairwise_features_chunk(
    s1_ids: List[str],
    cand_ids: List[str],
    ranks: np.ndarray,
    s1_names: List[str],
    cand_names: List[str],
    s1_addrs: List[str],
    cand_addrs: List[str],
    s1_countries: List[str],
    cand_countries: List[str],
    vectorizer: Optional[TfidfVectorizer] = None,
) -> pd.DataFrame:
    """Vectorized pairwise feature calculation for a chunk of candidate pairs.

    Zero Python-level row loops: uses RapidFuzz C implementations, NumPy array
    math, and SciPy sparse row-wise multiplications.

    Parameters
    ----------
    s1_ids : list[str]
    cand_ids : list[str]
    ranks : np.ndarray (int16/int32)
        Candidate rank within Source 1 blocking results.
    s1_names, cand_names : list[str]
        Pre-normalised business names.
    s1_addrs, cand_addrs : list[str]
        Pre-normalised business addresses.
    s1_countries, cand_countries : list[str]
    vectorizer : TfidfVectorizer, optional
        Pre-fitted TF-IDF vectorizer for name cosine similarity.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns specified by ``OUTPUT_COLUMNS``.
    """
    n_pairs = len(s1_ids)
    if n_pairs == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # 1. Jaro-Winkler similarity on normalized names (RapidFuzz C-level)
    name_jw = np.array(
        [JaroWinkler.similarity(a, b) for a, b in zip(s1_names, cand_names)],
        dtype=np.float32,
    )

    # 2. Levenshtein ratio on normalized names (RapidFuzz C-level)
    name_lev = np.array(
        [Levenshtein.normalized_similarity(a, b) for a, b in zip(s1_names, cand_names)],
        dtype=np.float32,
    )

    # 3. Token-set Jaccard similarity on normalized name tokens
    # Using list comprehensions with set operations (executed at C speed)
    s1_tok_sets = [set(n.split()) for n in s1_names]
    cand_tok_sets = [set(n.split()) for n in cand_names]
    name_jacc = np.array(
        [
            len(a & b) / (len(a) + len(b) - len(a & b)) if (a or b) else 1.0
            for a, b in zip(s1_tok_sets, cand_tok_sets)
        ],
        dtype=np.float32,
    )

    # 4. TF-IDF cosine similarity on normalized names
    # Vectorized via unique-transform caching + sparse row-wise dot product
    if vectorizer is not None:
        try:
            u_s1, inv_s1 = np.unique(s1_names, return_inverse=True)
            u_cand, inv_cand = np.unique(cand_names, return_inverse=True)

            mat_s1 = vectorizer.transform(u_s1)[inv_s1]
            mat_cand = vectorizer.transform(u_cand)[inv_cand]

            # Row-wise dot product of L2-normalized sparse vectors
            name_tfidf = np.asarray(
                mat_s1.multiply(mat_cand).sum(axis=1),
                dtype=np.float32,
            ).ravel()
            # Clip numerical precision to [0.0, 1.0]
            name_tfidf = np.clip(name_tfidf, 0.0, 1.0)
        except Exception as e:
            logger.warning("Error computing batch TF-IDF cosine: %s", e)
            name_tfidf = np.zeros(n_pairs, dtype=np.float32)
    else:
        name_tfidf = np.zeros(n_pairs, dtype=np.float32)

    # 5. Name length difference (absolute, normalized by max length)
    s1_name_lens = np.array([len(x) for x in s1_names], dtype=np.float32)
    cand_name_lens = np.array([len(x) for x in cand_names], dtype=np.float32)
    max_name_lens = np.maximum(np.maximum(s1_name_lens, cand_name_lens), 1.0)
    name_len_diff = np.abs(s1_name_lens - cand_name_lens) / max_name_lens

    # 6. Exact country match flag (1.0 / 0.0)
    country_match = np.array(
        [1.0 if (a and b and a.lower() == b.lower()) else 0.0
         for a, b in zip(s1_countries, cand_countries)],
        dtype=np.float32,
    )

    # 7. Empty-address flags & address features with -1.0 sentinel
    s1_empty = np.array([1.0 if not a.strip() else 0.0 for a in s1_addrs], dtype=np.float32)
    cand_empty = np.array([1.0 if not a.strip() else 0.0 for a in cand_addrs], dtype=np.float32)
    either_empty = (s1_empty > 0.5) | (cand_empty > 0.5)

    addr_overlap = np.full(n_pairs, SENTINEL_EMPTY_ADDR, dtype=np.float32)
    addr_edit_sim = np.full(n_pairs, SENTINEL_EMPTY_ADDR, dtype=np.float32)
    addr_len_diff = np.full(n_pairs, SENTINEL_EMPTY_ADDR, dtype=np.float32)

    valid_indices = np.where(~either_empty)[0]
    if len(valid_indices) > 0:
        val_s1_addrs = [s1_addrs[i] for i in valid_indices]
        val_cand_addrs = [cand_addrs[i] for i in valid_indices]

        # Address edit distance (normalized Levenshtein similarity in [0, 1])
        addr_edit_sim[valid_indices] = [
            Levenshtein.normalized_similarity(a, b)
            for a, b in zip(val_s1_addrs, val_cand_addrs)
        ]

        # Address token overlap ratio: len(A & B) / min(len(A), len(B))
        val_s1_toks = [set(a.split()) for a in val_s1_addrs]
        val_cand_toks = [set(b.split()) for b in val_cand_addrs]
        addr_overlap[valid_indices] = [
            len(a & b) / min(len(a), len(b)) if min(len(a), len(b)) > 0 else 0.0
            for a, b in zip(val_s1_toks, val_cand_toks)
        ]

        # Address length difference normalized by max length
        v_s1_len = np.array([len(x) for x in val_s1_addrs], dtype=np.float32)
        v_cand_len = np.array([len(x) for x in val_cand_addrs], dtype=np.float32)
        v_max_len = np.maximum(np.maximum(v_s1_len, v_cand_len), 1.0)
        addr_len_diff[valid_indices] = np.abs(v_s1_len - v_cand_len) / v_max_len

    # Assemble chunk DataFrame
    chunk_df = pd.DataFrame({
        "source1_entity_id": s1_ids,
        "candidate_entity_id": cand_ids,
        "blocking_rank": ranks.astype(np.int16),
        "name_jaro_winkler": name_jw,
        "name_levenshtein_ratio": name_lev,
        "name_token_jaccard": name_jacc,
        "name_tfidf_cosine": name_tfidf,
        "name_len_diff": name_len_diff,
        "addr_token_overlap": addr_overlap,
        "addr_edit_sim": addr_edit_sim,
        "addr_len_diff": addr_len_diff,
        "country_match": country_match,
        "s1_empty_addr": s1_empty,
        "cand_empty_addr": cand_empty,
    })

    return chunk_df


# ── In-Memory Reference Cache Loader ─────────────────────────────────

def load_and_normalize_entity_cache(
    root: Path,
    country_filter: Optional[str] = None,
    checkpoint_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Tuple[str, str, str]], TfidfVectorizer]:
    """Load Source 1, 2, and 3 records and construct the reference entity cache.

    Reuses ``normalize.py`` functions (``normalize_name``, ``normalize_address``)
    without re-implementing any normalization logic.
    Caches normalized parquet files in ``checkpoint_dir`` to eliminate repeated
    normalization overhead on subsequent invocations.

    Parameters
    ----------
    root : Path
        Project root directory.
    country_filter : str, optional
        Country identifier (e.g. 'India', 'US'). If specified, only records
        matching this country are loaded and indexed.
    checkpoint_dir : Path, optional
        Directory to read/write normalized parquet checkpoints.

    Returns
    -------
    entity_cache : dict[str, tuple[str, str, str]]
        Mapping ``entity_id -> (name_clean, address_clean, interned_country)``.
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer for normalized names in this partition.
    """
    from business_entity_resolution.io_utils import load_config, load_source
    from business_entity_resolution.normalize import normalize_name, normalize_address

    cfg_dir = root / "code" / "business_entity_resolution" / "configs"
    paths_cfg = load_config(cfg_dir / "paths.yaml")
    data_cfg = paths_cfg["data"]["train"]

    ckpt_dir = checkpoint_dir or (root / "checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    c_tag = country_filter.lower() if country_filter else "all"

    # Check for existing normalized parquet cache
    s1_parquet = ckpt_dir / f"norm_source1_{c_tag}.parquet"
    s2_parquet = ckpt_dir / f"norm_source2_{c_tag}.parquet"
    s3_parquet = ckpt_dir / f"norm_source3_{c_tag}.parquet"

    if s1_parquet.is_file() and s2_parquet.is_file() and s3_parquet.is_file():
        logger.info("Loading pre-normalized entities from parquet checkpoints (%s)...", c_tag)
        t0 = time.time()
        s1_df = pd.read_parquet(s1_parquet)
        s2_df = pd.read_parquet(s2_parquet)
        s3_df = pd.read_parquet(s3_parquet)
        logger.info("Loaded parquet caches in %.2fs", time.time() - t0)
    else:
        logger.info("Loading raw TSVs and applying Unicode normalization (%s)...", c_tag)
        t0 = time.time()
        s1_raw = load_source(root / data_cfg["source1"], expected_source=1)
        s2_raw = load_source(root / data_cfg["source2"], expected_source=2)
        s3_raw = load_source(root / data_cfg["source3"], expected_source=3)

        if country_filter:
            s1_raw = s1_raw[s1_raw["country"] == country_filter].copy()
            s2_raw = s2_raw[s2_raw["country"] == country_filter].copy()
            s3_raw = s3_raw[s3_raw["country"] == country_filter].copy()
            logger.info("Filtered to country '%s': S1=%d, S2=%d, S3=%d",
                        country_filter, len(s1_raw), len(s2_raw), len(s3_raw))

        def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
            names = [normalize_name(n, country=c) for n, c in zip(df["business_name"], df["country"])]
            addrs = [normalize_address(a, country=c) for a, c in zip(df["business_address"], df["country"])]
            return pd.DataFrame({
                "entity_id": df["entity_id"].values,
                "name_clean": names,
                "address_clean": addrs,
                "country": df["country"].values,
            })

        s1_df = _normalize_df(s1_raw)
        s2_df = _normalize_df(s2_raw)
        s3_df = _normalize_df(s3_raw)

        # Save parquet checkpoints for instant reuse
        s1_df.to_parquet(s1_parquet, index=False)
        s2_df.to_parquet(s2_parquet, index=False)
        s3_df.to_parquet(s3_parquet, index=False)
        logger.info("Saved normalized parquet caches in %.2fs", time.time() - t0)

    # Build compact in-memory lookup dictionary
    # entity_id -> (name_clean, address_clean, country)
    t_dict = time.time()
    cache: Dict[str, Tuple[str, str, str]] = {}

    for df in (s1_df, s2_df, s3_df):
        eids = df["entity_id"].values
        names = df["name_clean"].values
        addrs = df["address_clean"].values
        countries = [sys.intern(str(c)) for c in df["country"].values]
        for i in range(len(eids)):
            cache[eids[i]] = (str(names[i]), str(addrs[i]), countries[i])

    logger.info("Constructed entity lookup cache with %d entities in %.2fs (RAM: %.1f MB)",
                len(cache), time.time() - t_dict, get_current_rss_mb())

    # Fit TF-IDF vectorizer on reference names (S2 + S3 names)
    ref_names = list(s2_df["name_clean"].values) + list(s3_df["name_clean"].values)
    vectorizer = fit_name_tfidf_vectorizer(ref_names)

    return cache, vectorizer


# ── Streaming Chunked Feature Pipeline ───────────────────────────────

def extract_features_streaming(
    candidate_pairs_path: Union[str, Path],
    entity_cache: Dict[str, Tuple[str, str, str]],
    output_path: Union[str, Path],
    country_filter: Optional[str] = None,
    vectorizer: Optional[TfidfVectorizer] = None,
    chunk_size: int = 100000,
    max_pairs: Optional[int] = None,
    log_every: int = 200000,
) -> Dict[str, Any]:
    """Stream candidate pairs, compute features in vectorized chunks, and write to disk.

    Parameters
    ----------
    candidate_pairs_path : Path or str
        Path to ``output/candidate_pairs.tsv``.
    entity_cache : dict[str, tuple[str, str, str]]
        In-memory lookup dictionary from ``load_and_normalize_entity_cache``.
    output_path : Path or str
        Destination path (.parquet or .tsv).
    country_filter : str, optional
        If specified (e.g. 'India'), skips any S1 entity not matching this country.
    vectorizer : TfidfVectorizer, optional
        Pre-fitted vectorizer for TF-IDF cosine similarity.
    chunk_size : int, default 100000
        Number of candidate pairs to accumulate before vectorized feature calculation.
    max_pairs : int, optional
        Maximum candidate pairs to process (for benchmarking / sampling).
    log_every : int, default 200000
        Logging interval for pair progress.

    Returns
    -------
    dict[str, Any]
        Summary metrics: total_pairs, elapsed_seconds, pairs_per_sec, peak_memory_mb.
    """
    cand_path = Path(candidate_pairs_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    is_parquet = out_path.suffix.lower() == ".parquet"
    writer = None

    # Trackers
    total_pairs = 0
    total_s1_processed = 0
    t_start = time.time()
    t_last_log = t_start

    # Chunk accumulators
    chunk_s1_ids: List[str] = []
    chunk_cand_ids: List[str] = []
    chunk_ranks: List[int] = []
    chunk_s1_names: List[str] = []
    chunk_cand_names: List[str] = []
    chunk_s1_addrs: List[str] = []
    chunk_cand_addrs: List[str] = []
    chunk_s1_countries: List[str] = []
    chunk_cand_countries: List[str] = []

    def _flush_chunk() -> None:
        nonlocal writer, chunk_s1_ids, chunk_cand_ids, chunk_ranks
        nonlocal chunk_s1_names, chunk_cand_names, chunk_s1_addrs, chunk_cand_addrs
        nonlocal chunk_s1_countries, chunk_cand_countries

        if not chunk_s1_ids:
            return

        chunk_df = compute_pairwise_features_chunk(
            s1_ids=chunk_s1_ids,
            cand_ids=chunk_cand_ids,
            ranks=np.array(chunk_ranks, dtype=np.int16),
            s1_names=chunk_s1_names,
            cand_names=chunk_cand_names,
            s1_addrs=chunk_s1_addrs,
            cand_addrs=chunk_cand_addrs,
            s1_countries=chunk_s1_countries,
            cand_countries=chunk_cand_countries,
            vectorizer=vectorizer,
        )

        if is_parquet:
            import pyarrow as pa
            import pyarrow.parquet as pq
            table = pa.Table.from_pandas(chunk_df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
            writer.write_table(table)
        else:
            # TSV output
            is_first = not out_path.is_file() or out_path.stat().st_size == 0
            chunk_df.to_csv(out_path, sep="\t", index=False, header=is_first, mode="a")

        # Clear chunk accumulators
        chunk_s1_ids.clear()
        chunk_cand_ids.clear()
        chunk_ranks.clear()
        chunk_s1_names.clear()
        chunk_cand_names.clear()
        chunk_s1_addrs.clear()
        chunk_cand_addrs.clear()
        chunk_s1_countries.clear()
        chunk_cand_countries.clear()

    logger.info("Streaming candidate pairs from %s...", cand_path)
    with open(cand_path, "r", encoding="utf-8") as f:
        header = f.readline()  # Skip header (source1_entity_id\tcandidate_entity_ids)

        for line in f:
            line_str = line.rstrip("\r\n")
            if not line_str:
                continue

            tab_pos = line_str.find("\t")
            if tab_pos == -1:
                continue

            s1_id = line_str[:tab_pos]
            cand_str = line_str[tab_pos + 1:]
            if not cand_str:
                continue

            s1_rec = entity_cache.get(s1_id)
            if s1_rec is None:
                # Entity not in cache (e.g. filtered out by country)
                continue

            s1_name, s1_addr, s1_country = s1_rec
            if country_filter and s1_country.lower() != country_filter.lower():
                continue

            cands = cand_str.split(",")
            total_s1_processed += 1

            for rank, cid in enumerate(cands):
                cand_rec = entity_cache.get(cid)
                if cand_rec is None:
                    continue

                c_name, c_addr, c_country = cand_rec

                chunk_s1_ids.append(s1_id)
                chunk_cand_ids.append(cid)
                chunk_ranks.append(rank)
                chunk_s1_names.append(s1_name)
                chunk_cand_names.append(c_name)
                chunk_s1_addrs.append(s1_addr)
                chunk_cand_addrs.append(c_addr)
                chunk_s1_countries.append(s1_country)
                chunk_cand_countries.append(c_country)

                total_pairs += 1

                if len(chunk_s1_ids) >= chunk_size:
                    _flush_chunk()

                if max_pairs is not None and total_pairs >= max_pairs:
                    break

            if total_pairs % log_every < len(cands) or time.time() - t_last_log >= 10.0:
                dt = time.time() - t_start
                qps = total_pairs / dt if dt > 0 else 0
                logger.info("  Processed %s pairs (S1=%s) — %.0f pairs/sec | Peak RAM: %.1f MB",
                            f"{total_pairs:,}", f"{total_s1_processed:,}", qps, get_peak_memory_mb())
                t_last_log = time.time()

            if max_pairs is not None and total_pairs >= max_pairs:
                break

    # Flush any remaining pairs in the last chunk
    _flush_chunk()

    if writer is not None:
        writer.close()

    elapsed = time.time() - t_start
    qps = total_pairs / elapsed if elapsed > 0 else 0
    peak_ram = get_peak_memory_mb()

    logger.info("Feature extraction complete: %s pairs in %.2fs (%.0f pairs/sec) | Peak RAM: %.1f MB",
                f"{total_pairs:,}", elapsed, qps, peak_ram)

    return {
        "total_pairs": total_pairs,
        "total_s1_entities": total_s1_processed,
        "elapsed_seconds": elapsed,
        "pairs_per_sec": qps,
        "peak_memory_mb": peak_ram,
        "output_path": str(out_path),
    }


# ── Milestone 2 Compatibility API ────────────────────────────────────

def build_feature_matrix(
    pairs: List[Tuple[str, str]],
    source1: pd.DataFrame,
    candidates: pd.DataFrame,
    vectorizer: Optional[TfidfVectorizer] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Build a feature matrix for a list of candidate pairs.

    Parameters
    ----------
    pairs : list[tuple[str, str]]
        ``(source1_id, candidate_id)`` pairs.
    source1 : pd.DataFrame
        Source 1 records (must have ``entity_id``, and ideally ``name_clean``,
        ``address_clean``, ``country``).
    candidates : pd.DataFrame
        Combined Source 2 + Source 3 records.
    vectorizer : TfidfVectorizer, optional

    Returns
    -------
    pair_df : pd.DataFrame
        DataFrame with entity IDs and feature columns.
    X : np.ndarray
        Numeric feature matrix (shape N x num_features) ready for model training.
    """
    if not pairs:
        pair_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
        X = np.empty((0, len(FEATURE_COLUMNS)), dtype=np.float32)
        return pair_df, X

    from business_entity_resolution.normalize import normalize_name, normalize_address

    # Build entity cache from provided DataFrames
    entity_cache: Dict[str, Tuple[str, str, str]] = {}
    for df in (source1, candidates):
        eids = df["entity_id"].values
        has_clean_n = "name_clean" in df.columns
        has_clean_a = "address_clean" in df.columns
        raw_names = df["business_name"].values
        raw_addrs = df["business_address"].values
        countries = [str(c) if pd.notna(c) else "" for c in df.get("country", [""] * len(df))]

        clean_names = df["name_clean"].values if has_clean_n else [
            normalize_name(n, country=c) for n, c in zip(raw_names, countries)
        ]
        clean_addrs = df["address_clean"].values if has_clean_a else [
            normalize_address(a, country=c) for a, c in zip(raw_addrs, countries)
        ]

        for i in range(len(eids)):
            entity_cache[eids[i]] = (str(clean_names[i]), str(clean_addrs[i]), countries[i])

    s1_ids = [p[0] for p in pairs]
    cand_ids = [p[1] for p in pairs]
    ranks = np.zeros(len(pairs), dtype=np.int16)

    s1_names = [entity_cache.get(sid, ("", "", ""))[0] for sid in s1_ids]
    cand_names = [entity_cache.get(cid, ("", "", ""))[0] for cid in cand_ids]
    s1_addrs = [entity_cache.get(sid, ("", "", ""))[1] for sid in s1_ids]
    cand_addrs = [entity_cache.get(cid, ("", "", ""))[1] for cid in cand_ids]
    s1_countries = [entity_cache.get(sid, ("", "", ""))[2] for sid in s1_ids]
    cand_countries = [entity_cache.get(cid, ("", "", ""))[2] for cid in cand_ids]

    pair_df = compute_pairwise_features_chunk(
        s1_ids=s1_ids,
        cand_ids=cand_ids,
        ranks=ranks,
        s1_names=s1_names,
        cand_names=cand_names,
        s1_addrs=s1_addrs,
        cand_addrs=cand_addrs,
        s1_countries=s1_countries,
        cand_countries=cand_countries,
        vectorizer=vectorizer,
    )

    X = pair_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    return pair_df, X


# ── CLI Interface ────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vectorized chunked feature extraction for entity resolution candidate pairs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Repository root path.",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="Path to candidate_pairs.tsv. Defaults to output/candidate_pairs.tsv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (.parquet or .tsv). Defaults to output/features_<country>.parquet.",
    )
    parser.add_argument(
        "--country",
        type=str,
        default=None,
        help="Optional country filter ('India', 'US'). Restricts processing to S1 entities from that country.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Batch size (number of candidate pairs) per chunk. Default: 100,000.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Maximum pairs to process (for sampling/benchmarking).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    root = args.root.resolve()
    cand_path = args.candidates or (root / "output" / "candidate_pairs.tsv")

    c_tag = args.country.lower() if args.country else "all"
    out_path = args.output or (root / "output" / f"features_{c_tag}.parquet")

    logger.info("Starting feature extraction: country=%s, chunk_size=%d, max_pairs=%s",
                args.country, args.chunk_size, args.max_pairs)

    entity_cache, vectorizer = load_and_normalize_entity_cache(
        root=root,
        country_filter=args.country,
    )

    stats = extract_features_streaming(
        candidate_pairs_path=cand_path,
        entity_cache=entity_cache,
        output_path=out_path,
        country_filter=args.country,
        vectorizer=vectorizer,
        chunk_size=args.chunk_size,
        max_pairs=args.max_pairs,
    )

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION BENCHMARK REPORT")
    print("=" * 60)
    print(f"Country Filter:        {args.country or 'ALL'}")
    print(f"Total Pairs Processed: {stats['total_pairs']:,}")
    print(f"S1 Entities Processed: {stats['total_s1_entities']:,}")
    print(f"Elapsed Time:          {stats['elapsed_seconds']:.2f} s")
    print(f"Processing Speed:      {stats['pairs_per_sec']:,.0f} pairs/sec")
    print(f"Peak Memory Usage:     {stats['peak_memory_mb']:.1f} MB")
    print(f"Output File:           {stats['output_path']}")

    total_440m = 440091505
    if stats["pairs_per_sec"] > 0:
        est_full_sec = total_440m / stats["pairs_per_sec"]
        print(f"Extrapolated Full 440M Runtime: {est_full_sec / 60.0:.1f} minutes ({est_full_sec / 3600.0:.2f} hours)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
