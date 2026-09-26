"""
blocking.py  –  High-performance candidate-pair generation for Entity Resolution.

Combines multiple complementary blocking signals:
1. **Country partitioning** – records from different countries are never paired.
2. **Exact normalized name index** – instant O(1) matching for identical clean names.
3. **Significant name-word index** – matches on distinctive entity words (pruned for frequency).
4. **Compound key index** – matches on (name_prefix_3, address_token) for high precision.
5. **Address token & bigram index** – matches on street/locality/PIN code tokens.
6. **Name-only fallback** – records with empty addresses are indexed by name and never dropped.
7. **Score-based candidate ranking & capping** – bounds candidates per entity to maintain
   extremely high reduction ratio while preserving recall.

All structures use fast in-memory inverted indices with integer row references
and frequency pruning, running in O(N) time and fitting easily in RAM.
"""

from __future__ import annotations

import gc
import heapq
import math
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from datasketch import MinHash, MinHashLSH

from business_entity_resolution.io_utils import (
    append_candidate_pairs,
    load_candidate_pairs,
    write_candidate_pairs,
)
from business_entity_resolution.normalize import normalize_address, normalize_name

logger = logging.getLogger(__name__)

# Legal and generic business words that should not serve as single-token blocking keys
NAME_STOP_WORDS: Set[str] = {
    "pvt", "ltd", "limited", "private", "corp", "corporation", "inc", "co",
    "company", "llc", "sa", "gmbh", "enterprises", "enterprise", "services",
    "service", "trading", "industries", "industry", "solutions", "solution",
    "group", "india", "stores", "store", "agency", "agencies", "associates",
    "associate", "international", "holdings", "holding", "tech", "technologies",
    "technology", "global", "systems", "system", "consulting", "consultancy",
    "ventures", "venture", "products", "product", "world", "india", "us",
    "usa", "national", "state", "city",
}

# US state codes for compound-key filtering (2-letter tokens in addresses)
_US_STATE_CODES: Set[str] = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga",
    "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md",
    "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
    "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc",
    "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy",
    "dc",
}

# Threshold for "distinctive" token: if a query has any name-word with
# posting list <= this size, we skip iterating larger lists entirely.
_SELECTIVE_EXPANSION_THRESHOLD = 5000

_LOG_EVERY = 500_000


# ── Shingling and MinHash helpers (for API & unit test compatibility) ────────

def _char_ngrams(text: str, n: int = 3) -> List[str]:
    """Extract character n-grams from *text*.

    Parameters
    ----------
    text : str
        Pre-normalised text.
    n : int
        N-gram size.

    Returns
    -------
    list[str]
        Character n-grams.  Returns the full string as a single
        element if it is shorter than *n*.
    """
    if len(text) < n:
        return [text] if text else []
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def _build_minhash(text: str, num_perm: int = 128, ngram_size: int = 3) -> MinHash:
    """Create a MinHash signature for *text*.

    Parameters
    ----------
    text : str
        Pre-normalised text string.
    num_perm : int
        Number of permutation functions.
    ngram_size : int
        Character n-gram size for shingling.

    Returns
    -------
    MinHash
    """
    m = MinHash(num_perm=num_perm)
    for ng in _char_ngrams(text, ngram_size):
        m.update(ng.encode("utf-8"))
    return m


def build_minhash_lsh_index(
    records: pd.DataFrame,
    text_col: str = "name_clean",
    id_col: str = "entity_id",
    num_perm: int = 128,
    threshold: float = 0.3,
    ngram_size: int = 3,
) -> Tuple[MinHashLSH, Dict[str, MinHash]]:
    """Build a MinHashLSH index over *records*.

    Parameters
    ----------
    records : pd.DataFrame
        Must contain *id_col* and *text_col*.
    text_col : str
        Column to shingle.
    id_col : str
        Column with unique entity identifiers.
    num_perm : int
        Number of permutation functions for MinHash.
    threshold : float
        Jaccard threshold for the LSH index.
    ngram_size : int
        Character n-gram size for shingling.

    Returns
    -------
    tuple[MinHashLSH, dict[str, MinHash]]
        The LSH index and a dict mapping entity_id → MinHash.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: Dict[str, MinHash] = {}
    total = len(records)

    eids = records[id_col].values
    texts = records[text_col].values

    for i in range(total):
        eid = eids[i]
        text_str = str(texts[i]) if texts[i] else ""
        mh = _build_minhash(text_str, num_perm, ngram_size)
        minhashes[eid] = mh
        try:
            lsh.insert(eid, mh, check_duplication=False)
        except ValueError:
            pass

        if (i + 1) % _LOG_EVERY == 0:
            logger.info("    LSH index: %d/%d inserted (%.0f%%)",
                        i + 1, total, 100 * (i + 1) / total)

    return lsh, minhashes


def query_candidates(
    lsh_index: MinHashLSH,
    query_records: pd.DataFrame,
    text_col: str = "name_clean",
    id_col: str = "entity_id",
    num_perm: int = 128,
    ngram_size: int = 3,
) -> Dict[str, List[str]]:
    """Query the LSH index for candidate matches.

    Parameters
    ----------
    lsh_index : MinHashLSH
    query_records : pd.DataFrame
    text_col : str
    id_col : str
    num_perm : int
    ngram_size : int

    Returns
    -------
    dict[str, list[str]]
        Mapping from query entity_id to list of candidate entity_ids.
    """
    results: Dict[str, List[str]] = {}
    total = len(query_records)

    eids = query_records[id_col].values
    texts = query_records[text_col].values

    for i in range(total):
        eid = eids[i]
        text_str = str(texts[i]) if texts[i] else ""
        mh = _build_minhash(text_str, num_perm, ngram_size)
        candidates = lsh_index.query(mh)
        results[eid] = candidates

        if (i + 1) % _LOG_EVERY == 0:
            logger.info("    LSH query: %d/%d (%.0f%%)",
                        i + 1, total, 100 * (i + 1) / total)

    return results


# ── Address token inverted index helpers ────────────────────────────

def _extract_significant_tokens(address: str, min_len: int = 3) -> List[str]:
    """Extract significant tokens from a normalized address.

    Preserves alphanumeric words of length >= *min_len* and numeric
    tokens (PIN/ZIP codes) of length >= 4.

    Parameters
    ----------
    address : str
        Pre-normalised address string.
    min_len : int
        Minimum token length to keep.

    Returns
    -------
    list[str]
    """
    if not address:
        return []
    return [
        t for t in address.split()
        if len(t) >= min_len and (not t.isdigit() or len(t) >= 4)
    ]


def build_address_token_index(
    records: pd.DataFrame,
    addr_col: str = "address_clean",
    id_col: str = "entity_id",
    min_token_len: int = 3,
    max_token_freq: int = 50000,
) -> Dict[str, Set[str]]:
    """Build an inverted index: address_token → set of entity_ids.

    Tokens that appear in more than *max_token_freq* records are dropped.

    Parameters
    ----------
    records : pd.DataFrame
    addr_col : str
    id_col : str
    min_token_len : int
    max_token_freq : int

    Returns
    -------
    dict[str, set[str]]
    """
    index: Dict[str, Set[str]] = defaultdict(set)

    eids = records[id_col].values
    addrs = records[addr_col].values

    for i in range(len(records)):
        addr = str(addrs[i]) if addrs[i] else ""
        eid = eids[i]
        for token in _extract_significant_tokens(addr, min_token_len):
            index[token].add(eid)

    pruned = {
        tok: cands for tok, cands in index.items()
        if len(cands) <= max_token_freq
    }
    n_pruned = len(index) - len(pruned)
    if n_pruned > 0:
        logger.info("    Pruned %d high-frequency address tokens (freq > %d)",
                    n_pruned, max_token_freq)

    return pruned


def query_address_candidates(
    addr_index: Dict[str, Set[str]],
    query_records: pd.DataFrame,
    addr_col: str = "address_clean",
    id_col: str = "entity_id",
    min_token_len: int = 3,
    min_shared_tokens: int = 1,
) -> Dict[str, Set[str]]:
    """Query the address token index for candidate matches.

    Parameters
    ----------
    addr_index : dict[str, set[str]]
    query_records : pd.DataFrame
    addr_col : str
    id_col : str
    min_token_len : int
    min_shared_tokens : int

    Returns
    -------
    dict[str, set[str]]
    """
    results: Dict[str, Set[str]] = {}

    eids = query_records[id_col].values
    addrs = query_records[addr_col].values

    for i in range(len(query_records)):
        eid = eids[i]
        addr = str(addrs[i]) if addrs[i] else ""
        tokens = _extract_significant_tokens(addr, min_token_len)
        if not tokens:
            results[eid] = set()
            continue

        if min_shared_tokens <= 1:
            candidates: Set[str] = set()
            for tok in tokens:
                if tok in addr_index:
                    candidates.update(addr_index[tok])
            results[eid] = candidates
        else:
            counts: Counter = Counter()
            for tok in tokens:
                if tok in addr_index:
                    counts.update(addr_index[tok])
            results[eid] = {c for c, cnt in counts.items() if cnt >= min_shared_tokens}

    return results


# ── Normalisation helper ────────────────────────────────────────────

def _normalize_source(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``name_clean`` and ``address_clean`` columns to a source DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have ``business_name``, ``business_address``, and optional ``country``.

    Returns
    -------
    pd.DataFrame
        Copy with two new normalized columns.
    """
    df = df.copy()
    countries = df["country"] if "country" in df.columns else [None] * len(df)

    df["name_clean"] = [
        normalize_name(n, country=c)
        for n, c in zip(df["business_name"], countries)
    ]
    df["address_clean"] = [
        normalize_address(a, country=c)
        for a, c in zip(df["business_address"], countries)
    ]
    return df


# ── High-performance candidate generator ───────────────────────────

def _build_partition_indices(
    ref_eids: List[str],
    ref_names: List[str],
    ref_addrs: List[str],
    name_enabled: bool = True,
    addr_enabled: bool = True,
    prefix_len: int = 3,
    hard_cap_name: int = 50000,
    hard_cap_addr: int = 5000,
    hard_cap_bigram: int = 5000,
    country: Optional[str] = None,
) -> Tuple[
    Dict[str, List[str]],
    Dict[str, List[str]],
    Dict[Tuple[str, str], List[str]],
    Dict[str, List[str]],
    Dict[Tuple[str, str], List[str]],
]:
    """Build fast inverted indices over the reference partition.

    Uses asymmetric frequency capping:
    - Name words: capped at hard_cap_name (50,000) so discriminative words
      like 'advanced', 'care', 'primary' are retained and softly down-weighted by IDF.
    - Address tokens: capped at hard_cap_addr (5,000) to keep posting lists tight.
    - Compound keys: (name_prefix_3, addr_token) includes 2-letter state codes
      and short digits (len >= 2).
    """
    name_exact_idx: Dict[str, List[str]] = defaultdict(list)
    name_word_idx: Dict[str, List[str]] = defaultdict(list)
    compound_idx: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    addr_token_idx: Dict[str, List[str]] = defaultdict(list)
    addr_bigram_idx: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    n_records = len(ref_eids)

    for i in range(n_records):
        eid = ref_eids[i]
        n = ref_names[i]
        a = ref_addrs[i]

        if name_enabled and n:
            name_exact_idx[n].append(eid)
            for w in n.split():
                if len(w) >= 3 and w not in NAME_STOP_WORDS:
                    name_word_idx[w].append(eid)

        if addr_enabled and a:
            a_words = a.split()
            toks = [t for t in a_words if len(t) >= 3 and (not t.isdigit() or len(t) >= 4)]
            for t in toks:
                addr_token_idx[t].append(eid)

            if name_enabled and n and len(n) >= prefix_len:
                pfx = n[:prefix_len]
                # Restrict 2-letter tokens to US state codes when country=US
                # to avoid combinatorial explosion from common short words
                # ("no", "of", "in", etc.). For other countries, keep all
                # tokens >= 3 chars (2-letter filtering is US-specific).
                if country is not None and country.upper() == "US":
                    compound_toks = [
                        t for t in a_words
                        if len(t) >= 3 or (len(t) == 2 and t in _US_STATE_CODES)
                    ]
                else:
                    compound_toks = [t for t in a_words if len(t) >= 3]
                for t in compound_toks:
                    compound_idx[(pfx, t)].append(eid)

            for j in range(len(a_words) - 1):
                w1, w2 = a_words[j], a_words[j + 1]
                if len(w1) >= 3 and len(w2) >= 3:
                    addr_bigram_idx[(w1, w2)].append(eid)

    # Asymmetric hard caps
    n_pruned = 0
    if hard_cap_name > 0:
        before = len(name_word_idx)
        name_word_idx = {k: v for k, v in name_word_idx.items() if len(v) <= hard_cap_name}
        n_pruned += before - len(name_word_idx)
    if hard_cap_addr > 0:
        before = len(addr_token_idx)
        addr_token_idx = {k: v for k, v in addr_token_idx.items() if len(v) <= hard_cap_addr}
        n_pruned += before - len(addr_token_idx)
        before_c = len(compound_idx)
        compound_idx = {k: v for k, v in compound_idx.items() if len(v) <= hard_cap_addr}
        n_pruned += before_c - len(compound_idx)
    if hard_cap_bigram > 0:
        before = len(addr_bigram_idx)
        addr_bigram_idx = {k: v for k, v in addr_bigram_idx.items() if len(v) <= hard_cap_bigram}
        n_pruned += before - len(addr_bigram_idx)
    if n_pruned > 0:
        logger.info("    Pruned frequent tokens (name > %d, addr > %d, total pruned: %d)",
                    hard_cap_name, hard_cap_addr, n_pruned)

    return (
        name_exact_idx,
        name_word_idx,
        compound_idx,
        addr_token_idx,
        addr_bigram_idx,
    )


def _query_partition_candidates(
    s1_eids: List[str],
    s1_names: List[str],
    s1_addrs: List[str],
    indices: Tuple[
        Dict[str, List[str]],
        Dict[str, List[str]],
        Dict[Tuple[str, str], List[str]],
        Dict[str, List[str]],
        Dict[Tuple[str, str], List[str]],
    ],
    n_ref: int = 1,
    name_enabled: bool = True,
    addr_enabled: bool = True,
    prefix_len: int = 3,
    max_candidates: int = 100,
    ref_lookup_name: Optional[Dict[str, str]] = None,
    ref_lookup_addr: Optional[Dict[str, str]] = None,
    country: Optional[str] = None,
) -> Dict[str, List[str]]:
    """Query candidates for each S1 entity in the partition.

    Scoring uses IDF-weighted contributions rather than flat integer
    increments, so common tokens still contribute a small positive
    score instead of being hard-excluded.
    """
    (
        name_exact_idx,
        name_word_idx,
        compound_idx,
        addr_token_idx,
        addr_bigram_idx,
    ) = indices

    # Pre-compute IDF weights for every token in the indices.
    # IDF = log2(N / freq) where N = total reference entities in partition.
    # Clamp minimum weight to 0.1 so even the most common token gives *something*.
    log_n = math.log2(max(n_ref, 2))

    name_word_idf: Dict[str, float] = {}
    for w, postings in name_word_idx.items():
        idf = max(log_n - math.log2(max(len(postings), 1)), 0.1)
        name_word_idf[w] = idf

    addr_token_idf: Dict[str, float] = {}
    for t, postings in addr_token_idx.items():
        idf = max(log_n - math.log2(max(len(postings), 1)), 0.1)
        addr_token_idf[t] = idf

    addr_bigram_idf: Dict[Tuple[str, str], float] = {}
    for bg, postings in addr_bigram_idx.items():
        idf = max(log_n - math.log2(max(len(postings), 1)), 0.1)
        addr_bigram_idf[bg] = idf

    results: Dict[str, List[str]] = {}
    n_queries = len(s1_eids)
    t_query_start = time.time()
    log_interval = min(max(n_queries // 10, 5000), 20000)

    for i in range(n_queries):
        eid = s1_eids[i]
        n = s1_names[i]
        a = s1_addrs[i]

        scores: Dict[str, float] = defaultdict(float)

        # 1. Exact name match — high fixed weight (always discriminative)
        if name_enabled and n and n in name_exact_idx:
            for c in name_exact_idx[n]:
                scores[c] += 10.0

        # 2. Significant name words — IDF-weighted (base weight 3.0 * idf)
        #    Selective expansion: only traverse large posting lists when no
        #    cheaper distinctive token is available in the query.
        if name_enabled and n:
            q_name_words = [
                w for w in set(n.split())
                if len(w) >= 3 and w not in NAME_STOP_WORDS and w in name_word_idx
            ]
            # Check if query has at least one distinctive (small) posting list
            has_distinctive = any(
                len(name_word_idx[w]) <= _SELECTIVE_EXPANSION_THRESHOLD
                for w in q_name_words
            )
            for w in q_name_words:
                posting_size = len(name_word_idx[w])
                # If we have a distinctive token, skip traversing huge lists
                if has_distinctive and posting_size > _SELECTIVE_EXPANSION_THRESHOLD:
                    continue
                w_idf = name_word_idf[w]
                for c in name_word_idx[w]:
                    scores[c] += 3.0 * w_idf

        # 3. Address features — IDF-weighted
        if addr_enabled and a:
            a_words = a.split()
            toks = {t for t in a_words if len(t) >= 3 and (not t.isdigit() or len(t) >= 4)}

            # Compound key: (name_prefix, addr_token)
            # Mirror the build-time filter for consistency
            if name_enabled and n and len(n) >= prefix_len:
                pfx = n[:prefix_len]
                if country is not None and country.upper() == "US":
                    compound_toks = {
                        t for t in a_words
                        if len(t) >= 3 or (len(t) == 2 and t in _US_STATE_CODES)
                    }
                else:
                    compound_toks = {t for t in a_words if len(t) >= 3}
                for t in compound_toks:
                    key = (pfx, t)
                    if key in compound_idx:
                        for c in compound_idx[key]:
                            scores[c] += 2.0

            # Address tokens — IDF-weighted (base weight 1.0 * idf)
            for t in toks:
                if t in addr_token_idx:
                    t_idf = addr_token_idf[t]
                    for c in addr_token_idx[t]:
                        scores[c] += 1.0 * t_idf

            # Address bigrams — IDF-weighted (base weight 2.0 * idf)
            if len(a_words) >= 2:
                bigrams = {
                    (a_words[j], a_words[j + 1])
                    for j in range(len(a_words) - 1)
                    if len(a_words[j]) >= 3 and len(a_words[j + 1]) >= 3
                }
                for bg in bigrams:
                    if bg in addr_bigram_idx:
                        bg_idf = addr_bigram_idf[bg]
                        for c in addr_bigram_idx[bg]:
                            scores[c] += 2.0 * bg_idf

        if not scores:
            results[eid] = []
        elif max_candidates is not None and len(scores) > max_candidates:
            if ref_lookup_name is not None and ref_lookup_addr is not None:
                # Stage 1: coarse pool from IDF scores
                pool_size = min(len(scores), max(max_candidates * 2, 1000))
                pool = heapq.nlargest(pool_size, scores, key=scores.get)

                # Stage 2: Jaccard re-scoring within pool
                n_tokens = set(n.split()) if n else set()
                a_tokens = set(a.split()) if a else set()

                fine_scores: Dict[str, float] = {}
                for c in pool:
                    c_n = ref_lookup_name.get(c, "")
                    c_a = ref_lookup_addr.get(c, "")

                    c_n_tokens = set(c_n.split()) if c_n else set()
                    c_a_tokens = set(c_a.split()) if c_a else set()

                    n_union = len(n_tokens | c_n_tokens)
                    a_union = len(a_tokens | c_a_tokens)
                    n_sim = len(n_tokens & c_n_tokens) / n_union if n_union else 0.0
                    a_sim = len(a_tokens & c_a_tokens) / a_union if a_union else 0.0

                    fine_scores[c] = scores[c] + (n_sim * 0.6) + (a_sim * 0.4)

                top = heapq.nlargest(max_candidates, fine_scores, key=fine_scores.get)
                results[eid] = sorted(top, key=lambda x: fine_scores[x], reverse=True)
            else:
                top = heapq.nlargest(max_candidates, scores, key=scores.get)
                results[eid] = sorted(top, key=lambda x: scores[x], reverse=True)
        else:
            results[eid] = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        if (i + 1) % log_interval == 0 or (i + 1) == n_queries:
            elapsed_q = time.time() - t_query_start
            qps = (i + 1) / max(elapsed_q, 0.001)
            logger.info("    Query progress: %d/%d (%.0f%%) — %.0f q/s",
                        i + 1, n_queries, 100 * (i + 1) / n_queries, qps)

    return results


def generate_candidate_pairs(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    blocking_cfg: Optional[Dict[str, Any]] = None,
    *,
    checkpoint_dir: Optional[Union[Path, str]] = None,
    output_path: Optional[Union[Path, str]] = None,
) -> Dict[str, List[str]]:
    """Generate all candidate pairs (source1_id → [candidate_ids]) via blocking.

    Features:
    1. Country partitioning: records from different countries are never paired.
       Open set: works on any country label (India, US, France, etc.).
    2. Multi-signal inverted index: exact name, distinctive words, compound keys,
       address tokens and bigrams.
    3. Empty-address handling: records with empty addresses are blocked via name.
    4. Candidate ranking and capping: keeps high recall with controlled candidate set.
    5. Memory efficient: partition-by-partition processing, low RAM footprint.
    6. Incremental checkpointing: saves each partition immediately upon completion
       to avoid work loss, and appends to output_path incrementally.

    Parameters
    ----------
    source1 : pd.DataFrame
    source2 : pd.DataFrame
    source3 : pd.DataFrame
    blocking_cfg : dict, optional
        Configuration dictionary (from blocking.yaml).
    checkpoint_dir : Path or str, optional
        Directory to read/write per-partition checkpoint TSVs
        (e.g., ``checkpoints/candidates_india.tsv``).
    output_path : Path or str, optional
        Target TSV path (e.g. ``output/candidate_pairs.tsv``) to write
        incrementally as each partition finishes.

    Returns
    -------
    dict[str, list[str]]
        Mapping ``{s1_id: [candidate_ids]}``. Every S1 entity has an entry.
    """
    t0 = time.time()
    if blocking_cfg is None:
        blocking_cfg = {}

    # ── Checkpointing setup ─────────────────────────────────────────
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    out_file = Path(output_path) if output_path is not None else None
    needs_header = True
    if out_file is not None:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        if out_file.is_file() and out_file.stat().st_size > 0:
            needs_header = False

    # ── Config parameters ───────────────────────────────────────────
    country_cfg = blocking_cfg.get("country_blocking", {})
    country_enabled = country_cfg.get("enabled", True)

    name_cfg = blocking_cfg.get("name_blocking", {})
    name_enabled = name_cfg.get("enabled", True)
    prefix_len = name_cfg.get("prefix_len", 3)

    addr_cfg = blocking_cfg.get("address_blocking", {})
    addr_enabled = addr_cfg.get("enabled", True)

    ranking_cfg = blocking_cfg.get("candidate_ranking", {})
    max_cands = ranking_cfg.get("max_candidates_per_entity", 100)
    hard_cap_name = ranking_cfg.get("hard_cap_name_token_freq", 50000)
    hard_cap_addr = ranking_cfg.get("hard_cap_addr_token_freq", 5000)
    hard_cap_bigram = ranking_cfg.get("hard_cap_bigram_freq", 5000)

    # ── Initialize candidate container for all S1 entities ──────────
    all_s1_eids = list(source1["entity_id"])
    all_candidates: Dict[str, List[str]] = {eid: [] for eid in all_s1_eids}

    # ── Determine partitions ────────────────────────────────────────
    has_country = (
        country_enabled
        and "country" in source1.columns
        and "country" in source2.columns
        and "country" in source3.columns
    )

    if has_country:
        raw_countries = source1["country"].dropna().unique()
        countries = sorted(str(c) for c in raw_countries)
        logger.info("Country partitioning enabled: %s", countries)
    else:
        countries = [None]
        logger.info("Country partitioning disabled: single partition")

    for country in countries:
        t_part = time.time()
        c_label = country if country is not None else "ALL"

        # Check for existing partition checkpoint
        ckpt_file = ckpt_dir / f"candidates_{c_label.lower()}.tsv" if ckpt_dir is not None else None
        if ckpt_file is not None and not ckpt_file.is_file():
            alt_ckpt = ckpt_dir / f"checkpoint_candidates_{c_label.lower()}.tsv"
            if alt_ckpt.is_file():
                ckpt_file = alt_ckpt

        if ckpt_file is not None and ckpt_file.is_file():
            logger.info("  [%s] Found checkpoint: %s — loading from disk instead of querying", c_label, ckpt_file)
            part_cands = load_candidate_pairs(ckpt_file)
            logger.info("  [%s] Loaded %d entities from checkpoint", c_label, len(part_cands))
            for eid, cands in part_cands.items():
                all_candidates[eid] = cands

            if out_file is not None and (not out_file.is_file() or out_file.stat().st_size == 0):
                append_candidate_pairs(part_cands, out_file, write_header=needs_header)
                needs_header = False

            continue

        # ── Slice partition ─────────────────────────────────────────
        if country is not None:
            s1_mask = source1["country"] == country
            s2_mask = source2["country"] == country
            s3_mask = source3["country"] == country
        else:
            s1_mask = pd.Series(True, index=source1.index)
            s2_mask = pd.Series(True, index=source2.index)
            s3_mask = pd.Series(True, index=source3.index)

        s1_part = source1.loc[s1_mask]
        s2_part = source2.loc[s2_mask]
        s3_part = source3.loc[s3_mask]

        if len(s1_part) == 0:
            continue

        ref_part = pd.concat([s2_part, s3_part], ignore_index=True)
        del s2_part, s3_part
        logger.info("  [%s] S1: %s rows, Ref (S2+S3): %s rows",
                    c_label, f"{len(s1_part):,}", f"{len(ref_part):,}")

        if len(ref_part) == 0:
            logger.info("  [%s] No reference records found — skipping partition", c_label)
            del ref_part
            continue

        # ── Normalise partition ─────────────────────────────────────
        t_norm = time.time()
        country_arg = country if country is not None else None
        s1_eids = s1_part["entity_id"].tolist()
        s1_names = [normalize_name(n, country=country_arg) for n in s1_part["business_name"]]
        s1_addrs = [normalize_address(a, country=country_arg) for a in s1_part["business_address"]]

        ref_eids = ref_part["entity_id"].tolist()
        ref_names = [normalize_name(n, country=country_arg) for n in ref_part["business_name"]]
        ref_addrs = [normalize_address(a, country=country_arg) for a in ref_part["business_address"]]

        del s1_part, ref_part
        gc.collect()
        logger.info("  [%s] Normalised in %.1fs", c_label, time.time() - t_norm)

        # ── Build inverted indices ──────────────────────────────────
        t_idx = time.time()
        n_ref_count = len(ref_eids)
        indices = _build_partition_indices(
            ref_eids=ref_eids,
            ref_names=ref_names,
            ref_addrs=ref_addrs,
            name_enabled=name_enabled,
            addr_enabled=addr_enabled,
            prefix_len=prefix_len,
            hard_cap_name=hard_cap_name,
            hard_cap_addr=hard_cap_addr,
            hard_cap_bigram=hard_cap_bigram,
            country=country,
        )
        logger.info("  [%s] Inverted indices built in %.1fs (name_exact=%d, name_word=%d, compound=%d, addr_token=%d)",
                    c_label, time.time() - t_idx,
                    len(indices[0]), len(indices[1]), len(indices[2]), len(indices[3]))

        # Keep ref_names and ref_addrs for post-retrieval re-scoring
        ref_lookup_name = {eid: n for eid, n in zip(ref_eids, ref_names)}
        ref_lookup_addr = {eid: a for eid, a in zip(ref_eids, ref_addrs)}
        del ref_eids, ref_names, ref_addrs
        gc.collect()

        # ── Query partition candidates ──────────────────────────────
        t_query = time.time()
        part_cands = _query_partition_candidates(
            s1_eids=s1_eids,
            s1_names=s1_names,
            s1_addrs=s1_addrs,
            indices=indices,
            n_ref=n_ref_count,
            name_enabled=name_enabled,
            addr_enabled=addr_enabled,
            prefix_len=prefix_len,
            max_candidates=max_cands,
            ref_lookup_name=ref_lookup_name,
            ref_lookup_addr=ref_lookup_addr,
            country=country,
        )
        t_query_done = time.time()
        part_duration = t_query_done - t_part
        logger.info("  [%s] Queried %d entities in %.1fs (%.0f/s) — total partition time: %.1fs (%.2fh)",
                    c_label, len(s1_eids), t_query_done - t_query,
                    len(s1_eids) / max(t_query_done - t_query, 0.001),
                    part_duration, part_duration / 3600.0)

        # ── 1. Checkpoint partition immediately to disk ─────────────
        if ckpt_dir is not None:
            save_ckpt = ckpt_dir / f"candidates_{c_label.lower()}.tsv"
            write_candidate_pairs(part_cands, save_ckpt)
            logger.info("  [%s] Checkpoint saved: %s (%d entities)",
                        c_label, save_ckpt, len(part_cands))

        # ── 2. Incrementally write/append to output candidate TSV ────
        if out_file is not None:
            append_candidate_pairs(part_cands, out_file, write_header=needs_header)
            needs_header = False
            logger.info("  [%s] Incrementally appended %d entities to %s",
                        c_label, len(part_cands), out_file)

        # ── 3. Merge partition results in memory ─────────────────────
        for eid, cands in part_cands.items():
            all_candidates[eid] = cands

        del indices, part_cands, s1_eids, s1_names, s1_addrs, ref_lookup_name, ref_lookup_addr
        gc.collect()
        logger.info("  [%s] Partition completed in %.1fs", c_label, time.time() - t_part)

    elapsed = time.time() - t0
    n_pairs = sum(len(v) for v in all_candidates.values())
    n_with = sum(1 for v in all_candidates.values() if v)
    mean_c = n_pairs / len(all_candidates) if all_candidates else 0

    logger.info(
        "Blocking complete: %d S1 entities, %d total candidate pairs, "
        "%.1f mean candidates/entity, %d S1 with >=1 candidate, %.1fs elapsed",
        len(all_candidates), n_pairs, mean_c, n_with, elapsed,
    )

    return all_candidates
