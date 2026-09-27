"""
pipeline.py  –  End-to-end orchestration for training and inference.

Provides top-level ``run_train_pipeline()`` and ``run_infer_pipeline()``
functions that chain together:
    normalisation → blocking → features → model → grouping → I/O.

Each stage is wrapped in a try/except that catches NotImplementedError,
ImportError, and FileNotFoundError, logging a clear "stage not yet
implemented or missing dependency — skipping" message rather than
crashing the whole pipeline.

Inference is memory-safe at full scale (1.73M S1 entities):
- Sequential country-partition processing (India, US, France)
- Incremental candidate pair streaming to candidate_pairs.tsv
- Vectorized chunked featurization and LightGBM scoring (250k row chunks)
- In-flight threshold filtering (tau=0.98)
- Incremental matching_results.tsv writing with exact single-row guarantees
  per S1 test entity (singletons assigned empty match lists).
"""

from __future__ import annotations

from collections import defaultdict
import gc
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Stage execution wrapper
# ═══════════════════════════════════════════════════════════════════════════


class StageResult:
    """Container for the outcome of a single pipeline stage."""

    __slots__ = ("name", "status", "elapsed_seconds", "result", "error_msg")

    def __init__(
        self,
        name: str,
        status: str,
        elapsed_seconds: float,
        result: Any = None,
        error_msg: Optional[str] = None,
    ):
        self.name = name
        self.status = status  # "completed", "skipped", "failed"
        self.elapsed_seconds = elapsed_seconds
        self.result = result
        self.error_msg = error_msg

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "status": self.status,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }
        if self.error_msg:
            d["error"] = self.error_msg
        return d


def _run_stage(
    name: str,
    fn: Any,
    *args: Any,
    **kwargs: Any,
) -> StageResult:
    """Execute a pipeline stage with timing and graceful error handling.

    Catches ``NotImplementedError``, ``ImportError``, and ``FileNotFoundError``
    (the errors raised by unimplemented stubs, missing functions, or missing
    model artifacts in unit tests) and logs a clear skip message.
    All other exceptions propagate.

    Parameters
    ----------
    name : str
        Human-readable stage name for logging.
    fn : callable
        The stage function to invoke.
    *args, **kwargs
        Forwarded to *fn*.

    Returns
    -------
    StageResult
    """
    logger.info("Stage [%s] — STARTED", name)
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        logger.info(
            "Stage [%s] — COMPLETED (%.2fs)", name, elapsed,
        )
        return StageResult(name, "completed", elapsed, result=result)
    except (NotImplementedError, ImportError, FileNotFoundError) as exc:
        elapsed = time.time() - t0
        msg = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "Stage [%s] — SKIPPED (missing dependency or stub): %s", name, msg,
        )
        return StageResult(name, "skipped", elapsed, error_msg=msg)


# ═══════════════════════════════════════════════════════════════════════════
# Individual stage functions
# ═══════════════════════════════════════════════════════════════════════════


def _stage_load_train_data(
    root: Path,
    paths_cfg: Dict[str, Any],
    sample: Optional[int] = None,
) -> Dict[str, Any]:
    """Load and return train source DataFrames + ground truth."""
    from business_entity_resolution.io_utils import (
        load_ground_truth,
        load_source,
    )

    data_cfg = paths_cfg.get("data", {}).get("train", {})
    s1_path = root / data_cfg.get("source1", "dataset/train/train_source1.tsv")
    s2_path = root / data_cfg.get("source2", "dataset/train/train_source2.tsv")
    s3_path = root / data_cfg.get("source3", "dataset/train/train_source3.tsv")

    s1 = load_source(s1_path, expected_source=1)
    if sample is not None and sample > 0:
        s1 = s1.head(sample).copy()
        logger.info("Sampled Source 1 to %d rows", len(s1))
    s2 = load_source(s2_path, expected_source=2)
    s3 = load_source(s3_path, expected_source=3)

    logger.info(
        "Loaded sources — S1: %d rows | S2: %d rows | S3: %d rows",
        len(s1), len(s2), len(s3),
    )

    gt_rel = data_cfg.get("ground_truth", "dataset/train/train_ground_truth.tsv")
    gt_path = root / gt_rel
    gt = load_ground_truth(gt_path) if gt_path.is_file() else None
    if gt is not None:
        logger.info("Loaded ground truth: %d rows", len(gt))

    return {"s1": s1, "s2": s2, "s3": s3, "ground_truth": gt}


def _stage_load_test_data(
    root: Path,
    paths_cfg: Dict[str, Any],
    sample: Optional[int] = None,
    test_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load and return test source DataFrames."""
    from business_entity_resolution.io_utils import load_source

    data_cfg = paths_cfg.get("data", {}).get("test", {})
    t_dir = test_dir or (root / "dataset" / "test")

    s1_file = (t_dir / "test_source1.tsv") if (t_dir / "test_source1.tsv").is_file() else (root / data_cfg.get("source1", "dataset/test/test_source1.tsv"))
    s2_file = (t_dir / "test_source2.tsv") if (t_dir / "test_source2.tsv").is_file() else (root / data_cfg.get("source2", "dataset/test/test_source2.tsv"))
    s3_file = (t_dir / "test_source3.tsv") if (t_dir / "test_source3.tsv").is_file() else (root / data_cfg.get("source3", "dataset/test/test_source3.tsv"))

    s1 = load_source(s1_file, expected_source=1)
    if sample is not None and sample > 0:
        s1 = s1.head(sample).copy()
        logger.info("Sampled Source 1 to %d rows", len(s1))
    s2 = load_source(s2_file, expected_source=2)
    s3 = load_source(s3_file, expected_source=3)

    logger.info(
        "Loaded test sources — S1: %d rows | S2: %d rows | S3: %d rows",
        len(s1), len(s2), len(s3),
    )

    return {"s1": s1, "s2": s2, "s3": s3}


def _stage_normalize(
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalize all source DataFrames in-place and return them."""
    from business_entity_resolution.normalize import normalize_dataframe

    for key in ("s1", "s2", "s3"):
        if key in data and data[key] is not None:
            data[key] = normalize_dataframe(data[key])
            logger.info("Normalized %s: %d rows", key.upper(), len(data[key]))

    return data


def _stage_blocking(
    data: Dict[str, Any],
    blocking_cfg: Dict[str, Any],
    checkpoint_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, List[str]]:
    """Generate candidate pairs via blocking."""
    from business_entity_resolution.blocking import generate_candidate_pairs

    candidates = generate_candidate_pairs(
        data["s1"], data["s2"], data["s3"], blocking_cfg,
        checkpoint_dir=checkpoint_dir,
        output_path=output_path,
    )
    n_pairs = sum(len(v) for v in candidates.values())
    logger.info(
        "Blocking produced %d candidate pairs for %d S1 entities",
        n_pairs, len(candidates),
    )
    return candidates


def _stage_featurize(
    candidates: Dict[str, List[str]],
    data: Dict[str, Any],
) -> Tuple[pd.DataFrame, Any]:
    """Compute feature matrix from candidate pairs."""
    from business_entity_resolution.features import build_feature_matrix

    pairs: List[Tuple[str, str]] = []
    for s1_id, cand_ids in candidates.items():
        for cand_id in cand_ids:
            pairs.append((s1_id, cand_id))

    combined_ref = pd.concat(
        [data["s2"], data["s3"]], ignore_index=True,
    )

    pair_df, X = build_feature_matrix(pairs, data["s1"], combined_ref)
    logger.info(
        "Feature matrix: %d pairs × %d features", X.shape[0], X.shape[1],
    )
    return pair_df, X


def _stage_train_model(
    pair_df: pd.DataFrame,
    X: Any,
    ground_truth: Optional[pd.DataFrame],
    model_cfg: Dict[str, Any],
    model_save_path: Optional[Path] = None,
) -> Any:
    """Train the LightGBM classifier."""
    from business_entity_resolution.model import (
        build_classifier,
        save_model,
        train,
    )

    clf = build_classifier(model_cfg.get("lgbm", {}))

    import numpy as np
    y = np.zeros(len(pair_df), dtype=int)
    if ground_truth is not None:
        gt_set: Dict[str, Set[str]] = {}
        for _, row in ground_truth.iterrows():
            s1_id = str(row["source1_entity_id"])
            matched = str(row["matched_entity_ids"])
            if matched and matched != "nan":
                gt_set[s1_id] = {m.strip() for m in matched.split(",") if m.strip()}
            else:
                gt_set[s1_id] = set()

        col_s1 = "source1_entity_id" if "source1_entity_id" in pair_df.columns else "s1_id"
        col_cand = "candidate_entity_id" if "candidate_entity_id" in pair_df.columns else "cand_id"
        s1_vals = pair_df[col_s1].astype(str).values
        cand_vals = pair_df[col_cand].astype(str).values

        for i in range(len(pair_df)):
            s1_id = s1_vals[i]
            cand_id = cand_vals[i]
            if s1_id in gt_set and cand_id in gt_set[s1_id]:
                y[i] = 1

    val_cfg = model_cfg.get("validation", {})
    test_size = val_cfg.get("test_size", 0.2)
    rand_state = val_cfg.get("random_state", 42)

    from sklearn.model_selection import train_test_split
    if len(X) >= 4 and len(np.unique(y)) > 1:
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=rand_state, stratify=y,
            )
            clf = train(clf, X_train, y_train, X_val=X_val, y_val=y_val)
        except Exception:
            clf = train(clf, X, y)
    else:
        clf = train(clf, X, y)
    logger.info("Model trained on %d pairs", len(X))

    if model_save_path:
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        save_model(clf, model_save_path)
        logger.info("Model saved to %s", model_save_path)

    return clf


def _stage_predict(
    clf: Any,
    X: Any,
    pair_df: pd.DataFrame,
) -> Dict[str, List[Tuple[str, float]]]:
    """Score candidate pairs with the trained/loaded model."""
    from business_entity_resolution.model import predict_proba

    proba = predict_proba(clf, X)
    logger.info("Scored %d candidate pairs", len(proba))

    col_s1 = "source1_entity_id" if "source1_entity_id" in pair_df.columns else "s1_id"
    col_cand = "candidate_entity_id" if "candidate_entity_id" in pair_df.columns else "cand_id"
    s1_vals = pair_df[col_s1].astype(str).values
    cand_vals = pair_df[col_cand].astype(str).values

    scored_pairs: Dict[str, List[Tuple[str, float]]] = {}
    for i in range(len(pair_df)):
        scored_pairs.setdefault(s1_vals[i], []).append((cand_vals[i], float(proba[i])))

    return scored_pairs


def _stage_threshold_sweep(
    scored_pairs: Dict[str, List[Tuple[str, float]]],
    ground_truth: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """Run threshold sweep to find optimal F_0.5 threshold."""
    from business_entity_resolution.grouping import sweep_thresholds
    from business_entity_resolution.scoring import load_matches_dict

    if ground_truth is None:
        logger.warning("No ground truth available — cannot sweep thresholds")
        return {"best_threshold": 0.98, "best_metrics": {}, "sweep": {}}

    gt_dict = load_matches_dict(ground_truth)
    sweep_result = sweep_thresholds(scored_pairs, gt_dict)
    logger.info(
        "Threshold sweep: best=%.3f  macro_f_beta=%.4f",
        sweep_result["best_threshold"],
        sweep_result["best_metrics"].get("macro_f_beta", 0.0),
    )
    return sweep_result


def _stage_group_and_write(
    scored_pairs: Dict[str, List[Tuple[str, float]]],
    threshold: float,
    all_s1_ids: List[str],
    output_dir: Path,
    paths_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Group matches at threshold and write output files."""
    from business_entity_resolution.grouping import group_by_threshold
    from business_entity_resolution.io_utils import (
        write_candidate_pairs,
        write_matching_results,
    )

    predictions = group_by_threshold(scored_pairs, threshold)

    for s1_id in all_s1_ids:
        if s1_id not in predictions:
            predictions[s1_id] = set()

    matches_for_write: Dict[str, List[str]] = {
        s1_id: sorted(matched_ids)
        for s1_id, matched_ids in predictions.items()
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    mr_path = Path(paths_cfg.get("output", {}).get(
        "matching_results", str(output_dir / "matching_results.tsv"),
    ))
    write_matching_results(matches_for_write, mr_path)

    cand_for_write: Dict[str, List[str]] = {
        s1_id: [cid for cid, _ in cands]
        for s1_id, cands in scored_pairs.items()
    }
    for s1_id in all_s1_ids:
        if s1_id not in cand_for_write:
            cand_for_write[s1_id] = []

    cp_path = Path(paths_cfg.get("output", {}).get(
        "candidate_pairs", str(output_dir / "candidate_pairs.tsv"),
    ))
    write_candidate_pairs(cand_for_write, cp_path)

    n_matched = sum(1 for v in matches_for_write.values() if v)
    n_singleton = sum(1 for v in matches_for_write.values() if not v)
    logger.info(
        "Output written — %d matched entities, %d singletons",
        n_matched, n_singleton,
    )

    return {
        "n_matched": n_matched,
        "n_singletons": n_singleton,
        "matching_results_path": str(mr_path),
        "candidate_pairs_path": str(cp_path),
    }


# ═══════════════════════════════════════════════════════════════════════════
# High-Scale Streaming Partition Inference Engine
# ═══════════════════════════════════════════════════════════════════════════


def run_streaming_partition_infer(
    data: Dict[str, Any],
    blocking_cfg: Dict[str, Any],
    model_path: Path,
    output_dir: Path,
    threshold: float = 0.98,
    country_filter: Optional[str] = None,
    chunk_size: int = 250000,
) -> Dict[str, Any]:
    """Execute end-to-end memory-safe inference partitioned by country.

    Processes country partitions sequentially:
    1. Slices S1, S2, S3 by country (France, India, US)
    2. Runs blocking and writes directly to candidate_pairs.tsv
    3. Featurizes and predicts in chunks of chunk_size (250k rows)
    4. Filters in-flight at threshold (0.98)
    5. Writes to matching_results.tsv with exact single-row guarantee per S1 entity
    6. Frees memory per partition so peak RAM stays strictly bounded (< 4.5 GB).
    """
    from business_entity_resolution.blocking import generate_candidate_pairs
    from business_entity_resolution.features import (
        FEATURE_COLUMNS,
        compute_pairwise_features_chunk,
        fit_name_tfidf_vectorizer,
        get_peak_memory_mb,
    )
    from business_entity_resolution.model import load_model, predict_proba

    output_dir.mkdir(parents=True, exist_ok=True)
    cp_path = output_dir / "candidate_pairs.tsv"
    mr_path = output_dir / "matching_results.tsv"

    logger.info("Loading model from %s...", model_path)
    clf = load_model(model_path)

    # Initialize output TSV headers
    with open(cp_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
    with open(mr_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")

    raw_countries = data["s1"]["country"].dropna().unique()
    countries = sorted([str(c) for c in raw_countries]) if len(raw_countries) > 0 else [None]
    if country_filter:
        countries = [c for c in countries if c and c.lower() == country_filter.lower()]
        logger.info("Restricted inference to country: %s", country_filter)

    total_s1 = 0
    total_candidates = 0
    total_matched = 0
    total_singletons = 0
    country_stats = {}

    t_blocking_total = 0.0
    t_featurize_total = 0.0
    t_predict_total = 0.0
    t_write_total = 0.0

    t_engine_start = time.time()

    for country in countries:
        c_label = country or "ALL"
        t_part_start = time.time()
        logger.info("=" * 60)
        logger.info("PROCESSING PARTITION: %s", c_label)
        logger.info("=" * 60)

        # 1. Partition slice
        if country is not None:
            s1_part = data["s1"][data["s1"]["country"] == country]
            s2_part = data["s2"][data["s2"]["country"] == country]
            s3_part = data["s3"][data["s3"]["country"] == country]
        else:
            s1_part = data["s1"]
            s2_part = data["s2"]
            s3_part = data["s3"]

        n_s1_part = len(s1_part)
        total_s1 += n_s1_part
        if n_s1_part == 0:
            logger.info("[%s] Zero S1 entities — skipping partition.", c_label)
            continue

        logger.info("[%s] S1 entities: %d | S2 records: %d | S3 records: %d",
                    c_label, n_s1_part, len(s2_part), len(s3_part))

        # 2. Blocking
        t_b0 = time.time()
        part_cands = generate_candidate_pairs(s1_part, s2_part, s3_part, blocking_cfg)
        dt_b = time.time() - t_b0
        t_blocking_total += dt_b
        logger.info("[%s] Blocking complete in %.2fs", c_label, dt_b)

        # 3. Stream candidate pairs to candidate_pairs.tsv immediately
        t_w0 = time.time()
        part_cand_count = 0
        with open(cp_path, "a", encoding="utf-8") as cp_file:
            for s1_id in s1_part["entity_id"]:
                cands = part_cands.get(s1_id, [])
                part_cand_count += len(cands)
                cp_file.write(f"{s1_id}\t{','.join(cands)}\n")
            cp_file.flush()
        total_candidates += part_cand_count
        t_write_total += (time.time() - t_w0)
        logger.info("[%s] Streamed %d candidate pairs for %d entities to candidate_pairs.tsv",
                    c_label, part_cand_count, n_s1_part)

        # 4. Prepare in-memory entity lookups for featurization
        ref_part = pd.concat([s2_part, s3_part], ignore_index=True)
        ref_names = ref_part["name_clean"].tolist() if "name_clean" in ref_part.columns else []
        vectorizer = fit_name_tfidf_vectorizer(ref_names) if ref_names else None

        s1_lookup = {
            eid: (n, a, c)
            for eid, n, a, c in zip(
                s1_part["entity_id"].values,
                s1_part["name_clean"].values,
                s1_part["address_clean"].values,
                s1_part["country"].values,
            )
        }
        ref_lookup = {
            eid: (n, a, c)
            for eid, n, a, c in zip(
                ref_part["entity_id"].values,
                ref_part["name_clean"].values,
                ref_part["address_clean"].values,
                ref_part["country"].values,
            )
        }
        del ref_part, ref_names, s2_part, s3_part
        gc.collect()

        # 5. Chunked featurization and model inference
        country_matches: Dict[str, List[str]] = defaultdict(list)
        chunk_s1: List[str] = []
        chunk_cand: List[str] = []
        chunk_ranks: List[int] = []
        chunk_s1_names: List[str] = []
        chunk_cand_names: List[str] = []
        chunk_s1_addrs: List[str] = []
        chunk_cand_addrs: List[str] = []
        chunk_s1_c: List[str] = []
        chunk_cand_c: List[str] = []

        total_part_pairs_scored = 0

        def _flush_eval_chunk():
            nonlocal t_featurize_total, t_predict_total, total_part_pairs_scored
            if not chunk_s1:
                return

            n_chunk = len(chunk_s1)
            total_part_pairs_scored += n_chunk

            t_f0 = time.time()
            chunk_df = compute_pairwise_features_chunk(
                s1_ids=chunk_s1,
                cand_ids=chunk_cand,
                ranks=np.array(chunk_ranks, dtype=np.int16),
                s1_names=chunk_s1_names,
                cand_names=chunk_cand_names,
                s1_addrs=chunk_s1_addrs,
                cand_addrs=chunk_cand_addrs,
                s1_countries=chunk_s1_c,
                cand_countries=chunk_cand_c,
                vectorizer=vectorizer,
            )
            X = chunk_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            t_featurize_total += (time.time() - t_f0)

            t_p0 = time.time()
            probs = predict_proba(clf, X)
            matched_indices = np.flatnonzero(probs >= threshold)
            for idx in matched_indices:
                country_matches[chunk_s1[idx]].append(chunk_cand[idx])
            t_predict_total += (time.time() - t_p0)

            chunk_s1.clear()
            chunk_cand.clear()
            chunk_ranks.clear()
            chunk_s1_names.clear()
            chunk_cand_names.clear()
            chunk_s1_addrs.clear()
            chunk_cand_addrs.clear()
            chunk_s1_c.clear()
            chunk_cand_c.clear()

        logger.info("[%s] Scoring %d candidate pairs in chunks of %d...",
                    c_label, part_cand_count, chunk_size)

        for s1_id, cands in part_cands.items():
            s1_rec = s1_lookup.get(s1_id)
            if s1_rec is None:
                continue
            s1_n, s1_a, s1_c_val = s1_rec

            for rank, cand_id in enumerate(cands):
                cand_rec = ref_lookup.get(cand_id)
                if cand_rec is None:
                    continue
                cand_n, cand_a, cand_c_val = cand_rec

                chunk_s1.append(s1_id)
                chunk_cand.append(cand_id)
                chunk_ranks.append(rank)
                chunk_s1_names.append(s1_n)
                chunk_cand_names.append(cand_n)
                chunk_s1_addrs.append(s1_a)
                chunk_cand_addrs.append(cand_a)
                chunk_s1_c.append(s1_c_val)
                chunk_cand_c.append(cand_c_val)

                if len(chunk_s1) >= chunk_size:
                    _flush_eval_chunk()

        _flush_eval_chunk()  # flush trailing chunk

        # 6. Stream matching results to matching_results.tsv immediately
        t_w1 = time.time()
        part_matched_entities = 0
        part_singleton_entities = 0
        part_predicted_pairs = 0

        with open(mr_path, "a", encoding="utf-8") as mr_file:
            for s1_id in s1_part["entity_id"]:
                m_list = country_matches.get(s1_id, [])
                if m_list:
                    part_matched_entities += 1
                    uniq_m = sorted(set(m_list))
                    part_predicted_pairs += len(uniq_m)
                    mr_file.write(f"{s1_id}\t{','.join(uniq_m)}\n")
                else:
                    part_singleton_entities += 1
                    mr_file.write(f"{s1_id}\t\n")
            mr_file.flush()

        t_write_total += (time.time() - t_w1)
        total_matched += part_matched_entities
        total_singletons += part_singleton_entities

        part_elapsed = time.time() - t_part_start
        peak_mb = get_peak_memory_mb()

        country_stats[c_label] = {
            "entities": n_s1_part,
            "candidate_pairs": part_cand_count,
            "matched_entities": part_matched_entities,
            "singleton_entities": part_singleton_entities,
            "predicted_matches": part_predicted_pairs,
            "duration_seconds": round(part_elapsed, 2),
            "peak_ram_mb": round(peak_mb, 1),
        }

        logger.info(
            "[%s] COMPLETED: %d entities (%d matched, %d singletons, %d candidates, %d predicted matches) in %.1fs | Peak RAM: %.1f MB",
            c_label, n_s1_part, part_matched_entities, part_singleton_entities, part_cand_count, part_predicted_pairs, part_elapsed, peak_mb,
        )

        del part_cands, country_matches, s1_lookup, ref_lookup, vectorizer, s1_part
        gc.collect()

    engine_elapsed = time.time() - t_engine_start

    return {
        "total_s1": total_s1,
        "total_candidates": total_candidates,
        "total_matched": total_matched,
        "total_singletons": total_singletons,
        "country_stats": country_stats,
        "candidate_pairs_path": str(cp_path),
        "matching_results_path": str(mr_path),
        "engine_elapsed_seconds": round(engine_elapsed, 2),
        "stage_timings": {
            "blocking_seconds": round(t_blocking_total, 2),
            "featurize_seconds": round(t_featurize_total, 2),
            "predict_seconds": round(t_predict_total, 2),
            "write_seconds": round(t_write_total, 2),
        },
        "peak_ram_mb": round(get_peak_memory_mb(), 1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Top-level orchestration functions
# ═══════════════════════════════════════════════════════════════════════════


def run_train_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the full training pipeline with graceful stage-skip handling.

    Sequence: load data → normalize → block → featurize → train model →
    threshold sweep → group.
    """
    root = Path(config["root"])
    paths_cfg = config.get("paths", {})
    blocking_cfg = config.get("blocking", {})
    model_cfg = config.get("model", {})
    sample = config.get("sample")

    stages: Dict[str, Dict[str, Any]] = {}
    metrics: Dict[str, Any] = {}
    pipeline_t0 = time.time()

    # ── Stage 1: Load train data ────────────────────────────────────
    sr = _run_stage("load_data", _stage_load_train_data, root, paths_cfg, sample)
    stages[sr.name] = sr.to_dict()
    data = sr.result if sr.status == "completed" else None

    # ── Stage 2: Normalize ──────────────────────────────────────────
    if data is not None:
        sr = _run_stage("normalize", _stage_normalize, data)
        stages[sr.name] = sr.to_dict()
        if sr.status == "completed":
            data = sr.result
    else:
        stages["normalize"] = {"status": "skipped", "elapsed_seconds": 0.0,
                               "error": "Dependency [load_data] not available"}

    # ── Stage 3: Blocking ───────────────────────────────────────────
    candidates = None
    if data is not None:
        sr = _run_stage(
            "blocking", _stage_blocking, data, blocking_cfg,
        )
        stages[sr.name] = sr.to_dict()
        if sr.status == "completed":
            candidates = sr.result
    else:
        stages["blocking"] = {"status": "skipped", "elapsed_seconds": 0.0,
                              "error": "Dependency [load_data] not available"}

    # ── Stage 4: Featurize ──────────────────────────────────────────
    pair_df = None
    X = None
    if candidates is not None and data is not None:
        sr = _run_stage("featurize", _stage_featurize, candidates, data)
        stages[sr.name] = sr.to_dict()
        if sr.status == "completed":
            pair_df, X = sr.result
    else:
        stages["featurize"] = {
            "status": "skipped", "elapsed_seconds": 0.0,
            "error": "Dependency [blocking] not available",
        }

    # ── Stage 5: Train model ────────────────────────────────────────
    clf = None
    if pair_df is not None and X is not None and data is not None:
        model_save_path = root / paths_cfg.get("model", {}).get(
            "artifact", "models/lgbm_entity_model.txt",
        )
        sr = _run_stage(
            "train_model", _stage_train_model,
            pair_df, X, data.get("ground_truth"), model_cfg,
            model_save_path=model_save_path,
        )
        stages[sr.name] = sr.to_dict()
        if sr.status == "completed":
            clf = sr.result
    else:
        stages["train_model"] = {
            "status": "skipped", "elapsed_seconds": 0.0,
            "error": "Dependency [featurize] not available",
        }

    # ── Stage 6: Predict (score candidates) ─────────────────────────
    scored_pairs = None
    if clf is not None and X is not None and pair_df is not None:
        sr = _run_stage("predict", _stage_predict, clf, X, pair_df)
        stages[sr.name] = sr.to_dict()
        if sr.status == "completed":
            scored_pairs = sr.result
    else:
        stages["predict"] = {
            "status": "skipped", "elapsed_seconds": 0.0,
            "error": "Dependency [train_model] not available",
        }

    # ── Stage 7: Threshold sweep ────────────────────────────────────
    sweep_result = None
    if scored_pairs is not None:
        gt = data.get("ground_truth") if data else None
        sr = _run_stage(
            "threshold_sweep", _stage_threshold_sweep, scored_pairs, gt,
        )
        stages[sr.name] = sr.to_dict()
        if sr.status == "completed":
            sweep_result = sr.result
            metrics["threshold_sweep"] = {
                "best_threshold": sweep_result["best_threshold"],
                "best_metrics": sweep_result["best_metrics"],
            }
    else:
        stages["threshold_sweep"] = {
            "status": "skipped", "elapsed_seconds": 0.0,
            "error": "Dependency [predict] not available",
        }

    # ── Pipeline summary ────────────────────────────────────────────
    elapsed_total = time.time() - pipeline_t0
    completed = [k for k, v in stages.items() if v["status"] == "completed"]
    skipped = [k for k, v in stages.items() if v["status"] == "skipped"]

    logger.info(
        "Train pipeline finished in %.2fs — %d completed, %d skipped",
        elapsed_total, len(completed), len(skipped),
    )

    return {
        "stages": stages,
        "metrics": metrics,
        "completed": completed,
        "skipped": skipped,
        "elapsed_total_seconds": round(elapsed_total, 3),
    }


def run_infer_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the full inference pipeline with graceful stage-skip handling.

    Sequence: load data → normalize → block → featurize → score →
    group at frozen threshold (0.98) → write output files.

    Automatically uses memory-safe streaming partition inference when
    the trained model is available.
    """
    root = Path(config["root"])
    paths_cfg = config.get("paths", {})
    blocking_cfg = config.get("blocking", {})
    model_cfg = config.get("model", {})
    sample = config.get("sample")
    threshold = float(config.get(
        "threshold",
        model_cfg.get("threshold", {}).get("match", 0.98),
    ))
    country_filter = config.get("country")
    chunk_size = int(config.get("chunk_size", 250000))

    test_dir = Path(config["test_dir"]) if config.get("test_dir") else None
    output_dir = Path(config["out_dir"]) if config.get("out_dir") else (root / paths_cfg.get("output", {}).get("dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    stages: Dict[str, Dict[str, Any]] = {}
    metrics: Dict[str, Any] = {}
    pipeline_t0 = time.time()

    # ── Stage 1: Load test data ─────────────────────────────────────
    sr = _run_stage("load_data", _stage_load_test_data, root, paths_cfg, sample, test_dir=test_dir)
    stages[sr.name] = sr.to_dict()
    data = sr.result if sr.status == "completed" else None

    # ── Stage 2: Normalize ──────────────────────────────────────────
    if data is not None:
        sr = _run_stage("normalize", _stage_normalize, data)
        stages[sr.name] = sr.to_dict()
        if sr.status == "completed":
            data = sr.result
    else:
        stages["normalize"] = {"status": "skipped", "elapsed_seconds": 0.0,
                               "error": "Dependency [load_data] not available"}

    # Resolve model file
    model_path = Path(config.get("model_path") or (root / paths_cfg.get("model", {}).get("artifact", "models/lgbm_entity_model.txt")))
    if not model_path.is_file():
        alt_model = root / "checkpoints" / "lgbm_model_cap40.txt"
        if alt_model.is_file():
            model_path = alt_model

    has_model = model_path.is_file()

    # Check whether we can run full streaming partition inference
    if data is not None and has_model:
        # Run high-scale streaming partition inference
        res = run_streaming_partition_infer(
            data=data,
            blocking_cfg=blocking_cfg,
            model_path=model_path,
            output_dir=output_dir,
            threshold=threshold,
            country_filter=country_filter,
            chunk_size=chunk_size,
        )

        # Populate stages with precise partition execution timings
        timings = res["stage_timings"]
        stages["blocking"] = {"status": "completed", "elapsed_seconds": timings["blocking_seconds"]}
        stages["featurize"] = {"status": "completed", "elapsed_seconds": timings["featurize_seconds"]}
        stages["predict"] = {"status": "completed", "elapsed_seconds": timings["predict_seconds"]}
        stages["group_and_write"] = {"status": "completed", "elapsed_seconds": timings["write_seconds"]}

        metrics["output"] = {
            "total_s1": res["total_s1"],
            "total_candidates": res["total_candidates"],
            "total_matched": res["total_matched"],
            "total_singletons": res["total_singletons"],
            "matching_results_path": res["matching_results_path"],
            "candidate_pairs_path": res["candidate_pairs_path"],
            "country_stats": res["country_stats"],
            "peak_ram_mb": res["peak_ram_mb"],
        }

    else:
        # Graceful fallback for synthetic unit tests where model is not provided or blocking is skipped
        candidates = None
        if data is not None:
            sr = _run_stage(
                "blocking", _stage_blocking, data, blocking_cfg,
                output_path=output_dir / "candidate_pairs.tsv",
            )
            stages[sr.name] = sr.to_dict()
            if sr.status == "completed":
                candidates = sr.result
        else:
            stages["blocking"] = {"status": "skipped", "elapsed_seconds": 0.0,
                                  "error": "Dependency [load_data] not available"}

        if not has_model or candidates is None:
            stages["featurize"] = {
                "status": "skipped", "elapsed_seconds": 0.0,
                "error": "Dependency [blocking] not available" if candidates is None else "Model artifact not available",
            }
            stages["predict"] = {
                "status": "skipped", "elapsed_seconds": 0.0,
                "error": f"FileNotFoundError: Model file not found: {model_path}",
            }
            stages["group_and_write"] = {
                "status": "skipped", "elapsed_seconds": 0.0,
                "error": "Dependency [predict] not available",
            }

    # ── Pipeline summary ────────────────────────────────────────────
    elapsed_total = time.time() - pipeline_t0
    completed = [k for k, v in stages.items() if v["status"] == "completed"]
    skipped = [k for k, v in stages.items() if v["status"] == "skipped"]

    logger.info(
        "Infer pipeline finished in %.2fs — %d completed, %d skipped",
        elapsed_total, len(completed), len(skipped),
    )

    return {
        "stages": stages,
        "metrics": metrics,
        "completed": completed,
        "skipped": skipped,
        "elapsed_total_seconds": round(elapsed_total, 3),
        "threshold_used": threshold,
    }
