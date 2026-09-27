"""
model.py  –  LightGBM wrapper & hard-negative mining for entity-resolution matching.

Wraps ``lightgbm.LGBMClassifier`` with helpers for:
1. Stratified entity splitting (by country and singleton status, zero leakage).
2. Streaming hard-negative mining over 440M candidate pairs.
3. LightGBM classifier instantiation, training with class imbalance & early stopping.
4. Fast batch validation prediction and official challenge macro F_0.5 evaluation.
5. Model persistence (.txt native booster and .pkl scikit wrapper).
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import lightgbm as lgb
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from business_entity_resolution.scoring import compute_f_beta, macro_f_beta, score_entity

logger = logging.getLogger(__name__)

# ── Feature Schema Constants ─────────────────────────────────────────
DEFAULT_FEATURE_COLUMNS: List[str] = [
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

# ── Peak Memory Monitoring ───────────────────────────────────────────
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
    """Return peak process memory (working set) in MB."""
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
    try:
        import resource
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:
        return 0.0


# ── Stratified Entity Split ──────────────────────────────────────────
def stratified_entity_split(
    s1_ids: Sequence[str],
    s1_country_map: Dict[str, str],
    gt_dict: Dict[str, Set[str]],
    val_ratio: float = 0.1,
    random_state: int = 42,
) -> Tuple[Set[str], Set[str]]:
    """Partition S1 entities into (train_entities, val_entities).

    Stratification is performed jointly on (country, is_singleton) to preserve
    the identical data distribution between training and validation sets.
    Partitioning is strictly at the entity level: no candidate pairs from a validation
    entity are ever seen during training or hard-negative tuning.

    Parameters
    ----------
    s1_ids : Sequence[str]
        List of all Source 1 entity IDs.
    s1_country_map : Dict[str, str]
        Mapping {s1_id: country_name}.
    gt_dict : Dict[str, Set[str]]
        Ground truth mapping {s1_id: set_of_matched_ids}.
    val_ratio : float, default=0.1
        Fraction of entities held out for validation (typically 0.10 - 0.15).
    random_state : int, default=42
        Reproducibility seed.

    Returns
    -------
    Tuple[Set[str], Set[str]]
        (train_entities, val_entities)
    """
    from sklearn.model_selection import train_test_split

    strata: List[str] = []
    for eid in s1_ids:
        c = s1_country_map.get(eid, "Unknown")
        is_singleton = 1 if len(gt_dict.get(eid, set())) == 0 else 0
        strata.append(f"{c}_{is_singleton}")

    train_list, val_list = train_test_split(
        s1_ids,
        test_size=val_ratio,
        stratify=strata,
        random_state=random_state,
    )
    return set(train_list), set(val_list)


# ── Streaming Hard-Negative Mining ───────────────────────────────────
def mine_hard_negatives_and_prepare_datasets(
    feature_paths: List[Path],
    gt_dict: Dict[str, Set[str]],
    train_entities: Set[str],
    val_entities: Set[str],
    output_train_parquet: Path,
    output_val_candidates_parquet: Path,
    output_val_mined_parquet: Optional[Path] = None,
    feature_columns: Optional[List[str]] = None,
    neg_pos_ratio: int = 5,
    max_negatives_has_match: int = 15,
    min_negatives_has_match: int = 5,
    singleton_negatives: int = 10,
    batch_row_groups: int = 5,
    max_pairs: Optional[int] = None,
) -> Dict[str, Any]:
    """Stream candidate pairs from feature Parquet files, mine hard negatives, and cache splits.

    Memory Safety:
    Reads row groups in batches via PyArrow memory mapping. Working memory is bounded
    by a few row groups (< 100 MB), never loading the full 440M dataset into memory.

    Mining Strategy:
    1. For each training entity:
       - Keep ALL true-positive candidate pairs (label = 1).
       - Select hard negatives (label = 0) with top ``name_jaro_winkler`` scores.
       - Entities with matches: sample ``min(len(negs), max(5, min(15, 5 * n_pos)))``.
       - Singleton entities: sample up to ``10`` hard negatives to teach the model
         to resist high-similarity false positives where no true match exists.
    2. For validation entities:
       - Stream all candidate pairs to ``output_val_candidates_parquet`` for full evaluation.
       - Sample a mined subset to ``output_val_mined_parquet`` for LightGBM early stopping.
    """
    feats = feature_columns or DEFAULT_FEATURE_COLUMNS
    cols_to_read = ["source1_entity_id", "candidate_entity_id"] + feats
    output_train_parquet.parent.mkdir(parents=True, exist_ok=True)
    output_val_candidates_parquet.parent.mkdir(parents=True, exist_ok=True)
    if output_val_mined_parquet:
        output_val_mined_parquet.parent.mkdir(parents=True, exist_ok=True)

    # Initialize Parquet writers
    train_writer: Optional[pq.ParquetWriter] = None
    val_writer: Optional[pq.ParquetWriter] = None
    val_mined_writer: Optional[pq.ParquetWriter] = None

    t0 = time.time()
    total_pairs_scanned = 0
    total_train_rows = 0
    total_train_pos = 0
    total_train_neg = 0
    total_val_candidates = 0
    total_val_mined_rows = 0

    leftover_df: Optional[pd.DataFrame] = None
    leftover_tbl: Optional[pa.Table] = None

    for f_idx, feat_path in enumerate(feature_paths):
        p = Path(feat_path)
        if not p.is_file():
            logger.warning(f"Feature file not found: {p}")
            continue

        pf = pq.ParquetFile(p)
        n_rgs = pf.num_row_groups
        logger.info(f"Streaming from {p.name} ({n_rgs} row groups, {pf.metadata.num_rows:,} rows)...")

        for rg_start in range(0, n_rgs, batch_row_groups):
            if max_pairs is not None and total_pairs_scanned >= max_pairs:
                break

            rg_end = min(rg_start + batch_row_groups, n_rgs)
            tbl_batch = pf.read_row_groups(list(range(rg_start, rg_end)), columns=cols_to_read)

            # Prepend leftover from previous batch if exists
            if leftover_tbl is not None:
                tbl_batch = pa.concat_tables([leftover_tbl, tbl_batch])
                leftover_tbl = None

            n_rows_batch = len(tbl_batch)
            if n_rows_batch == 0:
                continue

            s1_list = tbl_batch["source1_entity_id"].to_pylist()
            cand_list = tbl_batch["candidate_entity_id"].to_pylist()
            jw_arr = tbl_batch["name_jaro_winkler"].to_numpy(zero_copy_only=False)

            is_last_batch = (f_idx == len(feature_paths) - 1) and (rg_end == n_rgs)

            train_indices: List[int] = []
            train_labels: List[int] = []
            val_indices: List[int] = []
            val_mined_indices: List[int] = []
            val_mined_labels: List[int] = []

            i = 0
            n = len(s1_list)
            while i < n:
                curr_s1 = s1_list[i]
                start = i
                while i < n and s1_list[i] == curr_s1:
                    i += 1
                end = i

                # If this entity is cut off at the batch boundary and not the last batch, save for next
                if i == n and not is_last_batch:
                    leftover_tbl = tbl_batch.slice(start, end - start)
                    break

                is_val = curr_s1 in val_entities
                is_train = curr_s1 in train_entities
                if not is_val and not is_train:
                    continue

                ent_cands = cand_list[start:end]
                true_m = gt_dict.get(curr_s1, set())
                is_singleton = (len(true_m) == 0)

                if is_singleton:
                    pos_rel: List[int] = []
                    neg_rel = list(range(len(ent_cands)))
                else:
                    pos_rel = [j for j, c in enumerate(ent_cands) if c in true_m]
                    neg_rel = [j for j, c in enumerate(ent_cands) if c not in true_m]

                ent_jw = jw_arr[start:end]

                if is_singleton:
                    k_neg = min(len(neg_rel), singleton_negatives)
                else:
                    k_neg = min(len(neg_rel), max(min_negatives_has_match, min(max_negatives_has_match, neg_pos_ratio * len(pos_rel))))

                if len(neg_rel) <= k_neg:
                    top_neg_rel = neg_rel
                elif k_neg > 0:
                    neg_jw = ent_jw[neg_rel]
                    top_idx = np.argsort(-neg_jw)[:k_neg]
                    top_neg_rel = [neg_rel[idx] for idx in top_idx]
                else:
                    top_neg_rel = []

                if is_val:
                    val_indices.extend(range(start, end))
                    if output_val_mined_parquet:
                        val_mined_indices.extend(start + j for j in pos_rel)
                        val_mined_labels.extend([1] * len(pos_rel))
                        val_mined_indices.extend(start + j for j in top_neg_rel)
                        val_mined_labels.extend([0] * len(top_neg_rel))
                elif is_train:
                    train_indices.extend(start + j for j in pos_rel)
                    train_labels.extend([1] * len(pos_rel))
                    train_indices.extend(start + j for j in top_neg_rel)
                    train_labels.extend([0] * len(top_neg_rel))

            total_pairs_scanned += (n - (len(leftover_tbl) if leftover_tbl is not None else 0))

            # Write train chunk
            if train_indices:
                df_batch = tbl_batch.to_pandas(types_mapper=pd.ArrowDtype)
                sub_train = df_batch.iloc[train_indices].copy()
                sub_train["label"] = np.array(train_labels, dtype=np.int8)

                train_tbl = pa.Table.from_pandas(sub_train, preserve_index=False)
                if train_writer is None:
                    train_writer = pq.ParquetWriter(output_train_parquet, train_tbl.schema, compression="SNAPPY")
                train_writer.write_table(train_tbl)

                total_train_rows += len(train_indices)
                n_pos = sum(train_labels)
                total_train_pos += n_pos
                total_train_neg += (len(train_labels) - n_pos)

            # Write val candidates chunk
            if val_indices:
                if 'df_batch' not in locals():
                    df_batch = tbl_batch.to_pandas(types_mapper=pd.ArrowDtype)
                sub_val = df_batch.iloc[val_indices]
                val_tbl = pa.Table.from_pandas(sub_val, preserve_index=False)
                if val_writer is None:
                    val_writer = pq.ParquetWriter(output_val_candidates_parquet, val_tbl.schema, compression="SNAPPY")
                val_writer.write_table(val_tbl)
                total_val_candidates += len(val_indices)

            # Write val mined chunk
            if val_mined_indices and output_val_mined_parquet:
                if 'df_batch' not in locals():
                    df_batch = tbl_batch.to_pandas(types_mapper=pd.ArrowDtype)
                sub_val_mined = df_batch.iloc[val_mined_indices].copy()
                sub_val_mined["label"] = np.array(val_mined_labels, dtype=np.int8)
                val_mined_tbl = pa.Table.from_pandas(sub_val_mined, preserve_index=False)
                if val_mined_writer is None:
                    val_mined_writer = pq.ParquetWriter(output_val_mined_parquet, val_mined_tbl.schema, compression="SNAPPY")
                val_mined_writer.write_table(val_mined_tbl)
                total_val_mined_rows += len(val_mined_indices)

            if "df_batch" in locals():
                del df_batch

            if (rg_start // batch_row_groups) % 20 == 0:
                elapsed_cur = time.time() - t0
                speed = total_pairs_scanned / elapsed_cur if elapsed_cur > 0 else 0
                logger.info(
                    f"Processed {total_pairs_scanned:,} pairs ({speed:,.0f} pairs/s) | "
                    f"Train: {total_train_rows:,} (pos={total_train_pos:,}, neg={total_train_neg:,}) | "
                    f"Val: {total_val_candidates:,} | Peak RAM: {get_peak_memory_mb():.1f} MB"
                )

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()
    if val_mined_writer is not None:
        val_mined_writer.close()

    elapsed = time.time() - t0
    stats = {
        "total_pairs_scanned": total_pairs_scanned,
        "total_train_rows": total_train_rows,
        "total_train_pos": total_train_pos,
        "total_train_neg": total_train_neg,
        "total_val_candidates": total_val_candidates,
        "total_val_mined_rows": total_val_mined_rows,
        "train_imbalance_ratio": (total_train_neg / total_train_pos) if total_train_pos > 0 else 0.0,
        "elapsed_seconds": elapsed,
        "pairs_per_sec": total_pairs_scanned / elapsed if elapsed > 0 else 0.0,
        "peak_memory_mb": get_peak_memory_mb(),
    }
    return stats


# ── Classifier Builder ───────────────────────────────────────────────
def build_classifier(
    model_cfg: Dict[str, Any],
    scale_pos_weight: Optional[float] = None,
) -> LGBMClassifier:
    """Instantiate an ``LGBMClassifier`` from config values.

    Parameters
    ----------
    model_cfg : dict
        Parsed ``configs/model.yaml["lgbm"]``.
    scale_pos_weight : float | None
        Optional explicit positive class weight (e.g. n_neg / n_pos).
        If provided and > 0, overrides ``class_weight``.

    Returns
    -------
    LGBMClassifier
    """
    params = {
        "boosting_type": model_cfg.get("boosting_type", "gbdt"),
        "num_leaves": model_cfg.get("num_leaves", 63),
        "max_depth": model_cfg.get("max_depth", -1),
        "learning_rate": model_cfg.get("learning_rate", 0.05),
        "n_estimators": model_cfg.get("n_estimators", 500),
        "subsample": model_cfg.get("subsample", 0.8),
        "colsample_bytree": model_cfg.get("colsample_bytree", 0.8),
        "reg_alpha": model_cfg.get("reg_alpha", 0.1),
        "reg_lambda": model_cfg.get("reg_lambda", 1.0),
        "random_state": model_cfg.get("random_state", 42),
        "n_jobs": model_cfg.get("n_jobs", -1),
        "objective": "binary",
        "verbose": -1,
    }

    if scale_pos_weight is not None and scale_pos_weight > 0:
        params["scale_pos_weight"] = float(scale_pos_weight)
        params["class_weight"] = None
    else:
        params["class_weight"] = model_cfg.get("class_weight", "balanced")

    return LGBMClassifier(**params)


# ── Model Training ───────────────────────────────────────────────────
def train(
    clf: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    callbacks: Optional[list] = None,
    early_stopping_rounds: int = 50,
    verbose_eval: int = 50,
) -> Any:
    """Fit the classifier on training data.

    Parameters
    ----------
    clf : LGBMClassifier
    X_train : np.ndarray
    y_train : np.ndarray
    X_val : np.ndarray | None
    y_val : np.ndarray | None
    callbacks : list | None
    early_stopping_rounds : int, default=50
    verbose_eval : int, default=50

    Returns
    -------
    LGBMClassifier
        The fitted classifier.
    """
    if X_val is not None and y_val is not None:
        if callbacks is None:
            callbacks = [
                early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                log_evaluation(period=verbose_eval),
            ]
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
    else:
        clf.fit(X_train, y_train)

    return clf


# ── Prediction & Scoring ─────────────────────────────────────────────
def predict_proba(clf: Any, X: np.ndarray) -> np.ndarray:
    """Return match probabilities (1-D array of P(match)).

    Parameters
    ----------
    clf : LGBMClassifier | Booster
    X : np.ndarray

    Returns
    -------
    np.ndarray
        1-D array of P(match).
    """
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    elif hasattr(clf, "predict"):
        return clf.predict(X)
    else:
        raise TypeError(f"Unsupported model type: {type(clf)}")


def evaluate_validation(
    clf: Any,
    val_candidates_path: Path,
    val_gt: Dict[str, Set[str]],
    s1_country_map: Optional[Dict[str, str]] = None,
    feature_columns: Optional[List[str]] = None,
    threshold: float = 0.5,
    batch_size: int = 500000,
) -> Dict[str, Any]:
    """Score the held-out validation set using official macro_f_beta.

    Streams the full validation candidate set in batches, predicts match
    probabilities with the trained model, applies ``threshold``, and scores
    with ``scoring.macro_f_beta``.

    Parameters
    ----------
    clf : LGBMClassifier | Booster
    val_candidates_path : Path
        Parquet file containing all candidate pairs for validation entities.
    val_gt : Dict[str, Set[str]]
        Authoritative validation ground truth.
    s1_country_map : Dict[str, str] | None
        Optional mapping {s1_id: country} to compute per-country macro F_0.5.
    feature_columns : List[str] | None
    threshold : float, default=0.5
        Classification decision threshold.
    batch_size : int, default=500000

    Returns
    -------
    Dict[str, Any]
        Scoring report with macro_f_beta, singleton_mean, has_match_mean, etc.
    """
    feats = feature_columns or DEFAULT_FEATURE_COLUMNS
    cols_to_read = ["source1_entity_id", "candidate_entity_id"] + feats

    pf = pq.ParquetFile(val_candidates_path)
    total_val_pairs = pf.metadata.num_rows

    predictions: Dict[str, Set[str]] = {eid: set() for eid in val_gt.keys()}

    t0 = time.time()
    pairs_scored = 0

    for batch in pf.iter_batches(batch_size=batch_size, columns=cols_to_read):
        s1_ids = batch["source1_entity_id"].to_pylist()
        cand_ids = batch["candidate_entity_id"].to_pylist()

        # Build feature matrix
        X_batch = np.column_stack([
            batch[col].to_numpy(zero_copy_only=False) for col in feats
        ]).astype(np.float32)

        probs = predict_proba(clf, X_batch)
        match_idx = np.where(probs >= threshold)[0]

        for idx in match_idx:
            eid = s1_ids[idx]
            if eid in predictions:
                predictions[eid].add(cand_ids[idx])

        pairs_scored += len(s1_ids)

    elapsed = time.time() - t0
    logger.info(f"Scored {pairs_scored:,} validation pairs in {elapsed:.2f}s ({pairs_scored/elapsed:,.0f} pairs/s)")

    metrics = macro_f_beta(predictions, val_gt, beta=0.5)
    metrics["threshold"] = threshold
    metrics["val_pairs_scored"] = pairs_scored
    metrics["eval_elapsed_seconds"] = elapsed

    if s1_country_map:
        val_gt_india = {eid: val_gt[eid] for eid in val_gt if s1_country_map.get(eid) == "India"}
        val_gt_us = {eid: val_gt[eid] for eid in val_gt if s1_country_map.get(eid) == "US"}
        if val_gt_india:
            metrics["india"] = macro_f_beta(predictions, val_gt_india, beta=0.5)
        if val_gt_us:
            metrics["us"] = macro_f_beta(predictions, val_gt_us, beta=0.5)

    return metrics


# ── Model Persistence ────────────────────────────────────────────────
def save_model(clf: Any, path: Path) -> None:
    """Persist a trained model to disk (.txt and .pkl formats).

    Parameters
    ----------
    clf : LGBMClassifier | Booster
    path : Path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save native booster text format
    txt_path = path if path.suffix == ".txt" else path.with_suffix(".txt")
    if hasattr(clf, "booster_"):
        clf.booster_.save_model(str(txt_path))
    elif hasattr(clf, "save_model"):
        clf.save_model(str(txt_path))

    # Save pickle format for scikit wrapper
    pkl_path = path if path.suffix in (".pkl", ".joblib") else path.with_suffix(".pkl")
    try:
        import joblib
        joblib.dump(clf, pkl_path)
    except Exception as e:
        logger.warning(f"Could not save joblib model to {pkl_path}: {e}")


def load_model(path: Path) -> Any:
    """Load a previously saved model from disk.

    Parameters
    ----------
    path : Path

    Returns
    -------
    LGBMClassifier | Booster
    """
    path = Path(path)
    if not path.is_file():
        if path.with_suffix(".pkl").is_file():
            path = path.with_suffix(".pkl")
        elif path.with_suffix(".txt").is_file():
            path = path.with_suffix(".txt")
        else:
            raise FileNotFoundError(f"Model file not found: {path}")

    if path.suffix in (".pkl", ".joblib"):
        import joblib
        return joblib.load(path)
    elif path.suffix == ".txt":
        return lgb.Booster(model_file=str(path))
    else:
        try:
            import joblib
            return joblib.load(path)
        except Exception:
            return lgb.Booster(model_file=str(path))
