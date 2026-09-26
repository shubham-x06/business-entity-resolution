"""
pipeline.py  –  End-to-end orchestration for training and inference.

Provides top-level ``run_train_pipeline()`` and ``run_infer_pipeline()``
functions that chain together:
    normalisation → blocking → features → model → grouping → I/O.

Each stage is wrapped in a try/except that catches NotImplementedError and
AttributeError, logging a clear "stage not yet implemented — skipping"
message rather than crashing the whole pipeline.  This lets the skeleton
run end-to-end today even though real stages don't all exist yet.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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

    Catches ``NotImplementedError`` and ``AttributeError`` (the two errors
    raised by unimplemented stubs or missing functions) and logs a clear
    skip message.  All other exceptions propagate.

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
    except (NotImplementedError, AttributeError) as exc:
        elapsed = time.time() - t0
        msg = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "Stage [%s] — SKIPPED (not yet implemented): %s", name, msg,
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

    data_cfg = paths_cfg["data"]["train"]
    s1 = load_source(root / data_cfg["source1"], expected_source=1)
    if sample is not None and sample > 0:
        s1 = s1.head(sample).copy()
        logger.info("Sampled Source 1 to %d rows", len(s1))
    s2 = load_source(root / data_cfg["source2"], expected_source=2)
    s3 = load_source(root / data_cfg["source3"], expected_source=3)

    logger.info(
        "Loaded sources — S1: %d rows | S2: %d rows | S3: %d rows",
        len(s1), len(s2), len(s3),
    )

    gt_path = root / data_cfg["ground_truth"]
    gt = load_ground_truth(gt_path) if gt_path.is_file() else None
    if gt is not None:
        logger.info("Loaded ground truth: %d rows", len(gt))

    return {"s1": s1, "s2": s2, "s3": s3, "ground_truth": gt}


def _stage_load_test_data(
    root: Path,
    paths_cfg: Dict[str, Any],
    sample: Optional[int] = None,
) -> Dict[str, Any]:
    """Load and return test source DataFrames."""
    from business_entity_resolution.io_utils import load_source

    data_cfg = paths_cfg["data"]["test"]
    s1 = load_source(root / data_cfg["source1"], expected_source=1)
    if sample is not None and sample > 0:
        s1 = s1.head(sample).copy()
        logger.info("Sampled Source 1 to %d rows", len(s1))
    s2 = load_source(root / data_cfg["source2"], expected_source=2)
    s3 = load_source(root / data_cfg["source3"], expected_source=3)

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

    # Flatten candidates dict to list of (s1_id, cand_id) pairs
    pairs: List[Tuple[str, str]] = []
    for s1_id, cand_ids in candidates.items():
        for cand_id in cand_ids:
            pairs.append((s1_id, cand_id))

    # Combine S2 + S3 as the reference pool
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

    # Build label vector from ground_truth
    # (This is a placeholder call sequence — real label construction
    # will depend on how features.py aligns pair_df with ground_truth)
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

        for i, row in pair_df.iterrows():
            s1_id = str(row["s1_id"])
            cand_id = str(row["cand_id"])
            if s1_id in gt_set and cand_id in gt_set[s1_id]:
                y[i] = 1

    # Validation split
    val_cfg = model_cfg.get("validation", {})
    test_size = val_cfg.get("test_size", 0.2)
    rand_state = val_cfg.get("random_state", 42)

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=rand_state, stratify=y,
    )

    clf = train(clf, X_train, y_train, X_val=X_val, y_val=y_val)
    logger.info("Model trained on %d pairs", len(X_train))

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

    # Group scores by S1 entity: {s1_id: [(cand_id, score), ...]}
    scored_pairs: Dict[str, List[Tuple[str, float]]] = {}
    for i, row in pair_df.iterrows():
        s1_id = str(row["s1_id"])
        cand_id = str(row["cand_id"])
        scored_pairs.setdefault(s1_id, []).append((cand_id, float(proba[i])))

    return scored_pairs


def _stage_threshold_sweep(
    scored_pairs: Dict[str, List[Tuple[str, float]]],
    ground_truth: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """Run threshold sweep to find optimal F_0.5 threshold.

    NOTE: Depends on grouping.py from feature/grouping branch.
    If that branch is not yet merged, this stage will raise ImportError.
    """
    from business_entity_resolution.grouping import sweep_thresholds
    from business_entity_resolution.scoring import load_matches_dict

    if ground_truth is None:
        logger.warning("No ground truth available — cannot sweep thresholds")
        return {"best_threshold": 0.5, "best_metrics": {}, "sweep": {}}

    # Convert ground_truth DataFrame to dict for scoring
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
    """Group matches at threshold and write output files.

    NOTE: Depends on grouping.py from feature/grouping branch.
    """
    from business_entity_resolution.grouping import group_by_threshold
    from business_entity_resolution.io_utils import (
        write_candidate_pairs,
        write_matching_results,
    )

    # Group predictions at the chosen threshold
    predictions = group_by_threshold(scored_pairs, threshold)

    # Ensure all S1 entities appear (singletons with empty set)
    for s1_id in all_s1_ids:
        if s1_id not in predictions:
            predictions[s1_id] = set()

    # Convert sets to sorted lists for output writers
    matches_for_write: Dict[str, List[str]] = {
        s1_id: sorted(matched_ids)
        for s1_id, matched_ids in predictions.items()
    }

    # Write matching_results.tsv
    output_dir.mkdir(parents=True, exist_ok=True)
    mr_path = Path(paths_cfg.get("output", {}).get(
        "matching_results", str(output_dir / "matching_results.tsv"),
    ))
    write_matching_results(matches_for_write, mr_path)

    # Write candidate_pairs.tsv (all candidates, not just above threshold)
    cand_for_write: Dict[str, List[str]] = {
        s1_id: [cid for cid, _ in cands]
        for s1_id, cands in scored_pairs.items()
    }
    # Include S1 entities with no candidates
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
# Top-level orchestration functions
# ═══════════════════════════════════════════════════════════════════════════


def run_train_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the full training pipeline with graceful stage-skip handling.

    Sequence: load data → normalize → block → featurize → train model →
    threshold sweep → group.

    Each stage that raises NotImplementedError or AttributeError is logged
    as skipped and the pipeline continues to the next independent stage.
    Stages that depend on prior stage output are automatically skipped if
    their dependency was skipped.

    Parameters
    ----------
    config : dict
        Merged configuration containing keys from paths.yaml, blocking.yaml,
        and model.yaml.  Must include at minimum:
        - "root": Path to the repository root
        - "paths": contents of paths.yaml
        - "blocking": contents of blocking.yaml
        - "model": contents of model.yaml
        Optional:
        - "sample": int, subsample S1 rows

    Returns
    -------
    dict
        Summary with keys:
        - "stages": dict mapping stage name → {status, elapsed_seconds, ...}
        - "metrics": any metrics produced by stages that ran
        - "completed": list of stage names that completed
        - "skipped": list of stage names that were skipped
    """
    root = Path(config["root"])
    paths_cfg = config.get("paths", {})
    blocking_cfg = config.get("blocking", {})
    model_cfg = config.get("model", {})
    sample = config.get("sample")

    stages: Dict[str, Dict[str, Any]] = {}
    metrics: Dict[str, Any] = {}
    pipeline_t0 = time.time()

    # ── Stage 1: Load training data ─────────────────────────────────
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
        output_dir = root / paths_cfg.get("output", {}).get("dir", "output")
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
        model_path = root / paths_cfg.get("model", {}).get(
            "artifact", "models/lgbm_entity_model.txt",
        )
        sr = _run_stage(
            "train_model", _stage_train_model,
            pair_df, X, data.get("ground_truth"), model_cfg, model_path,
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
    group at frozen threshold → write output files.

    Parameters
    ----------
    config : dict
        Merged configuration.  Must include:
        - "root": Path to the repository root
        - "paths": contents of paths.yaml
        - "blocking": contents of blocking.yaml
        - "model": contents of model.yaml
        Optional:
        - "sample": int, subsample S1 rows
        - "threshold": float, match threshold (defaults to model.yaml value)

    Returns
    -------
    dict
        Same shape as ``run_train_pipeline`` return.
    """
    root = Path(config["root"])
    paths_cfg = config.get("paths", {})
    blocking_cfg = config.get("blocking", {})
    model_cfg = config.get("model", {})
    sample = config.get("sample")
    threshold = config.get(
        "threshold",
        model_cfg.get("threshold", {}).get("match", 0.5),
    )

    stages: Dict[str, Dict[str, Any]] = {}
    metrics: Dict[str, Any] = {}
    pipeline_t0 = time.time()

    # ── Stage 1: Load test data ─────────────────────────────────────
    sr = _run_stage("load_data", _stage_load_test_data, root, paths_cfg, sample)
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
        output_dir = root / paths_cfg.get("output", {}).get("dir", "output")
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

    # ── Stage 5: Load model + predict ───────────────────────────────
    scored_pairs = None
    if pair_df is not None and X is not None:
        # Load trained model
        model_path = root / paths_cfg.get("model", {}).get(
            "artifact", "models/lgbm_entity_model.txt",
        )

        def _load_and_predict(
            model_path: Path, X: Any, pair_df: pd.DataFrame,
        ) -> Dict[str, List[Tuple[str, float]]]:
            from business_entity_resolution.model import (
                load_model,
                predict_proba,
            )
            clf = load_model(model_path)
            return _stage_predict(clf, X, pair_df)

        sr = _run_stage("predict", _load_and_predict, model_path, X, pair_df)
        stages[sr.name] = sr.to_dict()
        if sr.status == "completed":
            scored_pairs = sr.result
    else:
        stages["predict"] = {
            "status": "skipped", "elapsed_seconds": 0.0,
            "error": "Dependency [featurize] not available",
        }

    # ── Stage 6: Group + write output ───────────────────────────────
    if scored_pairs is not None and data is not None:
        output_dir = root / paths_cfg.get("output", {}).get("dir", "output")
        all_s1_ids = data["s1"]["entity_id"].tolist()
        sr = _run_stage(
            "group_and_write", _stage_group_and_write,
            scored_pairs, threshold, all_s1_ids, output_dir, paths_cfg,
        )
        stages[sr.name] = sr.to_dict()
        if sr.status == "completed":
            metrics["output"] = sr.result
    else:
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
