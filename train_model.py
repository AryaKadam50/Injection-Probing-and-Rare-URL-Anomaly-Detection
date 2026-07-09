"""
train_model.py
================
Fits IsolationForest + LOF + StandardScaler on a window of entity-window
rows (a 14-day window of TRAIN data in the rolling architecture) and
packages everything a scoring process needs into one joblib bundle.

train_models() is byte-for-byte the same algorithm as uc09_v3.2's
train_models(): same StandardScaler, same IsolationForest(n_estimators=200,
contamination="auto"), same LOF(n_neighbors=20, novelty=True), same
score-bound percentiles, same fused-threshold / catastrophic-threshold
derivation. Nothing here changes detection behavior — it just no longer
assumes it's the only thing running (it doesn't touch global state, so it's
safe to call from a background thread while another model is scoring live
traffic).
"""

import time
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from feature_extraction import (
    MODEL_VERSION, FEATURE_COLS, to_matrix, current_config,
    SCORE_BOUND_PCT_LOW, SCORE_BOUND_PCT_HIGH, ALERT_PCT, CATASTROPHIC_PCT,
)


def train_models(train_rows: list):
    """Unchanged from uc09_v3.2.train_models()."""
    t0 = time.time()
    X_train = to_matrix(train_rows)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    iforest = IsolationForest(
        n_estimators=200, contamination="auto", random_state=42, n_jobs=-1,
    )
    iforest.fit(X_scaled)

    lof = LocalOutlierFactor(
        n_neighbors=20, novelty=True, contamination="auto", n_jobs=-1,
    )
    lof.fit(X_scaled)

    if_raw_train = iforest.decision_function(X_scaled)
    lof_raw_train = lof.score_samples(X_scaled)
    if_bounds = (
        float(np.percentile(if_raw_train, SCORE_BOUND_PCT_LOW)),
        float(np.percentile(if_raw_train, SCORE_BOUND_PCT_HIGH)),
    )
    lof_bounds = (
        float(np.percentile(lof_raw_train, SCORE_BOUND_PCT_LOW)),
        float(np.percentile(lof_raw_train, SCORE_BOUND_PCT_HIGH)),
    )

    if_min, if_max = if_bounds
    lof_min, lof_max = lof_bounds
    if_score_tr = np.clip(1 - (if_raw_train - if_min) / (if_max - if_min + 1e-9), 0.0, 1.0)
    lof_score_tr = np.clip(1 - (lof_raw_train - lof_min) / (lof_max - lof_min + 1e-9), 0.0, 1.0)
    legit_tr = np.array([r["legit_path_ratio"] for r in train_rows])
    fused_tr = (1.0 - 0.5 * legit_tr) * 0.5 * (if_score_tr + lof_score_tr)
    fused_threshold = float(np.percentile(fused_tr, ALERT_PCT))
    catastrophic_threshold = float(np.percentile(fused_tr, CATASTROPHIC_PCT))

    elapsed = time.time() - t0
    print(f"[train_model] Trained IF+LOF on {len(train_rows):,} windows in {elapsed:,.1f}s | "
          f"fused_threshold={fused_threshold:.4f} catastrophic={catastrophic_threshold:.4f}", flush=True)

    return {
        "scaler": scaler,
        "iforest": iforest,
        "lof": lof,
        "if_bounds": if_bounds,
        "lof_bounds": lof_bounds,
        "fused_threshold": fused_threshold,
        "catastrophic_threshold": catastrophic_threshold,
        "fused_train_scores": fused_tr,      # kept in-memory for drift comparison
        "if_train_scores": if_score_tr,
        "lof_train_scores": lof_score_tr,
        "train_matrix": X_train,             # kept for PSI/KL feature drift
    }


def build_bundle(model_id: str, week_start, week_end, fit_result: dict,
                  path_freq: dict, path_total: int, common_cutoff: float,
                  path_baseline: dict, train_rarity_p95: float,
                  n_train_windows: int) -> dict:
    """
    Assembles the full save-able bundle. Same fields save_model_bundle()
    wrote in the batch tool, plus versioning metadata needed by
    model_library.py (model_id, training window, row count, creation time).
    """
    return {
        "model_id": model_id,
        "version": MODEL_VERSION,
        "feature_cols": FEATURE_COLS,
        "config": current_config(),
        "created_at": time.time(),
        "train_window_start": week_start,
        "train_window_end": week_end,
        "n_train_windows": n_train_windows,

        "scaler": fit_result["scaler"],
        "iforest": fit_result["iforest"],
        "lof": fit_result["lof"],
        "if_bounds": fit_result["if_bounds"],
        "lof_bounds": fit_result["lof_bounds"],
        "fused_threshold": fit_result["fused_threshold"],
        "catastrophic_threshold": fit_result["catastrophic_threshold"],

        "path_freq": path_freq,
        "path_total": path_total,
        "common_cutoff": common_cutoff,
        "path_baseline": path_baseline,
        "train_rarity_p95": train_rarity_p95,

        # Retained only for drift_detection.py; not needed for scoring.
        "_drift_reference": {
            "train_matrix": fit_result["train_matrix"],
            "fused_train_scores": fit_result["fused_train_scores"],
            "if_train_scores": fit_result["if_train_scores"],
            "lof_train_scores": fit_result["lof_train_scores"],
        },
    }


def save_bundle_to_disk(bundle: dict, path: str) -> None:
    joblib.dump(bundle, path)
    print(f"[train_model] Bundle '{bundle['model_id']}' saved to {path}", flush=True)


def train_candidate_from_records(model_id: str, week_records: list, week_start, week_end) -> dict:
    """
    End-to-end candidate training for a rolling window: builds the path
    popularity table, per-path baselines, and entity-window features from
    this window's records (exactly like the batch tool built them from
    TRAIN), then fits models. This is the function scheduler.py calls in a
    background thread.
    """
    from feature_extraction import (
        build_path_popularity, build_path_query_baseline, build_entity_windows,
    )

    path_freq, path_total, common_cutoff = build_path_popularity(week_records)
    path_baseline = build_path_query_baseline(week_records)
    train_rows = build_entity_windows(
        week_records, path_freq, path_total, common_cutoff, path_baseline, label=model_id
    )

    if len(train_rows) < 10:
        raise ValueError(
            f"[train_model] Only {len(train_rows)} entity-windows in this window's data — "
            f"too few to train a meaningful model for {model_id}."
        )

    train_rarity_p95 = float(np.percentile([r["path_rarity_max"] for r in train_rows], 95))
    fit_result = train_models(train_rows)

    bundle = build_bundle(
        model_id=model_id, week_start=week_start, week_end=week_end, fit_result=fit_result,
        path_freq=path_freq, path_total=path_total, common_cutoff=common_cutoff,
        path_baseline=path_baseline, train_rarity_p95=train_rarity_p95,
        n_train_windows=len(train_rows),
    )
    return bundle
    iforest = IsolationForest(
        n_estimators=200, contamination="auto", random_state=42, n_jobs=-1,
    )
    iforest.fit(X_scaled)

    lof = LocalOutlierFactor(
        n_neighbors=20, novelty=True, contamination="auto", n_jobs=-1,
    )
    lof.fit(X_scaled)

    if_raw_train = iforest.decision_function(X_scaled)
    lof_raw_train = lof.score_samples(X_scaled)
    if_bounds = (
        float(np.percentile(if_raw_train, SCORE_BOUND_PCT_LOW)),
        float(np.percentile(if_raw_train, SCORE_BOUND_PCT_HIGH)),
    )
    lof_bounds = (
        float(np.percentile(lof_raw_train, SCORE_BOUND_PCT_LOW)),
        float(np.percentile(lof_raw_train, SCORE_BOUND_PCT_HIGH)),
    )

    if_min, if_max = if_bounds
    lof_min, lof_max = lof_bounds
    if_score_tr = np.clip(1 - (if_raw_train - if_min) / (if_max - if_min + 1e-9), 0.0, 1.0)
    lof_score_tr = np.clip(1 - (lof_raw_train - lof_min) / (lof_max - lof_min + 1e-9), 0.0, 1.0)
    legit_tr = np.array([r["legit_path_ratio"] for r in train_rows])
    fused_tr = (1.0 - 0.5 * legit_tr) * 0.5 * (if_score_tr + lof_score_tr)
    fused_threshold = float(np.percentile(fused_tr, ALERT_PCT))
    catastrophic_threshold = float(np.percentile(fused_tr, CATASTROPHIC_PCT))

    elapsed = time.time() - t0
    print(f"[train_model] Trained IF+LOF on {len(train_rows):,} windows in {elapsed:,.1f}s | "
          f"fused_threshold={fused_threshold:.4f} catastrophic={catastrophic_threshold:.4f}", flush=True)

    return {
        "scaler": scaler,
        "iforest": iforest,
        "lof": lof,
        "if_bounds": if_bounds,
        "lof_bounds": lof_bounds,
        "fused_threshold": fused_threshold,
        "catastrophic_threshold": catastrophic_threshold,
        "fused_train_scores": fused_tr,      # kept in-memory for drift comparison
        "if_train_scores": if_score_tr,
        "lof_train_scores": lof_score_tr,
        "train_matrix": X_train,             # kept for PSI/KL feature drift
    }


def build_bundle(model_id: str, week_start, week_end, fit_result: dict,
                  path_freq: dict, path_total: int, common_cutoff: float,
                  path_baseline: dict, train_rarity_p95: float,
                  n_train_windows: int) -> dict:
    """
    Assembles the full save-able bundle. Same fields save_model_bundle()
    wrote in the batch tool, plus versioning metadata needed by
    model_library.py (model_id, training window, row count, creation time).
    """
    return {
        "model_id": model_id,
        "version": MODEL_VERSION,
        "feature_cols": FEATURE_COLS,
        "config": current_config(),
        "created_at": time.time(),
        "train_window_start": week_start,
        "train_window_end": week_end,
        "n_train_windows": n_train_windows,

        "scaler": fit_result["scaler"],
        "iforest": fit_result["iforest"],
        "lof": fit_result["lof"],
        "if_bounds": fit_result["if_bounds"],
        "lof_bounds": fit_result["lof_bounds"],
        "fused_threshold": fit_result["fused_threshold"],
        "catastrophic_threshold": fit_result["catastrophic_threshold"],

        "path_freq": path_freq,
        "path_total": path_total,
        "common_cutoff": common_cutoff,
        "path_baseline": path_baseline,
        "train_rarity_p95": train_rarity_p95,

        # Retained only for drift_detection.py; not needed for scoring.
        "_drift_reference": {
            "train_matrix": fit_result["train_matrix"],
            "fused_train_scores": fit_result["fused_train_scores"],
            "if_train_scores": fit_result["if_train_scores"],
            "lof_train_scores": fit_result["lof_train_scores"],
        },
    }


def save_bundle_to_disk(bundle: dict, path: str) -> None:
    joblib.dump(bundle, path)
    print(f"[train_model] Bundle '{bundle['model_id']}' saved to {path}", flush=True)


def train_candidate_from_records(model_id: str, week_records: list, week_start, week_end) -> dict:
    """
    End-to-end candidate training for a rolling window: builds the path
    popularity table, per-path baselines, and entity-window features from
    this window's records (exactly like the batch tool built them from
    TRAIN), then fits models. This is the function scheduler.py calls in a
    background thread.
    """
    from feature_extraction import (
        build_path_popularity, build_path_query_baseline, build_entity_windows,
    )

    path_freq, path_total, common_cutoff = build_path_popularity(week_records)
    path_baseline = build_path_query_baseline(week_records)
    train_rows = build_entity_windows(
        week_records, path_freq, path_total, common_cutoff, path_baseline, label=model_id
    )

    if len(train_rows) < 10:
        raise ValueError(
            f"[train_model] Only {len(train_rows)} entity-windows in this window's data — "
            f"too few to train a meaningful model for {model_id}."
        )

    train_rarity_p95 = float(np.percentile([r["path_rarity_max"] for r in train_rows], 95))
    fit_result = train_models(train_rows)

    bundle = build_bundle(
        model_id=model_id, week_start=week_start, week_end=week_end, fit_result=fit_result,
        path_freq=path_freq, path_total=path_total, common_cutoff=common_cutoff,
        path_baseline=path_baseline, train_rarity_p95=train_rarity_p95,
        n_train_windows=len(train_rows),
    )
    return bundle
