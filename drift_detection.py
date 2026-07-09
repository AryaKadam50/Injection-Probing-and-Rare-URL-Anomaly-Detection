"""
drift_detection.py
====================
Compares a candidate model against the currently approved model along three
axes — feature drift, model (score) drift, threshold drift — and produces a
single drift_score plus a requires_approval flag.

This does not change how IsolationForest/LOF/GMM score anything; it only
looks at the *distributions* each model was fit against / produced, using
the "_drift_reference" arrays train_model.py stows in every bundle
(train_matrix + fused/IF/LOF train scores).

DRIFT_APPROVAL_THRESHOLD is the single knob that decides "auto-promote"
vs. "ask a human". Raised from 0.50 -> 0.59.
"""

import numpy as np

from feature_extraction import FEATURE_COLS

DRIFT_APPROVAL_THRESHOLD = 0.59

# Weights for combining the three drift axes into one score. Feature drift
# gets the most weight since it's the earliest, most interpretable signal
# that the traffic distribution itself has moved; score/threshold drift
# matter more once feature drift is already borderline.
WEIGHT_FEATURE = 0.5
WEIGHT_SCORE = 0.3
WEIGHT_THRESHOLD = 0.2

_EPS = 1e-9


def _psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index between two 1-D samples."""
    breakpoints = np.linspace(0, 100, buckets + 1)
    edges = np.unique(np.percentile(expected, breakpoints))
    if len(edges) < 3:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)

    exp_pct = exp_counts / max(exp_counts.sum(), 1)
    act_pct = act_counts / max(act_counts.sum(), 1)
    exp_pct = np.where(exp_pct == 0, _EPS, exp_pct)
    act_pct = np.where(act_pct == 0, _EPS, act_pct)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _kl_divergence(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """KL(actual || expected) over the same histogram binning as PSI."""
    breakpoints = np.linspace(0, 100, buckets + 1)
    edges = np.unique(np.percentile(expected, breakpoints))
    if len(edges) < 3:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)

    exp_pct = exp_counts / max(exp_counts.sum(), 1)
    act_pct = act_counts / max(act_counts.sum(), 1)
    exp_pct = np.where(exp_pct == 0, _EPS, exp_pct)
    act_pct = np.where(act_pct == 0, _EPS, act_pct)

    return float(np.sum(act_pct * np.log(act_pct / exp_pct)))


def compute_feature_drift(approved_bundle: dict, candidate_bundle: dict) -> dict:
    """
    Per-feature PSI + KL divergence between the approved model's training
    matrix and the candidate's training matrix (i.e. last week's traffic
    shape vs. this week's). Returns per-feature scores plus a single
    aggregate (mean PSI, normalized to roughly 0-1 via a soft cap at PSI=1).
    """
    approved_ref = approved_bundle.get("_drift_reference")
    candidate_ref = candidate_bundle.get("_drift_reference")
    if approved_ref is None or candidate_ref is None:
        return {"per_feature": {}, "aggregate_psi": 0.0, "aggregate_kl": 0.0, "note": "no drift reference stored"}

    X_old = approved_ref["train_matrix"]
    X_new = candidate_ref["train_matrix"]

    per_feature = {}
    psis, kls = [], []
    for i, col in enumerate(FEATURE_COLS):
        psi = _psi(X_old[:, i], X_new[:, i])
        kl = _kl_divergence(X_old[:, i], X_new[:, i])
        per_feature[col] = {"psi": round(psi, 4), "kl": round(kl, 4)}
        psis.append(psi)
        kls.append(kl)

    aggregate_psi = float(np.mean(psis))
    aggregate_kl = float(np.mean(kls))
    return {
        "per_feature": per_feature,
        "aggregate_psi": aggregate_psi,
        "aggregate_kl": aggregate_kl,
        # Soft-capped 0-1 normalization: PSI >= 0.5 is already "major shift"
        # by common convention (0.1 = moderate, 0.25 = major), so cap there.
        "normalized": min(aggregate_psi / 0.5, 1.0),
    }


def compute_score_drift(approved_bundle: dict, candidate_bundle: dict) -> dict:
    """PSI between IF / LOF / fused score distributions of the two models' training sets."""
    approved_ref = approved_bundle.get("_drift_reference")
    candidate_ref = candidate_bundle.get("_drift_reference")
    if approved_ref is None or candidate_ref is None:
        return {"if_psi": 0.0, "lof_psi": 0.0, "fused_psi": 0.0, "normalized": 0.0}

    if_psi = _psi(approved_ref["if_train_scores"], candidate_ref["if_train_scores"])
    lof_psi = _psi(approved_ref["lof_train_scores"], candidate_ref["lof_train_scores"])
    fused_psi = _psi(approved_ref["fused_train_scores"], candidate_ref["fused_train_scores"])

    aggregate = float(np.mean([if_psi, lof_psi, fused_psi]))
    return {
        "if_psi": round(if_psi, 4),
        "lof_psi": round(lof_psi, 4),
        "fused_psi": round(fused_psi, 4),
        "normalized": min(aggregate / 0.5, 1.0),
    }


def compute_threshold_drift(approved_bundle: dict, candidate_bundle: dict) -> dict:
    """Relative change in fused_threshold / catastrophic_threshold between models."""
    old_fused = approved_bundle["fused_threshold"]
    new_fused = candidate_bundle["fused_threshold"]
    old_cat = approved_bundle["catastrophic_threshold"]
    new_cat = candidate_bundle["catastrophic_threshold"]

    fused_rel_change = abs(new_fused - old_fused) / max(abs(old_fused), _EPS)
    cat_rel_change = abs(new_cat - old_cat) / max(abs(old_cat), _EPS)
    aggregate = float(np.mean([fused_rel_change, cat_rel_change]))

    return {
        "old_fused_threshold": old_fused,
        "new_fused_threshold": new_fused,
        "old_catastrophic_threshold": old_cat,
        "new_catastrophic_threshold": new_cat,
        "fused_rel_change": round(fused_rel_change, 4),
        "catastrophic_rel_change": round(cat_rel_change, 4),
        # A doubling (100% relative change) is treated as maximal drift.
        "normalized": min(aggregate / 1.0, 1.0),
    }


def compare_models(approved_bundle: dict, candidate_bundle: dict,
                    approval_threshold: float = DRIFT_APPROVAL_THRESHOLD) -> dict:
    """
    Top-level entry point. Combines feature drift, score drift, and
    threshold drift into one drift_score in [0, 1], and decides whether the
    change is small enough to auto-promote or needs human review.
    """
    feature_drift = compute_feature_drift(approved_bundle, candidate_bundle)
    score_drift = compute_score_drift(approved_bundle, candidate_bundle)
    threshold_drift = compute_threshold_drift(approved_bundle, candidate_bundle)

    drift_score = (
        WEIGHT_FEATURE * feature_drift["normalized"] +
        WEIGHT_SCORE * score_drift["normalized"] +
        WEIGHT_THRESHOLD * threshold_drift["normalized"]
    )
    drift_score = round(float(drift_score), 4)

    return {
        "drift_score": drift_score,
        "requires_approval": drift_score > approval_threshold,
        "approval_threshold": approval_threshold,
        "feature_drift": feature_drift,
        "score_drift": score_drift,
        "threshold_drift": threshold_drift,
    }    edges = np.unique(np.percentile(expected, breakpoints))
    if len(edges) < 3:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)

    exp_pct = exp_counts / max(exp_counts.sum(), 1)
    act_pct = act_counts / max(act_counts.sum(), 1)
    exp_pct = np.where(exp_pct == 0, _EPS, exp_pct)
    act_pct = np.where(act_pct == 0, _EPS, act_pct)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _kl_divergence(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """KL(actual || expected) over the same histogram binning as PSI."""
    breakpoints = np.linspace(0, 100, buckets + 1)
    edges = np.unique(np.percentile(expected, breakpoints))
    if len(edges) < 3:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)

    exp_pct = exp_counts / max(exp_counts.sum(), 1)
    act_pct = act_counts / max(act_counts.sum(), 1)
    exp_pct = np.where(exp_pct == 0, _EPS, exp_pct)
    act_pct = np.where(act_pct == 0, _EPS, act_pct)

    return float(np.sum(act_pct * np.log(act_pct / exp_pct)))


def compute_feature_drift(approved_bundle: dict, candidate_bundle: dict) -> dict:
    """
    Per-feature PSI + KL divergence between the approved model's training
    matrix and the candidate's training matrix (i.e. last week's traffic
    shape vs. this week's). Returns per-feature scores plus a single
    aggregate (mean PSI, normalized to roughly 0-1 via a soft cap at PSI=1).
    """
    approved_ref = approved_bundle.get("_drift_reference")
    candidate_ref = candidate_bundle.get("_drift_reference")
    if approved_ref is None or candidate_ref is None:
        return {"per_feature": {}, "aggregate_psi": 0.0, "aggregate_kl": 0.0, "note": "no drift reference stored"}

    X_old = approved_ref["train_matrix"]
    X_new = candidate_ref["train_matrix"]

    per_feature = {}
    psis, kls = [], []
    for i, col in enumerate(FEATURE_COLS):
        psi = _psi(X_old[:, i], X_new[:, i])
        kl = _kl_divergence(X_old[:, i], X_new[:, i])
        per_feature[col] = {"psi": round(psi, 4), "kl": round(kl, 4)}
        psis.append(psi)
        kls.append(kl)

    aggregate_psi = float(np.mean(psis))
    aggregate_kl = float(np.mean(kls))
    return {
        "per_feature": per_feature,
        "aggregate_psi": aggregate_psi,
        "aggregate_kl": aggregate_kl,
        # Soft-capped 0-1 normalization: PSI >= 0.5 is already "major shift"
        # by common convention (0.1 = moderate, 0.25 = major), so cap there.
        "normalized": min(aggregate_psi / 0.5, 1.0),
    }


def compute_score_drift(approved_bundle: dict, candidate_bundle: dict) -> dict:
    """PSI between IF / LOF / fused score distributions of the two models' training sets."""
    approved_ref = approved_bundle.get("_drift_reference")
    candidate_ref = candidate_bundle.get("_drift_reference")
    if approved_ref is None or candidate_ref is None:
        return {"if_psi": 0.0, "lof_psi": 0.0, "fused_psi": 0.0, "normalized": 0.0}

    if_psi = _psi(approved_ref["if_train_scores"], candidate_ref["if_train_scores"])
    lof_psi = _psi(approved_ref["lof_train_scores"], candidate_ref["lof_train_scores"])
    fused_psi = _psi(approved_ref["fused_train_scores"], candidate_ref["fused_train_scores"])

    aggregate = float(np.mean([if_psi, lof_psi, fused_psi]))
    return {
        "if_psi": round(if_psi, 4),
        "lof_psi": round(lof_psi, 4),
        "fused_psi": round(fused_psi, 4),
        "normalized": min(aggregate / 0.5, 1.0),
    }


def compute_threshold_drift(approved_bundle: dict, candidate_bundle: dict) -> dict:
    """Relative change in fused_threshold / catastrophic_threshold between models."""
    old_fused = approved_bundle["fused_threshold"]
    new_fused = candidate_bundle["fused_threshold"]
    old_cat = approved_bundle["catastrophic_threshold"]
    new_cat = candidate_bundle["catastrophic_threshold"]

    fused_rel_change = abs(new_fused - old_fused) / max(abs(old_fused), _EPS)
    cat_rel_change = abs(new_cat - old_cat) / max(abs(old_cat), _EPS)
    aggregate = float(np.mean([fused_rel_change, cat_rel_change]))

    return {
        "old_fused_threshold": old_fused,
        "new_fused_threshold": new_fused,
        "old_catastrophic_threshold": old_cat,
        "new_catastrophic_threshold": new_cat,
        "fused_rel_change": round(fused_rel_change, 4),
        "catastrophic_rel_change": round(cat_rel_change, 4),
        # A doubling (100% relative change) is treated as maximal drift.
        "normalized": min(aggregate / 1.0, 1.0),
    }


def compare_models(approved_bundle: dict, candidate_bundle: dict,
                    approval_threshold: float = DRIFT_APPROVAL_THRESHOLD) -> dict:
    """
    Top-level entry point. Combines feature drift, score drift, and
    threshold drift into one drift_score in [0, 1], and decides whether the
    change is small enough to auto-promote or needs human review.
    """
    feature_drift = compute_feature_drift(approved_bundle, candidate_bundle)
    score_drift = compute_score_drift(approved_bundle, candidate_bundle)
    threshold_drift = compute_threshold_drift(approved_bundle, candidate_bundle)

    drift_score = (
        WEIGHT_FEATURE * feature_drift["normalized"] +
        WEIGHT_SCORE * score_drift["normalized"] +
        WEIGHT_THRESHOLD * threshold_drift["normalized"]
    )
    drift_score = round(float(drift_score), 4)

    return {
        "drift_score": drift_score,
        "requires_approval": drift_score > approval_threshold,
        "approval_threshold": approval_threshold,
        "feature_drift": feature_drift,
        "score_drift": score_drift,
        "threshold_drift": threshold_drift,
    }
