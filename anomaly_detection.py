"""
anomaly_detection.py
======================
Scoring + explainability, unchanged from uc09_v3.2: IF/LOF fusion with the
legit-path discount, GMM posterior as an informational confidence score
only (never the alert gate itself), the two-level burst/catastrophic gate,
severity tiers, and the same signal/URL explainability helpers. This module
always operates against whatever model bundle realtime_pipeline.py hands
it — the CURRENTLY APPROVED one, per model_library.get_active_model().
"""

import time
import urllib.parse

import numpy as np
from sklearn.mixture import GaussianMixture

from feature_extraction import (
    to_matrix, _non_alnum_ratio, _transition_rate, MIN_LEN_FOR_RATIO,
    MIN_BREAK_URLS, MIN_BREAK_RATIO, WINDOW_MINUTES,
)


def score_rows(rows: list, scaler, iforest, lof, if_bounds, lof_bounds) -> list:
    if not rows:
        return []
    X = to_matrix(rows)
    X_scaled = scaler.transform(X)

    if_min, if_max = if_bounds
    lof_min, lof_max = lof_bounds

    if_raw = iforest.decision_function(X_scaled)
    if_score = np.clip(1 - (if_raw - if_min) / (if_max - if_min + 1e-9), 0.0, 1.0)

    lof_raw = lof.score_samples(X_scaled)
    lof_score = np.clip(1 - (lof_raw - lof_min) / (lof_max - lof_min + 1e-9), 0.0, 1.0)

    for i, row in enumerate(rows):
        legit_discount = 1.0 - 0.5 * row["legit_path_ratio"]
        fused = legit_discount * 0.5 * (if_score[i] + lof_score[i])
        row["if_score"] = round(float(if_score[i]), 4)
        row["lof_score"] = round(float(lof_score[i]), 4)
        row["fused_score"] = round(float(fused), 4)

    return rows


def gmm_anomaly_flags(fused_scores: np.ndarray):
    x = fused_scores.reshape(-1, 1)
    if len(x) < 4 or np.std(x) < 1e-9:
        return np.zeros(len(x), dtype=bool), np.zeros(len(x))

    gmm = GaussianMixture(n_components=2, random_state=42, n_init=5)
    gmm.fit(x)
    anomalous_component = int(np.argmax(gmm.means_.ravel()))
    posterior = gmm.predict_proba(x)[:, anomalous_component]
    flags = posterior > 0.5
    return flags, posterior


def apply_alert_gate(rows: list, fused_threshold: float, catastrophic_threshold: float) -> list:
    """
    Applies the same two-level gate as uc09_v3.2.main(), plus a break-ratio
    floor added on top of the burst gate: burst gate (fused >= train-derived
    threshold AND enough grammar-breaking URLs BOTH by absolute count
    [MIN_BREAK_URLS] AND by fraction of the window [MIN_BREAK_RATIO]) OR
    catastrophic gate (very high fused score AND a landed 5xx). The ratio
    floor exists because absolute count alone lets high-volume, low-signal
    recon/enumeration traffic (many requests, few of them actually
    grammar-breaking) clear the same raw count as a much smaller, genuinely
    surgical injection campaign. Also attaches the GMM posterior as an
    informational confidence score, plus a `relative_score` normalized
    against THIS model's own calibration:

        relative_score = (fused_score - fused_threshold)
                          / (catastrophic_threshold - fused_threshold)

    0.0 = just cleared this model's burst gate; 1.0 = at this model's
    catastrophic threshold; >1.0 = beyond it. Because every model's
    fused_threshold/catastrophic_threshold are derived the same way (99.7th
    / 99.99th percentile of ITS OWN training scores — see train_model.py),
    relative_score means roughly the same thing across model versions, so
    a single fixed cutoff on it stays meaningful even as models retrain
    and fused_score's raw scale drifts. Raw fused_score/severity do NOT
    have this property — see README "Comparing alerts across models".
    """
    if not rows:
        return rows
    fused = np.array([r["fused_score"] for r in rows])
    _, posterior = gmm_anomaly_flags(fused)
    denom = max(catastrophic_threshold - fused_threshold, 1e-9)
    for r, p in zip(rows, posterior):
        burst = (
            (r["fused_score"] >= fused_threshold)
            and (r["break_count"] >= MIN_BREAK_URLS)
            and (r["break_ratio"] >= MIN_BREAK_RATIO)
        )
        catastrophic = (r["fused_score"] >= catastrophic_threshold) and (r["status_500_ratio"] > 0)
        r["anomaly_flag"] = bool(burst or catastrophic)
        r["alert_kind"] = "catastrophic" if (catastrophic and not burst) else "burst"
        r["anomaly_prob"] = round(float(p), 4)
        r["relative_score"] = round(float((r["fused_score"] - fused_threshold) / denom), 4)
    return rows


def assign_severities(alert_rows: list) -> None:
    if not alert_rows:
        return
    scores = np.array([r["fused_score"] for r in alert_rows])
    t33, t66 = np.percentile(scores, [33, 66])
    for r in alert_rows:
        s = r["fused_score"]
        if s >= t66:
            r["severity"] = "HIGH"
        elif s >= t33:
            r["severity"] = "MEDIUM"
        else:
            r["severity"] = "LOW"


def top_signals(row: dict) -> list:
    signals = []
    if row.get("break_count", 0) > 0:
        signals.append(f"grammar-breaking URLs in window: {row['break_count']} of {row['n_requests']} "
                        f"({row['break_ratio']*100:.0f}%)")
    if row.get("rare_path_count", 0) > 0:
        signals.append(f"never-before-seen paths in window: {row['rare_path_count']} of {row['n_requests']}")
    if row["query_non_alnum_max"] > 0.15:
        signals.append(f"query non-alphanumeric char ratio (max): {row['query_non_alnum_max']:.3f}")
    if row["path_non_alnum_max"] > 0.15:
        signals.append(f"path non-alphanumeric char ratio (max): {row['path_non_alnum_max']:.3f}")
    if row["query_encoded_ratio"] > 0.08:
        signals.append(f"query pct-encoded char ratio: {row['query_encoded_ratio']:.3f}")
    if row["q_len_max"] > 200:
        signals.append(f"max query length: {row['q_len_max']:.0f} chars")
    if row["token_entropy_max"] > 4.0:
        signals.append(f"query token entropy (max): {row['token_entropy_max']:.2f}")
    if row["max_param_entropy"] > 4.0:
        signals.append(f"single-parameter entropy (max): {row['max_param_entropy']:.2f}")
    if row["transition_rate_max"] > 0.5:
        signals.append(f"char-class transition rate (max): {row['transition_rate_max']:.2f} (syntax-like structure)")
    if row["status_500_ratio"] > 0.05:
        signals.append(f"500-error ratio: {row['status_500_ratio']:.2f} (possible landed payload)")
    if row["status_4xx_ratio"] > 0.6:
        signals.append(f"4xx-error ratio: {row['status_4xx_ratio']:.2f}")
    if row["query_diversity"] > 0.8:
        signals.append(f"query diversity: {row['query_diversity']:.2f} (many unique payload shapes)")
    if row["path_rarity_max"] > row.get("_train_rarity_p95", float("inf")):
        signals.append(f"path rarity (max): {row['path_rarity_max']:.2f} (rare vs. training traffic)")
    if row["qlen_z_max"] > 3.0:
        signals.append(f"query length deviation for this path: {row['qlen_z_max']:.1f} std devs from this path's norm")
    if row["entropy_z_max"] > 3.0:
        signals.append(f"query entropy deviation for this path: {row['entropy_z_max']:.1f} std devs from this path's norm")
    if row["burst_rate"] > 20:
        signals.append(f"burst rate: {row['burst_rate']:.1f} req/min")
    return signals or ["general statistical URL anomaly — no single dominant signal"]


def worst_urls(reqs: list, n: int = None) -> list:
    """
    Returns URLs from this window's requests, ranked by anomaly score
    (most suspicious first), with exact duplicate URLs collapsed to one
    entry. By default (n=None) returns ALL distinct URLs — no top-N cap —
    since for real-time alerting every anomalous URL should get its own
    alert, not just the top few. Pass an explicit n to cap the list.
    """
    def score_req(r):
        q = r["query"] or ""
        p = r["path"]
        dq = urllib.parse.unquote(q)
        dp = urllib.parse.unquote(p)
        q_ratio = _non_alnum_ratio(dq) if len(dq) >= MIN_LEN_FOR_RATIO else 0.0
        p_ratio = _non_alnum_ratio(dp) if len(dp) >= MIN_LEN_FOR_RATIO else 0.0
        trans = _transition_rate(dq) if dq else 0.0
        return q_ratio * 20 + p_ratio * 10 + trans * 15 + (q + p).count("%") * 0.5 + (len(dq) > 200) * 5

    ranked = sorted(reqs, key=score_req, reverse=True)

    seen = set()
    urls = []
    for r in ranked:
        url = r["path"] + ("?" + r["query"] if r["query"] else "")
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)

    return urls if n is None else urls[:n]