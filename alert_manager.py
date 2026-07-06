"""
alert_manager.py
==================
Alert presentation. print_alert() is unchanged from uc09_v3.2. save_alert_json()
is new: it emits one JSON object per alert, suitable for a SOC pipeline
(SIEM ingestion, ticketing webhook, etc.) without changing what counts as
an alert or how severity/signals are computed.
"""

import json
import os
import time

from anomaly_detection import top_signals, worst_urls
from feature_extraction import WINDOW_MINUTES


def print_alert(row: dict, model_id: str = None) -> None:
    sig = top_signals(row)
    urls = worst_urls(row["_reqs"], n=3)  # console display only — capped for readability
    ts_str = row["window_ts"].strftime("%Y-%m-%d %H:%M UTC")

    print()
    print("=" * 70)
    print("  ALERT — UC-09 ANOMALOUS URL/QUERY TRAFFIC DETECTED")
    if model_id:
        print(f"  Detected by   : {model_id}")
    print(f"  Severity      : {row['severity']}")
    print(f"  Fused score   : {row['fused_score']:.3f}   GMM anomaly posterior: {row['anomaly_prob']:.3f}")
    print(f"  IP            : {row['ip']}")
    print(f"  Window        : {ts_str}  ({row['n_requests']} requests in {WINDOW_MINUTES} min)")
    print(f"  IF score      : {row['if_score']:.3f}  |  LOF score : {row['lof_score']:.3f}")
    print()
    print("  Key signals:")
    for s in sig:
        print(f"    • {s}")
    print()
    print("  Worst URLs in window (by statistical anomaly, not keyword match):")
    for u in urls:
        display = u if len(u) <= 100 else u[:97] + "..."
        print(f"    > {display}")
    print("=" * 70)


def alert_to_dict(row: dict, model_id: str = None) -> dict:
    """Flat, JSON-safe representation of one alert for SOC integration.
    Kept for backward compatibility (bundles multiple URLs into one
    alert) — prefer alerts_per_url() for one-alert-per-URL output."""
    return {
        "detected_by_model": model_id,
        "alert_kind": row.get("alert_kind"),
        "severity": row.get("severity"),
        "ip": row["ip"],
        "window_start_utc": row["window_ts"].isoformat(),
        "window_minutes": WINDOW_MINUTES,
        "n_requests": row["n_requests"],
        "if_score": row["if_score"],
        "lof_score": row["lof_score"],
        "fused_score": row["fused_score"],
        "anomaly_prob": row["anomaly_prob"],
        "signals": top_signals(row),
        "worst_urls": worst_urls(row["_reqs"]),
        "generated_at": time.time(),
    }


def alerts_per_url(row: dict, model_id: str = None, n_urls: int = None) -> list:
    """
    Same alert context (IP, window, scores, severity, signals) as
    alert_to_dict(), but returns ONE dict PER URL instead of bundling URLs
    into a single "worst_urls" list. Each dict has a single "url" field.
    By default (n_urls=None) every distinct anomalous URL in the window
    gets its own alert — no top-N cap, since this is real-time alerting,
    not a summary. Pass an explicit n_urls to cap the list if needed.
    """
    urls = worst_urls(row["_reqs"], n=n_urls)
    sig = top_signals(row)
    base = {
        "detected_by_model": model_id,
        "alert_kind": row.get("alert_kind"),
        "severity": row.get("severity"),
        "ip": row["ip"],
        "window_start_utc": row["window_ts"].isoformat(),
        "window_minutes": WINDOW_MINUTES,
        "n_requests": row["n_requests"],
        "if_score": row["if_score"],
        "lof_score": row["lof_score"],
        "fused_score": row["fused_score"],
        "relative_score": row.get("relative_score"),
        "anomaly_prob": row["anomaly_prob"],
        "signals": sig,
    }
    out = []
    for u in urls:
        d = dict(base)
        d["url"] = u
        d["generated_at"] = time.time()
        out.append(d)
    return out


def save_alert_json(rows: list, out_path: str, model_id: str = None) -> str:
    """
    Writes all given alert rows to a single JSON array file at out_path,
    for downstream SOC tooling to poll/consume. One entry per URL (not one
    entry per window with a bundled worst_urls list) — every alerting URL
    gets its own JSON object. Creates parent dirs as needed.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = []
    for r in rows:
        payload.extend(alerts_per_url(r, model_id=model_id))
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[alert_manager] {len(payload)} alert(s) written to {out_path}", flush=True)
    return out_path