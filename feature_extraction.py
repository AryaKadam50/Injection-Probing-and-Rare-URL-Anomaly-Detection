"""
feature_extraction.py
======================
UC-09 feature engineering, extracted unchanged from uc09_v3.2.py.

Nothing about the detection math has been altered: same normalize_path(),
same normalize_query_shape(), same per-path baselines, same
build_entity_windows() aggregation, same FEATURE_COLS. This module is a
straight lift so the realtime pipeline and the batch trainer score
identically.

Only additions vs. the original file:
  - CONFIG is exposed as a dict (CONFIG_SNAPSHOT()) so model_library.py /
    drift_detection.py can fingerprint config drift, same as the original
    _current_config() did.
  - load_logs() is unchanged but scheduler.py additionally exposes a
    streaming-friendly `parse_event` for line-by-line ingestion instead of
    whole-file batch loads.
"""

import json
import math
import re
import string
import time
import urllib.parse
from collections import defaultdict
from datetime import datetime

import numpy as np

# ──────────────────────────────────────────────
# CONFIG (unchanged from uc09_v3.2)
# ──────────────────────────────────────────────
MODEL_VERSION = "uc09-v3.2"

WINDOW_MINUTES = 60
MIN_REQUESTS = 20
LOF_NEIGHBORS = 20

SCORE_BOUND_PCT_LOW = 0.5
SCORE_BOUND_PCT_HIGH = 99.5

MIN_PATH_SAMPLES_FOR_BASELINE = 20
MIN_LEN_FOR_RATIO = 8
# Raised from 10 -> 25: a window now needs a much larger volume of
# grammar-breaking URLs before the burst gate can even fire. This targets
# small/isolated probes and reduces alert volume at the gate itself,
# independent of the percentile thresholds below.
MIN_BREAK_URLS = 25
# New: minimum FRACTION of a window's requests that must be grammar-breaking
# for the burst gate to fire, in addition to the absolute MIN_BREAK_URLS
# count above. Absolute count alone can't distinguish a small, surgical
# injection campaign (e.g. 363 breaks out of 1,759 requests = 21%) from
# high-volume recon/enumeration noise that happens to cross the same raw
# count purely on volume (e.g. 63 breaks out of 38,377 requests = 0.16%).
# Both MIN_BREAK_URLS and MIN_BREAK_RATIO must be satisfied.
MIN_BREAK_RATIO = 0.05

BREAK_Z_THRESH = 3.0

# Raised from 99.7 -> 99.9: fewer training windows clear the burst gate,
# so fused_threshold sits further out in the tail and alert volume drops
# at the source (every future retrained model will use the new cutoff).
ALERT_PCT = 99.9
# Left at 99.99 by default. Only push this higher (e.g. 99.995) if your
# training window has enough rows for that percentile to be stable --
# check n_train_windows in the candidate's metadata first; with too few
# samples the extreme tail estimate becomes noisy rather than meaningful.
CATASTROPHIC_PCT = 99.99

Z_CLIP = 20.0

LOG_EVERY_LINES = 1_000_000
LOG_EVERY_RECORDS = 1_000_000

SYNTHETIC_CAMPAIGN_SIZE = MIN_REQUESTS + 10  # kept for parity with batch tool

PATH_ID_PATTERN = re.compile(r"\d+")
PARAM_NUMERIC_PATTERN = re.compile(r"^\d+$")
_ALNUM = set(string.ascii_letters + string.digits)

FEATURE_COLS = [
    "path_non_alnum_mean", "path_non_alnum_max",
    "query_non_alnum_mean", "query_non_alnum_max",
    "encoded_ratio", "query_encoded_ratio",
    "q_len_mean", "q_len_max",
    "token_entropy_mean", "token_entropy_max", "max_param_entropy",
    "transition_rate_mean", "transition_rate_max",
    "param_count",
    "status_500_ratio", "status_4xx_ratio",
    "query_diversity",
    "path_rarity_mean", "path_rarity_max",
    "qlen_z_max", "entropy_z_max",
    "break_ratio", "rare_path_ratio",
    "burst_rate",
]


def current_config() -> dict:
    """Config fingerprint used for drift-warning / bundle-compat checks."""
    return {
        "WINDOW_MINUTES": WINDOW_MINUTES,
        "MIN_REQUESTS": MIN_REQUESTS,
        "LOF_NEIGHBORS": LOF_NEIGHBORS,
        "BREAK_Z_THRESH": BREAK_Z_THRESH,
        "MIN_PATH_SAMPLES_FOR_BASELINE": MIN_PATH_SAMPLES_FOR_BASELINE,
        "MIN_LEN_FOR_RATIO": MIN_LEN_FOR_RATIO,
        "MIN_BREAK_URLS": MIN_BREAK_URLS,
        "MIN_BREAK_RATIO": MIN_BREAK_RATIO,
        "ALERT_PCT": ALERT_PCT,
        "CATASTROPHIC_PCT": CATASTROPHIC_PCT,
        "Z_CLIP": Z_CLIP,
        "SCORE_BOUND_PCT_LOW": SCORE_BOUND_PCT_LOW,
        "SCORE_BOUND_PCT_HIGH": SCORE_BOUND_PCT_HIGH,
    }


# ──────────────────────────────────────────────
# LOG PARSING (unchanged)
# ──────────────────────────────────────────────

def parse_event(ev: dict):
    ts_raw = ev.get("@timestamp", "")
    try:
        ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
    except Exception:
        return None

    url_block = ev.get("url", {}) or {}
    http_block = ev.get("http", {}) or {}
    req_block = http_block.get("request", {}) or {}
    resp_block = http_block.get("response", {}) or {}
    body_block = resp_block.get("body", {}) or {}

    path = url_block.get("path") or ""
    query = url_block.get("query") or ""

    return {
        "ts": ts,
        "ip": (ev.get("source", {}) or {}).get("ip", ""),
        "method": req_block.get("method", "GET"),
        "path": path,
        "query": query,
        "status": resp_block.get("status_code", 0),
        "bytes": body_block.get("bytes", 0),
        "ua": (ev.get("user_agent", {}) or {}).get("original", ""),
        "referrer": req_block.get("referrer", "") or "",
    }


def detect_format(filepath: str) -> str:
    with open(filepath, "r") as f:
        while True:
            ch = f.read(1)
            if not ch:
                return "empty"
            if ch.isspace():
                continue
            return "array" if ch == "[" else "ndjson"


def load_logs(filepath: str, label: str = "") -> list:
    tag = f"[{label}] " if label else ""
    fmt = detect_format(filepath)
    print(f"{tag}[~] Detected format: {fmt} ({filepath})", flush=True)

    if fmt == "empty":
        print(f"{tag}[!] File is empty.")
        return []

    t0 = time.time()
    records = []
    bad = 0

    if fmt == "array":
        with open(filepath, "r") as f:
            data = json.load(f)
        total = len(data)
        for i, ev in enumerate(data, 1):
            r = parse_event(ev)
            if r and r["ip"]:
                records.append(r)
            else:
                bad += 1
            if i % LOG_EVERY_LINES == 0:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(f"{tag}[~] Parsed {i:,}/{total:,} | valid {len(records):,} | "
                      f"bad {bad:,} | {rate:,.0f}/sec", flush=True)
    else:
        lines_seen = 0
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                lines_seen += 1
                try:
                    ev = json.loads(line)
                    r = parse_event(ev)
                except json.JSONDecodeError:
                    r = None
                if r and r["ip"]:
                    records.append(r)
                else:
                    bad += 1
                if lines_seen % LOG_EVERY_LINES == 0:
                    elapsed = time.time() - t0
                    rate = lines_seen / elapsed if elapsed > 0 else 0
                    print(f"{tag}[~] Read {lines_seen:,} lines | valid {len(records):,} | "
                          f"bad {bad:,} | {rate:,.0f} lines/sec", flush=True)

    records.sort(key=lambda x: x["ts"])
    elapsed = time.time() - t0
    print(f"{tag}[+] Loaded {len(records):,} valid log entries ({bad:,} bad) in {elapsed:,.1f}s", flush=True)
    return records


# ──────────────────────────────────────────────
# NORMALIZATION HELPERS (unchanged)
# ──────────────────────────────────────────────

def normalize_path(path: str) -> str:
    return PATH_ID_PATTERN.sub("#", path)


def normalize_query_shape(query: str) -> str:
    if not query:
        return query
    parts = query.split("&")
    out = []
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            if PARAM_NUMERIC_PATTERN.match(v):
                v = "#"
            out.append(f"{k}={v}")
        else:
            out.append(p)
    return "&".join(out)


def _string_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = defaultdict(int)
    for c in s:
        freq[c] += 1
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())


def _non_alnum_ratio(s: str) -> float:
    if not s:
        return 0.0
    non_alnum = sum(1 for c in s if c not in _ALNUM)
    return non_alnum / len(s)


def _char_class(c: str) -> int:
    return 0 if c in _ALNUM else 1


def _transition_rate(s: str) -> float:
    if len(s) < 2:
        return 0.0
    transitions = sum(1 for i in range(len(s) - 1) if _char_class(s[i]) != _char_class(s[i + 1]))
    return transitions / (len(s) - 1)


def _max_param_entropy(query: str) -> float:
    if not query:
        return 0.0
    parts = query.split("&")
    entropies = []
    for p in parts:
        val = p.split("=", 1)[1] if "=" in p else p
        if val:
            entropies.append(_string_entropy(val))
    return max(entropies) if entropies else 0.0


# ──────────────────────────────────────────────
# PATH POPULARITY / RARITY (unchanged)
# ──────────────────────────────────────────────

def build_path_popularity(records: list):
    counts = defaultdict(int)
    for r in records:
        counts[normalize_path(r["path"])] += 1
    total = sum(counts.values())

    unique_rarities = [-math.log10((c + 1) / (total + 1)) for c in counts.values()]
    common_cutoff = float(np.median(unique_rarities)) if unique_rarities else 0.0

    print(f"[~] Path popularity table: {len(counts):,} unique normalized paths from "
          f"{total:,} requests | common-rarity cutoff (median): {common_cutoff:.3f}", flush=True)
    return dict(counts), total, common_cutoff


def path_rarity(path: str, path_freq: dict, total: int) -> float:
    count = path_freq.get(normalize_path(path), 0)
    return -math.log10((count + 1) / (total + 1))


# ──────────────────────────────────────────────
# PER-REQUEST URL FEATURES (unchanged)
# ──────────────────────────────────────────────

def url_features(path: str, query: str, path_freq: dict, total: int, path_baseline: dict) -> dict:
    decoded_path = urllib.parse.unquote(path)
    decoded_query = urllib.parse.unquote(query) if query else ""

    path_non_alnum_ratio = _non_alnum_ratio(decoded_path) if len(decoded_path) >= MIN_LEN_FOR_RATIO else 0.0
    query_non_alnum_ratio = _non_alnum_ratio(decoded_query) if len(decoded_query) >= MIN_LEN_FOR_RATIO else 0.0

    full = path + ("?" + query if query else "")
    encoded_count = full.count("%")
    encoded_ratio = encoded_count / max(len(full), 1)
    query_encoded_ratio = (query.count("%") / max(len(query), 1)) if query else 0.0

    q_len = len(query)
    param_count = len(query.split("&")) if query else 0
    token_entropy = _string_entropy(decoded_query) if decoded_query else 0.0
    max_param_entropy = _max_param_entropy(decoded_query)
    transition_rate = _transition_rate(decoded_query) if decoded_query else 0.0
    rarity = path_rarity(path, path_freq, total)

    p = normalize_path(path)
    struct_base = path_baseline["struct_per_path"].get(p, path_baseline["struct_global"])
    non_alnum_z = min(abs(query_non_alnum_ratio - struct_base["non_alnum_mean"]) / struct_base["non_alnum_std"], Z_CLIP)
    transition_z = min(abs(transition_rate - struct_base["transition_mean"]) / struct_base["transition_std"], Z_CLIP)

    is_break = (
        (len(decoded_query) >= MIN_LEN_FOR_RATIO and non_alnum_z > BREAK_Z_THRESH) or
        (len(decoded_query) >= MIN_LEN_FOR_RATIO and transition_z > BREAK_Z_THRESH) or
        (q_len > 200) or
        (len(decoded_path) >= MIN_LEN_FOR_RATIO and path_non_alnum_ratio > 0.30)
    )

    is_rare_path = (
        path_freq.get(normalize_path(path), 0) == 0
        and len(decoded_path) >= MIN_LEN_FOR_RATIO
    )

    return {
        "path_non_alnum_ratio": path_non_alnum_ratio,
        "query_non_alnum_ratio": query_non_alnum_ratio,
        "encoded_ratio": encoded_ratio,
        "query_encoded_ratio": query_encoded_ratio,
        "q_len": q_len,
        "param_count": param_count,
        "token_entropy": token_entropy,
        "max_param_entropy": max_param_entropy,
        "transition_rate": transition_rate,
        "path_rarity": rarity,
        "is_break": is_break,
        "is_rare_path": is_rare_path,
    }


# ──────────────────────────────────────────────
# PER-PATH QUERY BASELINE (unchanged)
# ──────────────────────────────────────────────

def build_path_query_baseline(records: list) -> dict:
    per_path_qlens = defaultdict(list)
    per_path_entropy = defaultdict(list)
    per_path_non_alnum = defaultdict(list)
    per_path_transition = defaultdict(list)
    all_qlens, all_entropy, all_non_alnum, all_transition = [], [], [], []

    for r in records:
        p = normalize_path(r["path"])
        q = r["query"] or ""
        dq = urllib.parse.unquote(q) if q else ""
        qlen = len(q)
        ent = _string_entropy(dq) if dq else 0.0
        na = _non_alnum_ratio(dq) if len(dq) >= MIN_LEN_FOR_RATIO else 0.0
        tr = _transition_rate(dq) if dq else 0.0

        per_path_qlens[p].append(qlen)
        per_path_entropy[p].append(ent)
        per_path_non_alnum[p].append(na)
        per_path_transition[p].append(tr)
        all_qlens.append(qlen)
        all_entropy.append(ent)
        all_non_alnum.append(na)
        all_transition.append(tr)

    baseline = {}
    struct_per_path = {}
    for p in per_path_qlens:
        n = len(per_path_qlens[p])
        if n >= MIN_PATH_SAMPLES_FOR_BASELINE:
            baseline[p] = {
                "n": n,
                "qlen_mean": float(np.mean(per_path_qlens[p])),
                "qlen_std": max(float(np.std(per_path_qlens[p])), 1.0),
                "entropy_mean": float(np.mean(per_path_entropy[p])),
                "entropy_std": max(float(np.std(per_path_entropy[p])), 0.5),
            }
            struct_per_path[p] = {
                "non_alnum_mean": float(np.mean(per_path_non_alnum[p])),
                "non_alnum_std": max(float(np.std(per_path_non_alnum[p])), 0.05),
                "transition_mean": float(np.mean(per_path_transition[p])),
                "transition_std": max(float(np.std(per_path_transition[p])), 0.05),
            }

    global_baseline = {
        "qlen_mean": float(np.mean(all_qlens)) if all_qlens else 0.0,
        "qlen_std": max(float(np.std(all_qlens)), 1.0) if all_qlens else 1.0,
        "entropy_mean": float(np.mean(all_entropy)) if all_entropy else 0.0,
        "entropy_std": max(float(np.std(all_entropy)), 0.5) if all_entropy else 0.5,
    }
    struct_global = {
        "non_alnum_mean": float(np.mean(all_non_alnum)) if all_non_alnum else 0.0,
        "non_alnum_std": max(float(np.std(all_non_alnum)), 0.05) if all_non_alnum else 0.05,
        "transition_mean": float(np.mean(all_transition)) if all_transition else 0.0,
        "transition_std": max(float(np.std(all_transition)), 0.05) if all_transition else 0.05,
    }

    print(f"[~] Per-path query baseline: {len(baseline):,} paths with >= "
          f"{MIN_PATH_SAMPLES_FOR_BASELINE} TRAIN samples (of {len(per_path_qlens):,} unique paths)", flush=True)

    return {
        "per_path": baseline,
        "global": global_baseline,
        "struct_per_path": struct_per_path,
        "struct_global": struct_global,
    }


def query_deviation(path: str, query: str, path_baseline: dict):
    p = normalize_path(path)
    base = path_baseline["per_path"].get(p, path_baseline["global"])

    qlen = len(query) if query else 0
    ent = _string_entropy(urllib.parse.unquote(query)) if query else 0.0

    qlen_z = min(abs(qlen - base["qlen_mean"]) / base["qlen_std"], Z_CLIP)
    ent_z = min(abs(ent - base["entropy_mean"]) / base["entropy_std"], Z_CLIP)
    return qlen_z, ent_z


# ──────────────────────────────────────────────
# ENTITY-WINDOW AGGREGATION (unchanged)
# ──────────────────────────────────────────────

def build_entity_windows(records: list, path_freq: dict, total: int, common_cutoff: float,
                          path_baseline: dict, label: str = "") -> list:
    tag = f"[{label}] " if label else ""
    t0 = time.time()
    buckets = defaultdict(list)
    n_records = len(records)

    for r in records:
        window_key = r["ts"].replace(
            minute=(r["ts"].minute // WINDOW_MINUTES) * WINDOW_MINUTES,
            second=0, microsecond=0
        )
        buckets[(r["ip"], window_key)].append(r)

    rows = []
    for (ip, window_ts), reqs in buckets.items():
        if len(reqs) < MIN_REQUESTS:
            continue

        per_req_feats = [url_features(r["path"], r["query"], path_freq, total, path_baseline) for r in reqs]
        n = len(reqs)

        path_non_alnum_mean = np.mean([f["path_non_alnum_ratio"] for f in per_req_feats])
        path_non_alnum_max = np.max([f["path_non_alnum_ratio"] for f in per_req_feats])
        query_non_alnum_mean = np.mean([f["query_non_alnum_ratio"] for f in per_req_feats])
        query_non_alnum_max = np.max([f["query_non_alnum_ratio"] for f in per_req_feats])
        encoded_ratio_mean = np.mean([f["encoded_ratio"] for f in per_req_feats])
        query_encoded_mean = np.mean([f["query_encoded_ratio"] for f in per_req_feats])
        q_len_mean = np.mean([f["q_len"] for f in per_req_feats])
        q_len_max = np.max([f["q_len"] for f in per_req_feats])
        token_entropy_mean = np.mean([f["token_entropy"] for f in per_req_feats])
        token_entropy_max = np.max([f["token_entropy"] for f in per_req_feats])
        max_param_entropy_w = np.max([f["max_param_entropy"] for f in per_req_feats])
        transition_rate_mean = np.mean([f["transition_rate"] for f in per_req_feats])
        transition_rate_max = np.max([f["transition_rate"] for f in per_req_feats])
        param_count_mean = np.mean([f["param_count"] for f in per_req_feats])
        path_rarity_mean = np.mean([f["path_rarity"] for f in per_req_feats])
        path_rarity_max = np.max([f["path_rarity"] for f in per_req_feats])

        deviations = [query_deviation(r["path"], r["query"], path_baseline) for r in reqs]
        qlen_z_max = max(d[0] for d in deviations) if deviations else 0.0
        entropy_z_max = max(d[1] for d in deviations) if deviations else 0.0

        break_count = int(sum(f["is_break"] for f in per_req_feats))
        break_ratio = break_count / n
        rare_path_count = int(sum(f["is_rare_path"] for f in per_req_feats))
        rare_path_ratio = rare_path_count / n

        statuses = [r["status"] for r in reqs]
        status_500_ratio = sum(1 for s in statuses if s == 500) / n
        status_4xx_ratio = sum(1 for s in statuses if 400 <= s < 500) / n

        unique_queries = len(set(normalize_query_shape(r["query"]) for r in reqs if r["query"]))
        query_diversity = unique_queries / n

        legit_path_ratio = sum(f["path_rarity"] < common_cutoff for f in per_req_feats) / n
        burst_rate = n / WINDOW_MINUTES

        rows.append({
            "ip": ip,
            "window_ts": window_ts,
            "n_requests": n,
            "path_non_alnum_mean": path_non_alnum_mean,
            "path_non_alnum_max": path_non_alnum_max,
            "query_non_alnum_mean": query_non_alnum_mean,
            "query_non_alnum_max": query_non_alnum_max,
            "encoded_ratio": encoded_ratio_mean,
            "query_encoded_ratio": query_encoded_mean,
            "q_len_mean": q_len_mean,
            "q_len_max": q_len_max,
            "token_entropy_mean": token_entropy_mean,
            "token_entropy_max": token_entropy_max,
            "max_param_entropy": max_param_entropy_w,
            "transition_rate_mean": transition_rate_mean,
            "transition_rate_max": transition_rate_max,
            "param_count": param_count_mean,
            "status_500_ratio": status_500_ratio,
            "status_4xx_ratio": status_4xx_ratio,
            "query_diversity": query_diversity,
            "path_rarity_mean": path_rarity_mean,
            "path_rarity_max": path_rarity_max,
            "qlen_z_max": qlen_z_max,
            "entropy_z_max": entropy_z_max,
            "break_count": break_count,
            "break_ratio": break_ratio,
            "rare_path_count": rare_path_count,
            "rare_path_ratio": rare_path_ratio,
            "legit_path_ratio": legit_path_ratio,
            "burst_rate": burst_rate,
            "_reqs": reqs,
        })

    total_elapsed = time.time() - t0
    print(f"{tag}[+] Built {len(rows):,} entity-window feature vectors in {total_elapsed:,.1f}s "
          f"from {n_records:,} records", flush=True)
    return rows


def to_matrix(rows: list) -> np.ndarray:
    return np.array([[r[c] for c in FEATURE_COLS] for r in rows], dtype=float)