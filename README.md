# UC-09 — Real-Time Anomalous URL/Query Detection Pipeline

UC-09 is a real-time, self-retraining anomaly detection pipeline for web
traffic logs. It flags requests with statistically anomalous
URLs/query strings (injection-style probing, mass recon/enumeration,
grammar-breaking payloads, etc.) using an IsolationForest + Local Outlier
Factor ensemble, fuses those scores with a GMM-based confidence measure,
and retrains itself on a rolling 14-day cadence — without ever blocking
live detection.

This repo's files are uploaded individually with no sub-folders. This
README explains how the pieces fit together, how to run the pipeline
locally, and how to deploy it as a persistent `systemd` service.

## Contents

1. [Repo layout](#1-repo-layout)
2. [Requirements](#2-requirements)
3. [Quickstart](#3-quickstart)
4. [How it works](#4-how-it-works)
5. [CLI commands](#5-cli-commands)
6. [Production deployment: systemd](#6-production-deployment-systemd)
7. [Analyst approval workflow](#7-analyst-approval-workflow---approval-mode-json)
8. [Model lifecycle](#8-model-lifecycle-at-a-glance)
9. [Key tunables](#9-key-tunables)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Repo layout

All eight modules must live in the **same directory** — `realtime_pipeline.py`
imports its siblings directly (`import model_library`, `from
anomaly_detection import ...`, etc.), so a flat layout like this is
required for Python's import resolution to work:

```
uc09_workspace/
└── scripts/
    ├── feature_extraction.py
    ├── anomaly_detection.py
    ├── train_model.py
    ├── model_library.py
    ├── drift_detection.py
    ├── scheduler.py
    ├── alert_manager.py
    └── realtime_pipeline.py   ← entrypoint, run this one
```

If your repo's files landed loose in one directory when cloned, you're
already set — just make sure nothing else shares a name that would shadow
`model_library` (keep the model storage root, e.g. `model_store/`, named
differently from any importable package).

| File | Role |
|---|---|
| `feature_extraction.py` | Log parsing, path/query normalization, per-path baselines, entity-window feature aggregation (`FEATURE_COLS`), and all detection constants (window size, gate thresholds, etc.). |
| `anomaly_detection.py` | Scores entity-windows against a model bundle (IF + LOF fusion, legit-path discount), applies the burst/catastrophic alert gate, assigns severity, and provides explainability helpers (`top_signals`, `worst_urls`). |
| `train_model.py` | Fits a fresh `StandardScaler` + `IsolationForest` + `LocalOutlierFactor` on one 14-day window of data and packages everything into a joblib "bundle". |
| `model_library.py` | Versioned on-disk storage for model bundles (`candidate/`, `approved/`, `metadata/`). Only one model is ever "approved & active" for scoring at a time. |
| `drift_detection.py` | Compares a candidate model against the currently approved one (feature drift via PSI/KL, score drift, threshold drift) and decides whether the change needs human review. |
| `scheduler.py` | Owns the rolling train → drift-review → approve/reject lifecycle. Runs candidate training on a background thread so it never blocks live detection. |
| `alert_manager.py` | Alert presentation: human-readable console alerts and JSON output for SOC/SIEM ingestion (one JSON object **per anomalous URL**, not per window). |
| `realtime_pipeline.py` | The production entrypoint / CLI. Wires the modules above into `bootstrap`, `score`, `stream`, `tail`, and `serve` commands. |

---

## 2. Requirements

```bash
pip install numpy scikit-learn joblib
```

Python 3.9+ recommended.

---

## 3. Quickstart

A minimal local run, from inside the directory containing all eight
`.py` files:

```bash
# 1. Train + auto-approve the first model from your oldest 14 days of logs
python realtime_pipeline.py bootstrap --input logs_day1_14.json

# 2. Score the next batch against it
python realtime_pipeline.py score --input logs_day15_28.json \
    --alert-json-out alerts/batch1.json

# 3. Inspect what fired
cat alerts/batch1.json
```

`model_store/` and `alerts/` are created automatically on first run. Once
this works locally, jump to [Production deployment](#6-production-deployment-systemd)
to run it continuously.

---

## 4. How it works

### 4.1 Detection flow

```
Incoming Logs
      │
      ▼
Feature Extraction   (feature_extraction.py)
      │
      ▼
Load Active Model    (model_library.get_active_model — always the
      │                currently APPROVED model)
      ▼
Score                (anomaly_detection.py — IF + LOF + GMM fusion)
      │
      ▼
Alert Manager        (alert_manager.py — console and/or SOC JSON)
```

In parallel, every batch of records is also fed into
`RollingRetrainingScheduler`, which accumulates a rolling 14-day buffer and,
once a window completes, trains the next candidate model **on a background
thread** and runs it through drift review — all without pausing detection.

### 4.2 From raw requests to an entity-window

Logs aren't scored one request at a time. `build_entity_windows()` first
buckets requests into `(ip, 60-minute window)` groups, then computes one
aggregated feature vector per bucket (entropy stats, non-alphanumeric
ratios, break counts, status-code ratios, etc. — see `FEATURE_COLS`). Only
buckets with at least `MIN_REQUESTS` (20) requests are scored at all —
below that, the statistics are too noisy to trust.

### 4.3 Scoring and the alert gate

Each entity-window's feature vector is scored by both an `IsolationForest`
and a `LocalOutlierFactor`, normalized to `[0, 1]`, and combined into a
single `fused_score` (with a discount applied for traffic that's mostly
hitting well-known/legit paths). A `GaussianMixture` model separately
produces an `anomaly_prob` — an *informational* confidence score that is
never itself the alert trigger.

An entity-window only becomes an **alert** if it clears one of two gates in
`apply_alert_gate()`:

- **Burst gate** — `fused_score >= fused_threshold` **and** enough
  grammar-breaking URLs, judged two ways at once:
  - `break_count >= MIN_BREAK_URLS` (25) — an absolute floor, so a couple
    of odd-looking URLs in an otherwise huge, normal window can't trigger
    on their own.
  - `break_ratio >= MIN_BREAK_RATIO` (0.05) — a proportional floor on top
    of that, so the breaking traffic also has to be a *meaningful share*
    of the window. This is what tells apart a small, surgical injection
    campaign (e.g. 363 of 1,759 requests breaking — 21%) from high-volume
    recon/enumeration noise that clears the same raw count purely on
    volume (e.g. 63 of 38,377 — 0.16%).
- **Catastrophic gate** — a very high `fused_score` (past
  `catastrophic_threshold`) *and* a landed 5xx response in the window.

Alerts that clear either gate get a `severity` (`LOW`/`MEDIUM`/`HIGH`,
relative to that batch's own score distribution) and a `relative_score`
normalized against the *active model's own* thresholds (`0.0` = just
cleared the burst gate, `1.0` = at the catastrophic threshold). Because
every model's thresholds are derived the same way from its own training
data, `relative_score` stays comparable across model versions even as raw
`fused_score` drifts between retrains — this is why `--min-relative-score`
is the recommended way to set a stable alerting bar, not a hand-picked
`fused_score` cutoff.

---

## 5. CLI commands

All commands are run through `realtime_pipeline.py` and accept
`--model-root` (default `./model_store`) to control where model bundles are
stored.

### `bootstrap` — train the very first model (Window 1)

```bash
python realtime_pipeline.py bootstrap --input moodle_day1_14.json
```

Trains directly from the input file and auto-approves it, since there is no
existing approved model to compare drift against yet.

### `score` — score one batch against the currently approved model

```bash
python realtime_pipeline.py score --input moodle_day15_28.json \
    --alert-json-out alerts/batch1.json
```

### `stream` — simulate continuous ingestion + rolling retraining

```bash
# One file per arrival batch (e.g. day-by-day dumps)
python realtime_pipeline.py stream --input-dir ./daily_logs --poll-seconds 5

# A single file spanning many days — auto-sliced into 14-day windows
python realtime_pipeline.py stream --input-file dataset/moodle_last60days.json \
    --auto-approve --alert-json-dir alerts/
```

### `tail` — watch ONE continuously-appended NDJSON file (recommended for production)

```bash
python realtime_pipeline.py tail \
    --input-file /path/to/live_logs.ndjson \
    --alert-out-file alerts/alerts.jsonl \
    --approval-mode safe \
    --poll-seconds 30
```

Runs forever, re-reading only newly appended bytes on each poll. This is
the mode used in the systemd setup below. Key flags:

| Flag | Purpose |
|---|---|
| `--approval-mode {safe,auto,console,json}` | How retrain candidates get promoted. `safe`: auto-promote unless drift is extreme (auto-reject instead, never blocks). `auto`: always promote. `console`: blocks on a Y/N prompt (interactive use only). `json`: writes a `pending_approval/approval.json` file and waits for an analyst to edit it (headless-friendly, human-in-the-loop). |
| `--history {auto,resume,replay}` | `auto` (default): no active model yet → full progressive replay of file history; active model already exists → resume from saved byte offset. Safe to use for both first launch and every restart. |
| `--min-relative-score` | Drop alerts below this `relative_score` (normalized 0–1 against each model's own thresholds — stays meaningful across model versions, unlike raw `fused_score`). |
| `--high-only` | Only keep `HIGH` severity alerts. |
| `--require-landed-error` | Only keep alerts where a request actually returned a 5xx (filters out pure recon/scanning noise). |
| `--max-urls-per-alert` | Caps how many distinct-URL alert lines one anomalous window can emit. |

### `serve` — watch a directory for new/rotated log files

```bash
python realtime_pipeline.py serve --watch-dir ./incoming_logs \
    --alert-json-dir alerts/ --approval-mode safe
```

---

## 6. Production deployment: `systemd`

The pipeline is designed to run unattended via `tail` mode under `systemd`.

### 6.1 Create the service file

```bash
sudo nano /etc/systemd/system/uc09.service
```

```ini
[Unit]
Description=UC09 Real-Time Detection Pipeline
After=network.target

[Service]
Type=simple
User=fusion
WorkingDirectory=/home/fusion/framework

ExecStart=/usr/bin/python3 \
    /home/fusion/framework/uc09_workspace/scripts/realtime_pipeline.py \
    --model-root /home/fusion/framework/uc09_workspace/model_store \
    tail \
    --input-file /home/fusion/framework/data/moodle_last60days.json \
    --alert-out-file /home/fusion/framework/uc09_workspace/alerts/alerts.jsonl \
    --approval-mode json \
    --poll-seconds 30 \
    --min-relative-score 1.0 \
    --high-only \
    --max-urls-per-alert 5

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

> Adjust `User`, `WorkingDirectory`, and all paths to match your actual
> layout. Since this repo's files are uploaded individually with no
> subfolders, place them together under a single directory (e.g.
> `uc09_workspace/scripts/`) before pointing `ExecStart` at
> `realtime_pipeline.py` — it imports its sibling modules
> (`model_library`, `feature_extraction`, `anomaly_detection`,
> `alert_manager`, `scheduler`, `train_model`, `drift_detection`) directly,
> so they must all live in the same directory.

### 6.2 Reload systemd

```bash
sudo systemctl daemon-reload
```

### 6.3 Enable auto-start on boot

```bash
sudo systemctl enable uc09
```

### 6.4 Start the service

```bash
sudo systemctl start uc09
```

### 6.5 Check status

```bash
sudo systemctl status uc09
```

Expect to see:

```
active (running)
```

### 6.6 View live logs

```bash
journalctl -u uc09 -f
```

Equivalent to watching the terminal output of a manually-started run.

### 6.7 Stop / restart

```bash
sudo systemctl stop uc09
sudo systemctl restart uc09
```

---

## 7. Analyst approval workflow (`--approval-mode json`)

When a retrained candidate's drift score **exceeds**
`DRIFT_APPROVAL_THRESHOLD` (see `drift_detection.py`), the scheduler cannot
safely auto-promote it and instead writes a pending-approval record:

```
/home/fusion/framework/uc09_workspace/model_store/pending_approval/approval.json
```

**Check the pending request:**

```bash
cat /home/fusion/framework/uc09_workspace/model_store/pending_approval/approval.json
```

It contains the candidate's `drift_score`, the `approval_threshold` it
exceeded, and a `status` field starting at `"PENDING"`.

**Approve:**

```bash
sed -i 's/PENDING/APPROVED/' \
  /home/fusion/framework/uc09_workspace/model_store/pending_approval/approval.json
```

**Reject:**

```bash
sed -i 's/PENDING/REJECTED/' \
  /home/fusion/framework/uc09_workspace/model_store/pending_approval/approval.json
```

The scheduler polls this file roughly every 5 seconds and, on seeing
`APPROVED`, promotes the candidate to active; on `REJECTED`, discards it
and keeps serving on the current model. No service restart is required
either way.

### End-to-end service flow

1. System boots.
2. `uc09.service` starts automatically.
3. Logs are tailed continuously from the byte offset last processed.
4. Alerts are written to `alerts.jsonl` (one JSON object per anomalous URL).
5. Retraining happens automatically every completed 14-day window, on a
   background thread — detection never pauses for it.
6. If drift is small → the new model auto-promotes.
7. If drift exceeds the threshold → `approval.json` is created and the
   pipeline keeps serving on the current model while it waits.
8. Analyst edits `approval.json`, changing `PENDING` to `APPROVED` or
   `REJECTED`.
9. The service continues running throughout — no restart needed.
10. `Restart=always` in the unit file means the service also survives
    crashes/reboots and resumes cleanly (`--history auto` picks up from the
    saved offset instead of reprocessing everything).

---

## 8. Model lifecycle at a glance

```
model_store/
  approved/         # .joblib bundles currently or previously active
  candidate/        # awaiting drift review / approval
  metadata/         # one JSON record per model_id + active.json
  pending_approval/ # created only when --approval-mode json needs a human
```

- Exactly one model is "approved & active" for scoring at any time
  (`model_library.get_active_model`).
- `approve_model()` is the only function that changes what real-time
  detection actually scores against.
- `rollback_model()` can revert to any previously approved version if a
  newly promoted model turns out to alert-flood or go silent in
  production.

---

## 9. Key tunables (`feature_extraction.py`)

| Constant | Current value | Effect |
|---|---|---|
| `WINDOW_MINUTES` | 60 | Size of each per-IP aggregation window. |
| `MIN_REQUESTS` | 20 | Minimum requests a window needs to be scored at all — see below. |
| `MIN_BREAK_URLS` | 25 | Minimum absolute count of grammar-breaking URLs for the burst gate — see below. |
| `MIN_BREAK_RATIO` | 0.05 | Minimum *fraction* of a window's requests that must be grammar-breaking — see below. |
| `ALERT_PCT` | 99.9 | Training-score percentile used to derive `fused_threshold`. |
| `CATASTROPHIC_PCT` | 99.99 | Training-score percentile used to derive `catastrophic_threshold`. |
| `DRIFT_APPROVAL_THRESHOLD` (`drift_detection.py`) | 0.59 | Drift score above which a retrained candidate requires human/analyst approval instead of auto-promoting. |
| `SAFE_AUTO_REJECT_ABOVE` (`scheduler.py`) | 0.85 | In `safe` approval mode, drift above this is auto-rejected rather than promoted. |

`MIN_REQUESTS`, `MIN_BREAK_URLS`, and `MIN_BREAK_RATIO` act at different
stages of the pipeline and are easy to conflate:

- **`MIN_REQUESTS`** is an *eligibility* filter, applied in
  `build_entity_windows()` before any scoring happens. If an IP made fewer
  than 20 requests in an hour, that bucket is discarded outright — most of
  the statistical features (entropy, non-alnum ratio, z-scores) aren't
  meaningful on a handful of requests, so this isn't a maliciousness
  judgment, it's a "is there enough data to trust the stats" filter.
- **`MIN_BREAK_URLS`** is an *absolute* floor on the burst gate, applied in
  `apply_alert_gate()` after scoring. A window needs at least 25 URLs
  flagged as grammar-breaking (`is_break` in `url_features()` — extreme
  non-alphanumeric ratio, extreme char-class transition rate, oversized
  query, etc.) before the burst gate can fire at all, regardless of how
  anomalous any individual URL looks.
- **`MIN_BREAK_RATIO`** is the *proportional* companion, required on top of
  `MIN_BREAK_URLS`, not instead of it. It exists because raw count alone
  can't distinguish a small, surgical injection campaign from high-volume
  recon/enumeration noise that happens to cross the same absolute count
  purely on volume — e.g. 363 breaking URLs out of 1,759 requests (21%,
  genuinely anomalous) vs. 63 breaking URLs out of 38,377 requests (0.16%,
  just noise in a big pile of scanning traffic). Both `break_count >= 25`
  **and** `break_ratio >= 0.05` must hold for the burst gate to trigger.

Changing any of these and wiping `model_store/` warrants an explicit
`--history replay` on the next `tail` run to rebuild models from scratch.

---

## 10. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'model_library'` (or similar) | Not all `.py` files are in the same directory as `realtime_pipeline.py` — see [Repo layout](#1-repo-layout). |
| `[realtime_pipeline] No approved model yet — cannot score.` | No model has been bootstrapped/approved yet. Run `bootstrap` first, or use `tail --history auto`, which bootstraps automatically on a fresh `model_store/`. |
| Pipeline hangs with no output under `systemd` | You're using `--approval-mode console`, which blocks on `input()`. There's no terminal attached under `systemd` to answer it. Use `safe`, `auto`, or `json` for headless deployment instead. |
| `approval.json` never resolves | The scheduler polls it every ~5 seconds but only while a candidate is genuinely pending — check `cat .../pending_approval/approval.json` for `"status": "PENDING"`, then flip it with the `sed` commands in [section 7](#7-analyst-approval-workflow---approval-mode-json). |
| No new alerts appear even though the file is growing | With `tail --history resume`, only bytes appended *after* the saved offset are read. Check the sidecar `<input_file>.tail_offset` file, and confirm new lines are complete, newline-terminated NDJSON (a trailing partial line is intentionally held back until it's finished). |
| Way more/fewer alerts than expected after a config change | Detection constants (`MIN_BREAK_URLS`, `ALERT_PCT`, etc.) are baked into each trained model's thresholds. Changing them requires retraining — wipe `model_store/` and rerun with `--history replay` (or `bootstrap`) so every model reflects the new config. |
| `journalctl -u uc09 -f` shows nothing | Confirm the unit is actually active with `systemctl status uc09`; if it's not, check `ExecStart`'s paths (`realtime_pipeline.py` path, `--input-file`, `--model-root`) are correct and the `User` running the service has read/write permission on them. |
