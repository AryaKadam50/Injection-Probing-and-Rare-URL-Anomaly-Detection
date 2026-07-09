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
anomaly_detection import ...`, `from scheduler import
RollingRetrainingScheduler`, etc.), so a flat layout like this is required
for Python's import resolution to work:

```
<your-install-dir>/
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

`<your-install-dir>` can be anywhere you have read/write access —
`/opt/uc09`, a directory under your own home folder, wherever. It does not
need to match any path shown in this README; the examples later on use
`/opt/uc09` purely as a placeholder to substitute your own.

If your repo's files landed loose in one directory when cloned, you're
already set — just make sure nothing else shares a name that would shadow
`model_library` (keep the model storage root, e.g. `models/`, named
differently from any importable package).

| File | Role |
|---|---|
| `feature_extraction.py` | Log parsing (`load_logs`, `parse_event`), path/query normalization, per-path baselines (`build_path_query_baseline`), entity-window feature aggregation (`build_entity_windows`, `FEATURE_COLS`), and all detection constants (window size, gate thresholds, etc.). |
| `anomaly_detection.py` | Scores entity-windows against a model bundle (`score_rows` — IF + LOF fusion with the legit-path discount), applies the burst/catastrophic alert gate (`apply_alert_gate`), assigns severity (`assign_severities`), and provides explainability helpers (`top_signals`, `worst_urls`). |
| `train_model.py` | Fits a fresh `StandardScaler` + `IsolationForest` + `LocalOutlierFactor` on one 14-day window of data (`train_models`) and packages everything into a joblib "bundle" (`build_bundle`, `train_candidate_from_records`). |
| `model_library.py` | Versioned on-disk storage for model bundles (`candidate/`, `approved/`, `metadata/`). Only one model is ever "approved & active" for scoring at a time (`get_active_model`, `approve_model`, `rollback_model`). |
| `drift_detection.py` | Compares a candidate model against the currently approved one (feature drift via PSI/KL, score drift, threshold drift) and decides whether the change needs human review (`compare_models`, `DRIFT_APPROVAL_THRESHOLD`). |
| `scheduler.py` | Owns the rolling train → drift-review → approve/reject lifecycle (`RollingRetrainingScheduler`). Runs candidate training on a background thread so it never blocks live detection. |
| `alert_manager.py` | Alert presentation: human-readable console alerts (`print_alert`) and JSON output for SOC/SIEM ingestion (`save_alert_json` — one JSON object **per anomalous URL**, not per window). |
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

`models/` (the default `--model-root`) and any `alerts/` directory you
point at are created automatically on first run. Once this works locally,
jump to [Production deployment](#6-production-deployment-systemd) to run
it continuously.

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

In parallel, every batch of records handed to `stream`, `tail`, or `serve`
is also fed into `RollingRetrainingScheduler.ingest_records()`, which
accumulates a rolling 14-day buffer (`RollingLogBuffer`, 2 weeks retention)
and, once a window completes, trains the next candidate model **on a
background thread** and runs it through drift review — all without
pausing detection.

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
hitting well-known/legit paths — the `legit_path_ratio` discount in
`score_rows()`). A `GaussianMixture` model separately produces an
`anomaly_prob` — an *informational* confidence score that is never itself
the alert trigger.

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
  `catastrophic_threshold`) *and* a landed 5xx response in the window
  (`status_500_ratio > 0`).

Alerts that clear either gate get a `severity` (`LOW`/`MEDIUM`/`HIGH`,
relative to that batch's own score distribution, via `assign_severities`)
and a `relative_score` normalized against the *active model's own*
thresholds (`0.0` = just cleared the burst gate, `1.0` = at the
catastrophic threshold). Because every model's thresholds are derived the
same way from its own training data (99.9th / 99.99th percentile of its
own training scores), `relative_score` stays comparable across model
versions even as raw `fused_score` drifts between retrains — this is why
`--min-relative-score` is the recommended way to set a stable alerting
bar, not a hand-picked `fused_score` cutoff.

Alert output is **one JSON object per anomalous URL**, not per window —
`alerts_per_url()` explodes each alerting window's `worst_urls()` list
into individual alert records that share the same IP/window/score
context, so downstream SOC tooling gets one ticket-worthy item per URL.

---

## 5. CLI commands

All commands are run through `realtime_pipeline.py` and accept
`--model-root` (default `./models`) to control where model bundles are
stored.

### `bootstrap` — train the very first model (Window 1)

```bash
python realtime_pipeline.py bootstrap --input access_logs_day1_14.json
```

Trains directly from the input file and auto-approves it, since there is no
existing approved model to compare drift against yet.

### `score` — score one batch against the currently approved model

```bash
python realtime_pipeline.py score --input access_logs_day15_28.json \
    --alert-json-out alerts/batch1.json
```

### `stream` — simulate continuous ingestion + rolling retraining

```bash
# One file per arrival batch (e.g. day-by-day dumps)
python realtime_pipeline.py stream --input-dir ./daily_logs --poll-seconds 5

# A single file spanning many days — auto-sliced into 14-day windows
python realtime_pipeline.py stream --input-file dataset/access_logs_last60days.json \
    --auto-approve --alert-json-dir alerts/
```

`--auto-approve` skips drift review entirely and promotes every candidate
— useful for backtests/demos, not recommended for production.

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
| `--approval-mode {safe,auto,console,json}` | How retrain candidates get promoted. `safe` (default): auto-promote unless drift is extreme (auto-reject instead, never blocks). `auto`: always promote, no safety net. `console`: blocks on a Y/N prompt (interactive use only — hangs under systemd). `json`: writes a `pending_approval/approval.json` file and waits for an analyst to edit it (headless-friendly, human-in-the-loop). |
| `--history {auto,resume,replay}` | `auto` (default): no active model yet → full progressive replay of file history; active model already exists → resume from saved byte offset. Safe to use for both first launch and every restart. `resume`: never reprocess history. `replay`: force a fresh progressive rebuild even if a model already exists. |
| `--min-relative-score` | Drop alerts below this `relative_score` (normalized 0–1 against each model's own thresholds — stays meaningful across model versions, unlike raw `fused_score`). Default: no filtering. |
| `--high-only` | Only keep `HIGH` severity alerts. Severity is relative to each scoring batch, not a fixed cutoff — combine with `--min-relative-score` for a stable bar. |
| `--require-landed-error` | Only keep alerts where `status_500_ratio > 0` — i.e. a request in the window actually returned a 5xx, not just a statistically loud burst. Filters out mass recon/scanning noise (probing `/.env`, `/wp-content/*`, `/cgi-bin/php`, etc.) that can clear a high `relative_score` purely on volume while being all-404s. |
| `--max-urls-per-alert` | Caps how many distinct-URL alert lines one anomalous window can emit (default: uncapped). Does not change which windows count as anomalous — only limits fan-out for one loud campaign hitting many paths in the same window. |

### `serve` — watch a directory for new/rotated log files

```bash
python realtime_pipeline.py serve --watch-dir ./incoming_logs \
    --alert-json-dir alerts/ --approval-mode safe
```

---

## 6. Production deployment: `systemd`

The pipeline is designed to run unattended via `tail` mode under `systemd`.
The paths below use a placeholder install directory,
`/opt/uc09` — substitute this with wherever *you* actually have
read/write access (it does **not** need to be under any particular user's
home directory; anywhere the service's `User` can read and write works).

### 6.1 Lay out the directories

Create one top-level directory and the sub-directories the pipeline reads
from / writes to:

```bash
sudo mkdir -p /opt/uc09/{scripts,data,models,alerts}
```

| Directory | Purpose |
|---|---|
| `scripts/` | All eight `.py` files from this repo, flat, no sub-folders (see [Repo layout](#1-repo-layout)). |
| `data/` | The NDJSON log file `tail` watches (e.g. `data/live_logs.ndjson`). Your log shipper should append here. |
| `models/` | Created and managed automatically by `model_library.py` (`approved/`, `candidate/`, `metadata/`, and `pending_approval/` if using `--approval-mode json`). You don't need to pre-create its contents — just the parent. |
| `alerts/` | Where `--alert-out-file` writes `alerts.jsonl`. |

Copy this repo's `.py` files into `scripts/`:

```bash
cp *.py /opt/uc09/scripts/
```

Then pick (or create) a user to run the service as — it needs read access
to `scripts/` and `data/`, and write access to `models/` and `alerts/`:

```bash
sudo chown -R <your-service-user>:<your-service-user> /opt/uc09
```

### 6.2 Create the service file

```bash
sudo nano /etc/systemd/system/uc09.service
```

```ini
[Unit]
Description=UC09 Real-Time Detection Pipeline
After=network.target

[Service]
Type=simple
User=<your-service-user>
WorkingDirectory=/opt/uc09

ExecStart=/usr/bin/python3 \
    /opt/uc09/scripts/realtime_pipeline.py \
    --model-root /opt/uc09/models \
    tail \
    --input-file /opt/uc09/data/live_logs.ndjson \
    --alert-out-file /opt/uc09/alerts/alerts.jsonl \
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

> Replace `<your-service-user>` and every `/opt/uc09` path with your own
> choices — nothing here is tied to a specific machine or account. The
> only hard requirement is that `scripts/` contains all eight `.py` files
> together (`realtime_pipeline.py` imports its siblings directly — see
> [Repo layout](#1-repo-layout)), and that `User` has read/write
> permission on whatever paths you point `--input-file`, `--model-root`,
> and `--alert-out-file` at.

### 6.3 Reload systemd

```bash
sudo systemctl daemon-reload
```

### 6.4 Enable auto-start on boot

```bash
sudo systemctl enable uc09
```

### 6.5 Start the service

```bash
sudo systemctl start uc09
```

### 6.6 Check status

```bash
sudo systemctl status uc09
```

Expect to see:

```
active (running)
```

### 6.7 View live logs

```bash
journalctl -u uc09 -f
```

Equivalent to watching the terminal output of a manually-started run.

### 6.8 Stop / restart

```bash
sudo systemctl stop uc09
sudo systemctl restart uc09
```

---

## 7. Analyst approval workflow (`--approval-mode json`)

When a retrained candidate's drift score **exceeds**
`DRIFT_APPROVAL_THRESHOLD` (see `drift_detection.py`), the scheduler cannot
safely auto-promote it and instead writes a pending-approval record under
your `--model-root` (using the layout from section 6):

```
/opt/uc09/models/pending_approval/approval.json
```

**Check the pending request:**

```bash
cat /opt/uc09/models/pending_approval/approval.json
```

It contains the candidate's `drift_score`, the `approval_threshold` it
exceeded, and a `status` field starting at `"PENDING"`.

**Approve:**

```bash
sed -i 's/PENDING/APPROVED/' \
  /opt/uc09/models/pending_approval/approval.json
```

**Reject:**

```bash
sed -i 's/PENDING/REJECTED/' \
  /opt/uc09/models/pending_approval/approval.json
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
models/
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
  production — it just repoints `active.json`, so it's instant and
  doesn't touch the `approved/` bundles themselves.
- `reject_candidate()` discards a candidate a human reviewer declined to
  promote, leaving the currently active model untouched.

---

## 9. Key tunables

### `feature_extraction.py`

| Constant | Current value | Effect |
|---|---|---|
| `WINDOW_MINUTES` | 60 | Size of each per-IP aggregation window. |
| `MIN_REQUESTS` | 20 | Minimum requests a window needs to be scored at all — see below. |
| `MIN_BREAK_URLS` | 25 | Minimum absolute count of grammar-breaking URLs for the burst gate — see below. |
| `MIN_BREAK_RATIO` | 0.05 | Minimum *fraction* of a window's requests that must be grammar-breaking — see below. |
| `ALERT_PCT` | 99.9 | Training-score percentile used to derive `fused_threshold`. |
| `CATASTROPHIC_PCT` | 99.99 | Training-score percentile used to derive `catastrophic_threshold`. |
| `SCORE_BOUND_PCT_LOW` / `SCORE_BOUND_PCT_HIGH` | 0.5 / 99.5 | Percentile bounds used to normalize raw IF/LOF scores into `[0, 1]`. |

### `drift_detection.py` / `scheduler.py`

| Constant | Current value | Effect |
|---|---|---|
| `DRIFT_APPROVAL_THRESHOLD` | 0.59 | Drift score above which a retrained candidate requires human/analyst approval instead of auto-promoting. |
| `SAFE_AUTO_REJECT_ABOVE` | 0.85 | In `safe` approval mode, drift above this is auto-rejected rather than promoted. |
| `WEIGHT_FEATURE` / `WEIGHT_SCORE` / `WEIGHT_THRESHOLD` | 0.5 / 0.3 / 0.2 | How feature drift (PSI/KL on the training matrix), score drift (PSI on IF/LOF/fused training scores), and threshold drift (relative change in `fused_threshold`/`catastrophic_threshold`) combine into one `drift_score`. |
| `WEEK` (retraining cadence) | 14 days | How much new data accumulates before the scheduler kicks off background training of the next candidate. |

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

Changing any of these and wiping `models/` warrants an explicit
`--history replay` on the next `tail` run to rebuild models from scratch.

---

## 10. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'model_library'` (or similar) | Not all `.py` files are in the same directory as `realtime_pipeline.py` — see [Repo layout](#1-repo-layout). |
| `[realtime_pipeline] No approved model yet — cannot score.` | No model has been bootstrapped/approved yet. Run `bootstrap` first, or use `tail --history auto`, which bootstraps automatically on a fresh `models/` directory. |
| Pipeline hangs with no output under `systemd` | You're using `--approval-mode console`, which blocks on `input()`. There's no terminal attached under `systemd` to answer it. Use `safe`, `auto`, or `json` for headless deployment instead. |
| `approval.json` never resolves | The scheduler polls it every ~5 seconds but only while a candidate is genuinely pending — check `cat .../pending_approval/approval.json` for `"status": "PENDING"`, then flip it with the `sed` commands in [section 7](#7-analyst-approval-workflow---approval-mode-json). |
| No new alerts appear even though the file is growing | With `tail --history resume`, only bytes appended *after* the saved offset are read. Check the sidecar `<input_file>.tail_offset` file, and confirm new lines are complete, newline-terminated NDJSON (a trailing partial line is intentionally held back until it's finished). |
| Way more/fewer alerts than expected after a config change | Detection constants (`MIN_BREAK_URLS`, `ALERT_PCT`, etc.) are baked into each trained model's thresholds. Changing them requires retraining — wipe `models/` and rerun with `--history replay` (or `bootstrap`) so every model reflects the new config. |
| `journalctl -u uc09 -f` shows nothing | Confirm the unit is actually active with `systemctl status uc09`; if it's not, check `ExecStart`'s paths (`realtime_pipeline.py` path, `--input-file`, `--model-root`) are correct and the `User` running the service has read/write permission on them. |
| `ValueError: Only N entity-windows ... too few to train` | The 14-day window handed to `train_candidate_from_records()` produced fewer than 10 entity-windows (usually too little traffic, or too many IPs under `MIN_REQUESTS`). Training for that window is skipped and the scheduler keeps serving on the current model — widen the window or check log volume. |
