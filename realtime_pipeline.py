"""
realtime_pipeline.py
======================
Production entrypoint.

    Incoming Logs
          |
          v
    Feature Extraction   (feature_extraction.py — unchanged UC-09 features)
          |
          v
    Load Active Model    (model_library.get_active_model — always the
          |                currently APPROVED model; never blocks on
          |                whatever candidate is training in the background)
          v
    Score                (anomaly_detection.py — same IF+LOF+GMM fusion)
          |
          v
    Alert Manager         (alert_manager.py — console + SOC JSON)

This script also drives the rolling scheduler: every batch of incoming
records is (a) scored immediately against the active model and (b) handed
to the scheduler's rolling buffer, which — independently, on a background
thread, without blocking step (a) — trains the next 14-day candidate once
a full window has accumulated and runs it through drift review.

Usage
-----
Batch/backfill mode (score one file against whatever is currently approved):

    python realtime_pipeline.py score --input access_logs_day15_28.json

Continuous mode, one file per arrival batch (day-by-day or window-by-window dumps):

    python realtime_pipeline.py stream --input-dir ./daily_logs --poll-seconds 5

Continuous mode from a SINGLE file spanning many days (e.g. a 60-day export)
— records are auto-sliced into 14-day windows by timestamp, no pre-splitting
into separate files required:

    python realtime_pipeline.py stream --input-file dataset/access_logs_100days.json \
        --auto-approve --alert-json-dir alerts/

Bootstrap (Window 1, no approved model yet — trains + auto-approves V1):

    python realtime_pipeline.py bootstrap --input access_logs_day1_14.json

See README.md for the full walkthrough across multiple windows.
"""

import argparse
import glob
import json
import os
import time
from datetime import datetime

import numpy as np

import model_library
from feature_extraction import (
    build_path_popularity, build_path_query_baseline, build_entity_windows, load_logs, parse_event,
)
from anomaly_detection import score_rows, apply_alert_gate, assign_severities
from alert_manager import print_alert, save_alert_json, alerts_per_url
from scheduler import RollingRetrainingScheduler


def score_against_active_model(records: list, root: str = model_library.DEFAULT_ROOT):
    """
    Feature Extraction -> Load Active Model -> Score, exactly following the
    realtime_pipeline.py workflow diagram in the spec. Path popularity /
    baselines used here are the ACTIVE MODEL's TRAIN-derived tables (never
    recomputed from the incoming batch) — this is what makes scoring
    consistent with how the model was fit, same as the original batch tool
    scoring a TEST file against a TRAIN-fit model.
    """
    model_id, bundle = model_library.get_active_model(root)
    if bundle is None:
        print("[realtime_pipeline] No approved model yet — cannot score. "
              "Run `bootstrap` first.", flush=True)
        return model_id, []

    rows = build_entity_windows(
        records, bundle["path_freq"], bundle["path_total"], bundle["common_cutoff"],
        bundle["path_baseline"], label=model_id,
    )
    if not rows:
        return model_id, []

    for r in rows:
        r["_train_rarity_p95"] = bundle["train_rarity_p95"]

    rows = score_rows(rows, bundle["scaler"], bundle["iforest"], bundle["lof"],
                       bundle["if_bounds"], bundle["lof_bounds"])
    rows = apply_alert_gate(rows, bundle["fused_threshold"], bundle["catastrophic_threshold"])

    rows.sort(key=lambda x: x["fused_score"], reverse=True)
    alerts = [r for r in rows if r["anomaly_flag"]]
    assign_severities(alerts)
    alerts.sort(key=lambda r: (r["fused_score"], r["break_count"]), reverse=True)

    return model_id, alerts


def run_score(input_path: str, root: str, alert_json_out: str = None) -> None:
    records = load_logs(input_path, label="LIVE")
    if not records:
        print("[realtime_pipeline] No records loaded.", flush=True)
        return

    model_id, alerts = score_against_active_model(records, root)
    print(f"[realtime_pipeline] Scored {len(records):,} records against '{model_id}'. "
          f"{len(alerts)} alert(s).", flush=True)
    for row in alerts:
        print_alert(row, model_id=model_id)
    if alert_json_out and alerts:
        save_alert_json(alerts, alert_json_out, model_id=model_id)


def run_bootstrap(input_path: str, root: str) -> None:
    """
    Window 1: train the very first model directly (no approved model exists
    to compare drift against, so this always auto-promotes), matching the
    spec's Window-1 behavior.
    """
    import train_model
    records = load_logs(input_path, label="WINDOW1")
    if not records:
        print("[realtime_pipeline] No records loaded — cannot bootstrap.", flush=True)
        return

    model_library.init_library(root)
    model_id = model_library.next_model_id(root)
    week_start = min(r["ts"] for r in records)
    week_end = max(r["ts"] for r in records)
    bundle = train_model.train_candidate_from_records(model_id, records, week_start, week_end)
    model_library.register_candidate_model(bundle, root=root)
    model_library.approve_model(model_id, root=root)
    print(f"[realtime_pipeline] Bootstrap complete — '{model_id}' is approved and active.", flush=True)


def run_stream(input_dir: str, root: str, poll_seconds: int, alert_json_dir: str = None,
                auto_approve: bool = False) -> None:
    """
    Demo of the continuous workflow: treats each file in input_dir (sorted
    by name, e.g. day-by-day dumps) as the next arriving batch of logs.
    Each batch is scored immediately against whichever model is currently
    approved, then fed into the rolling scheduler, which trains/reviews/
    promotes the next window's candidate in the background without pausing
    detection on later batches.
    """
    scheduler = RollingRetrainingScheduler(model_root=root, auto_approve=auto_approve)

    files = sorted(glob.glob(os.path.join(input_dir, "*")))
    if not files:
        print(f"[realtime_pipeline] No files found in {input_dir}.", flush=True)
        return

    for fpath in files:
        records = load_logs(fpath, label="STREAM")
        if not records:
            continue

        model_id, alerts = score_against_active_model(records, root)
        print(f"[realtime_pipeline] [{os.path.basename(fpath)}] Scored {len(records):,} records "
              f"against '{model_id}'. {len(alerts)} alert(s).", flush=True)
        for row in alerts:
            print_alert(row, model_id=model_id)
        if alert_json_dir and alerts:
            out_path = os.path.join(alert_json_dir, f"alerts_{os.path.basename(fpath)}.json")
            save_alert_json(alerts, out_path, model_id=model_id)

        scheduler.ingest_records(records)
        scheduler.maybe_start_weekly_training()

        time.sleep(poll_seconds)

    # Give any in-flight background training a moment to finish before exit,
    # purely so the demo prints its result instead of dying mid-thread.
    if scheduler._training_thread is not None:
        scheduler._training_thread.join(timeout=60)


def run_stream_from_file(input_file: str, root: str, alert_json_dir: str = None,
                          auto_approve: bool = False) -> None:
    """
    Same continuous workflow as run_stream(), but for a SINGLE file spanning
    many days (e.g. access_logs_last60days.json) instead of one file per
    arrival batch. Records are sliced into consecutive 14-day windows purely from
    their own timestamps — no pre-splitting into separate files needed.

    For each 14-day slice, in order:
      1. Score it against whichever model is currently approved (skipped
         for the very first slice, since nothing is approved yet — that
         first slice becomes the Window 1 bootstrap training set instead).
      2. Feed it into the rolling scheduler and let a full window's boundary
         trigger background candidate training + drift review, exactly as
         it would if these days had arrived one batch at a time.

    This is a REPLAY/backtest driver: each window's training is joined
    (waited on) before moving to the next slice, so a 60-day file produces
    a clean, deterministic run through all the windowed models. A true live
    deployment (scheduler running against a real-time feed) would NOT join
    between windows — see run_stream()/scheduler.py for that shape instead.
    """
    from datetime import timedelta

    records = load_logs(input_file, label="STREAM")
    if not records:
        print(f"[realtime_pipeline] No records loaded from {input_file}.", flush=True)
        return

    scheduler = RollingRetrainingScheduler(model_root=root, auto_approve=auto_approve)

    week_start = min(r["ts"] for r in records)
    end_ts = max(r["ts"] for r in records)
    week_num = 1

    # 14-day slicing window (was 7).
    SLICE = timedelta(days=14)

    while week_start <= end_ts:
        week_end = week_start + SLICE
        week_slice = [r for r in records if week_start <= r["ts"] < week_end]
        if not week_slice:
            week_start = week_end
            continue

        label = f"window{week_num} [{week_start.date()} -> {week_end.date()})"

        active_id, active_bundle = model_library.get_active_model(root)
        if active_bundle is not None:
            model_id, alerts = score_against_active_model(week_slice, root)
            print(f"[realtime_pipeline] [{label}] Scored {len(week_slice):,} records "
                  f"against '{model_id}'. {len(alerts)} alert(s).", flush=True)
            for row in alerts:
                print_alert(row, model_id=model_id)
            if alert_json_dir and alerts:
                out_path = os.path.join(alert_json_dir, f"alerts_window{week_num}.json")
                save_alert_json(alerts, out_path, model_id=model_id)
        else:
            print(f"[realtime_pipeline] [{label}] No approved model yet — this slice "
                  f"will bootstrap the first model instead of being scored.", flush=True)

        scheduler.ingest_records(week_slice)
        if scheduler.maybe_start_weekly_training() and scheduler._training_thread is not None:
            scheduler._training_thread.join()

        week_start = week_end
        week_num += 1

    print(f"[realtime_pipeline] Replay complete. Active model: "
          f"{model_library.get_active_model_id(root)}", flush=True)


def _offset_path(input_file: str) -> str:
    return input_file + ".tail_offset"


def _read_offset(input_file: str) -> int:
    p = _offset_path(input_file)
    if os.path.exists(p):
        with open(p) as f:
            return int(f.read().strip() or 0)
    return 0


def _write_offset(input_file: str, offset: int) -> None:
    with open(_offset_path(input_file), "w") as f:
        f.write(str(offset))


def _read_new_records(input_file: str, last_offset: int):
    """Reads only complete new lines appended to input_file since
    last_offset (NDJSON — one JSON object per line). Returns
    (records, new_offset); leaves a trailing partial line unread so it's
    picked up whole on the next poll."""
    with open(input_file, "r") as f:
        f.seek(last_offset)
        chunk = f.read()

    if not chunk:
        return [], last_offset

    if chunk.endswith("\n"):
        consumed = chunk
        new_offset = last_offset + len(chunk.encode("utf-8"))
    else:
        last_nl = chunk.rfind("\n")
        if last_nl == -1:
            return [], last_offset  # no complete line yet
        consumed = chunk[:last_nl + 1]
        new_offset = last_offset + len(consumed.encode("utf-8"))

    records, bad = [], 0
    for line in consumed.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = parse_event(json.loads(line))
        except json.JSONDecodeError:
            r = None
        if r and r["ip"]:
            records.append(r)
        else:
            bad += 1

    if bad:
        print(f"[realtime_pipeline] [tail] {bad} malformed line(s) skipped this poll.", flush=True)

    records.sort(key=lambda x: x["ts"])
    return records, new_offset


def _replay_history_into_scheduler(input_file: str, root: str, scheduler: "RollingRetrainingScheduler",
                                    alert_out_file: str, min_relative_score: float = None,
                                    high_only: bool = False, require_landed_error: bool = False,
                                    max_urls_per_alert: int = None) -> int:
    """
    Progressively replays every record currently in input_file through
    consecutive 14-day windows (V1, V2, V3, ... — same windowing as
    run_stream_from_file), using the SAME scheduler/buffer instance that
    the live tail loop will keep using afterward, so there's no gap and no
    restart between "catching up on history" and "watching for new data".
    Each window's training is joined before moving to the next, so this is
    deterministic. Returns the byte offset corresponding to the end of the
    file at the time this finished (so the tail loop knows where to resume
    reading from for genuinely NEW appends).
    """
    from datetime import timedelta

    print(f"[realtime_pipeline] [tail] --history replay: reprocessing all existing data in "
          f"'{input_file}' in 14-day windows before switching to live tailing.", flush=True)

    records = load_logs(input_file, label="REPLAY")
    end_offset = os.path.getsize(input_file)
    if not records:
        print(f"[realtime_pipeline] [tail] No records found to replay yet.", flush=True)
        return end_offset

    week_start = min(r["ts"] for r in records)
    end_ts = max(r["ts"] for r in records)
    week_num = 1
    SLICE = timedelta(days=14)

    while week_start <= end_ts:
        week_end = week_start + SLICE
        week_slice = [r for r in records if week_start <= r["ts"] < week_end]
        if not week_slice:
            week_start = week_end
            continue

        label = f"window{week_num} [{week_start.date()} -> {week_end.date()})"

        active_id, active_bundle = model_library.get_active_model(root)
        if active_bundle is not None:
            model_id, alerts = score_against_active_model(week_slice, root)
            pre_filter_count = len(alerts)
            if high_only:
                alerts = [r for r in alerts if r.get("severity") == "HIGH"]
            if require_landed_error:
                alerts = [r for r in alerts if r.get("status_500_ratio", 0.0) > 0]
            if min_relative_score is not None:
                alerts = [r for r in alerts if r.get("relative_score", 0.0) >= min_relative_score]
            print(f"[realtime_pipeline] [tail] [replay {label}] Scored {len(week_slice):,} records "
                  f"against '{model_id}'. {pre_filter_count} alert(s) before filtering, "
                  f"{len(alerts)} alert(s) after filtering.", flush=True)
            if alerts and alert_out_file:
                with open(alert_out_file, "a") as f:
                    for row in alerts:
                        for entry in alerts_per_url(row, model_id=model_id, n_urls=max_urls_per_alert):
                            f.write(json.dumps(entry, default=str) + "\n")
        else:
            print(f"[realtime_pipeline] [tail] [replay {label}] No approved model yet — this window "
                  f"will bootstrap the first model.", flush=True)

        scheduler.ingest_records(week_slice)
        if scheduler.maybe_start_weekly_training() and scheduler._training_thread is not None:
            scheduler._training_thread.join()

        week_start = week_end
        week_num += 1

    active_id = model_library.get_active_model_id(root)
    print(f"[realtime_pipeline] [tail] Replay of existing history complete. Active model: "
          f"{active_id}. Switching to live tailing.", flush=True)
    return end_offset


def run_tail(input_file: str, root: str, poll_seconds: int, alert_out_file: str,
             approval_mode: str = "safe", history: str = "auto",
             min_relative_score: float = None, high_only: bool = False,
             require_landed_error: bool = False, max_urls_per_alert: int = None) -> None:
    """
    TRUE "one file, appended forever" mode. Unlike `serve` (which watches a
    DIRECTORY for whole new files) or `stream --input-file` (which reads a
    STATIC file once and exits), this re-opens input_file on every poll and
    reads only the bytes appended since the last read — so the same file
    can keep growing indefinitely and this process never has to restart.

    Runs forever. The ONLY way it stops is Ctrl-C / SIGTERM / killing the
    process or terminal — there is no exit condition in the loop.

    history:
      "auto" (default) — self-detects which of the two modes below applies,
                 based on whether an active model already exists under
                 `root`. No active model -> behaves exactly like "replay"
                 (fresh start, full progressive training). Active model
                 already present -> behaves exactly like "resume" (continue
                 from the saved offset, no reprocessing). This is the mode
                 to point a systemd/unattended service at: the SAME command
                 is correct whether this is the very first launch ever or
                 the 500th restart after a crash/reboot.
      "resume" — do NOT reprocess existing file history. Picks up
                 from the last saved byte-offset (a sidecar
                 <input_file>.tail_offset file), so restarting this command
                 never reprocesses data it already saw. If nothing has been
                 approved yet, does ONE lightweight bootstrap on whatever is
                 currently in the file (a single model, not a progressive
                 series) and starts tailing from the current end of file.
      "replay" — reprocesses ALL existing history in the file first, sliced
                 into consecutive 14-day windows (producing V1, V2, V3...
                 progressively, exactly like the old `stream` replay), then
                 switches into live tailing from wherever the file ends.
                 Forces a fresh progressive rebuild even if a model already
                 exists — use explicitly only when you deliberately want
                 that (e.g. after changing detection-constant config and
                 wiping model_store/ yourself).

    min_relative_score:
      Filters out alerts below this cutoff on `relative_score` — a score
      normalized against EACH model's own fused_threshold/catastrophic_threshold,
      so unlike raw fused_score it stays meaningful across model versions
      (V1, V2, V3... each have their own scale for fused_score, but not for
      relative_score). None (default) = no filtering by score.

    high_only:
      If True, drops any alert whose `severity` isn't "HIGH" before writing.
      NOTE: severity is computed per-scoring-batch (percentile within that
      batch's alerts only), so it's a relative label, not a fixed cutoff —
      combine with min_relative_score for a stable bar across batches/models.

    max_urls_per_alert:
      Caps how many distinct-URL alert lines a single anomalous WINDOW can
      produce. None (default) = uncapped, one line per distinct URL in the
      window (original behavior). Does NOT change which windows count as
      anomalous — only limits fan-out in the output file, useful when one
      loud, genuine campaign against many paths would otherwise write
      hundreds of near-duplicate lines for the same underlying event.

    Alerts are NOT printed to the console — they are appended as one JSON
    object per line to alert_out_file.

    Requires input_file to be NDJSON (one JSON object per line). A JSON
    array file ([ ... ]) cannot be safely tailed this way.
    """
    model_library.init_library(root)
    scheduler = RollingRetrainingScheduler(model_root=root, approval_mode=approval_mode)

    if alert_out_file:
        os.makedirs(os.path.dirname(alert_out_file) or ".", exist_ok=True)

    if history == "auto":
        existing_active_id = model_library.get_active_model_id(root)
        if existing_active_id is None:
            print(f"[realtime_pipeline] [tail] --history auto: no active model found under "
                  f"'{root}' — treating this as a fresh start, running a full progressive "
                  f"replay of '{input_file}'.", flush=True)
            history = "replay"
        else:
            print(f"[realtime_pipeline] [tail] --history auto: active model '{existing_active_id}' "
                  f"already exists under '{root}' — resuming from the saved offset, no "
                  f"reprocessing.", flush=True)
            history = "resume"

    if history == "replay":
        offset = _replay_history_into_scheduler(input_file, root, scheduler, alert_out_file,
                                                  min_relative_score=min_relative_score, high_only=high_only,
                                                  require_landed_error=require_landed_error,
                                                  max_urls_per_alert=max_urls_per_alert)
        _write_offset(input_file, offset)
    else:
        # "resume": trust the saved offset if one exists; otherwise this is
        # a first run, so do a single lightweight bootstrap on whatever's
        # currently in the file (not a progressive replay) and start
        # tailing from the current end of file.
        saved_offset = _read_offset(input_file)
        active_id, active_bundle = model_library.get_active_model(root)

        if active_bundle is not None:
            offset = saved_offset
        else:
            print(f"[realtime_pipeline] [tail] --history resume: no approved model and no saved "
                  f"offset — bootstrapping once from whatever's currently in '{input_file}'.", flush=True)
            import train_model
            existing_records = load_logs(input_file, label="BOOTSTRAP")
            if existing_records:
                model_id = model_library.next_model_id(root)
                week_start = min(r["ts"] for r in existing_records)
                week_end = max(r["ts"] for r in existing_records)
                bundle = train_model.train_candidate_from_records(model_id, existing_records, week_start, week_end)
                model_library.register_candidate_model(bundle, root=root)
                model_library.approve_model(model_id, root=root)
                print(f"[realtime_pipeline] [tail] Bootstrap complete — '{model_id}' is approved and active.",
                      flush=True)
            offset = os.path.getsize(input_file)
            _write_offset(input_file, offset)

    print(f"[realtime_pipeline] [tail] Watching '{input_file}' from byte offset {offset}. "
          f"approval_mode='{approval_mode}'. Retrain window: 14 days. Alerts -> "
          f"{alert_out_file or '(discarded, no --alert-out-file given)'}. "
          f"Running forever — Ctrl-C or kill the process/terminal to stop.", flush=True)

    while True:
        try:
            records, new_offset = _read_new_records(input_file, offset)

            if records:
                model_id, alerts = score_against_active_model(records, root)
                pre_filter_count = len(alerts)

                if high_only:
                    alerts = [r for r in alerts if r.get("severity") == "HIGH"]
                if require_landed_error:
                    alerts = [r for r in alerts if r.get("status_500_ratio", 0.0) > 0]
                if min_relative_score is not None:
                    alerts = [r for r in alerts if r.get("relative_score", 0.0) >= min_relative_score]

                print(f"[realtime_pipeline] [tail] {len(records):,} new record(s) scored against "
                      f"'{model_id}'. {pre_filter_count} alert(s) before filtering, "
                      f"{len(alerts)} alert(s) after filtering.", flush=True)

                if alerts and alert_out_file:
                    with open(alert_out_file, "a") as f:
                        for row in alerts:
                            for entry in alerts_per_url(row, model_id=model_id, n_urls=max_urls_per_alert):
                                f.write(json.dumps(entry, default=str) + "\n")

                scheduler.ingest_records(records)
                scheduler.maybe_start_weekly_training()

                offset = new_offset
                _write_offset(input_file, offset)

        except Exception as e:
            # A single bad poll must never take detection down.
            print(f"[realtime_pipeline] [tail] ERROR during poll: {e} — continuing.", flush=True)

        time.sleep(poll_seconds)


def run_serve(watch_dir: str, root: str, poll_seconds: int, alert_json_dir: str = None,
              approval_mode: str = "safe") -> None:
    """
    TRUE continuous/production mode. Runs forever:

      1. Polls `watch_dir` every poll_seconds for files it hasn't processed
         yet (e.g. a log shipper dropping rotated/closed log files there).
      2. Any new file is scored IMMEDIATELY against the currently approved
         model and alerts are emitted right away — this never waits on
         training.
      3. Every batch is also fed into the rolling scheduler, which trains
         the next 14-day candidate in a background thread the instant that
         much data has accumulated, and promotes/rejects it automatically
         using `approval_mode` — no console prompt, no human required.

    This process is meant to be run under a supervisor (systemd, a
    container restart policy, etc.) since it never returns on its own;
    Ctrl-C / SIGTERM is the only normal way to stop it.
    """
    os.makedirs(watch_dir, exist_ok=True)
    scheduler = RollingRetrainingScheduler(model_root=root, approval_mode=approval_mode)
    processed = set()

    print(f"[realtime_pipeline] Serving (headless, approval_mode='{approval_mode}'). "
          f"Watching '{watch_dir}' every {poll_seconds}s. Ctrl-C to stop.", flush=True)

    while True:
        try:
            files = sorted(glob.glob(os.path.join(watch_dir, "*")))
            new_files = [f for f in files if f not in processed]

            for fpath in new_files:
                try:
                    records = load_logs(fpath, label="LIVE")
                except Exception as e:
                    print(f"[realtime_pipeline] Failed to parse '{fpath}': {e} — skipping.", flush=True)
                    processed.add(fpath)
                    continue

                if records:
                    model_id, alerts = score_against_active_model(records, root)
                    print(f"[realtime_pipeline] [{os.path.basename(fpath)}] Scored {len(records):,} "
                          f"records against '{model_id}'. {len(alerts)} alert(s).", flush=True)
                    for row in alerts:
                        print_alert(row, model_id=model_id)
                    if alert_json_dir and alerts:
                        out_path = os.path.join(alert_json_dir, f"alerts_{os.path.basename(fpath)}.json")
                        save_alert_json(alerts, out_path, model_id=model_id)

                    scheduler.ingest_records(records)
                    scheduler.maybe_start_weekly_training()

                processed.add(fpath)

        except Exception as e:
            # A single bad batch must never take detection down — log and
            # keep serving. This is the headless-availability guarantee.
            print(f"[realtime_pipeline] ERROR during serve loop: {e} — continuing to serve.", flush=True)

        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description="UC-09 real-time rolling-retraining pipeline (14-day windows)")
    parser.add_argument("--model-root", default=model_library.DEFAULT_ROOT,
                         help="Model library root directory (default: ./model_store)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_bootstrap = sub.add_parser("bootstrap", help="Train + auto-approve the very first (Window 1) model")
    p_bootstrap.add_argument("--input", required=True, help="Window 1 log file (Day1-14)")

    p_score = sub.add_parser("score", help="Score one batch of logs against the currently approved model")
    p_score.add_argument("--input", required=True, help="Log file to score")
    p_score.add_argument("--alert-json-out", default=None, help="Optional path to write alerts as JSON")

    p_stream = sub.add_parser("stream", help="Simulate continuous ingestion + 14-day rolling retraining")
    stream_input = p_stream.add_mutually_exclusive_group(required=True)
    stream_input.add_argument("--input-dir", default=None,
                               help="Directory of log files, one per arrival batch")
    stream_input.add_argument("--input-file", default=None,
                               help="Single file spanning many days — auto-sliced into "
                                    "14-day windows by timestamp, no pre-splitting needed")
    p_stream.add_argument("--poll-seconds", type=int, default=2,
                           help="Delay between batches (only used with --input-dir; demo pacing)")
    p_stream.add_argument("--alert-json-dir", default=None, help="Optional directory to write per-batch alert JSON")
    p_stream.add_argument("--auto-approve", action="store_true",
                           help="Skip human review and auto-promote every candidate (testing only)")

    p_tail = sub.add_parser("tail", help="Run forever: watch ONE file that keeps growing via appends "
                                          "(NDJSON), detect + retrain automatically, alerts appended to "
                                          "a file instead of printed")
    p_tail.add_argument("--input-file", required=True,
                         help="NDJSON log file that will keep being appended to")
    p_tail.add_argument("--poll-seconds", type=int, default=30,
                         help="How often to check the file for new appended lines (default: 30)")
    p_tail.add_argument("--alert-out-file", default=None,
                         help="Path to append alerts to, one JSON object per line (default: alerts discarded)")
    p_tail.add_argument("--approval-mode", choices=["safe", "auto", "console", "json"], default="safe",
                         help="'safe' (default): auto-promote unless drift is extreme, else auto-reject "
                              "and keep serving on the current model, never blocks. 'auto': always "
                              "promote, no safety net, never blocks. 'console': auto-promote when drift "
                              "is within the approval threshold; when it EXCEEDS the threshold, block "
                              "and print a Y/N prompt for a human analyst to approve/reject the "
                              "candidate. Requires an attached interactive terminal (stdin) -- do not "
                              "use this mode under systemd/headless deployment, since nothing will be "
                              "there to answer and the pipeline will hang waiting for input.")
    p_tail.add_argument("--history", choices=["auto", "resume", "replay"], default="auto",
                         help="'auto' (default, recommended for services): no active model yet -> "
                              "full progressive replay; active model already exists -> resume from "
                              "saved offset. Use the SAME command for first-ever launch and every "
                              "restart after. 'resume': never reprocess history. 'replay': force a "
                              "fresh progressive rebuild even if a model already exists.")
    p_tail.add_argument("--min-relative-score", type=float, default=None,
                         help="Only write alerts with relative_score >= this value. relative_score is "
                              "normalized against EACH model's own thresholds (0.0 = just cleared the "
                              "burst gate, 1.0 = at the catastrophic threshold), so unlike raw "
                              "fused_score this stays meaningful across model versions/retrains. "
                              "Default: no filtering.")
    p_tail.add_argument("--high-only", action="store_true",
                         help="Only write alerts with severity == HIGH (drops LOW/MEDIUM). NOTE: "
                              "severity is relative to each scoring batch, not a fixed cutoff — combine "
                              "with --min-relative-score for a stable bar.")
    p_tail.add_argument("--require-landed-error", action="store_true",
                         help="Only write alerts where status_500_ratio > 0 — meaning the window "
                              "actually returned a 5xx, not just a statistically loud burst. Mass "
                              "internet recon/scanning (probing /.env, /wp-content/*, /cgi-bin/php, etc.) "
                              "is genuinely anomalous vs. your normal traffic and can clear a high "
                              "relative_score purely on magnitude while being all-404s — this flag "
                              "filters those out. (Checks status_500_ratio directly, NOT alert_kind — "
                              "alert_kind's 'catastrophic' label is rarely applied since it's suppressed "
                              "whenever the burst condition also fires, which most real attacks also "
                              "trigger.)")
    p_tail.add_argument("--max-urls-per-alert", type=int, default=None,
                         help="Caps how many distinct-URL lines a single anomalous window can write "
                              "(default: uncapped). Does NOT change which windows count as anomalous -- "
                              "only limits fan-out in the output file for one loud campaign hitting many "
                              "paths in the same window.")

    p_serve = sub.add_parser("serve", help="Run forever: watch a DIRECTORY for new logs, detect + retrain "
                                            "automatically, no human intervention")
    p_serve.add_argument("--watch-dir", required=True,
                          help="Directory a log shipper drops new/rotated log files into")
    p_serve.add_argument("--poll-seconds", type=int, default=30,
                          help="How often to check watch-dir for new files (default: 30)")
    p_serve.add_argument("--alert-json-dir", default=None, help="Optional directory to write per-batch alert JSON")
    p_serve.add_argument("--approval-mode", choices=["safe", "auto", "console", "json"], default="safe",
                          help="'safe' (default): auto-promote unless drift is extreme, else auto-reject "
                               "and keep serving on the current model, never blocks. 'auto': always "
                               "promote, no safety net, never blocks. 'console': human Y/N approval "
                               "when drift exceeds the threshold -- requires an attached interactive "
                               "terminal, do not use headless.")

    args = parser.parse_args()

    if args.command == "bootstrap":
        run_bootstrap(args.input, args.model_root)
    elif args.command == "score":
        run_score(args.input, args.model_root, alert_json_out=args.alert_json_out)
    elif args.command == "stream":
        if args.input_file:
            run_stream_from_file(args.input_file, args.model_root,
                                  alert_json_dir=args.alert_json_dir, auto_approve=args.auto_approve)
        else:
            run_stream(args.input_dir, args.model_root, args.poll_seconds,
                       alert_json_dir=args.alert_json_dir, auto_approve=args.auto_approve)
    elif args.command == "tail":
        run_tail(args.input_file, args.model_root, args.poll_seconds,
                 args.alert_out_file, approval_mode=args.approval_mode, history=args.history,
                 min_relative_score=args.min_relative_score, high_only=args.high_only,
                 require_landed_error=args.require_landed_error,
                 max_urls_per_alert=args.max_urls_per_alert)
    elif args.command == "serve":
        run_serve(args.watch_dir, args.model_root, args.poll_seconds,
                  alert_json_dir=args.alert_json_dir, approval_mode=args.approval_mode)


if __name__ == "__main__":
    main()
