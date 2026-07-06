"""
scheduler.py
=============
Owns the rolling retraining lifecycle, now on a 14-day cadence:

    Window 1: train Day1-14   -> UC09_Model_V1 (approved, since nothing exists yet)
    Window 2: detect with V1, train candidate V2 on Day15-28 in the background
    Window 3: detect with V2, train candidate V3 on Day29-42 in the background
    ...

Training NEVER blocks detection: register_candidate_model() and the drift
comparison happen on a background thread; realtime_pipeline.py keeps
reading model_library.get_active_model() on every batch, which only
changes at the instant approve_model()/rollback_model() is called.

This module intentionally does not care whether "log arrival" is a Kafka
topic, a tailed file, or a directory of NDJSON dumps — it exposes
ingest_records() for whatever ingestion mechanism realtime_pipeline.py (or
a real log shipper) is using, and just tracks a rolling window in memory.
For anything beyond a demo/single-node deployment, replace RollingLogBuffer
with a real store (e.g. a 14-day retention topic / table) but keep the same
interface.
"""

import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timedelta

import model_library
import train_model
from drift_detection import compare_models, DRIFT_APPROVAL_THRESHOLD

# Retraining cadence. Was 7 days; now 14.
WEEK = timedelta(days=14)


class RollingLogBuffer:
    """Thread-safe rolling store of parsed log records, retained for RETENTION_WEEKS."""

    def __init__(self, retention_weeks: int = 2):
        self._lock = threading.Lock()
        self._records = deque()
        self.retention = retention_weeks * WEEK

    def add(self, records: list) -> None:
        if not records:
            return
        with self._lock:
            self._records.extend(records)
            self._prune_locked()

    def _prune_locked(self) -> None:
        if not self._records:
            return
        newest = max(r["ts"] for r in self._records)
        cutoff = newest - self.retention
        while self._records and self._records[0]["ts"] < cutoff:
            self._records.popleft()

    def window(self, start: datetime, end: datetime) -> list:
        with self._lock:
            return [r for r in self._records if start <= r["ts"] < end]

    def earliest_ts(self):
        with self._lock:
            return min((r["ts"] for r in self._records), default=None)

    def latest_ts(self):
        with self._lock:
            return max((r["ts"] for r in self._records), default=None)


class RollingRetrainingScheduler:
    """
    Drives the rolling train/approve/detect cycle (14-day windows).

    approval_callback(review_text: str, drift_result: dict) -> bool
        Called ONLY when drift_result["requires_approval"] is True. Default
        implementation prints the review block from the spec and blocks on
        console input; swap in a Slack/ticketing hook for production.
    """

    # A drift_score above this is treated as "too risky to auto-promote even
    # in headless mode" — the candidate is auto-rejected (old model keeps
    # serving) and the review block is only logged, never blocked on.
    SAFE_AUTO_REJECT_ABOVE = 0.85


    def __init__(self, model_root: str = model_library.DEFAULT_ROOT,
                 approval_callback=None, auto_approve: bool = False,
                 approval_mode: str = "console"):
        """
        approval_mode (used only when approval_callback is not explicitly
        given):
          "console" — default; blocks on input() for a human [Y]/[N]. Fine
                      for interactive backtests/demos, NEVER use for an
                      unattended deployment (nothing will be there to answer).
          "safe"    — headless/no-human-intervention mode. Never blocks.
                      Drift within SAFE_AUTO_REJECT_ABOVE -> auto-promote.
                      Drift above it -> auto-reject, keep the current model,
                      just log the review block for audit/ops visibility.
          "auto"    — headless, always promote regardless of drift (same
                      effect as auto_approve=True). Use with caution.
        """
        self.model_root = model_root
        self.buffer = RollingLogBuffer(retention_weeks=2)
        if approval_callback is not None:
            self.approval_callback = approval_callback
        elif approval_mode == "safe":
            self.approval_callback = self._safe_headless_approval
        elif approval_mode == "auto":
            self.approval_callback = self._auto_headless_approval
        elif approval_mode == "json":
            self.approval_callback = self._json_approval
        else:
            self.approval_callback = self._console_approval
        self.auto_approve = auto_approve or approval_mode == "auto"
        self._current_week_start = None
        self._training_thread = None
        model_library.init_library(model_root)

    # ── ingestion ──────────────────────────────────────────────────────
    def ingest_records(self, records: list) -> None:
        """Feed newly-arrived, already-parsed log records into the rolling buffer."""
        self.buffer.add(records)
        if self._current_week_start is None:
            earliest = self.buffer.earliest_ts()
            if earliest is not None:
                self._current_week_start = earliest

    def ingest_from_file(self, filepath: str) -> None:
        from feature_extraction import load_logs
        self.ingest_records(load_logs(filepath, label="STREAM"))

    # ── window boundary check ────────────────────────────────────────
    def maybe_start_weekly_training(self) -> bool:
        """
        Call this periodically (e.g. once per detection batch, or on a
        cron-style timer). If a full 14-day window of new data has
        accumulated since the last training boundary, kicks off background
        candidate training and returns True. Otherwise returns False
        immediately (non-blocking).
        """
        if self._current_week_start is None:
            return False

        latest = self.buffer.latest_ts()
        if latest is None:
            return False

        window_end = self._current_week_start + WEEK
        if latest < window_end:
            return False  # window not complete yet

        if self._training_thread is not None and self._training_thread.is_alive():
            return False  # previous window's candidate still training

        window_records = self.buffer.window(self._current_week_start, window_end)
        model_id = model_library.next_model_id(self.model_root)

        print(f"[scheduler] Window complete ({self._current_week_start} -> {window_end}, "
              f"{len(window_records):,} records). Launching background training for '{model_id}'.",
              flush=True)

        self._training_thread = threading.Thread(
            target=self._train_and_review,
            args=(model_id, window_records, self._current_week_start, window_end),
            daemon=True,
        )
        self._training_thread.start()

        self._current_week_start = window_end  # next window starts where this one ended
        return True

    # ── background worker ───────────────────────────────────────────
    def _train_and_review(self, model_id: str, week_records: list, week_start, week_end) -> None:
        try:
            bundle = train_model.train_candidate_from_records(model_id, week_records, week_start, week_end)
        except ValueError as e:
            print(f"[scheduler] Training skipped for '{model_id}': {e}", flush=True)
            return

        model_library.register_candidate_model(bundle, root=self.model_root)

        active_id, approved_bundle = model_library.get_active_model(self.model_root)
        if approved_bundle is None:
            # Bootstrap case: Window 1, nothing approved yet -> auto-promote.
            print(f"[scheduler] No approved model exists yet — auto-promoting '{model_id}' "
                  f"as the initial approved model.", flush=True)
            model_library.approve_model(model_id, root=self.model_root)
            return

        drift_result = compare_models(approved_bundle, bundle)
        self._resolve_candidate(active_id, model_id, drift_result)

    def _resolve_candidate(self, current_model_id: str, candidate_model_id: str, drift_result: dict) -> None:
        if not drift_result["requires_approval"] or self.auto_approve:
            print(f"[scheduler] Drift score {drift_result['drift_score']:.2f} <= "
                  f"{drift_result['approval_threshold']:.2f} — auto-promoting '{candidate_model_id}'.",
                  flush=True)
            model_library.approve_model(candidate_model_id, root=self.model_root)
            return

        review_text = self._format_review(current_model_id, candidate_model_id, drift_result)
        approved = self.approval_callback(review_text, drift_result)
        if approved:
            model_library.approve_model(candidate_model_id, root=self.model_root)
        else:
            model_library.reject_candidate(candidate_model_id, root=self.model_root)

    @staticmethod
    def _format_review(current_model_id: str, candidate_model_id: str, drift_result: dict) -> str:
        return (
            "\n=================================================\n"
            "\n              MODEL REPLACEMENT REVIEW\n\n"
            f"Current Model: {current_model_id}\n\n"
            f"Candidate Model: {candidate_model_id}\n\n"
            f"Feature Drift Score: {drift_result['drift_score']:.2f}\n\n"
            f"Threshold: {drift_result['approval_threshold']:.2f}\n\n"
            "Significant behaviour change detected.\n\n"
            "Approve replacement?\n\n"
            "[Y] Yes\n"
            "[N] No\n"
            "\n=================================================\n"
        )


    def _json_approval(self, review_text: str, drift_result: dict) -> bool:

        pending_dir = os.path.join(
            self.model_root,
            "pending_approval"
        )

        os.makedirs(pending_dir, exist_ok=True)

        approval_file = os.path.join(
            pending_dir,
            "approval.json"
        )

        with open(approval_file, "w") as f:
            json.dump(
                {
                    "drift_score": drift_result["drift_score"],
                    "approval_threshold": drift_result["approval_threshold"],
                    "status": "PENDING"
                },
                f,
                indent=4
            )

        print(
            f"Waiting for analyst decision in {approval_file}",
            flush=True
        )

        while True:

            with open(approval_file) as f:
                data = json.load(f)

            status = data.get("status", "PENDING").upper()

            if status == "APPROVED":
                return True

            if status == "REJECTED":
                return False

            time.sleep(5)
    @staticmethod
    def _console_approval(review_text: str, drift_result: dict) -> bool:
        print(review_text, flush=True)
        answer = input("Approve replacement? [Y/N]: ").strip().lower()
        return answer == "y"

    @classmethod
    def _safe_headless_approval(cls, review_text: str, drift_result: dict) -> bool:
        """
        Never blocks. Logs the same review block a human would see (for
        audit trails / ops dashboards), then decides automatically: promote
        unless drift is extreme, in which case reject and keep serving on
        the current model. This is the recommended mode for an unattended
        production deployment.
        """
        print(review_text, flush=True)
        if drift_result["drift_score"] <= cls.SAFE_AUTO_REJECT_ABOVE:
            print(f"[scheduler] Headless mode: drift {drift_result['drift_score']:.2f} <= "
                  f"safety ceiling {cls.SAFE_AUTO_REJECT_ABOVE:.2f} — auto-promoting without "
                  f"human input.", flush=True)
            return True
        print(f"[scheduler] Headless mode: drift {drift_result['drift_score']:.2f} EXCEEDS safety "
              f"ceiling {cls.SAFE_AUTO_REJECT_ABOVE:.2f} — auto-REJECTING this candidate and "
              f"continuing to serve on the current model. Flag this run for later ops review.",
              flush=True)
        return False

    @staticmethod
    def _auto_headless_approval(review_text: str, drift_result: dict) -> bool:
        """Never blocks, always promotes regardless of drift. No safety net — use with caution."""
        print(review_text, flush=True)
        print("[scheduler] Headless 'auto' mode: promoting regardless of drift score.", flush=True)
        return True

    # ── convenience for a long-running process ──────────────────────
    def run_forever(self, poll_seconds: int = 3600) -> None:
        """
        Simple polling loop for a standalone scheduler process: checks the
        window boundary once per poll_seconds. Real deployments will more
        likely call maybe_start_weekly_training() from whatever already
        drives log ingestion (a Kafka consumer loop, a cron job, etc.)
        instead of running this loop directly.
        """
        print(f"[scheduler] Running (checking every {poll_seconds}s for a completed 14-day window)...", flush=True)
        while True:
            self.maybe_start_weekly_training()
            time.sleep(poll_seconds)