"""
model_library.py
==================
Versioned storage for UC-09 model bundles.

Disk layout (default root: ./model_store — kept out of the way of this
module's own filename so `import model_library` never collides with a
same-named package directory):

    model_store/
      approved/
        UC09_Model_V1.joblib
        UC09_Model_V2.joblib
      candidate/
        UC09_Model_V3.joblib        <- present only while awaiting approval
      metadata/
        UC09_Model_V1.json
        UC09_Model_V2.json
        UC09_Model_V3.json
        active.json                 <- {"active_model_id": "UC09_Model_V2"}

Only ONE model is ever "approved & active" for real-time detection at a
time. Candidates sit in candidate/ until approve_model() promotes them
(moves the .joblib into approved/, updates active.json) or they're
discarded after a rejected review.
"""

import json
import os
import shutil
import time

import joblib

from feature_extraction import MODEL_VERSION, FEATURE_COLS, current_config

DEFAULT_ROOT = "model_store"


def _paths(root: str):
    approved = os.path.join(root, "approved")
    candidate = os.path.join(root, "candidate")
    metadata = os.path.join(root, "metadata")
    return approved, candidate, metadata


def init_library(root: str = DEFAULT_ROOT) -> None:
    approved, candidate, metadata = _paths(root)
    for d in (approved, candidate, metadata):
        os.makedirs(d, exist_ok=True)
    active_file = os.path.join(metadata, "active.json")
    if not os.path.exists(active_file):
        with open(active_file, "w") as f:
            json.dump({"active_model_id": None}, f)


def _metadata_path(root: str, model_id: str) -> str:
    return os.path.join(_paths(root)[2], f"{model_id}.json")


def _write_metadata(root: str, model_id: str, bundle: dict, status: str) -> None:
    meta = {
        "model_id": model_id,
        "status": status,  # "candidate" | "approved" | "rejected" | "rolled_back"
        "model_version": bundle.get("version"),
        "created_at": bundle.get("created_at"),
        "train_window_start": str(bundle.get("train_window_start")),
        "train_window_end": str(bundle.get("train_window_end")),
        "n_train_windows": bundle.get("n_train_windows"),
        "fused_threshold": bundle.get("fused_threshold"),
        "catastrophic_threshold": bundle.get("catastrophic_threshold"),
        "config": bundle.get("config"),
        "updated_at": time.time(),
    }
    with open(_metadata_path(root, model_id), "w") as f:
        json.dump(meta, f, indent=2)


def register_candidate_model(bundle: dict, root: str = DEFAULT_ROOT) -> str:
    """
    Saves a freshly trained bundle (from train_model.train_candidate_from_records)
    into candidate/ and writes its metadata record. Does NOT touch the
    currently active/approved model — detection keeps using it untouched.
    """
    init_library(root)
    _, candidate_dir, _ = _paths(root)
    model_id = bundle["model_id"]
    path = os.path.join(candidate_dir, f"{model_id}.joblib")
    joblib.dump(bundle, path)
    _write_metadata(root, model_id, bundle, status="candidate")
    print(f"[model_library] Candidate '{model_id}' registered at {path}", flush=True)
    return path


def approve_model(model_id: str, root: str = DEFAULT_ROOT) -> str:
    """
    Promotes a candidate to approved+active. Moves the .joblib from
    candidate/ to approved/, updates active.json, updates metadata status.
    This is the only function that changes what real-time detection uses.
    """
    approved_dir, candidate_dir, metadata_dir = _paths(root)
    src = os.path.join(candidate_dir, f"{model_id}.joblib")
    dst = os.path.join(approved_dir, f"{model_id}.joblib")
    if not os.path.exists(src):
        raise FileNotFoundError(f"[model_library] No candidate bundle found for '{model_id}' at {src}")

    shutil.move(src, dst)
    with open(os.path.join(metadata_dir, "active.json"), "w") as f:
        json.dump({"active_model_id": model_id, "promoted_at": time.time()}, f)

    meta_path = _metadata_path(root, model_id)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        meta["status"] = "approved"
        meta["approved_at"] = time.time()
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    print(f"[model_library] '{model_id}' APPROVED and now ACTIVE for real-time detection.", flush=True)
    return dst


def reject_candidate(model_id: str, root: str = DEFAULT_ROOT) -> None:
    """Discards a candidate that a human reviewer declined to promote."""
    _, candidate_dir, metadata_dir = _paths(root)
    src = os.path.join(candidate_dir, f"{model_id}.joblib")
    if os.path.exists(src):
        os.remove(src)
    meta_path = _metadata_path(root, model_id)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        meta["status"] = "rejected"
        meta["rejected_at"] = time.time()
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    print(f"[model_library] Candidate '{model_id}' rejected and discarded.", flush=True)


def rollback_model(target_model_id: str, root: str = DEFAULT_ROOT) -> str:
    """
    Reverts the active model to a previously approved version (e.g. the new
    approved model turns out to alert-flood or go silent in production).
    target_model_id must already exist in approved/.
    """
    approved_dir, _, metadata_dir = _paths(root)
    target_path = os.path.join(approved_dir, f"{target_model_id}.joblib")
    if not os.path.exists(target_path):
        raise FileNotFoundError(
            f"[model_library] Cannot roll back — '{target_model_id}' is not in approved/."
        )
    with open(os.path.join(metadata_dir, "active.json"), "w") as f:
        json.dump({"active_model_id": target_model_id, "rolled_back_at": time.time()}, f)
    print(f"[model_library] Rolled back — active model is now '{target_model_id}'.", flush=True)
    return target_path


def get_active_model_id(root: str = DEFAULT_ROOT):
    init_library(root)
    active_file = os.path.join(_paths(root)[2], "active.json")
    if not os.path.exists(active_file):
        return None
    with open(active_file) as f:
        return json.load(f).get("active_model_id")


def load_model_bundle(path: str) -> dict:
    """Unchanged behavior from uc09_v3.2.load_model_bundle(): loads + warns on drift."""
    bundle = joblib.load(path)

    if bundle.get("version") != MODEL_VERSION:
        print(f"[model_library] WARNING: bundle version '{bundle.get('version')}' != "
              f"current '{MODEL_VERSION}'. Retrain if unsure.", flush=True)
    if bundle.get("feature_cols") != FEATURE_COLS:
        print("[model_library] WARNING: FEATURE_COLS drift vs. loaded bundle. Retrain.", flush=True)
    mismatches = [k for k, v in current_config().items() if bundle.get("config", {}).get(k) != v]
    if mismatches:
        print(f"[model_library] WARNING: config drift vs. saved model for: {mismatches}", flush=True)

    return bundle


def get_active_model(root: str = DEFAULT_ROOT):
    """
    Convenience used by realtime_pipeline.py: resolves the active_model_id
    and loads its bundle from approved/. Returns (model_id, bundle) or
    (None, None) if nothing has been approved yet.
    """
    model_id = get_active_model_id(root)
    if model_id is None:
        return None, None
    approved_dir, _, _ = _paths(root)
    path = os.path.join(approved_dir, f"{model_id}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[model_library] active.json points to '{model_id}' but {path} is missing."
        )
    return model_id, load_model_bundle(path)


def list_models(root: str = DEFAULT_ROOT) -> list:
    """Lists all metadata records, most recent first — useful for a status CLI."""
    init_library(root)
    _, _, metadata_dir = _paths(root)
    records = []
    for fname in os.listdir(metadata_dir):
        if fname == "active.json" or not fname.endswith(".json"):
            continue
        with open(os.path.join(metadata_dir, fname)) as f:
            records.append(json.load(f))
    records.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return records


def next_model_id(root: str = DEFAULT_ROOT) -> str:
    """UC09_Model_V1, V2, V3... based on how many metadata records exist so far."""
    existing = list_models(root)
    return f"UC09_Model_V{len(existing) + 1}"