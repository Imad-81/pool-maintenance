"""Retrain job — spawn `ml.training.train` as a subprocess and promote if
the new run passes the evaluation gate.

Called by APScheduler (weekly) or manually via GET /api/admin/retrain.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from ml.training.artifacts import ArtifactStore
from ml.training.evaluate import should_promote

log = logging.getLogger(__name__)


def run_retrain(settings) -> dict:
    """Execute the training pipeline as a subprocess, then (if it wrote
    artifacts) evaluate promotion against the current active run.

    Returns a status dict suitable for the admin endpoint.
    """
    project_root = settings.project_root
    run_id = f"v6-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log.info("retrain: starting run %s", run_id)

    cmd = [sys.executable, "-m", "ml.training.train", "--run-id", run_id]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=settings.retrain_timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        log.error("retrain timed out after %ds", settings.retrain_timeout_seconds)
        return {"status": "timeout", "run_id": run_id}

    if proc.returncode != 0:
        log.error("retrain exited %d: %s", proc.returncode, proc.stderr[-500:] if proc.stderr else "no stderr")
        return {"status": "failed", "run_id": run_id, "stderr": proc.stderr[-500:]}

    # --- promotion gate ---
    models_dir = project_root / "models"
    run_dir = models_dir / run_id
    cfg_path = run_dir / "inference_config_v6.json"
    if not cfg_path.exists():
        log.warning("retrain ran but no config at %s", cfg_path)
        return {"status": "no_artifacts", "run_id": run_id}

    new_cfg = json.loads(cfg_path.read_text())
    new_metrics = new_cfg.get("metrics", {})

    old_metrics = None
    active_id = ArtifactStore.read_latest_pointer(models_dir)
    if active_id:
        old_cfg_path = models_dir / active_id / "inference_config_v6.json"
        if old_cfg_path.exists():
            old_metrics = json.loads(old_cfg_path.read_text()).get("metrics")

    promote, reason = should_promote(
        new_metrics, old_metrics,
        tol_cl=settings.model_config.get("..", 0.02),  ## TODO: wire from PipelineConfig
        tol_ph=settings.model_config.get("..", 0.005),
    )
    # Use defaults since settings doesn't have these directly:
    promote, reason = should_promote(new_metrics, old_metrics, tol_cl=0.02, tol_ph=0.005)

    if promote:
        ArtifactStore.write_latest_pointer(models_dir, run_id)
        log.info("retrain PROMOTED %s — %s", run_id, reason)
        return {"status": "promoted", "run_id": run_id, "reason": reason}
    else:
        log.warning("retrain NOT promoted %s — %s", run_id, reason)
        return {"status": "rejected", "run_id": run_id, "reason": reason}


def should_retrain(settings, session) -> bool:
    """Check if enough new readings have accumulated since the last run."""
    from backend.store import repo
    active = repo.get_active_model_run(session)
    if active is None:
        return False
    # Count readings added after the active run's creation date
    from backend.store.schema import Reading
    count = session.query(Reading).filter(
        Reading.created_at > active.created_at
    ).count()
    log.info("retrain check: %d new readings (threshold %d)", count, settings.retrain_min_new_readings)
    return count >= settings.retrain_min_new_readings