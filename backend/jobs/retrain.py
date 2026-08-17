"""
Retrain job — spawn `ml.training.train` as a subprocess, evaluate against promotion gate,
and atomically hot-swap active model in PostgreSQL and filesystem pointer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from prisma import Prisma

from backend.store.client import db
from backend.store import repo
from ml.training.artifacts import ArtifactStore
from ml.training.evaluate import should_promote

log = logging.getLogger("backend.jobs.retrain")

# Retrain execution lock to prevent concurrent subprocesses
_retrain_lock = asyncio.Lock()


async def run_retrain(settings, client: Prisma = db) -> dict:
    """Execute the training pipeline as a subprocess, evaluate promotion against
    current active run, and record in PostgreSQL registry."""
    if _retrain_lock.locked():
        log.warning("Retrain job already running; skipping.")
        return {"status": "busy", "message": "A retraining job is currently in progress."}

    async with _retrain_lock:
        project_root = settings.project_root
        run_id = f"v6-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        log.info("Starting retraining run: %s", run_id)

        cmd = [sys.executable, "-m", "ml.training.train", "--run-id", run_id]
        loop = asyncio.get_event_loop()

        def _run_subproc():
            return subprocess.run(
                cmd,
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=settings.retrain_timeout_seconds,
            )

        try:
            proc = await loop.run_in_executor(None, _run_subproc)
        except subprocess.TimeoutExpired:
            log.error("Retrain job timed out after %d seconds.", settings.retrain_timeout_seconds)
            return {"status": "timeout", "run_id": run_id}
        except Exception as e:
            log.error("Retrain execution failed: %s", e)
            return {"status": "error", "run_id": run_id, "detail": str(e)}

        if proc.returncode != 0:
            log.error("Retrain subprocess failed (code %d): %s", proc.returncode, proc.stderr[-500:] if proc.stderr else "")
            return {"status": "failed", "run_id": run_id, "stderr": proc.stderr[-500:]}

        # --- Promotion Gate ---
        models_dir = settings.models_dir_path
        run_dir = models_dir / run_id
        cfg_path = run_dir / "inference_config_v6.json"
        if not cfg_path.exists():
            log.warning("Retrain finished but config not found at %s", cfg_path)
            return {"status": "no_artifacts", "run_id": run_id}

        new_cfg = json.loads(cfg_path.read_text())
        new_metrics = new_cfg.get("metrics", {})
        schema_json = json.dumps(new_cfg.get("feature_schema", []))

        old_metrics = None
        active_id = ArtifactStore.read_latest_pointer(models_dir)
        if active_id:
            old_cfg_path = models_dir / active_id / "inference_config_v6.json"
            if old_cfg_path.exists():
                old_metrics = json.loads(old_cfg_path.read_text()).get("metrics")

        promote, reason = should_promote(
            new_metrics,
            old_metrics,
            tol_cl=0.02,
            tol_ph=0.005,
            tol_turb=0.01,
        )

        # Register run in PostgreSQL
        await repo.add_model_run(
            run_id=run_id,
            artifact_dir=str(run_dir),
            metrics_json=json.dumps(new_metrics),
            feature_schema_json=schema_json,
            is_active=1 if promote else 0,
            client=client,
        )

        if promote:
            ArtifactStore.write_latest_pointer(models_dir, run_id)
            await repo.set_active_model_run(run_id, reason, client=client)
            log.info("Retrain PROMOTED %s: %s", run_id, reason)
            return {"status": "promoted", "run_id": run_id, "reason": reason, "metrics": new_metrics}
        else:
            log.warning("Retrain REJECTED %s: %s", run_id, reason)
            return {"status": "rejected", "run_id": run_id, "reason": reason, "metrics": new_metrics}


async def should_retrain(settings, client: Prisma = db) -> bool:
    """Check if accumulated new readings since active run exceed retrain threshold."""
    active = await repo.get_active_model_run(client=client)
    if active is None:
        return False
    count = await client.reading.count(
        where={"created_at": {"gt": active.created_at}}
    )
    log.info("Retrain check: %d new readings (threshold %d)", count, settings.retrain_min_new_readings)
    return count >= settings.retrain_min_new_readings