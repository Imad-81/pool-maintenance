"""Admin endpoints — model runs, retrain trigger, weather status, ingest log."""

import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlmodel import Session

from backend.store.schema import get_session
from backend.store import repo
from backend.jobs.retrain import run_retrain
from backend.jobs.weather_refresh import run_weather_refresh

router = APIRouter(prefix="/api/admin", tags=["admin"])
log = logging.getLogger(__name__)


@router.get("/runs")
def list_runs(session: Session = Depends(get_session)):
    runs = repo.list_model_runs(session)
    return [
        {
            "run_id": r.run_id,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "is_active": bool(r.is_active),
            "metrics": json.loads(r.metrics_json) if r.metrics_json else None,
            "promoted_at": r.promoted_at.isoformat() if r.promoted_at else None,
            "promote_reason": r.promote_reason,
        }
        for r in runs
    ]


@router.post("/retrain")
def trigger_retrain(request: Request):
    """Trigger a retraining run now. Blocks until complete."""
    log.info("manual retrain triggered via admin")
    from backend.settings import settings
    try:
        result = run_retrain(settings)
    except Exception as e:
        raise HTTPException(500, f"Retrain failed: {e}")
    # Reload the prediction service with new models
    svc = request.app.state.prediction_service
    if svc:
        svc.reload()
    return {"status": "completed", "result": result}


@router.get("/weather-status")
def weather_status(session: Session = Depends(get_session)):
    latest = repo.get_latest_weather_date(session)
    return {"latest_weather_date": str(latest) if latest else "never fetched"}


@router.post("/weather-refresh")
def trigger_weather_refresh(session: Session = Depends(get_session)):
    try:
        n = run_weather_refresh(session)
    except Exception as e:
        raise HTTPException(500, f"Weather refresh failed: {e}")
    return {"status": "ok", "rows_upserted": n}


@router.get("/ingest-log")
def ingest_log(session: Session = Depends(get_session)):
    logs = repo.list_ingest_logs(session)
    return [
        {
            "source": l.source, "filename": l.filename,
            "pool_count": l.pool_count, "row_count": l.row_count,
            "skipped_count": l.skipped_count,
            "created_at": l.created_at.isoformat() if l.created_at else None,
        }
        for l in logs
    ]