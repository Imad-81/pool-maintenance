"""
Admin endpoints — model registry, retrain triggers, weather synchronization, and ingestion audit logs.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from prisma import Prisma

from backend.store.client import get_db, db
from backend.store import repo
from backend.jobs.retrain import run_retrain
from backend.weather.provider import refresh as refresh_weather
from backend.settings import settings

router = APIRouter(prefix="/api/admin", tags=["admin"])
log = logging.getLogger("backend.api.admin")


def _check_admin_token(authorization: Optional[str] = Header(None)) -> None:
    """Validate admin Bearer token if `admin_token` is configured."""
    token = settings.admin_token
    if token is None:
        return
    if not authorization:
        raise HTTPException(401, "Authorization header required for admin operations.")
    parts = authorization.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or parts[1] != token:
        raise HTTPException(403, "Invalid admin token.")


@router.get("/runs")
async def list_runs(
    authorization: Optional[str] = Header(None),
    client: Prisma = Depends(get_db),
):
    _check_admin_token(authorization)
    runs = await repo.list_model_runs(limit=30, client=client)
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
async def trigger_retrain(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Trigger a retraining run on demand."""
    _check_admin_token(authorization)
    log.info("Manual model retraining triggered via admin API.")
    try:
        result = await run_retrain(settings)
    except Exception as e:
        log.exception("Retrain failed: %s", e)
        raise HTTPException(500, f"Retraining failed: {e}")

    # Hot-reload the prediction service
    svc = getattr(request.app.state, "prediction_service", None)
    if svc:
        try:
            svc.reload()
            log.info("Prediction service reloaded with new active model.")
        except Exception as e:
            log.warning("Model reload after retrain notice: %s", e)

    return {"status": "completed", "result": result}


@router.get("/weather-status")
async def weather_status(
    authorization: Optional[str] = Header(None),
    client: Prisma = Depends(get_db),
):
    _check_admin_token(authorization)
    latest = await repo.get_latest_weather_date(client=client)
    return {
        "latest_weather_date": str(latest.date()) if latest else "never fetched",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/weather-refresh")
async def trigger_weather_refresh(
    authorization: Optional[str] = Header(None),
    client: Prisma = Depends(get_db),
):
    """Trigger weather synchronization from Open-Meteo."""
    _check_admin_token(authorization)
    log.info("Manual weather refresh triggered via admin API.")
    try:
        n = await refresh_weather(client=client)
    except Exception as e:
        log.exception("Weather refresh failed: %s", e)
        raise HTTPException(500, f"Weather refresh failed: {e}")
    return {"status": "ok", "rows_upserted": n}


@router.get("/ingest-log")
async def get_ingest_log(
    authorization: Optional[str] = Header(None),
    client: Prisma = Depends(get_db),
):
    _check_admin_token(authorization)
    logs = await repo.list_ingest_logs(limit=50, client=client)
    return [
        {
            "id": l.id,
            "source": l.source,
            "filename": l.filename,
            "pool_count": l.pool_count,
            "row_count": l.row_count,
            "skipped_count": l.skipped_count,
            "created_at": l.created_at.isoformat() if l.created_at else None,
            "detail": json.loads(l.detail_json) if l.detail_json else None,
        }
        for l in logs
    ]