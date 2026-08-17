"""
Health-check and readiness endpoints for Docker and Kubernetes orchestration.
"""

from datetime import datetime, timezone
from fastapi import APIRouter, Request, Response
from backend.store.client import is_db_connected
from backend.store import repo

router = APIRouter(tags=["health"])


@router.get("/healthz")
@router.get("/healthz/live")
def liveness():
    """Liveness probe to ensure process is running."""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/healthz/ready")
async def readiness(request: Request, response: Response):
    """Readiness probe checking PostgreSQL database, Model, and Weather freshness."""
    db_ok = await is_db_connected()
    svc = getattr(request.app.state, "prediction_service", None)
    model_ok = svc.is_loaded() if svc else False

    status_data = {
        "status": "ready" if (db_ok and model_ok) else "degraded",
        "database": "connected" if db_ok else "disconnected",
        "model": svc.status() if svc else {"loaded": False},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if not db_ok:
        response.status_code = 503
    return status_data