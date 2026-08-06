"""FastAPI dependency injection — providers for PredictionService, Store
session, Scheduler, and the Weather lookup callable.

All deps are lightweight (no DB on import) — the router uses `Depends()` to
get them per-request.
"""

from __future__ import annotations

import logging

from fastapi import Request

from backend.store.schema import get_session
from backend.store import repo
from backend.weather.provider import make_lookup
from ml.inference.predictor import PredictionService

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton in app.state (set in lifespan)
# ---------------------------------------------------------------------------

def get_prediction_service(request: Request) -> PredictionService:
    svc: PredictionService = request.app.state.prediction_service
    if svc is None:
        raise RuntimeError("PredictionService not initialised (check app lifespan)")
    return svc


def get_weather_lookup(request: Request):
    """Return a callable for `predict_forward` that queries the SQLite
    weather cache and returns NaN for missing dates."""
    session = next(get_session())
    try:
        return make_lookup(session)
    finally:
        session.close()


# ---------------------------------------------------------------------------
# get_session is re-exported from store.schema for convenience
# ---------------------------------------------------------------------------
__all__ = ["get_prediction_service", "get_weather_lookup", "get_session"]