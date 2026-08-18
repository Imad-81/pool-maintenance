"""
FastAPI dependency injection — providers for PredictionService, Prisma client, and Weather lookup.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Callable

# pyrefly: ignore [missing-import]
from fastapi import Request
from prisma import Prisma

from backend.store.client import db, get_db
from backend.weather.provider import make_lookup, warm_weather_cache
from ml.inference.predictor import PredictionService
from backend.settings import settings

log = logging.getLogger("backend.deps")


def get_prediction_service(request: Request) -> PredictionService:
    """Return PredictionService singleton from app state or lazy-initialize if in test mode."""
    svc = getattr(request.app.state, "prediction_service", None)
    if svc is None:
        svc = PredictionService(settings.models_dir_path)
        try:
            svc.load()
        except Exception as e:
            log.warning("Lazy PredictionService load: %s", e)
        request.app.state.prediction_service = svc
    return svc


async def get_weather_lookup_provider(request: Request) -> Callable:
    """Return synchronous weather lookup callable backed by warmed cache."""
    cache = await warm_weather_cache(client=db)
    return make_lookup(cache)


__all__ = ["get_prediction_service", "get_weather_lookup_provider", "get_db"]