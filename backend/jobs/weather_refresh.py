"""
Weather refresh background job.
"""

from __future__ import annotations

import logging
from prisma import Prisma

from backend.store.client import db
from backend.weather.provider import refresh

log = logging.getLogger("backend.jobs.weather_refresh")


async def run_weather_refresh(client: Prisma = db) -> int:
    """Fetch yesterday + 7-day forecast and commit to PostgreSQL weather_daily."""
    log.info("Starting scheduled weather_refresh job...")
    try:
        n = await refresh(client=client)
        log.info("weather_refresh job finished — %d rows upserted.", n)
        return n
    except Exception as e:
        log.exception("weather_refresh job failed: %s", e)
        raise