"""
APScheduler integration — Async background scheduler for weather refresh and periodic retraining.
"""

from __future__ import annotations

import logging
# pyrefly: ignore [missing-import]
from apscheduler.schedulers.asyncio import AsyncIOScheduler
# pyrefly: ignore [missing-import]
from apscheduler.triggers.cron import CronTrigger

from backend.store.client import db
from backend.jobs.weather_refresh import run_weather_refresh
from backend.jobs.retrain import should_retrain, run_retrain

log = logging.getLogger("backend.jobs.scheduler")

_scheduler: AsyncIOScheduler | None = None


def start_scheduler(settings) -> AsyncIOScheduler:
    """Create and start AsyncIOScheduler with weather refresh and retrain jobs."""
    global _scheduler
    sched = AsyncIOScheduler()

    @sched.scheduled_job(
        CronTrigger.from_crontab(settings.weather_refresh_cron),
        id="weather_refresh",
        name="Daily Open-Meteo Weather Refresh",
    )
    async def _weather_job():
        try:
            await run_weather_refresh(client=db)
        except Exception:
            log.exception("Error in scheduled weather_refresh job")

    @sched.scheduled_job(
        CronTrigger.from_crontab(settings.retrain_cron),
        id="retrain",
        name="Periodic ML Retraining & Promotion",
    )
    async def _retrain_job():
        try:
            if await should_retrain(settings, client=db):
                await run_retrain(settings, client=db)
        except Exception:
            log.exception("Error in scheduled retrain job")

    sched.start()
    _scheduler = sched
    log.info(
        "Async scheduler started: weather_cron='%s', retrain_cron='%s'",
        settings.weather_refresh_cron,
        settings.retrain_cron,
    )
    return sched


def shutdown_scheduler() -> None:
    """Gracefully shutdown scheduler."""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        log.info("Scheduler shutdown.")