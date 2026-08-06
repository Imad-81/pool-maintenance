"""APScheduler integration — start background jobs on application startup."""

from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

log = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def start_scheduler(settings, session_factory) -> BackgroundScheduler:
    """Create a BackgroundScheduler with the two recurring jobs and start it.

    `session_factory` is a callable that returns a new SQLModel Session
    (e.g. `lambda: next(get_session())`).
    """
    global _scheduler
    sched = BackgroundScheduler(daemon=True)

    @sched.scheduled_job(CronTrigger.from_crontab(settings.weather_refresh_cron),
                         id="weather_refresh")
    def _weather_job():
        session = session_factory()
        try:
            from backend.jobs.weather_refresh import run_weather_refresh
            run_weather_refresh(session)
        except Exception:
            log.exception("weather_refresh job error")
        finally:
            session.close()

    @sched.scheduled_job(CronTrigger.from_crontab(settings.retrain_cron),
                         id="retrain")
    def _retrain_job():
        from backend.jobs.retrain import should_retrain, run_retrain
        session = session_factory()
        try:
            if should_retrain(settings, session):
                run_retrain(settings)
        except Exception:
            log.exception("retrain job error")
        finally:
            session.close()

    sched.start()
    _scheduler = sched
    log.info("scheduler started: weather=%s retrain=%s",
             settings.weather_refresh_cron, settings.retrain_cron)
    return sched


def shutdown_scheduler():
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        log.info("scheduler shutdown")