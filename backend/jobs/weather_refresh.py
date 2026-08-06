"""Weather refresh job — daily upsert into weather_daily."""

import logging
from backend.store import repo
from backend.weather.provider import refresh

log = logging.getLogger(__name__)


def run_weather_refresh(session) -> int:
    """Fetch yesterday + 7-day forecast and commit to weather_daily.
    Returns the number of new or updated rows."""
    log.info("weather_refresh job starting")
    try:
        n = refresh(session)
        log.info("weather_refresh job complete — %d rows", n)
        return n
    except Exception:
        log.exception("weather_refresh job FAILED")
        raise