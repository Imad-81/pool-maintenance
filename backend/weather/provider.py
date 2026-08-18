"""
Live weather provider wrapping Open-Meteo via `fetch_weather.py` with Prisma and in-memory caching.

The daily weather_refresh job calls `await refresh()` which fetches the
archive (yesterday) + 7-day forecast and upserts into `weather_daily`.

`get_weather_cache()` loads the recent weather window into an in-memory cache
so `make_lookup()` returns a high-performance synchronous callable compatible
with `predict_forward`.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Optional

import numpy as np
import pandas as pd
from prisma import Prisma

from backend.store.client import db
from backend.store import repo

log = logging.getLogger("backend.weather")

# In-memory weather cache: {pd.Timestamp: {col: val}}
_weather_cache: dict[pd.Timestamp, dict] = {}
_cache_last_warmed: float = 0.0
CACHE_TTL_SECONDS: float = 600.0  # 10 minutes


async def warm_weather_cache(client: Prisma = db, force: bool = False) -> dict[pd.Timestamp, dict]:
    """Load weather records from database into memory cache."""
    global _weather_cache, _cache_last_warmed
    now = time.time()
    if not force and _weather_cache and (now - _cache_last_warmed < CACHE_TTL_SECONDS):
        return _weather_cache

    try:
        # Load weather range (e.g. past 90 days to future 14 days)
        cutoff_start = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=90)
        cutoff_end = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=14)
        rows = await client.weatherdaily.find_many(
            where={"date": {"gte": cutoff_start, "lte": cutoff_end}},
            order={"date": "asc"},
        )
        new_cache = {}
        for row in rows:
            d = pd.Timestamp(row.date)
            if d.tz is not None:
                d = d.tz_localize(None)
            d = d.normalize()
            new_cache[d] = {
                "w_temp_max": row.w_temp_max,
                "w_temp_mean": row.w_temp_mean,
                "w_uv_max": row.w_uv_max,
                "w_uv_clear_sky_max": row.w_uv_clear_sky_max,
                "w_solar_radiation": row.w_solar_radiation,
                "w_sunshine_hours": row.w_sunshine_hours,
                "w_precipitation_mm": row.w_precipitation_mm,
                "w_wind_max_kmh": row.w_wind_max_kmh,
                "w_et0": row.w_et0,
            }
        _weather_cache = new_cache
        _cache_last_warmed = now
        log.debug("Weather cache warmed with %d days.", len(_weather_cache))
    except Exception as e:
        log.warning("Failed to warm weather cache from DB: %s", e)
    return _weather_cache


async def refresh(client: Prisma = db) -> int:
    """
    Fetch yesterday's archive + today through +7 days forecast from Open-Meteo.
    Upsert into weather_daily. Returns count of new/updated rows.
    """
    from fetch_weather import fetch_daily_weather
    from ml.config import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    end_forecast = (date.today() + timedelta(days=8)).strftime("%Y-%m-%d")
    log.info("weather_refresh: fetching %s → %s from Open-Meteo", yesterday, end_forecast)

    try:
        rows, _units = fetch_daily_weather(
            latitude=cfg.alicante_lat,
            longitude=cfg.alicante_lon,
            start_date=yesterday,
            end_date=end_forecast,
            timezone=cfg.alicante_tz,
        )
    except Exception as e:
        log.error("Open-Meteo API fetch failed: %s", e)
        return 0

    WX_RENAME = {
        "temperature_2m_max":      "w_temp_max",
        "temperature_2m_mean":     "w_temp_mean",
        "uv_index_max":            "w_uv_max",
        "uv_index_clear_sky_max":  "w_uv_clear_sky_max",
        "shortwave_radiation_sum": "w_solar_radiation",
        "sunshine_duration":       "w_sunshine_hours",
        "precipitation_sum":       "w_precipitation_mm",
        "wind_speed_10m_max":      "w_wind_max_kmh",
        "et0_fao_evapotranspiration": "w_et0",
    }
    records = []
    for r in rows:
        rec = {}
        rec["date"] = pd.Timestamp(r["date"]).normalize().to_pydatetime()
        for src, dst in WX_RENAME.items():
            if src in r:
                val = r[src]
                rec[dst] = float(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else None
        rec["w_weather_code"] = int(r["weather_code"]) if "weather_code" in r and r["weather_code"] is not None else None
        records.append(rec)

    if not records:
        log.warning("weather_refresh returned empty records.")
        return 0

    n = await repo.upsert_weather_batch(records, client=client)
    await warm_weather_cache(client=client, force=True)
    log.info("weather_refresh upserted %d days and invalidated cache.", n)
    return n


def make_lookup(cache: Optional[dict[pd.Timestamp, dict]] = None) -> Callable[[pd.Timestamp, list[str]], dict]:
    """
    Return a synchronous `WeatherLookup` callable for use with `predict_forward`.
    """
    active_cache = cache if cache is not None else _weather_cache

    def lookup(date_val, cols):
        d = pd.Timestamp(date_val)
        if d.tz is not None:
            d = d.tz_localize(None)
        d = d.normalize()
        day_data = active_cache.get(d, {})
        return {
            c: (float(day_data[c]) if c in day_data and day_data[c] is not None else np.nan)
            for c in cols
        }

    return lookup