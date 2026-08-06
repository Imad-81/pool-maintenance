"""
Live weather provider wrapping Open-Meteo via `fetch_weather.py`.

The daily weather_refresh job calls `refresh(session)` which fetches the
archive (yesterday) + 7-day forecast and upserts into `weather_daily`.  The
API then uses `make_lookup(session)` to create a callable compatible with
`predict_forward`'s `weather_lookup` parameter.

If Open-Meteo is unreachable the lookup simply returns NaN for that date,
and the predictor's fill_values handle it gracefully — no crash.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from backend.store import repo

log = logging.getLogger(__name__)


def refresh(session) -> int:
    """
    Fetch yesterday's archive + today through +7 days forecast from Open-Meteo.
    Upsert into weather_daily. Returns count of new/updated rows.
    """
    from fetch_weather import fetch_daily_weather
    from ml.config import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    end_forecast = (date.today() + timedelta(days=8)).strftime("%Y-%m-%d")
    log.info("weather_refresh: %s → %s", yesterday, end_forecast)

    rows, _units = fetch_daily_weather(
        latitude=cfg.alicante_lat,
        longitude=cfg.alicante_lon,
        start_date=yesterday,
        end_date=end_forecast,
        timezone=cfg.alicante_tz,
    )
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
        if "weather_code" in r:
            rec["w_weather_code"] = int(r["weather_code"]) if r["weather_code"] is not None else None
        else:
            rec["w_weather_code"] = None
        records.append(rec)
    if not records:
        log.warning("weather_refresh returned empty — API down?")
        return 0
    n = repo.upsert_weather_batch(session, records)
    session.commit()
    log.info("weather_refresh upserted %d days", n)
    return n


def make_lookup(session):
    """
    Return a `WeatherLookup` callable for use with `predict_forward`.

    The callable accepts (pd.Timestamp, list[str col_names]) and returns
    {col: value}. Missing dates → NaN.
    """
    # warm a local lookup dict
    from backend.store.schema import WeatherDaily
    wx_cache: dict[pd.Timestamp, dict] = {}

    def _warm():
        all_rows = session.query(WeatherDaily).all()
        for row in all_rows:
            d = pd.Timestamp(row.date)
            wx_cache[d] = {
                "w_temp_max":         row.w_temp_max,
                "w_temp_mean":        row.w_temp_mean,
                "w_uv_max":           row.w_uv_max,
                "w_uv_clear_sky_max": row.w_uv_clear_sky_max,
                "w_solar_radiation":  row.w_solar_radiation,
                "w_sunshine_hours":   row.w_sunshine_hours,
                "w_precipitation_mm":  row.w_precipitation_mm,
                "w_wind_max_kmh":     row.w_wind_max_kmh,
                "w_et0":              row.w_et0,
            }
    _warm()

    def lookup(date, cols):
        d = pd.Timestamp(date).normalize()
        day_data = wx_cache.get(d, {})
        return {c: (float(day_data[c]) if c in day_data and day_data[c] is not None else np.nan)
                for c in cols}

    return lookup