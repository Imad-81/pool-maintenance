"""
Repository layer — async Prisma CRUD helpers consumed by API routers and jobs.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from prisma import Prisma
# pyrefly: ignore [missing-import]
from prisma.models import Pool, Reading, WeatherDaily, ModelRun, IngestLog, DailyPrediction

from backend.store.client import db
from ml.config import (
    REG_CHLORINE_MIN,
    REG_CHLORINE_CLOSE,
    REG_PH_MIN,
    REG_PH_MAX,
    DEFAULT_CONFIG,
)
from ml.features import add_setpoint_features

log = logging.getLogger("backend.store.repo")

URGENCY_ORDER_MAP: dict[str, int] = {
    "Immediate": 0,
    "URGENT": 1,
    "Advised": 2,
    "Soon": 3,
    "Monitor": 4,
    "Routine": 5,
    "Extended": 6,
}


# ---------------------------------------------------------------------------
# Pool metadata
# ---------------------------------------------------------------------------

async def upsert_pool(row: dict, client: Prisma = db) -> Pool:
    """Upsert pool static metadata."""
    pool_id = str(row["pool_id"]).strip()
    create_data = {k: v for k, v in row.items() if v is not None}
    create_data["pool_id"] = pool_id
    update_data = {k: v for k, v in row.items() if k != "pool_id" and v is not None}
    
    return await client.pool.upsert(
        where={"pool_id": pool_id},
        data={"create": create_data, "update": update_data},  # type: ignore
    )


async def get_pool(pool_id: str, client: Prisma = db) -> Optional[Pool]:
    """Retrieve pool by ID."""
    return await client.pool.find_unique(where={"pool_id": pool_id})


async def get_all_pool_ids(client: Prisma = db) -> list[str]:
    """Return all unique pool IDs."""
    pools = await client.pool.find_many(order={"pool_id": "asc"})
    return [p.pool_id for p in pools if p.pool_id]


# ---------------------------------------------------------------------------
# Readings
# ---------------------------------------------------------------------------

async def upsert_readings_batch(
    rows: list[dict], source: str = "ingest", client: Prisma = db
) -> int:
    """Insert or update a batch of readings (upsert on pool_id + reading_date)."""
    count = 0
    # Process in chunks of 50 inside transaction for optimal throughput & safety
    chunk_size = 50
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i : i + chunk_size]
        async with client.tx() as tx:
            for r in chunk:
                pid = str(r["pool_id"]).strip()
                rd = r["reading_date"]
                if isinstance(rd, str):
                    rd = pd.to_datetime(rd).to_pydatetime()
                elif isinstance(rd, pd.Timestamp):
                    rd = rd.to_pydatetime()

                create_data = {
                    "pool_id": pid,
                    "reading_date": rd,
                    "source": source,
                }
                update_data = {"source": source}

                for field in [
                    "technician",
                    "community_name",
                    "ph",
                    "free_chlorine",
                    "turbidity",
                    "sunscreen_abuse",
                    "hypochlorite_dosing_pct",
                    "hypochlorite_dosing_hours",
                    "ph_dosing_pct",
                    "ph_dosing_hours",
                    "daily_filtration_hours",
                    "water_temperature",
                ]:
                    if field in r and r[field] is not None:
                        val = r[field]
                        if isinstance(val, (int, float)) and np.isnan(val):
                            val = None
                        create_data[field] = val
                        update_data[field] = val

                await tx.reading.upsert(
                    where={
                        "pool_id_reading_date": {
                            "pool_id": pid,
                            "reading_date": rd,
                        }
                    },
                    data={"create": create_data, "update": update_data},  # type: ignore
                )
                count += 1
    return count


async def get_readings_for_pool(
    pool_id: str, limit: int = 500, client: Prisma = db
) -> list[Reading]:
    """Return chronological history of readings for a pool."""
    readings = await client.reading.find_many(
        where={"pool_id": pool_id},
        order={"reading_date": "desc"},
        take=limit,
    )
    return readings[::-1]  # Return chronological


async def get_pool_latest_reading(
    pool_id: str, client: Prisma = db
) -> Optional[Reading]:
    """Return the most recent reading for a pool."""
    return await client.reading.find_first(
        where={"pool_id": pool_id},
        order={"reading_date": "desc"},
    )


async def get_active_pool_ids(
    as_of: datetime, days_back: int = 30, client: Prisma = db
) -> list[str]:
    """Return pool IDs having readings in the last N days (falls back to all pools)."""
    cutoff = as_of - timedelta(days=days_back)
    readings = await client.reading.find_many(
        where={"reading_date": {"gte": cutoff}},
        distinct=["pool_id"],
    )
    pids = [r.pool_id for r in readings if r.pool_id]
    if not pids:
        return await get_all_pool_ids(client=client)
    return pids


async def count_readings_by_pool(client: Prisma = db) -> dict[str, int]:
    """Count readings per pool."""
    pools = await client.pool.find_many(include={"readings": True})
    return {p.pool_id: len(p.readings or []) for p in pools}


async def count_readings_by_date(client: Prisma = db) -> list[tuple[datetime, int]]:
    """Return reading counts grouped chronologically."""
    # Using Prisma raw query for fast date-grouping in PostgreSQL
    results = await client.query_raw(
        "SELECT DATE(reading_date) as day, COUNT(id)::int as count FROM readings GROUP BY DATE(reading_date) ORDER BY day ASC;"
    )
    return [(r["day"], r["count"]) for r in results]


# ---------------------------------------------------------------------------
# Master row (for ML Predictor)
# ---------------------------------------------------------------------------

async def get_master_row(pool_id: str, client: Prisma = db) -> Optional[dict]:
    """Combine static pool attributes + latest reading + post-treatment setpoints."""
    pool = await client.pool.find_unique(where={"pool_id": pool_id})
    reading = await get_pool_latest_reading(pool_id, client)
    if reading is None:
        return None

    row = {}
    if pool is not None:
        row.update(pool.model_dump(exclude={"community_name", "readings"}))
    row.update(reading.model_dump(exclude={"id"}))

    # Inject post-treatment setpoint features
    df = pd.DataFrame([row])
    df = add_setpoint_features(
        df,
        setpoint_cl=DEFAULT_CONFIG.setpoint_free_chlorine,
        setpoint_ph=DEFAULT_CONFIG.setpoint_ph,
        setpoint_turb=DEFAULT_CONFIG.setpoint_turbidity,
    )
    return df.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

async def upsert_weather_batch(rows: list[dict], client: Prisma = db) -> int:
    """Upsert daily weather records."""
    count = 0
    chunk_size = 50
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i : i + chunk_size]
        async with client.tx() as tx:
            for r in chunk:
                d = r["date"]
                if isinstance(d, str):
                    d = pd.to_datetime(d).to_pydatetime()
                elif isinstance(d, pd.Timestamp):
                    d = d.to_pydatetime()

                create_data = {"date": d}
                update_data = {}
                for field in [
                    "w_temp_max",
                    "w_temp_mean",
                    "w_uv_max",
                    "w_uv_clear_sky_max",
                    "w_solar_radiation",
                    "w_sunshine_hours",
                    "w_precipitation_mm",
                    "w_wind_max_kmh",
                    "w_et0",
                    "w_weather_code",
                ]:
                    if field in r and r[field] is not None:
                        val = r[field]
                        if isinstance(val, (int, float)) and np.isnan(val):
                            val = None
                        create_data[field] = val
                        update_data[field] = val

                await tx.weatherdaily.upsert(
                    where={"date": d},
                    data={"create": create_data, "update": update_data},  # type: ignore
                )
                count += 1
    return count


async def get_weather_row(date: datetime, client: Prisma = db) -> Optional[WeatherDaily]:
    d = pd.Timestamp(date).normalize().to_pydatetime()
    return await client.weatherdaily.find_unique(where={"date": d})


async def get_weather_range(
    start: datetime, end: datetime, client: Prisma = db
) -> list[WeatherDaily]:
    start_d = pd.Timestamp(start).normalize().to_pydatetime()
    end_d = pd.Timestamp(end).normalize().to_pydatetime()
    return await client.weatherdaily.find_many(
        where={"date": {"gte": start_d, "lte": end_d}},
        order={"date": "asc"},
    )


async def get_latest_weather_date(client: Prisma = db) -> Optional[datetime]:
    latest = await client.weatherdaily.find_first(order={"date": "desc"})
    return latest.date if latest else None


# ---------------------------------------------------------------------------
# Model runs registry
# ---------------------------------------------------------------------------

async def add_model_run(
    run_id: str,
    artifact_dir: str,
    metrics_json: Optional[str] = None,
    feature_schema_json: Optional[str] = None,
    is_active: int = 0,
    client: Prisma = db,
) -> ModelRun:
    return await client.modelrun.create(
        data={
            "run_id": run_id,
            "artifact_dir": artifact_dir,
            "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
            "is_active": is_active,
            "metrics_json": metrics_json,
            "feature_schema_json": feature_schema_json,
        }
    )


async def get_active_model_run(client: Prisma = db) -> Optional[ModelRun]:
    return await client.modelrun.find_first(
        where={"is_active": 1},
        order={"promoted_at": "desc"},
    )


async def set_active_model_run(run_id: str, reason: str, client: Prisma = db) -> None:
    """Atomically demote all current active runs and promote target run_id."""
    async with client.tx() as tx:
        await tx.modelrun.update_many(
            where={"is_active": 1},
            data={"is_active": 0},
        )
        await tx.modelrun.update(
            where={"run_id": run_id},
            data={
                "is_active": 1,
                "promoted_at": datetime.now(timezone.utc).replace(tzinfo=None),
                "promote_reason": reason,
            },
        )


async def list_model_runs(limit: int = 20, client: Prisma = db) -> list[ModelRun]:
    return await client.modelrun.find_many(
        order={"created_at": "desc"},
        take=limit,
    )


# ---------------------------------------------------------------------------
# Ingest audit
# ---------------------------------------------------------------------------

async def add_ingest_log(
    source: str,
    filename: Optional[str] = None,
    pool_count: int = 0,
    row_count: int = 0,
    skipped_count: int = 0,
    detail_json: Optional[str] = None,
    client: Prisma = db,
) -> IngestLog:
    return await client.ingestlog.create(
        data={
            "source": source,
            "filename": filename,
            "pool_count": pool_count,
            "row_count": row_count,
            "skipped_count": skipped_count,
            "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
            "detail_json": detail_json,
        }
    )


async def list_ingest_logs(limit: int = 50, client: Prisma = db) -> list[IngestLog]:
    return await client.ingestlog.find_many(
        order={"created_at": "desc"},
        take=limit,
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

async def compute_fleet_stats(client: Prisma = db) -> dict:
    total_pools = await client.pool.count()
    total_readings = await client.reading.count()
    return {"total_pools": total_pools, "total_readings": total_readings}


def classify_urgency(cl: Optional[float], ph: Optional[float]) -> str:
    if cl is not None and (cl < REG_CHLORINE_MIN or cl > REG_CHLORINE_CLOSE):
        return "Immediate"
    if ph is not None and (ph < REG_PH_MIN or ph > REG_PH_MAX):
        return "Immediate"
    return "Routine"


# ---------------------------------------------------------------------------
# Daily Predictions Pre-computation & Fast Serving
# ---------------------------------------------------------------------------

async def get_master_rows_bulk(
    as_of: datetime, client: Prisma = db, pool_ids: Optional[list[str]] = None
) -> list[dict]:
    """
    Retrieve master rows for multiple or all pools up to `as_of` in a single query.
    Falls back gracefully if Prisma raw querying is unavailable in testing.
    """
    try:
        # 1. Fetch all matching pool metadata
        pool_filter = {"pool_id": {"in": pool_ids}} if pool_ids else {}
        pools = await client.pool.find_many(where=pool_filter)
        if not pools:
            return []

        # 2. Bulk query latest reading per pool on or before as_of date
        as_of_ts = pd.Timestamp(as_of).to_pydatetime()
        target_pids = [p.pool_id for p in pools]

        # Use Prisma raw query in PostgreSQL if supported
        raw_rows = await client.query_raw(
            """
            SELECT DISTINCT ON (r.pool_id)
                p.pool_id, p.community_name, p.pool_type, p.deck_type,
                p.pool_volume_m3, p.pool_surface_m2, p.filter_diameter,
                p.filter_count, p.motor_count, p.pool_heated, p.pool_community,
                p.pool_outdoor, p.pool_private, p.pool_public, p.pool_skimmer,
                p.pool_overflow, p.pool_oval, p.pool_round, p.pool_rectangular_0714,
                p.pool_rectangular_07, p.vegetation_contamination,
                p.deck_grass, p.deck_mixed, p.deck_paved,
                r.reading_date, r.technician, r.ph, r.free_chlorine, r.turbidity,
                r.hypochlorite_dosing_pct, r.hypochlorite_dosing_hours,
                r.ph_dosing_pct, r.ph_dosing_hours, r.daily_filtration_hours,
                r.water_temperature
            FROM pools p
            INNER JOIN readings r ON p.pool_id = r.pool_id
            WHERE r.reading_date <= $1
            ORDER BY r.pool_id, r.reading_date DESC;
            """,
            as_of_ts,
        )

        if raw_rows:
            df = pd.DataFrame(raw_rows)
            df = add_setpoint_features(
                df,
                setpoint_cl=DEFAULT_CONFIG.setpoint_free_chlorine,
                setpoint_ph=DEFAULT_CONFIG.setpoint_ph,
                setpoint_turb=DEFAULT_CONFIG.setpoint_turbidity,
            )
            return df.to_dict(orient="records")
    except Exception as e:
        log.debug("Bulk query using raw SQL fallback to standard Prisma ORM: %s", e)

    # Fallback to ORM fetching
    all_pids = pool_ids or await get_all_pool_ids(client=client)
    rows = []
    for pid in all_pids:
        row = await get_master_row(pid, client=client)
        if row:
            rows.append(row)
    return rows


import dataclasses


def _sanitize_forecast_dict(d: dict) -> dict:
    clean = {}
    for k, v in d.items():
        if isinstance(v, (datetime, pd.Timestamp, date)):
            clean[k] = str(v)
        elif dataclasses.is_dataclass(v):
            clean[k] = dataclasses.asdict(v)
        elif hasattr(v, "_asdict"):  # NamedTuple such as UncertaintyBand
            clean[k] = {sk: float(sv) for sk, sv in v._asdict().items()}
        elif isinstance(v, (np.floating, float)):
            clean[k] = None if np.isnan(v) else float(v)
        elif isinstance(v, (np.integer, int)):
            clean[k] = int(v)
        elif isinstance(v, (np.bool_, bool)):
            clean[k] = bool(v)
        else:
            clean[k] = v
    return clean



async def compute_and_store_daily_predictions(
    as_of: datetime,
    svc,
    wx_lookup,
    client: Prisma = db,
    pool_ids: Optional[list[str]] = None,
) -> int:
    """
    Generate ML forecasts for target pools as of `as_of` and bulk-upsert into daily_predictions table.
    """
    as_of_d = pd.Timestamp(as_of).normalize().to_pydatetime()
    master_rows = await get_master_rows_bulk(as_of, client=client, pool_ids=pool_ids)
    if not master_rows:
        return 0

    records = []
    for row in master_rows:
        pid = row["pool_id"]
        try:
            series = pd.Series(row)
            forecast = svc.forecast(pid, series, as_of_d, wx_lookup, horizon_days=2)
        except Exception as e:
            log.warning("Forecast skipped for %s on %s: %s", pid, as_of_d.date(), e)
            continue

        if "error" in forecast:
            continue

        df = forecast.get("forecast")
        if df is None or len(df) == 0:
            continue

        dashboard = df[df["is_today"] | df["is_tomorrow"]]
        if len(dashboard) == 0:
            dashboard = df.tail(1)

        item_urgency = (
            dashboard[dashboard["urgency"] != "Routine"].iloc[0]["urgency"]
            if (dashboard["urgency"] != "Routine").any()
            else (dashboard.iloc[-1]["urgency"] if len(dashboard) else "Routine")
        )
        urg_order = URGENCY_ORDER_MAP.get(item_urgency, 5)

        today_fc = forecast.get("today_forecast")
        tomorrow_fc = forecast.get("tomorrow_forecast")
        today_data = None
        if today_fc and len(today_fc) > 0:
            today_data = _sanitize_forecast_dict(today_fc[0])
        elif len(df) > 0:
            today_data = _sanitize_forecast_dict(df.iloc[-1].to_dict())

        tomorrow_data = None
        if tomorrow_fc and len(tomorrow_fc) > 0:
            tomorrow_data = _sanitize_forecast_dict(tomorrow_fc[0])

        rec_visit = forecast.get("recommended_visit")
        rec_visit_data = _sanitize_forecast_dict(rec_visit) if rec_visit else None

        last_rd = row.get("reading_date")
        if isinstance(last_rd, str):
            last_rd = pd.to_datetime(last_rd).to_pydatetime()
        elif isinstance(last_rd, pd.Timestamp):
            last_rd = last_rd.to_pydatetime()

        pred_cl_today = float(today_data.get("predicted_cl")) if today_data and today_data.get("predicted_cl") is not None else None
        pred_ph_today = float(today_data.get("predicted_ph")) if today_data and today_data.get("predicted_ph") is not None else None
        pred_turb_today = float(today_data.get("predicted_turb")) if today_data and today_data.get("predicted_turb") is not None else None

        pred_cl_tmrw = float(tomorrow_data.get("predicted_cl")) if tomorrow_data and tomorrow_data.get("predicted_cl") is not None else None
        pred_ph_tmrw = float(tomorrow_data.get("predicted_ph")) if tomorrow_data and tomorrow_data.get("predicted_ph") is not None else None
        pred_turb_tmrw = float(tomorrow_data.get("predicted_turb")) if tomorrow_data and tomorrow_data.get("predicted_turb") is not None else None

        breach_proba = float(any(dashboard["cl_breach"])) if len(dashboard) and "cl_breach" in dashboard else 0.0

        records.append({
            "pool_id": pid,
            "as_of_date": as_of_d,
            "last_reading_date": last_rd,
            "ph": row.get("ph"),
            "free_chlorine": row.get("free_chlorine"),
            "turbidity": row.get("turbidity"),
            "urgency": item_urgency,
            "urgency_order": urg_order,
            "breach_proba": breach_proba,
            "predicted_cl_today": pred_cl_today,
            "predicted_ph_today": pred_ph_today,
            "predicted_turb_today": pred_turb_today,
            "predicted_cl_tmrw": pred_cl_tmrw,
            "predicted_ph_tmrw": pred_ph_tmrw,
            "predicted_turb_tmrw": pred_turb_tmrw,
            "today_forecast_json": json.dumps(today_data) if today_data else None,
            "tomorrow_forecast_json": json.dumps(tomorrow_data) if tomorrow_data else None,
            "recommended_visit_json": json.dumps(rec_visit_data) if rec_visit_data else None,
            "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
        })

    if not records:
        return 0

    chunk_size = 50
    count = 0
    for i in range(0, len(records), chunk_size):
        chunk = records[i : i + chunk_size]
        async with client.tx() as tx:
            for rec in chunk:
                await tx.dailyprediction.upsert(
                    where={
                        "pool_id_as_of_date": {
                            "pool_id": rec["pool_id"],
                            "as_of_date": rec["as_of_date"],
                        }
                    },
                    data={"create": rec, "update": rec},  # type: ignore
                )
                count += 1

    log.info("Stored %d daily predictions for date %s", count, as_of_d.date())
    return count


async def count_daily_predictions(as_of: datetime, client: Prisma = db) -> int:
    """Return count of stored predictions for a given date."""
    as_of_d = pd.Timestamp(as_of).normalize().to_pydatetime()
    return await client.dailyprediction.count(where={"as_of_date": as_of_d})


async def get_daily_predictions_paged(
    as_of: datetime,
    q: Optional[str] = None,
    urgency: Optional[str] = None,
    page: int = 0,
    page_size: int = 50,
    client: Prisma = db,
) -> tuple[list[dict], int]:
    """
    Retrieve stored daily predictions for as_of date with SQL filtering, search, and pagination.
    """
    as_of_d = pd.Timestamp(as_of).normalize().to_pydatetime()
    where: dict = {"as_of_date": as_of_d}

    if urgency:
        where["urgency"] = urgency

    if q:
        q_clean = q.strip()
        where["OR"] = [
            {"pool_id": {"contains": q_clean, "mode": "insensitive"}},
            {"pool": {"community_name": {"contains": q_clean, "mode": "insensitive"}}},
        ]

    total = await client.dailyprediction.count(where=where)
    preds = await client.dailyprediction.find_many(
        where=where,
        include={"pool": True},
        order=[{"urgency_order": "asc"}, {"pool_id": "asc"}],
        take=page_size,
        skip=page * page_size,
    )

    items = []
    for p in preds:
        today_data = json.loads(p.today_forecast_json) if p.today_forecast_json else None
        tomorrow_data = json.loads(p.tomorrow_forecast_json) if p.tomorrow_forecast_json else None
        rec_visit = json.loads(p.recommended_visit_json) if getattr(p, "recommended_visit_json", None) else None
        items.append({
            "pool_id": p.pool_id,
            "community_name": p.pool.community_name if p.pool and p.pool.community_name else "",
            "last_reading_date": str(p.last_reading_date.date()) if p.last_reading_date else "",
            "ph": p.ph,
            "free_chlorine": p.free_chlorine,
            "turbidity": p.turbidity,
            "urgency": p.urgency,
            "breach_proba": p.breach_proba,
            "today_forecast": today_data,
            "tomorrow_forecast": tomorrow_data,
            "recommended_visit": rec_visit,
            "prediction_source": "model",
        })

    return items, total


async def get_daily_fleet_summary(
    as_of: datetime,
    client: Prisma = db,
) -> dict:
    """
    Fast SQL aggregation of fleet health, urgency counts, and compliance for a given date.
    """
    as_of_d = pd.Timestamp(as_of).normalize().to_pydatetime()
    preds = await client.dailyprediction.find_many(
        where={"as_of_date": as_of_d},
    )
    if not preds:
        return {
            "total": 0,
            "counts": {"Immediate": 0, "Advised": 0, "Routine": 0, "Extended": 0},
            "compliance_rate": 100,
            "as_of_date": str(as_of_d.date()),
        }

    counts = {"Immediate": 0, "Advised": 0, "Routine": 0, "Extended": 0}
    compliant = 0
    total = len(preds)
    for p in preds:
        if p.urgency in ("Immediate", "URGENT"):
            counts["Immediate"] += 1
        elif p.urgency in ("Advised", "Soon", "Monitor"):
            counts["Advised"] += 1
        elif p.urgency == "Extended":
            counts["Extended"] += 1
        else:
            counts["Routine"] += 1

        cl = p.predicted_cl_today if p.predicted_cl_today is not None else p.free_chlorine
        ph_val = p.predicted_ph_today if p.predicted_ph_today is not None else p.ph
        if cl is not None and 0.5 <= cl <= 2.0 and ph_val is not None and 7.2 <= ph_val <= 8.0:
            compliant += 1

    compliance_rate = round((compliant / total) * 100) if total > 0 else 100
    return {
        "total": total,
        "counts": counts,
        "compliance_rate": compliance_rate,
        "as_of_date": str(as_of_d.date()),
    }


async def get_pool_daily_prediction(
    pool_id: str, as_of: datetime, client: Prisma = db
) -> Optional[DailyPrediction]:
    """Retrieve stored prediction for single pool on target date."""
    as_of_d = pd.Timestamp(as_of).normalize().to_pydatetime()
    return await client.dailyprediction.find_unique(
        where={"pool_id_as_of_date": {"pool_id": pool_id, "as_of_date": as_of_d}}
    )