"""
Repository layer — async Prisma CRUD helpers consumed by API routers and jobs.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from prisma import Prisma
# pyrefly: ignore [missing-import]
from prisma.models import Pool, Reading, WeatherDaily, ModelRun, IngestLog

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
    """Return pool IDs having readings in the last N days."""
    cutoff = as_of - timedelta(days=days_back)
    readings = await client.reading.find_many(
        where={"reading_date": {"gte": cutoff}},
        distinct=["pool_id"],
    )
    return [r.pool_id for r in readings if r.pool_id]


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