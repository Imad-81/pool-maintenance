"""
Repository layer — typed CRUD helpers consumed by the API routers and jobs.

All functions receive a `session` parameter (SQLModel `Session`) so the
caller controls the transaction boundary; this keeps the store decoupled from
the HTTP layer.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sqlmodel import Session, select, func

from backend.store.schema import (
    Pool,
    Reading,
    WeatherDaily,
    ModelRun,
    IngestLog,
)

from ml.config import (
    REG_CHLORINE_MIN,
    REG_CHLORINE_CLOSE,
    REG_PH_MIN,
    REG_PH_MAX,
)


# ---------------------------------------------------------------------------
# Pool metadata
# ---------------------------------------------------------------------------

def upsert_pool(session: Session, row: dict) -> None:
    existing = session.get(Pool, row["pool_id"])
    if existing:
        for k, v in row.items():
            setattr(existing, k, v)
    else:
        session.add(Pool(**row))
    session.flush()


def get_pool(session: Session, pool_id: str) -> Optional[Pool]:
    return session.get(Pool, pool_id)


def get_all_pool_ids(session: Session) -> list[str]:
    return [str(r) for r in session.exec(select(Pool.pool_id)).all() if r]


# ---------------------------------------------------------------------------
# Readings
# ---------------------------------------------------------------------------

def upsert_readings_batch(session: Session, rows: list[dict], source: str = "ingest") -> int:
    """Insert (or update) a batch of readings. Returns count processed.

    Upsert on (pool_id, reading_date) — if a reading for this pool-day exists,
    keep the later one (the import scripts de-duplicate before calling this).
    """
    count = 0
    for r in rows:
        pool_id = r["pool_id"]
        rd = r["reading_date"]
        if not isinstance(rd, datetime):
            rd = pd.to_datetime(rd).to_pydatetime()
        existing = session.exec(
            select(Reading).where(Reading.pool_id == pool_id, Reading.reading_date == rd)
        ).first()
        if existing:
            for k, v in r.items():
                if k != "id":
                    setattr(existing, k, v)
        else:
            session.add(Reading(source=source, **{k: v for k, v in r.items() if k != "source"}))
        count += 1
    session.flush()
    return count


def get_readings_for_pool(session: Session, pool_id: str, limit: int = 500) -> list[Reading]:
    return session.exec(
        select(Reading)
        .where(Reading.pool_id == pool_id)
        .order_by(Reading.reading_date.desc())
        .limit(limit)
    ).all()[::-1]  # return chronological


def get_pool_latest_reading(session: Session, pool_id: str) -> Optional[Reading]:
    return session.exec(
        select(Reading)
        .where(Reading.pool_id == pool_id)
        .order_by(Reading.reading_date.desc())
    ).first()


def get_active_pool_ids(session: Session, as_of: datetime, days_back: int = 30) -> list[str]:
    cutoff = as_of - timedelta(days=days_back)
    return [
        str(r) for r in session.exec(
            select(Reading.pool_id)
            .where(Reading.reading_date >= cutoff)
            .distinct()
        ).all()
        if r
    ]


def count_readings_by_pool(session: Session) -> dict[str, int]:
    rows = session.exec(
        select(Reading.pool_id, func.count(Reading.id).label("c"))
        .group_by(Reading.pool_id)
    ).all()
    return {r[0]: r[1] for r in rows}


def count_readings_by_date(session: Session) -> list[tuple[datetime, int]]:
    return [(r[0], r[1]) for r in session.exec(
        select(Reading.reading_date, func.count(Reading.id))
        .group_by(Reading.reading_date)
        .order_by(Reading.reading_date)
    ).all()]


# ---------------------------------------------------------------------------
# Master row — merge latest reading with static pool attributes for the
# predictor (which expects one row with all features present).
# ---------------------------------------------------------------------------

def get_master_row(session: Session, pool_id: str) -> Optional[dict]:
    """Return a dict combining the pool's static attrs with its latest
    reading, suitable as the `latest_row` arg to `predict_forward`."""
    pool = session.get(Pool, pool_id)
    reading = get_pool_latest_reading(session, pool_id)
    if reading is None:
        return None
    row = {}
    if pool is not None:
        row.update(pool.model_dump(exclude={"community_name"}))
    row.update(reading.model_dump(exclude={"id"}))
    return row


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

def upsert_weather_batch(session: Session, rows: list[dict]) -> int:
    count = 0
    for r in rows:
        d = r["date"]
        if not isinstance(d, datetime):
            d = pd.to_datetime(d).to_pydatetime()
        existing = session.get(WeatherDaily, d)
        if existing:
            for k, v in r.items():
                setattr(existing, k, v)
        else:
            session.add(WeatherDaily(**r))
        count += 1
    session.flush()
    return count


def get_weather_row(session: Session, date: datetime) -> Optional[WeatherDaily]:
    d = pd.Timestamp(date).normalize().to_pydatetime()
    return session.get(WeatherDaily, d)


def get_weather_range(session: Session, start: datetime, end: datetime) -> list[WeatherDaily]:
    return session.exec(
        select(WeatherDaily)
        .where(WeatherDaily.date >= start, WeatherDaily.date <= end)
        .order_by(WeatherDaily.date)
    ).all()


def get_latest_weather_date(session: Session) -> Optional[datetime]:
    r = session.exec(
        select(func.max(WeatherDaily.date))
    ).first()
    return r


# ---------------------------------------------------------------------------
# Model runs
# ---------------------------------------------------------------------------

def add_model_run(session: Session, run_id: str, artifact_dir: str,
                  metrics_json: Optional[str] = None,
                  feature_schema_json: Optional[str] = None,
                  is_active: int = 0) -> None:
    session.add(ModelRun(
        run_id=run_id,
        artifact_dir=artifact_dir,
        created_at=datetime.utcnow(),
        is_active=is_active,
        metrics_json=metrics_json,
        feature_schema_json=feature_schema_json,
    ))
    session.flush()


def get_active_model_run(session: Session) -> Optional[ModelRun]:
    return session.exec(
        select(ModelRun).where(ModelRun.is_active == 1).order_by(ModelRun.promoted_at.desc())
    ).first()


def set_active_model_run(session: Session, run_id: str, reason: str) -> None:
    """Demote all current active runs and promote `run_id`."""
    for r in session.exec(select(ModelRun).where(ModelRun.is_active == 1)).all():
        r.is_active = 0
    target = session.get(ModelRun, run_id)
    if target:
        target.is_active = 1
        target.promoted_at = datetime.utcnow()
        target.promote_reason = reason
    session.flush()


def list_model_runs(session: Session, limit: int = 20) -> list[ModelRun]:
    return session.exec(
        select(ModelRun).order_by(ModelRun.created_at.desc()).limit(limit)
    ).all()


# ---------------------------------------------------------------------------
# Ingest audit
# ---------------------------------------------------------------------------

def add_ingest_log(session: Session, source: str, filename: Optional[str] = None,
                   pool_count: int = 0, row_count: int = 0,
                   skipped_count: int = 0, detail_json: Optional[str] = None) -> None:
    session.add(IngestLog(
        source=source,
        filename=filename,
        pool_count=pool_count,
        row_count=row_count,
        skipped_count=skipped_count,
        detail_json=detail_json,
    ))
    session.flush()


def list_ingest_logs(session: Session, limit: int = 50) -> list[IngestLog]:
    return session.exec(
        select(IngestLog).order_by(IngestLog.created_at.desc()).limit(limit)
    ).all()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_fleet_stats(session: Session) -> dict:
    total = session.exec(select(func.count(Reading.pool_id.distinct()))).first() or 0
    total_rows = session.exec(select(func.count(Reading.id))).first() or 0
    return {"total_pools": int(total), "total_readings": int(total_rows)}


def classify_urgency(cl: Optional[float], ph: Optional[float]) -> str:
    if cl is not None and (cl < REG_CHLORINE_MIN or cl > REG_CHLORINE_CLOSE):
        return "Immediate"
    if ph is not None and (ph < REG_PH_MIN or ph > REG_PH_MAX):
        return "Immediate"
    return "Routine"  # full urgency comes from the predictor, not static lookup