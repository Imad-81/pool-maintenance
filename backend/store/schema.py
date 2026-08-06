"""
SQLite schema for the pool predictive maintenance system.

Tables mirror the three sub-tables from pipeline_v6 (readings, operations,
products) plus a weather cache, an audit log for data ingestion, and a
model_runs registry used by the scheduler for retrain hot-swap.

Every table lives in a single SQLite file at the path given by the `DATABASE_URL`
environment variable (default: `<project_root>/data/store.db`).  WAL mode is
enabled on connection to allow concurrent reads during retrain writes.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# pyrefly: ignore [missing-import]
from sqlmodel import Field, SQLModel, Session, create_engine

# ---------------------------------------------------------------------------
# Default database path
# ---------------------------------------------------------------------------
_default_db_url = (
    f"sqlite:///{Path(__file__).resolve().parent.parent.parent / 'data' / 'store.db'}"
)
DATABASE_URL = os.environ.get("DATABASE_URL", _default_db_url)


# ---------------------------------------------------------------------------
# Pool metadata (static attributes; one row per pool)
# ---------------------------------------------------------------------------
class Pool(SQLModel, table=True):
    pool_id: str = Field(primary_key=True)
    community_name: Optional[str] = None
    pool_type: Optional[str] = None
    deck_type: Optional[str] = None
    pool_volume_m3: Optional[float] = None
    pool_surface_m2: Optional[float] = None
    filter_diameter: Optional[float] = None
    filter_count: Optional[float] = None
    motor_count: Optional[float] = None
    pool_heated: int = 0
    pool_community: int = 0
    pool_outdoor: int = 0
    pool_private: int = 0
    pool_public: int = 0
    pool_skimmer: int = 0
    pool_overflow: int = 0
    pool_oval: int = 0
    pool_round: int = 0
    pool_rectangular_0714: int = 0
    pool_rectangular_07: int = 0
    vegetation_contamination: int = 0
    deck_grass: float = 0.0
    deck_mixed: float = 0.0
    deck_paved: float = 0.0


# ---------------------------------------------------------------------------
# Readings (one row per pool-visit)
# ---------------------------------------------------------------------------
class Reading(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    pool_id: str = Field(index=True)
    reading_date: datetime
    technician: Optional[str] = None
    community_name: Optional[str] = None
    ph: Optional[float] = None
    free_chlorine: Optional[float] = None
    turbidity: Optional[float] = None
    sunscreen_abuse: Optional[str] = None
    # control
    hypochlorite_dosing_pct: Optional[float] = None
    hypochlorite_dosing_hours: Optional[float] = None
    ph_dosing_pct: Optional[float] = None
    ph_dosing_hours: Optional[float] = None
    daily_filtration_hours: Optional[float] = None
    water_temperature: Optional[float] = None
    # source tracking
    source: str = "upload"    # "upload" | "manual" | "ingest" | "master"
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Weather cache (daily, upserted by weather_refresh job)
# ---------------------------------------------------------------------------
class WeatherDaily(SQLModel, table=True):
    date: datetime = Field(primary_key=True)
    w_temp_max: Optional[float] = None
    w_temp_mean: Optional[float] = None
    w_uv_max: Optional[float] = None
    w_uv_clear_sky_max: Optional[float] = None
    w_solar_radiation: Optional[float] = None
    w_sunshine_hours: Optional[float] = None
    w_precipitation_mm: Optional[float] = None
    w_wind_max_kmh: Optional[float] = None
    w_et0: Optional[float] = None
    w_weather_code: Optional[int] = None
    fetched_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Model runs registry (one row per training run)
# ---------------------------------------------------------------------------
class ModelRun(SQLModel, table=True):
    run_id: str = Field(primary_key=True)
    artifact_dir: str
    created_at: datetime
    is_active: int = 0                 # 1 if the backend should serve this run
    metrics_json: Optional[str] = None  # JSON blob of {chlorine_next: {mae,...}, ...}
    feature_schema_json: Optional[str] = None
    promoted_at: Optional[datetime] = None
    promote_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Ingestion audit log
# ---------------------------------------------------------------------------
class IngestLog(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    source: str           # "upload" | "manual" | "ingest_api"
    filename: Optional[str] = None
    pool_count: int = 0
    row_count: int = 0
    skipped_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    detail_json: Optional[str] = None


# ---------------------------------------------------------------------------
# Engine & Helper (module-level singleton — FastAPI dependents get a session via
# deps.get_session)
# ---------------------------------------------------------------------------
def get_db_file_path(url: str) -> Optional[Path]:
    """Extract filesystem Path from a sqlite:// database URL."""
    if not url.startswith("sqlite:"):
        return None
    clean = url
    if clean.startswith("sqlite:////"):
        clean = "/" + clean[11:]
    elif clean.startswith("sqlite:///"):
        clean = clean[10:]
    elif clean.startswith("sqlite://"):
        clean = clean[9:]
    if not clean or clean == ":memory:":
        return None
    return Path(clean)


def ensure_db_dir_exists() -> None:
    """Ensure the directory containing the SQLite database exists."""
    db_path = get_db_file_path(DATABASE_URL)
    if db_path and not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)


ensure_db_dir_exists()
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)


def create_all() -> None:
    """Create all tables. Idempotent."""
    ensure_db_dir_exists()
    SQLModel.metadata.create_all(engine)


def enable_wal() -> None:
    """Enable WAL journal mode for SQLite (no-op on PostgreSQL)."""
    if not DATABASE_URL.startswith("sqlite"):
        return
    import sqlite3
    db_path = get_db_file_path(DATABASE_URL)
    if db_path is None:
        return
    ensure_db_dir_exists()
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")


def get_session():
    """Yield a new session — used as FastAPI dependency."""
    with Session(engine) as session:
        yield session