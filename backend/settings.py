"""
Backend settings loaded from environment variables with sensible defaults.

All paths are relative to the repository root, making the same settings
work in dev (`python -m backend.main`), Docker (env-file overrides) and tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = dict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- project root (inferred, not set via env) -------------------------
    project_root: Path = Path(__file__).resolve().parent.parent

    # --- database ---------------------------------------------------------
    database_url: str = ""

    @property
    def db_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{self.project_root / 'data' / 'store.db'}"

    # --- models -----------------------------------------------------------
    models_dir: str = ""
    @property
    def models_dir_path(self) -> Path:
        if self.models_dir:
            return Path(self.models_dir)
        return self.project_root / "models"

    # --- weather (Alicante) ------------------------------------------------
    weather_lat: float = 38.3452
    weather_lon: float = -0.4815
    weather_tz: str   = "Europe/Madrid"

    # --- scheduler ---------------------------------------------------------
    weather_refresh_cron: str = "0 4 * * *"       # daily at 4am
    retrain_cron: str         = "0 3 * * 1"       # weekly Monday 3am
    retrain_min_new_readings: int = 200            # skip retrain if < N new rows
    retrain_timeout_seconds: int  = 900            # 15 min for subprocess

    # --- ingestion ---------------------------------------------------------
    ingestion_token: Optional[str] = None           # no auth by default

    # --- server ------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

settings = Settings()