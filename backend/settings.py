"""
Backend settings loaded from environment variables with sensible defaults for production.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = dict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- project root ---
    project_root: Path = Path(__file__).resolve().parent.parent

    # --- database (PostgreSQL 16) ---
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql://pool:pool_secret@localhost:5432/pool_db"
    )

    # --- models ---
    models_dir: str = ""
    @property
    def models_dir_path(self) -> Path:
        if self.models_dir:
            return Path(self.models_dir)
        return self.project_root / "models"

    # --- weather (Alicante) ---
    weather_lat: float = 38.3452
    weather_lon: float = -0.4815
    weather_tz: str   = "Europe/Madrid"

    # --- scheduler ---
    enable_scheduler: bool    = True
    weather_refresh_cron: str = "0 4 * * *"       # daily at 4am
    retrain_cron: str         = "0 3 * * 1"       # weekly Monday 3am
    retrain_min_new_readings: int = 200            # skip retrain if < N new rows
    retrain_timeout_seconds: int  = 900            # 15 min for subprocess

    # --- security & auth ---
    ingestion_token: Optional[str] = os.getenv("INGESTION_TOKEN", None)
    admin_token: Optional[str] = os.getenv("ADMIN_TOKEN", None)
    cors_origins: str = os.getenv("CORS_ORIGINS", "*")

    @property
    def allowed_cors_origins(self) -> List[str]:
        if self.cors_origins == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # --- uploads & rate limits ---
    max_upload_size_mb: int = 15

    # --- server ---
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    log_level: str = "INFO"


settings = Settings()