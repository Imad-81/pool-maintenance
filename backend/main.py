"""
FastAPI application entry point.

Startup (lifespan):
    1. Create/verify SQLite schema
    2. Load the active trained model run from models/latest.json
    3. Start APScheduler (weather refresh + retrain)

Usage:
    python -m uvicorn backend.main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.settings import settings
from backend.store.schema import DATABASE_URL
from backend.store.schema import create_all, enable_wal, get_session
from backend.store import repo
from ml.inference.predictor import PredictionService
from backend.jobs.scheduler import start_scheduler, shutdown_scheduler

log = logging.getLogger("backend")

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    log.info("backend starting — db=%s  models=%s", DATABASE_URL, settings.models_dir_path)
    create_all()
    enable_wal()
    log.info("db ready")

    svc = PredictionService(settings.models_dir_path)
    try:
        svc.load()
    except Exception:
        log.warning("model load FAILED — starting in degraded mode")
    app.state.prediction_service = svc
    log.info("prediction service loaded: %s", svc.status())

    sched = start_scheduler(settings, get_session)
    log.info("scheduler started")

    yield

    # shutdown
    shutdown_scheduler()
    log.info("backend shutdown complete")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Pool Predictive Maintenance API",
    version="6.0",
    docs_url="/docs",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- routers ---
from backend.api.health import router as health_router
from backend.api.fleet import router as fleet_router
from backend.api.pool import router as pool_router
from backend.api.upload import router as upload_router, manual_router
from backend.api.optimise import router as optimise_router
from backend.api.admin import router as admin_router
from backend.api.ingest import router as ingest_router

app.include_router(health_router)
app.include_router(fleet_router)
app.include_router(pool_router)
app.include_router(upload_router)
app.include_router(manual_router)
app.include_router(optimise_router)
app.include_router(admin_router)
app.include_router(ingest_router)

# --- status ---
@app.get("/api/status")
def api_status():
    svc = app.state.prediction_service
    return {"status": "ok", "prediction": svc.status() if svc else {"loaded": False}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")