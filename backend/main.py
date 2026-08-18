"""
FastAPI production application entry point.

Startup (lifespan):
    1. Connect to PostgreSQL via Prisma Client
    2. Warm weather cache
    3. Load active trained model run
    4. Start APScheduler (weather refresh + retrain) if enabled
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

# pyrefly: ignore [missing-import]
from fastapi import FastAPI, Request, Response
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.responses import JSONResponse

from backend.settings import settings
from backend.store.client import connect_db, disconnect_db, db
from backend.weather.provider import warm_weather_cache
from backend.jobs.scheduler import start_scheduler, shutdown_scheduler
from ml.inference.predictor import PredictionService

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("backend")



# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    log.info("Starting Pool Predictive Maintenance API (v6.0) — db=%s", settings.database_url)
    try:
        await connect_db()
        log.info("PostgreSQL database connected via Prisma.")
        await warm_weather_cache(client=db)
    except Exception as e:
        log.error("Database connection initialization warning: %s", e)

    svc = PredictionService(settings.models_dir_path)
    try:
        svc.load()
        log.info("Prediction service loaded: %s", svc.status())
    except Exception as e:
        log.warning("Model load FAILED: %s — starting in degraded mode", e)
    app.state.prediction_service = svc

    # Pre-compute today's daily predictions if missing
    try:
        from backend.store import repo
        from backend.weather.provider import make_lookup
        as_of = datetime.now(timezone.utc).replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0)
        cnt = await repo.count_daily_predictions(as_of, client=db)
        if cnt == 0 and svc.is_loaded():
            wx_lookup = make_lookup()
            log.info("Pre-computing daily predictions for today (%s)...", as_of.date())
            await repo.compute_and_store_daily_predictions(as_of, svc, wx_lookup, client=db)
            log.info("Daily predictions pre-computed successfully.")
    except Exception as e:
        log.warning("Startup daily predictions pre-warm notice: %s", e)

    if settings.enable_scheduler:
        start_scheduler(settings)
        log.info("Background scheduler started.")


    yield

    # --- Shutdown ---
    if settings.enable_scheduler:
        shutdown_scheduler()
    await disconnect_db()
    log.info("Backend shutdown complete.")


# ---------------------------------------------------------------------------
# App Definition & Middleware
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Spain Pool Predictive Maintenance API",
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_tracing_middleware(request: Request, call_next) -> Response:
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    start_time = time.time()

    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

    if request.url.path not in ("/healthz", "/healthz/live"):
        log.info(
            "%s %s -> %d (%.2fms) [req_id=%s]",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    req_id = getattr(request.state, "request_id", "unknown")
    log.exception("Unhandled exception on %s [req_id=%s]: %s", request.url.path, req_id, exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An unexpected error occurred.",
            "request_id": req_id,
        },
    )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
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


# --- Legacy status helper ---
@app.get("/api/status", tags=["health"])
def api_status():
    svc = getattr(app.state, "prediction_service", None)
    return {"status": "ok", "prediction": svc.status() if svc else {"loaded": False}}


if __name__ == "__main__":
    # pyrefly: ignore [missing-import]
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.host, port=settings.port, reload=settings.debug)