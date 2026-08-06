"""Pool detail endpoint."""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlmodel import Session

from backend.store.schema import get_session
from backend.store import repo
from backend.deps import get_prediction_service, get_weather_lookup
from ml.inference.predictor import PredictionService

router = APIRouter(prefix="/api/pool", tags=["pool"])
log = logging.getLogger(__name__)


@router.get("/{pool_id}")
def get_pool_detail(
    pool_id: str,
    request: Request,
    horizon: Optional[int] = Query(None, description="Forecast horizon days (default 2, max 7)"),
    session: Session = Depends(get_session),
):
    svc: PredictionService = request.app.state.prediction_service
    row = repo.get_master_row(session, pool_id)
    if row is None:
        raise HTTPException(404, f"Pool {pool_id} not found in database")

    wx_lookup = get_weather_lookup(request)
    as_of = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    import pandas as pd
    latest_series = pd.Series(row)

    # --- forecast ---
    try:
        forecast = svc.forecast(pool_id, latest_series, as_of, wx_lookup, horizon_days=horizon)
    except Exception as e:
        log.error("forecast error for %s: %s", pool_id, e)
        forecast = {"error": str(e)}

    # --- optimiser ---
    try:
        opt = svc.optimise(pool_id, latest_series)
    except Exception as e:
        log.error("optimiser error for %s: %s", pool_id, e)
        opt = None

    # --- history ---
    readings = repo.get_readings_for_pool(session, pool_id, limit=500)
    history = []
    for r in readings:
        history.append({
            "pool_id": r.pool_id,
            "reading_date": r.reading_date.isoformat() if r.reading_date else None,
            "ph": r.ph,
            "free_chlorine": r.free_chlorine,
            "turbidity": r.turbidity,
            "water_temperature": r.water_temperature,
        })

    # --- forecast serialisation ---
    fc_serialised = []
    if "forecast" in forecast and hasattr(forecast["forecast"], "to_dict"):
        for _, frow in forecast["forecast"].iterrows():
            serialised = {
                "date": str(frow["date"]),
                "day": frow.get("day", ""),
                "days_from_visit": int(frow.get("days_from_visit", 0)),
                "day_offset_from_today": int(frow.get("day_offset_from_today", 0)),
                "predicted_cl": float(frow["predicted_cl"]),
                "predicted_ph": float(frow["predicted_ph"]),
                "predicted_turb": float(frow["predicted_turb"]),
                "cl_breach": bool(frow.get("cl_breach", False)),
                "ph_breach": bool(frow.get("ph_breach", False)),
                "urgency": frow.get("urgency", ""),
                "status": frow.get("status", ""),
                "is_today": bool(frow.get("is_today", False)),
                "is_tomorrow": bool(frow.get("is_tomorrow", False)),
            }
            band = frow.get("uncertainty_band")
            if band is not None:
                serialised["uncertainty_band"] = {
                    "cl_low": float(band.cl_low), "cl_high": float(band.cl_high),
                    "ph_low": float(band.ph_low), "ph_high": float(band.ph_high),
                    "turb_low": float(band.turb_low), "turb_high": float(band.turb_high),
                }
            fc_serialised.append(serialised)

    # --- response ---
    result = {
        "pool_id": pool_id,
        "community_name": row.get("community_name", ""),
        "latest": {
            "reading_date": str(row.get("reading_date", "")),
            "ph": row.get("ph"),
            "free_chlorine": row.get("free_chlorine"),
            "turbidity": row.get("turbidity"),
        },
        "forecast": fc_serialised,
        "visit_needed": forecast.get("visit_needed", False),
        "today_forecast": forecast.get("today_forecast"),
        "tomorrow_forecast": forecast.get("tomorrow_forecast"),
        "prediction": {
            "source": "model" if "error" not in forecast else "error",
            "error": forecast.get("error"),
        },
        "history": history,
        "pool_volume_m3": row.get("pool_volume_m3"),
    }
    if opt is not None:
        result["optimiser"] = {
            "recommended_dosing": opt.recommended_dosing,
            "predicted_tomorrow": opt.predicted_tomorrow,
            "feasible_configurations": opt.feasible_configurations,
            "top_3_configs": opt.top_3_configs,
            "urgency": opt.urgency,
            "reasons": opt.reasons,
        }
    return result