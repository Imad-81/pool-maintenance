"""Fleet overview endpoints."""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlmodel import Session

from backend.store.schema import get_session
from backend.store import repo
from backend.deps import get_prediction_service, get_weather_lookup
from ml.inference.predictor import PredictionService

router = APIRouter(prefix="/api/fleet", tags=["fleet"])
log = logging.getLogger(__name__)


@router.get("")
def get_fleet(
    request: Request,
    date: Optional[str] = Query(None),
    q: Optional[str] = Query(None),
    urgency: Optional[str] = Query(None),
    page: int = Query(0, ge=0),
    page_size: int = Query(50, ge=1, le=200),
    session: Session = Depends(get_session),
):
    """Return urgency-sorted fleet data with chained forecasts per pool."""
    svc: PredictionService = request.app.state.prediction_service
    wx_lookup = get_weather_lookup(request)
    as_of = datetime.utcnow()

    pool_ids = repo.get_active_pool_ids(session, as_of)
    # Filter out obviously corrupt pool_ids (empty, only whitespace, or missing
    # the expected " (NNN)" reference suffix pattern). These are artifacts from
    # CSV parsing of quoted names that overflow into the pool_id column.
    pool_ids = [p for p in pool_ids if p and p.strip() and "(" in p]
    if date:
        try:
            as_of = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(400, f"Invalid date: {date}. Use YYYY-MM-DD.")
    else:
        as_of = as_of.replace(hour=0, minute=0, second=0, microsecond=0)

    results = []
    for pid in pool_ids:
        row = repo.get_master_row(session, pid)
        if row is None:
            continue
        try:
            forecast = svc.forecast(pid, pd_row_to_series(row, svc), as_of, wx_lookup, horizon_days=2)
        except Exception as e:
            log.warning("forecast skip %s: %s", pid, e)
            continue
        if "error" in forecast:
            continue
        df = forecast["forecast"]
        dashboard = df[df["is_today"] | df["is_tomorrow"]]
        if len(dashboard) == 0:
            continue
        latest_row = dashboard.iloc[-1] if len(dashboard) else dashboard.iloc[0]
        results.append({
            "pool_id": pid,
            "community_name": row.get("community_name", ""),
            "last_reading_date": str(forecast.get("last_visit_date", "")),
            "ph": row.get("ph"),
            "free_chlorine": row.get("free_chlorine"),
            "turbidity": row.get("turbidity"),
            "urgency": dashboard[dashboard["urgency"] != "Routine"].iloc[0]["urgency"] if (dashboard["urgency"] != "Routine").any() else "Routine",
            "breach_proba": float(any(dashboard["cl_breach"])),
            "today_forecast": {k: str(v) if isinstance(v, (datetime,)) else v for k, v in forecast["today_forecast"][0].items()} if forecast["today_forecast"] else None,
            "tomorrow_forecast": {k: str(v) if isinstance(v, (datetime,)) else v for k, v in forecast["tomorrow_forecast"][0].items()} if forecast["tomorrow_forecast"] else None,
            "prediction_source": "model",
        })

    urgency_order = {"Immediate": 0, "URGENT": 1, "Advised": 2, "Soon": 3, "Monitor": 4, "Routine": 5, "Extended": 6}
    results.sort(key=lambda x: urgency_order.get(x["urgency"], 9))

    if q:
        results = [r for r in results if q.lower() in (r["pool_id"] + (r.get("community_name") or "")).lower()]
    if urgency:
        results = [r for r in results if r["urgency"] == urgency]

    total = len(results)
    start = page * page_size
    paged = results[start:start + page_size]
    return {"items": paged, "total": total, "page": page, "page_size": page_size}


@router.get("/pool-ids")
def get_pool_ids(session: Session = Depends(get_session)):
    return sorted(repo.get_all_pool_ids(session))


@router.get("/dates")
def get_fleet_dates(session: Session = Depends(get_session)):
    dates = repo.count_readings_by_date(session)
    if not dates:
        return {"min": "", "max": "", "count": 0}
    return {"min": str(dates[0][0]), "max": str(dates[-1][0]), "count": len(dates)}


def pd_row_to_series(row: dict, svc: PredictionService) -> "pd.Series":
    import pandas as pd
    return pd.Series(row)
