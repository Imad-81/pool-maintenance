"""
Fleet overview endpoints — async Prisma-backed multi-pool forecast aggregation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from prisma import Prisma

from backend.store.client import get_db
from backend.store import repo
from backend.deps import get_prediction_service, get_weather_lookup_provider
from ml.inference.predictor import PredictionService

router = APIRouter(prefix="/api/fleet", tags=["fleet"])
log = logging.getLogger("backend.api.fleet")


class FleetItemResponse(BaseModel):
    pool_id: str
    community_name: Optional[str] = ""
    last_reading_date: Optional[str] = ""
    ph: Optional[float] = None
    free_chlorine: Optional[float] = None
    turbidity: Optional[float] = None
    urgency: str
    breach_proba: float
    today_forecast: Optional[Dict[str, Any]] = None
    tomorrow_forecast: Optional[Dict[str, Any]] = None
    prediction_source: str = "model"


class FleetListResponse(BaseModel):
    items: List[FleetItemResponse]
    total: int
    page: int
    page_size: int


@router.get("", response_model=FleetListResponse)
async def get_fleet(
    request: Request,
    date: Optional[str] = Query(None, description="Query date in YYYY-MM-DD format"),
    q: Optional[str] = Query(None, description="Search term for pool ID or community"),
    urgency: Optional[str] = Query(None, description="Filter by urgency level"),
    page: int = Query(0, ge=0),
    page_size: int = Query(50, ge=1, le=200),
    client: Prisma = Depends(get_db),
):
    svc: PredictionService = get_prediction_service(request)
    wx_lookup = await get_weather_lookup_provider(request)
    as_of = datetime.now(timezone.utc).replace(tzinfo=None)

    if date:
        try:
            as_of = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(400, f"Invalid date format: {date}. Use YYYY-MM-DD.")
    else:
        as_of = as_of.replace(hour=0, minute=0, second=0, microsecond=0)

    pool_ids = await repo.get_active_pool_ids(as_of, client=client)
    # Sanitize pool IDs
    pool_ids = [p for p in pool_ids if p and p.strip()]

    results = []
    for pid in pool_ids:
        row = await repo.get_master_row(pid, client=client)
        if row is None:
            continue
        try:
            series = pd.Series(row)
            forecast = svc.forecast(pid, series, as_of, wx_lookup, horizon_days=2)
        except Exception as e:
            log.warning("Forecast skipped for %s: %s", pid, e)
            continue

        if "error" in forecast:
            continue

        df = forecast["forecast"]
        dashboard = df[df["is_today"] | df["is_tomorrow"]]
        if len(dashboard) == 0:
            continue

        item_urgency = (
            dashboard[dashboard["urgency"] != "Routine"].iloc[0]["urgency"]
            if (dashboard["urgency"] != "Routine").any()
            else "Routine"
        )

        results.append(
            FleetItemResponse(
                pool_id=pid,
                community_name=row.get("community_name") or "",
                last_reading_date=str(forecast.get("last_visit_date", "")),
                ph=row.get("ph"),
                free_chlorine=row.get("free_chlorine"),
                turbidity=row.get("turbidity"),
                urgency=item_urgency,
                breach_proba=float(any(dashboard["cl_breach"])),
                today_forecast=(
                    {k: str(v) if isinstance(v, datetime) else v for k, v in forecast["today_forecast"][0].items()}
                    if forecast.get("today_forecast")
                    else None
                ),
                tomorrow_forecast=(
                    {k: str(v) if isinstance(v, datetime) else v for k, v in forecast["tomorrow_forecast"][0].items()}
                    if forecast.get("tomorrow_forecast")
                    else None
                ),
                prediction_source="model",
            )
        )

    urgency_order = {
        "Immediate": 0,
        "URGENT": 1,
        "Advised": 2,
        "Soon": 3,
        "Monitor": 4,
        "Routine": 5,
        "Extended": 6,
    }
    results.sort(key=lambda x: urgency_order.get(x.urgency, 9))

    if q:
        q_low = q.lower()
        results = [
            r for r in results
            if q_low in (r.pool_id + (r.community_name or "")).lower()
        ]
    if urgency:
        results = [r for r in results if r.urgency.lower() == urgency.lower()]

    total = len(results)
    start = page * page_size
    paged = results[start : start + page_size]
    return FleetListResponse(items=paged, total=total, page=page, page_size=page_size)


@router.get("/pool-ids", response_model=List[str])
async def get_pool_ids(client: Prisma = Depends(get_db)):
    return await repo.get_all_pool_ids(client=client)


@router.get("/dates")
async def get_fleet_dates(client: Prisma = Depends(get_db)):
    dates = await repo.count_readings_by_date(client=client)
    if not dates:
        return {"min": "", "max": "", "count": 0}
    return {
        "min": str(dates[0][0]),
        "max": str(dates[-1][0]),
        "count": len(dates),
    }
