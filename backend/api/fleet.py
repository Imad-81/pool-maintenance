"""
Fleet overview endpoints — async Prisma-backed multi-pool forecast aggregation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
# pyrefly: ignore [missing-import]
from fastapi import APIRouter, Depends, HTTPException, Query, Request
# pyrefly: ignore [missing-import]
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
    recommended_visit: Optional[Dict[str, Any]] = None
    prediction_source: str = "model"


class FleetListResponse(BaseModel):
    items: List[FleetItemResponse]
    total: int
    page: int
    page_size: int


class FleetSummaryResponse(BaseModel):
    total: int
    counts: Dict[str, int]
    compliance_rate: int
    as_of_date: str


class RunInferenceResponse(BaseModel):
    success: bool
    predictions_generated: int
    as_of_date: str
    message: str


@router.post("/run-inference", response_model=RunInferenceResponse)
async def trigger_fleet_inference(
    request: Request,
    date: Optional[str] = Query(None, description="Query date in YYYY-MM-DD format"),
    client: Prisma = Depends(get_db),
):
    """Force re-run AI inference and update daily predictions table."""
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

    try:
        count = await repo.compute_and_store_daily_predictions(as_of, svc, wx_lookup, client=client)
        return RunInferenceResponse(
            success=True,
            predictions_generated=count,
            as_of_date=str(as_of.date()),
            message=f"Successfully re-calculated inference for {count} pools.",
        )
    except Exception as e:
        log.exception("Run inference failed: %s", e)
        raise HTTPException(500, f"Failed to execute fleet inference: {e}")



@router.get("/summary", response_model=FleetSummaryResponse)
async def get_fleet_summary(
    request: Request,
    date: Optional[str] = Query(None, description="Query date in YYYY-MM-DD format"),
    client: Prisma = Depends(get_db),
):
    """Return instant fleet aggregate KPIs (total pools, urgency counts, compliance rate)."""
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

    try:
        stored_count = await repo.count_daily_predictions(as_of, client=client)
    except Exception:
        stored_count = 0

    if stored_count == 0:
        await repo.compute_and_store_daily_predictions(as_of, svc, wx_lookup, client=client)

    try:
        summary = await repo.get_daily_fleet_summary(as_of, client=client)
        return FleetSummaryResponse(**summary)
    except Exception as e:
        log.warning("Daily fleet summary calculation fallback: %s", e)
        return FleetSummaryResponse(
            total=0,
            counts={"Immediate": 0, "Advised": 0, "Routine": 0, "Extended": 0},
            compliance_rate=100,
            as_of_date=str(as_of.date()),
        )


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
    """Retrieve paginated, precomputed fleet predictions with sub-20ms SQL performance."""
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

    try:
        stored_count = await repo.count_daily_predictions(as_of, client=client)
    except Exception:
        stored_count = 0

    # Auto-generate predictions if this date was not yet precomputed (cold start / historical date)
    if stored_count == 0:
        await repo.compute_and_store_daily_predictions(as_of, svc, wx_lookup, client=client)

    try:
        items, total = await repo.get_daily_predictions_paged(
            as_of=as_of,
            q=q,
            urgency=urgency,
            page=page,
            page_size=page_size,
            client=client,
        )
        if total > 0 or not q:
            return FleetListResponse(
                items=[FleetItemResponse(**item) for item in items],
                total=total,
                page=page,
                page_size=page_size,
            )
    except Exception as e:
        log.warning("Daily predictions paged read fallback: %s", e)

    # Fallback to direct computation if storage query fails
    pool_ids = await repo.get_active_pool_ids(as_of, client=client)
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

        df = forecast.get("forecast")
        if df is None or len(df) == 0:
            continue

        dashboard = df[df["is_today"] | df["is_tomorrow"]]
        if len(dashboard) == 0:
            dashboard = df.tail(1)

        item_urgency = (
            dashboard[dashboard["urgency"] != "Routine"].iloc[0]["urgency"]
            if (dashboard["urgency"] != "Routine").any()
            else (dashboard.iloc[-1]["urgency"] if len(dashboard) else "Routine")
        )

        today_fc = forecast.get("today_forecast")
        tomorrow_fc = forecast.get("tomorrow_forecast")
        today_data = None
        if today_fc and len(today_fc) > 0:
            today_data = {k: str(v) if isinstance(v, datetime) else v for k, v in today_fc[0].items()}
        elif len(df) > 0:
            today_data = {k: str(v) if isinstance(v, datetime) else v for k, v in df.iloc[-1].to_dict().items()}

        tomorrow_data = None
        if tomorrow_fc and len(tomorrow_fc) > 0:
            tomorrow_data = {k: str(v) if isinstance(v, datetime) else v for k, v in tomorrow_fc[0].items()}

        results.append(
            FleetItemResponse(
                pool_id=pid,
                community_name=row.get("community_name") or "",
                last_reading_date=str(forecast.get("last_visit_date", "")),
                ph=row.get("ph"),
                free_chlorine=row.get("free_chlorine"),
                turbidity=row.get("turbidity"),
                urgency=item_urgency,
                breach_proba=float(any(dashboard["cl_breach"])) if len(dashboard) and "cl_breach" in dashboard else 0.0,
                today_forecast=today_data,
                tomorrow_forecast=tomorrow_data,
                recommended_visit=forecast.get("recommended_visit"),
                prediction_source="model",
            )
        )

    results.sort(key=lambda x: repo.URGENCY_ORDER_MAP.get(x.urgency, 9))
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

