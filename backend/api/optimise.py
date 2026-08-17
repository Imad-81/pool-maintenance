"""
Dosing optimisation endpoint.
"""

from __future__ import annotations

import logging
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from prisma import Prisma

from backend.store.client import get_db
from backend.store import repo
from backend.deps import get_prediction_service
from ml.inference.predictor import PredictionService

router = APIRouter(prefix="/api/optimise", tags=["optimise"])
log = logging.getLogger("backend.api.optimise")


@router.get("/{pool_id}")
async def get_optimise(
    pool_id: str,
    request: Request,
    client: Prisma = Depends(get_db),
):
    svc: PredictionService = get_prediction_service(request)
    row = await repo.get_master_row(pool_id, client=client)
    if row is None:
        raise HTTPException(404, f"Pool '{pool_id}' not found in database.")

    opt = svc.optimise(pool_id, pd.Series(row))
    if opt is None:
        raise HTTPException(500, f"Optimisation could not be computed for pool {pool_id}.")

    return {
        "pool_id": pool_id,
        "pool_volume_m3": opt.pool_volume_m3,
        "current_readings": opt.current_readings,
        "recommended_dosing": opt.recommended_dosing,
        "predicted_tomorrow": opt.predicted_tomorrow,
        "feasible_configurations": opt.feasible_configurations,
        "top_3_configs": opt.top_3_configs,
        "urgency": opt.urgency,
        "reasons": opt.reasons,
    }