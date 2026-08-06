"""Dosing optimisation endpoint."""

import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlmodel import Session

from backend.store.schema import get_session
from backend.store import repo
from backend.deps import get_prediction_service
from ml.inference.predictor import PredictionService

router = APIRouter(prefix="/api/optimise", tags=["optimise"])
log = logging.getLogger(__name__)


@router.get("/{pool_id}")
def get_optimise(pool_id: str, request: Request, session: Session = Depends(get_session)):
    svc: PredictionService = request.app.state.prediction_service
    row = repo.get_master_row(session, pool_id)
    if row is None:
        raise HTTPException(404, f"Pool {pool_id} not found")
    import pandas as pd
    opt = svc.optimise(pool_id, pd.Series(row))
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