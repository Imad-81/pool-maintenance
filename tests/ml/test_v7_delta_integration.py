"""
Integration tests for v7 LightGBM Delta models and chemical dosing optimizer.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ml.inference.predictor import PredictionService, _build_v7_delta_feature_vector
from ml.inference.optimiser import Optimiser


@pytest.fixture
def prediction_service():
    models_dir = Path(__file__).resolve().parents[2] / "models"
    svc = PredictionService(models_dir)
    svc.load()
    return svc


def test_v7_model_family_loaded(prediction_service):
    status = prediction_service.status()
    assert status["loaded"] is True
    assert status["model_family"] == "lightgbm_delta"
    assert status["run_id"] == "v7-daily-delta"


def test_v7_feature_vector_generation(prediction_service):
    state = prediction_service._state
    expected_cols = state["all_numeric_features"]
    assert len(expected_cols) == 121

    sample_row = pd.Series({
        "pool_id": "TEST_POOL",
        "free_chlorine": 1.2,
        "ph": 7.4,
        "turbidity": 0.25,
        "pool_volume_m3": 150.0,
        "reading_date": "2026-08-01",
    })

    df = _build_v7_delta_feature_vector(
        sample_row,
        step_date=pd.Timestamp("2026-08-02"),
        anchor_cl=1.2,
        anchor_ph=7.4,
        anchor_turb=0.25,
        anchor_cya=30.0,
        cl_history=[1.2],
        ph_history=[7.4],
        turb_history=[0.25],
        step=1,
        feature_names=expected_cols,
        fill_values=state["fill_values"],
    )

    assert df.shape == (1, 121)
    assert not df.isna().any().any()
    assert list(df.columns) == expected_cols


def test_v7_optimizer_recommends_dosing_for_deficient_pool(prediction_service):
    deficient_row = pd.Series({
        "pool_id": "DEFICIENT_POOL",
        "free_chlorine": 0.1,  # critically low
        "ph": 7.4,
        "turbidity": 0.3,
        "pool_volume_m3": 120.0,
        "reading_date": "2026-08-01",
    })

    res = prediction_service.optimise("DEFICIENT_POOL", deficient_row)
    assert res is not None
    assert res.pool_id == "DEFICIENT_POOL"
    assert res.urgency == "Immediate"  # below 0.5 regulatory min
    assert res.recommended_dosing["hypochlorite_dosing_hours"] > 0
    assert res.recommended_dosing["hypochlorite_dosing_pct"] > 0
    assert res.predicted_tomorrow["free_chlorine"] > 0.1


def test_v7_optimizer_safe_on_missing_values(prediction_service):
    # Edge-case: row with missing free_chlorine and ph
    sparse_row = pd.Series({
        "pool_id": "SPARSE_POOL",
        "pool_volume_m3": 100.0,
    })

    res = prediction_service.optimise("SPARSE_POOL", sparse_row)
    assert res is not None
    assert res.recommended_dosing is not None


def test_v7_multi_day_chained_forecast(prediction_service):
    sample_row = pd.Series({
        "pool_id": "CHAIN_POOL",
        "free_chlorine": 2.2,
        "ph": 7.4,
        "turbidity": 0.3,
        "pool_volume_m3": 100.0,
        "reading_date": "2026-08-01",
    })

    def mock_wx(date, cols):
        return {c: 25.0 if "temp" in c else 0.0 for c in cols}

    res = prediction_service.forecast(
        "CHAIN_POOL", sample_row, pd.Timestamp("2026-08-02"), mock_wx, horizon_days=5
    )

    assert "forecast" in res
    fc_df = res["forecast"]
    assert len(fc_df) == 6  # day 0 (today) through day 5 inclusive
    assert all("predicted_cl" in fc_df.columns for _ in [1])
    assert all("predicted_ph" in fc_df.columns for _ in [1])
    assert all("predicted_turb" in fc_df.columns for _ in [1])
    assert all(band is not None for band in fc_df["uncertainty_band"])
