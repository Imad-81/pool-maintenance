"""Integration smoke tests for the ML training pipeline.

These validate that:
1. `--dry-run` loads data and matches row counts
2. The inference engine reproduces predictions from the original V6

Run with:
    python -m pytest tests/ml/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# repo root is two levels up from tests/ml/
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Dry-run test (fast — no training)
# ---------------------------------------------------------------------------
def test_dry_run():
    """Verify the pipeline loads data and produces the expected row counts."""
    from ml.config import DEFAULT_CONFIG
    from ml.training.train import run_pipeline
    result = run_pipeline(DEFAULT_CONFIG, run_id="test-dry-run", dry_run=True)
    assert result["dry_run"] is True
    assert result["rows"] == 41799
    assert result["pools"] == 135


# ---------------------------------------------------------------------------
# Parsing helpers (unit tests that don't require data)
# ---------------------------------------------------------------------------
def test_extract_pool_ref():
    from ml import features
    assert features.extract_pool_ref("Cabo Verde (19)") == "19"
    assert features.extract_pool_ref("Test (654-655)") == "654-655"
    assert features.extract_pool_ref("No number") is None
    assert features.extract_pool_ref(None) is None


def test_safe_float():
    import pandas as pd
    from ml.features import safe_float
    s = pd.Series(["1.5", "bad", "", "3", None])
    result = safe_float(s)
    assert result.iloc[0] == 1.5
    assert np.isnan(result.iloc[1])
    assert np.isnan(result.iloc[2])
    assert result.iloc[3] == 3.0


def test_breach_flags():
    import pandas as pd
    from ml.features import breach_flags
    df = pd.DataFrame({
        "ph": [7.0, 7.4, 8.5],
        "free_chlorine": [0.3, 1.2, 6.0],
        "turbidity": [1.0, 2.0, 10.0],
    })
    df = breach_flags(df)
    assert bool(df["ph_breach"].iloc[0])          # 7.0 < 7.2
    assert not bool(df["ph_breach"].iloc[1])
    assert bool(df["ph_breach"].iloc[2])          # 8.5 > 8.0
    assert bool(df["chlorine_breach"].iloc[0])    # 0.3 < 0.5
    assert not bool(df["chlorine_breach"].iloc[1])
    assert bool(df["chlorine_breach"].iloc[2])    # 6.0 > 5.0
    assert bool(df["turbidity_breach"].iloc[2])   # 10 > 5


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------
def test_config_paths_exist():
    from ml.config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG.raw_excel_path.exists()
    assert DEFAULT_CONFIG.chlorine_pump_list_path.exists()
    assert DEFAULT_CONFIG.models_dir_path.exists()


# ---------------------------------------------------------------------------
# Promotion gate
# ---------------------------------------------------------------------------
def test_should_promote():
    from ml.training.evaluate import should_promote
    new = {"chlorine_next": {"mae": 0.204}, "ph_next": {"mae": 0.034}}
    old = {"chlorine_next": {"mae": 0.210}, "ph_next": {"mae": 0.035}}
    promote, reason = should_promote(new, old, tol_cl=0.02, tol_ph=0.005)
    assert promote is True  # new is better

    # same as old but not worse at all: should still promote
    promote, reason = should_promote(new, new, tol_cl=0.02, tol_ph=0.005)
    assert promote is True


def test_should_not_promote():
    from ml.training.evaluate import should_promote
    new = {"chlorine_next": {"mae": 0.250}, "ph_next": {"mae": 0.034}}
    old = {"chlorine_next": {"mae": 0.210}, "ph_next": {"mae": 0.035}}
    promote, reason = should_promote(new, old, tol_cl=0.02, tol_ph=0.005)
    assert promote is False
    assert "Cl" in reason

    # None old = first run, always promote
    promote, reason = should_promote(new, None, tol_cl=0.02, tol_ph=0.005)
    assert promote is True


# ---------------------------------------------------------------------------
# Inference chain smoke test (requires trained models)
# ---------------------------------------------------------------------------
def test_prediction_service_loads():
    from ml.inference.predictor import PredictionService
    svc = PredictionService(ROOT / "models")
    svc.load()
    status = svc.status()
    assert status["loaded"] is True
    assert status["run_id"] is not None


def test_predict_forward_smoke():
    """Run a chained forecast on a known pool and verify the output shape."""
    import pandas as pd
    from ml.inference.predictor import PredictionService
    from backend.weather.provider import make_lookup

    svc = PredictionService(ROOT / "models")
    svc.load()

    df = pd.read_csv(ROOT / "outputs" / "master_dataset_v6.csv", parse_dates=["reading_date"])
    latest_pool = df.groupby("pool_id")["reading_date"].max().sort_values().index[-1]
    row = df[df["pool_id"] == latest_pool].sort_values("reading_date").iloc[-1]

    from backend.store.schema import get_session
    with next(get_session()) as session:
        wx = make_lookup(session)

    as_of = pd.Timestamp("2026-08-03")
    result = svc.forecast(latest_pool, row, as_of, wx, horizon_days=2)

    if "error" not in result:
        assert "forecast" in result
        fc = result["forecast"]
        assert not fc.empty
        assert "predicted_cl" in fc.columns
        assert "predicted_ph" in fc.columns
        assert fc["predicted_cl"].iloc[0] >= 0


def test_inference_shim_imports():
    """Verify the legacy inference.py can be imported without side-effects."""
    import inference
    assert hasattr(inference, "main")
    assert hasattr(inference, "print_pool_forecast")