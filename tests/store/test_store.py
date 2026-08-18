"""
Unit tests for store repository helpers and parsing logic.
"""

from datetime import datetime
import pandas as pd
import numpy as np

from backend.store.repo import classify_urgency
from backend.weather.provider import make_lookup
from backend.api.upload import _auto_detect_mapping, _safe_float, _parse_date_flexible


def test_classify_urgency():
    assert classify_urgency(0.2, 7.4) == "Immediate"  # Cl < 0.5
    assert classify_urgency(6.0, 7.4) == "Immediate"  # Cl > 5.0
    assert classify_urgency(1.5, 6.8) == "Immediate"  # pH < 7.2
    assert classify_urgency(1.5, 8.5) == "Immediate"  # pH > 8.0
    assert classify_urgency(1.5, 7.4) == "Routine"


def test_make_lookup_cache():
    d1 = pd.Timestamp("2026-08-01").normalize()
    d2 = pd.Timestamp("2026-08-02").normalize()

    cache = {
        d1: {"w_temp_max": 32.5, "w_uv_max": 8.5},
        d2: {"w_temp_max": 33.0, "w_uv_max": 9.0},
    }
    lookup = make_lookup(cache)

    res1 = lookup(d1, ["w_temp_max", "w_uv_max", "missing_col"])
    assert res1["w_temp_max"] == 32.5
    assert res1["w_uv_max"] == 8.5
    assert np.isnan(res1["missing_col"])

    # Missing date returns NaN
    res_miss = lookup(pd.Timestamp("2026-08-10"), ["w_temp_max"])
    assert np.isnan(res_miss["w_temp_max"])


def test_auto_detect_mapping():
    cols = ["Piscina ID", "Fecha Medición", "Cloro Libre", "pH", "Turbidez", "Volumen"]
    mapping = _auto_detect_mapping(cols)
    assert mapping.get("pool_id") == "Piscina ID"
    assert mapping.get("reading_date") == "Fecha Medición"
    assert mapping.get("free_chlorine") == "Cloro Libre"
    assert mapping.get("ph") == "pH"
    assert mapping.get("turbidity") == "Turbidez"
    assert mapping.get("pool_volume_m3") == "Volumen"


def test_safe_float_edge_cases():
    assert _safe_float("1,25") == 1.25
    assert _safe_float("  3.5  ") == 3.5
    assert _safe_float(None) is None
    assert _safe_float("nan") is None
    assert _safe_float("invalid") is None


def test_parse_date_flexible():
    d = _parse_date_flexible("2026-08-15")
    assert isinstance(d, datetime)
    assert d.year == 2026 and d.month == 8 and d.day == 15

    d2 = _parse_date_flexible("15/08/2026 10:30")
    assert isinstance(d2, datetime)
    assert d2.day == 15 and d2.month == 8

    assert _parse_date_flexible("") is None
    assert _parse_date_flexible("invalid-text") is None


def test_safe_float_predictor_helper():
    from ml.inference.predictor import _safe_float
    assert _safe_float(None, 2.0) == 2.0
    assert _safe_float(np.nan, 7.4) == 7.4
    assert _safe_float("invalid", 0.5) == 0.5
    assert _safe_float(1.23, 0.0) == 1.23
    assert _safe_float("1.45", 0.0) == 1.45

