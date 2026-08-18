"""
Unit tests for the Pool Visit Recommendation Engine.
"""

from datetime import date, datetime
import pandas as pd
import pytest

from ml.inference.chaining import UncertaintyBand
from ml.inference.visit_recommender import (
    compute_recommended_visit,
    get_seasonal_baseline_days,
)


def test_seasonal_baseline_cadence():
    # Summer (Jun-Sep) = 2 days
    assert get_seasonal_baseline_days(6) == 2
    assert get_seasonal_baseline_days(7) == 2
    assert get_seasonal_baseline_days(8) == 2
    assert get_seasonal_baseline_days(9) == 2

    # May = 4 days
    assert get_seasonal_baseline_days(5) == 4

    # April & October = 6 days
    assert get_seasonal_baseline_days(4) == 6
    assert get_seasonal_baseline_days(10) == 6

    # Winter = 7 days
    assert get_seasonal_baseline_days(1) == 7
    assert get_seasonal_baseline_days(12) == 7


def test_immediate_chlorine_breach_today():
    as_of = pd.Timestamp("2026-08-18")
    last_visit = pd.Timestamp("2026-08-15")

    # Forecast has severe chlorine depletion today (0.35 mg/L < 0.50 mg/L)
    df = pd.DataFrame([
        {
            "date": "2026-08-18",
            "day": "Tue ◀ TODAY",
            "day_offset_from_today": 0,
            "days_from_visit": 3,
            "predicted_cl": 0.35,
            "predicted_ph": 7.45,
            "predicted_turb": 0.6,
            "cl_breach": True,
            "ph_breach": False,
            "urgency": "URGENT",
            "status": "🚨 Regulatory breach",
            "is_today": True,
            "is_tomorrow": False,
            "uncertainty_band": UncertaintyBand(0, 0.35, 0.35, 7.45, 7.45, 0.6, 0.6),
        },
        {
            "date": "2026-08-19",
            "day": "Wed ◀ TOMORROW",
            "day_offset_from_today": 1,
            "days_from_visit": 4,
            "predicted_cl": 0.20,
            "predicted_ph": 7.50,
            "predicted_turb": 0.7,
            "cl_breach": True,
            "ph_breach": False,
            "urgency": "URGENT",
            "status": "🚨 Regulatory breach",
            "is_today": False,
            "is_tomorrow": True,
            "uncertainty_band": UncertaintyBand(1, 0.20, 0.20, 7.50, 7.50, 0.7, 0.7),
        },
    ])

    rec = compute_recommended_visit(df, as_of, last_visit)
    assert rec["date"] == "2026-08-18"
    assert rec["day_offset_from_today"] == 0
    assert rec["urgency"] == "Immediate"
    assert rec["trigger"] == "regulatory_breach"
    assert rec["is_breach"] is True
    assert rec["predicted_cl"] == 0.35
    assert "0.50" in rec["reason"]


def test_target_decay_visit_recommendation():
    as_of = pd.Timestamp("2026-08-18")
    last_visit = pd.Timestamp("2026-08-17")

    # Today and tomorrow are in client target (1.0-1.5), but day +2 drops below 1.0
    df = pd.DataFrame([
        {
            "date": "2026-08-18",
            "day": "Tue ◀ TODAY",
            "day_offset_from_today": 0,
            "days_from_visit": 1,
            "predicted_cl": 1.40,
            "predicted_ph": 7.40,
            "predicted_turb": 0.5,
            "cl_breach": False,
            "ph_breach": False,
            "urgency": "Routine",
            "is_today": True,
            "is_tomorrow": False,
            "uncertainty_band": UncertaintyBand(0, 1.40, 1.40, 7.40, 7.40, 0.5, 0.5),
        },
        {
            "date": "2026-08-19",
            "day": "Wed ◀ TOMORROW",
            "day_offset_from_today": 1,
            "days_from_visit": 2,
            "predicted_cl": 1.15,
            "predicted_ph": 7.48,
            "predicted_turb": 0.6,
            "cl_breach": False,
            "ph_breach": False,
            "urgency": "Routine",
            "is_today": False,
            "is_tomorrow": True,
            "uncertainty_band": UncertaintyBand(1, 1.15, 1.15, 7.48, 7.48, 0.6, 0.6),
        },
        {
            "date": "2026-08-20",
            "day": "Thu",
            "day_offset_from_today": 2,
            "days_from_visit": 3,
            "predicted_cl": 0.85,
            "predicted_ph": 7.55,
            "predicted_turb": 0.7,
            "cl_breach": False,
            "ph_breach": False,
            "urgency": "Advised",
            "is_today": False,
            "is_tomorrow": False,
            "uncertainty_band": UncertaintyBand(2, 0.65, 1.05, 7.51, 7.59, 0.66, 0.74),
        },
    ])

    rec = compute_recommended_visit(df, as_of, last_visit)
    assert rec["date"] == "2026-08-20"
    assert rec["day_offset_from_today"] == 2
    assert rec["urgency"] == "Advised"
    assert rec["trigger"] == "target_decay"
    assert rec["predicted_cl"] == 0.85
    assert "1.0" in rec["reason"]


def test_seasonal_routine_when_chemistry_stable():
    as_of = pd.Timestamp("2026-08-18")      # August (Summer baseline = 2 days)
    last_visit = pd.Timestamp("2026-08-18") # Visited today

    # All days stay inside optimal 1.0-1.5 range
    df = pd.DataFrame([
        {
            "date": "2026-08-18",
            "day": "Tue ◀ TODAY",
            "day_offset_from_today": 0,
            "days_from_visit": 0,
            "predicted_cl": 1.50,
            "predicted_ph": 7.40,
            "predicted_turb": 0.5,
            "cl_breach": False,
            "ph_breach": False,
            "urgency": "Routine",
            "is_today": True,
            "is_tomorrow": False,
        },
        {
            "date": "2026-08-19",
            "day": "Wed ◀ TOMORROW",
            "day_offset_from_today": 1,
            "days_from_visit": 1,
            "predicted_cl": 1.35,
            "predicted_ph": 7.45,
            "predicted_turb": 0.5,
            "cl_breach": False,
            "ph_breach": False,
            "urgency": "Routine",
            "is_today": False,
            "is_tomorrow": True,
        },
        {
            "date": "2026-08-20",
            "day": "Thu",
            "day_offset_from_today": 2,
            "days_from_visit": 2,
            "predicted_cl": 1.20,
            "predicted_ph": 7.50,
            "predicted_turb": 0.6,
            "cl_breach": False,
            "ph_breach": False,
            "urgency": "Routine",
            "is_today": False,
            "is_tomorrow": False,
        },
    ])

    rec = compute_recommended_visit(df, as_of, last_visit)
    # Since Summer baseline is 2 days from last visit (Aug 18 -> Aug 20, day_offset = 2)
    assert rec["date"] == "2026-08-20"
    assert rec["day_offset_from_today"] == 2
    assert rec["urgency"] == "Routine"
    assert rec["trigger"] == "seasonal_routine"
    assert rec["predicted_cl"] == 1.20
