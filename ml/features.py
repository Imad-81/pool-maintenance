"""
Shared feature-engineering helpers used by both training (`ml.training`) and
inference (`ml.inference`).

Keeping these in one place guarantees that the feature vector a model is
trained on is *bit-for-bit identical* to the one built at inference time —
any drift would silently degrade prediction quality, so parity is enforced
by a golden-output test (`tests/ml/test_feature_parity.py`).
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

from ml.config import (
    CLIENT_CL_TARGET_MAX,
    CLIENT_CL_TARGET_MIN,
    PH_IDEAL,
    REG_CHLORINE_CLOSE,
    REG_CHLORINE_IDEAL_MAX,
    REG_CHLORINE_MIN,
    REG_PH_MAX,
    REG_PH_MIN,
    REG_TURBIDITY_MAX,
)


# ---------------------------------------------------------------------------
# Small primitives
# ---------------------------------------------------------------------------

def safe_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def parse_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="mixed", dayfirst=True, errors="coerce")


def extract_pool_ref(s) -> Optional[str]:
    """Extract the numeric reference ID from a pool name, e.g.
    'Cabo Verde (19)' -> '19'. Handles compound IDs (654-655, 1122-2)."""
    if pd.isna(s):
        return None
    m = re.search(r"\(\s*(\d[\d\-]*\d|\d)\s*\)", str(s))
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Categorical derivation from flag columns
# ---------------------------------------------------------------------------

_POOL_TYPE_FLAGS = (
    "pool_heated", "pool_outdoor", "pool_community", "pool_private", "pool_public",
)
_DECK_COLS = ("deck_grass", "deck_mixed", "deck_paved")


def make_pool_type(row: pd.Series) -> str:
    parts = []
    for flag in _POOL_TYPE_FLAGS:
        if row.get(flag, 0):
            parts.append(flag.replace("pool_", ""))
    return "_".join(parts) if parts else "unknown"


def make_deck_type(row: pd.Series) -> str:
    g = float(row.get("deck_grass", 0) or 0)
    p = float(row.get("deck_paved", 0) or 0)
    m = float(row.get("deck_mixed", 0) or 0)
    if m > 0:
        return "mixed"
    if g > 0 and p > 0:
        return "mixed"
    if g > 0:
        return "grass"
    if p > 0:
        return "paved"
    return "unknown"


# ---------------------------------------------------------------------------
# Breach flags
# ---------------------------------------------------------------------------

def breach_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` augmented with regulatory breach flag columns."""
    df = df.copy()
    if "ph" in df.columns:
        df["ph_breach"] = (
            ~df["ph"].between(REG_PH_MIN, REG_PH_MAX) & df["ph"].notna()
        )
    if "free_chlorine" in df.columns:
        df["chlorine_breach"] = (
            ((df["free_chlorine"] < REG_CHLORINE_MIN) |
             (df["free_chlorine"] > REG_CHLORINE_CLOSE))
            & df["free_chlorine"].notna()
        )
        df["chlorine_low"] = (
            (df["free_chlorine"] < REG_CHLORINE_MIN) & df["free_chlorine"].notna()
        )
        df["chlorine_over_ideal"] = (
            (df["free_chlorine"] > REG_CHLORINE_IDEAL_MAX) & df["free_chlorine"].notna()
        )
    if "turbidity" in df.columns:
        df["turbidity_breach"] = (
            (df["turbidity"] > REG_TURBIDITY_MAX) & df["turbidity"].notna()
        )
    df["any_breach"] = (
        df.get("ph_breach", False) | df.get("chlorine_breach", False) | df.get("turbidity_breach", False)
    )
    return df


# ---------------------------------------------------------------------------
# Multi-visit deduplication (keep last reading per pool-day)
# ---------------------------------------------------------------------------

def dedup_keep_last_per_day(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["reading_date_only"] = df["reading_date"].dt.normalize()
    visit_counts = df.groupby(["pool_id", "reading_date_only"])["reading_date"].transform("count")
    df["multi_visit_day"] = (visit_counts > 1).astype(int)
    df = (
        df.sort_values(["pool_id", "reading_date"])
           .drop_duplicates(subset=["pool_id", "reading_date_only"], keep="last")
           .sort_values(["pool_id", "reading_date"])
           .reset_index(drop=True)
    )
    return df


# ---------------------------------------------------------------------------
# Consecutive-clean-visit counter (vectorised-friendly helper)
# ---------------------------------------------------------------------------

def consecutive_clean(series: pd.Series) -> list[int]:
    """Count consecutive 0s (non-breach) ending at each position, in row order."""
    result: list[int] = []
    count = 0
    for val in series:
        if val == 0:
            count += 1
        else:
            count = 0
        result.append(count)
    return result


# ---------------------------------------------------------------------------
# Headroom features (used in training AND recomputed at every chained step)
# ---------------------------------------------------------------------------

def add_headroom_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["chlorine_headroom_low"]  = df["free_chlorine"] - REG_CHLORINE_MIN
    df["chlorine_headroom_high"] = REG_CHLORINE_CLOSE - df["free_chlorine"]
    df["ph_headroom_low"]        = df["ph"] - REG_PH_MIN
    df["ph_headroom_high"]       = REG_PH_MAX - df["ph"]
    df["turbidity_headroom"]     = REG_TURBIDITY_MAX - df.get("turbidity", 0)
    df["min_headroom"] = df[
        ["chlorine_headroom_low", "chlorine_headroom_high",
         "ph_headroom_low", "ph_headroom_high", "turbidity_headroom"]
    ].min(axis=1)
    df["cl_below_client_target"] = (CLIENT_CL_TARGET_MIN - df["free_chlorine"]).clip(lower=0)
    df["cl_above_client_target"] = (df["free_chlorine"] - CLIENT_CL_TARGET_MAX).clip(lower=0)
    return df


# ---------------------------------------------------------------------------
# Trend / rate features
# ---------------------------------------------------------------------------

def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ph_trend"]        = df["ph"]        - df["ph_lag1"]
    df["chlorine_trend"]  = df["free_chlorine"] - df["chlorine_lag1"]
    df["turbidity_trend"] = df["turbidity"] - df["turbidity_lag1"]
    safe_gap = df["days_since_last_visit"].replace(0, np.nan)
    df["ph_rate_per_day"]        = df["ph_trend"]        / safe_gap
    df["chlorine_rate_per_day"]  = df["chlorine_trend"]  / safe_gap
    df["turbidity_rate_per_day"] = df["turbidity_trend"] / safe_gap
    return df


def cl_effectiveness(cl: float, ph: float) -> float:
    """HOCl active-fraction proxy — chlorine effectiveness falls as pH rises
    above 7.5. Identical formulation to pipeline_v6 STEP 7."""
    ph_factor = np.where(
        ph <= 7.5,
        1.0,
        1.0 - 0.5 * ((ph - 7.5) / 0.5),
    )
    return float(cl * np.clip(ph_factor, 0.1, 1.0))


# ---------------------------------------------------------------------------
# Feature group definitions (return lists of column names present in `df`)
# ---------------------------------------------------------------------------

def static_features() -> list[str]:
    return [
        "pool_surface_m2", "pool_volume_m3", "filter_diameter",
        "filter_count", "motor_count",
    ]

def lag_features() -> list[str]:
    return [
        "ph_lag1", "ph_lag2", "chlorine_lag1", "chlorine_lag2",
        "turbidity_lag1", "turbidity_lag2",
    ]

def rolling_features() -> list[str]:
    return [
        "ph_roll3_mean", "ph_roll3_std",
        "chlorine_roll3_mean", "chlorine_roll3_std",
        "turbidity_roll3_mean",
    ]

def temporal_features() -> list[str]:
    return [
        "days_since_last_visit", "visit_month", "visit_is_summer",
        "visit_day_of_week", "visit_year", "pool_visit_number",
    ]

def control_features() -> list[str]:
    return [
        "hypochlorite_dosing_pct", "hypochlorite_dosing_hours",
        "ph_dosing_pct", "ph_dosing_hours",
        "daily_filtration_hours", "water_temperature",
    ]

def product_features() -> list[str]:
    return ["last_total_chlorine_applied", "total_ph_minus_product"]

def headroom_features() -> list[str]:
    return [
        "chlorine_headroom_low", "chlorine_headroom_high",
        "ph_headroom_low", "ph_headroom_high", "turbidity_headroom", "min_headroom",
        "cl_below_client_target", "cl_above_client_target",
    ]

def trend_features() -> list[str]:
    return [
        "ph_trend", "chlorine_trend", "turbidity_trend",
        "ph_rate_per_day", "chlorine_rate_per_day", "turbidity_rate_per_day",
    ]

def breach_history_features() -> list[str]:
    return [
        "consecutive_clean_visits", "breach_rate_last5",
        "current_any_breach", "current_ph_breach", "current_chlorine_breach",
        "multi_visit_day",
    ]

def chemistry_features() -> list[str]:
    return [
        "ph_deviation", "chlorine_deficit",
        "cl_effectiveness_index", "chlorine_dose_per_m3", "ph_minus_dose_per_m3",
        "chlorine_decay_per_m3",
    ]

def weather_current_features() -> list[str]:
    return [
        "w_temp_max", "w_temp_mean", "w_uv_max", "w_uv_clear_sky_max",
        "w_solar_radiation", "w_sunshine_hours", "w_precipitation_mm",
        "w_wind_max_kmh", "w_et0",
    ]

def weather_cumulative_features() -> list[str]:
    return ["w_uv_sum_since", "w_solar_sum_since", "w_precip_sum_since", "w_temp_mean_since"]

def weather_tomorrow_features_names() -> list[str]:
    """Raw weather column names used to derive the `w_tmrw_*` tomorrow block."""
    return [
        "w_temp_max", "w_temp_mean", "w_uv_max", "w_uv_clear_sky_max",
        "w_solar_radiation", "w_sunshine_hours", "w_precipitation_mm",
        "w_wind_max_kmh", "w_et0",
    ]


CATEGORICAL_FEATURES: tuple[str, ...] = ("pool_type", "deck_type")


def all_numeric_feature_groups() -> list[str]:
    """Concatenate every feature group in fixed training order."""
    return (
        static_features() + lag_features() + rolling_features() +
        temporal_features() + control_features() + product_features() +
        headroom_features() + trend_features() + breach_history_features() +
        chemistry_features() + weather_current_features() +
        weather_cumulative_features()
        # tomorrow-weather names are added dynamically by the trainer after the
        # asof join because their count depends on which weather columns exist.
    )