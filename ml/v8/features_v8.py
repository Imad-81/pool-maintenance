"""
V8 Gap-Aware Feature Engineering.

Builds the feature matrix for the v8 sparse-visit model from a DataFrame
that already has gap_since / gap_to columns (from sparse_dataset.add_gap_columns).

Key design principles:
  - ALL rate-of-change features divide by the actual gap (days_since_last),
    never assume daily cadence.
  - Physics-imputed columns (pure_forward, active_hocl, cya_cumulative, etc.)
    are intentionally excluded — they are artifacts of the v7 reconstruction
    and not available in real-time inference from sparse visits.
  - Sample weights are attached here (1/gap_days × seasonal_factor) so the
    trainer can pass them directly to LightGBM.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

from ml.v8.config_v8 import V8Config, DEFAULT_V8_CONFIG, SUMMER_MONTHS
from ml.v8.sparse_dataset import COL_POOL, COL_DATE, add_gap_columns

log = logging.getLogger("ml.v8.features_v8")


# ---------------------------------------------------------------------------
# Feature group definitions
# (These are the COLUMN NAMES that will be present after build_features())
# ---------------------------------------------------------------------------

def gap_features() -> List[str]:
    return [
        "gap_since",          # days since last visit (NaN for pool's first visit)
        "gap_log",            # log1p(gap_since)
        "gap_category",       # ordinal 1–5
        "is_long_gap_lag",    # 1 if gap_since > 14
        "seasonal_visit_freq",# avg visits per month this season for this pool
    ]


def chemistry_state_features() -> List[str]:
    return [
        "cl_pre",             # free_chlorine_pre_ppm at current visit
        "ph_pre",             # ph_pre at current visit
        "turb_pre",           # turbidity_pre at current visit
        "cl_post",            # free_chlorine_post_ppm (post-treatment at current visit)
        "ph_post",            # ph_post
        "turb_post",          # turbidity_post
        "cl_dosage_boost",    # chlorine dosage added this visit (ppm)
        "ph_delta",           # ph change at this visit
        "turb_delta",         # turbidity change at this visit
    ]


def rate_features() -> List[str]:
    return [
        "cl_per_gap_day",     # (cl_pre - prev_cl_post) / gap_since
        "ph_per_gap_day",     # (ph_pre - prev_ph_post) / gap_since
        "turb_per_gap_day",   # (turb_pre - prev_turb_post) / gap_since
    ]


def lag_features() -> List[str]:
    return [
        "cl_lag1",            # cl_post at previous visit
        "ph_lag1",            # ph_post at previous visit
        "turb_lag1",          # turb_post at previous visit
        "cl_lag2",            # cl_post two visits ago
        "ph_lag2",            # ph_post two visits ago
        "turb_lag2",          # turb_post two visits ago
        "cl_roll3_mean",      # 3-visit rolling mean of cl_post
        "ph_roll3_mean",
        "turb_roll3_mean",
        "cl_roll3_std",
        "ph_roll3_std",
    ]


def temporal_features() -> List[str]:
    return [
        "visit_month",
        "visit_day_of_week",
        "is_summer",
        "visit_year",
        "pool_visit_number",
    ]


def pool_static_features() -> List[str]:
    return [
        "pool_volume",
        "pool_surface_area",
        "community_pool",
        "outdoor_pool",
        "water_temperature_c",
    ]


def weather_features() -> List[str]:
    return [
        "solar_radiation_mj",
        "temperature_ambient_max_c",
        "temperature_ambient_mean_c",
        "precipitation_mm",
        "sunshine_duration_hrs",
        "uv_index_max",
        "wind_speed_max_kmh",
        "et0_evapotranspiration",
    ]


def all_feature_groups() -> List[str]:
    return (
        gap_features()
        + chemistry_state_features()
        + rate_features()
        + lag_features()
        + temporal_features()
        + pool_static_features()
        + weather_features()
    )


# ---------------------------------------------------------------------------
# Column mapping from raw timeseries names → feature names
# ---------------------------------------------------------------------------
_COL_MAP = {
    "free_chlorine_pre_ppm":        "cl_pre",
    "free_chlorine_post_ppm":       "cl_post",
    "chlorine_dosage_boost_ppm":    "cl_dosage_boost",
    "ph_pre":                       "ph_pre",
    "ph_post":                      "ph_post",
    "ph_delta":                     "ph_delta",
    "turbidity_pre":                "turb_pre",
    "turbidity_post":               "turb_post",
    "turbidity_delta":              "turb_delta",
    "water_temperature_c":          "water_temperature_c",
    "pool_volume":                  "pool_volume",
    "pool_surface_area":            "pool_surface_area",
    "community_pool":               "community_pool",
    "outdoor_pool":                 "outdoor_pool",
    "solar_radiation_mj":           "solar_radiation_mj",
    "temperature_ambient_mean_c":   "temperature_ambient_mean_c",
    "temperature_ambient_max_c":    "temperature_ambient_max_c",
    "precipitation_mm":             "precipitation_mm",
    "sunshine_duration_hrs":        "sunshine_duration_hrs",
    "uv_index_max":                 "uv_index_max",
    "wind_speed_max_kmh":           "wind_speed_max_kmh",
    "et0_evapotranspiration":       "et0_evapotranspiration",
}


def _safe_div(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    """Divide a by b, returning fill where b is zero or NaN."""
    b_safe = b.replace(0, np.nan)
    return (a / b_safe).fillna(fill)


def _seasonal_visit_freq(df: pd.DataFrame) -> pd.Series:
    """
    For each row, compute the pool's average visits/month in that season.
    Uses the whole dataset so this is a pool-level static feature per season.
    """
    if "visit_season" not in df.columns:
        df = df.copy()
        df["visit_season"] = df["visit_month"].map({
            12: "Winter", 1: "Winter", 2: "Winter",
             3: "Spring", 4: "Spring", 5: "Spring",
             6: "Summer", 7: "Summer", 8: "Summer", 9: "Summer",
            10: "Autumn", 11: "Autumn",
        })

    # count visits per pool per season
    pool_season_counts = (
        df.groupby([COL_POOL, "visit_season"])
          .size()
          .rename("season_visits")
          .reset_index()
    )
    # approximate months per season per pool (3 calendar months each)
    pool_season_counts["visits_per_month"] = pool_season_counts["season_visits"] / 3.0

    df2 = df.reset_index().merge(
        pool_season_counts[[COL_POOL, "visit_season", "visits_per_month"]],
        on=[COL_POOL, "visit_season"],
        how="left",
    ).set_index(df.index)
    return df2["visits_per_month"].fillna(0.0)


# ---------------------------------------------------------------------------
# MAIN: build_features
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    cfg: V8Config = DEFAULT_V8_CONFIG,
) -> pd.DataFrame:
    """
    Transform a sparse-visit DataFrame into a model-ready feature matrix.

    Input:  output of sparse_dataset.add_gap_columns()
    Output: DataFrame with all_feature_groups() columns + sample_weight

    Does NOT drop NaN rows — the caller (train_v8) handles that after
    target construction, so we don't lose the first visit per pool here.
    """
    df = df.copy()

    # --- 1. Rename raw columns to canonical feature names ---
    rename_present = {k: v for k, v in _COL_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_present)

    # --- 2. Ensure gap columns exist ---
    if "gap_since" not in df.columns:
        df = add_gap_columns(df)

    # --- 3. Gap features ---
    df["gap_log"]         = np.log1p(df["gap_since"].fillna(0))
    df["is_long_gap_lag"] = (df["gap_since"] > 14).astype(int)
    df["visit_month"]     = df[COL_DATE].dt.month
    df["seasonal_visit_freq"] = _seasonal_visit_freq(df).values

    # --- 4. Rate features (change per gap day) ---
    # Previous visit's post-treatment values (lag-1 of post columns)
    for col_post, rate_col, pre_col in [
        ("cl_post",   "cl_per_gap_day",   "cl_pre"),
        ("ph_post",   "ph_per_gap_day",   "ph_pre"),
        ("turb_post", "turb_per_gap_day", "turb_pre"),
    ]:
        if col_post in df.columns:
            prev_post = df.groupby(COL_POOL)[col_post].shift(1)
            current_pre = df.get(pre_col, df.get(col_post, pd.Series(0.0, index=df.index)))
            delta = current_pre - prev_post
            df[rate_col] = _safe_div(delta, df["gap_since"], fill=0.0)
        else:
            df[rate_col] = 0.0

    # --- 5. Lag features ---
    for col_post, lag1, lag2, r3m, r3s in [
        ("cl_post",   "cl_lag1",   "cl_lag2",   "cl_roll3_mean",   "cl_roll3_std"),
        ("ph_post",   "ph_lag1",   "ph_lag2",   "ph_roll3_mean",   "ph_roll3_std"),
        ("turb_post", "turb_lag1", "turb_lag2", "turb_roll3_mean", None),
    ]:
        if col_post not in df.columns:
            for c in [lag1, lag2, r3m]:
                df[c] = 0.0
            if r3s:
                df[r3s] = 0.0
            continue
        g = df.groupby(COL_POOL)[col_post]
        df[lag1] = g.shift(1)
        df[lag2] = g.shift(2)
        df[r3m]  = g.transform(lambda x: x.rolling(3, min_periods=2).mean())
        if r3s is not None:
            df[r3s] = g.transform(lambda x: x.rolling(3, min_periods=2).std())

    # --- 6. Temporal features ---
    df["visit_day_of_week"] = df[COL_DATE].dt.dayofweek
    df["visit_year"]        = df[COL_DATE].dt.year
    df["pool_visit_number"] = df.groupby(COL_POOL).cumcount()

    # --- 7. Sample weights ---
    # Down-weight summer (over-represented) and long-gap rows (less reliable targets)
    summer_mask = df[COL_DATE].dt.month.isin(SUMMER_MONTHS)
    base_weight = np.where(summer_mask, cfg.gap_weight_summer, cfg.gap_weight_non_summer)
    # Additional: de-weight by gap length (longer gaps = less certain target)
    gap_denom = df["gap_since"].fillna(1.0).clip(lower=1.0)
    df["sample_weight"] = base_weight / np.sqrt(gap_denom)
    df["sample_weight"] = df["sample_weight"] / df["sample_weight"].mean()  # normalise to mean=1

    log.info(
        "build_features: %d rows, %d feature columns",
        len(df), len(all_feature_groups()),
    )
    return df


# ---------------------------------------------------------------------------
# Fill-value computation (median from train set, used at inference)
# ---------------------------------------------------------------------------

def compute_fill_values(df_train: pd.DataFrame) -> dict:
    """Compute median fill values for all numeric feature columns from the train set."""
    feats = all_feature_groups()
    present = [c for c in feats if c in df_train.columns]
    fill = {}
    for col in present:
        m = df_train[col].median()
        fill[col] = float(m) if pd.notna(m) else 0.0
    return fill


def apply_fill_values(df: pd.DataFrame, fill_values: dict) -> pd.DataFrame:
    """Apply pre-computed fill values to a feature DataFrame (in-place copy)."""
    df = df.copy()
    for col, val in fill_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
        else:
            df[col] = val
    return df

