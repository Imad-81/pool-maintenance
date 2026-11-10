"""
V8 Grouped Models — Feature Engineering.

Builds feature matrices for Group 1 (short-gap) and Group 2 (long-gap)
separately, sharing a base feature set and each adding group-specific
domain features.

NEW vs the base V8 features_v8.py:
  — CYA features (cyanuric acid stabilizer — 0% null, high domain value)
  — Chemical dosing features (shock, solid Cl, acid, pH pump, clarifier)
  — Daylight hours (solar exposure driver separate from UV index)
  — G1-specific: stability scores, dosing streak, Cl range
  — G2-specific: physics decay estimate, cumulative weather × gap, reopening flag
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

from ml.v8.config_v8 import V8Config, DEFAULT_V8_CONFIG, SUMMER_MONTHS
from ml.v8.sparse_dataset import COL_POOL, COL_DATE, add_gap_columns
from ml.v8.grouped.grouper import G1_MAX_GAP, G2_MIN_GAP, G2_MAX_GAP

log = logging.getLogger("ml.v8.grouped.features_grouped")

# ---------------------------------------------------------------------------
# Raw-data column → canonical feature name map
# (extends the base v8 map with new columns)
# ---------------------------------------------------------------------------
_COL_MAP = {
    # Chemistry (pre / post / delta)
    "free_chlorine_pre_ppm":        "cl_pre",
    "free_chlorine_post_ppm":       "cl_post",
    "chlorine_dosage_boost_ppm":    "cl_dosage_boost",
    "ph_pre":                       "ph_pre",
    "ph_post":                      "ph_post",
    "ph_delta":                     "ph_delta",
    "turbidity_pre":                "turb_pre",
    "turbidity_post":               "turb_post",
    "turbidity_delta":              "turb_delta",
    # CYA (cyanuric acid stabilizer — NEW)
    "cya_pre_ppm":                  "cya_pre",
    "cya_post_ppm":                 "cya_post",
    "cya_added_ppm":                "cya_added",
    # Dosing (NEW)
    "shock_dosage_ppm":             "shock_ppm",
    "erodible_active_cl2_added_grams": "cl_solid_grams",
    "daily_pump_cl2_delivered_ppm": "pump_cl_ppm",
    "is_chemical_dosed_day":        "is_dosed",
    "acid_added_grams":             "acid_grams",
    "daily_pump_ph_minus_ml":       "pump_ph_minus_ml",
    "clarifier_added_grams":        "clarifier_grams",
    # Pool static
    "water_temperature_c":          "water_temperature_c",
    "pool_volume":                  "pool_volume",
    "pool_surface_area":            "pool_surface_area",
    "community_pool":               "community_pool",
    "outdoor_pool":                 "outdoor_pool",
    # Weather
    "solar_radiation_mj":           "solar_radiation_mj",
    "temperature_ambient_mean_c":   "temperature_ambient_mean_c",
    "temperature_ambient_max_c":    "temperature_ambient_max_c",
    "precipitation_mm":             "precipitation_mm",
    "sunshine_duration_hrs":        "sunshine_duration_hrs",
    "daylight_duration_hrs":        "daylight_duration_hrs",   # NEW
    "uv_index_max":                 "uv_index_max",
    "wind_speed_max_kmh":           "wind_speed_max_kmh",
    "et0_evapotranspiration":       "et0_evapotranspiration",
}

# ---------------------------------------------------------------------------
# Feature group definitions
# ---------------------------------------------------------------------------

def shared_base_features() -> List[str]:
    """Feature columns present in BOTH G1 and G2 feature matrices."""
    return [
        # Gap
        "gap_since", "gap_log", "gap_category", "is_long_gap_lag", "seasonal_visit_freq",
        # Chemistry state
        "cl_pre", "cl_post", "cl_dosage_boost",
        "ph_pre", "ph_post", "ph_delta",
        "turb_pre", "turb_post", "turb_delta",
        # CYA — NEW
        "cya_pre", "cya_post", "cya_added",
        "cya_lag1",           # CYA at previous visit
        "cya_cl_ratio",       # cya_pre / (cl_post + 0.01) — stabiliser intensity
        "cya_effectiveness",  # Effective Cl: cl_post / (1 + cya_pre / 100)
        # Dosing — NEW
        "is_dosed",
        "shock_ppm",
        "cl_solid_grams",
        "pump_cl_ppm",
        "acid_grams",
        "pump_ph_minus_ml",
        "clarifier_grams",
        "dosed_lag1",         # was previous visit dosed?
        "total_cl_input",     # cl_dosage_boost + shock_ppm + pump_cl_ppm
        # Rate features
        "cl_per_gap_day", "ph_per_gap_day", "turb_per_gap_day",
        # Lag features
        "cl_lag1", "cl_lag2", "ph_lag1", "ph_lag2", "turb_lag1", "turb_lag2",
        "cl_roll3_mean", "cl_roll3_std",
        "ph_roll3_mean", "ph_roll3_std",
        "turb_roll3_mean",
        # Temporal
        "visit_month", "visit_day_of_week", "is_summer", "visit_year", "pool_visit_number",
        # Pool static
        "pool_volume", "pool_surface_area", "community_pool", "outdoor_pool",
        "water_temperature_c",
        # Weather
        "solar_radiation_mj", "temperature_ambient_max_c", "temperature_ambient_mean_c",
        "precipitation_mm", "sunshine_duration_hrs", "daylight_duration_hrs",
        "uv_index_max", "wind_speed_max_kmh", "et0_evapotranspiration",
    ]


def g1_extra_features() -> List[str]:
    """Extra features for Group 1 (short-gap, high-autocorrelation regime)."""
    return [
        "cl_stability_score",    # 1 - std/mean of last 3 Cl readings (is state stable?)
        "cl_range_last3",        # max - min Cl over last 3 visits (spread)
        "ph_stability_score",    # 1 - std/mean pH last 3 visits
        "days_since_last_dosing",# visits since is_dosed==1
        "dosed_streak",          # consecutive visits with dosing
    ]


def g2_extra_features() -> List[str]:
    """Extra features for Group 2 (long-gap, decay-dominant regime)."""
    return [
        "gap_since_sq",             # squared gap — non-linear decay at longer windows
        "cya_modulated_decay",      # expected Cl decay given CYA buffer: gap × (1/(1+cya/100))
        "weather_uv_x_gap",         # uv_index_max × gap_since (accumulated UV exposure)
        "weather_et0_x_gap",        # et0 × gap_since (accumulated evaporation)
        "weather_temp_x_gap",       # temp_mean × gap_since (heat accumulation)
        "is_reopening",             # 1 if gap_since > 21 (likely seasonal re-opening)
        "seasonal_gap_product",     # is_summer × gap_since (summer ≈ faster decay)
        "ph_drift_expected",        # ph_post_last + small drift per gap day (seasonal)
    ]


def g1_all_features() -> List[str]:
    return shared_base_features() + g1_extra_features()


def g2_all_features() -> List[str]:
    return shared_base_features() + g2_extra_features()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    return (a / b.replace(0, np.nan)).fillna(fill)


def _seasonal_visit_freq(df: pd.DataFrame) -> pd.Series:
    """Pool's avg visits/month in this season."""
    if "visit_season" not in df.columns:
        month = pd.to_datetime(df[COL_DATE]).dt.month
        df = df.copy()
        df["visit_season"] = month.map({
            12: "Winter", 1: "Winter", 2: "Winter",
             3: "Spring", 4: "Spring", 5: "Spring",
             6: "Summer", 7: "Summer", 8: "Summer", 9: "Summer",
            10: "Autumn", 11: "Autumn",
        })
    counts = (
        df.groupby([COL_POOL, "visit_season"]).size()
          .rename("season_visits").reset_index()
    )
    counts["visits_per_month"] = counts["season_visits"] / 3.0
    df2 = df.reset_index().merge(
        counts[[COL_POOL, "visit_season", "visits_per_month"]],
        on=[COL_POOL, "visit_season"], how="left",
    ).set_index(df.index)
    return df2["visits_per_month"].fillna(0.0)


def _rolling_stability(series: pd.Series, window: int = 3) -> pd.Series:
    """
    Stability score = 1 - (rolling_std / rolling_mean).
    Returns NaN where mean is 0 or fewer than 2 periods available.
    Clipped to [0, 1].
    """
    r_mean = series.rolling(window, min_periods=2).mean()
    r_std  = series.rolling(window, min_periods=2).std()
    score  = 1.0 - _safe_div(r_std, r_mean.abs(), fill=np.nan)
    return score.clip(0.0, 1.0)


def _days_since_event(flag_series: pd.Series) -> pd.Series:
    """
    Given a binary 0/1 series (is_dosed_day), return the number of VISITS
    since the last event (not calendar days). NaN for the first occurrence.
    """
    out = pd.Series(np.nan, index=flag_series.index)
    last_event_idx = -1
    for i, (idx, val) in enumerate(flag_series.items()):
        if val == 1:
            last_event_idx = i
        elif last_event_idx >= 0:
            out[idx] = i - last_event_idx
    return out


def _dosed_streak(flag_series: pd.Series) -> pd.Series:
    """Consecutive visits where is_dosed == 1."""
    out = []
    streak = 0
    for v in flag_series:
        if v == 1:
            streak += 1
        else:
            streak = 0
        out.append(streak)
    return pd.Series(out, index=flag_series.index)


# ---------------------------------------------------------------------------
# MAIN: build_features_for_group
# ---------------------------------------------------------------------------

def build_features_for_group(
    df: pd.DataFrame,
    group_id: str,          # "G1" or "G2"
    cfg: V8Config = DEFAULT_V8_CONFIG,
) -> pd.DataFrame:
    """
    Build the full feature matrix for one group.

    Input:  sparse-visit DataFrame with gap columns + group column assigned.
            df should contain ONLY the rows for this group.
    Output: DataFrame with the appropriate all-features columns + sample_weight.
    """
    assert group_id in ("G1", "G2"), f"group_id must be 'G1' or 'G2', got '{group_id}'"
    df = df.copy()

    # --- 0. Rename raw columns to canonical names ---
    rename_present = {k: v for k, v in _COL_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_present)

    # --- 1. Ensure gap columns exist ---
    if "gap_since" not in df.columns:
        df = add_gap_columns(df)

    # --- 2. Base gap features ---
    df["gap_log"]           = np.log1p(df["gap_since"].fillna(0))
    df["is_long_gap_lag"]   = (df["gap_since"] > 14).astype(int)
    df["visit_month"]       = pd.to_datetime(df[COL_DATE]).dt.month
    df["visit_season"] = df["visit_month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
         3: "Spring", 4: "Spring", 5: "Spring",
         6: "Summer", 7: "Summer", 8: "Summer", 9: "Summer",
        10: "Autumn", 11: "Autumn",
    })
    df["seasonal_visit_freq"] = _seasonal_visit_freq(df).values

    # --- 3. CYA features ---
    if "cya_pre" in df.columns:
        df["cya_lag1"]        = df.groupby(COL_POOL)["cya_pre"].shift(1)
        df["cya_cl_ratio"]    = _safe_div(df["cya_pre"], df["cl_post"] + 0.01)
        # Effective chlorine: at high CYA, free Cl is "buffered" (less active but more stable)
        # Fraction of Cl that is HOCl (active form) decreases with CYA
        df["cya_effectiveness"] = df["cl_post"] / (1.0 + df["cya_pre"] / 100.0)
    else:
        for c in ["cya_lag1", "cya_cl_ratio", "cya_effectiveness"]:
            df[c] = 0.0

    # --- 4. Dosing features ---
    for col in ["is_dosed", "shock_ppm", "cl_solid_grams", "pump_cl_ppm",
                "acid_grams", "pump_ph_minus_ml", "clarifier_grams"]:
        if col not in df.columns:
            df[col] = 0.0

    if "cl_dosage_boost" not in df.columns:
        df["cl_dosage_boost"] = 0.0

    df["total_cl_input"] = (
        df["cl_dosage_boost"].fillna(0)
        + df["shock_ppm"].fillna(0)
        + df["pump_cl_ppm"].fillna(0)
    )
    df["dosed_lag1"] = df.groupby(COL_POOL)["is_dosed"].shift(1).fillna(0)

    # --- 5. Rate features (change per gap day) ---
    for col_post, rate_col, pre_col in [
        ("cl_post",   "cl_per_gap_day",   "cl_pre"),
        ("ph_post",   "ph_per_gap_day",   "ph_pre"),
        ("turb_post", "turb_per_gap_day", "turb_pre"),
    ]:
        if col_post in df.columns:
            prev_post = df.groupby(COL_POOL)[col_post].shift(1)
            current   = df.get(pre_col, df[col_post])
            df[rate_col] = _safe_div(current - prev_post, df["gap_since"], fill=0.0)
        else:
            df[rate_col] = 0.0

    # --- 6. Lag features ---
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

    # --- 7. Temporal features ---
    df["visit_day_of_week"] = pd.to_datetime(df[COL_DATE]).dt.dayofweek
    df["visit_year"]        = pd.to_datetime(df[COL_DATE]).dt.year
    df["is_summer"]         = df["visit_month"].isin(SUMMER_MONTHS).astype(int)
    df["pool_visit_number"] = df.groupby(COL_POOL).cumcount()

    # --- 8. Sample weights (same logic as v8) ---
    summer_mask = df["visit_month"].isin(SUMMER_MONTHS)
    base_w = np.where(summer_mask, cfg.gap_weight_summer, cfg.gap_weight_non_summer)
    gap_denom = df["gap_since"].fillna(1.0).clip(lower=1.0)
    df["sample_weight"] = base_w / np.sqrt(gap_denom)
    df["sample_weight"] = df["sample_weight"] / df["sample_weight"].mean()

    # =========================================================
    # GROUP-SPECIFIC FEATURES
    # =========================================================

    if group_id == "G1":
        # --- G1a: Chlorine stability over last 3 visits ---
        if "cl_post" in df.columns:
            g = df.groupby(COL_POOL)["cl_post"]
            df["cl_stability_score"] = g.transform(_rolling_stability)
            df["cl_range_last3"]     = g.transform(
                lambda x: x.rolling(3, min_periods=2).max() - x.rolling(3, min_periods=2).min()
            )
        else:
            df["cl_stability_score"] = 0.5
            df["cl_range_last3"]     = 0.0

        if "ph_post" in df.columns:
            df["ph_stability_score"] = df.groupby(COL_POOL)["ph_post"].transform(_rolling_stability)
        else:
            df["ph_stability_score"] = 0.5

        # --- G1b: Dosing history ---
        if "is_dosed" in df.columns:
            df["days_since_last_dosing"] = df.groupby(COL_POOL)["is_dosed"].transform(
                _days_since_event
            )
            df["dosed_streak"] = df.groupby(COL_POOL)["is_dosed"].transform(
                _dosed_streak
            )
        else:
            df["days_since_last_dosing"] = np.nan
            df["dosed_streak"]           = 0

    elif group_id == "G2":
        # --- G2a: Non-linear gap effect ---
        df["gap_since_sq"] = df["gap_since"].fillna(0) ** 2

        # --- G2b: CYA-modulated expected Cl decay ---
        # Physics: Cl decays at rate k ≈ k_base / (1 + CYA/100)
        # Over gap_since days: expected_remaining = cl_lag1 × exp(-k × gap_since)
        # We use an approximation: decay_factor = gap_since / (1 + cya_lag1 / 100)
        if "cya_lag1" in df.columns and "cl_lag1" in df.columns:
            cya_buf  = 1.0 + df["cya_lag1"].fillna(30) / 100.0  # higher CYA → slower decay
            k_approx = df["gap_since"].fillna(7) / cya_buf
            df["cya_modulated_decay"] = df["cl_lag1"].fillna(2.5) * np.exp(-0.05 * k_approx)
        else:
            df["cya_modulated_decay"] = 0.0

        # --- G2c: Accumulated weather over gap period ---
        # Approximate: today's daily rate × gap_since (slow-varying weather)
        if "uv_index_max" in df.columns:
            df["weather_uv_x_gap"]   = df["uv_index_max"].fillna(0)   * df["gap_since"].fillna(7)
        else:
            df["weather_uv_x_gap"] = 0.0
        if "et0_evapotranspiration" in df.columns:
            df["weather_et0_x_gap"]  = df["et0_evapotranspiration"].fillna(0) * df["gap_since"].fillna(7)
        else:
            df["weather_et0_x_gap"] = 0.0
        if "temperature_ambient_mean_c" in df.columns:
            df["weather_temp_x_gap"] = df["temperature_ambient_mean_c"].fillna(20) * df["gap_since"].fillna(7)
        else:
            df["weather_temp_x_gap"] = 0.0

        # --- G2d: Reopening flag and seasonal gap interaction ---
        df["is_reopening"]         = (df["gap_since"].fillna(0) > 21).astype(int)
        df["seasonal_gap_product"] = df["is_summer"] * df["gap_since"].fillna(0)

        # --- G2e: Expected pH drift ---
        # pH tends to rise ~0.02/day in summer, ~0.01/day in winter due to algae / CO2
        ph_drift_rate = np.where(df["is_summer"] == 1, 0.02, 0.01)
        if "ph_lag1" in df.columns:
            df["ph_drift_expected"] = df["ph_lag1"].fillna(7.4) + ph_drift_rate * df["gap_since"].fillna(7)
        else:
            df["ph_drift_expected"] = 7.4 + ph_drift_rate * df["gap_since"].fillna(7)

    log.info(
        "build_features_for_group(%s): %d rows, %d feature cols",
        group_id, len(df),
        len(g1_all_features() if group_id == "G1" else g2_all_features()),
    )
    return df


# ---------------------------------------------------------------------------
# Fill-value computation (median from train set)
# ---------------------------------------------------------------------------

def compute_fill_values(df_train: pd.DataFrame, feature_names: List[str]) -> dict:
    fill = {}
    for col in feature_names:
        if col in df_train.columns:
            m = df_train[col].median()
            fill[col] = float(m) if pd.notna(m) else 0.0
        else:
            fill[col] = 0.0
    return fill


def apply_fill_values(df: pd.DataFrame, fill_values: dict) -> pd.DataFrame:
    df = df.copy()
    for col, val in fill_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
        else:
            df[col] = val
    return df

