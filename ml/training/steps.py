"""
Training step functions — one pure function per STEP of the original
`pipeline_v6.py`. Each takes (and returns) plain pandas objects so the
whole pipeline is composable, testable, and free of side-effects.

The composition lives in ml.training.train.run_pipeline.
"""

from __future__ import annotations

import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from ml.config import (
    DEFAULT_CONFIG,
    PipelineConfig,
    REG_CHLORINE_CLOSE,
    REG_CHLORINE_IDEAL_MAX,
    REG_CHLORINE_MIN,
    REG_PH_MAX,
    REG_PH_MIN,
    REG_TURBIDITY_MAX,
    CLIENT_CL_TARGET_MAX,
    CLIENT_CL_TARGET_MIN,
    CLIENT_CL_IDEAL,
    PH_IDEAL,
    RENAME_MAP,
)
from ml import features as F

log = logging.getLogger(__name__)


# ===========================================================================
# STEP 1 — LOAD & RENAME
# ===========================================================================

def load_and_rename(cfg: PipelineConfig) -> pd.DataFrame:
    """Read the master Excel and apply the Spanish->snake_case rename map."""
    path = cfg.raw_excel_path
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")
    log.info("STEP 1  load Excel: %s", path)
    df = pd.read_excel(path, header=0)
    df = df.rename(columns=RENAME_MAP)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")
    df = df.dropna(how="all").reset_index(drop=True)
    log.info("STEP 1  loaded %d rows x %d cols", df.shape[0], df.shape[1])
    return df


# ===========================================================================
# STEP 1.5 — POOL FILTER (liquid chlorine dosing pump only)
# ===========================================================================

def filter_chlorine_pump_pools(df_raw: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Keep only pools with liquid-chlorine dosing pumps (per client brief)."""
    path = cfg.chlorine_pump_list_path
    if not path.exists():
        log.warning("STEP 1.5  chlorine-pump list %s not found — skipping filter", path)
        df_raw["pool_ref"] = df_raw["pool_id"].apply(F.extract_pool_ref)
        return df_raw.copy()

    log.info("STEP 1.5  filter by chlorine-pump pool list: %s", path)
    df_pump = pd.read_excel(path)
    pump_refs = {str(r).strip() for r in df_pump["Referencia"].dropna()}
    pump_communities = {str(c).lower().strip() for c in df_pump["Comunidad"].dropna()}

    df_raw = df_raw.copy()
    df_raw["pool_ref"] = df_raw["pool_id"].apply(F.extract_pool_ref)

    def _norm_community(s) -> str:
        if pd.isna(s):
            return ""
        return re.sub(r"\s+", " ", str(s).lower().strip())

    df_raw["community_normalized"] = df_raw["community_name"].apply(_norm_community)
    mask_ref = df_raw["pool_ref"].isin(pump_refs)
    mask_comm = df_raw["community_normalized"].apply(
        lambda c: any(pc in c or c in pc for pc in pump_communities if len(pc) > 4)
    )
    mask = mask_ref | mask_comm
    out = df_raw[mask].copy()
    log.info(
        "STEP 1.5  %d -> %d rows (ref=%d, community_fallback=%d)",
        len(df_raw), len(out), int(mask_ref.sum()), int((mask & ~mask_ref).sum()),
    )
    return out


# ===========================================================================
# STEP 2 — WEATHER (fetch or cache, then slim & rename)
# ===========================================================================

WEATHER_RENAME = {
    "temperature_2m_max":         "w_temp_max",
    "temperature_2m_mean":        "w_temp_mean",
    "uv_index_max":               "w_uv_max",
    "uv_index_clear_sky_max":     "w_uv_clear_sky_max",
    "shortwave_radiation_sum":    "w_solar_radiation",
    "sunshine_duration":          "w_sunshine_duration_s",
    "precipitation_sum":          "w_precipitation_mm",
    "wind_speed_10m_max":         "w_wind_max_kmh",
    "et0_fao_evapotranspiration": "w_et0",
    "weather_code":               "w_weather_code",
}


def load_or_fetch_weather(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Return a slim daily-weather DataFrame renamed to w_* model features."""
    dates = F.parse_date_series(df["reading_date"])
    start = dates.min().strftime("%Y-%m-%d")
    end   = dates.max().strftime("%Y-%m-%d")
    path = cfg.weather_csv_path

    needs_fetch = True
    if path.exists():
        w_dates = pd.read_csv(path, usecols=["date"])["date"]
        if w_dates.min() <= start and w_dates.max() >= end:
            log.info("STEP 2  weather cache OK (%s -> %s)", w_dates.min(), w_dates.max())
            needs_fetch = False
        else:
            log.info("STEP 2  cache insufficient (%s -> %s); re-fetching", w_dates.min(), w_dates.max())

    if needs_fetch:
        from fetch_weather import fetch_daily_weather, save_to_csv
        log.info("STEP 2  fetching weather %s -> %s", start, end)
        rows, units = fetch_daily_weather(
            latitude=cfg.alicante_lat, longitude=cfg.alicante_lon,
            start_date=start, end_date=end, timezone=cfg.alicante_tz,
        )
        save_to_csv(rows, str(path), units=units)

    w = pd.read_csv(path)
    w["date"] = pd.to_datetime(w["date"]).dt.normalize()
    keep = [c for c in cfg.weather_cols_keep if c in w.columns]
    w = w[keep].rename(columns=WEATHER_RENAME)
    if "w_sunshine_duration_s" in w.columns:
        w["w_sunshine_hours"] = w["w_sunshine_duration_s"] / 3600
        w = w.drop(columns=["w_sunshine_duration_s"])
    log.info("STEP 2  weather %d days, features=%s", len(w), [c for c in w.columns if c != "date"])
    return w


# ===========================================================================
# STEP 3 — separate three sub-tables (readings / ops / products)
# ===========================================================================

READING_COLS = [
    "pool_id", "community_name", "reading_date", "technician",
    "ph", "turbidity", "free_chlorine", "sunscreen_abuse",
    "ph_pump_flow_rate", "hypochlorite_pump_flow_rate", "motor_flow_rate",
    "filter_diameter", "filter_count", "motor_count",
    "pool_heated", "pool_community", "pool_skimmer", "pool_overflow",
    "pool_outdoor", "pool_oval", "pool_private", "pool_public",
    "pool_rectangular_0714", "pool_rectangular_07", "pool_round",
    "pool_surface_m2", "vegetation_contamination", "pool_volume_m3",
    "deck_grass", "deck_mixed", "deck_paved",
]
OPS_COLS = [
    "ops_technician", "ops_date",
    "ph_dosing_hours", "daily_filtration_hours", "ph_dosing_pct",
    "filter_wash_rinse_time", "hypochlorite_dosing_hours",
    "hypochlorite_dosing_pct", "water_temperature",
]
PROD_COLS = [
    "prod_technician", "prod_date",
    "prod_t500_qp", "prod_alboral_tablets_250g", "prod_flovil_tablets",
    "prod_hypo_jugs_20kg", "prod_hypo_gr_chloryte", "prod_hypo_granular_xaka",
    "prod_hypo_sticks_bayrol", "prod_hypo_tab_ritocal",
    "prod_hypo_tablets_200g_qp", "prod_hypo_tablets_xaka",
    "prod_ph_minus_granular_6kg", "prod_ph_minus_liquid_13_5kg",
    "prod_ph_minus_liquid_27kg", "prod_protect_shine",
    "prod_sg_xaka_agonet", "prod_superklar",
]


def separate_subtables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["pool_id"] = df["pool_id"].ffill()
    df["community_name"] = df["community_name"].ffill()

    def _present(cols): return [c for c in cols if c in df.columns]

    reading_cols = _present(READING_COLS)
    ops_cols     = _present(OPS_COLS)
    prod_cols    = _present(PROD_COLS)

    readings = df[reading_cols].copy()
    readings = readings.dropna(subset=["reading_date"])

    ops = df[["pool_id"] + [c for c in ops_cols if c != "pool_id"]].copy()
    ops = ops.dropna(subset=["ops_date"])
    key_ops = [c for c in
               ["ph_dosing_hours", "daily_filtration_hours", "water_temperature",
                "hypochlorite_dosing_hours", "hypochlorite_dosing_pct"]
               if c in ops.columns]
    ops = ops.dropna(subset=key_ops, how="all")

    products = df[["pool_id"] + [c for c in prod_cols if c != "pool_id"]].copy()
    products = products.dropna(subset=["prod_date"])

    log.info("STEP 3  readings=%d  ops=%d  products=%d", len(readings), len(ops), len(products))
    return readings, ops, products


# ===========================================================================
# STEP 4 — clean readings / ops / products
# ===========================================================================

POOL_TYPE_FLAGS = [
    "pool_heated", "pool_community", "pool_skimmer", "pool_overflow",
    "pool_outdoor", "pool_oval", "pool_private", "pool_public",
    "pool_rectangular_0714", "pool_rectangular_07", "pool_round",
]


def clean_readings(df_readings: pd.DataFrame) -> pd.DataFrame:
    df = df_readings.copy()
    df["reading_date"] = F.parse_date_series(df["reading_date"])
    for col in ["ph", "turbidity", "free_chlorine"]:
        if col in df.columns:
            df[col] = F.safe_float(df[col])
    df = F.breach_flags(df)
    for col in POOL_TYPE_FLAGS:
        if col in df.columns:
            df[col] = F.safe_float(df[col]).fillna(0).astype(int)
    for col in ["pool_surface_m2", "pool_volume_m3", "filter_diameter", "filter_count", "motor_count"]:
        if col in df.columns:
            df[col] = F.safe_float(df[col])
    df["pool_type"] = df.apply(F.make_pool_type, axis=1)
    df["deck_type"]  = df.apply(F.make_deck_type, axis=1)
    df = F.dedup_keep_last_per_day(df)
    log.info("STEP 4  readings cleaned -> %d", len(df))
    return df


def clean_operations(df_ops: pd.DataFrame) -> pd.DataFrame:
    df = df_ops.copy()
    df["ops_date"] = F.parse_date_series(df["ops_date"])
    numeric = [c for c in [
        "ph_dosing_hours", "daily_filtration_hours", "ph_dosing_pct",
        "filter_wash_rinse_time", "hypochlorite_dosing_hours",
        "hypochlorite_dosing_pct", "water_temperature",
    ] if c in df.columns]
    for col in numeric:
        df[col] = F.safe_float(df[col])
    df = df.groupby(["pool_id", "ops_date"], as_index=False)[numeric].mean()
    df = df.sort_values(["pool_id", "ops_date"]).reset_index(drop=True)
    log.info("STEP 4  ops cleaned -> %d", len(df))
    return df


def clean_products(df_products: pd.DataFrame) -> pd.DataFrame:
    df = df_products.copy()
    df["prod_date"] = F.parse_date_series(df["prod_date"])
    prod_value_cols = [c for c in PROD_COLS
                       if c not in ("prod_technician", "prod_date", "pool_id")
                       and c in df.columns]
    for col in prod_value_cols:
        df[col] = F.safe_float(df[col]).fillna(0)
    hypo = [c for c in prod_value_cols if "hypo" in c.lower()]
    ph_minus = [c for c in prod_value_cols if "ph_minus" in c.lower()]
    flocc = [c for c in
             ["prod_flovil_tablets", "prod_superklar", "prod_sg_xaka_agonet", "prod_alboral_tablets_250g"]
             if c in df.columns]
    df["total_chlorine_product"]  = df[hypo].sum(axis=1) if hypo else 0
    df["total_ph_minus_product"]  = df[ph_minus].sum(axis=1) if ph_minus else 0
    df["total_flocculant_product"] = df[flocc].sum(axis=1) if flocc else 0
    agg = [c for c in prod_value_cols if c in df.columns] + [
        "total_chlorine_product", "total_ph_minus_product", "total_flocculant_product",
    ]
    df = df.groupby(["pool_id", "prod_date"], as_index=False)[agg].max()
    df = df.sort_values(["pool_id", "prod_date"]).reset_index(drop=True)
    log.info("STEP 4  products cleaned -> %d", len(df))
    return df


# ===========================================================================
# STEP 4.5 — backfill static pool data (fleet median + per-pool max)
# ===========================================================================

NUMERIC_STATIC = ["pool_volume_m3", "pool_surface_m2", "filter_diameter", "filter_count", "motor_count"]
FLAG_STATIC = [
    "pool_heated", "pool_community", "pool_skimmer", "pool_overflow",
    "pool_outdoor", "pool_oval", "pool_private", "pool_public",
    "pool_rectangular_0714", "pool_rectangular_07", "pool_round",
    "vegetation_contamination",
]


def backfill_static(df_readings: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df_readings.copy()
    summary: dict = {}
    for col in [c for c in NUMERIC_STATIC if c in df.columns]:
        pool_vals = df.groupby("pool_id")[col].apply(
            lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan
        )
        clean = pool_vals.dropna()
        fleet_median = clean.median() if len(clean) else 0.0
        df[col] = df["pool_id"].map(pool_vals).fillna(fleet_median)
        summary[col] = {
            "pools_with_original_data": int(len(clean)),
            "pools_filled_with_median": int(len(pool_vals) - len(clean)),
            "fleet_median": float(fleet_median),
        }
    for col in [c for c in FLAG_STATIC if c in df.columns]:
        pool_vals = df.groupby("pool_id")[col].apply(lambda x: x.max() if x.notna().any() else 0)
        df[col] = df["pool_id"].map(pool_vals).fillna(0).astype(int)
    for col in [c for c in ["deck_grass", "deck_mixed", "deck_paved"] if c in df.columns]:
        pool_vals = df.groupby("pool_id")[col].apply(lambda x: x.max() if x.notna().any() else 0)
        df[col] = df["pool_id"].map(pool_vals).fillna(0)
    df["pool_type"] = df.apply(F.make_pool_type, axis=1)
    df["deck_type"]  = df.apply(F.make_deck_type, axis=1)
    log.info("STEP 4.5  static backfill complete")
    return df, summary


# ===========================================================================
# STEP 5 — merge_asof per pool (ops + products)
# ===========================================================================

def _merge_asof_by_pool(df_left, df_right, left_date, right_date, right_cols, tolerance):
    parts = []
    for pool_id in df_left["pool_id"].unique():
        lp = df_left[df_left["pool_id"] == pool_id].sort_values(left_date)
        rp = df_right[df_right["pool_id"] == pool_id].sort_values(right_date)
        if len(rp) == 0:
            for col in right_cols:
                lp[col] = np.nan
            parts.append(lp)
            continue
        merged = pd.merge_asof(
            lp,
            rp[[c for c in right_cols + [right_date] if c in rp.columns]],
            left_on=left_date, right_on=right_date,
            direction="backward", tolerance=tolerance,
        )
        parts.append(merged)
    return pd.concat(parts, ignore_index=True)


def merge_subtables(df_readings, df_ops, df_products, cfg):
    tol = pd.Timedelta(f"{cfg.merge_tolerance_days}D")
    ops_vals = [c for c in df_ops.columns if c not in ("pool_id", "ops_date")]
    master = _merge_asof_by_pool(df_readings, df_ops, "reading_date", "ops_date", ops_vals, tol)
    prod_vals = [c for c in df_products.columns if c not in ("pool_id", "prod_date")]
    master = _merge_asof_by_pool(master, df_products, "reading_date", "prod_date", prod_vals, tol)
    log.info("STEP 5  master = %d x %d", master.shape[0], master.shape[1])
    return master


# ===========================================================================
# STEP 6 — join weather (exact date) + tomorrow block
# ===========================================================================

def join_weather(df_master: pd.DataFrame, df_weather: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Join today's and tomorrow's weather. Returns (master, weather_today_cols, weather_tmrw_cols)."""
    df = df_master.copy()
    df["reading_date_only"] = pd.to_datetime(df["reading_date"]).dt.normalize()
    weather_today = [c for c in df_weather.columns if c != "date"]

    before = df.shape[0]
    df = pd.merge(df, df_weather.rename(columns={"date": "reading_date_only"}),
                  on="reading_date_only", how="left")
    assert df.shape[0] == before, "Weather today join inflated rows"

    # Tomorrow block: shift weather date back by 1 so reading_date_only aligns.
    tmrw_src = [c for c in F.weather_tomorrow_features_names() if c in df_weather.columns]
    tmrw_cols = [f"w_tmrw_{c[2:]}" for c in tmrw_src]
    w_tmrw = df_weather[["date"] + tmrw_src].rename(columns={c: f"w_tmrw_{c[2:]}" for c in tmrw_src})
    w_tmrw["reading_date_only"] = w_tmrw["date"] - pd.Timedelta(days=1)
    w_tmrw = w_tmrw.drop(columns=["date"])
    before2 = df.shape[0]
    df = pd.merge(df, w_tmrw, on="reading_date_only", how="left")
    assert df.shape[0] == before2, "Tomorrow weather join inflated rows"

    log.info("STEP 6  weather joined: today=%d cols, tomorrow=%d cols", len(weather_today), len(tmrw_cols))
    return df, weather_today, tmrw_cols


# ===========================================================================
# STEP 7 — feature engineering
# ===========================================================================

def engineer_features(df_master, df_weather, cfg: PipelineConfig):
    df = df_master.sort_values(["pool_id", "reading_date"]).reset_index(drop=True)

    # Lags
    for col, prefix in [("ph", "ph"), ("free_chlorine", "chlorine"), ("turbidity", "turbidity")]:
        df[f"{prefix}_lag1"] = df.groupby("pool_id")[col].shift(1)
        df[f"{prefix}_lag2"] = df.groupby("pool_id")[col].shift(2)

    # Rolling
    for col, prefix in [("ph", "ph"), ("free_chlorine", "chlorine"), ("turbidity", "turbidity")]:
        df[f"{prefix}_roll3_mean"] = df.groupby("pool_id")[col].transform(
            lambda x: x.rolling(window=3, min_periods=2).mean())
        if prefix != "turbidity":
            df[f"{prefix}_roll3_std"] = df.groupby("pool_id")[col].transform(
                lambda x: x.rolling(window=3, min_periods=2).std())

    # Temporal
    df["days_since_last_visit"] = df.groupby("pool_id")["reading_date"].diff().dt.days
    df["visit_day_of_week"] = df["reading_date"].dt.dayofweek
    df["visit_month"]       = df["reading_date"].dt.month
    df["visit_is_summer"]   = df["visit_month"].isin([6, 7, 8, 9]).astype(int)
    df["visit_year"]        = df["reading_date"].dt.year
    df["pool_visit_number"] = df.groupby("pool_id").cumcount()

    # Chemistry
    df["ph_deviation"] = (df["ph"] - PH_IDEAL).abs()
    df["chlorine_deficit"] = (REG_CHLORINE_MIN - df["free_chlorine"]).clip(lower=0)
    df["last_total_chlorine_applied"] = (
        df["total_chlorine_product"].fillna(0) if "total_chlorine_product" in df.columns else 0.0
    )
    df["cl_effectiveness_index"] = df["free_chlorine"] * np.clip(
        np.where(df["ph"] <= 7.5, 1.0, 1.0 - 0.5 * ((df["ph"] - 7.5) / 0.5)), 0.1, 1.0
    )

    # Headroom + trend
    df = F.add_headroom_features(df)
    df = F.add_trend_features(df)

    # Breach history
    df["current_any_breach"] = df["any_breach"].astype(int)
    df["current_ph_breach"] = df["ph_breach"].astype(int) if "ph_breach" in df.columns else 0
    df["current_chlorine_breach"] = df["chlorine_breach"].astype(int) if "chlorine_breach" in df.columns else 0
    df["consecutive_clean_visits"] = df.groupby("pool_id")["current_any_breach"].transform(
        lambda x: F.consecutive_clean(x.values))
    df["breach_rate_last5"] = df.groupby("pool_id")["current_any_breach"].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean())

    # Volume-normalised
    df["chlorine_dose_per_m3"] = df["last_total_chlorine_applied"] / df["pool_volume_m3"]
    df["ph_minus_dose_per_m3"] = (
        df["total_ph_minus_product"] if "total_ph_minus_product" in df.columns else 0
    ) / df["pool_volume_m3"]
    df["chlorine_decay_per_m3"] = df["chlorine_rate_per_day"] / df["pool_volume_m3"]

    # Cumulative weather since last visit
    df = _add_cumulative_weather(df, df_weather)

    # Post-treatment setpoint features (configurable per run)
    df = F.add_setpoint_features(
        df,
        setpoint_cl=cfg.setpoint_free_chlorine,
        setpoint_ph=cfg.setpoint_ph,
        setpoint_turb=cfg.setpoint_turbidity,
    )

    log.info("STEP 7  feature engineering -> %d x %d", df.shape[0], df.shape[1])
    return df


def _add_cumulative_weather(df_master, df_weather):
    wx = df_weather.set_index("date")
    cols = ["w_uv_max", "w_solar_radiation", "w_precipitation_mm", "w_temp_mean"]
    cols = [c for c in cols if c in wx.columns]
    out_cols = {
        "w_uv_max": "w_uv_sum_since",
        "w_solar_radiation": "w_solar_sum_since",
        "w_precipitation_mm": "w_precip_sum_since",
        "w_temp_mean": "w_temp_mean_since",
    }
    parts = []
    for _pid, grp in df_master.groupby("pool_id"):
        dates = grp["reading_date"].dt.normalize().values
        gaps = grp["days_since_last_visit"].values
        rows = []
        for i in range(len(grp)):
            end_d = pd.Timestamp(dates[i]); days_bk = gaps[i]
            if pd.isna(days_bk) or days_bk <= 0:
                rows.append({out_cols[c]: np.nan for c in cols}); continue
            start_d = end_d - pd.Timedelta(days=int(days_bk) - 1)
            sl = wx[(wx.index >= start_d) & (wx.index <= end_d)]
            if len(sl) == 0:
                rows.append({out_cols[c]: np.nan for c in cols})
            else:
                agg = {out_cols["w_uv_max"]: sl["w_uv_max"].sum(),
                       out_cols["w_solar_radiation"]: sl["w_solar_radiation"].sum(),
                       out_cols["w_precipitation_mm"]: sl["w_precipitation_mm"].sum(),
                       out_cols["w_temp_mean"]: sl["w_temp_mean"].mean()}
                rows.append({k: (float(v) if pd.notna(v) else np.nan) for k, v in agg.items()})
        parts.append(pd.DataFrame(rows, index=grp.index))
    cum = pd.concat(parts).sort_index()
    for c in cum.columns:
        df_master[c] = cum[c]
    return df_master


# ===========================================================================
# STEP 8 — define next-day targets (interpolated)
# ===========================================================================

def build_targets(df_master, cfg: PipelineConfig):
    df = df_master.copy()
    df["next_reading_date"] = df.groupby("pool_id")["reading_date"].shift(-1)
    df["days_to_next_visit"] = (df["next_reading_date"] - df["reading_date"]).dt.days
    df["cl_next_visit"]   = df.groupby("pool_id")["free_chlorine"].shift(-1)
    df["ph_next_visit"]   = df.groupby("pool_id")["ph"].shift(-1)
    df["turb_next_visit"] = df.groupby("pool_id")["turbidity"].shift(-1)

    # Solar/thermal-driven daily chlorine degradation rate (0.15 - 0.50 mg/L per day)
    solar = df["w_solar_radiation"].fillna(30.0) if "w_solar_radiation" in df.columns else 30.0
    cl_decay = np.clip(0.15 + 0.005 * solar, 0.15, 0.50)

    # Temperature/aeration-driven pH upward drift (+0.03 to +0.08 units per day)
    temp = df["w_temp_mean"].fillna(25.0) if "w_temp_mean" in df.columns else 25.0
    ph_drift = np.clip(0.03 + 0.001 * temp, 0.03, 0.08)

    # Wind/dust-driven turbidity accumulation (+0.04 to +0.12 NTU per day)
    wind = df["w_wind_max_kmh"].fillna(15.0) if "w_wind_max_kmh" in df.columns else 15.0
    turb_rise = np.clip(0.04 + 0.002 * wind, 0.04, 0.12)

    # Configurable post-treatment setpoint — the assumed water state right
    # after the technician treats the pool at the current visit. The dataset
    # only contains pre-treatment readings, so synthetic targets anchor to
    # the setpoint: tomorrow = setpoint + 1 day of degradation toward the
    # next observed reading (linear interpolation in 1/k). When there is no
    # next visit (NaN gap), we fall back to pure 1-day kinetic decay from
    # the setpoint.
    sp_cl   = float(cfg.setpoint_free_chlorine)
    sp_ph   = float(cfg.setpoint_ph)
    sp_turb = float(cfg.setpoint_turbidity)

    k = df["days_to_next_visit"]
    safe_k = k.replace(0, np.nan)

    # All three targets follow the same setpoint-anchored interpolation:
    #   tomorrow = setpoint + (next_reading - setpoint) / gap
    # This naturally handles both upward and downward movement, so no
    # ph_treated / turb_cleaned special-case bypasses are needed (they
    # existed only because the old decay-from-reading formulation could
    # not model downward movement).
    cl_interp   = sp_cl   + (df["cl_next_visit"]   - sp_cl)   / safe_k
    ph_interp   = sp_ph   + (df["ph_next_visit"]   - sp_ph)   / safe_k
    turb_interp = sp_turb + (df["turb_next_visit"] - sp_turb) / safe_k

    df["target_cl_tomorrow"] = np.where(
        df["days_to_next_visit"] == 1,
        df["cl_next_visit"],
        np.where(
            df["days_to_next_visit"].notna(),
            cl_interp,
            (sp_cl - cl_decay).clip(lower=0.0),
        ),
    )

    df["target_ph_tomorrow"] = np.where(
        df["days_to_next_visit"] == 1,
        df["ph_next_visit"],
        np.where(
            df["days_to_next_visit"].notna(),
            ph_interp,
            (sp_ph + ph_drift).clip(upper=8.6),
        ),
    )

    df["target_turb_tomorrow"] = np.where(
        df["days_to_next_visit"] == 1,
        df["turb_next_visit"],
        np.where(
            df["days_to_next_visit"].notna(),
            turb_interp,
            (sp_turb + turb_rise).clip(upper=5.0),
        ),
    )

    df = df.dropna(subset=["days_to_next_visit"]).copy()

    df["ph_breach_tomorrow"] = (
        (df["target_ph_tomorrow"] < REG_PH_MIN) | (df["target_ph_tomorrow"] > REG_PH_MAX))
    df["chlorine_breach_tomorrow"] = (
        (df["target_cl_tomorrow"] < REG_CHLORINE_MIN) | (df["target_cl_tomorrow"] > REG_CHLORINE_CLOSE))
    df["chlorine_in_client_range_tomorrow"] = df["target_cl_tomorrow"].between(
        CLIENT_CL_TARGET_MIN, CLIENT_CL_TARGET_MAX)
    df["any_breach_tomorrow"] = df["ph_breach_tomorrow"] | df["chlorine_breach_tomorrow"]
    df["any_breach_next"] = df["any_breach_tomorrow"]

    df_model    = df.dropna(subset=["ph", "free_chlorine"]).copy()
    df_model_wq = df_model.dropna(subset=["target_cl_tomorrow", "target_ph_tomorrow"]).copy()
    log.info("STEP 8  model rows=%d  WQ rows=%d  median gap=%.0fd",
             len(df_model), len(df_model_wq), df_model_wq["days_to_next_visit"].median())
    return df, df_model, df_model_wq


# ===========================================================================
# STEP 9 — feature selection & temporal split
# ===========================================================================

def select_features_and_split(df_model_wq: pd.DataFrame, cfg: PipelineConfig,
                               weather_tmrw_cols: list[str]):
    """Build feature matrix + temporal split. `weather_tmrw_cols` are the
    `w_tmrw_*` column names actually present in df_model_wq (from STEP 6)."""
    all_numeric = F.all_numeric_feature_groups() + weather_tmrw_cols
    # filter present + drop duplicates (defensive against double-inclusion)
    all_numeric = [c for c in all_numeric if c in df_model_wq.columns]
    seen = set(); dedup = []
    for c in all_numeric:
        if c not in seen:
            dedup.append(c); seen.add(c)
    all_numeric = dedup
    null_rates = df_model_wq[all_numeric].isnull().mean()
    high_null = null_rates[null_rates > cfg.feature_null_drop_threshold].index.tolist()
    if high_null:
        log.info("STEP 9  dropping >50%% null features: %s", high_null)
        all_numeric = [c for c in all_numeric if c not in high_null]

    categorical = list(F.CATEGORICAL_FEATURES)

    cutoff = df_model_wq["reading_date"].quantile(cfg.temporal_split_quantile)
    train = df_model_wq[df_model_wq["reading_date"] < cutoff].copy()
    test  = df_model_wq[df_model_wq["reading_date"] >= cutoff].copy()
    log.info("STEP 9  cutoff=%s  train=%d  test=%d", cutoff, len(train), len(test))

    fill_values = {}
    for col in all_numeric:
        m = train[col].median()
        fill_values[col] = float(m) if pd.notna(m) else 0.0

    for dfx in (train, test):
        missing = [c for c in all_numeric if c not in dfx.columns]
        if missing:
            for c in missing:
                dfx[c] = fill_values.get(c, 0.0)
        dfx[all_numeric] = dfx[all_numeric].fillna(fill_values)
        for c in categorical:
            dfx[c] = dfx[c].fillna("unknown")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("num", "passthrough", all_numeric),
        ],
        remainder="drop",
    )
    X_train = preprocessor.fit_transform(train[categorical + all_numeric])
    X_test  = preprocessor.transform(test[categorical + all_numeric])
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical).tolist()
    feature_names = cat_names + all_numeric
    log.info("STEP 9  X_train=%s  X_test=%s  features=%d", X_train.shape, X_test.shape, len(feature_names))
    return {
        "preprocessor": preprocessor,
        "X_train": X_train, "X_test": X_test,
        "train": train, "test": test,
        "all_numeric": all_numeric, "categorical": categorical,
        "feature_names": feature_names,
        "fill_values": fill_values, "cutoff": cutoff,
    }