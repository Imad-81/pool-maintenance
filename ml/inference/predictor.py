"""
Pure chained multi-day forecast engine + the `PredictionService` the FastAPI
backend binds to at startup.

`predict_forward` is a pure function — it accepts the latest master row for a
pool, an as-of date, a live weather-lookup callable, and the loaded
models/preprocessor/config. It never touches disk, never logs files, never
imports globals. That makes it golden-output testable and lets the backend
plug in a SQLite-backed weather provider without monkey-patching.

`PredictionService` loads the *active* model run once (read from
`models/latest.json`) and re-loads when the scheduler hot-swaps. Hot-swap is
graceful: a failed reload logs and keeps the previously loaded artefacts, so
live prediction never dies mid-request.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ml.config import (
    CLIENT_CL_TARGET_MAX,
    CLIENT_CL_TARGET_MIN,
    REG_CHLORINE_CLOSE,
    REG_CHLORINE_IDEAL_MAX,
    REG_CHLORINE_MIN,
    REG_PH_MAX,
    REG_PH_MIN,
    SETPOINT_FREE_CHLORINE,
    SETPOINT_PH,
    SETPOINT_TURBIDITY,
)
from ml.inference.chaining import (
    DEFAULT_HORIZON_DAYS,
    MAX_HORIZON_DAYS,
    UncertaintyBand,
    clamp_horizon,
    warning_band,
)
from ml.inference.visit_recommender import compute_recommended_visit

log = logging.getLogger(__name__)


# Type alias for the weather-lookup callable: given a normalised date and a
# list of weather column names, return {col_name: value} (NaN if missing).
WeatherLookup = Callable[[pd.Timestamp, list[str]], dict]


# ---------------------------------------------------------------------------
# Pure chained forecast
# ---------------------------------------------------------------------------

def _safe_float(val, default: float = 0.0) -> float:
    if val is None or pd.isna(val):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _normalize_ts(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tz is not None:
        t = t.tz_localize(None)
    return t.normalize()


def _build_v7_delta_feature_vector(
    base_row: pd.Series,
    step_date: pd.Timestamp,
    anchor_cl: float,
    anchor_ph: float,
    anchor_turb: float,
    anchor_cya: float,
    cl_history: list[float],
    ph_history: list[float],
    turb_history: list[float],
    step: int,
    feature_names: list[str],
    fill_values: dict[str, float],
) -> pd.DataFrame:
    row_dict = {}

    m = int(step_date.month)
    dow = int(step_date.dayofweek)
    doy = int(step_date.dayofyear)
    row_dict["year"] = int(step_date.year)
    row_dict["month"] = m
    row_dict["day_of_week"] = dow
    row_dict["is_weekend"] = 1 if dow in (5, 6) else 0

    row_dict["free_chlorine_pre_ppm"] = float(anchor_cl)
    row_dict["free_chlorine_post_ppm"] = float(anchor_cl)
    row_dict["chlorine_dosage_boost_ppm"] = 0.0
    row_dict["free_chlorine_estimated_daily_mean_ppm"] = float(anchor_cl)
    row_dict["ph_pre"] = float(anchor_ph)
    row_dict["ph_post"] = float(anchor_ph)
    row_dict["ph_delta"] = 0.0
    row_dict["ph"] = float(anchor_ph)
    row_dict["turbidity_pre"] = float(anchor_turb)
    row_dict["turbidity_post"] = float(anchor_turb)
    row_dict["turbidity_delta"] = 0.0
    row_dict["turbidity"] = float(anchor_turb)
    row_dict["cya_pre_ppm"] = float(anchor_cya)
    row_dict["cya_post_ppm"] = float(anchor_cya)
    row_dict["cya_added_ppm"] = 0.0
    row_dict["cya_cumulative_ppm"] = float(anchor_cya)

    row_dict["water_temperature_c"] = _safe_float(base_row.get("water_temperature"), 24.0)
    row_dict["shock_dosage_ppm"] = 0.0
    row_dict["erodible_active_cl2_added_grams"] = 0.0
    row_dict["clarifier_added_grams"] = 0.0
    row_dict["acid_added_grams"] = 0.0
    row_dict["daily_pump_ph_minus_ml"] = 0.0

    vol = max(10.0, _safe_float(base_row.get("pool_volume_m3") or base_row.get("pool_volume"), 150.0))
    area = max(5.0, _safe_float(base_row.get("pool_surface_m2") or base_row.get("pool_surface_area"), 75.0))
    row_dict["pool_volume"] = vol
    row_dict["pool_surface_area"] = area
    row_dict["specific_surface_ratio"] = area / vol
    row_dict["estimated_mean_depth"] = vol / area

    row_dict["community_pool"] = 1.0 if _safe_float(base_row.get("pool_community"), 1.0) > 0 else 0.0
    row_dict["outdoor_pool"] = 1.0 if _safe_float(base_row.get("pool_outdoor"), 1.0) > 0 else 0.0
    row_dict["oval_pool"] = _safe_float(base_row.get("pool_oval"), 0.0)
    row_dict["overflow_pool"] = _safe_float(base_row.get("pool_overflow"), 0.0)
    row_dict["rectangular_pool_07"] = _safe_float(base_row.get("pool_rectangular_07"), 0.0)
    row_dict["rectangular_pool_0714"] = _safe_float(base_row.get("pool_rectangular_0714"), 0.0)
    row_dict["round_pool"] = _safe_float(base_row.get("pool_round"), 0.0)
    row_dict["skimmer_pool"] = _safe_float(base_row.get("pool_skimmer"), 0.0)
    row_dict["number_of_filters"] = _safe_float(base_row.get("filter_count") or base_row.get("number_of_filters"), 1.0)
    row_dict["number_of_motors"] = _safe_float(base_row.get("motor_count") or base_row.get("number_of_motors"), 1.0)
    row_dict["filter_diameter"] = _safe_float(base_row.get("filter_diameter"), 600.0)
    row_dict["motor_pump_flow_rate"] = _safe_float(base_row.get("motor_flow_rate") or base_row.get("motor_pump_flow_rate"), 15.0)
    row_dict["hypochlorite_pump_flow_rate"] = _safe_float(base_row.get("hypochlorite_pump_flow_rate"), 4.0)
    row_dict["ph_pump_flow_rate"] = _safe_float(base_row.get("ph_pump_flow_rate"), 2.0)
    row_dict["heated_pool"] = _safe_float(base_row.get("pool_heated") or base_row.get("heated_pool"), 0.0)
    row_dict["contaminating_vegetation"] = _safe_float(base_row.get("vegetation_contamination") or base_row.get("contaminating_vegetation"), 0.0)
    row_dict["sunscreen_overuse"] = 1.0 if base_row.get("sunscreen_abuse") in ("SI", "si", 1, True, "1") else 0.0
    row_dict["deck_grass_area"] = _safe_float(base_row.get("deck_grass"), 0.0)
    row_dict["deck_mixed_area"] = _safe_float(base_row.get("deck_mixed"), 0.0)
    row_dict["deck_paved_area"] = _safe_float(base_row.get("deck_paved"), 1.0)

    filt_hours = _safe_float(base_row.get("daily_filtration_hours"), 8.0)
    hypo_hrs = _safe_float(base_row.get("hypochlorite_dosing_hours"), 0.0) if step == 1 else 0.0
    hypo_pct = _safe_float(base_row.get("hypochlorite_dosing_pct"), 0.0) if step == 1 else 0.0
    row_dict["daily_filtration_hours"] = filt_hours
    row_dict["hypo_dosing_hours"] = hypo_hrs
    row_dict["hypo_dosing_percentage"] = hypo_pct
    row_dict["ph_dosing_hours"] = _safe_float(base_row.get("ph_dosing_hours"), 0.0) if step == 1 else 0.0
    row_dict["ph_dosing_percentage"] = _safe_float(base_row.get("ph_dosing_pct"), 0.0) if step == 1 else 0.0
    pump_flow = _safe_float(base_row.get("hypochlorite_pump_flow_rate"), 4.0)
    delivered_cl2 = (hypo_hrs * (hypo_pct / 100.0) * pump_flow * 130.0) / vol if step == 1 else 0.0
    row_dict["daily_pump_cl2_delivered_ppm"] = float(np.clip(delivered_cl2, 0.0, 0.40))

    rad = _safe_float(base_row.get("w_solar_radiation") or base_row.get("solar_radiation_mj"), 25.0)
    row_dict["solar_radiation_mj"] = rad
    row_dict["forecast_solar_radiation_mj"] = _safe_float(base_row.get("w_tmrw_solar_radiation") or base_row.get("forecast_solar_radiation_mj"), rad)
    row_dict["temperature_ambient_mean_c"] = _safe_float(base_row.get("w_temp_mean") or base_row.get("temperature_ambient_mean_c"), 25.0)
    row_dict["forecast_temperature_mean_c"] = _safe_float(base_row.get("w_tmrw_temp_mean") or base_row.get("forecast_temperature_mean_c"), 25.0)
    row_dict["temperature_ambient_max_c"] = _safe_float(base_row.get("w_temp_max") or base_row.get("temperature_ambient_max_c"), 30.0)
    row_dict["forecast_temperature_max_c"] = _safe_float(base_row.get("w_tmrw_temp_max") or base_row.get("forecast_temperature_max_c"), 30.0)
    row_dict["precipitation_mm"] = _safe_float(base_row.get("w_precipitation_mm") or base_row.get("precipitation_mm"), 0.0)
    row_dict["forecast_precipitation_mm"] = _safe_float(base_row.get("w_tmrw_precipitation_mm") or base_row.get("forecast_precipitation_mm"), 0.0)
    row_dict["sunshine_duration_hrs"] = _safe_float(base_row.get("w_sunshine_hours") or base_row.get("sunshine_duration_hrs"), 10.0)
    row_dict["forecast_sunshine_duration_hrs"] = _safe_float(base_row.get("w_tmrw_sunshine_hours") or base_row.get("forecast_sunshine_duration_hrs"), 10.0)
    row_dict["wind_speed_max_kmh"] = _safe_float(base_row.get("w_wind_max_kmh") or base_row.get("wind_speed_max_kmh"), 15.0)
    row_dict["forecast_wind_speed_max_kmh"] = _safe_float(base_row.get("w_tmrw_wind_max_kmh") or base_row.get("forecast_wind_speed_max_kmh"), 15.0)
    row_dict["daylight_duration_hrs"] = 14.0
    row_dict["uv_index_max"] = _safe_float(base_row.get("w_uv_max") or base_row.get("uv_index_max"), 7.0)
    row_dict["et0_evapotranspiration"] = _safe_float(base_row.get("w_et0") or base_row.get("et0_evapotranspiration"), 5.0)
    row_dict["forecast_et0_evapotranspiration"] = _safe_float(base_row.get("w_tmrw_et0") or base_row.get("forecast_et0_evapotranspiration"), 5.0)

    hocl_fraction = 1.0 / (1.0 + 10.0 ** (anchor_ph - 7.53))
    turnover = float(np.clip((15.0 * filt_hours) / vol, 0.0, 10.0))
    spec_surface = area / vol
    is_outdoor = row_dict["outdoor_pool"]
    is_community = row_dict["community_pool"]
    k_photo = 0.025 * (rad / 20.0) * float(np.clip(spec_surface, 0.5, 3.0)) * is_outdoor
    k_temp = 0.015 * float(np.clip(row_dict["water_temperature_c"] / 25.0, 0.5, 2.0))
    k_turb = 0.020 * float(np.clip(anchor_turb, 0.1, 5.0))
    k_cya = 0.015 * float(np.clip(anchor_cya / 50.0, 0.0, 0.8))
    k_bather = 0.025 * (1.0 if dow >= 5 else 0.0) * is_community
    decay_k = float(np.clip(0.04 + k_photo + k_temp + k_turb + k_bather - k_cya, 0.02, 0.80))

    row_dict["theoretical_decay_k"] = decay_k
    row_dict["theoretical_retained_chlorine"] = float(anchor_cl * np.exp(-decay_k))
    row_dict["active_hocl_fraction"] = hocl_fraction
    row_dict["active_hocl_pre_ppm"] = float(anchor_cl * hocl_fraction)
    row_dict["active_hocl_post_ppm"] = float(anchor_cl * hocl_fraction)
    row_dict["active_hocl_ppm"] = float(anchor_cl * hocl_fraction)
    row_dict["daily_turnover_ratio"] = turnover
    row_dict["hocl_demand_proxy"] = float(hocl_fraction * anchor_cl * turnover * (1.0 + anchor_turb))
    row_dict["cl_decay_potential"] = float(anchor_cl * decay_k)
    row_dict["chlorine_post_treatment_theoretical"] = float(anchor_cl)
    row_dict["clarifier_dosed_today"] = 0.0
    row_dict["clarifier_dose_per_vol"] = 0.0
    row_dict["acid_dosed_today"] = 0.0
    row_dict["acid_dose_per_vol"] = 0.0

    row_dict["free_chlorine_post_lag1"] = float(cl_history[-1])
    row_dict["free_chlorine_post_lag2"] = float(cl_history[-2] if len(cl_history) >= 2 else cl_history[-1])
    row_dict["free_chlorine_rolling_mean_3"] = float(np.mean(cl_history[-3:]))
    row_dict["free_chlorine_pre_lag1"] = float(cl_history[-1])
    row_dict["ph_post_lag1"] = float(ph_history[-1])
    row_dict["turbidity_post_lag1"] = float(turb_history[-1])
    row_dict["cl_diff_lag1"] = float(cl_history[-1] - (cl_history[-2] if len(cl_history) >= 2 else cl_history[-1]))
    row_dict["cl_diff_rolling"] = float(anchor_cl - np.mean(cl_history[-3:]))
    row_dict["cl_decay_slope_recent"] = float((anchor_cl - cl_history[0]) / max(1, len(cl_history)))

    row_dict["pool_cl_hist_mean"] = _safe_float(base_row.get("pool_cl_hist_mean"), 2.2)
    row_dict["pool_cl_hist_std"] = _safe_float(base_row.get("pool_cl_hist_std"), 0.8)
    row_dict["pool_cl_hist_min"] = _safe_float(base_row.get("pool_cl_hist_min"), 0.5)
    row_dict["pool_cl_hist_max"] = _safe_float(base_row.get("pool_cl_hist_max"), 4.5)
    row_dict["pool_cl_hist_q25"] = _safe_float(base_row.get("pool_cl_hist_q25"), 1.5)
    row_dict["pool_cl_hist_q75"] = _safe_float(base_row.get("pool_cl_hist_q75"), 2.8)
    row_dict["pool_visit_count"] = _safe_float(base_row.get("pool_visit_number"), 50.0)

    row_dict["window_solar_rad_sum_mj"] = rad * step
    row_dict["window_solar_rad_mean_mj"] = rad
    row_dict["window_temp_mean_c"] = row_dict["temperature_ambient_mean_c"]
    row_dict["window_temp_max_c"] = row_dict["temperature_ambient_max_c"]
    row_dict["window_precip_sum_mm"] = row_dict["precipitation_mm"]
    row_dict["window_sunshine_hours_sum"] = row_dict["sunshine_duration_hrs"] * step
    row_dict["window_wind_max_kmh"] = row_dict["wind_speed_max_kmh"]
    row_dict["window_et0_sum_mm"] = row_dict["et0_evapotranspiration"] * step

    row_dict["rad_per_volume"] = rad / vol
    row_dict["pump_flow_per_vol"] = 15.0 / vol
    row_dict["bather_surge_index"] = 1.5 if (dow >= 5 and is_community) else 1.0
    row_dict["weekend_exposure_ratio"] = 1.0 if dow >= 5 else 0.0
    row_dict["cl_times_turnover"] = float(anchor_cl * turnover)
    row_dict["evap_stress_index"] = float((rad * row_dict["wind_speed_max_kmh"]) / 100.0)

    row_dict["day_of_year"] = doy
    row_dict["day_of_year_sin"] = float(np.sin(2 * np.pi * doy / 365.25))
    row_dict["day_of_year_cos"] = float(np.cos(2 * np.pi * doy / 365.25))
    row_dict["month_sin"] = float(np.sin(2 * np.pi * m / 12.0))
    row_dict["month_cos"] = float(np.cos(2 * np.pi * m / 12.0))

    feat_series = {col: row_dict.get(col, fill_values.get(col, 0.0)) for col in feature_names}
    return pd.DataFrame([feat_series])[feature_names]


def predict_forward(
    *,
    pool_id: str,
    latest_row: pd.Series,
    as_of_date: pd.Timestamp,
    weather_lookup: WeatherLookup,
    models: dict,                # {"chlorine": ..., "ph": ..., "turbidity": ...}
    preprocessor,
    config: dict,                # the inference_config_v6.json / v7 payload
    horizon_days: Optional[int] = None,
) -> dict:
    """Chain 1-day-forward predictions from the last visit to `as_of_date +
    horizon_days`.

    Returns a dict with pool_id, last_visit_date, days_since_visit, last
    readings, a `forecast` DataFrame (one row per day with per-day
    UncertaintyBand), the today/tomorrow sections highlighted, and an
    overall `visit_needed` flag.
    """
    as_of_date = _normalize_ts(as_of_date)
    horizon = clamp_horizon(horizon_days)

    last_visit_date = _normalize_ts(latest_row["reading_date"])
    days_since = int((as_of_date - last_visit_date).days)
    if days_since < 0:
        return {"error": f"as_of_date {as_of_date.date()} is before last visit {last_visit_date.date()}"}

    all_numeric = list(config.get("all_numeric_features", []))
    categorical = list(config.get("categorical_features", []))
    fill_values = {k: float(v) for k, v in config.get("fill_values", {}).items()}
    today_wx = list(config.get("weather_current_features", []))
    tmrw_wx = list(config.get("weather_tomorrow_features", []))
    is_delta = (config.get("model_family") == "lightgbm_delta") or (preprocessor is None)

    # Configurable post-treatment setpoint
    sp = config.get("treatment_setpoint", {})
    sp_cl   = _safe_float(sp.get("free_chlorine"), SETPOINT_FREE_CHLORINE)
    sp_ph   = _safe_float(sp.get("ph"),            SETPOINT_PH)
    sp_turb = _safe_float(sp.get("turbidity"),     SETPOINT_TURBIDITY)

    base = latest_row.copy()
    for col in all_numeric:
        if col not in base.index or pd.isna(base.get(col, np.nan)):
            base[col] = fill_values.get(col, 0.0)
    for col in categorical:
        if col not in base.index or pd.isna(base.get(col, np.nan)):
            base[col] = "unknown"

    cur_cl   = _safe_float(latest_row.get("free_chlorine"), fill_values.get("free_chlorine", 2.0))
    cur_ph   = _safe_float(latest_row.get("ph"),            fill_values.get("ph",            7.4))
    cur_turb = _safe_float(latest_row.get("turbidity"),     fill_values.get("turbidity",     0.5))

    row = base.copy()
    total_steps = days_since + horizon
    prev_cl, prev_ph, prev_turb = cur_cl, cur_ph, cur_turb
    cl_history = [cur_cl, cur_cl, cur_cl]
    ph_history = [cur_ph, cur_ph, cur_ph]
    turb_history = [cur_turb, cur_turb, cur_turb]
    forecast_rows: list[dict] = []

    model_cl, model_ph, model_turb = models["chlorine"], models["ph"], models["turbidity"]

    sim_horizon = max(horizon, 7)
    for step in range(1, total_steps + 1):
        if step > (days_since + sim_horizon):
            break
        step_date = last_visit_date + pd.Timedelta(days=step)

        # temporal features
        row["visit_month"] = int(step_date.month)
        row["visit_day_of_week"] = int(step_date.dayofweek)
        row["visit_is_summer"] = int(step_date.month in (6, 7, 8, 9))
        row["visit_year"] = int(step_date.year)

        # weather injection — today's + tomorrow's
        row = _inject_weather(row, weather_lookup, step_date, today_wx, tmrw_wx, last_visit_date=last_visit_date)

        anchor_cl   = sp_cl   if (step == 1 and not is_delta) else cur_cl
        anchor_ph   = sp_ph   if (step == 1 and not is_delta) else cur_ph
        anchor_turb = sp_turb if (step == 1 and not is_delta) else cur_turb

        if is_delta:
            anchor_cya = _safe_float(latest_row.get("cya_post_ppm") or latest_row.get("cya_ppm"), 30.0)
            feat_df = _build_v7_delta_feature_vector(
                row, step_date, anchor_cl, anchor_ph, anchor_turb, anchor_cya,
                cl_history, ph_history, turb_history, step, all_numeric, fill_values
            )
            raw_delta_cl = float(model_cl.predict(feat_df)[0])
            raw_delta_ph = float(model_ph.predict(feat_df)[0])
            raw_delta_turb = float(model_turb.predict(feat_df)[0])

            decay_k = float(feat_df.get("theoretical_decay_k", [0.15])[0])
            pump_delivered = float(feat_df.get("daily_pump_cl2_delivered_ppm", [0.0])[0])

            # Physical kinetics ceiling: chlorine cannot spontaneously increase without chemical addition
            kinetic_cl_ceiling = anchor_cl * np.exp(-decay_k) + pump_delivered
            raw_pred_cl = anchor_cl + raw_delta_cl

            if pump_delivered <= 0.01:
                pred_cl = float(np.clip(min(raw_pred_cl, kinetic_cl_ceiling), 0.05, 5.0))
            else:
                pred_cl = float(np.clip(raw_pred_cl, 0.05, 5.0))

            pred_ph = float(np.clip(anchor_ph + raw_delta_ph, 6.8, 8.8))
            pred_turb = float(np.clip(anchor_turb + raw_delta_turb, 0.05, 5.0))

            cl_breach = pred_cl < REG_CHLORINE_MIN or pred_cl > REG_CHLORINE_CLOSE
            ph_breach = pred_ph < REG_PH_MIN or pred_ph > REG_PH_MAX
            urgency, status = _classify(pred_cl, pred_ph, cl_breach, ph_breach)

            is_today    = (step_date == as_of_date)
            is_tomorrow = (step_date == as_of_date + pd.Timedelta(days=1))
            day_offset_from_today = int((step_date - as_of_date).days)

            band = warning_band(day_offset_from_today, pred_cl, pred_ph, pred_turb,
                                step_error_cl=0.084, step_error_ph=0.012, step_error_turb=0.015)
        else:
            # recompute chemistry-dependent features on the previous step's state
            row = _recompute_features(row, cur_cl, cur_ph, cur_turb,
                                      step=step, prev_cl=prev_cl, prev_ph=prev_ph,
                                      sp_cl=sp_cl, sp_ph=sp_ph, sp_turb=sp_turb)

            # build the preprocessor frame
            feat = pd.DataFrame([row])
            for col in all_numeric:
                if col not in feat.columns:
                    feat[col] = fill_values.get(col, 0.0)
                feat[col] = pd.to_numeric(feat[col], errors="coerce").fillna(fill_values.get(col, 0.0))
            for col in categorical:
                if col not in feat.columns:
                    feat[col] = "unknown"
            X = preprocessor.transform(feat[categorical + all_numeric])
            raw_cl   = max(0.0, float(model_cl.predict(X)[0]))
            raw_ph   = float(model_ph.predict(X)[0])
            raw_turb = max(0.0, float(model_turb.predict(X)[0]))

            # --- Physical Kinetics Rate Integration Engine ---
            cl_added = float(row.get("last_total_chlorine_applied", 0) or 0) > 0 or float(row.get("chlorine_dose_per_m3", 0) or 0) > 0
            if cl_added:
                pred_cl = raw_cl
            else:
                solar_rad = float(row.get("w_solar_radiation", 25.0) or 25.0)
                decay_k = 0.15 + 0.003 * max(0.0, solar_rad - 15.0)
                cl_kinetic = anchor_cl * np.exp(-decay_k / 3.0)
                pred_cl = max(0.0, min(raw_cl, cl_kinetic))

            acid_added = float(row.get("total_ph_minus_product", 0) or 0) > 0 or float(row.get("ph_minus_dose_per_m3", 0) or 0) > 0
            if acid_added:
                pred_ph = raw_ph
            else:
                temp_max = float(row.get("w_temp_max", 30.0) or 30.0)
                daily_ph_drift = 0.035 + 0.0015 * max(0.0, temp_max - 25.0)
                ph_kinetic = anchor_ph + daily_ph_drift
                pred_ph = min(8.6, max(raw_ph, ph_kinetic))

            wind_max = float(row.get("w_wind_max_kmh", 15.0) or 15.0)
            daily_turb_rise = 0.045 + 0.002 * max(0.0, wind_max - 10.0)
            turb_kinetic = anchor_turb + daily_turb_rise
            pred_turb = min(5.0, max(raw_turb, turb_kinetic))

            cl_breach = pred_cl < REG_CHLORINE_MIN or pred_cl > REG_CHLORINE_CLOSE
            ph_breach = pred_ph < REG_PH_MIN or pred_ph > REG_PH_MAX
            urgency, status = _classify(pred_cl, pred_ph, cl_breach, ph_breach)

            is_today    = (step_date == as_of_date)
            is_tomorrow = (step_date == as_of_date + pd.Timedelta(days=1))
            day_offset_from_today = int((step_date - as_of_date).days)

            band = warning_band(day_offset_from_today, pred_cl, pred_ph, pred_turb)

        day_label = step_date.strftime("%a")
        if is_today:
            day_label += " ◀ TODAY"
        elif is_tomorrow:
            day_label += " ◀ TOMORROW"

        forecast_rows.append({
            "date":            step_date.date(),
            "day":             day_label,
            "days_from_visit": step,
            "day_offset_from_today": day_offset_from_today,
            "predicted_cl":    round(pred_cl, 3),
            "predicted_ph":    round(pred_ph, 3),
            "predicted_turb":  round(pred_turb, 3),
            "cl_breach":       bool(cl_breach),
            "ph_breach":       bool(ph_breach),
            "urgency":         urgency,
            "status":          status,
            "is_today":        bool(is_today),
            "is_tomorrow":     bool(is_tomorrow),
            "uncertainty_band": band,
        })

        prev_cl, prev_ph, prev_turb = cur_cl, cur_ph, cur_turb
        cur_cl, cur_ph, cur_turb = pred_cl, pred_ph, pred_turb
        cl_history.append(pred_cl)
        ph_history.append(pred_ph)
        turb_history.append(pred_turb)

    fc_full = pd.DataFrame(forecast_rows)
    if fc_full.empty:
        return {"pool_id": pool_id, "error": "no forecast produced — check days_since/inputs"}

    recommended_visit = compute_recommended_visit(fc_full, as_of_date, last_visit_date)

    # Slice returned forecast table to the requested horizon window
    fc = fc_full[fc_full["day_offset_from_today"] <= horizon].copy()
    if fc.empty:
        fc = fc_full

    dashboard = fc[fc["is_today"] | fc["is_tomorrow"]]
    visit_needed = bool(
        (dashboard["cl_breach"].any() if len(dashboard) else False) or
        (dashboard["ph_breach"].any() if len(dashboard) else False) or
        ((dashboard["urgency"] == "Advised").any() if len(dashboard) else False)
    )

    return {
        "pool_id":           pool_id,
        "last_visit_date":   last_visit_date.date(),
        "days_since_visit":  days_since,
        "last_readings":     {
            "free_chlorine": round(_safe_float(latest_row.get("free_chlorine"), 0.0), 3),
            "ph":            round(_safe_float(latest_row.get("ph"),            0.0), 3),
            "turbidity":     round(_safe_float(latest_row.get("turbidity"),     0.0), 3),
        },
        "forecast":          fc,
        "today_forecast":    fc[fc["is_today"]].to_dict("records"),
        "tomorrow_forecast": fc[fc["is_tomorrow"]].to_dict("records"),
        "visit_needed":      visit_needed,
        "recommended_visit": recommended_visit,
    }


# ---------------------------------------------------------------------------
# Feature recomputation (mirrors inference.py `_recompute_features` + the V6
# pipeline's headroom/trend logic from ml/features.py)
# ---------------------------------------------------------------------------

def _recompute_features(row, pred_cl, pred_ph, pred_turb, step, prev_cl, prev_ph,
                        sp_cl=SETPOINT_FREE_CHLORINE, sp_ph=SETPOINT_PH, sp_turb=SETPOINT_TURBIDITY):
    row = row.copy()

    # current state
    row["free_chlorine"] = pred_cl
    row["ph"]            = pred_ph
    row["turbidity"]     = max(0.0, pred_turb)

    # Reset manual dosing product inputs for post-visit days (step > 1)
    # on unvisited days no technician is present to add manual chemicals
    if step > 1:
        row["last_total_chlorine_applied"] = 0.0
        row["total_ph_minus_product"] = 0.0
        row["chlorine_dose_per_m3"] = 0.0
        row["ph_minus_dose_per_m3"] = 0.0

    # lags
    row["chlorine_lag2"] = row.get("chlorine_lag1", pred_cl)
    row["chlorine_lag1"] = prev_cl
    row["ph_lag2"]       = row.get("ph_lag1", pred_ph)
    row["ph_lag1"]       = prev_ph
    row["turbidity_lag1"] = row.get("turbidity_lag1", pred_turb)
    row["turbidity_lag2"] = row.get("turbidity_lag2", pred_turb)

    # rolling (3-step window)
    vals_cl   = [pred_cl,   prev_cl,   row.get("chlorine_lag2", pred_cl)]
    vals_ph   = [pred_ph,   prev_ph,   row.get("ph_lag2", pred_ph)]
    vals_turb = [pred_turb, row.get("turbidity_lag1", pred_turb), row.get("turbidity_lag2", pred_turb)]
    row["chlorine_roll3_mean"]  = float(np.mean(vals_cl))
    row["chlorine_roll3_std"]   = float(np.std(vals_cl))
    row["ph_roll3_mean"]        = float(np.mean(vals_ph))
    row["ph_roll3_std"]         = float(np.std(vals_ph))
    row["turbidity_roll3_mean"] = float(np.mean(vals_turb))

    # temporal
    row["days_since_last_visit"] = step

    # headroom
    from ml.features import add_headroom_features
    single = pd.DataFrame([row])
    single = add_headroom_features(single)
    for c in ["chlorine_headroom_low", "chlorine_headroom_high",
              "ph_headroom_low", "ph_headroom_high",
              "turbidity_headroom", "min_headroom",
              "cl_below_client_target", "cl_above_client_target"]:
        row[c] = float(single[c].iloc[0])

    # trends
    row["chlorine_trend"]        = pred_cl - prev_cl
    row["ph_trend"]              = pred_ph - prev_ph
    row["turbidity_trend"]       = pred_turb - row.get("turbidity_lag1", pred_turb)
    safe_step = step if step else np.nan
    row["chlorine_rate_per_day"] = (pred_cl - prev_cl) / safe_step if safe_step else 0.0
    row["ph_rate_per_day"]       = (pred_ph - prev_ph) / safe_step if safe_step else 0.0
    row["turbidity_rate_per_day"] = row["turbidity_trend"] / safe_step if safe_step else 0.0

    # effectiveness index
    from ml.features import cl_effectiveness
    row["cl_effectiveness_index"] = cl_effectiveness(pred_cl, pred_ph)

    # post-treatment setpoint features (constant setpoint; deltas/rates vs the
    # running predicted state and `step`)
    row["setpoint_free_chlorine"]  = float(sp_cl)
    row["setpoint_ph"]             = float(sp_ph)
    row["setpoint_turbidity"]      = float(sp_turb)
    row["cl_degradation_from_setpoint"]   = float(sp_cl)   - pred_cl
    row["ph_drift_from_setpoint"]         = pred_ph        - float(sp_ph)
    row["turb_accumulation_from_setpoint"] = pred_turb     - float(sp_turb)
    safe_gap = step if step else np.nan
    row["cl_degradation_rate_from_setpoint"]    = (float(sp_cl) - pred_cl) / safe_gap if safe_gap else 0.0
    row["ph_drift_rate_from_setpoint"]          = (pred_ph - float(sp_ph)) / safe_gap if safe_gap else 0.0
    row["turb_accumulation_rate_from_setpoint"] = (pred_turb - float(sp_turb)) / safe_gap if safe_gap else 0.0

    return row


# ---------------------------------------------------------------------------
# Weather injection
# ---------------------------------------------------------------------------

def _inject_weather(row, weather_lookup, step_date, today_wx, tmrw_wx, last_visit_date=None):
    today = _normalize_ts(step_date)
    today_vals = weather_lookup(today, today_wx)
    for col, v in today_vals.items():
        row[col] = v
    # tomorrow block: lookup weather at (step_date+1), rename to w_tmrw_*
    tomorrow = today + pd.Timedelta(days=1)
    tmrw_src = [c.replace("w_tmrw_", "w_") for c in tmrw_wx]
    tmrw_vals = weather_lookup(tomorrow, tmrw_src)
    for t_col, s_col in zip(tmrw_wx, tmrw_src):
        row[t_col] = tmrw_vals.get(s_col, np.nan)

    # Cumulative weather since last visit
    if last_visit_date is not None:
        lvl = _normalize_ts(last_visit_date)
        num_days = min(max(1, int((today - lvl).days)), 30)
        uv_vals, solar_vals, precip_vals, temp_vals = [], [], [], []
        for d_off in range(1, num_days + 1):
            d_curr = lvl + pd.Timedelta(days=d_off)
            w_curr = weather_lookup(d_curr, ["w_uv_max", "w_solar_radiation", "w_precipitation_mm", "w_temp_mean"])
            if pd.notna(w_curr.get("w_uv_max")): uv_vals.append(w_curr["w_uv_max"])
            if pd.notna(w_curr.get("w_solar_radiation")): solar_vals.append(w_curr["w_solar_radiation"])
            if pd.notna(w_curr.get("w_precipitation_mm")): precip_vals.append(w_curr["w_precipitation_mm"])
            if pd.notna(w_curr.get("w_temp_mean")): temp_vals.append(w_curr["w_temp_mean"])
        row["w_uv_sum_since"] = float(np.sum(uv_vals)) if uv_vals else today_vals.get("w_uv_max", 0.0)
        row["w_solar_sum_since"] = float(np.sum(solar_vals)) if solar_vals else today_vals.get("w_solar_radiation", 0.0)
        row["w_precip_sum_since"] = float(np.sum(precip_vals)) if precip_vals else today_vals.get("w_precipitation_mm", 0.0)
        row["w_temp_mean_since"] = float(np.mean(temp_vals)) if temp_vals else today_vals.get("w_temp_mean", 25.0)

    return row


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _classify(pred_cl, pred_ph, cl_breach, ph_breach):
    if cl_breach or ph_breach:
        return "URGENT", "🚨 Regulatory breach — URGENT visit"
    if pred_cl < CLIENT_CL_TARGET_MIN:
        return "Advised", "⚠️  Cl below client target — visit advised"
    if pred_cl > REG_CHLORINE_IDEAL_MAX:
        return "Monitor", "⚠️  Cl above optimal range — monitor"
    return "Routine", "✅ OK"


# ---------------------------------------------------------------------------
# PredictionService — loaded once at backend startup, hot-swappable
# ---------------------------------------------------------------------------

class PredictionService:
    """Loads the active V6 run + creates an `Optimiser` on demand.

    Thread-safe: a single `reload()` flips `_state` atomically under a lock,
    and live `forecast`/`optimise` reads use snapshots so concurrent requests
    during a retrain promotion are safe.
    """

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self._lock = threading.RLock()
        self._state: Optional[dict] = None

    # --- public API --------------------------------------------------------

    def is_loaded(self) -> bool:
        with self._lock:
            return self._state is not None

    def load(self) -> None:
        with self._lock:
            self._state = self._load_active_run()

    def reload(self) -> None:
        """Called by the scheduler after a retrain promotion. On failure,
        keep the previously loaded state and log — never kill live serving."""
        try:
            new_state = self._load_active_run()
        except Exception as e:  # pragma: no cover
            log.error("reload FAILED — keeping previous models: %s", e)
            return
        with self._lock:
            self._state = new_state
        log.info("reloaded active run %s", new_state.get("run_id"))

    def forecast(self, pool_id: str, latest_row: pd.Series,
                 as_of_date: pd.Timestamp, weather_lookup: WeatherLookup,
                 horizon_days: Optional[int] = None) -> dict:
        with self._lock:
            state = self._state
        if state is None:
            raise RuntimeError("PredictionService not loaded")
        return predict_forward(
            pool_id=pool_id, latest_row=latest_row, as_of_date=as_of_date,
            weather_lookup=weather_lookup, models=state["models"],
            preprocessor=state["preprocessor"], config=state["config"],
            horizon_days=horizon_days,
        )

    def optimise(self, pool_id: str, latest_row: pd.Series):
        from ml.inference.optimiser import Optimiser
        with self._lock:
            state = self._state
        if state is None:
            raise RuntimeError("PredictionService not loaded")
        opt = Optimiser(
            state["cfg"], state["models"]["chlorine"], state["models"]["ph"],
            state["preprocessor"], state["all_numeric_features"],
            state["categorical_features"], state["fill_values"],
            is_delta=(state.get("model_family") == "lightgbm_delta"),
        )
        return opt.optimise(pool_id, latest_row)

    def status(self) -> dict:
        with self._lock:
            state = self._state
        if state is None:
            return {"loaded": False}
        return {
            "loaded": True,
            "run_id": state["run_id"],
            "model_family": state.get("model_family", "xgboost_direct"),
            "feature_schema": state["config"].get("feature_schema"),
            "metrics": state["config"].get("metrics"),
        }

    # --- internals ---------------------------------------------------------

    def _load_active_run(self) -> dict:
        from ml.config import DEFAULT_CONFIG
        pointer_path = self.models_dir / "latest.json"
        if not pointer_path.exists():
            # fall back to the legacy flat layout in models/ if present
            log.warning("no latest.json — falling back to legacy flat models")
            return self._load_legacy_flat(DEFAULT_CONFIG)

        pointer = json.loads(pointer_path.read_text())
        run_id = pointer["active_run_id"]
        run_dir = self.models_dir / run_id
        if not run_dir.exists():
            log.error("active run dir %s missing — falling back to legacy", run_dir)
            return self._load_legacy_flat(DEFAULT_CONFIG)

        return self._load_from_dir(run_dir, DEFAULT_CONFIG, run_id)

    def _load_from_dir(self, run_dir: Path, cfg, run_id: str) -> dict:
        cfg_files = sorted(list(run_dir.glob("inference_config*.json")))
        if not cfg_files:
            raise FileNotFoundError(f"No inference_config found in {run_dir}")
        with open(cfg_files[0]) as f:
            config = json.load(f)

        preprocessor = None
        prep_path = run_dir / "preprocessor_v6.pkl"
        if prep_path.exists():
            with open(prep_path, "rb") as f:
                preprocessor = pickle.load(f)

        models = {}
        lgb_cl = run_dir / "best_daily_chlorine_model_LightGBM_Delta.pkl"
        if lgb_cl.exists():
            import joblib
            models["chlorine"] = joblib.load(lgb_cl)
            models["ph"] = joblib.load(run_dir / "best_daily_ph_model_LightGBM_Delta.pkl")
            models["turbidity"] = joblib.load(run_dir / "best_daily_turbidity_model_LightGBM_Delta.pkl")
            model_family = "lightgbm_delta"
        else:
            import xgboost as xgb
            for name in ("chlorine", "ph", "turbidity"):
                m = xgb.XGBRegressor()
                m.load_model(run_dir / f"xgb_{name}_next.json")
                models[name] = m
            model_family = "xgboost_direct"

        log.info("loaded run %s from %s (family: %s)", run_id, run_dir, model_family)
        return {
            "run_id": run_id,
            "model_family": model_family,
            "config": config,
            "models": models,
            "preprocessor": preprocessor,
            "all_numeric_features": list(config.get("all_numeric_features", [])),
            "categorical_features": list(config.get("categorical_features", [])),
            "fill_values": {k: float(v) for k, v in config.get("fill_values", {}).items()},
            "cfg": cfg,
        }

    def _load_legacy_flat(self, cfg) -> dict:
        """Read the existing `models/xgb_*_next.json` + `preprocessor_v6.pkl`
        + `inference_config_v6.json` layout shipped before the refactor so the
        backend can boot against the already-trained artifacts."""
        import xgboost as xgb
        d = self.models_dir
        with open(d / "inference_config_v6.json") as f:
            config = json.load(f)
        with open(d / "preprocessor_v6.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        models = {}
        for name in ("chlorine", "ph", "turbidity"):
            m = xgb.XGBRegressor()
            m.load_model(d / f"xgb_{name}_next.json")
            models[name] = m
        log.info("loaded LEGACY flat model layout")
        return {
            "run_id": config.get("pipeline_version", "legacy"),
            "config": config,
            "models": models,
            "preprocessor": preprocessor,
            "all_numeric_features": list(config["all_numeric_features"]),
            "categorical_features": list(config["categorical_features"]),
            "fill_values": {k: float(v) for k, v in config["fill_values"].items()},
            "cfg": cfg,
        }