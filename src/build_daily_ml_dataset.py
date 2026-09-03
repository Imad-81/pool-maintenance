#!/usr/bin/env python3
"""
ML-Ready Daily Pool Dataset Builder.

Takes the continuous daily time series with dual pre/post states (156K rows) and enriches it
through 8 sequential passes into a complete, physically rigorous ML-ready feature set (~110 features).

Transition Modeling:
- Today's State: Refreshed Departure State (C_post, pH_post, Turb_post, CYA_post, pump setpoints, weather)
- Tomorrow's Target: Arrival Pre-Treatment State (C_pre_t+1, pH_pre_t+1)
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.build_ml_dataset import (
    load_raw_data,
    disaggregate_tables,
    impute_pool_profiles,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pass 1: Pool Profile Join
# ─────────────────────────────────────────────────────────────────────────────
def pass1_pool_profile_join(df: pd.DataFrame, df_profile: pd.DataFrame) -> pd.DataFrame:
    """Joins 20 static pool profile features onto every daily row."""
    logger.info("Pass 1: Joining pool profile features...")

    profile_cols = [
        'pool_clean',
        'oval_pool', 'overflow_pool', 'rectangular_pool_07',
        'rectangular_pool_0714', 'round_pool', 'skimmer_pool',
        'number_of_filters', 'number_of_motors', 'filter_diameter',
        'motor_pump_flow_rate', 'hypochlorite_pump_flow_rate', 'ph_pump_flow_rate',
        'heated_pool', 'contaminating_vegetation', 'sunscreen_overuse',
        'deck_grass_area', 'deck_mixed_area', 'deck_paved_area',
    ]

    available = [c for c in profile_cols if c in df_profile.columns]
    prof_subset = df_profile[available].copy()

    bool_cols = ['oval_pool', 'overflow_pool', 'rectangular_pool_07',
                 'rectangular_pool_0714', 'round_pool', 'skimmer_pool',
                 'heated_pool', 'contaminating_vegetation', 'sunscreen_overuse']
    for c in bool_cols:
        if c in prof_subset.columns:
            prof_subset[c] = prof_subset[c].fillna(0).astype(int)

    numeric_profile = ['number_of_filters', 'number_of_motors', 'filter_diameter',
                       'motor_pump_flow_rate', 'hypochlorite_pump_flow_rate',
                       'ph_pump_flow_rate', 'deck_grass_area', 'deck_mixed_area',
                       'deck_paved_area']
    for c in numeric_profile:
        if c in prof_subset.columns:
            prof_subset[c] = prof_subset[c].fillna(prof_subset[c].median())

    df = df.merge(prof_subset, on='pool_clean', how='left', suffixes=('', '_profile'))

    dup_cols = [c for c in df.columns if c.endswith('_profile')]
    if dup_cols:
        df = df.drop(columns=dup_cols)

    df['estimated_mean_depth'] = (df['pool_volume'] / df['pool_surface_area']).round(2)
    df['specific_surface_ratio'] = (df['pool_surface_area'] / df['pool_volume']).round(4)

    logger.info(f"  Added profile features. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Pass 2: Operational Setpoints As-Of Join
# ─────────────────────────────────────────────────────────────────────────────
def pass2_operational_setpoints(df: pd.DataFrame, df_ops: pd.DataFrame) -> pd.DataFrame:
    """Joins most-recent operational settings for each pool-date via as-of merge."""
    logger.info("Pass 2: As-of joining operational setpoints...")

    ops_cols = ['daily_filtration_hours', 'hypo_dosing_hours',
                'hypo_dosing_percentage', 'ph_dosing_hours', 'ph_dosing_percentage']

    df_ops = df_ops.copy()
    df_ops['merge_dt'] = pd.to_datetime(df_ops['date_dt'])
    df_ops = df_ops.sort_values(['pool_clean', 'merge_dt'])

    ops_pivot = df_ops.groupby(['pool_clean', 'merge_dt'])[ops_cols].last().reset_index()

    df['merge_dt'] = pd.to_datetime(df['date'])
    df = df.sort_values(['pool_clean', 'merge_dt'])
    ops_pivot = ops_pivot.sort_values(['pool_clean', 'merge_dt'])

    result_frames = []
    for pool_name in df['pool_clean'].unique():
        pool_daily = df[df['pool_clean'] == pool_name].copy()
        pool_ops = ops_pivot[ops_pivot['pool_clean'] == pool_name].copy()

        if len(pool_ops) == 0:
            for c in ops_cols:
                pool_daily[c] = np.nan
            result_frames.append(pool_daily)
            continue

        pool_daily_sorted = pool_daily.sort_values('merge_dt')
        pool_ops_sorted = pool_ops.sort_values('merge_dt')

        merged = pd.merge_asof(
            pool_daily_sorted,
            pool_ops_sorted[['merge_dt'] + ops_cols],
            on='merge_dt',
            direction='backward',
            suffixes=('', '_ops')
        )
        dup = [c for c in merged.columns if c.endswith('_ops')]
        if dup:
            for d in dup:
                orig = d.replace('_ops', '')
                if orig in merged.columns:
                    merged[orig] = merged[orig].fillna(merged[d])
                merged = merged.drop(columns=[d])

        result_frames.append(merged)

    df = pd.concat(result_frames, ignore_index=True)

    defaults = {
        'daily_filtration_hours': 10.0,
        'hypo_dosing_hours': 8.0,
        'hypo_dosing_percentage': 10.0,
        'ph_dosing_hours': 1.0,
        'ph_dosing_percentage': 2.0,
    }
    for c, v in defaults.items():
        if c in df.columns:
            df[c] = df[c].fillna(v)
        else:
            df[c] = v

    df = df.drop(columns=['merge_dt'], errors='ignore')
    logger.info(f"  Added operational setpoint features. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Pass 3: Temporal Lag Features (Dual-State Aware)
# ─────────────────────────────────────────────────────────────────────────────
def pass3_temporal_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Computes temporal lag features tracking both pre (arrival) and post (refreshed) states."""
    logger.info("Pass 3: Computing dual-state temporal lag features...")

    df = df.sort_values(['pool_clean', 'date']).reset_index(drop=True)
    grouped = df.groupby('pool_clean')

    # Lags of the refreshed departure state (which drove decay)
    df['free_chlorine_post_lag1'] = grouped['free_chlorine_post_ppm'].shift(1)
    df['free_chlorine_post_lag2'] = grouped['free_chlorine_post_ppm'].shift(2)
    df['free_chlorine_rolling_mean_3'] = grouped['free_chlorine_post_ppm'].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    
    # Lags of the arrival state
    df['free_chlorine_pre_lag1'] = grouped['free_chlorine_pre_ppm'].shift(1)
    df['ph_post_lag1'] = grouped['ph_post'].shift(1)
    df['turbidity_post_lag1'] = grouped['turbidity_post'].shift(1)
    
    # Differential movement
    df['cl_diff_lag1'] = df['free_chlorine_post_ppm'] - df['free_chlorine_post_lag1']
    df['cl_diff_rolling'] = grouped['cl_diff_lag1'].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    df['cl_decay_slope_recent'] = grouped['free_chlorine_post_ppm'].transform(
        lambda s: s.rolling(5, min_periods=2).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0.0,
            raw=True
        )
    )

    lag_cols = [
        'free_chlorine_post_lag1', 'free_chlorine_post_lag2',
        'free_chlorine_rolling_mean_3', 'free_chlorine_pre_lag1',
        'ph_post_lag1', 'turbidity_post_lag1',
        'cl_diff_lag1', 'cl_diff_rolling', 'cl_decay_slope_recent'
    ]
    for c in lag_cols:
        df[c] = df.groupby('pool_clean')[c].transform(lambda s: s.bfill().ffill())
        df[c] = df[c].fillna(0.0)

    logger.info(f"  Added dual-state temporal lag features. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Pass 4: Physics & Kinetic Features (Forward from Post-Treatment)
# ─────────────────────────────────────────────────────────────────────────────
def pass4_physics_kinetics(df: pd.DataFrame) -> pd.DataFrame:
    """Derives physics kinetics starting forward from Today's Post-Treatment State."""
    logger.info("Pass 4: Computing physics & kinetic features from departure state...")

    vol = df['pool_volume']
    area = df['pool_surface_area']
    cl_post = df['free_chlorine_post_ppm']
    rad = df['solar_radiation_mj']
    temp = df['temperature_ambient_mean_c']
    turb_post = df['turbidity_post']
    cya_post = df['cya_post_ppm']
    ph_post = df['ph_post']
    outdoor = df['outdoor_pool']

    spec_surface = area / vol

    # theoretical_decay_k
    k_photo = 0.025 * (rad / 20.0) * np.clip(spec_surface, 0.5, 3.0) * outdoor
    k_temp  = 0.015 * np.clip(temp / 25.0, 0.5, 2.0)
    k_turb  = 0.020 * np.clip(turb_post, 0.1, 5.0)
    k_cya   = 0.015 * np.clip(cya_post / 50.0, 0.0, 0.8)
    k_bather= 0.025 * df['is_weekend'] * df['community_pool']
    df['theoretical_decay_k'] = np.clip(
        0.04 + k_photo + k_temp + k_turb + k_bather - k_cya, 0.02, 0.80
    ).round(4)

    # Theoretical retained chlorine decaying from today's refreshed post-treatment state
    df['theoretical_retained_chlorine'] = (cl_post * np.exp(-df['theoretical_decay_k'])).round(3)

    # Active HOCl fraction based on refreshed pH
    df['active_hocl_fraction'] = (1.0 / (1.0 + 10.0 ** (ph_post - 7.53))).round(4)

    # Hydraulic turnover
    motor_flow = df.get('motor_pump_flow_rate', pd.Series(15.0, index=df.index))
    filt_hours = df.get('daily_filtration_hours', pd.Series(10.0, index=df.index))
    df['daily_turnover_ratio'] = ((motor_flow * filt_hours) / vol).clip(0.0, 10.0).round(3)

    # HOCl demand proxy
    df['hocl_demand_proxy'] = (
        df['active_hocl_fraction'] * cl_post * df['daily_turnover_ratio'] * (1.0 + turb_post)
    ).round(3)

    # Decay potential
    df['cl_decay_potential'] = ((rad * temp) / np.sqrt(vol.clip(lower=1.0))).round(3)

    # Theoretical forward projection
    pump_dose = df['daily_pump_cl2_delivered_ppm']
    df['chlorine_post_treatment_theoretical'] = (
        df['theoretical_retained_chlorine'] + pump_dose
    ).clip(0.0, 5.0).round(3)

    logger.info(f"  Added physics features. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Pass 5: Pool Historical Baselines (Leak-Free)
# ─────────────────────────────────────────────────────────────────────────────
def pass5_pool_history(df: pd.DataFrame) -> pd.DataFrame:
    """Computes pool-level historical baselines strictly from 2023-2025 training data."""
    logger.info("Pass 5: Computing leak-free pool historical baselines...")

    cl_col = 'free_chlorine_pre_ppm'

    if 'is_train_split' not in df.columns:
        df['is_train_split'] = (df['year'] <= 2025).astype(int)

    train_obs = df[
        (df['is_train_split'] == 1) &
        (df['is_observed_measurement_day'] == 1)
    ].copy()

    pool_stats = train_obs.groupby('pool_clean')[cl_col].agg(
        pool_cl_hist_mean='mean',
        pool_cl_hist_std='std',
        pool_cl_hist_min='min',
        pool_cl_hist_max='max',
        pool_cl_hist_q25=lambda x: x.quantile(0.25),
        pool_cl_hist_q75=lambda x: x.quantile(0.75),
    ).reset_index()

    visit_counts = train_obs.groupby('pool_clean').size().reset_index(name='pool_visit_count')
    pool_stats = pool_stats.merge(visit_counts, on='pool_clean', how='left')
    pool_stats['pool_cl_hist_std'] = pool_stats['pool_cl_hist_std'].fillna(0.5)

    global_mean = train_obs[cl_col].mean()
    global_std = train_obs[cl_col].std()

    df = df.merge(pool_stats, on='pool_clean', how='left')

    hist_cols = ['pool_cl_hist_mean', 'pool_cl_hist_std', 'pool_cl_hist_min',
                 'pool_cl_hist_max', 'pool_cl_hist_q25', 'pool_cl_hist_q75']
    defaults = {
        'pool_cl_hist_mean': global_mean,
        'pool_cl_hist_std': global_std,
        'pool_cl_hist_min': 0.0,
        'pool_cl_hist_max': 5.0,
        'pool_cl_hist_q25': 1.5,
        'pool_cl_hist_q75': 3.5,
        'pool_visit_count': 0,
    }
    for c, v in defaults.items():
        df[c] = df[c].fillna(v)

    for c in hist_cols:
        df[c] = df[c].round(3)

    logger.info(f"  Added pool history baselines. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Pass 6: Weather Rolling Windows
# ─────────────────────────────────────────────────────────────────────────────
def pass6_weather_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Computes trailing 3-day rolling weather aggregates per pool."""
    logger.info("Pass 6: Computing 3-day trailing weather windows...")

    df = df.sort_values(['pool_clean', 'date']).reset_index(drop=True)
    g = df.groupby('pool_clean')

    df['window_solar_rad_sum_mj'] = g['solar_radiation_mj'].transform(
        lambda s: s.rolling(3, min_periods=1).sum()
    ).round(2)

    df['window_solar_rad_mean_mj'] = g['solar_radiation_mj'].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    ).round(2)

    df['window_temp_mean_c'] = g['temperature_ambient_mean_c'].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    ).round(1)

    df['window_temp_max_c'] = g['temperature_ambient_max_c'].transform(
        lambda s: s.rolling(3, min_periods=1).max()
    ).round(1)

    df['window_precip_sum_mm'] = g['precipitation_mm'].transform(
        lambda s: s.rolling(3, min_periods=1).sum()
    ).round(1)

    df['window_sunshine_hours_sum'] = g['sunshine_duration_hrs'].transform(
        lambda s: s.rolling(3, min_periods=1).sum()
    ).round(2)

    df['window_wind_max_kmh'] = g['wind_speed_max_kmh'].transform(
        lambda s: s.rolling(3, min_periods=1).max()
    ).round(1)

    df['window_et0_sum_mm'] = g['et0_evapotranspiration'].transform(
        lambda s: s.rolling(3, min_periods=1).sum()
    ).round(2)

    logger.info(f"  Added weather window features. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Pass 7: Interaction & Domain Features
# ─────────────────────────────────────────────────────────────────────────────
def pass7_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Creates cross-feature interaction terms."""
    logger.info("Pass 7: Computing interaction & domain features...")

    vol = df['pool_volume']
    cl_post = df['free_chlorine_post_ppm']

    df['rad_per_volume'] = (df['solar_radiation_mj'] / vol).round(4)

    pump_flow = df.get('hypochlorite_pump_flow_rate', pd.Series(4.0, index=df.index))
    df['pump_flow_per_vol'] = (pump_flow / vol).round(6)

    df['bather_surge_index'] = (
        df['is_weekend'] * df['community_pool'] *
        df['temperature_ambient_mean_c'] / 25.0
    ).round(3)

    df = df.sort_values(['pool_clean', 'date']).reset_index(drop=True)
    df['weekend_exposure_ratio'] = df.groupby('pool_clean')['is_weekend'].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    ).round(3)

    turnover = df.get('daily_turnover_ratio', pd.Series(1.0, index=df.index))
    df['cl_times_turnover'] = (cl_post * turnover).round(3)

    et0 = df.get('et0_evapotranspiration', pd.Series(4.0, index=df.index))
    wind = df.get('wind_speed_max_kmh', pd.Series(10.0, index=df.index))
    df['evap_stress_index'] = (et0 * (1.0 + wind / 30.0)).round(3)

    logger.info(f"  Added interaction features. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Pass 8: Cyclical Encodings, Multi-Chemical Targets & Split Labels
# ─────────────────────────────────────────────────────────────────────────────
def pass8_cyclical_target_split(df: pd.DataFrame) -> pd.DataFrame:
    """Adds cyclical time encodings, target variables, and train/test split."""
    logger.info("Pass 8: Adding cyclical encodings and multi-chemical targets...")

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12).round(4)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12).round(4)

    df['day_of_year'] = pd.to_datetime(df['date']).dt.dayofyear
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365).round(4)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365).round(4)

    df['is_train_split'] = (df['year'] <= 2025).astype(int)

    # Primary Target: Tomorrow's Arrival Pre-Treatment Free Chlorine
    df = df.sort_values(['pool_clean', 'date']).reset_index(drop=True)
    df['target_next_day_free_chlorine'] = df.groupby('pool_clean')['free_chlorine_pre_ppm'].shift(-1)

    # Secondary Chemical Targets for Tomorrow's Arrival State
    df['target_next_day_ph'] = df.groupby('pool_clean')['ph_pre'].shift(-1)
    df['target_next_day_turbidity'] = df.groupby('pool_clean')['turbidity_pre'].shift(-1)

    # Compliance band based on tomorrow's arrival chlorine
    def _band(x):
        if pd.isna(x):
            return np.nan
        if x < 1.0:
            return 'under'
        elif x <= 3.0:
            return 'compliant'
        else:
            return 'over'

    df['target_next_day_compliance_band'] = df['target_next_day_free_chlorine'].apply(_band)

    before = len(df)
    df = df.dropna(subset=['target_next_day_free_chlorine']).reset_index(drop=True)
    logger.info(f"  Dropped {before - len(df)} rows with no next-day target (last day per pool)")

    logger.info(f"  Added cyclical, multi-chemical targets and split. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Final NaN cleanup & export
# ─────────────────────────────────────────────────────────────────────────────
def final_cleanup_and_export(df: pd.DataFrame,
                             output_csv: str = "data/processed/pool_daily_ml_ready.csv",
                             meta_json: str = "data/processed/pool_daily_ml_feature_metadata.json") -> None:
    """Final NaN sweep, export CSV and metadata JSON."""
    logger.info("Final cleanup & export...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    nan_report = df[numeric_cols].isna().sum()
    nan_cols = nan_report[nan_report > 0]
    if len(nan_cols) > 0:
        logger.warning(f"  Filling {len(nan_cols)} columns with remaining NaNs:")
        for c in nan_cols.index:
            fill_val = df[c].median() if df[c].notna().any() else 0.0
            logger.warning(f"    {c}: {nan_cols[c]} NaNs → filled with {fill_val:.3f}")
            df[c] = df[c].fillna(fill_val)

    total_nans = int(df.isna().sum().sum())
    assert total_nans == 0, f"Still have {total_nans} NaN cells after cleanup!"

    df = df.sort_values(['pool_clean', 'date']).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved ML-ready dataset to {output_csv}: {df.shape}")

    feature_categories = {
        'identity': ['pool_clean', 'date', 'year', 'month'],
        'flags': ['is_observed_measurement_day', 'is_chemical_dosed_day', 'is_weekend', 'is_train_split'],
        'chlorine_dual_state': [
            'free_chlorine_pre_ppm', 'free_chlorine_post_ppm',
            'chlorine_dosage_boost_ppm', 'free_chlorine_estimated_daily_mean_ppm'
        ],
        'ph_dual_state': ['ph_pre', 'ph_post', 'ph_delta', 'ph'],
        'active_hocl_dual_state': ['active_hocl_pre_ppm', 'active_hocl_post_ppm', 'active_hocl_ppm'],
        'turbidity_dual_state': ['turbidity_pre', 'turbidity_post', 'turbidity_delta', 'turbidity'],
        'cya_dual_state': ['cya_pre_ppm', 'cya_post_ppm', 'cya_added_ppm', 'cya_cumulative_ppm'],
        'temperature_and_dosing': [
            'water_temperature_c', 'shock_dosage_ppm',
            'erodible_active_cl2_added_grams', 'daily_pump_cl2_delivered_ppm'
        ],
        'pool_profile': [
            'pool_volume', 'pool_surface_area', 'community_pool', 'outdoor_pool',
            'oval_pool', 'overflow_pool', 'rectangular_pool_07',
            'rectangular_pool_0714', 'round_pool', 'skimmer_pool',
            'number_of_filters', 'number_of_motors', 'filter_diameter',
            'motor_pump_flow_rate', 'hypochlorite_pump_flow_rate',
            'ph_pump_flow_rate', 'heated_pool',
            'contaminating_vegetation', 'sunscreen_overuse',
            'deck_grass_area', 'deck_mixed_area', 'deck_paved_area',
            'estimated_mean_depth', 'specific_surface_ratio'
        ],
        'operational': [
            'daily_filtration_hours', 'hypo_dosing_hours',
            'hypo_dosing_percentage', 'ph_dosing_hours', 'ph_dosing_percentage'
        ],
        'temporal_lags': [
            'free_chlorine_post_lag1', 'free_chlorine_post_lag2',
            'free_chlorine_rolling_mean_3', 'free_chlorine_pre_lag1',
            'ph_post_lag1', 'turbidity_post_lag1',
            'cl_diff_lag1', 'cl_diff_rolling', 'cl_decay_slope_recent'
        ],
        'physics': [
            'theoretical_decay_k', 'theoretical_retained_chlorine',
            'active_hocl_fraction', 'daily_turnover_ratio',
            'hocl_demand_proxy', 'cl_decay_potential',
            'chlorine_post_treatment_theoretical'
        ],
        'pool_history': [
            'pool_cl_hist_mean', 'pool_cl_hist_std', 'pool_cl_hist_min',
            'pool_cl_hist_max', 'pool_cl_hist_q25', 'pool_cl_hist_q75',
            'pool_visit_count'
        ],
        'weather_daily': [
            'solar_radiation_mj', 'temperature_ambient_mean_c',
            'temperature_ambient_max_c', 'precipitation_mm',
            'sunshine_duration_hrs', 'uv_index_max', 'wind_speed_max_kmh',
            'daylight_duration_hrs', 'et0_evapotranspiration'
        ],
        'weather_windows': [
            'window_solar_rad_sum_mj', 'window_solar_rad_mean_mj',
            'window_temp_mean_c', 'window_temp_max_c',
            'window_precip_sum_mm', 'window_sunshine_hours_sum',
            'window_wind_max_kmh', 'window_et0_sum_mm'
        ],
        'interactions': [
            'rad_per_volume', 'pump_flow_per_vol', 'bather_surge_index',
            'weekend_exposure_ratio', 'cl_times_turnover', 'evap_stress_index'
        ],
        'cyclical': [
            'month_sin', 'month_cos', 'day_of_year',
            'day_of_year_sin', 'day_of_year_cos', 'day_of_week'
        ],
        'targets': [
            'target_next_day_free_chlorine', 'target_next_day_ph',
            'target_next_day_turbidity', 'target_next_day_compliance_band'
        ],
        'metadata': ['imputation_confidence_score', 'imputation_method'],
    }

    metadata = {
        "dataset_name": "ML-Ready Daily Pool Water Quality Dataset (Multi-Chemical Dual-State Architecture)",
        "created_at": datetime.now().isoformat(),
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_nans": 0,
        "unique_pools": int(df['pool_clean'].nunique()),
        "train_rows": int((df['is_train_split'] == 1).sum()),
        "test_rows": int((df['is_train_split'] == 0).sum()),
        "date_range": {"start": str(df['date'].min()), "end": str(df['date'].max())},
        "target_variable": "target_next_day_free_chlorine",
        "feature_categories": feature_categories,
        "all_columns": list(df.columns),
    }

    with open(meta_json, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved feature metadata to {meta_json}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    logger.info("=== Building Multi-Chemical Dual-State ML-Ready Dataset ===")

    daily_csv = "data/processed/pool_daily_reconstructed_timeseries.csv"
    df = pd.read_csv(daily_csv)
    logger.info(f"Loaded daily dataset: {df.shape}")

    df_raw, _ = load_raw_data()
    _, df_profile, df_ops, _ = disaggregate_tables(df_raw)

    train_pools = set(df[df['year'] <= 2025]['pool_clean'].unique())
    df_profile = impute_pool_profiles(df_profile, train_pools)

    df['is_train_split'] = (df['year'] <= 2025).astype(int)

    df = pass1_pool_profile_join(df, df_profile)
    df = pass2_operational_setpoints(df, df_ops)
    df = pass3_temporal_lags(df)
    df = pass4_physics_kinetics(df)
    df = pass5_pool_history(df)
    df = pass6_weather_windows(df)
    df = pass7_interactions(df)
    df = pass8_cyclical_target_split(df)

    final_cleanup_and_export(df)

    logger.info("=== Multi-Chemical ML-Ready Dataset Build Complete! ===")


if __name__ == "__main__":
    main()
