"""
feature_pipeline.py — Shared feature engineering for live inference.

Mirrors Step 5 of pipeline_v3.py exactly, operating on a single pool's
sorted reading history instead of the full fleet DataFrame.

Exported symbols
----------------
build_features(df_pool)   -> pd.DataFrame  (same column schema as df_master in training)
MIN_READINGS_FOR_MODEL    -> int           (threshold below which we fall back to rules)
"""

import numpy as np
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------------
# Regulatory constants (mirror pipeline_v3.py exactly)
# ---------------------------------------------------------------------------
REG_CHLORINE_MIN = 0.5
REG_CHLORINE_IDEAL_MAX = 2.0
REG_CHLORINE_CLOSE = 5.0
REG_PH_MIN = 7.2
REG_PH_MAX = 8.0
REG_TURBIDITY_MAX = 5.0
PH_IDEAL = 7.2
CHLORINE_IDEAL = 1.25

# ---------------------------------------------------------------------------
# Minimum readings threshold
# A pool needs at least 3 readings so that:
#   lag1  = shift(1)  → computable from reading 2 onwards
#   lag2  = shift(2)  → computable from reading 3 onwards
#   roll3 = rolling(3, min_periods=2) mean → computable from reading 2 onwards
# With 3 readings, all lag/rolling features on the LAST row are from real data.
# ---------------------------------------------------------------------------
MIN_READINGS_FOR_MODEL = 3


# ---------------------------------------------------------------------------
# Internal helpers (mirror pipeline_v3.py)
# ---------------------------------------------------------------------------

def _consecutive_clean(series_values):
    """Count consecutive non-breach (0) visits ending at each position."""
    result = []
    count = 0
    for val in series_values:
        if val == 0:
            count += 1
        else:
            count = 0
        result.append(count)
    return result


def _make_pool_type(row):
    parts = []
    if row.get('pool_heated', 0): parts.append('heated')
    if row.get('pool_outdoor', 0): parts.append('outdoor')
    if row.get('pool_community', 0): parts.append('community')
    if row.get('pool_private', 0): parts.append('private')
    if row.get('pool_public', 0): parts.append('public')
    return '_'.join(parts) if parts else 'unknown'


def _make_deck_type(row):
    g = float(row.get('deck_grass', 0) or 0)
    p = float(row.get('deck_paved', 0) or 0)
    m = float(row.get('deck_mixed', 0) or 0)
    if m > 0: return 'mixed'
    if g > 0 and p > 0: return 'mixed'
    if g > 0: return 'grass'
    if p > 0: return 'paved'
    return 'unknown'


# ---------------------------------------------------------------------------
# build_features
# ---------------------------------------------------------------------------

def build_features(df_pool: pd.DataFrame, fill_values: dict = None,
                   pool_volume_default: float = 220.0) -> pd.DataFrame:
    """
    Compute all training features for a single pool's reading history.

    Parameters
    ----------
    df_pool : DataFrame
        Readings for ONE pool, must contain at minimum:
            reading_date, ph, free_chlorine, turbidity
        May optionally contain pool static fields and ops/product cols.
    fill_values : dict
        Values from inference_config.json["fill_values"].
        Used to fill any NaN that remain after real computation.
    pool_volume_default : float
        Fallback when pool_volume_m3 is missing (220 m³ = training fleet median).

    Returns
    -------
    DataFrame with all feature columns on the same schema as df_master
    during training. Columns that are unavailable will be filled from
    fill_values.  The DataFrame is sorted by reading_date ascending.
    """
    if fill_values is None:
        fill_values = {}

    df = df_pool.copy()

    # ---- Normalise date ----
    if not pd.api.types.is_datetime64_any_dtype(df['reading_date']):
        df['reading_date'] = pd.to_datetime(df['reading_date'], errors='coerce')
    df = df.sort_values('reading_date').reset_index(drop=True)

    # ---- Ensure numeric core measurements ----
    for col in ('ph', 'free_chlorine', 'turbidity'):
        df[col] = pd.to_numeric(df.get(col, pd.Series([np.nan] * len(df))), errors='coerce')

    # ---- Static pool fields — propagate then fill with defaults ----
    # Numeric statics
    numeric_statics = {
        'pool_volume_m3': pool_volume_default,
        'pool_surface_m2': 155.0,
        'filter_diameter': 850.0,
        'filter_count': 1.0,
        'motor_count': 1.0,
    }
    for col, default_val in numeric_statics.items():
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Propagate first non-null value across all rows
        first_known = df[col].dropna().iloc[0] if df[col].notna().any() else np.nan
        df[col] = df[col].fillna(first_known if pd.notna(first_known) else default_val)

    # Flag statics (binary, fill with 0)
    flag_cols = [
        'pool_heated', 'pool_community', 'pool_skimmer', 'pool_overflow',
        'pool_outdoor', 'pool_oval', 'pool_private', 'pool_public',
        'pool_rectangular_0714', 'pool_rectangular_07', 'pool_round',
        'vegetation_contamination', 'deck_grass', 'deck_mixed', 'deck_paved',
    ]
    for col in flag_cols:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Categorical
    if 'pool_type' not in df.columns:
        df['pool_type'] = df.apply(_make_pool_type, axis=1)
    df['pool_type'] = df['pool_type'].fillna('unknown')

    if 'deck_type' not in df.columns:
        df['deck_type'] = df.apply(_make_deck_type, axis=1)
    df['deck_type'] = df['deck_type'].fillna('unknown')

    # ---- Ops/product fields — fill with 0/NaN as in training ----
    ops_fields = {
        'daily_filtration_hours': np.nan,
        'water_temperature': np.nan,
        'ph_dosing_pct': np.nan,
        'hypochlorite_dosing_pct': np.nan,
    }
    for col, default_val in ops_fields.items():
        if col not in df.columns:
            df[col] = default_val

    product_fields = {
        'total_chlorine_product': 0.0,
        'total_ph_minus_product': 0.0,
        'last_total_chlorine_applied': 0.0,
    }
    for col, default_val in product_fields.items():
        if col not in df.columns:
            df[col] = default_val

    # ---- Step 3 breach flags (current reading) ----
    df['ph_breach'] = (~df['ph'].between(REG_PH_MIN, REG_PH_MAX)) & df['ph'].notna()
    df['chlorine_breach'] = (
        (df['free_chlorine'] < REG_CHLORINE_MIN) |
        (df['free_chlorine'] > REG_CHLORINE_CLOSE)
    ) & df['free_chlorine'].notna()
    df['turbidity_breach'] = (df['turbidity'] > REG_TURBIDITY_MAX) & df['turbidity'].notna()
    df['any_breach'] = df['ph_breach'] | df['chlorine_breach'] | df['turbidity_breach']

    # ---- Step 5 — Lag features ----
    for col, prefix in [('ph', 'ph'), ('free_chlorine', 'chlorine'), ('turbidity', 'turbidity')]:
        df[f'{prefix}_lag1'] = df[col].shift(1)
        df[f'{prefix}_lag2'] = df[col].shift(2)

    # ---- Rolling statistics (window=3, min_periods=2) ----
    for col, prefix in [('ph', 'ph'), ('free_chlorine', 'chlorine'), ('turbidity', 'turbidity')]:
        df[f'{prefix}_roll3_mean'] = df[col].rolling(window=3, min_periods=2).mean()
        if prefix != 'turbidity':
            df[f'{prefix}_roll3_std'] = df[col].rolling(window=3, min_periods=2).std()

    # ---- Visit interval features ----
    df['days_since_last_visit'] = df['reading_date'].diff().dt.days
    df['visit_day_of_week'] = df['reading_date'].dt.dayofweek
    df['visit_month'] = df['reading_date'].dt.month
    df['visit_is_summer'] = df['visit_month'].isin([6, 7, 8, 9]).astype(int)

    # ---- Chemistry features ----
    df['ph_deviation'] = (df['ph'] - PH_IDEAL).abs()
    df['chlorine_deficit'] = (REG_CHLORINE_MIN - df['free_chlorine']).clip(lower=0)

    # ---- Regulatory headroom ----
    df['chlorine_headroom_low']  = df['free_chlorine'] - REG_CHLORINE_MIN
    df['chlorine_headroom_high'] = REG_CHLORINE_CLOSE - df['free_chlorine']
    df['ph_headroom_low']        = df['ph'] - REG_PH_MIN
    df['ph_headroom_high']       = REG_PH_MAX - df['ph']
    df['turbidity_headroom']     = REG_TURBIDITY_MAX - df['turbidity']
    df['min_headroom'] = df[[
        'chlorine_headroom_low', 'chlorine_headroom_high',
        'ph_headroom_low', 'ph_headroom_high', 'turbidity_headroom',
    ]].min(axis=1)

    # ---- Trend features ----
    df['ph_trend']        = df['ph']           - df['ph_lag1']
    df['chlorine_trend']  = df['free_chlorine'] - df['chlorine_lag1']
    df['turbidity_trend'] = df['turbidity']    - df['turbidity_lag1']

    days_safe = df['days_since_last_visit'].replace(0, np.nan)
    df['ph_rate_per_day']        = df['ph_trend']        / days_safe
    df['chlorine_rate_per_day']  = df['chlorine_trend']  / days_safe
    df['turbidity_rate_per_day'] = df['turbidity_trend'] / days_safe

    # ---- Breach history features ----
    df['current_any_breach']      = df['any_breach'].astype(int)
    df['current_ph_breach']       = df['ph_breach'].astype(int)
    df['current_chlorine_breach'] = df['chlorine_breach'].astype(int)

    df['consecutive_clean_visits'] = _consecutive_clean(df['current_any_breach'].values)
    df['breach_rate_last5'] = df['current_any_breach'].rolling(window=5, min_periods=1).mean()

    # ---- V3 features ----
    # 1. pH-Chlorine Effectiveness Index
    ph_correction = np.where(
        df['ph'] <= 7.5, 1.0,
        1.0 - 0.5 * ((df['ph'] - 7.5) / 0.5)
    )
    df['cl_effectiveness_index'] = df['free_chlorine'] * np.clip(ph_correction, 0.1, 1.0)

    # 2. Dose per m3
    df['chlorine_dose_per_m3']  = df['last_total_chlorine_applied'] / df['pool_volume_m3']
    df['ph_minus_dose_per_m3']  = df['total_ph_minus_product']      / df['pool_volume_m3']

    # 3. Chlorine decay rate per m3
    df['chlorine_decay_per_m3'] = df['chlorine_rate_per_day'] / df['pool_volume_m3']

    # 4. Pool visit number (cumcount)
    df['pool_visit_number'] = range(len(df))

    # ---- Fill remaining NaN from fill_values ----
    # (Any column that was uncomputable due to short history, or truly missing)
    all_numeric = [
        'pool_surface_m2', 'pool_volume_m3', 'filter_diameter', 'filter_count', 'motor_count',
        'ph_lag1', 'ph_lag2', 'chlorine_lag1', 'chlorine_lag2', 'turbidity_lag1', 'turbidity_lag2',
        'ph_roll3_mean', 'ph_roll3_std', 'chlorine_roll3_mean', 'chlorine_roll3_std',
        'turbidity_roll3_mean',
        'last_total_chlorine_applied', 'total_ph_minus_product',
        'days_since_last_visit', 'visit_month', 'visit_is_summer', 'visit_day_of_week',
        'chlorine_headroom_low', 'chlorine_headroom_high',
        'ph_headroom_low', 'ph_headroom_high', 'turbidity_headroom', 'min_headroom',
        'ph_trend', 'chlorine_trend', 'turbidity_trend',
        'ph_rate_per_day', 'chlorine_rate_per_day', 'turbidity_rate_per_day',
        'consecutive_clean_visits', 'breach_rate_last5',
        'current_any_breach', 'current_ph_breach', 'current_chlorine_breach',
        'ph_deviation', 'chlorine_deficit',
        'cl_effectiveness_index', 'chlorine_dose_per_m3', 'ph_minus_dose_per_m3',
        'chlorine_decay_per_m3', 'pool_visit_number',
    ]
    for col in all_numeric:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(fill_values.get(col, 0.0))

    return df


# ---------------------------------------------------------------------------
# Determine which features on the LAST row were from real vs fill data
# ---------------------------------------------------------------------------

def feature_quality_report(df_features: pd.DataFrame) -> dict:
    """
    Given the output of build_features() for a pool, examine the LAST row
    and report which key features were real vs filled.

    Returns
    -------
    dict:
        real_count   : number of key lag/rolling features from real history
        fill_count   : number that were fallback-filled
        used_fills   : list of feature names that needed fill values
        history_len  : total number of readings for this pool
    """
    last = df_features.iloc[-1]
    n = len(df_features)

    lag_features = [
        'ph_lag1', 'ph_lag2',
        'chlorine_lag1', 'chlorine_lag2',
        'turbidity_lag1', 'turbidity_lag2',
        'ph_roll3_mean', 'chlorine_roll3_mean', 'turbidity_roll3_mean',
        'ph_trend', 'chlorine_trend', 'turbidity_trend',
        'days_since_last_visit',
    ]

    used_fills = []
    # Check which would have been null before fill (by checking n)
    # With n=1: all lag/rolling NaN
    # With n=2: lag1 real, lag2 NaN, rolling real (min_periods=2)
    # With n>=3: all real
    for feat in lag_features:
        needs_fill = False
        if 'lag2' in feat and n < 3:
            needs_fill = True
        elif ('lag1' in feat or 'roll3' in feat or 'trend' in feat or
              feat == 'days_since_last_visit') and n < 2:
            needs_fill = True
        if needs_fill:
            used_fills.append(feat)

    return {
        'real_count': len(lag_features) - len(used_fills),
        'fill_count': len(used_fills),
        'used_fills': used_fills,
        'history_len': n,
    }
