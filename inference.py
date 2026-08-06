#!/usr/bin/env python3
"""
Pool Predictive Maintenance — V6 Inference
==========================================
Copyright (c) 2026 shaik imaduddin. All rights reserved.
Private and Proprietary. Unauthorized use or copying is prohibited.

Dashboard use-case
------------------
If the last technician visit was Monday and the dashboard is opened on Wednesday,
this module produces today (Wednesday) AND tomorrow (Thursday) predicted values
by chaining 1-day-forward predictions:

    Monday (actual) → predict Tuesday → predict Wednesday → predict Thursday

Usage
-----
    python inference.py                      # all active pools, today + tomorrow
    python inference.py --pool 461           # single pool
    python inference.py --date 2026-08-10    # specific query date
"""

import os
import sys
import json
import pickle
import warnings
import argparse

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(SCRIPT_DIR, 'models')
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, 'outputs')
DATA_DIR    = os.path.join(SCRIPT_DIR, 'data')

# ─── Regulatory constants (must match pipeline_v6 CONFIG) ────────────────────
REG_CHLORINE_MIN     = 0.5
REG_CHLORINE_CLOSE   = 5.0
REG_PH_MIN           = 7.2
REG_PH_MAX           = 8.0
REG_TURBIDITY_MAX    = 5.0
CLIENT_CL_TARGET_MIN = 1.0
CLIENT_CL_TARGET_MAX = 1.5


# ─── Load artifacts ───────────────────────────────────────────────────────────

def load_models():
    """Load all V6 models, preprocessor, and inference config."""
    print("Loading V6 models...")
    with open(os.path.join(MODELS_DIR, 'inference_config_v6.json')) as f:
        config = json.load(f)
    with open(os.path.join(MODELS_DIR, 'preprocessor_v6.pkl'), 'rb') as f:
        preprocessor = pickle.load(f)

    import xgboost as xgb
    model_cl   = xgb.XGBRegressor()
    model_ph   = xgb.XGBRegressor()
    model_turb = xgb.XGBRegressor()
    model_cl.load_model(os.path.join(MODELS_DIR,   'xgb_chlorine_next.json'))
    model_ph.load_model(os.path.join(MODELS_DIR,   'xgb_ph_next.json'))
    model_turb.load_model(os.path.join(MODELS_DIR, 'xgb_turbidity_next.json'))

    print("  ✓ Chlorine | ✓ pH | ✓ Turbidity | ✓ Preprocessor")
    return model_cl, model_ph, model_turb, preprocessor, config


def load_data():
    """Load master dataset and weather data (with same column rename as pipeline_v6)."""
    df_master  = pd.read_csv(os.path.join(OUTPUTS_DIR, 'master_dataset_v6.csv'),
                              parse_dates=['reading_date'])
    df_weather = pd.read_csv(os.path.join(DATA_DIR, 'weather_alicante_2023_2026.csv'),
                              parse_dates=['date'])
    df_master['reading_date_only'] = pd.to_datetime(df_master['reading_date']).dt.normalize()
    df_weather['date'] = pd.to_datetime(df_weather['date']).dt.normalize()

    # Apply same rename map as pipeline_v6 STEP 4
    WEATHER_RENAME = {
        'temperature_2m_max':          'w_temp_max',
        'temperature_2m_mean':         'w_temp_mean',
        'uv_index_max':                'w_uv_max',
        'uv_index_clear_sky_max':      'w_uv_clear_sky_max',
        'shortwave_radiation_sum':     'w_solar_radiation',
        'sunshine_duration':           'w_sunshine_hours',
        'precipitation_sum':           'w_precipitation_mm',
        'wind_speed_10m_max':          'w_wind_max_kmh',
        'et0_fao_evapotranspiration':  'w_et0',
    }
    df_weather = df_weather.rename(columns=WEATHER_RENAME)

    print(f"  Master: {len(df_master):,} rows | "
          f"Weather: {len(df_weather):,} days "
          f"({df_weather['date'].min().date()} to {df_weather['date'].max().date()})")
    return df_master, df_weather


# ─── Feature helpers ──────────────────────────────────────────────────────────

def _get_weather(df_weather, target_date, cols):
    row = df_weather[df_weather['date'] == pd.Timestamp(target_date).normalize()]
    if len(row) == 0:
        return {c: np.nan for c in cols}
    return row.iloc[0][cols].to_dict()


def _inject_weather(row, df_weather, step_date, today_cols, tmrw_cols):
    """Inject today's and tomorrow's weather into the feature row."""
    row = row.copy()
    today_wx = _get_weather(df_weather, step_date, today_cols)
    for col, val in today_wx.items():
        row[col] = val
    tmrw_date     = step_date + pd.Timedelta(days=1)
    tmrw_src_cols = [c.replace('w_tmrw_', 'w_') for c in tmrw_cols]
    tmrw_wx       = _get_weather(df_weather, tmrw_date, tmrw_src_cols)
    for t_col, s_col in zip(tmrw_cols, tmrw_src_cols):
        row[t_col] = tmrw_wx.get(s_col, np.nan)
    return row


def _recompute_features(row, pred_cl, pred_ph, pred_turb, step, prev_cl, prev_ph):
    """Update all features that depend on the current chemistry state."""
    row = row.copy()

    # Current state
    row['free_chlorine'] = pred_cl
    row['ph']            = pred_ph
    row['turbidity']     = max(0.0, pred_turb)

    # Lag shifts
    row['chlorine_lag2']  = row.get('chlorine_lag1', pred_cl)
    row['chlorine_lag1']  = prev_cl
    row['ph_lag2']        = row.get('ph_lag1', pred_ph)
    row['ph_lag1']        = prev_ph
    row['turbidity_lag1'] = row.get('turbidity_lag1', pred_turb)
    row['turbidity_lag2'] = row.get('turbidity_lag2', pred_turb)

    # Rolling approximation (3-step window)
    vals_cl   = [pred_cl,   prev_cl,   row.get('chlorine_lag2', pred_cl)]
    vals_ph   = [pred_ph,   prev_ph,   row.get('ph_lag2', pred_ph)]
    vals_turb = [pred_turb, row.get('turbidity_lag1', pred_turb), row.get('turbidity_lag2', pred_turb)]
    row['chlorine_roll3_mean']  = float(np.mean(vals_cl))
    row['chlorine_roll3_std']   = float(np.std(vals_cl))
    row['ph_roll3_mean']        = float(np.mean(vals_ph))
    row['ph_roll3_std']         = float(np.std(vals_ph))
    row['turbidity_roll3_mean'] = float(np.mean(vals_turb))

    # Temporal
    row['days_since_last_visit'] = step

    # Headroom
    row['chlorine_headroom_low']  = pred_cl - REG_CHLORINE_MIN
    row['chlorine_headroom_high'] = REG_CHLORINE_CLOSE - pred_cl
    row['ph_headroom_low']        = pred_ph - REG_PH_MIN
    row['ph_headroom_high']       = REG_PH_MAX - pred_ph
    row['turbidity_headroom']     = REG_TURBIDITY_MAX - max(0.0, pred_turb)
    row['min_headroom']           = min(
        row['chlorine_headroom_low'], row['chlorine_headroom_high'],
        row['ph_headroom_low'],       row['ph_headroom_high'],
        row['turbidity_headroom'],
    )
    row['cl_below_client_target'] = max(0.0, CLIENT_CL_TARGET_MIN - pred_cl)
    row['cl_above_client_target'] = max(0.0, pred_cl - CLIENT_CL_TARGET_MAX)

    # Trends
    row['chlorine_trend']        = pred_cl - prev_cl
    row['ph_trend']              = pred_ph - prev_ph
    row['chlorine_rate_per_day'] = pred_cl - prev_cl
    row['ph_rate_per_day']       = pred_ph - prev_ph
    row['chlorine_acceleration'] = 0.0
    row['ph_acceleration']       = 0.0

    # Effectiveness index
    ph_factor = max(0.1, 1.0 - abs(pred_ph - 7.4) / 2.0)
    row['cl_effectiveness_index'] = pred_cl * np.clip(ph_factor, 0, 1)

    return row


# ─── Core inference ───────────────────────────────────────────────────────────

def predict_forward(pool_id, as_of_date,
                    df_master, model_cl, model_ph, model_turb,
                    preprocessor, config, df_weather):
    """
    Chain 1-day-forward predictions from last visit to as_of_date + 1 (tomorrow).

    Returns a dict with:
        pool_id, last_visit_date, days_since_visit, last_readings,
        forecast (DataFrame with one row per day),
        today_forecast, tomorrow_forecast, visit_needed
    """
    as_of_date = pd.Timestamp(as_of_date).normalize()

    pool_rows = df_master[df_master['pool_id'] == pool_id].sort_values('reading_date')
    if len(pool_rows) == 0:
        return {'error': f'No data for pool_id: {pool_id}'}

    last_row        = pool_rows.iloc[-1].copy()
    last_visit_date = pd.Timestamp(last_row['reading_date']).normalize()
    days_since      = int((as_of_date - last_visit_date).days)

    if days_since < 0:
        return {'error': f'as_of_date {as_of_date.date()} is before last visit {last_visit_date.date()}'}

    # Config
    all_numeric          = config["all_numeric_features"]
    categorical_features = config['categorical_features']
    today_wx_cols        = config.get('weather_current_features', [])
    tmrw_wx_cols         = config.get('weather_tomorrow_features', [])
    fill_values          = config.get('fill_values', {})

    # Base row from last visit — fill NaNs
    base = last_row.copy()
    for col in all_numeric:
        if col not in base.index or pd.isna(base.get(col, np.nan)):
            base[col] = fill_values.get(col, 0.0)
    for col in categorical_features:
        if col not in base.index or pd.isna(base.get(col, np.nan)):
            base[col] = 'unknown'

    # Starting chemistry
    cur_cl   = float(last_row.get('free_chlorine', fill_values.get('free_chlorine', 2.0)))
    cur_ph   = float(last_row.get('ph',            fill_values.get('ph',            7.4)))
    cur_turb = float(last_row.get('turbidity',     fill_values.get('turbidity',     0.5)))

    row          = base.copy()
    forecast_rows = []
    total_steps  = days_since + 1  # last_visit+1 … as_of_date+1

    prev_cl, prev_ph, prev_turb = cur_cl, cur_ph, cur_turb

    for step in range(1, total_steps + 1):
        step_date = last_visit_date + pd.Timedelta(days=step)

        # Update date-based temporal features
        row['visit_month']       = step_date.month
        row['visit_day_of_week'] = step_date.dayofweek
        row['visit_is_summer']   = int(step_date.month in [6, 7, 8, 9])
        row['visit_year']        = step_date.year

        # Inject weather
        row = _inject_weather(row, df_weather, step_date, today_wx_cols, tmrw_wx_cols)

        # Recompute chemistry-dependent features
        row = _recompute_features(row, cur_cl, cur_ph, cur_turb,
                                  step=step, prev_cl=prev_cl, prev_ph=prev_ph)

        # Build feature DataFrame
        feat_df = pd.DataFrame([row])
        for col in all_numeric:
            if col not in feat_df.columns:
                feat_df[col] = fill_values.get(col, 0.0)
            feat_df[col] = pd.to_numeric(feat_df[col], errors='coerce').fillna(fill_values.get(col, 0.0))
        for col in categorical_features:
            if col not in feat_df.columns:
                feat_df[col] = 'unknown'
            feat_df[col] = feat_df[col].fillna('unknown').astype(str)

        X = preprocessor.transform(feat_df[categorical_features + all_numeric])

        pred_cl   = max(0.0, float(model_cl.predict(X)[0]))
        pred_ph   = float(model_ph.predict(X)[0])
        pred_turb = max(0.0, float(model_turb.predict(X)[0]))

        # Status
        cl_breach = pred_cl < REG_CHLORINE_MIN or pred_cl > REG_CHLORINE_CLOSE
        ph_breach = pred_ph < REG_PH_MIN or pred_ph > REG_PH_MAX

        if cl_breach or ph_breach:
            urgency = 'URGENT'
            status  = '🚨 Regulatory breach — URGENT visit'
        elif pred_cl < CLIENT_CL_TARGET_MIN:
            urgency = 'Advised'
            status  = '⚠️  Cl below client target — visit advised'
        elif pred_cl > 2.0:
            urgency = 'Monitor'
            status  = '⚠️  Cl above optimal range — monitor'
        else:
            urgency = 'Routine'
            status  = '✅ OK'

        is_today    = (step_date == as_of_date)
        is_tomorrow = (step_date == as_of_date + pd.Timedelta(days=1))
        day_label   = step_date.strftime('%a')
        if is_today:
            day_label += ' ◀ TODAY'
        elif is_tomorrow:
            day_label += ' ◀ TOMORROW'

        forecast_rows.append({
            'date':           step_date.date(),
            'day':            day_label,
            'days_from_visit': step,
            'predicted_cl':   round(pred_cl,   3),
            'predicted_ph':   round(pred_ph,   3),
            'predicted_turb': round(pred_turb, 3),
            'cl_breach':      cl_breach,
            'ph_breach':      ph_breach,
            'urgency':        urgency,
            'status':         status,
            'is_today':       is_today,
            'is_tomorrow':    is_tomorrow,
        })

        # Chain
        prev_cl, prev_ph, prev_turb = cur_cl, cur_ph, cur_turb
        cur_cl,  cur_ph,  cur_turb  = pred_cl, pred_ph, pred_turb

    fc = pd.DataFrame(forecast_rows)

    dashboard = fc[fc['is_today'] | fc['is_tomorrow']]
    visit_needed = (
        dashboard['cl_breach'].any() or
        dashboard['ph_breach'].any() or
        (dashboard['urgency'] == 'Advised').any()
    )

    return {
        'pool_id':           pool_id,
        'last_visit_date':   last_visit_date.date(),
        'days_since_visit':  days_since,
        'last_readings':     {
            'free_chlorine': round(float(last_row.get('free_chlorine', 0)), 3),
            'ph':            round(float(last_row.get('ph',            0)), 3),
            'turbidity':     round(float(last_row.get('turbidity',     0)), 3),
        },
        'forecast':          fc,
        'today_forecast':    fc[fc['is_today']].to_dict('records'),
        'tomorrow_forecast': fc[fc['is_tomorrow']].to_dict('records'),
        'visit_needed':      visit_needed,
    }


# ─── Pretty print ─────────────────────────────────────────────────────────────

def print_pool_forecast(result):
    if 'error' in result:
        print(f"  ERROR: {result['error']}")
        return

    lr  = result['last_readings']
    fc  = result['forecast']
    print(f"\n{'─'*72}")
    print(f"  Pool : {result['pool_id']}")
    print(f"  Last visit: {result['last_visit_date']} "
          f"({result['days_since_visit']} days ago)  |  "
          f"Cl={lr['free_chlorine']:.2f}  pH={lr['ph']:.2f}  Turb={lr['turbidity']:.2f}")
    print()
    print(f"  {'Date':<12}{'Day':<22}{'Cl (mg/L)':<12}{'pH':<8}{'Turb (NTU)':<12}Status")
    print(f"  {'─'*12}{'─'*22}{'─'*12}{'─'*8}{'─'*12}{'─'*30}")
    for _, row in fc.iterrows():
        print(f"  {str(row['date']):<12}{row['day']:<22}"
              f"{row['predicted_cl']:<12.3f}{row['predicted_ph']:<8.3f}"
              f"{row['predicted_turb']:<12.3f}{row['status']}")
    vn = result['visit_needed']
    print()
    print(f"  📋 {'⚠️  VISIT NEEDED in next 2 days' if vn else '✅ No visit needed in next 2 days'}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Pool V6 Inference — chained daily forecast')
    parser.add_argument('--pool', type=str, default=None)
    parser.add_argument('--date', type=str, default=None)
    parser.add_argument('--top',  type=int, default=10)
    args = parser.parse_args()

    as_of = pd.Timestamp(args.date).normalize() if args.date else pd.Timestamp.now().normalize()

    print(f"\n{'='*72}")
    print(f"  POOL PREDICTIVE MAINTENANCE V6 — DASHBOARD FORECAST")
    print(f"  Query date (today): {as_of.date()}")
    print(f"{'='*72}")

    model_cl, model_ph, model_turb, preprocessor, config = load_models()
    df_master, df_weather = load_data()

    if args.pool:
        pool_ids = [args.pool]
    else:
        latest = df_master.groupby('pool_id')['reading_date'].max()
        cutoff  = as_of - pd.Timedelta(days=30)
        pool_ids = latest[latest >= cutoff].index.tolist()[:args.top]
        print(f"\n  Active pools (reading within past 30 days): {len(pool_ids)}")

    urgent_pools  = []
    advised_pools = []
    routine_pools = []

    for pid in pool_ids:
        res = predict_forward(pid, as_of, df_master,
                              model_cl, model_ph, model_turb,
                              preprocessor, config, df_weather)
        if 'error' in res:
            continue
        print_pool_forecast(res)

        fc_dt = res['forecast']
        dt_rows = fc_dt[fc_dt['is_today'] | fc_dt['is_tomorrow']]
        if (dt_rows['cl_breach'] | dt_rows['ph_breach']).any():
            urgent_pools.append(pid)
        elif (dt_rows['urgency'] == 'Advised').any():
            advised_pools.append(pid)
        else:
            routine_pools.append(pid)

    print(f"\n{'='*72}")
    print(f"  SUMMARY — {as_of.date()}")
    print(f"{'='*72}")
    print(f"  🚨 URGENT  ({len(urgent_pools)}): {', '.join(str(p) for p in urgent_pools) or 'None'}")
    print(f"  ⚠️  ADVISED ({len(advised_pools)}): {', '.join(str(p) for p in advised_pools) or 'None'}")
    print(f"  ✅ ROUTINE ({len(routine_pools)}): {len(routine_pools)} pools OK")
    print()


if __name__ == '__main__':
    main()
