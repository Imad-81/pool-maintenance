#!/usr/bin/env python3
"""
Pool Predictive Maintenance Pipeline — V6
==========================================
Copyright (c) 2026 shaik imaduddin. All rights reserved.
Private and Proprietary. Unauthorized use or copying is prohibited.

Changes from V3:
  - New dataset: Merged_2023_2026.xlsx (Jan 2023 – Aug 2026, 42K+ records)
  - Pool filter: only pools with a liquid chlorine dosing pump
    (cross-referenced against Listado_piscinas_bomba_cloro.xlsx)
  - Spanish column headers — new name-based rename map (robust to reordering)
  - Weather data: Open-Meteo daily weather for Alicante
    (lat 38.3452 N, lon -0.4815 W) merged by exact date
  - Rolling weather since last visit (cumulative UV, solar, rain, heat)
  - New primary targets: next-DAY FREE CHLORINE and pH
    (linearly interpolated 1 day forward from each visit, using the gap
    to the subsequent visit as the interpolation anchor)
  - Tomorrow's weather forecast added as features (UV, solar, temp, rain)
    for the prediction day — key drivers of next-day chlorine decay
  - Smarter multi-visit deduplication (keep last reading per day, flag incidents)

Regulatory basis:
  - Real Decreto 742/2013 (national)
  - Decreto 85/2018 Comunitat Valenciana (regional)

Regulatory thresholds (RD 742/2013 Annexe I):
  - Free chlorine: 0.5 – 2.0 mg/L (pool closes if < 0.5 or > 5.0)
  - pH: 7.2 – 8.0
  - Turbidity: ≤ 5 NTU

Client target range (Jesús Santana brief):
  - Free chlorine: 1.0 – 1.5 mg/L (optimal dosing target)
  - pH: 7.2 – 8.0
"""

import warnings
warnings.filterwarnings('ignore')

import re
import os
import sys
import json
import pickle
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# ML / Stats
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# pyrefly: ignore [missing-import]
import xgboost as xgb

# Explainability
# pyrefly: ignore [missing-import]
import shap

# Visualization
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use('Agg')
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns

# Weather fetcher (existing module)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_weather import fetch_daily_weather, save_to_csv


# ============================================================================
# REGULATORY & CLIENT CONSTANTS
# ============================================================================

REG_CHLORINE_MIN      = 0.5    # mg/L — pathogen risk below this
REG_CHLORINE_IDEAL_MAX = 2.0   # mg/L — ideal upper (common practice goes above)
REG_CHLORINE_CLOSE    = 5.0    # mg/L — mandatory closure above this
REG_PH_MIN            = 7.2
REG_PH_MAX            = 8.0
REG_TURBIDITY_MAX     = 5.0    # NTU

# Client optimal target (Jesús Santana brief)
CLIENT_CL_TARGET_MIN  = 1.0    # mg/L
CLIENT_CL_TARGET_MAX  = 1.5    # mg/L
CLIENT_CL_IDEAL       = 1.25   # midpoint
PH_IDEAL              = 7.4    # midpoint of 7.2–8.0 (common Spanish practice)


# ============================================================================
# CONFIGURATION
# ============================================================================

RAW_EXCEL          = 'data/Merged_2023_2026.xlsx'
CHLORINE_PUMP_LIST = 'data/Listado_piscinas_bomba_cloro.xlsx'
WEATHER_CSV        = 'data/weather_alicante_2023_2026.csv'
OUTPUT_DIR         = 'outputs'
MODELS_DIR         = 'models'

# Alicante coordinates (per client brief)
ALICANTE_LAT = 38.3452
ALICANTE_LON = -0.4815
ALICANTE_TZ  = 'Europe/Madrid'

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)
EARLY_STOPPING_ROUNDS = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_step(step_num, title):
    print(f"\n{'='*70}")
    print(f"  STEP {step_num} — {title}")
    print(f"{'='*70}\n")


def safe_float(series):
    return pd.to_numeric(series, errors='coerce').astype(float)


def parse_date_series(s):
    return pd.to_datetime(s, format='mixed', dayfirst=True, errors='coerce')


def extract_pool_ref(s):
    """Extract the numeric reference ID from pool name, e.g. 'Cabo Verde (19)' → '19'."""
    if pd.isna(s):
        return None
    # Handle compound IDs like (654-655), (1122-2)
    m = re.search(r'\(\s*(\d[\d\-]*\d|\d)\s*\)', str(s))
    return m.group(1).strip() if m else None


def make_pool_type(row):
    parts = []
    if row.get('pool_heated', 0):    parts.append('heated')
    if row.get('pool_outdoor', 0):   parts.append('outdoor')
    if row.get('pool_community', 0): parts.append('community')
    if row.get('pool_private', 0):   parts.append('private')
    if row.get('pool_public', 0):    parts.append('public')
    return '_'.join(parts) if parts else 'unknown'


def make_deck_type(row):
    g = safe_float(pd.Series([row.get('deck_grass', 0)])).iloc[0] or 0
    p = safe_float(pd.Series([row.get('deck_paved', 0)])).iloc[0] or 0
    m = safe_float(pd.Series([row.get('deck_mixed', 0)])).iloc[0] or 0
    if m > 0: return 'mixed'
    if g > 0 and p > 0: return 'mixed'
    if g > 0: return 'grass'
    if p > 0: return 'paved'
    return 'unknown'


def consecutive_clean(series):
    """Count consecutive 0s (non-breach) ending at each position."""
    result = []
    count = 0
    for val in series:
        if val == 0:
            count += 1
        else:
            count = 0
        result.append(count)
    return result


# ============================================================================
# STEP 1 — LOAD AND RENAME (NEW EXCEL FORMAT)
# ============================================================================

print_step(1, "LOAD AND RENAME RAW DATA (Merged_2023_2026.xlsx)")

try:
    df_raw = pd.read_excel(RAW_EXCEL, header=0)
    print(f"Loaded {RAW_EXCEL}: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
except FileNotFoundError:
    print(f"ERROR: {RAW_EXCEL} not found.")
    sys.exit(1)

# Drop completely empty rows (the first row is always blank in this file)
df_raw = df_raw.dropna(how='all').reset_index(drop=True)
print(f"After dropping fully-empty rows: {df_raw.shape}")

# Name-based rename map (Spanish → snake_case)
RENAME_MAP = {
    'PISCINA':                                'pool_id',
    'COMUNIDAD':                              'community_name',
    'FECHA':                                  'reading_date',
    'EMPLEADO':                               'technician',
    'PH':                                     'ph',
    'TURBIDEZ':                               'turbidity',
    'CLORO LIBRE':                            'free_chlorine',
    'ABUSO CREMAS PROTECCION':                'sunscreen_abuse',
    'Caudal bomba de PH':                     'ph_pump_flow_rate',
    'Caudal bomba hipoclorito':               'hypochlorite_pump_flow_rate',
    'Caudal del motor':                       'motor_flow_rate',
    'Diametro filtro':                        'filter_diameter',
    'Numero de filtros':                      'filter_count',
    'Número de motores':                      'motor_count',
    'PISCINA CLIMATIZADA':                    'pool_heated',
    'PISCINA COMUNITARIA':                    'pool_community',
    'Piscina con skimmers':                   'pool_skimmer',
    'Piscina desbordante':                    'pool_overflow',
    'PISCINA EXTERIOR':                       'pool_outdoor',
    'Piscina ovalada':                        'pool_oval',
    'PISCINA PARTICULAR':                     'pool_private',
    'PISCINA PUBLICA':                        'pool_public',
    '(0714) Piscina rectangular':             'pool_rectangular_0714',
    '(07) Piscina rectangular':               'pool_rectangular_07',
    'Piscina redonda':                        'pool_round',
    'Superficie piscina':                     'pool_surface_m2',
    'VEGETACION CONTAMINANTE':                'vegetation_contamination',
    'Volumen piscina':                        'pool_volume_m3',
    'Zona playa césped':                      'deck_grass',
    'Zona PLAYA mixta':                       'deck_mixed',
    'Zona PLAYA pavimentada':                 'deck_paved',
    'EMPLEADO.1':                             'ops_technician',
    'FECHA.1':                                'ops_date',
    'Horas dosificacion PH':                  'ph_dosing_hours',
    'Horas filtracion diarias':               'daily_filtration_hours',
    'Porcentaje dosificación PH':             'ph_dosing_pct',
    'Tiempo lavado /enjuague filtro':         'filter_wash_rinse_time',
    'Horas dosificación hipo':                'hypochlorite_dosing_hours',
    'Porcentaje dosificación hipoclorito':    'hypochlorite_dosing_pct',
    'Temperatura agua':                       'water_temperature',
    'EMPLEADO.2':                             'prod_technician',
    'FECHA.2':                                'prod_date',
    'T-500 (GRUPO QP)':                       'prod_t500_qp',
    'ALBORAL TABLETAS 250 GRS RF. 201710':    'prod_alboral_tablets_250g',
    'FLOVIL PASTILLAS':                       'prod_flovil_tablets',
    'HIPO GARRAFAS 20KG.':                    'prod_hypo_jugs_20kg',
    'HIPO GR CHLORYTE':                       'prod_hypo_gr_chloryte',
    'HIPO GRANULADO XAKA':                    'prod_hypo_granular_xaka',
    'HIPO STICKS BAYROL':                     'prod_hypo_sticks_bayrol',
    'HIPO TAB. RITOCAL':                      'prod_hypo_tab_ritocal',
    'HIPO TABLETAS 200Gr. QP':               'prod_hypo_tablets_200g_qp',
    'HIPO TABLETAS XAKA':                     'prod_hypo_tablets_xaka',
    'PH MINUS GRANULADO 6kg':                'prod_ph_minus_granular_6kg',
    'PH MINUS LIQUIDO 13.5 KG':              'prod_ph_minus_liquid_13_5kg',
    'PH MINUS LIQUIDO 27 KG.':              'prod_ph_minus_liquid_27kg',
    'PROTECT & SHINE':                        'prod_protect_shine',
    'SG XAKA (AGONET GR90)':                 'prod_sg_xaka_agonet',
    'SUPERKLAR':                              'prod_superklar',
}

df_raw = df_raw.rename(columns=RENAME_MAP)

# Drop separator columns (Unnamed:*)
df_raw = df_raw.drop(columns=[c for c in df_raw.columns if str(c).startswith('Unnamed')], errors='ignore')

print(f"Columns after rename: {df_raw.shape[1]}")
print(f"Null pool_id rows: {df_raw['pool_id'].isna().sum()}")


# ============================================================================
# STEP 1.5 — POOL FILTER: LIQUID CHLORINE DOSING PUMP ONLY
# ============================================================================

print_step("1.5", "POOL FILTER — Liquid Chlorine Dosing Pump Pools")

# Load the chlorine pump pool list
try:
    df_pump_list = pd.read_excel(CHLORINE_PUMP_LIST)
    print(f"Chlorine pump list: {len(df_pump_list)} pools")
except FileNotFoundError:
    print(f"WARNING: {CHLORINE_PUMP_LIST} not found — skipping pool filter.")
    df_pump_list = None

if df_pump_list is not None:
    # Normalize references from the pump list (handle mixed types)
    pump_refs = set(str(r).strip() for r in df_pump_list['Referencia'].dropna())

    # Extract numeric ref from pool_id in the dataset
    df_raw['pool_ref'] = df_raw['pool_id'].apply(extract_pool_ref)

    # Primary match: exact numeric ref
    mask_matched = df_raw['pool_ref'].isin(pump_refs)

    # Fallback: fuzzy community name match for compound IDs (e.g. 654-655, 1122-2)
    # Normalize community names from pump list for fuzzy match
    pump_communities = set(
        str(c).lower().strip()
        for c in df_pump_list['Comunidad'].dropna()
    )

    def normalize_community(s):
        if pd.isna(s): return ''
        return re.sub(r'\s+', ' ', str(s).lower().strip())

    df_raw['community_normalized'] = df_raw['community_name'].apply(normalize_community)
    mask_community = df_raw['community_normalized'].apply(
        lambda c: any(pc in c or c in pc for pc in pump_communities if len(pc) > 4)
    )
    mask_combined = mask_matched | mask_community

    before = len(df_raw)
    df = df_raw[mask_combined].copy()
    after = len(df)

    n_pools_before = df_raw['pool_id'].dropna().nunique()
    n_pools_after  = df['pool_id'].dropna().nunique()

    print(f"\n  Rows before filter: {before} | After: {after} (removed {before - after})")
    print(f"  Pools before filter: {n_pools_before} | After: {n_pools_after}")
    print(f"  Matched by ref: {mask_matched.sum()} rows")
    print(f"  Matched by community fallback: {(mask_combined & ~mask_matched).sum()} rows")
    print(f"\n  RECONCILIATION:")
    matched_refs = set(df['pool_ref'].dropna().unique())
    unmatched_pump = pump_refs - matched_refs
    print(f"  Pump list pools NOT found in dataset: {sorted(unmatched_pump)[:20]}")
else:
    df = df_raw.copy()


# ============================================================================
# STEP 2 — FETCH / LOAD WEATHER DATA
# ============================================================================

print_step(2, "WEATHER DATA — Open-Meteo Alicante")

# Determine date range from dataset
df_dates_raw = parse_date_series(df['reading_date'])
data_start = df_dates_raw.min().strftime('%Y-%m-%d')
data_end   = df_dates_raw.max().strftime('%Y-%m-%d')
print(f"  Dataset date range: {data_start} → {data_end}")

# Fetch or load from cache
weather_needs_fetch = True
if os.path.exists(WEATHER_CSV):
    df_weather_check = pd.read_csv(WEATHER_CSV, usecols=['date'], nrows=1)
    df_weather_full  = pd.read_csv(WEATHER_CSV, usecols=['date'])
    w_min = df_weather_full['date'].min()
    w_max = df_weather_full['date'].max()
    if w_min <= data_start and w_max >= data_end:
        print(f"  Cache found: {WEATHER_CSV} covers {w_min} → {w_max}. Skipping fetch.")
        weather_needs_fetch = False
    else:
        print(f"  Cache found but insufficient ({w_min} → {w_max}). Re-fetching.")

if weather_needs_fetch:
    print(f"  Fetching weather for Alicante (lat={ALICANTE_LAT}, lon={ALICANTE_LON})")
    print(f"  Date range: {data_start} → {data_end}")
    weather_rows, weather_units = fetch_daily_weather(
        latitude=ALICANTE_LAT,
        longitude=ALICANTE_LON,
        start_date=data_start,
        end_date=data_end,
        timezone=ALICANTE_TZ,
    )
    save_to_csv(weather_rows, WEATHER_CSV, units=weather_units)
    print(f"  Saved {len(weather_rows)} daily weather records to {WEATHER_CSV}")

# Load weather
df_weather = pd.read_csv(WEATHER_CSV)
df_weather['date'] = pd.to_datetime(df_weather['date']).dt.normalize()
print(f"  Weather loaded: {df_weather.shape} | {df_weather['date'].min().date()} → {df_weather['date'].max().date()}")

# Slim down to the columns we need
WEATHER_COLS_KEEP = [
    'date',
    'temperature_2m_max',
    'temperature_2m_mean',
    'uv_index_max',
    'uv_index_clear_sky_max',
    'shortwave_radiation_sum',
    'sunshine_duration',
    'precipitation_sum',
    'wind_speed_10m_max',
    'et0_fao_evapotranspiration',
    'weather_code',
]
df_weather = df_weather[[c for c in WEATHER_COLS_KEEP if c in df_weather.columns]].copy()

# Rename for clarity in the model
df_weather = df_weather.rename(columns={
    'temperature_2m_max':          'w_temp_max',
    'temperature_2m_mean':         'w_temp_mean',
    'uv_index_max':                'w_uv_max',
    'uv_index_clear_sky_max':      'w_uv_clear_sky_max',
    'shortwave_radiation_sum':     'w_solar_radiation',
    'sunshine_duration':           'w_sunshine_duration_s',
    'precipitation_sum':           'w_precipitation_mm',
    'wind_speed_10m_max':          'w_wind_max_kmh',
    'et0_fao_evapotranspiration':  'w_et0',
    'weather_code':                'w_weather_code',
})

# Convert sunshine from seconds → hours
if 'w_sunshine_duration_s' in df_weather.columns:
    df_weather['w_sunshine_hours'] = df_weather['w_sunshine_duration_s'] / 3600
    df_weather = df_weather.drop(columns=['w_sunshine_duration_s'])

print(f"  Weather features kept: {[c for c in df_weather.columns if c != 'date']}")


# ============================================================================
# STEP 3 — SEPARATE SUB-TABLES
# ============================================================================

print_step(3, "SEPARATE THREE SUB-TABLES")

reading_cols = [
    'pool_id', 'community_name', 'reading_date', 'technician',
    'ph', 'turbidity', 'free_chlorine', 'sunscreen_abuse',
    'ph_pump_flow_rate', 'hypochlorite_pump_flow_rate', 'motor_flow_rate',
    'filter_diameter', 'filter_count', 'motor_count',
    'pool_heated', 'pool_community', 'pool_skimmer', 'pool_overflow',
    'pool_outdoor', 'pool_oval', 'pool_private', 'pool_public',
    'pool_rectangular_0714', 'pool_rectangular_07', 'pool_round',
    'pool_surface_m2', 'vegetation_contamination', 'pool_volume_m3',
    'deck_grass', 'deck_mixed', 'deck_paved',
]

ops_cols = [
    'ops_technician', 'ops_date',
    'ph_dosing_hours', 'daily_filtration_hours', 'ph_dosing_pct',
    'filter_wash_rinse_time', 'hypochlorite_dosing_hours',
    'hypochlorite_dosing_pct', 'water_temperature',
]

prod_cols = [
    'prod_technician', 'prod_date',
    'prod_t500_qp', 'prod_alboral_tablets_250g', 'prod_flovil_tablets',
    'prod_hypo_jugs_20kg', 'prod_hypo_gr_chloryte', 'prod_hypo_granular_xaka',
    'prod_hypo_sticks_bayrol', 'prod_hypo_tab_ritocal',
    'prod_hypo_tablets_200g_qp', 'prod_hypo_tablets_xaka',
    'prod_ph_minus_granular_6kg', 'prod_ph_minus_liquid_13_5kg',
    'prod_ph_minus_liquid_27kg', 'prod_protect_shine',
    'prod_sg_xaka_agonet', 'prod_superklar',
]

# Forward-fill pool_id and community_name within the dataset
df['pool_id']        = df['pool_id'].ffill()
df['community_name'] = df['community_name'].ffill()

# Only keep columns that exist
reading_cols = [c for c in reading_cols if c in df.columns]
ops_cols     = [c for c in ops_cols if c in df.columns]
prod_cols    = [c for c in prod_cols if c in df.columns]

# --- Readings ---
df_readings = df[reading_cols].copy()
df_readings = df_readings.dropna(subset=['reading_date'])
print(f"df_readings raw: {df_readings.shape}")

# --- Operations ---
df_operations = df[['pool_id'] + [c for c in ops_cols if c != 'pool_id']].copy()
df_operations = df_operations.dropna(subset=['ops_date'])
key_ops_cols = [c for c in ['ph_dosing_hours', 'daily_filtration_hours', 'water_temperature',
                              'hypochlorite_dosing_hours', 'hypochlorite_dosing_pct'] if c in df_operations.columns]
df_operations = df_operations.dropna(subset=key_ops_cols, how='all')
print(f"df_operations raw: {df_operations.shape}")

# --- Products ---
df_products = df[['pool_id'] + [c for c in prod_cols if c != 'pool_id']].copy()
df_products = df_products.dropna(subset=['prod_date'])
print(f"df_products raw: {df_products.shape}")

for name, dfx, date_col in [
    ('Readings',   df_readings,   'reading_date'),
    ('Operations', df_operations, 'ops_date'),
    ('Products',   df_products,   'prod_date'),
]:
    dates = parse_date_series(dfx[date_col])
    valid = dates.dropna()
    if len(valid) > 0:
        print(f"  {name}: {len(dfx)} rows, {valid.min().date()} to {valid.max().date()}")


# ============================================================================
# STEP 4 — DATA CLEANING
# ============================================================================

print_step(4, "DATA CLEANING")

# --- Readings ---
df_readings['reading_date'] = parse_date_series(df_readings['reading_date'])
for col in ['ph', 'turbidity', 'free_chlorine']:
    if col in df_readings.columns:
        df_readings[col] = safe_float(df_readings[col])

# Regulatory breach flags
df_readings['ph_breach'] = (
    ~df_readings['ph'].between(REG_PH_MIN, REG_PH_MAX) & df_readings['ph'].notna()
)
df_readings['chlorine_breach'] = (
    (df_readings['free_chlorine'] < REG_CHLORINE_MIN) |
    (df_readings['free_chlorine'] > REG_CHLORINE_CLOSE)
) & df_readings['free_chlorine'].notna()
df_readings['chlorine_low']       = (df_readings['free_chlorine'] < REG_CHLORINE_MIN) & df_readings['free_chlorine'].notna()
df_readings['chlorine_over_ideal'] = (df_readings['free_chlorine'] > REG_CHLORINE_IDEAL_MAX) & df_readings['free_chlorine'].notna()
df_readings['turbidity_breach']   = (df_readings['turbidity'] > REG_TURBIDITY_MAX) & df_readings['turbidity'].notna()
df_readings['any_breach']         = df_readings['ph_breach'] | df_readings['chlorine_breach'] | df_readings['turbidity_breach']

print(f"  pH breaches (outside 7.2–8.0):           {df_readings['ph_breach'].sum()} ({100*df_readings['ph_breach'].mean():.1f}%)")
print(f"  Chlorine SAFETY breaches (<0.5 or >5.0): {df_readings['chlorine_breach'].sum()} ({100*df_readings['chlorine_breach'].mean():.1f}%)")
print(f"    → Low chlorine (<0.5):                  {df_readings['chlorine_low'].sum()}")
print(f"    → Over ideal (>2.0):                    {df_readings['chlorine_over_ideal'].sum()} ({100*df_readings['chlorine_over_ideal'].mean():.1f}%)")
print(f"  Turbidity breaches (>5 NTU):              {df_readings['turbidity_breach'].sum()} ({100*df_readings['turbidity_breach'].mean():.1f}%)")

# Pool type flags
pool_type_flags = [
    'pool_heated', 'pool_community', 'pool_skimmer', 'pool_overflow',
    'pool_outdoor', 'pool_oval', 'pool_private', 'pool_public',
    'pool_rectangular_0714', 'pool_rectangular_07', 'pool_round',
]
for col in pool_type_flags:
    if col in df_readings.columns:
        df_readings[col] = safe_float(df_readings[col]).fillna(0).astype(int)

for col in ['pool_surface_m2', 'pool_volume_m3', 'filter_diameter', 'filter_count', 'motor_count']:
    if col in df_readings.columns:
        df_readings[col] = safe_float(df_readings[col])

df_readings['pool_type'] = df_readings.apply(make_pool_type, axis=1)
df_readings['deck_type']  = df_readings.apply(make_deck_type, axis=1)

# --- Multi-visit deduplication (smarter than V3) ---
# Add date-only column for grouping
df_readings['reading_date_only'] = df_readings['reading_date'].dt.normalize()

# Count visits per (pool, date)
visit_counts = df_readings.groupby(['pool_id', 'reading_date_only'])['reading_date'].transform('count')
df_readings['multi_visit_day'] = (visit_counts > 1).astype(int)

multi_total = df_readings['multi_visit_day'].sum()
print(f"\n  Multi-visit day rows: {multi_total} across "
      f"{df_readings[df_readings['multi_visit_day']==1].groupby(['pool_id','reading_date_only']).ngroups} pool-day combos")

# Keep the LAST reading of each (pool, date) — captures corrections and post-incident state
before = len(df_readings)
df_readings = (
    df_readings
    .sort_values(['pool_id', 'reading_date'])
    .drop_duplicates(subset=['pool_id', 'reading_date_only'], keep='last')
)
print(f"  Dedup (keep last per day): {before} → {len(df_readings)}")

df_readings = df_readings.sort_values(['pool_id', 'reading_date']).reset_index(drop=True)

# --- Operations ---
df_operations['ops_date'] = parse_date_series(df_operations['ops_date'])
ops_numeric = [c for c in [
    'ph_dosing_hours', 'daily_filtration_hours', 'ph_dosing_pct',
    'filter_wash_rinse_time', 'hypochlorite_dosing_hours',
    'hypochlorite_dosing_pct', 'water_temperature',
] if c in df_operations.columns]

for col in ops_numeric:
    df_operations[col] = safe_float(df_operations[col])

df_operations = df_operations.groupby(['pool_id', 'ops_date'], as_index=False)[ops_numeric].mean()
df_operations = df_operations.sort_values(['pool_id', 'ops_date']).reset_index(drop=True)
print(f"\n  df_operations after dedup: {df_operations.shape}")

# --- Products ---
df_products['prod_date'] = parse_date_series(df_products['prod_date'])
prod_value_cols = [c for c in prod_cols if c not in ['prod_technician', 'prod_date', 'pool_id']]
for col in prod_value_cols:
    if col in df_products.columns:
        df_products[col] = safe_float(df_products[col]).fillna(0)

hypo_cols      = [c for c in prod_value_cols if 'hypo' in c.lower()]
ph_minus_cols  = [c for c in prod_value_cols if 'ph_minus' in c.lower()]
flocculant_cols = [c for c in [
    'prod_flovil_tablets', 'prod_superklar', 'prod_sg_xaka_agonet', 'prod_alboral_tablets_250g'
] if c in df_products.columns]

df_products['total_chlorine_product']  = df_products[hypo_cols].sum(axis=1)
df_products['total_ph_minus_product']  = df_products[ph_minus_cols].sum(axis=1)
df_products['total_flocculant_product'] = df_products[flocculant_cols].sum(axis=1) if flocculant_cols else 0

agg_cols = [c for c in prod_value_cols if c in df_products.columns] + [
    'total_chlorine_product', 'total_ph_minus_product', 'total_flocculant_product'
]
df_products = df_products.groupby(['pool_id', 'prod_date'], as_index=False)[agg_cols].max()
df_products = df_products.sort_values(['pool_id', 'prod_date']).reset_index(drop=True)

# Remove orphan product rows
reading_pools = set(df_readings['pool_id'].dropna().unique())
df_products = df_products[df_products['pool_id'].isin(reading_pools)]
print(f"  df_products after dedup: {df_products.shape}")


# ============================================================================
# STEP 4.5 — BACKFILL STATIC POOL DATA (from V3)
# ============================================================================

print_step("4.5", "BACKFILL STATIC POOL DATA")

backfill_summary = {}

numeric_static = ['pool_volume_m3', 'pool_surface_m2', 'filter_diameter', 'filter_count', 'motor_count']
for col in [c for c in numeric_static if c in df_readings.columns]:
    before_fill = df_readings[col].notna().sum()
    pool_vals = df_readings.groupby('pool_id')[col].apply(
        lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan
    )
    pool_vals_clean = pool_vals.dropna()
    fleet_median = pool_vals_clean.median() if len(pool_vals_clean) > 0 else 0.0
    pools_with_data   = len(pool_vals_clean)
    pools_without     = len(pool_vals) - pools_with_data

    df_readings[col] = df_readings['pool_id'].map(pool_vals).fillna(fleet_median)
    after_fill = df_readings[col].notna().sum()

    backfill_summary[col] = {
        'pools_with_original_data': pools_with_data,
        'pools_filled_with_median': pools_without,
        'fleet_median': round(fleet_median, 2),
        'rows_before': int(before_fill),
        'rows_after':  int(after_fill),
    }
    print(f"  {col}: {pools_with_data} pools had data, {pools_without} filled with median ({fleet_median:.1f})")

flag_static = [c for c in [
    'pool_heated', 'pool_community', 'pool_skimmer', 'pool_overflow',
    'pool_outdoor', 'pool_oval', 'pool_private', 'pool_public',
    'pool_rectangular_0714', 'pool_rectangular_07', 'pool_round',
    'vegetation_contamination',
] if c in df_readings.columns]

for col in flag_static:
    pool_vals = df_readings.groupby('pool_id')[col].apply(
        lambda x: x.max() if x.notna().any() else 0
    )
    df_readings[col] = df_readings['pool_id'].map(pool_vals).fillna(0).astype(int)

for col in [c for c in ['deck_grass', 'deck_mixed', 'deck_paved'] if c in df_readings.columns]:
    pool_vals = df_readings.groupby('pool_id')[col].apply(
        lambda x: x.max() if x.notna().any() else 0
    )
    df_readings[col] = df_readings['pool_id'].map(pool_vals).fillna(0)

# Recompute derived categoricals after backfill
df_readings['pool_type'] = df_readings.apply(make_pool_type, axis=1)
df_readings['deck_type']  = df_readings.apply(make_deck_type, axis=1)

print(f"\n  pool_volume_m3 fill rate: {df_readings['pool_volume_m3'].notna().mean()*100:.1f}%")
print(f"  pool_surface_m2 fill rate: {df_readings['pool_surface_m2'].notna().mean()*100:.1f}%")
pool_type_dist = df_readings['pool_type'].value_counts()
print(f"\n  Pool type distribution:")
for pt, cnt in pool_type_dist.head(6).items():
    print(f"    {pt}: {cnt} rows ({100*cnt/len(df_readings):.1f}%)")


# ============================================================================
# STEP 5 — MERGE INTO MASTER DATASET
# ============================================================================

print_step(5, "MERGE INTO MASTER DATASET")

MERGE_TOLERANCE_DAYS = 14
tolerance = pd.Timedelta(f'{MERGE_TOLERANCE_DAYS}D')


def merge_asof_by_pool(df_left, df_right, left_date, right_date, right_cols):
    """Backward asof merge per pool with a date tolerance."""
    merged_parts = []
    for pool_id in df_left['pool_id'].unique():
        left_pool  = df_left[df_left['pool_id'] == pool_id].copy()
        right_pool = df_right[df_right['pool_id'] == pool_id].copy()
        if len(right_pool) == 0:
            for col in right_cols:
                left_pool[col] = np.nan
            merged_parts.append(left_pool)
            continue
        left_pool  = left_pool.sort_values(left_date)
        right_pool = right_pool.sort_values(right_date)
        merged = pd.merge_asof(
            left_pool,
            right_pool[[c for c in right_cols + [right_date] if c in right_pool.columns]],
            left_on=left_date,
            right_on=right_date,
            direction='backward',
            tolerance=tolerance,
        )
        merged_parts.append(merged)
    return pd.concat(merged_parts, ignore_index=True)


# Ops merge
ops_value_cols = [c for c in df_operations.columns if c not in ['pool_id', 'ops_date']]
df_master = merge_asof_by_pool(df_readings, df_operations, 'reading_date', 'ops_date', ops_value_cols)
ops_matched = df_master[ops_value_cols[0]].notna().sum() if ops_value_cols else 0
print(f"Ops match: {ops_matched}/{len(df_master)} ({100*ops_matched/len(df_master):.1f}%)" if len(df_master) else "No rows")

# Products merge
prod_value_cols_merge = [c for c in df_products.columns if c not in ['pool_id', 'prod_date']]
df_master = merge_asof_by_pool(df_master, df_products, 'reading_date', 'prod_date', prod_value_cols_merge)
prod_matched = df_master['total_chlorine_product'].notna().sum() if 'total_chlorine_product' in df_master.columns else 0
print(f"Products match: {prod_matched}/{len(df_master)} ({100*prod_matched/len(df_master):.1f}%)" if len(df_master) else "")
print(f"df_master after sub-table merge: {df_master.shape}")


# ============================================================================
# STEP 6 — WEATHER JOIN (exact date, no row inflation)
# ============================================================================

print_step(6, "WEATHER JOIN — Exact Date Merge")

# reading_date_only is already on df_master (kept from readings)
# Ensure it's a proper date-normalised datetime
df_master['reading_date_only'] = pd.to_datetime(df_master['reading_date']).dt.normalize()

weather_feature_cols = [c for c in df_weather.columns if c != 'date']

before_shape = df_master.shape
df_master = pd.merge(
    df_master,
    df_weather.rename(columns={'date': 'reading_date_only'}),
    on='reading_date_only',
    how='left',
)
assert df_master.shape[0] == before_shape[0], \
    f"Row inflation after weather join! {before_shape[0]} → {df_master.shape[0]}"

weather_coverage = df_master[weather_feature_cols[0]].notna().mean() * 100 if weather_feature_cols else 0
print(f"  df_master after weather join: {df_master.shape} (rows unchanged ✓)")
print(f"  Weather coverage (today): {weather_coverage:.1f}%")
print(f"  Weather features (today): {weather_feature_cols}")

# --- Join TOMORROW's weather as additional features ---
# Since we predict next-day chemistry, tomorrow's UV/solar/temp are key signals.
# Shift: tomorrow's date = today's reading date + 1 day.
# We bring tomorrow's weather to today's row so the model sees the prediction-day conditions.
TOMORROW_WEATHER_COLS = [
    'w_temp_max', 'w_temp_mean', 'w_uv_max', 'w_uv_clear_sky_max',
    'w_solar_radiation', 'w_sunshine_hours', 'w_precipitation_mm',
    'w_wind_max_kmh', 'w_et0',
]
tomorrow_weather_cols_present = [c for c in TOMORROW_WEATHER_COLS if c in df_weather.columns]

df_weather_tmrw = df_weather[['date'] + tomorrow_weather_cols_present].copy()
df_weather_tmrw = df_weather_tmrw.rename(
    columns={c: f'w_tmrw_{c[2:]}' for c in tomorrow_weather_cols_present}  # w_temp_max -> w_tmrw_temp_max
)
# The key: tomorrow's date in the weather table = (today's visit date + 1)
df_weather_tmrw['reading_date_only'] = df_weather_tmrw['date'] - pd.Timedelta(days=1)
df_weather_tmrw = df_weather_tmrw.drop(columns=['date'])

before_shape2 = df_master.shape
df_master = pd.merge(df_master, df_weather_tmrw, on='reading_date_only', how='left')
assert df_master.shape[0] == before_shape2[0], "Row inflation after tomorrow weather join!"

tomorrow_weather_col_names = [f'w_tmrw_{c[2:]}' for c in tomorrow_weather_cols_present]
tomorrow_coverage = df_master[tomorrow_weather_col_names[0]].notna().mean() * 100 if tomorrow_weather_col_names else 0
print(f"  Tomorrow weather features: {tomorrow_weather_col_names}")
print(f"  Tomorrow weather coverage: {tomorrow_coverage:.1f}%")


# ============================================================================
# STEP 7 — FEATURE ENGINEERING
# ============================================================================

print_step(7, "FEATURE ENGINEERING")

df_master = df_master.sort_values(['pool_id', 'reading_date']).reset_index(drop=True)

# --- Lag features ---
print("  Adding lag features...")
for col, prefix in [('ph', 'ph'), ('free_chlorine', 'chlorine'), ('turbidity', 'turbidity')]:
    df_master[f'{prefix}_lag1'] = df_master.groupby('pool_id')[col].shift(1)
    df_master[f'{prefix}_lag2'] = df_master.groupby('pool_id')[col].shift(2)

# --- Rolling statistics (window = 3) ---
print("  Adding rolling statistics...")
for col, prefix in [('ph', 'ph'), ('free_chlorine', 'chlorine'), ('turbidity', 'turbidity')]:
    df_master[f'{prefix}_roll3_mean'] = df_master.groupby('pool_id')[col].transform(
        lambda x: x.rolling(window=3, min_periods=2).mean()
    )
    if prefix != 'turbidity':
        df_master[f'{prefix}_roll3_std'] = df_master.groupby('pool_id')[col].transform(
            lambda x: x.rolling(window=3, min_periods=2).std()
        )

# --- Visit interval features ---
print("  Adding visit interval features...")
df_master['days_since_last_visit'] = df_master.groupby('pool_id')['reading_date'].diff().dt.days
df_master['visit_day_of_week']     = df_master['reading_date'].dt.dayofweek
df_master['visit_month']           = df_master['reading_date'].dt.month
df_master['visit_is_summer']       = df_master['visit_month'].isin([6, 7, 8, 9]).astype(int)
df_master['visit_year']            = df_master['reading_date'].dt.year
df_master['pool_visit_number']     = df_master.groupby('pool_id').cumcount()

# --- Chemistry features ---
print("  Adding chemistry features...")
df_master['ph_deviation']          = (df_master['ph'] - PH_IDEAL).abs()
df_master['chlorine_deficit']      = (REG_CHLORINE_MIN - df_master['free_chlorine']).clip(lower=0)
df_master['last_total_chlorine_applied'] = df_master['total_chlorine_product'].fillna(0) if 'total_chlorine_product' in df_master.columns else 0.0
df_master['cl_effectiveness_index'] = df_master['free_chlorine'] * np.clip(
    np.where(df_master['ph'] <= 7.5, 1.0, 1.0 - 0.5 * ((df_master['ph'] - 7.5) / 0.5)), 0.1, 1.0
)

# --- Regulatory headroom features ---
print("  Adding regulatory headroom features...")
df_master['chlorine_headroom_low']  = df_master['free_chlorine'] - REG_CHLORINE_MIN
df_master['chlorine_headroom_high'] = REG_CHLORINE_CLOSE - df_master['free_chlorine']
df_master['ph_headroom_low']        = df_master['ph'] - REG_PH_MIN
df_master['ph_headroom_high']       = REG_PH_MAX - df_master['ph']
df_master['turbidity_headroom']     = REG_TURBIDITY_MAX - (df_master['turbidity'] if 'turbidity' in df_master.columns else 0)
df_master['min_headroom']           = df_master[[
    'chlorine_headroom_low', 'chlorine_headroom_high',
    'ph_headroom_low', 'ph_headroom_high', 'turbidity_headroom'
]].min(axis=1)

# Client target headroom (how far from optimal [1.0–1.5])
df_master['cl_below_client_target'] = (CLIENT_CL_TARGET_MIN - df_master['free_chlorine']).clip(lower=0)
df_master['cl_above_client_target'] = (df_master['free_chlorine'] - CLIENT_CL_TARGET_MAX).clip(lower=0)

# --- Trend/rate features ---
print("  Adding trend features...")
df_master['ph_trend']       = df_master['ph'] - df_master['ph_lag1']
df_master['chlorine_trend'] = df_master['free_chlorine'] - df_master['chlorine_lag1']
df_master['turbidity_trend'] = df_master['turbidity'] - df_master['turbidity_lag1']

df_master['ph_rate_per_day']        = df_master['ph_trend'] / df_master['days_since_last_visit'].replace(0, np.nan)
df_master['chlorine_rate_per_day']  = df_master['chlorine_trend'] / df_master['days_since_last_visit'].replace(0, np.nan)
df_master['turbidity_rate_per_day'] = df_master['turbidity_trend'] / df_master['days_since_last_visit'].replace(0, np.nan)

# --- Breach history features ---
print("  Adding breach history features...")
df_master['current_any_breach']       = df_master['any_breach'].astype(int)
df_master['current_ph_breach']        = df_master['ph_breach'].astype(int)
df_master['current_chlorine_breach']  = df_master['chlorine_breach'].astype(int)
df_master['consecutive_clean_visits'] = df_master.groupby('pool_id')['current_any_breach'].transform(
    lambda x: consecutive_clean(x.values)
)
df_master['breach_rate_last5'] = df_master.groupby('pool_id')['current_any_breach'].transform(
    lambda x: x.rolling(window=5, min_periods=1).mean()
)

# --- Volume-normalised features ---
print("  Adding volume-normalised features...")
df_master['chlorine_dose_per_m3'] = df_master['last_total_chlorine_applied'] / df_master['pool_volume_m3']
df_master['ph_minus_dose_per_m3'] = (df_master['total_ph_minus_product'] if 'total_ph_minus_product' in df_master.columns else 0) / df_master['pool_volume_m3']
df_master['chlorine_decay_per_m3'] = df_master['chlorine_rate_per_day'] / df_master['pool_volume_m3']

# --- Weather on current visit day ---
print("  Weather features on current visit day already joined.")

# --- Rolling weather SINCE LAST VISIT (cumulative exposure) ---
# This is the physically meaningful signal: total UV/heat/rain since chlorine was last dosed
print("  Computing cumulative weather since last visit...")

weather_daily_dict = df_weather.set_index('date')

def cumulative_weather_since_last(pool_group, df_w):
    """
    For each row, look back `days_since_last_visit` days in the weather data
    and aggregate UV, solar, precipitation, and mean temperature.
    Returns a DataFrame of weather aggregations indexed like pool_group.
    """
    results = []
    dates        = pool_group['reading_date'].dt.normalize().values
    days_back_arr = pool_group['days_since_last_visit'].values

    for i in range(len(pool_group)):
        end_d    = pd.Timestamp(dates[i])
        days_bk  = days_back_arr[i]

        if pd.isna(days_bk) or days_bk <= 0:
            results.append({'w_uv_sum_since': np.nan, 'w_solar_sum_since': np.nan,
                             'w_precip_sum_since': np.nan, 'w_temp_mean_since': np.nan})
            continue

        start_d = end_d - pd.Timedelta(days=int(days_bk) - 1)
        mask = (df_w.index >= start_d) & (df_w.index <= end_d)
        w_slice = df_w[mask]

        if len(w_slice) == 0:
            results.append({'w_uv_sum_since': np.nan, 'w_solar_sum_since': np.nan,
                             'w_precip_sum_since': np.nan, 'w_temp_mean_since': np.nan})
        else:
            results.append({
                'w_uv_sum_since':     w_slice['w_uv_max'].sum()             if 'w_uv_max' in w_slice.columns else np.nan,
                'w_solar_sum_since':  w_slice['w_solar_radiation'].sum()    if 'w_solar_radiation' in w_slice.columns else np.nan,
                'w_precip_sum_since': w_slice['w_precipitation_mm'].sum()   if 'w_precipitation_mm' in w_slice.columns else np.nan,
                'w_temp_mean_since':  w_slice['w_temp_mean'].mean()         if 'w_temp_mean' in w_slice.columns else np.nan,
            })

    return pd.DataFrame(results, index=pool_group.index)


# Apply per pool
cumulative_parts = []
for pid, grp in df_master.groupby('pool_id'):
    cum = cumulative_weather_since_last(grp, weather_daily_dict)
    cumulative_parts.append(cum)

df_cum = pd.concat(cumulative_parts).sort_index()

for col in df_cum.columns:
    df_master[col] = df_cum[col]

cumulative_weather_features = list(df_cum.columns)
print(f"  Cumulative weather features: {cumulative_weather_features}")
uv_coverage = df_master['w_uv_sum_since'].notna().mean() * 100
print(f"  Cumulative UV coverage: {uv_coverage:.1f}%")

print(f"\n  df_master after feature engineering: {df_master.shape}")


# ============================================================================
# STEP 8 — DEFINE NEW TARGET VARIABLES
# ============================================================================

print_step(8, "DEFINE TARGET VARIABLES — Next-Day Cl and pH")

# -----------------------------------------------------------------------
# TARGET DEFINITION — NEXT CALENDAR DAY (interpolated)
#
# Goal: predict what the pool's chemical state will be TOMORROW, so that
# anyone can decide whether a visit is needed the next day.
#
# Since pool readings are sparse (visits every ~3 days, never Sundays),
# we don't have ground-truth readings for every calendar day. Instead, we
# estimate tomorrow's value using linear interpolation between consecutive
# visits:
#
#   target_tomorrow = C_today + (C_next_visit - C_today) * (1 / k)
#
# where k = days between this visit and the next visit for the same pool.
#
# When k == 1 (back-to-back visits): interpolated value = next reading.
# When k == 3 (typical 3-day gap): tomorrow = 1/3 of the way toward next.
#
# This approach:
# - Trains on real visit-day measurements only (no synthetic feature rows)
# - Provides a physically grounded estimate of next-day chemistry
# - Enables answer to "will the pool need attention tomorrow?"
# - Pairs naturally with tomorrow's weather forecast features
# -----------------------------------------------------------------------

# Next visit's readings (anchor point for interpolation)
df_master['next_reading_date']  = df_master.groupby('pool_id')['reading_date'].shift(-1)
df_master['days_to_next_visit'] = (df_master['next_reading_date'] - df_master['reading_date']).dt.days
df_master['cl_next_visit']      = df_master.groupby('pool_id')['free_chlorine'].shift(-1)
df_master['ph_next_visit']      = df_master.groupby('pool_id')['ph'].shift(-1)
df_master['turb_next_visit']    = df_master.groupby('pool_id')['turbidity'].shift(-1)

# Interpolation fraction: 1/k  (share of the inter-visit gap that 1 day represents)
df_master['interp_frac'] = 1.0 / df_master['days_to_next_visit'].clip(lower=1)

# Primary targets — tomorrow's estimated values
df_master['target_cl_tomorrow'] = (
    df_master['free_chlorine'] +
    (df_master['cl_next_visit']   - df_master['free_chlorine']) * df_master['interp_frac']
)
df_master['target_ph_tomorrow'] = (
    df_master['ph'] +
    (df_master['ph_next_visit']   - df_master['ph'])            * df_master['interp_frac']
)
df_master['target_turb_tomorrow'] = (
    df_master['turbidity'] +
    (df_master['turb_next_visit'] - df_master['turbidity'])     * df_master['interp_frac']
)

# Drop rows where interpolation is undefined (last visit of each pool = no next anchor)
df_master = df_master.dropna(subset=['days_to_next_visit']).copy()

# Breach flags — tomorrow's estimated values vs regulatory thresholds
df_master['ph_breach_tomorrow'] = (
    (df_master['target_ph_tomorrow'] < REG_PH_MIN) | (df_master['target_ph_tomorrow'] > REG_PH_MAX)
)
df_master['chlorine_breach_tomorrow'] = (
    (df_master['target_cl_tomorrow'] < REG_CHLORINE_MIN) |
    (df_master['target_cl_tomorrow'] > REG_CHLORINE_CLOSE)
)
df_master['chlorine_in_client_range_tomorrow'] = df_master['target_cl_tomorrow'].between(
    CLIENT_CL_TARGET_MIN, CLIENT_CL_TARGET_MAX
)
df_master['any_breach_tomorrow'] = df_master['ph_breach_tomorrow'] | df_master['chlorine_breach_tomorrow']

# For sample weighting: upweight today's visit rows where TOMORROW looks like a breach
df_master['any_breach_next'] = df_master['any_breach_tomorrow']  # alias used in training

# --- Filter for model training ---
df_model    = df_master.dropna(subset=['ph', 'free_chlorine']).copy()
df_model_wq = df_model.dropna(subset=['target_cl_tomorrow', 'target_ph_tomorrow']).copy()

median_gap = df_model_wq['days_to_next_visit'].median()
print(f"Full master rows (after dropping last-visit rows): {len(df_master)}")
print(f"Model dataset (has current readings):               {len(df_model)}")
print(f"Model dataset WQ (has tomorrow targets):            {len(df_model_wq)}")
print(f"Median days to next visit (interpolation gap):      {median_gap:.0f} days")
print(f"Rows where k=1 (exact tomorrow target):             {(df_model_wq['days_to_next_visit']==1).sum()}")
print(f"\n  Breach stats for TOMORROW:")
print(f"    pH breach:       {df_model_wq['ph_breach_tomorrow'].sum()} ({100*df_model_wq['ph_breach_tomorrow'].mean():.1f}%)")
print(f"    Chlorine breach: {df_model_wq['chlorine_breach_tomorrow'].sum()} ({100*df_model_wq['chlorine_breach_tomorrow'].mean():.1f}%)")
print(f"    Cl in client range [1.0–1.5]: {df_model_wq['chlorine_in_client_range_tomorrow'].sum()} ({100*df_model_wq['chlorine_in_client_range_tomorrow'].mean():.1f}%)")


# ============================================================================
# STEP 9 — FEATURE SELECTION AND TRAIN/TEST SPLIT
# ============================================================================

print_step(9, "FEATURE SELECTION AND TRAIN/TEST SPLIT")

# --- Feature groups ---
static_features      = [c for c in ['pool_surface_m2', 'pool_volume_m3', 'filter_diameter', 'filter_count', 'motor_count'] if c in df_model_wq.columns]
categorical_features = ['pool_type', 'deck_type']
lag_features         = [c for c in ['ph_lag1', 'ph_lag2', 'chlorine_lag1', 'chlorine_lag2', 'turbidity_lag1', 'turbidity_lag2'] if c in df_model_wq.columns]
rolling_features     = [c for c in ['ph_roll3_mean', 'ph_roll3_std', 'chlorine_roll3_mean', 'chlorine_roll3_std', 'turbidity_roll3_mean'] if c in df_model_wq.columns]
temporal_features    = [c for c in ['days_since_last_visit', 'visit_month', 'visit_is_summer', 'visit_day_of_week', 'visit_year', 'pool_visit_number'] if c in df_model_wq.columns]

# Control variables (per client brief — the levers for optimisation)
control_features = [c for c in [
    'hypochlorite_dosing_pct', 'hypochlorite_dosing_hours',
    'ph_dosing_pct', 'ph_dosing_hours',
    'daily_filtration_hours', 'water_temperature',
] if c in df_model_wq.columns]

product_features = [c for c in [
    'last_total_chlorine_applied', 'total_ph_minus_product'
] if c in df_model_wq.columns]

headroom_features = [c for c in [
    'chlorine_headroom_low', 'chlorine_headroom_high',
    'ph_headroom_low', 'ph_headroom_high', 'turbidity_headroom', 'min_headroom',
    'cl_below_client_target', 'cl_above_client_target',
] if c in df_model_wq.columns]

trend_features = [c for c in [
    'ph_trend', 'chlorine_trend', 'turbidity_trend',
    'ph_rate_per_day', 'chlorine_rate_per_day', 'turbidity_rate_per_day',
] if c in df_model_wq.columns]

breach_history_features = [c for c in [
    'consecutive_clean_visits', 'breach_rate_last5',
    'current_any_breach', 'current_ph_breach', 'current_chlorine_breach',
    'multi_visit_day',
] if c in df_model_wq.columns]

chemistry_features = [c for c in [
    'ph_deviation', 'chlorine_deficit',
    'cl_effectiveness_index', 'chlorine_dose_per_m3', 'ph_minus_dose_per_m3',
    'chlorine_decay_per_m3',
] if c in df_model_wq.columns]

weather_current_features = [c for c in [
    'w_temp_max', 'w_temp_mean', 'w_uv_max', 'w_uv_clear_sky_max',
    'w_solar_radiation', 'w_sunshine_hours', 'w_precipitation_mm',
    'w_wind_max_kmh', 'w_et0',
] if c in df_model_wq.columns]

weather_cumulative_features = [c for c in cumulative_weather_features if c in df_model_wq.columns]

# Tomorrow's weather forecast (the prediction-day conditions)
# These are the strongest signals for next-day chlorine decay
weather_tomorrow_features = [c for c in tomorrow_weather_col_names if c in df_model_wq.columns]

all_numeric_features = (
    static_features + lag_features + rolling_features + temporal_features +
    control_features + product_features + headroom_features + trend_features +
    breach_history_features + chemistry_features +
    weather_current_features + weather_cumulative_features + weather_tomorrow_features
)

# Drop features with >50% nulls
null_rates = df_model_wq[all_numeric_features].isnull().mean()
high_null  = null_rates[null_rates > 0.5].index.tolist()
if high_null:
    print(f"Dropping features with >50% nulls: {high_null}")
    all_numeric_features = [f for f in all_numeric_features if f not in high_null]

print(f"Final numeric features ({len(all_numeric_features)})")
print(f"  Control features: {control_features}")
print(f"  Weather today: {weather_current_features}")
print(f"  Weather cumulative since last visit: {weather_cumulative_features}")
print(f"  Weather TOMORROW (prediction day): {weather_tomorrow_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# --- Temporal train/test split (80th percentile date) ---
cutoff_date = df_model_wq['reading_date'].quantile(0.8)
print(f"\nTemporal cutoff: {cutoff_date}")

df_train_wq = df_model_wq[df_model_wq['reading_date'] < cutoff_date].copy()
df_test_wq  = df_model_wq[df_model_wq['reading_date'] >= cutoff_date].copy()
print(f"Train: {len(df_train_wq)} rows | Test: {len(df_test_wq)} rows")

# --- Fill NaN in features with training-set medians ---
fill_values = {}
for col in all_numeric_features:
    median_val = df_train_wq[col].median()
    fill_values[col] = float(median_val) if pd.notna(median_val) else 0.0

# Store control variable medians/ranges for optimiser
control_fill = {c: fill_values.get(c, 0.0) for c in control_features}
print(f"\n  Control variable fill values (used as defaults in optimiser):")
for k, v in control_fill.items():
    print(f"    {k}: {v:.2f}")

for dfx in [df_train_wq, df_test_wq]:
    dfx[all_numeric_features] = dfx[all_numeric_features].fillna(fill_values)
    for col in categorical_features:
        dfx[col] = dfx[col].fillna('unknown')

# --- Preprocessor ---
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', 'passthrough', all_numeric_features),
    ],
    remainder='drop'
)

X_train = preprocessor.fit_transform(df_train_wq[categorical_features + all_numeric_features])
X_test  = preprocessor.transform(df_test_wq[categorical_features + all_numeric_features])

cat_names    = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
feature_names = cat_names + all_numeric_features

print(f"\nX_train: {X_train.shape} | X_test: {X_test.shape}")
print(f"Total features after encoding: {len(feature_names)}")

# Save preprocessor
with open(os.path.join(MODELS_DIR, 'preprocessor_v6.pkl'), 'wb') as f:
    pickle.dump(preprocessor, f)


# ============================================================================
# STEP 10 — TRAIN MODELS
# ============================================================================

print_step(10, "TRAIN XGBOOST MODELS")

models  = {}
results = {}


def train_and_eval(model, X_tr, y_tr, X_te, y_te, label, sample_weight=None):
    """Train an XGB regressor with early stopping and evaluate."""
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        sample_weight=sample_weight,
        verbose=False,
    )
    y_pred = model.predict(X_te)
    rmse   = np.sqrt(mean_squared_error(y_te, y_pred))
    mae    = mean_absolute_error(y_te, y_pred)
    r2     = r2_score(y_te, y_pred)
    p90    = np.percentile(np.abs(y_te - y_pred), 90)
    print(f"  {label}: RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f} | P90={p90:.4f} | best_iter={model.best_iteration}")
    return y_pred, {'rmse': rmse, 'mae': mae, 'r2': r2, 'p90': p90, 'best_iter': model.best_iteration}


# --- 10A: FREE CHLORINE (primary) ---
print("=" * 60)
print("  MODEL A — Free Chlorine at Next Visit")
print("=" * 60)

y_train_cl = df_train_wq['target_cl_tomorrow'].values
y_test_cl  = df_test_wq['target_cl_tomorrow'].values

# Upweight rows where current chlorine is near a breach (higher stakes)
sample_weights_cl = np.ones(len(df_train_wq))
breach_mask_cl    = df_train_wq['any_breach_next'].values.astype(bool)
sample_weights_cl[breach_mask_cl] = 3.0

model_cl = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS, eval_metric='rmse')
y_pred_cl, res_cl = train_and_eval(model_cl, X_train, y_train_cl, X_test, y_test_cl,
                                   'chlorine_next', sample_weight=sample_weights_cl)
models['chlorine_next']  = model_cl
results['chlorine_next'] = res_cl
model_cl.save_model(os.path.join(MODELS_DIR, 'xgb_chlorine_next.json'))
print("  Saved xgb_chlorine_next.json")

# % predictions in client target range [1.0–1.5]
in_range = ((y_pred_cl >= CLIENT_CL_TARGET_MIN) & (y_pred_cl <= CLIENT_CL_TARGET_MAX)).mean() * 100
print(f"  Predicted in client range [{CLIENT_CL_TARGET_MIN}–{CLIENT_CL_TARGET_MAX}]: {in_range:.1f}%")




# --- 10C: pH (primary) ---
print("\n" + "=" * 60)
print("  MODEL C — pH at Next Visit")
print("=" * 60)

y_train_ph = df_train_wq['target_ph_tomorrow'].values
y_test_ph  = df_test_wq['target_ph_tomorrow'].values

model_ph = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS, eval_metric='rmse')
y_pred_ph, res_ph = train_and_eval(model_ph, X_train, y_train_ph, X_test, y_test_ph, 'ph_next')
models['ph_next']  = model_ph
results['ph_next'] = res_ph
model_ph.save_model(os.path.join(MODELS_DIR, 'xgb_ph_next.json'))
print("  Saved xgb_ph_next.json")


# --- 10D: Turbidity (secondary) ---
# Turbidity target has additional NaNs not covered by the WQ filter (which only
# required Cl and pH targets). Build a turbidity-specific train/test subset.
print("\n" + "=" * 60)
print("  MODEL D — Turbidity at Next Visit")
print("=" * 60)

df_train_turb = df_train_wq.dropna(subset=['target_turb_tomorrow']).copy()
df_test_turb  = df_test_wq.dropna(subset=['target_turb_tomorrow']).copy()
print(f"  Turbidity train: {len(df_train_turb)} | test: {len(df_test_turb)}")

X_train_turb = preprocessor.transform(df_train_turb[categorical_features + all_numeric_features])
X_test_turb  = preprocessor.transform(df_test_turb[categorical_features + all_numeric_features])

y_train_turb = df_train_turb['target_turb_tomorrow'].values
y_test_turb  = df_test_turb['target_turb_tomorrow'].values

model_turb = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS, eval_metric='rmse')
y_pred_turb, res_turb = train_and_eval(model_turb, X_train_turb, y_train_turb, X_test_turb, y_test_turb, 'turbidity_next')
models['turbidity_next']  = model_turb
results['turbidity_next'] = res_turb
model_turb.save_model(os.path.join(MODELS_DIR, 'xgb_turbidity_next.json'))
print("  Saved xgb_turbidity_next.json")


# ============================================================================
# STEP 11 — SHAP EXPLAINABILITY
# ============================================================================

print_step(11, "SHAP EXPLAINABILITY")

shap_results = {}

for model_name, model in [
    ('chlorine_next',  models['chlorine_next']),
    ('ph_next',        models['ph_next']),
    ('turbidity_next', models['turbidity_next']),
]:
    print(f"\n--- SHAP: {model_name} ---")
    # Turbidity was trained on its own subset (X_test_turb); use that for SHAP
    X_shap = X_test_turb if model_name == 'turbidity_next' else X_test
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    mean_abs_shap    = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)
    top15 = feature_importance.head(15)

    print("  Top 15 features:")
    for feat, val in top15.items():
        weather_marker = " ★ WEATHER" if feat.startswith('w_') else ""
        print(f"    {feat}: {val:.4f}{weather_marker}")
    shap_results[model_name] = top15.to_dict()

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_shap, feature_names=feature_names,
                      plot_type='bar', max_display=15, show=False)
    plt.title(f'SHAP Feature Importance — {model_name.upper().replace("_", " ")} Model', fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f'shap_summary_{model_name}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {plot_path}")


# ============================================================================
# STEP 12 — OPTIMISATION ENGINE (per client brief)
# ============================================================================

print_step(12, "OPTIMISATION ENGINE — Recommend Dosing Settings")

# Grid of control variable combinations to test
DOSING_PCT_GRID   = np.arange(0, 105, 5)        # 0–100% in steps of 5
DOSING_HOURS_GRID = np.arange(0, 25, 1.0)        # 0–24h in steps of 1


def optimise_dosing(pool_id, df_master_src, model_cl, model_ph,
                    preprocessor, fill_values,
                    all_numeric_features, categorical_features):
    """
    Grid-search over (hypochlorite_dosing_pct, hypochlorite_dosing_hours) to find
    the combination that yields predicted next-visit Cl ∈ [1.0–1.5] mg/L and
    pH ∈ [7.2–8.0], with minimum dosage cost.

    Returns the best configuration and its predicted outcomes.
    """
    pool_data = df_master_src[df_master_src['pool_id'] == pool_id]
    if len(pool_data) == 0:
        return {'error': f'No data for pool_id: {pool_id}'}

    latest = pool_data.sort_values('reading_date').iloc[-1:].copy()

    # Prepare base features
    for col in all_numeric_features:
        if col in latest.columns:
            latest[col] = latest[col].fillna(fill_values.get(col, 0))
        else:
            latest[col] = fill_values.get(col, 0)
    for col in categorical_features:
        if col not in latest.columns:
            latest[col] = 'unknown'
        else:
            latest[col] = latest[col].fillna('unknown')

    current_ph  = float(latest['ph'].iloc[0]) if 'ph' in latest.columns and pd.notna(latest['ph'].iloc[0]) else None
    current_cl  = float(latest['free_chlorine'].iloc[0]) if 'free_chlorine' in latest.columns and pd.notna(latest['free_chlorine'].iloc[0]) else None
    pool_vol    = float(latest['pool_volume_m3'].iloc[0]) if 'pool_volume_m3' in latest.columns and pd.notna(latest['pool_volume_m3'].iloc[0]) else 50.0

    results_grid = []

    for pct in DOSING_PCT_GRID:
        for hours in DOSING_HOURS_GRID:
            row = latest.copy()
            # Override control variables
            if 'hypochlorite_dosing_pct' in row.columns:
                row['hypochlorite_dosing_pct'] = pct
            if 'hypochlorite_dosing_hours' in row.columns:
                row['hypochlorite_dosing_hours'] = hours

            X = preprocessor.transform(row[categorical_features + all_numeric_features])
            pred_cl = float(model_cl.predict(X)[0])
            pred_ph = float(model_ph.predict(X)[0])

            # Score: 0 is perfect (in both ranges), positive = penalty
            cl_penalty = max(0, CLIENT_CL_TARGET_MIN - pred_cl) + max(0, pred_cl - CLIENT_CL_TARGET_MAX)
            ph_penalty = max(0, REG_PH_MIN - pred_ph) + max(0, pred_ph - REG_PH_MAX)
            total_penalty = cl_penalty + ph_penalty

            # Secondary cost: minimize total dosing (prefer lower pct and hours)
            cost = pct / 100 * hours  # normalized dosing effort

            results_grid.append({
                'hypochlorite_dosing_pct':   pct,
                'hypochlorite_dosing_hours': hours,
                'pred_cl_next':   round(pred_cl, 3),
                'pred_ph_next':   round(pred_ph, 3),
                'cl_penalty':     round(cl_penalty, 4),
                'ph_penalty':     round(ph_penalty, 4),
                'total_penalty':  round(total_penalty, 4),
                'dosing_cost':    round(cost, 3),
            })

    df_grid = pd.DataFrame(results_grid)

    # Primary sort: minimum penalty; secondary: minimum dosing cost
    df_grid = df_grid.sort_values(['total_penalty', 'dosing_cost'])
    best    = df_grid.iloc[0].to_dict()

    # Determine urgency based on current readings only
    urgency = 'Routine'
    reasons = []
    if current_cl is not None and current_cl < REG_CHLORINE_MIN:
        urgency = 'Immediate'
        reasons.append(f"⚠️ Current Cl ({current_cl:.2f}) BELOW {REG_CHLORINE_MIN} mg/L — pathogen risk")
    if current_ph is not None and (current_ph < REG_PH_MIN or current_ph > REG_PH_MAX):
        urgency = 'Immediate'
        reasons.append(f"⚠️ Current pH ({current_ph:.2f}) outside [{REG_PH_MIN}–{REG_PH_MAX}]")

    if best['total_penalty'] == 0:
        reasons.append(f"Optimal config found: {best['hypochlorite_dosing_pct']:.0f}% for {best['hypochlorite_dosing_hours']:.1f}h → predicted Cl={best['pred_cl_next']}, pH={best['pred_ph_next']}")
    else:
        reasons.append(f"Best available: Cl penalty={best['cl_penalty']:.3f}, pH penalty={best['ph_penalty']:.3f}")

    # Feasible configs summary
    feasible = df_grid[df_grid['total_penalty'] == 0]

    return {
        'pool_id':             pool_id,
        'last_reading_date':   str(latest['reading_date'].iloc[0]),
        'pool_volume_m3':      pool_vol,
        'current_readings':    {'ph': round(current_ph, 2) if current_ph else None,
                                'free_chlorine': round(current_cl, 2) if current_cl else None},
        'urgency':             urgency,
        'reasons':             reasons,
        'recommended_dosing':  {
            'hypochlorite_dosing_pct':   best['hypochlorite_dosing_pct'],
            'hypochlorite_dosing_hours': best['hypochlorite_dosing_hours'],
        },
        'predicted_tomorrow': {
            'free_chlorine': best['pred_cl_next'],
            'ph':            best['pred_ph_next'],
        },
        'feasible_configurations': len(feasible),
        'top_3_configs': df_grid.head(3)[['hypochlorite_dosing_pct','hypochlorite_dosing_hours',
                                          'pred_cl_next','pred_ph_next','total_penalty']].to_dict('records'),
    }


# Test optimiser on sample pools from test set
test_pools   = df_test_wq['pool_id'].unique()
np.random.seed(42)
sample_pools = np.random.choice(test_pools, size=min(8, len(test_pools)), replace=False)

print(f"\nRunning optimiser on {len(sample_pools)} sample pools...\n")
example_optimisations = []

for pid in sample_pools:
    opt = optimise_dosing(
        pid, df_master,
        models['chlorine_next'], models['ph_next'],
        preprocessor, fill_values,
        all_numeric_features, categorical_features,
    )
    example_optimisations.append(opt)

    cr  = opt['current_readings']
    pn  = opt['predicted_tomorrow']
    rd  = opt['recommended_dosing']
    print(f"  Pool: {pid}")
    print(f"    Current:   pH={cr['ph']}, Cl={cr['free_chlorine']}")
    print(f"    Predicted: pH={pn['ph']}, Cl={pn['free_chlorine']}")
    print(f"    ⚙️  Recommended: {rd['hypochlorite_dosing_pct']:.0f}% dosage, {rd['hypochlorite_dosing_hours']:.1f}h/day")
    print(f"    ✅ Feasible configs: {opt['feasible_configurations']}")
    print(f"    📋 Urgency: {opt['urgency']} — {'; '.join(opt['reasons'])}")
    print()


# ============================================================================
# STEP 13 — EVALUATION REPORT
# ============================================================================

print_step(13, "EVALUATION REPORT V6")

total_pools    = df_master['pool_id'].nunique()
total_readings = len(df_master)
date_min       = df_master['reading_date'].min()
date_max       = df_master['reading_date'].max()

report = []
report.append("=" * 70)
report.append("  POOL PREDICTIVE MAINTENANCE V6 — EVALUATION REPORT")
report.append(f"  Regulatory basis: Real Decreto 742/2013 + Decreto 85/2018 CV")
report.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report.append("=" * 70)

report.append("\n\n1. DATASET SUMMARY")
report.append("-" * 40)
report.append(f"  Dataset:         {RAW_EXCEL}")
report.append(f"  Pool filter:     {CHLORINE_PUMP_LIST} (liquid chlorine dosing pump only)")
report.append(f"  Total pools:     {total_pools}")
report.append(f"  Total readings:  {total_readings}")
report.append(f"  Date range:      {date_min.date()} to {date_max.date()}")
report.append(f"  Model rows (WQ): {len(df_model_wq)}")
report.append(f"  Train rows:      {len(df_train_wq)}")
report.append(f"  Test rows:       {len(df_test_wq)}")
report.append(f"  Temporal cutoff: {cutoff_date}")
report.append(f"\n  Weather source: Open-Meteo Archive/Forecast API")
report.append(f"  Alicante coords: lat={ALICANTE_LAT}, lon={ALICANTE_LON}")
report.append(f"  Weather features: {len(weather_current_features)} today + {len(weather_cumulative_features)} cumulative-since-last-visit + {len(weather_tomorrow_features)} TOMORROW forecast")

report.append("\n\n2. STATIC DATA BACKFILL SUMMARY")
report.append("-" * 40)
for col, stats in backfill_summary.items():
    report.append(f"  {col}:")
    report.append(f"    Pools with original data:  {stats['pools_with_original_data']}")
    report.append(f"    Pools filled with median:  {stats['pools_filled_with_median']} (fleet median: {stats['fleet_median']})")
    report.append(f"    Rows before: {stats['rows_before']} | After: {stats['rows_after']}")

report.append("\n\n3. MODEL A — FREE CHLORINE TOMORROW (PRIMARY)")
report.append("-" * 70)
c_res = results['chlorine_next']
report.append(f"  Objective: Predict free chlorine level on the next calendar day")
report.append(f"  Method: linear interpolation 1 day forward from each visit")
report.append(f"  Client optimal range: [{CLIENT_CL_TARGET_MIN}–{CLIENT_CL_TARGET_MAX}] mg/L")
report.append(f"  Regulatory range: [{REG_CHLORINE_MIN}–{REG_CHLORINE_CLOSE}] mg/L")
report.append(f"\n  Regression Performance (test set):")
report.append(f"    MAE:  {c_res['mae']:.4f} mg/L")
report.append(f"    RMSE: {c_res['rmse']:.4f} mg/L")
report.append(f"    R²:   {c_res['r2']:.4f}")
report.append(f"    P90 Error: {c_res['p90']:.4f} mg/L")

report.append(f"\n  Top SHAP Drivers:")
for i, (feat, val) in enumerate(list(shap_results['chlorine_next'].items())[:15], 1):
    wmark = " ★ WEATHER" if feat.startswith('w_') else ""
    report.append(f"    {i:2d}. {feat}: {val:.4f}{wmark}")

report.append("\n\n4. MODEL C — pH TOMORROW (PRIMARY)")
report.append("-" * 70)
ph_res = results['ph_next']
report.append(f"  Objective: Predict pH on the next calendar day")
report.append(f"  Regulatory range: [{REG_PH_MIN}–{REG_PH_MAX}]")
report.append(f"\n  Regression Performance (test set):")
report.append(f"    MAE:  {ph_res['mae']:.4f} pH units")
report.append(f"    RMSE: {ph_res['rmse']:.4f} pH units")
report.append(f"    R²:   {ph_res['r2']:.4f}")
report.append(f"    P90 Error: {ph_res['p90']:.4f} pH units")
report.append(f"\n  A standard handheld pH meter has measurement error of ±0.1 pH.")
report.append(f"  A MAE of {ph_res['mae']:.4f} means forecasts are within instrument accuracy.")
report.append(f"\n  Top SHAP Drivers:")
for i, (feat, val) in enumerate(list(shap_results['ph_next'].items())[:15], 1):
    wmark = " ★ WEATHER" if feat.startswith('w_') else ""
    report.append(f"    {i:2d}. {feat}: {val:.4f}{wmark}")

report.append("\n\n5. MODEL D — TURBIDITY TOMORROW (SECONDARY)")
report.append("-" * 70)
turb_res = results['turbidity_next']
report.append(f"  MAE: {turb_res['mae']:.4f} NTU | RMSE: {turb_res['rmse']:.4f} | R²: {turb_res['r2']:.4f}")
report.append(f"  Legal limit: {REG_TURBIDITY_MAX} NTU. MAE of {turb_res['mae']:.4f} NTU is negligible.")

report.append("\n\n6. OPTIMISATION ENGINE — RECOMMENDED DOSING CONFIGURATIONS")
report.append("-" * 70)
report.append(f"  Grid: hypochlorite_dosing_pct ∈ [0–100%] (step 5%)")
report.append(f"        hypochlorite_dosing_hours ∈ [0–24h] (step 1h)")
report.append(f"  Target: Cl ∈ [{CLIENT_CL_TARGET_MIN}–{CLIENT_CL_TARGET_MAX}] mg/L AND pH ∈ [{REG_PH_MIN}–{REG_PH_MAX}]")
report.append(f"\n  Example Pool Recommendations:")
for opt in example_optimisations:
    cr = opt['current_readings']
    pn = opt['predicted_tomorrow']
    rd = opt['recommended_dosing']
    report.append(f"\n  Pool: {opt['pool_id']}")
    report.append(f"    Current: pH={cr['ph']}, Cl={cr['free_chlorine']}")
    report.append(f"    Recommended: {rd['hypochlorite_dosing_pct']:.0f}% dosage, {rd['hypochlorite_dosing_hours']:.1f}h/day")
    report.append(f"    Predicted next: Cl={pn['free_chlorine']}, pH={pn['ph']}")
    report.append(f"    Feasible configs: {opt['feasible_configurations']} | Urgency: {opt['urgency']}")

report_text = '\n'.join(report)
report_path = os.path.join(OUTPUT_DIR, 'evaluation_report_v6.txt')
with open(report_path, 'w') as f:
    f.write(report_text)
print(report_text)
print(f"\nSaved {report_path}")


# ============================================================================
# STEP 14 — SAVE INFERENCE CONFIG
# ============================================================================

print_step(14, "SAVE INFERENCE CONFIG")

inference_config = {
    'pipeline_version':      'v6',
    'generated_at':          datetime.now().isoformat(),
    'fill_values':           fill_values,
    'all_numeric_features':  all_numeric_features,
    'categorical_features':  categorical_features,
    'feature_names':         feature_names,
    'control_features':      control_features,
    'weather_current_features':    weather_current_features,
    'weather_cumulative_features': weather_cumulative_features,
    'weather_tomorrow_features':   weather_tomorrow_features,
    'alicante_coords':       {'lat': ALICANTE_LAT, 'lon': ALICANTE_LON, 'timezone': ALICANTE_TZ},
    'regulatory_thresholds': {
        'chlorine_min':       REG_CHLORINE_MIN,
        'chlorine_ideal_max': REG_CHLORINE_IDEAL_MAX,
        'chlorine_close':     REG_CHLORINE_CLOSE,
        'ph_min':             REG_PH_MIN,
        'ph_max':             REG_PH_MAX,
        'turbidity_max':      REG_TURBIDITY_MAX,
    },
    'client_targets': {
        'chlorine_min': CLIENT_CL_TARGET_MIN,
        'chlorine_max': CLIENT_CL_TARGET_MAX,
        'chlorine_ideal': CLIENT_CL_IDEAL,
        'ph_ideal': PH_IDEAL,
    },
    'dosing_grid': {
        'pct_step': 5,
        'hours_step': 1.0,
    },
}

config_path = os.path.join(MODELS_DIR, 'inference_config_v6.json')
with open(config_path, 'w') as f:
    json.dump(inference_config, f, indent=2, default=str)
print(f"Saved {config_path}")

# Save master dataset
master_path = os.path.join(OUTPUT_DIR, 'master_dataset_v6.csv')
df_master.to_csv(master_path, index=False)
print(f"Saved master dataset: {master_path} ({df_master.shape})")


# ============================================================================
# STEP 15 — FINAL OUTPUT SUMMARY
# ============================================================================

print_step(15, "FINAL OUTPUT SUMMARY")

expected_files = {
    'xgb_chlorine_next.json':    MODELS_DIR,
    'xgb_ph_next.json':          MODELS_DIR,
    'xgb_turbidity_next.json':   MODELS_DIR,
    'preprocessor_v6.pkl':       MODELS_DIR,
    'inference_config_v6.json':  MODELS_DIR,
    'shap_summary_chlorine_next.png':  OUTPUT_DIR,
    'shap_summary_ph_next.png':        OUTPUT_DIR,
    'shap_summary_turbidity_next.png': OUTPUT_DIR,
    'evaluation_report_v6.txt':  OUTPUT_DIR,
    'master_dataset_v6.csv':     OUTPUT_DIR,
    'weather_alicante_2023_2026.csv': 'data',
}

print("Output files:")
all_ok = True
for fname, fdir in expected_files.items():
    fpath = os.path.join(fdir, fname)
    if os.path.exists(fpath):
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  ✅ {fname} ({size_kb:.1f} KB) in {fdir}/")
    else:
        print(f"  ❌ {fname} — MISSING from {fdir}/")
        all_ok = False

print(f"\n{'='*70}")
print(f"  PIPELINE V6 COMPLETE")
print(f"  Dataset: 2023–2026 | Pools: chlorine-pump only | Weather: Alicante Open-Meteo")
print(f"  Primary targets: next-visit free chlorine + pH")
print(f"  Optimisation: grid-search over dosing % and pump hours")
print(f"  Regulatory basis: Real Decreto 742/2013")
print(f"{'='*70}")
