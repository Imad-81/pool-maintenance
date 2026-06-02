#!/usr/bin/env python3
"""
Pool Predictive Maintenance Pipeline
=====================================
XGBoost-based system for forecasting water quality parameters (pH, free chlorine,
turbidity) and prescribing chemical dosages for the next technician visit.

Data source: Pepe Gutiérrez / SPP System (Spanish pool services company)
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os
import sys
import pickle
import json
from datetime import datetime

# ML / Stats
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Explainability
import shap

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================
RAW_CSV = 'raw_data.csv'
OUTPUT_DIR = '.'  # All outputs in project root

# pH / chlorine / turbidity acceptable ranges
PH_RANGE = (6.5, 8.5)
CHLORINE_RANGE = (0.5, 5.0)
TURBIDITY_MAX = 5.0
PH_IDEAL = 7.2
CHLORINE_MIN_ACCEPTABLE = 1.0

# Merge tolerance
MERGE_TOLERANCE_DAYS = 14

# XGBoost hyperparameters
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

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def print_step(step_num, title):
    """Pretty print step header."""
    print(f"\n{'='*70}")
    print(f"  STEP {step_num} — {title}")
    print(f"{'='*70}\n")


def safe_float(series):
    """Convert series to float, coercing errors to NaN."""
    return pd.to_numeric(series, errors='coerce').astype(float)


def clean_pool_id(s):
    """Strip whitespace and lowercase pool IDs for consistency."""
    if pd.isna(s):
        return np.nan
    return ' '.join(str(s).strip().lower().split())


# ============================================================================
# STEP 1 — LOAD AND INSPECT RAW DATA
# ============================================================================
print_step(1, "LOAD AND INSPECT RAW DATA")

try:
    df = pd.read_csv(RAW_CSV)
    print(f"Loaded {RAW_CSV}: {df.shape[0]} rows × {df.shape[1]} columns")
except FileNotFoundError:
    print(f"ERROR: {RAW_CSV} not found. Please run the Numbers-to-CSV conversion first.")
    sys.exit(1)

# Column rename mapping — positional since some column names repeat
# The actual columns from the file (61 cols):
# [0]  PISCINA
# [1]  COMUNIDAD
# [2]  FECHA
# [3]  EMPLEADO
# [4]  PH
# [5]  TURBIDEZ
# [6]  CLORO LIBRE
# [7]  (separator)
# [8]  ABUSO CREMAS PROTECCION
# [9]  Caudal bomba de PH
# [10] Caudal bomba hipoclorito
# [11] Caudal del motor
# [12] Diametro filtro
# [13] Numero de filtros
# [14] Número de motores
# [15] PISCINA CLIMATIZADA
# [16] PISCINA COMUNITARIA
# [17] Piscina con skimmers
# [18] Piscina desbordante
# [19] PISCINA EXTERIOR
# [20] Piscina ovalada
# [21] PISCINA PARTICULAR
# [22] PISCINA PUBLICA
# [23] (0714) Piscina rectangular
# [24] (07) Piscina rectangular
# [25] Piscina redonda
# [26] Superficie piscina
# [27] VEGETACION CONTAMINANTE
# [28] Volumen piscina
# [29] Zona playa césped
# [30] Zona PLAYA mixta
# [31] Zona PLAYA pavimentada
# [32] (separator)
# [33] EMPLEADO (ops)
# [34] FECHA (ops)
# [35] Horas dosificacion PH
# [36] Horas filtracion diarias
# [37] Porcentaje dosificación PH
# [38] Tiempo lavado /enjuague filtro
# [39] Horas dosificación hipo
# [40] Porcentaje dosificación hipoclorito
# [41] Temperatura agua
# [42] (separator)
# [43] EMPLEADO (products)
# [44] FECHA (products)
# [45] T-500 (GRUPO QP)
# [46] ALBORAL TABLETAS 250 GRS RF. 201710
# [47] FLOVIL PASTILLAS
# [48] HIPO GARRAFAS 20KG.
# [49] HIPO GR CHLORYTE
# [50] HIPO GRANULADO XAKA
# [51] HIPO STICKS BAYROL
# [52] HIPO TAB. RITOCAL
# [53] HIPO TABLETAS 200Gr. QP
# [54] HIPO TABLETAS XAKA
# [55] PH MINUS GRANULADO 6kg
# [56] PH MINUS LIQUIDO 13.5 KG
# [57] PH MINUS LIQUIDO 27 KG.
# [58] PROTECT & SHINE
# [59] SG XAKA (AGONET GR90)
# [60] SUPERKLAR

POSITIONAL_RENAME = {
    0: 'pool_id',
    1: 'community_name',
    2: 'reading_date',
    3: 'technician',
    4: 'ph',
    5: 'turbidity',
    6: 'free_chlorine',
    7: '_sep1',
    8: 'sunscreen_abuse',
    9: 'ph_pump_flow_rate',
    10: 'hypochlorite_pump_flow_rate',
    11: 'motor_flow_rate',
    12: 'filter_diameter',
    13: 'filter_count',
    14: 'motor_count',
    15: 'pool_heated',
    16: 'pool_community',
    17: 'pool_skimmer',
    18: 'pool_overflow',
    19: 'pool_outdoor',
    20: 'pool_oval',
    21: 'pool_private',
    22: 'pool_public',
    23: 'pool_rectangular_0714',
    24: 'pool_rectangular_07',
    25: 'pool_round',
    26: 'pool_surface_m2',
    27: 'vegetation_contamination',
    28: 'pool_volume_m3',
    29: 'deck_grass',
    30: 'deck_mixed',
    31: 'deck_paved',
    32: '_sep2',
    33: 'ops_technician',
    34: 'ops_date',
    35: 'ph_dosing_hours',
    36: 'daily_filtration_hours',
    37: 'ph_dosing_pct',
    38: 'filter_wash_rinse_time',
    39: 'hypochlorite_dosing_hours',
    40: 'hypochlorite_dosing_pct',
    41: 'water_temperature',
    42: '_sep3',
    43: 'prod_technician',
    44: 'prod_date',
    45: 'prod_t500_qp',
    46: 'prod_alboral_tablets_250g',
    47: 'prod_flovil_tablets',
    48: 'prod_hypo_jugs_20kg',
    49: 'prod_hypo_gr_chloryte',
    50: 'prod_hypo_granular_xaka',
    51: 'prod_hypo_sticks_bayrol',
    52: 'prod_hypo_tab_ritocal',
    53: 'prod_hypo_tablets_200g_qp',
    54: 'prod_hypo_tablets_xaka',
    55: 'prod_ph_minus_granular_6kg',
    56: 'prod_ph_minus_liquid_13_5kg',
    57: 'prod_ph_minus_liquid_27kg',
    58: 'prod_protect_shine',
    59: 'prod_sg_xaka_agonet',
    60: 'prod_superklar',
}

# Rename by position
df.columns = [POSITIONAL_RENAME.get(i, f'unknown_{i}') for i in range(len(df.columns))]

# Drop separator columns
df = df.drop(columns=[c for c in df.columns if c.startswith('_sep')])

# Clean pool_id
df['pool_id'] = df['pool_id'].apply(clean_pool_id)

print("\n--- df.head() ---")
print(df.head())
print(f"\n--- df.shape: {df.shape} ---")
print(f"\n--- df.dtypes ---")
print(df.dtypes)
print(f"\n--- Null counts ---")
print(df.isnull().sum())

# ============================================================================
# STEP 2 — UNDERSTAND THE DATA STRUCTURE / SEPARATE SUB-TABLES
# ============================================================================
print_step(2, "SEPARATE THREE SUB-TABLES")

# ---- READINGS sub-table ----
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

# Forward-fill pool_id and community_name (they appear once per pool block)
df['pool_id'] = df['pool_id'].ffill()
df['community_name'] = df['community_name'].ffill()

# --- Extract readings ---
df_readings = df[reading_cols].copy()
df_readings = df_readings.dropna(subset=['reading_date'])
print(f"df_readings shape (after dropping null reading_date): {df_readings.shape}")

# --- Extract operations ---
# Operations sub-table needs pool_id from context
df_operations = df[['pool_id'] + ops_cols].copy()
df_operations = df_operations.dropna(subset=['ops_date'])
# Drop rows where all key ops columns are null
key_ops_cols = ['ph_dosing_hours', 'daily_filtration_hours', 'water_temperature']
df_operations = df_operations.dropna(subset=key_ops_cols, how='all')
print(f"df_operations shape (after dropping null ops_date and all-null key ops): {df_operations.shape}")

# --- Extract products ---
df_products = df[['pool_id'] + prod_cols].copy()
df_products = df_products.dropna(subset=['prod_date'])
print(f"df_products shape (after dropping null prod_date): {df_products.shape}")

# Report date ranges
def parse_date_series(s):
    """Parse date strings in DD-MM-YYYY or DD-MM-YYYY HH:MM format."""
    return pd.to_datetime(s, format='mixed', dayfirst=True, errors='coerce')

for name, dfx, date_col in [
    ('df_readings', df_readings, 'reading_date'),
    ('df_operations', df_operations, 'ops_date'),
    ('df_products', df_products, 'prod_date')
]:
    dates = parse_date_series(dfx[date_col])
    valid = dates.dropna()
    if len(valid) > 0:
        print(f"  {name}: {len(dfx)} rows, date range: {valid.min().date()} to {valid.max().date()}")
    else:
        print(f"  {name}: {len(dfx)} rows, no valid dates parsed!")

# ============================================================================
# STEP 3 — DATA CLEANING
# ============================================================================
print_step(3, "DATA CLEANING")

# ----------- Clean df_readings -----------
print("Cleaning df_readings...")

# Parse dates
df_readings['reading_date'] = parse_date_series(df_readings['reading_date'])

# Cast numeric columns
for col in ['ph', 'turbidity', 'free_chlorine']:
    df_readings[col] = safe_float(df_readings[col])

# Outlier flags
df_readings['ph_outlier'] = ~df_readings['ph'].between(*PH_RANGE) & df_readings['ph'].notna()
df_readings['chlorine_outlier'] = ~df_readings['free_chlorine'].between(*CHLORINE_RANGE) & df_readings['free_chlorine'].notna()
df_readings['turbidity_outlier'] = (df_readings['turbidity'] > TURBIDITY_MAX) & df_readings['turbidity'].notna()

print(f"  pH outliers: {df_readings['ph_outlier'].sum()}")
print(f"  Chlorine outliers: {df_readings['chlorine_outlier'].sum()}")
print(f"  Turbidity outliers: {df_readings['turbidity_outlier'].sum()}")

# Cast pool type flags to boolean (1/0)
pool_type_flags = [
    'pool_heated', 'pool_community', 'pool_skimmer', 'pool_overflow',
    'pool_outdoor', 'pool_oval', 'pool_private', 'pool_public',
    'pool_rectangular_0714', 'pool_rectangular_07', 'pool_round',
]
for col in pool_type_flags:
    df_readings[col] = safe_float(df_readings[col]).fillna(0).astype(int)

# Cast dimensions
df_readings['pool_surface_m2'] = safe_float(df_readings['pool_surface_m2'])
df_readings['pool_volume_m3'] = safe_float(df_readings['pool_volume_m3'])
df_readings['filter_diameter'] = safe_float(df_readings['filter_diameter'])
df_readings['filter_count'] = safe_float(df_readings['filter_count'])
df_readings['motor_count'] = safe_float(df_readings['motor_count'])

# Derived: pool_type categorical
def make_pool_type(row):
    parts = []
    if row.get('pool_heated', 0): parts.append('heated')
    if row.get('pool_outdoor', 0): parts.append('outdoor')
    if row.get('pool_community', 0): parts.append('community')
    if row.get('pool_private', 0): parts.append('private')
    if row.get('pool_public', 0): parts.append('public')
    return '_'.join(parts) if parts else 'unknown'

df_readings['pool_type'] = df_readings.apply(make_pool_type, axis=1)

# Derived: deck_type categorical
def make_deck_type(row):
    g = safe_float(pd.Series([row.get('deck_grass', 0)])).iloc[0] or 0
    p = safe_float(pd.Series([row.get('deck_paved', 0)])).iloc[0] or 0
    m = safe_float(pd.Series([row.get('deck_mixed', 0)])).iloc[0] or 0
    if m > 0: return 'mixed'
    if g > 0 and p > 0: return 'mixed'
    if g > 0: return 'grass'
    if p > 0: return 'paved'
    return 'unknown'

df_readings['deck_type'] = df_readings.apply(make_deck_type, axis=1)

# Deduplicate
before_dedup = len(df_readings)
df_readings = df_readings.drop_duplicates(subset=['pool_id', 'reading_date'])
print(f"  Deduplication: {before_dedup} → {len(df_readings)} rows")

# Sort
df_readings = df_readings.sort_values(['pool_id', 'reading_date']).reset_index(drop=True)

# ----------- Clean df_operations -----------
print("\nCleaning df_operations...")

df_operations['ops_date'] = parse_date_series(df_operations['ops_date'])

ops_numeric = [
    'ph_dosing_hours', 'daily_filtration_hours', 'ph_dosing_pct',
    'filter_wash_rinse_time', 'hypochlorite_dosing_hours',
    'hypochlorite_dosing_pct', 'water_temperature',
]
for col in ops_numeric:
    df_operations[col] = safe_float(df_operations[col])

# Deduplicate: mean for operational columns per pool_id + ops_date
df_operations = df_operations.groupby(['pool_id', 'ops_date'], as_index=False)[ops_numeric].mean()
print(f"  After dedup (mean aggregation): {df_operations.shape}")

# Sort
df_operations = df_operations.sort_values(['pool_id', 'ops_date']).reset_index(drop=True)

# ----------- Clean df_products -----------
print("\nCleaning df_products...")

df_products['prod_date'] = parse_date_series(df_products['prod_date'])

prod_value_cols = [c for c in prod_cols if c not in ['prod_technician', 'prod_date']]
for col in prod_value_cols:
    df_products[col] = safe_float(df_products[col]).fillna(0)

# Derived aggregates
hypo_cols = [c for c in prod_value_cols if 'hypo' in c.lower()]
ph_minus_cols = [c for c in prod_value_cols if 'ph_minus' in c.lower()]
flocculant_cols = ['prod_flovil_tablets', 'prod_superklar', 'prod_sg_xaka_agonet', 'prod_alboral_tablets_250g']
# Only keep flocculant cols that exist
flocculant_cols = [c for c in flocculant_cols if c in df_products.columns]

df_products['total_chlorine_product'] = df_products[hypo_cols].sum(axis=1)
df_products['total_ph_minus_product'] = df_products[ph_minus_cols].sum(axis=1)
df_products['total_flocculant_product'] = df_products[flocculant_cols].sum(axis=1)

# Deduplicate: max for product columns per pool_id + prod_date
agg_cols = prod_value_cols + ['total_chlorine_product', 'total_ph_minus_product', 'total_flocculant_product']
df_products = df_products.groupby(['pool_id', 'prod_date'], as_index=False)[agg_cols].max()
print(f"  After dedup (max aggregation): {df_products.shape}")

# Sort
df_products = df_products.sort_values(['pool_id', 'prod_date']).reset_index(drop=True)

# --- Identify orphan product records ---
reading_pools = set(df_readings['pool_id'].dropna().unique())
product_pools = set(df_products['pool_id'].dropna().unique())
orphan_pools = product_pools - reading_pools
if orphan_pools:
    print(f"\n  ⚠ Orphan product pools (in products but not in readings): {orphan_pools}")
    df_products = df_products[df_products['pool_id'].isin(reading_pools)]
    print(f"  After removing orphans: {df_products.shape}")
else:
    print("  No orphan product pools found.")

# Print summary statistics
print("\n--- Value counts after cleaning ---")
print(f"  df_readings pool_type distribution:")
print(df_readings['pool_type'].value_counts().to_string())
print(f"\n  df_readings deck_type distribution:")
print(df_readings['deck_type'].value_counts().to_string())

print(f"\n--- Null percentages after cleaning ---")
for name, dfx in [('df_readings', df_readings), ('df_operations', df_operations), ('df_products', df_products)]:
    print(f"\n  {name}:")
    null_pct = (dfx.isnull().sum() / len(dfx) * 100).round(1)
    for col, pct in null_pct.items():
        if pct > 0:
            print(f"    {col}: {pct}%")

# ============================================================================
# STEP 4 — MERGE INTO A SINGLE ANALYTICAL DATASET
# ============================================================================
print_step(4, "MERGE INTO MASTER DATASET")

# Ensure all are sorted by pool_id and date for merge_asof
df_readings = df_readings.sort_values(['pool_id', 'reading_date']).reset_index(drop=True)
df_operations = df_operations.sort_values(['pool_id', 'ops_date']).reset_index(drop=True)
df_products = df_products.sort_values(['pool_id', 'prod_date']).reset_index(drop=True)

# merge_asof requires the "on" column to be sorted, so we merge per-pool
tolerance = pd.Timedelta(f'{MERGE_TOLERANCE_DAYS}D')

# --- Merge operations onto readings ---
def merge_asof_by_pool(df_left, df_right, left_date, right_date, right_cols):
    """Merge_asof grouped by pool_id."""
    merged_parts = []
    for pool_id in df_left['pool_id'].unique():
        left_pool = df_left[df_left['pool_id'] == pool_id].copy()
        right_pool = df_right[df_right['pool_id'] == pool_id].copy()
        
        if len(right_pool) == 0:
            # No matching records — add NaN columns
            for col in right_cols:
                left_pool[col] = np.nan
            merged_parts.append(left_pool)
            continue
        
        left_pool = left_pool.sort_values(left_date)
        right_pool = right_pool.sort_values(right_date)
        
        merged = pd.merge_asof(
            left_pool,
            right_pool[right_cols + [right_date]],
            left_on=left_date,
            right_on=right_date,
            direction='backward',
            tolerance=tolerance,
        )
        merged_parts.append(merged)
    
    return pd.concat(merged_parts, ignore_index=True)

# Ops columns to bring in (excluding pool_id and ops_date which is the join key)
ops_merge_cols = [c for c in df_operations.columns if c not in ['pool_id']]
ops_value_cols = [c for c in ops_merge_cols if c != 'ops_date']

df_master = merge_asof_by_pool(
    df_readings, df_operations,
    'reading_date', 'ops_date',
    ops_value_cols
)

ops_matched = df_master[ops_value_cols[0]].notna().sum()
print(f"Readings with ops match (within {MERGE_TOLERANCE_DAYS}d): {ops_matched}/{len(df_master)} ({100*ops_matched/len(df_master):.1f}%)")

# --- Merge products onto master ---
prod_merge_cols = [c for c in df_products.columns if c not in ['pool_id']]
prod_value_cols = [c for c in prod_merge_cols if c != 'prod_date']

df_master = merge_asof_by_pool(
    df_master, df_products,
    'reading_date', 'prod_date',
    prod_value_cols
)

prod_matched = df_master['total_chlorine_product'].notna().sum()
print(f"Readings with products match (within {MERGE_TOLERANCE_DAYS}d): {prod_matched}/{len(df_master)} ({100*prod_matched/len(df_master):.1f}%)")

print(f"\ndf_master shape: {df_master.shape}")

# ============================================================================
# STEP 5 — TIME-SERIES FEATURE ENGINEERING
# ============================================================================
print_step(5, "TIME-SERIES FEATURE ENGINEERING")

df_master = df_master.sort_values(['pool_id', 'reading_date']).reset_index(drop=True)

# --- Lag features ---
for col, prefix in [('ph', 'ph'), ('free_chlorine', 'chlorine'), ('turbidity', 'turbidity')]:
    df_master[f'{prefix}_lag1'] = df_master.groupby('pool_id')[col].shift(1)
    df_master[f'{prefix}_lag2'] = df_master.groupby('pool_id')[col].shift(2)

# --- Rolling statistics (window = 3 readings per pool) ---
for col, prefix in [('ph', 'ph'), ('free_chlorine', 'chlorine'), ('turbidity', 'turbidity')]:
    rolling = df_master.groupby('pool_id')[col].transform(
        lambda x: x.rolling(window=3, min_periods=2).mean()
    )
    df_master[f'{prefix}_roll3_mean'] = rolling
    
    if prefix != 'turbidity':  # std for ph and chlorine
        rolling_std = df_master.groupby('pool_id')[col].transform(
            lambda x: x.rolling(window=3, min_periods=2).std()
        )
        df_master[f'{prefix}_roll3_std'] = rolling_std

# --- Visit interval features ---
df_master['days_since_last_visit'] = df_master.groupby('pool_id')['reading_date'].diff().dt.days
df_master['visit_day_of_week'] = df_master['reading_date'].dt.dayofweek
df_master['visit_month'] = df_master['reading_date'].dt.month
df_master['visit_is_summer'] = df_master['visit_month'].isin([6, 7, 8, 9]).astype(int)

# --- Derived chemistry features ---
df_master['ph_deviation'] = (df_master['ph'] - PH_IDEAL).abs()
df_master['chlorine_deficit'] = (CHLORINE_MIN_ACCEPTABLE - df_master['free_chlorine']).clip(lower=0)

# last_total_chlorine_applied is already in via the merge (total_chlorine_product)
df_master['last_total_chlorine_applied'] = df_master['total_chlorine_product'].fillna(0)

# Drop rows with NaN in lag columns (first 1-2 readings per pool)
lag_cols = ['ph_lag1', 'ph_lag2', 'chlorine_lag1', 'chlorine_lag2', 'turbidity_lag1', 'turbidity_lag2']
before_lag_drop = len(df_master)
df_master = df_master.dropna(subset=lag_cols)
print(f"Dropped {before_lag_drop - len(df_master)} rows with null lag features")
print(f"df_master after feature engineering: {df_master.shape}")

# Print feature summary
feature_cols = lag_cols + [
    'ph_roll3_mean', 'ph_roll3_std', 'chlorine_roll3_mean', 'chlorine_roll3_std',
    'turbidity_roll3_mean', 'days_since_last_visit', 'visit_day_of_week',
    'visit_month', 'visit_is_summer', 'ph_deviation', 'chlorine_deficit',
    'last_total_chlorine_applied',
]
print(f"\nEngineered features ({len(feature_cols)}):")
for fc in feature_cols:
    non_null = df_master[fc].notna().sum()
    print(f"  {fc}: {non_null}/{len(df_master)} non-null ({100*non_null/len(df_master):.1f}%)")

# ============================================================================
# STEP 6 — DEFINE PREDICTION TARGETS
# ============================================================================
print_step(6, "DEFINE PREDICTION TARGETS")

df_master['target_ph_next'] = df_master.groupby('pool_id')['ph'].shift(-1)
df_master['target_chlorine_next'] = df_master.groupby('pool_id')['free_chlorine'].shift(-1)
df_master['target_turbidity_next'] = df_master.groupby('pool_id')['turbidity'].shift(-1)

target_cols = ['target_ph_next', 'target_chlorine_next', 'target_turbidity_next']

# Also exclude rows where current ph/chlorine readings are null (tech logged visit but didn't record)
df_model = df_master.dropna(subset=target_cols + ['ph', 'free_chlorine', 'turbidity']).copy()

print(f"Rows with all targets available: {len(df_model)}")
print(f"\n--- Target distributions ---")
for tc in target_cols:
    desc = df_model[tc].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    print(f"\n{tc}:")
    print(desc.to_string())

# ============================================================================
# STEP 7 — FEATURE SELECTION AND TRAIN/TEST SPLIT
# ============================================================================
print_step(7, "FEATURE SELECTION AND TRAIN/TEST SPLIT")

# Define feature columns
static_features = ['pool_surface_m2', 'pool_volume_m3', 'filter_diameter', 'filter_count', 'motor_count']
categorical_features = ['pool_type', 'deck_type']
lag_features = ['ph_lag1', 'ph_lag2', 'chlorine_lag1', 'chlorine_lag2', 'turbidity_lag1', 'turbidity_lag2']
rolling_features = ['ph_roll3_mean', 'ph_roll3_std', 'chlorine_roll3_mean', 'chlorine_roll3_std', 'turbidity_roll3_mean']
ops_features = ['daily_filtration_hours', 'water_temperature', 'ph_dosing_pct', 'hypochlorite_dosing_pct']
product_features = ['last_total_chlorine_applied', 'total_ph_minus_product']
temporal_features = ['days_since_last_visit', 'visit_month', 'visit_is_summer', 'visit_day_of_week']

all_numeric_features = static_features + lag_features + rolling_features + ops_features + product_features + temporal_features

# Drop features with >50% nulls
null_rates = df_model[all_numeric_features].isnull().mean()
high_null_features = null_rates[null_rates > 0.5].index.tolist()
if high_null_features:
    print(f"Dropping features with >50% nulls: {high_null_features}")
    all_numeric_features = [f for f in all_numeric_features if f not in high_null_features]

print(f"\nFinal numeric features ({len(all_numeric_features)}): {all_numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# --- Temporal train/test split ---
cutoff_date = df_model['reading_date'].quantile(0.8)
print(f"\nTemporal cutoff date (80th percentile): {cutoff_date}")

train_mask = df_model['reading_date'] < cutoff_date
test_mask = df_model['reading_date'] >= cutoff_date

df_train = df_model[train_mask].copy()
df_test = df_model[test_mask].copy()

print(f"Train size: {len(df_train)}")
print(f"Test size: {len(df_test)}")
print(f"Train date range: {df_train['reading_date'].min()} to {df_train['reading_date'].max()}")
print(f"Test date range: {df_test['reading_date'].min()} to {df_test['reading_date'].max()}")

# --- Fill remaining NaN in numeric features with median from train set ---
fill_values = {}
for col in all_numeric_features:
    median_val = df_train[col].median()
    fill_values[col] = median_val if pd.notna(median_val) else 0.0

df_train[all_numeric_features] = df_train[all_numeric_features].fillna(fill_values)
df_test[all_numeric_features] = df_test[all_numeric_features].fillna(fill_values)

# Also fill categorical NaN
for col in categorical_features:
    df_train[col] = df_train[col].fillna('unknown')
    df_test[col] = df_test[col].fillna('unknown')

# --- Build ColumnTransformer ---
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', 'passthrough', all_numeric_features),
    ],
    remainder='drop'
)

X_train = preprocessor.fit_transform(df_train[categorical_features + all_numeric_features])
X_test = preprocessor.transform(df_test[categorical_features + all_numeric_features])

# Get feature names after preprocessing
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist()
feature_names = cat_feature_names + all_numeric_features

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Total features after encoding: {len(feature_names)}")

# Save preprocessor
with open(os.path.join(OUTPUT_DIR, 'preprocessor.pkl'), 'wb') as f:
    pickle.dump(preprocessor, f)
print("Saved preprocessor.pkl")

# Also save fill_values and feature lists for inference
inference_config = {
    'fill_values': {k: float(v) for k, v in fill_values.items()},
    'all_numeric_features': all_numeric_features,
    'categorical_features': categorical_features,
    'feature_names': feature_names,
}
with open(os.path.join(OUTPUT_DIR, 'inference_config.json'), 'w') as f:
    json.dump(inference_config, f, indent=2, default=str)

# ============================================================================
# STEP 8 — TRAIN THREE XGBOOST MODELS
# ============================================================================
print_step(8, "TRAIN XGBOOST MODELS")

models = {}
results = {}

for target_name, target_col in [
    ('ph', 'target_ph_next'),
    ('chlorine', 'target_chlorine_next'),
    ('turbidity', 'target_turbidity_next'),
]:
    print(f"\n--- Training model for: {target_name} ---")
    
    y_train = df_train[target_col].values
    y_test = df_test[target_col].values
    
    model = xgb.XGBRegressor(
        **XGB_PARAMS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        eval_metric='rmse',
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    
    # Best iteration
    best_iter = model.best_iteration
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"  Best trees: {best_iter}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    models[target_name] = model
    results[target_name] = {'rmse': rmse, 'mae': mae, 'r2': r2, 'best_iter': best_iter}
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, f'xgb_{target_name}.json')
    model.save_model(model_path)
    print(f"  Saved {model_path}")

# ============================================================================
# STEP 9 — SHAP EXPLAINABILITY
# ============================================================================
print_step(9, "SHAP EXPLAINABILITY")

shap_results = {}

for target_name, model in models.items():
    print(f"\n--- SHAP analysis for: {target_name} ---")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Mean absolute SHAP values per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)
    
    top5 = feature_importance.head(5)
    print(f"  Top 5 features:")
    for feat, val in top5.items():
        print(f"    {feat}: {val:.4f}")
    
    shap_results[target_name] = top5.to_dict()
    
    # Generate SHAP summary bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_names,
        plot_type='bar',
        max_display=15,
        show=False,
    )
    plt.title(f'SHAP Feature Importance — {target_name.upper()} Model', fontsize=14)
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_DIR, f'shap_summary_{target_name}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {plot_path}")

# ============================================================================
# STEP 10 — PRESCRIPTION LOGIC
# ============================================================================
print_step(10, "PRESCRIPTION LOGIC")


def prescribe_visit(pool_id, df_master_src, model_ph, model_chlorine, model_turbidity, 
                    preprocessor, fill_values, all_numeric_features, categorical_features):
    """
    Prescribe chemical dosages for the next visit to a pool.
    
    Parameters
    ----------
    pool_id : str
        The pool identifier (cleaned/lowercased).
    df_master_src : pd.DataFrame
        The master dataset with all features.
    model_ph, model_chlorine, model_turbidity : xgb.XGBRegressor
        Trained models for each target.
    preprocessor : ColumnTransformer
        Fitted preprocessing pipeline.
    fill_values : dict
        Median fill values for numeric features.
    all_numeric_features : list
        List of numeric feature column names.
    categorical_features : list
        List of categorical feature column names.
    
    Returns
    -------
    dict
        Predictions and recommendations.
    """
    pool_data = df_master_src[df_master_src['pool_id'] == pool_id]
    
    if len(pool_data) == 0:
        return {'error': f'No data found for pool_id: {pool_id}'}
    
    # Get the most recent row
    latest = pool_data.sort_values('reading_date').iloc[-1:].copy()
    
    # Prepare features
    for col in all_numeric_features:
        if col in latest.columns:
            latest[col] = latest[col].fillna(fill_values.get(col, 0))
        else:
            latest[col] = fill_values.get(col, 0)
    
    for col in categorical_features:
        if col in latest.columns:
            latest[col] = latest[col].fillna('unknown')
        else:
            latest[col] = 'unknown'
    
    X = preprocessor.transform(latest[categorical_features + all_numeric_features])
    
    # Predict
    pred_ph = float(model_ph.predict(X)[0])
    pred_cl = float(model_chlorine.predict(X)[0])
    pred_turb = float(model_turbidity.predict(X)[0])
    
    # Get pool volume for dosing calculations
    pool_vol = float(latest['pool_volume_m3'].iloc[0]) if pd.notna(latest['pool_volume_m3'].iloc[0]) else 50.0
    
    # --- Chlorine prescription ---
    if pred_cl < 1.0:
        cl_action = "⚠️ URGENT — add chlorine product"
        cl_kg = max(0, (2.0 - pred_cl) * pool_vol * 0.0025)
    elif pred_cl < 1.5:
        cl_action = "Add light maintenance dose"
        cl_kg = max(0, (2.0 - pred_cl) * pool_vol * 0.0015)
    else:
        cl_action = "✅ No chlorine action needed"
        cl_kg = 0.0
    
    # --- pH prescription ---
    if pred_ph > 7.6:
        ph_action = "Add pH minus"
        ph_kg = (pred_ph - 7.2) * pool_vol * 0.001
    elif pred_ph < 7.0:
        ph_action = "Add pH plus"
        ph_kg = (7.2 - pred_ph) * pool_vol * 0.001
    else:
        ph_action = "✅ pH in range, no action"
        ph_kg = 0.0
    
    # --- Turbidity prescription ---
    if pred_turb > 2.0:
        turb_action = "Add flocculant"
    else:
        turb_action = "✅ No flocculant needed"
    
    return {
        'pool_id': pool_id,
        'last_reading_date': str(latest['reading_date'].iloc[0]),
        'current_ph': float(latest['ph'].iloc[0]) if pd.notna(latest['ph'].iloc[0]) else None,
        'current_chlorine': float(latest['free_chlorine'].iloc[0]) if pd.notna(latest['free_chlorine'].iloc[0]) else None,
        'current_turbidity': float(latest['turbidity'].iloc[0]) if pd.notna(latest['turbidity'].iloc[0]) else None,
        'pool_volume_m3': pool_vol,
        'predictions': {
            'next_ph': round(pred_ph, 2),
            'next_chlorine': round(pred_cl, 2),
            'next_turbidity': round(pred_turb, 2),
        },
        'prescriptions': {
            'chlorine': {'action': cl_action, 'kg_needed': round(cl_kg, 3)},
            'ph': {'action': ph_action, 'kg_needed': round(ph_kg, 3)},
            'turbidity': {'action': turb_action},
        },
    }


# Test on 5 different pool IDs from the test set
test_pools = df_test['pool_id'].unique()[:5]
print(f"\nTesting prescriptions on {len(test_pools)} pools:")

example_prescriptions = []
for pid in test_pools:
    result = prescribe_visit(
        pid, df_master, models['ph'], models['chlorine'], models['turbidity'],
        preprocessor, fill_values, all_numeric_features, categorical_features
    )
    example_prescriptions.append(result)
    print(f"\n  Pool: {pid}")
    print(f"    Current:     pH={result.get('current_ph')}, Cl={result.get('current_chlorine')}, Turb={result.get('current_turbidity')}")
    preds = result['predictions']
    print(f"    Predicted:   pH={preds['next_ph']}, Cl={preds['next_chlorine']}, Turb={preds['next_turbidity']}")
    presc = result['prescriptions']
    print(f"    Chlorine Rx: {presc['chlorine']['action']} ({presc['chlorine']['kg_needed']} kg)")
    print(f"    pH Rx:       {presc['ph']['action']} ({presc['ph']['kg_needed']} kg)")
    print(f"    Turbidity:   {presc['turbidity']['action']}")

# ============================================================================
# STEP 11 — EVALUATION REPORT
# ============================================================================
print_step(11, "EVALUATION REPORT")

# --- Compute stats ---
total_pools = df_master['pool_id'].nunique()
total_readings = len(df_master)
date_min = df_master['reading_date'].min()
date_max = df_master['reading_date'].max()

# % readings with ops data (at least one ops column non-null)
ops_check_cols = [c for c in ['daily_filtration_hours', 'water_temperature', 'ph_dosing_pct'] if c in df_master.columns]
pct_with_ops = (df_master[ops_check_cols].notna().any(axis=1).sum() / total_readings * 100)

# % readings with products data
prod_check_cols = ['total_chlorine_product', 'total_ph_minus_product']
prod_check_cols = [c for c in prod_check_cols if c in df_master.columns]
pct_with_prods = (df_master[prod_check_cols].notna().any(axis=1).sum() / total_readings * 100) if prod_check_cols else 0

# Feature null rates for the final model dataset
feature_null_rates = df_model[all_numeric_features + categorical_features].isnull().mean() * 100

# --- Write report ---
report_lines = []
report_lines.append("=" * 70)
report_lines.append("  POOL PREDICTIVE MAINTENANCE — EVALUATION REPORT")
report_lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append("=" * 70)

report_lines.append("\n\n1. DATASET SUMMARY")
report_lines.append("-" * 40)
report_lines.append(f"Total pools:                    {total_pools}")
report_lines.append(f"Total readings (master):        {total_readings}")
report_lines.append(f"Date range:                     {date_min.date()} to {date_max.date()}")
report_lines.append(f"Readings with ops data:         {pct_with_ops:.1f}%")
report_lines.append(f"Readings with products data:    {pct_with_prods:.1f}%")
report_lines.append(f"Model training rows:            {len(df_train)}")
report_lines.append(f"Model test rows:                {len(df_test)}")
report_lines.append(f"Temporal cutoff:                {cutoff_date}")

report_lines.append("\n\n2. FEATURE NULL RATES (final model dataset)")
report_lines.append("-" * 40)
for feat, rate in feature_null_rates.sort_values(ascending=False).items():
    report_lines.append(f"  {feat:40s} {rate:6.1f}%")

report_lines.append("\n\n3. MODEL PERFORMANCE")
report_lines.append("-" * 40)
report_lines.append(f"{'Model':<15} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'Best Trees':>12}")
report_lines.append("-" * 55)
for name, res in results.items():
    report_lines.append(f"{name:<15} {res['rmse']:8.4f} {res['mae']:8.4f} {res['r2']:8.4f} {res['best_iter']:>12}")

report_lines.append("\n\n4. TOP 5 SHAP FEATURES PER MODEL")
report_lines.append("-" * 40)
for name, top5 in shap_results.items():
    report_lines.append(f"\n  {name.upper()} model:")
    for i, (feat, val) in enumerate(top5.items(), 1):
        report_lines.append(f"    {i}. {feat}: {val:.4f}")

report_lines.append("\n\n5. EXAMPLE PRESCRIPTIONS")
report_lines.append("-" * 40)
for ex in example_prescriptions:
    report_lines.append(f"\n  Pool: {ex['pool_id']}")
    report_lines.append(f"  Last reading: {ex['last_reading_date']}")
    report_lines.append(f"  Current: pH={ex.get('current_ph')}, Cl={ex.get('current_chlorine')}, Turb={ex.get('current_turbidity')}")
    preds = ex['predictions']
    report_lines.append(f"  Predicted next: pH={preds['next_ph']}, Cl={preds['next_chlorine']}, Turb={preds['next_turbidity']}")
    presc = ex['prescriptions']
    report_lines.append(f"  → Chlorine: {presc['chlorine']['action']} ({presc['chlorine']['kg_needed']} kg)")
    report_lines.append(f"  → pH:       {presc['ph']['action']} ({presc['ph']['kg_needed']} kg)")
    report_lines.append(f"  → Turbidity: {presc['turbidity']['action']}")

report_text = '\n'.join(report_lines)

report_path = os.path.join(OUTPUT_DIR, 'evaluation_report.txt')
with open(report_path, 'w') as f:
    f.write(report_text)
print(f"Saved {report_path}")
print(report_text)

# --- Save master dataset ---
master_path = os.path.join(OUTPUT_DIR, 'master_dataset.csv')
df_master.to_csv(master_path, index=False)
print(f"\nSaved {master_path}: {df_master.shape}")

# ============================================================================
# STEP 12 — FINAL OUTPUT SUMMARY
# ============================================================================
print_step(12, "FINAL OUTPUT SUMMARY")

expected_files = [
    'master_dataset.csv',
    'xgb_ph.json',
    'xgb_chlorine.json',
    'xgb_turbidity.json',
    'shap_summary_ph.png',
    'shap_summary_chlorine.png',
    'shap_summary_turbidity.png',
    'evaluation_report.txt',
    'preprocessor.pkl',
    'inference_config.json',
]

print("Output files:")
for fname in expected_files:
    fpath = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(fpath):
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  ✅ {fname} ({size_kb:.1f} KB)")
    else:
        print(f"  ❌ {fname} — MISSING")

print("\n" + "=" * 70)
print("  PIPELINE COMPLETE")
print("=" * 70)
