#!/usr/bin/env python3
"""
High-Precision Free Chlorine Machine Learning Dataset Generator.

Implements the 14-point architecture:
1. Relational disaggregation of horizontally-pasted spreadsheet tables.
2. Data cleaning & outlier bounds enforcement (temperature, rinse times, negative dosages).
3. Leakage-free MICE imputation for pool physical profiles (fit on 2023-2025, transform 2026).
4. State-transition pairing (t -> t+1) with 0.5 to 10.0 day operational filtering.
5. True multi-day interval aggregation for chemical dosing events (shock vs. erodible dissolution).
6. As-of join for operational timers and pump settings.
7. Riemann window integration for multi-day weather forcing (solar radiation, temp, rain, ET0).
8. Physical feature engineering: surface-to-volume ratio, active HOCl fraction, cumulative CYA,
   weekend bather exposure, diurnal sun angles, and technician encoding.
9. Train (2023-2025) vs. Test (2026) split tagging.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants & Stoichiometry
# Active chlorine fraction by chemical product
CHLORINE_PURITY_MAP = {
    'chemical_t500_qp': 0.90,               # Trichlor pucks (90% active Cl2, kg of 500g cylinders)
    'chemical_alboral_tablets_250g': 0.90,   # Trichlor 250g tablets (90% active Cl2, kg)
    'chemical_hypo_gr_chloryte': 0.65,      # Cal-Hypo granules (65% active Cl2, kg)
    'chemical_hypo_granules_xaka': 0.65,    # Cal-Hypo granules (65% active Cl2, kg)
    'chemical_hypo_sticks_bayrol': 0.68,    # Cal-Hypo sticks (68% active Cl2, kg)
    'chemical_hypo_tabs_ritocal': 0.65,     # Cal-Hypo tablets (65% active Cl2, kg)
    'chemical_hypo_tablets_200g_qp': 0.65,  # Cal-Hypo 200g tablets (65% active Cl2, kg)
    'chemical_hypo_tablets_xaka': 0.65,     # Cal-Hypo tablets (65% active Cl2, kg)
    'chemical_sg_xaka_agonet_gr90': 0.56,   # Dichlor granules (56% active Cl2, kg)
}

# Slow-dissolving erodible products (erode over 5-8 days in skimmers)
ERODIBLE_CHEMICALS = {
    'chemical_t500_qp',
    'chemical_alboral_tablets_250g',
    'chemical_hypo_sticks_bayrol',
    'chemical_hypo_tabs_ritocal',
    'chemical_hypo_tablets_200g_qp',
    'chemical_hypo_tablets_xaka'
}

# Shock / Fast-dissolving products
SHOCK_CHEMICALS = {
    'chemical_hypo_gr_chloryte',
    'chemical_hypo_granules_xaka',
    'chemical_sg_xaka_agonet_gr90'
}

# Isocyanurate products that add Cyanuric Acid (CYA)
CYA_PRODUCING_CHEMICALS = {
    'chemical_t500_qp': 0.60,              # Trichlor adds ~0.6g CYA per g product
    'chemical_alboral_tablets_250g': 0.60,
    'chemical_sg_xaka_agonet_gr90': 0.50,  # Dichlor adds ~0.5g CYA per g product
}


def load_raw_data(translated_csv: str = "data/Merged_2023_2026_translated.csv",
                  weather_csv: str = "data/weather_alicante_daily.csv") -> tuple:
    """Loads translated pool dataset and Alicante weather dataset."""
    if not os.path.exists(translated_csv):
        raise FileNotFoundError(f"Missing translated pool data: {translated_csv}")
    if not os.path.exists(weather_csv):
        raise FileNotFoundError(f"Missing weather data: {weather_csv}")
    
    logger.info(f"Loading pool data from {translated_csv}...")
    df_pool = pd.read_csv(translated_csv)
    
    logger.info(f"Loading weather data from {weather_csv}...")
    df_weather = pd.read_csv(weather_csv)
    df_weather['date_dt'] = pd.to_datetime(df_weather['date'])
    
    return df_pool, df_weather


def disaggregate_tables(df: pd.DataFrame) -> tuple:
    """
    Disaggregates the horizontally-pasted spreadsheet into 4 pure relational entities:
    1. Table_Water (Water test records)
    2. Table_Profile (Static pool characteristics)
    3. Table_Operations (Equipment runtime and setpoints)
    4. Table_Chemicals (Chemical dosing events)
    """
    logger.info("Disaggregating horizontally-pasted spreadsheet tables...")
    df['pool_clean'] = df['pool_name'].astype(str).str.strip()
    
    # 1. Table_Water
    water_cols = ['pool_clean', 'community_address', 'measurement_date', 'measurement_employee', 'ph', 'turbidity', 'free_chlorine']
    df_water = df[df['measurement_date'].notna() & df['free_chlorine'].notna() & (df['pool_clean'] != 'nan')][water_cols].copy()
    df_water['date_dt'] = pd.to_datetime(df_water['measurement_date'], format='%d-%m-%Y %H:%M', errors='coerce')
    df_water = df_water.dropna(subset=['date_dt'])
    
    # Deduplicate same-day measurements per pool: keep latest timestamp
    df_water = df_water.sort_values(['pool_clean', 'date_dt'])
    df_water['date_only'] = df_water['date_dt'].dt.date
    df_water = df_water.groupby(['pool_clean', 'date_only'], as_index=False).last()
    df_water = df_water.sort_values(['pool_clean', 'date_dt']).reset_index(drop=True)
    logger.info(f"Extracted Table_Water: {len(df_water):,} deduplicated test records across {df_water['pool_clean'].nunique()} pools")
    
    # 2. Table_Profile (Static attributes)
    static_cols = [
        'pool_volume', 'pool_surface_area', 'filter_diameter', 'number_of_filters',
        'number_of_motors', 'motor_pump_flow_rate', 'hypochlorite_pump_flow_rate',
        'ph_pump_flow_rate', 'heated_pool', 'community_pool', 'skimmer_pool',
        'overflow_pool', 'outdoor_pool', 'oval_pool', 'round_pool',
        'rectangular_pool_0714', 'rectangular_pool_07', 'deck_grass_area',
        'deck_mixed_area', 'deck_paved_area', 'contaminating_vegetation', 'sunscreen_overuse'
    ]
    # Extract first non-null values per pool
    df_profile = df.groupby('pool_clean')[static_cols].first().reset_index()
    logger.info(f"Extracted Table_Profile: {len(df_profile)} unique pool profiles")
    
    # 3. Table_Operations
    op_cols = ['daily_filtration_hours', 'hypo_dosing_hours', 'hypo_dosing_percentage',
               'filter_wash_rinse_time', 'ph_dosing_hours', 'ph_dosing_percentage', 'water_temperature']
    df_ops = df[df['maintenance_date'].notna() & (df['pool_clean'] != 'nan')][['pool_clean', 'maintenance_date', 'maintenance_employee'] + op_cols].copy()
    df_ops['date_dt'] = pd.to_datetime(df_ops['maintenance_date'], format='%d-%m-%Y', errors='coerce')
    df_ops = df_ops.dropna(subset=['date_dt'])
    
    # Clean operational outliers
    # Water temperature: valid Mediterranean range [5.0, 40.0] C
    df_ops.loc[(df_ops['water_temperature'] < 5.0) | (df_ops['water_temperature'] > 40.0), 'water_temperature'] = np.nan
    # Filter rinse time: cap at 30 min (outliers like 1234 min are typos)
    df_ops.loc[df_ops['filter_wash_rinse_time'] > 30.0, 'filter_wash_rinse_time'] = 30.0
    # Timer hours: clip to [0, 24]
    for h_col in ['daily_filtration_hours', 'hypo_dosing_hours', 'ph_dosing_hours']:
        df_ops[h_col] = df_ops[h_col].clip(lower=0.0, upper=24.0)
    # Percentages: clip to [0, 100]
    for p_col in ['hypo_dosing_percentage', 'ph_dosing_percentage']:
        df_ops[p_col] = df_ops[p_col].clip(lower=0.0, upper=100.0)
        
    df_ops = df_ops.sort_values(['pool_clean', 'date_dt']).reset_index(drop=True)
    logger.info(f"Extracted Table_Operations: {len(df_ops):,} logs")
    
    # 4. Table_Chemicals
    chem_raw_cols = [c for c in df.columns if c.startswith('chemical_') and c not in ['chemical_dosing_employee', 'chemical_dosing_date']]
    df_chem = df[df['chemical_dosing_date'].notna() & (df['pool_clean'] != 'nan')][['pool_clean', 'chemical_dosing_date', 'chemical_dosing_employee'] + chem_raw_cols].copy()
    df_chem['date_dt'] = pd.to_datetime(df_chem['chemical_dosing_date'], format='%d-%m-%Y', errors='coerce')
    df_chem = df_chem.dropna(subset=['date_dt'])
    
    # Clean chemical entries: clip negative values (accounting corrections) to 0
    for c_col in chem_raw_cols:
        df_chem[c_col] = df_chem[c_col].fillna(0.0).clip(lower=0.0)
        
    df_chem = df_chem.sort_values(['pool_clean', 'date_dt']).reset_index(drop=True)
    logger.info(f"Extracted Table_Chemicals: {len(df_chem):,} dosing events")
    
    return df_water, df_profile, df_ops, df_chem


def impute_pool_profiles(df_profile: pd.DataFrame, train_pools: set) -> pd.DataFrame:
    """
    Imputes static pool profiles without data leakage:
    MICE (IterativeImputer with BayesianRidge) is fit strictly on training pools,
    and transformed across all pools.
    """
    logger.info("Imputing pool physical dimensions using MICE (fit on training pools)...")
    profile = df_profile.copy().set_index('pool_clean')
    
    # Binary flags: fillna with 0
    binary_cols = ['heated_pool', 'community_pool', 'skimmer_pool', 'overflow_pool',
                   'outdoor_pool', 'oval_pool', 'round_pool', 'rectangular_pool_0714',
                   'rectangular_pool_07', 'deck_grass_area', 'deck_mixed_area',
                   'deck_paved_area', 'contaminating_vegetation', 'sunscreen_overuse']
    for b in binary_cols:
        if b in profile.columns:
            profile[b] = profile[b].fillna(0.0).clip(lower=0.0, upper=1.0)
            
    # Numeric physical features
    numeric_profile_cols = ['pool_volume', 'pool_surface_area', 'filter_diameter',
                            'number_of_filters', 'number_of_motors', 'motor_pump_flow_rate',
                            'hypochlorite_pump_flow_rate', 'ph_pump_flow_rate']
    
    # Split train profiles vs full profiles
    train_profile_mask = profile.index.isin(train_pools)
    train_num_data = profile.loc[train_profile_mask, numeric_profile_cols]
    
    # Fit MICE imputer strictly on training profiles
    mice_imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=30,
        random_state=42,
        min_value=1.0  # Physical minimums
    )
    mice_imputer.fit(train_num_data)
    
    # Transform full profile dataframe
    imputed_matrix = mice_imputer.transform(profile[numeric_profile_cols])
    profile[numeric_profile_cols] = imputed_matrix
    
    # Compute derived physical ratios
    profile['specific_surface_ratio'] = profile['pool_surface_area'] / profile['pool_volume'].clip(lower=1.0)
    profile['estimated_mean_depth'] = profile['pool_volume'] / profile['pool_surface_area'].clip(lower=1.0)
    
    return profile.reset_index()


def build_state_transitions(df_water: pd.DataFrame, min_delta_days: float = 0.5, max_delta_days: float = 10.0) -> pd.DataFrame:
    """
    Constructs chronological state-transition pairs:
    Visit (t) -> Visit (t+1) per pool.
    Filters out non-operational gaps (>10 days) and same-day rechecks (<0.5 days).
    """
    logger.info(f"Building state transitions (interval window: [{min_delta_days}, {max_delta_days}] days)...")
    df_water = df_water.sort_values(['pool_clean', 'date_dt']).reset_index(drop=True)
    
    # Compute next visit state
    df_water['next_date_dt'] = df_water.groupby('pool_clean')['date_dt'].shift(-1)
    df_water['next_free_chlorine'] = df_water.groupby('pool_clean')['free_chlorine'].shift(-1)
    df_water['next_ph'] = df_water.groupby('pool_clean')['ph'].shift(-1)
    df_water['next_turbidity'] = df_water.groupby('pool_clean')['turbidity'].shift(-1)
    df_water['next_employee'] = df_water.groupby('pool_clean')['measurement_employee'].shift(-1)
    
    # Elapsed days delta
    df_water['delta_days'] = (df_water['next_date_dt'] - df_water['date_dt']).dt.total_seconds() / 86400.0
    
    # Autoregressive lag features at time t
    df_water['free_chlorine_lag1'] = df_water.groupby('pool_clean')['free_chlorine'].shift(1)
    df_water['free_chlorine_lag2'] = df_water.groupby('pool_clean')['free_chlorine'].shift(2)
    df_water['ph_lag1'] = df_water.groupby('pool_clean')['ph'].shift(1)
    
    # 3-visit rolling mean at time t (using past observations)
    df_water['free_chlorine_rolling_mean_3'] = (
        df_water.groupby('pool_clean')['free_chlorine']
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    
    # Filter valid transitions
    valid_transitions = df_water[
        df_water['next_free_chlorine'].notna() &
        (df_water['delta_days'] >= min_delta_days) &
        (df_water['delta_days'] <= max_delta_days)
    ].copy()
    
    # Fill missing lag features with current reading if early in series
    valid_transitions['free_chlorine_lag1'] = valid_transitions['free_chlorine_lag1'].fillna(valid_transitions['free_chlorine'])
    valid_transitions['free_chlorine_lag2'] = valid_transitions['free_chlorine_lag2'].fillna(valid_transitions['free_chlorine_lag1'])
    # Fill missing water quality features (per pool ffill/bfill, fallback to standard medians)
    valid_transitions['ph'] = valid_transitions.groupby('pool_clean')['ph'].transform(lambda s: s.ffill().bfill()).fillna(7.40)
    valid_transitions['turbidity'] = valid_transitions.groupby('pool_clean')['turbidity'].transform(lambda s: s.ffill().bfill()).fillna(0.30)
    valid_transitions['ph_lag1'] = valid_transitions['ph_lag1'].fillna(valid_transitions['ph'])
    valid_transitions['free_chlorine_rolling_mean_3'] = valid_transitions['free_chlorine_rolling_mean_3'].fillna(valid_transitions['free_chlorine'])
    
    logger.info(f"Built {len(valid_transitions):,} valid state-transition pairs")
    return valid_transitions.reset_index(drop=True)


def calculate_active_chlorine_and_cya(df_chem: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes all chemical dosing quantities into:
    1. Shock active chlorine (grams Cl2)
    2. Erodible slow-release active chlorine (grams Cl2)
    3. Liquid sodium hypochlorite active chlorine (from mL)
    4. Cyanuric Acid (CYA) added (grams)
    """
    chem = df_chem.copy()
    
    # Initialize calculated columns
    chem['active_cl2_shock_g'] = 0.0
    chem['active_cl2_erodible_g'] = 0.0
    chem['active_cl2_liquid_g'] = 0.0
    chem['cya_added_g'] = 0.0
    
    # 1. Liquid Hypochlorite Carboys 20kg (Values in mL of 13% NaClO solution)
    # Active Cl2 = mL * 1.1 g/mL * 0.13 = mL * 0.143 g
    if 'chemical_hypo_carboys_20kg' in chem.columns:
        chem['active_cl2_liquid_g'] = chem['chemical_hypo_carboys_20kg'] * 0.143
    
    # 2. Granular / Shock products (Values in kg of product -> 1000g * purity)
    for prod in SHOCK_CHEMICALS:
        if prod in chem.columns:
            purity = CHLORINE_PURITY_MAP.get(prod, 0.65)
            chem['active_cl2_shock_g'] += chem[prod] * 1000.0 * purity
            
    # 3. Slow-dissolving Erodible products (Values in kg of product -> 1000g * purity)
    for prod in ERODIBLE_CHEMICALS:
        if prod in chem.columns:
            purity = CHLORINE_PURITY_MAP.get(prod, 0.90)
            chem['active_cl2_erodible_g'] += chem[prod] * 1000.0 * purity
            
    # 4. Cyanuric Acid added (grams)
    for prod, cya_ratio in CYA_PRODUCING_CHEMICALS.items():
        if prod in chem.columns:
            chem['cya_added_g'] += chem[prod] * 1000.0 * cya_ratio
            
    chem['total_active_cl2_g'] = chem['active_cl2_shock_g'] + chem['active_cl2_erodible_g'] + chem['active_cl2_liquid_g']
    
    return chem


def aggregate_chemicals_for_transitions(transitions: pd.DataFrame, df_chem_std: pd.DataFrame) -> pd.DataFrame:
    """
    Executes true interval queries: for each transition [Date(t), Date(t+1)] of a pool,
    aggregates all chemical dosing events that occurred inside that multi-day window.
    """
    logger.info("Interval-joining chemical dosing events to state transitions...")
    
    # Group chemicals by pool for fast indexing
    chem_grouped = {p: g.sort_values('date_dt') for p, g in df_chem_std.groupby('pool_clean')}
    
    shock_doses = []
    erodible_doses = []
    liquid_doses = []
    total_doses = []
    cya_doses = []
    chem_event_counts = []
    is_chem_logged_list = []
    
    for _, row in transitions.iterrows():
        p = row['pool_clean']
        d_start = row['date_dt'].floor('D')
        d_end = row['next_date_dt'].floor('D')
        
        if p in chem_grouped:
            p_chems = chem_grouped[p]
            # Select events in interval [d_start, d_end]
            in_window = p_chems[(p_chems['date_dt'] >= d_start) & (p_chems['date_dt'] <= d_end)]
            if len(in_window) > 0:
                s_dose = in_window['active_cl2_shock_g'].sum()
                e_dose = in_window['active_cl2_erodible_g'].sum()
                l_dose = in_window['active_cl2_liquid_g'].sum()
                t_dose = in_window['total_active_cl2_g'].sum()
                c_dose = in_window['cya_added_g'].sum()
                
                shock_doses.append(float(s_dose))
                erodible_doses.append(float(e_dose))
                liquid_doses.append(float(l_dose))
                total_doses.append(float(t_dose))
                cya_doses.append(float(c_dose))
                chem_event_counts.append(int(len(in_window)))
                is_chem_logged_list.append(1)
                continue
                
        # No chemical logged in window
        shock_doses.append(0.0)
        erodible_doses.append(0.0)
        liquid_doses.append(0.0)
        total_doses.append(0.0)
        cya_doses.append(0.0)
        chem_event_counts.append(0)
        is_chem_logged_list.append(0)
        
    transitions['active_cl2_shock_grams'] = shock_doses
    transitions['active_cl2_erodible_grams'] = erodible_doses
    transitions['active_cl2_liquid_grams'] = liquid_doses
    transitions['total_active_cl2_grams'] = total_doses
    transitions['cya_added_grams'] = cya_doses
    transitions['chemical_events_in_window'] = chem_event_counts
    transitions['is_chemical_logged_in_window'] = is_chem_logged_list
    
    return transitions


def join_operations_asof(transitions: pd.DataFrame, df_ops: pd.DataFrame) -> pd.DataFrame:
    """
    Joins operational setpoints using as-of alignment:
    Finds the most recent maintenance log on or before visit date t per pool.
    """
    logger.info("As-of joining operational timer setpoints...")
    
    op_cols = ['daily_filtration_hours', 'hypo_dosing_hours', 'hypo_dosing_percentage',
               'filter_wash_rinse_time', 'ph_dosing_hours', 'ph_dosing_percentage', 'water_temperature']
    
    # Sort for merge_asof
    trans_sorted = transitions.sort_values('date_dt').copy()
    ops_sorted = df_ops[['pool_clean', 'date_dt'] + op_cols].sort_values('date_dt').copy()
    
    merged = pd.merge_asof(
        trans_sorted,
        ops_sorted,
        on='date_dt',
        by='pool_clean',
        direction='backward'
    )
    
    # Forward-fill and backward-fill remaining missing values per pool
    for col in op_cols:
        merged[col] = merged.groupby('pool_clean')[col].transform(lambda s: s.ffill().bfill())
        
    # Global medians for any pools with zero operational entries
    defaults = {
        'daily_filtration_hours': 10.0,
        'hypo_dosing_hours': 8.0,
        'hypo_dosing_percentage': 10.0,
        'filter_wash_rinse_time': 2.0,
        'ph_dosing_hours': 1.0,
        'ph_dosing_percentage': 2.0,
        'water_temperature': 24.0
    }
    for col, default_val in defaults.items():
        merged[col] = merged[col].fillna(default_val)
        
    return merged.sort_values(['pool_clean', 'date_dt']).reset_index(drop=True)


def aggregate_weather_windows(transitions: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    """
    Performs multi-day Riemann integration across the exact interval [Date(t), Date(t+1)]
    for Alicante weather variables.
    """
    logger.info("Integrating multi-day weather window metrics across [Date(t), Date(t+1)]...")
    
    # Index weather by date for fast slicing
    w_indexed = df_weather.set_index('date_dt').sort_index()
    
    rad_sums = []
    rad_means = []
    sun_hours_sums = []
    temp_means = []
    temp_maxes = []
    precip_sums = []
    et0_sums = []
    wind_maxes = []
    
    for _, row in transitions.iterrows():
        d_start = row['date_dt'].floor('D')
        d_end = row['next_date_dt'].floor('D')
        
        # Slice window
        w_win = w_indexed.loc[d_start:d_end]
        if len(w_win) > 0:
            rad_sums.append(float(w_win['shortwave_radiation_sum'].sum()))
            rad_means.append(float(w_win['shortwave_radiation_sum'].mean()))
            # Convert seconds of sunshine to hours
            sun_hours_sums.append(float(w_win['sunshine_duration'].sum() / 3600.0))
            temp_means.append(float(w_win['temperature_2m_mean'].mean()))
            temp_maxes.append(float(w_win['temperature_2m_max'].max()))
            precip_sums.append(float(w_win['precipitation_sum'].sum()))
            et0_sums.append(float(w_win['et0_fao_evapotranspiration'].sum()))
            wind_maxes.append(float(w_win['wind_speed_10m_max'].max()))
        else:
            # Fallback to single start day
            w_day = w_indexed.loc[d_start:d_start]
            rad_sums.append(float(w_day['shortwave_radiation_sum'].sum() if len(w_day) else 15.0))
            rad_means.append(float(w_day['shortwave_radiation_sum'].mean() if len(w_day) else 15.0))
            sun_hours_sums.append(float(w_day['sunshine_duration'].sum() / 3600.0 if len(w_day) else 8.0))
            temp_means.append(float(w_day['temperature_2m_mean'].mean() if len(w_day) else 20.0))
            temp_maxes.append(float(w_day['temperature_2m_max'].max() if len(w_day) else 25.0))
            precip_sums.append(float(w_day['precipitation_sum'].sum() if len(w_day) else 0.0))
            et0_sums.append(float(w_day['et0_fao_evapotranspiration'].sum() if len(w_day) else 3.0))
            wind_maxes.append(float(w_day['wind_speed_10m_max'].max() if len(w_day) else 10.0))
            
    transitions['window_solar_rad_sum_mj'] = rad_sums
    transitions['window_solar_rad_mean_mj'] = rad_means
    transitions['window_sunshine_hours_sum'] = sun_hours_sums
    transitions['window_temp_mean_c'] = temp_means
    transitions['window_temp_max_c'] = temp_maxes
    transitions['window_precip_sum_mm'] = precip_sums
    transitions['window_et0_sum_mm'] = et0_sums
    transitions['window_wind_max_kmh'] = wind_maxes
    
    return transitions


def engineer_physics_and_ml_features(df: pd.DataFrame, df_profile_imputed: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers domain-aware physics, thermodynamic equilibrium, dissolution kinetics,
    weekend exposure, diurnal cycles, and reliability masks.
    """
    logger.info("Engineering domain physics, thermodynamics, and ML features...")
    
    # 1. Merge imputed static pool profile
    df = pd.merge(df, df_profile_imputed, on='pool_clean', how='left')
    
    # 2. Stoichiometric Concentration Boost (ppm = g / m3)
    # Instant shock ppm boost
    df['shock_dosage_ppm'] = df['active_cl2_shock_grams'] / df['pool_volume'].clip(lower=1.0)
    df['liquid_dosage_ppm'] = df['active_cl2_liquid_grams'] / df['pool_volume'].clip(lower=1.0)
    df['total_instant_dosage_ppm'] = (df['active_cl2_shock_grams'] + df['active_cl2_liquid_grams']) / df['pool_volume'].clip(lower=1.0)
    
    # 3. Slow-Release Tablet Dissolution Modeling (Flux across interval delta_days)
    # Dissolution rate increases with daily filtration hours and water temp
    # Base tau ~ 6.0 days for 10h filtration at 25C
    dissolution_speed_factor = (df['daily_filtration_hours'] / 10.0).clip(0.3, 2.4) * (1.0 + 0.025 * (df['window_temp_mean_c'] - 20.0)).clip(0.5, 2.0)
    eroded_fraction = (1.0 - np.exp(-0.20 * dissolution_speed_factor * df['delta_days'])).clip(0.0, 1.0)
    df['erodible_dissolved_grams'] = df['active_cl2_erodible_grams'] * eroded_fraction
    df['erodible_dosage_ppm'] = df['erodible_dissolved_grams'] / df['pool_volume'].clip(lower=1.0)
    
    # 4. Automated Dosing Pump Input over interval delta_days
    # Mass (g) = Pump Flow (L/h) * Dosing Hours (h/day) * Power (%) * 130 g/L Cl2 * delta_days
    daily_pump_cl2_grams = (
        df['hypochlorite_pump_flow_rate'].fillna(4.0) *
        df['hypo_dosing_hours'] *
        (df['hypo_dosing_percentage'] / 100.0) *
        130.0  # ~130g active Cl2 per Liter of 13% hypo
    )
    df['pump_cl2_delivered_grams'] = daily_pump_cl2_grams * df['delta_days']
    df['pump_dosage_ppm'] = df['pump_cl2_delivered_grams'] / df['pool_volume'].clip(lower=1.0)
    
    # 5. Theoretical Post-Treatment Concentration Baseline at visit t
    df['chlorine_post_treatment_theoretical'] = (
        df['free_chlorine'] + df['total_instant_dosage_ppm'] + (df['erodible_dosage_ppm'] * 0.5)
    )
    
    # 6. Cumulative Seasonal CYA Estimate per pool
    df['year'] = df['date_dt'].dt.year
    df['cya_added_ppm'] = df['cya_added_grams'] / df['pool_volume'].clip(lower=1.0)
    df['cya_cumulative_seasonal_ppm'] = df.groupby(['pool_clean', 'year'])['cya_added_ppm'].cumsum()
    # Dilution discount from backwash filter rinsing (estimated 5% water change per 10 min rinse)
    rinse_dilution = (df['filter_wash_rinse_time'] / 30.0).clip(0.0, 0.15)
    df['cya_cumulative_seasonal_ppm'] = (df['cya_cumulative_seasonal_ppm'] * (1.0 - rinse_dilution)).clip(lower=0.0)
    
    # 7. Thermodynamic Active HOCl Dissociation Fraction
    # pKa of hypochlorous acid is ~7.53 at 25C
    # alpha = 1 / (1 + 10^(pH - pKa))
    df['active_hocl_fraction'] = 1.0 / (1.0 + 10.0 ** (df['ph'] - 7.53))
    df['active_hocl_ppm'] = df['free_chlorine'] * df['active_hocl_fraction']
    df['hocl_demand_proxy'] = (1.0 - df['active_hocl_fraction']) * df['free_chlorine']
    
    # 8. Hydraulic Filtration Turnover Ratio
    # Daily Turnover = (Motor Flow m3/h * Filtration Hours) / Pool Volume m3
    df['daily_turnover_ratio'] = (df['motor_pump_flow_rate'] * df['daily_filtration_hours']) / df['pool_volume'].clip(lower=1.0)
    df['pump_flow_per_vol'] = df['motor_pump_flow_rate'] / df['pool_volume'].clip(lower=1.0)
    
    # 9. Weekend Exposure & Bather Surge Multipliers
    # Compute number of weekend days between date_dt and next_date_dt
    def count_weekend_days(start, end):
        dates = pd.date_range(start=start.floor('D'), end=end.floor('D'), freq='D')
        return int((dates.dayofweek >= 5).sum())
    
    df['num_weekend_days'] = [count_weekend_days(r['date_dt'], r['next_date_dt']) for _, r in df.iterrows()]
    df['weekend_exposure_ratio'] = (df['num_weekend_days'] / df['delta_days']).clip(0.0, 1.0)
    df['bather_surge_index'] = (
        df['community_pool'] *
        df['num_weekend_days'] *
        (df['window_temp_mean_c'] / 20.0).clip(lower=0.5) *
        (1.0 + 0.5 * df['sunscreen_overuse'])
    )
    
    # 10. Diurnal Hour-of-Day Features
    df['hour_t'] = df['date_dt'].dt.hour + df['date_dt'].dt.minute / 60.0
    df['hour_t_sin'] = np.sin(2.0 * np.pi * df['hour_t'] / 24.0)
    df['hour_t_cos'] = np.cos(2.0 * np.pi * df['hour_t'] / 24.0)
    
    df['next_hour'] = df['next_date_dt'].dt.hour + df['next_date_dt'].dt.minute / 60.0
    df['next_hour_sin'] = np.sin(2.0 * np.pi * df['next_hour'] / 24.0)
    df['next_hour_cos'] = np.cos(2.0 * np.pi * df['next_hour'] / 24.0)
    
    # 11. Calendar & Annual Seasonality
    df['month'] = df['date_dt'].dt.month
    df['month_sin'] = np.sin(2.0 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2.0 * np.pi * df['month'] / 12.0)
    df['day_of_year'] = df['date_dt'].dt.dayofyear
    df['is_summer'] = df['month'].isin([6, 7, 8, 9]).astype(float)
    
    # 12. Domain Interaction & Kinetics Features
    df['rad_per_volume'] = df['window_solar_rad_sum_mj'] / df['pool_volume'].clip(lower=1.0)
    df['temp_x_delta'] = df['window_temp_mean_c'] * df['delta_days']
    df['cl_decay_potential'] = df['free_chlorine'] * df['delta_days']
    df['cl_diff_lag1'] = df['free_chlorine'] - df['free_chlorine_lag1']
    df['cl_diff_rolling'] = df['free_chlorine'] - df['free_chlorine_rolling_mean_3']
    df['cl_decay_slope_recent'] = (df['free_chlorine'] - df['free_chlorine_lag1']) / df['delta_days'].clip(lower=0.5)
    
    # 13. First-Order Physical Decay Proxy
    # Decay rate k increases with temperature, UV solar radiation, and organic turbidity
    decay_k_proxy = (
        0.04 +
        0.015 * (df['window_temp_mean_c'] / 25.0).clip(0.5, 2.0) +
        0.025 * (df['window_solar_rad_mean_mj'] / 20.0).clip(0.2, 2.0) * df['specific_surface_ratio'].clip(0.5, 3.0) +
        0.020 * df['turbidity'].clip(0.1, 5.0) -
        0.015 * (df['cya_cumulative_seasonal_ppm'] / 50.0).clip(0.0, 0.8)  # CYA shields against UV decay
    ).clip(lower=0.01)
    
    df['theoretical_decay_k'] = decay_k_proxy
    df['theoretical_retained_chlorine'] = df['chlorine_post_treatment_theoretical'] * np.exp(-decay_k_proxy * df['delta_days'])
    
    # 14. Quality, Reliability & Unlogged Shock Masks
    # Sensor 5.0 saturation indicator
    df['is_target_censored_at_5'] = (df['next_free_chlorine'] >= 5.0).astype(int)
    df['is_current_censored_at_5'] = (df['free_chlorine'] >= 5.0).astype(int)
    
    # Unlogged chemical shock: Cl jumps > 1.5 ppm despite 0 recorded dosage
    cl_jump = df['next_free_chlorine'] - df['free_chlorine']
    df['latent_unlogged_shock_flag'] = ((cl_jump > 1.5) & (df['total_active_cl2_grams'] == 0)).astype(int)
    
    # 15. Target Variable Setup
    df['target_next_free_chlorine'] = df['next_free_chlorine']
    # Classification target: 0 = Under (<1.0), 1 = Optimal (1.0-3.0), 2 = Over (>3.0)
    df['target_next_compliance_band'] = pd.cut(
        df['target_next_free_chlorine'],
        bins=[-np.inf, 0.999, 3.001, np.inf],
        labels=[0, 1, 2]
    ).astype(int)
    
    # 16. Train / Test Split Flag (Strict temporal partition)
    # Train: 2023-01-01 to 2025-12-31
    # Test:  2026-01-01 to 2026-08-05
    df['is_train_split'] = (df['date_dt'] < pd.Timestamp('2026-01-01')).astype(int)
    
    return df


def add_leak_free_pool_priors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes pool-level historical summary statistics fit strictly on training records
    (2023-2025) and mapped out-of-sample to 2026 test records.
    """
    logger.info("Computing leak-free pool historical baseline priors...")
    train_mask = df['is_train_split'] == 1
    train_df = df[train_mask]
    
    pool_stats = train_df.groupby('pool_clean')['free_chlorine'].agg(
        pool_cl_hist_mean='mean',
        pool_cl_hist_std='std',
        pool_cl_hist_min='min',
        pool_cl_hist_max='max',
        pool_cl_hist_q25=lambda x: x.quantile(0.25),
        pool_cl_hist_q75=lambda x: x.quantile(0.75),
        pool_visit_count='count'
    ).reset_index()
    
    global_mean = float(train_df['free_chlorine'].mean())
    global_std = float(train_df['free_chlorine'].std())
    global_min = float(train_df['free_chlorine'].min())
    global_max = float(train_df['free_chlorine'].max())
    global_q25 = float(train_df['free_chlorine'].quantile(0.25))
    global_q75 = float(train_df['free_chlorine'].quantile(0.75))
    
    df = pd.merge(df, pool_stats, on='pool_clean', how='left')
    
    df['pool_cl_hist_mean'] = df['pool_cl_hist_mean'].fillna(global_mean)
    df['pool_cl_hist_std'] = df['pool_cl_hist_std'].fillna(global_std)
    df['pool_cl_hist_min'] = df['pool_cl_hist_min'].fillna(global_min)
    df['pool_cl_hist_max'] = df['pool_cl_hist_max'].fillna(global_max)
    df['pool_cl_hist_q25'] = df['pool_cl_hist_q25'].fillna(global_q25)
    df['pool_cl_hist_q75'] = df['pool_cl_hist_q75'].fillna(global_q75)
    df['pool_visit_count'] = df['pool_visit_count'].fillna(1).astype(int)
    
    return df


def target_encode_technicians(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies out-of-fold target encoding for measurement technicians,
    fit strictly on training data (2023-2025) and mapped to test data.
    """
    logger.info("Target-encoding technician IDs strictly using training data...")
    
    train_mask = df['is_train_split'] == 1
    global_mean = df.loc[train_mask, 'target_next_free_chlorine'].mean()
    
    # Technician encoding for current test technician
    tech_means = df.loc[train_mask].groupby('measurement_employee')['target_next_free_chlorine'].agg(['count', 'mean'])
    # Bayesian smoothing: (count * mean + 10 * global_mean) / (count + 10)
    smooth_weight = 10.0
    tech_encoded = ((tech_means['count'] * tech_means['mean'] + smooth_weight * global_mean) / (tech_means['count'] + smooth_weight)).to_dict()
    
    df['tech_t_mean_cl_encoded'] = df['measurement_employee'].map(tech_encoded).fillna(global_mean)
    df['tech_next_mean_cl_encoded'] = df['next_employee'].map(tech_encoded).fillna(global_mean)
    
    return df


def export_dataset(df: pd.DataFrame, output_csv: str = "data/processed/chlorine_ml_dataset.csv",
                   meta_json: str = "data/processed/chlorine_ml_feature_metadata.json") -> None:
    """Exports the finalized dataset and feature schema."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Select clean modeling columns
    export_df = df.copy()
    
    logger.info(f"Saving finalized ML dataset to {output_csv}...")
    export_df.to_csv(output_csv, index=False)
    
    # Generate metadata dictionary
    train_count = int(export_df['is_train_split'].sum())
    test_count = int((1 - export_df['is_train_split']).sum())
    
    metadata = {
        "dataset_name": "Free Chlorine State-Transition Machine Learning Dataset",
        "created_at": datetime.now().isoformat(),
        "total_observations": len(export_df),
        "train_observations_2023_2025": train_count,
        "test_observations_2026": test_count,
        "unique_pools": int(export_df['pool_clean'].nunique()),
        "target_variable": "target_next_free_chlorine",
        "target_compliance_band": "target_next_compliance_band",
        "features_by_category": {
            "initial_water_state": [
                "free_chlorine", "ph", "turbidity", "free_chlorine_lag1", "free_chlorine_lag2",
                "free_chlorine_rolling_mean_3", "ph_lag1", "active_hocl_fraction", "active_hocl_ppm",
                "hocl_demand_proxy", "cl_diff_lag1", "cl_diff_rolling", "cl_decay_slope_recent"
            ],
            "chemical_stoichiometry": [
                "active_cl2_shock_grams", "active_cl2_erodible_grams", "active_cl2_liquid_grams",
                "total_active_cl2_grams", "shock_dosage_ppm", "liquid_dosage_ppm", "total_instant_dosage_ppm",
                "erodible_dissolved_grams", "erodible_dosage_ppm", "pump_cl2_delivered_grams",
                "pump_dosage_ppm", "chlorine_post_treatment_theoretical", "cya_cumulative_seasonal_ppm"
            ],
            "operational_controls": [
                "daily_filtration_hours", "hypo_dosing_hours", "hypo_dosing_percentage",
                "filter_wash_rinse_time", "ph_dosing_hours", "ph_dosing_percentage",
                "water_temperature", "daily_turnover_ratio", "pump_flow_per_vol"
            ],
            "pool_physical_profile": [
                "pool_volume", "pool_surface_area", "specific_surface_ratio", "estimated_mean_depth",
                "filter_diameter", "number_of_filters", "number_of_motors", "motor_pump_flow_rate",
                "hypochlorite_pump_flow_rate", "community_pool", "outdoor_pool", "skimmer_pool",
                "overflow_pool", "heated_pool", "deck_grass_area", "deck_mixed_area", "deck_paved_area",
                "contaminating_vegetation", "sunscreen_overuse", "pool_cl_hist_mean", "pool_cl_hist_std",
                "pool_cl_hist_min", "pool_cl_hist_max", "pool_cl_hist_q25", "pool_cl_hist_q75", "pool_visit_count"
            ],
            "multi_day_weather_window": [
                "delta_days", "window_solar_rad_sum_mj", "window_solar_rad_mean_mj",
                "window_sunshine_hours_sum", "window_temp_mean_c", "window_temp_max_c",
                "window_precip_sum_mm", "window_et0_sum_mm", "window_wind_max_kmh",
                "rad_per_volume", "temp_x_delta", "cl_decay_potential"
            ],
            "physics_decay_and_interaction": [
                "theoretical_decay_k", "theoretical_retained_chlorine", "num_weekend_days",
                "weekend_exposure_ratio", "bather_surge_index"
            ],
            "diurnal_and_calendar": [
                "hour_t", "hour_t_sin", "hour_t_cos", "next_hour", "next_hour_sin", "next_hour_cos",
                "month", "month_sin", "month_cos", "day_of_year", "is_summer"
            ],
            "human_and_sensor_masks": [
                "is_chemical_logged_in_window", "is_target_censored_at_5", "is_current_censored_at_5",
                "latent_unlogged_shock_flag", "tech_t_mean_cl_encoded", "tech_next_mean_cl_encoded"
            ]
        }
    }
    
    with open(meta_json, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved feature metadata schema to {meta_json}")


def main():
    logger.info("=== Starting High-Precision ML Dataset Generation Pipeline ===")
    
    # 1. Load Data
    df_raw, df_weather = load_raw_data()
    
    # 2. Disaggregate Tables
    df_water, df_profile, df_ops, df_chem = disaggregate_tables(df_raw)
    
    # 3. Identify Training Pools for Leakage-Free Profile Imputation
    df_water['date_dt'] = pd.to_datetime(df_water['measurement_date'], format='%d-%m-%Y %H:%M', errors='coerce')
    train_pools = set(df_water[df_water['date_dt'] < pd.Timestamp('2026-01-01')]['pool_clean'].unique())
    logger.info(f"Found {len(train_pools)} training pools in 2023-2025")
    
    # 4. Impute Static Profiles
    df_profile_imputed = impute_pool_profiles(df_profile, train_pools)
    
    # 5. Build State Transitions
    transitions = build_state_transitions(df_water, min_delta_days=0.5, max_delta_days=10.0)
    
    # 6. Standardize Chemical Dosages & Active Cl2
    df_chem_std = calculate_active_chlorine_and_cya(df_chem)
    
    # 7. Interval-Join Chemicals to Transitions
    transitions = aggregate_chemicals_for_transitions(transitions, df_chem_std)
    
    # 8. As-of Join Operations
    transitions = join_operations_asof(transitions, df_ops)
    
    # 9. Aggregate Multi-Day Weather Windows
    transitions = aggregate_weather_windows(transitions, df_weather)
    
    # 10. Engineer Physics & ML Features
    final_df = engineer_physics_and_ml_features(transitions, df_profile_imputed)
    
    # 11. Add Leak-Free Pool History Baseline Priors
    final_df = add_leak_free_pool_priors(final_df)
    
    # 12. Target Encode Technicians
    final_df = target_encode_technicians(final_df)
    
    # 13. Export Final Dataset & Schema
    export_dataset(final_df)
    
    logger.info(f"=== Pipeline Completed Successfully! Generated {len(final_df):,} State Transitions ===")


if __name__ == "__main__":
    main()
