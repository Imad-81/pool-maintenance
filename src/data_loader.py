"""
Data Loader & Preprocessing Module for Pool Data Analysis.
Handles ingestion of Merged_2023_2026.xlsx, table extraction, pool metadata propagation,
date standardization, relational merging, and feature engineering for chlorine prediction.
"""

import os
import re
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

# Define static pool attribute columns
STATIC_POOL_COLS = [
    'Volumen piscina',
    'Superficie piscina',
    'Caudal del motor',
    'Caudal bomba hipoclorito',
    'Caudal bomba de PH',
    'Diametro filtro',
    'Numero de filtros',
    'Número de motores',
    'PISCINA CLIMATIZADA',
    'PISCINA COMUNITARIA',
    'Piscina con skimmers',
    'Piscina desbordante',
    'PISCINA EXTERIOR',
    'Piscina ovalada',
    'PISCINA PARTICULAR',
    'PISCINA PUBLICA',
    '(0714) Piscina rectangular',
    '(07) Piscina rectangular',
    'Piscina redonda',
    'ABUSO CREMAS PROTECCION',
    'VEGETACION CONTAMINANTE',
    'Zona playa césped',
    'Zona PLAYA mixta',
    'Zona PLAYA pavimentada'
]

# Primary operational control columns
OPERATIONAL_COLS = [
    'Horas dosificacion PH',
    'Horas filtracion diarias',
    'Porcentaje dosificación PH',
    'Tiempo lavado /enjuague filtro',
    'Horas dosificación hipo',
    'Porcentaje dosificación hipoclorito',
    'Temperatura agua'
]

# Chemical dosage columns
CHEMICAL_COLS = [
    'T-500 (GRUPO QP)',
    'ALBORAL TABLETAS 250 GRS RF. 201710',
    'FLOVIL PASTILLAS',
    'HIPO GARRAFAS 20KG.',
    'HIPO GR CHLORYTE',
    'HIPO GRANULADO XAKA',
    'HIPO STICKS BAYROL',
    'HIPO TAB. RITOCAL',
    'HIPO TABLETAS 200Gr. QP',
    'HIPO TABLETAS XAKA',
    'PH MINUS GRANULADO 6kg',
    'PH MINUS LIQUIDO 13.5 KG',
    'PH MINUS LIQUIDO 27 KG.',
    'PROTECT & SHINE',
    'SG XAKA (AGONET GR90)',
    'SUPERKLAR'
]

# Chlorine-related chemical dosage columns (to aggregate total chlorine added)
CHLORINE_CHEMICALS = [
    'T-500 (GRUPO QP)',
    'ALBORAL TABLETAS 250 GRS RF. 201710',
    'HIPO GARRAFAS 20KG.',
    'HIPO GR CHLORYTE',
    'HIPO GRANULADO XAKA',
    'HIPO STICKS BAYROL',
    'HIPO TAB. RITOCAL',
    'HIPO TABLETAS 200Gr. QP',
    'HIPO TABLETAS XAKA'
]


def load_raw_data(filepath: str = "Merged_2023_2026.xlsx") -> pd.DataFrame:
    """Loads raw Excel file into DataFrame."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at: {filepath}")
    df = pd.read_excel(filepath)
    return df


def clean_pool_names(series: pd.Series) -> pd.Series:
    """Cleans and standardizes pool name strings."""
    return series.astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)


def extract_static_pool_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts static pool characteristics by grouping per pool.
    In the raw dataset, pool attributes were recorded sparsely on pool header rows.
    """
    valid_df = df[df['PISCINA'].notna() & (df['PISCINA'] != 'nan')].copy()
    valid_df['PISCINA_CLEAN'] = clean_pool_names(valid_df['PISCINA'])
    
    # Pool metadata aggregation
    agg_dict = {}
    for col in STATIC_POOL_COLS:
        if col in valid_df.columns:
            agg_dict[col] = 'first'
    if 'COMUNIDAD' in valid_df.columns:
        agg_dict['COMUNIDAD'] = 'first'
        
    pool_profiles = valid_df.groupby('PISCINA_CLEAN').agg(agg_dict).reset_index()
    
    # Consolidate rectangular pool columns if present
    rect_cols = [c for c in pool_profiles.columns if 'rectangular' in c.lower()]
    if rect_cols:
        pool_profiles['Piscina rectangular'] = pool_profiles[rect_cols].bfill(axis=1).iloc[:, 0]
        
    # Consolidate pool shape
    pool_profiles['Forma piscina'] = 'Desconocida'
    if 'Piscina rectangular' in pool_profiles.columns:
        pool_profiles.loc[pool_profiles['Piscina rectangular'].notna() & (pool_profiles['Piscina rectangular'] > 0), 'Forma piscina'] = 'Rectangular'
    if 'Piscina ovalada' in pool_profiles.columns:
        pool_profiles.loc[pool_profiles['Piscina ovalada'].notna() & (pool_profiles['Piscina ovalada'] > 0), 'Forma piscina'] = 'Ovalada'
    if 'Piscina redonda' in pool_profiles.columns:
        pool_profiles.loc[pool_profiles['Piscina redonda'].notna() & (pool_profiles['Piscina redonda'] > 0), 'Forma piscina'] = 'Redonda'

    # Consolidate beach zone
    pool_profiles['Tipo playa'] = 'No especificado'
    if 'Zona playa césped' in pool_profiles.columns:
        pool_profiles.loc[pool_profiles['Zona playa césped'] > 0, 'Tipo playa'] = 'Césped'
    if 'Zona PLAYA pavimentada' in pool_profiles.columns:
        pool_profiles.loc[pool_profiles['Zona PLAYA pavimentada'] > 0, 'Tipo playa'] = 'Pavimentada'
    if 'Zona PLAYA mixta' in pool_profiles.columns:
        pool_profiles.loc[pool_profiles['Zona PLAYA mixta'] > 0, 'Tipo playa'] = 'Mixta'

    return pool_profiles


def extract_and_align_subtables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Separates the 3 horizontal blocks into clean standalone relational tables:
    1. Water Measurements (FECHA, PISCINA, PH, TURBIDEZ, CLORO LIBRE)
    2. Operational & Dosing Controls (FECHA.1, PISCINA, Horas filtracion, Horas hipo, Temperatura)
    3. Chemical Consumptions (FECHA.2, PISCINA, T-500, HIPO..., PH MINUS...)
    """
    # 1. Water Measurements Table
    m_cols = ['PISCINA', 'COMUNIDAD', 'FECHA', 'EMPLEADO', 'PH', 'TURBIDEZ', 'CLORO LIBRE']
    measurements = df[[c for c in m_cols if c in df.columns]].dropna(subset=['FECHA', 'PISCINA']).copy()
    measurements['PISCINA_CLEAN'] = clean_pool_names(measurements['PISCINA'])
    measurements['FECHA_DT'] = pd.to_datetime(measurements['FECHA'], format='mixed', dayfirst=True, errors='coerce')
    measurements = measurements.dropna(subset=['FECHA_DT'])
    measurements['FECHA_DATE'] = measurements['FECHA_DT'].dt.date
    measurements['HORA_MEDICION'] = measurements['FECHA_DT'].dt.hour + measurements['FECHA_DT'].dt.minute / 60.0

    # 2. Operations & Dosing Table
    o_cols = ['PISCINA', 'EMPLEADO.1', 'FECHA.1'] + [c for c in OPERATIONAL_COLS if c in df.columns]
    operations = df[[c for c in o_cols if c in df.columns]].dropna(subset=['FECHA.1', 'PISCINA']).copy()
    operations['PISCINA_CLEAN'] = clean_pool_names(operations['PISCINA'])
    operations['FECHA_DT'] = pd.to_datetime(operations['FECHA.1'], format='mixed', dayfirst=True, errors='coerce')
    operations = operations.dropna(subset=['FECHA_DT'])
    operations['FECHA_DATE'] = operations['FECHA_DT'].dt.date
    if 'EMPLEADO.1' in operations.columns:
        operations = operations.rename(columns={'EMPLEADO.1': 'EMPLEADO_OPERACIONES'})

    # 3. Chemical Consumptions Table
    c_cols = ['PISCINA', 'EMPLEADO.2', 'FECHA.2'] + [c for c in CHEMICAL_COLS if c in df.columns]
    chemicals = df[[c for c in c_cols if c in df.columns]].dropna(subset=['FECHA.2', 'PISCINA']).copy()
    chemicals['PISCINA_CLEAN'] = clean_pool_names(chemicals['PISCINA'])
    chemicals['FECHA_DT'] = pd.to_datetime(chemicals['FECHA.2'], format='mixed', dayfirst=True, errors='coerce')
    chemicals = chemicals.dropna(subset=['FECHA_DT'])
    chemicals['FECHA_DATE'] = chemicals['FECHA_DT'].dt.date
    if 'EMPLEADO.2' in chemicals.columns:
        chemicals = chemicals.rename(columns={'EMPLEADO.2': 'EMPLEADO_QUIMICOS'})

    return measurements, operations, chemicals


def build_unified_dataset(filepath: str = "Merged_2023_2026.xlsx") -> pd.DataFrame:
    """
    Builds the unified, enriched, analysis-ready dataset by:
    - Extracting and propagating pool static profiles
    - Daily aggregating operations and chemical dosages
    - Merging on (PISCINA_CLEAN, FECHA_DATE)
    - Engineering temporal, lag, and dosage features.
    """
    raw_df = load_raw_data(filepath)
    
    # 1. Pool Static Metadata
    pool_profiles = extract_static_pool_profile(raw_df)
    
    # 2. Extract Subtables
    measurements, operations, chemicals = extract_and_align_subtables(raw_df)
    
    # 3. Aggregate Operations by Pool and Date (handling multiple daily entries via mean)
    op_num_cols = [c for c in OPERATIONAL_COLS if c in operations.columns]
    op_daily = operations.groupby(['PISCINA_CLEAN', 'FECHA_DATE'])[op_num_cols].agg('mean').reset_index()

    # 4. Aggregate Chemicals by Pool and Date (summing chemical quantities applied per day)
    chem_num_cols = [c for c in CHEMICAL_COLS if c in chemicals.columns]
    chem_daily = chemicals.groupby(['PISCINA_CLEAN', 'FECHA_DATE'])[chem_num_cols].agg('sum').reset_index()

    # Calculate total chlorine product dose units added per day
    present_cl_chems = [c for c in CHLORINE_CHEMICALS if c in chem_daily.columns]
    chem_daily['TOTAL_CLORO_QUIMICO_DOSIS'] = chem_daily[present_cl_chems].sum(axis=1)

    # 5. Merge measurements with static pool profiles
    merged = pd.merge(measurements, pool_profiles, on='PISCINA_CLEAN', how='left')

    # 6. Merge with daily operations and chemical logs
    merged = pd.merge(merged, op_daily, on=['PISCINA_CLEAN', 'FECHA_DATE'], how='left')
    merged = pd.merge(merged, chem_daily, on=['PISCINA_CLEAN', 'FECHA_DATE'], how='left')

    # Fill 0 for chemicals on measurement days where no chemical addition was logged
    for c in chem_num_cols + ['TOTAL_CLORO_QUIMICO_DOSIS']:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0.0)

    # 7. Sort by Pool and Date for Time-Series Lag Engineering
    merged = merged.sort_values(by=['PISCINA_CLEAN', 'FECHA_DT']).reset_index(drop=True)

    # 8. Temporal Features
    merged['ANIO'] = merged['FECHA_DT'].dt.year
    merged['MES'] = merged['FECHA_DT'].dt.month
    merged['DIA'] = merged['FECHA_DT'].dt.day
    merged['DIA_SEMANA'] = merged['FECHA_DT'].dt.dayofweek  # 0=Monday, 6=Sunday
    merged['ES_FIN_DE_SEMANA'] = (merged['DIA_SEMANA'] >= 5).astype(int)
    merged['ES_VERANO'] = merged['MES'].isin([6, 7, 8, 9]).astype(int)
    merged['TRIMESTRE'] = merged['FECHA_DT'].dt.quarter
    merged['DIA_DEL_ANIO'] = merged['FECHA_DT'].dt.dayofyear

    # Cyclical seasonal encoding
    merged['MES_SIN'] = np.sin(2 * np.pi * merged['MES'] / 12.0)
    merged['MES_COS'] = np.cos(2 * np.pi * merged['MES'] / 12.0)
    merged['HORA_SIN'] = np.sin(2 * np.pi * merged['HORA_MEDICION'] / 24.0)
    merged['HORA_COS'] = np.cos(2 * np.pi * merged['HORA_MEDICION'] / 24.0)

    # 9. Time-Series Lag & Rolling Features per Pool
    merged['CLORO_LIBRE_LAG1'] = merged.groupby('PISCINA_CLEAN')['CLORO LIBRE'].shift(1)
    merged['CLORO_LIBRE_LAG2'] = merged.groupby('PISCINA_CLEAN')['CLORO LIBRE'].shift(2)
    merged['PH_LAG1'] = merged.groupby('PISCINA_CLEAN')['PH'].shift(1)
    
    # Days elapsed since previous measurement
    prev_date = merged.groupby('PISCINA_CLEAN')['FECHA_DT'].shift(1)
    merged['DIAS_DESDE_ULTIMA_MEDICION'] = (merged['FECHA_DT'] - prev_date).dt.total_seconds() / (24 * 3600.0)

    # Rolling average chlorine of previous 3 measurements
    merged['CLORO_ROLLING_MEAN_3'] = (
        merged.groupby('PISCINA_CLEAN')['CLORO LIBRE']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    # Lagged chemical dosage
    merged['TOTAL_CLORO_QUIMICO_LAG1'] = merged.groupby('PISCINA_CLEAN')['TOTAL_CLORO_QUIMICO_DOSIS'].shift(1).fillna(0)
    merged['TOTAL_CLORO_QUIMICO_SUM3D'] = (
        merged.groupby('PISCINA_CLEAN')['TOTAL_CLORO_QUIMICO_DOSIS']
        .transform(lambda x: x.rolling(3, min_periods=1).sum())
    )

    # Derived physical ratio: Pool Volume / Surface (Mean Depth indicator)
    if 'Volumen piscina' in merged.columns and 'Superficie piscina' in merged.columns:
        merged['PROFUNDIDAD_MEDIA_EST'] = merged['Volumen piscina'] / merged['Superficie piscina'].replace(0, np.nan)

    # Hypochlorite dosage intensity per m3 volume
    if 'Horas dosificación hipo' in merged.columns and 'Volumen piscina' in merged.columns:
        merged['DOSIS_HIPO_POR_M3'] = (
            merged['Horas dosificación hipo'] * merged['Porcentaje dosificación hipoclorito'].fillna(100) / 100.0
        ) / merged['Volumen piscina'].replace(0, np.nan)

    return merged


def save_processed_data(df: pd.DataFrame, output_dir: str = "data/processed") -> Dict[str, str]:
    """Saves processed dataset to CSV and Parquet formats."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "clean_merged_pool_data.csv")
    parquet_path = os.path.join(output_dir, "clean_merged_pool_data.parquet")

    df_parquet = df.copy()
    if 'FECHA_DATE' in df_parquet.columns:
        df_parquet['FECHA_DATE'] = df_parquet['FECHA_DATE'].astype(str)
        
    df.to_csv(csv_path, index=False)
    try:
        df_parquet.to_parquet(parquet_path, index=False)
    except Exception as e:
        parquet_path = None

    return {"csv": csv_path, "parquet": parquet_path}


if __name__ == "__main__":
    print("Loading raw dataset and building unified pool dataset...")
    df = build_unified_dataset()
    print(f"Dataset successfully built! Shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {df.columns.tolist()[:15]}...")
    print(f"Target 'CLORO LIBRE' non-null count: {df['CLORO LIBRE'].notna().sum()} / {len(df)}")
    saved_files = save_processed_data(df)
    print(f"Saved processed data to: {saved_files}")
