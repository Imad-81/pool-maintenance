"""
Missing Data Analysis Module for Pool Dataset.
Analyzes missing data at raw, sub-table, pool-level, and temporal dimensions.
Generates comprehensive statistics and publication-grade visualizations.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import (
    load_raw_data,
    build_unified_dataset,
    extract_static_pool_profile,
    extract_and_align_subtables,
    STATIC_POOL_COLS,
    OPERATIONAL_COLS,
    CHEMICAL_COLS
)

# Set global plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.titleweight'] = 'bold'


def analyze_raw_missingness(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Computes missing value summary for all raw columns in Merged_2023_2026.xlsx."""
    total_rows = len(raw_df)
    missing_counts = raw_df.isnull().sum()
    missing_pct = (missing_counts / total_rows) * 100.0
    
    summary = pd.DataFrame({
        'Column': raw_df.columns,
        'Missing_Count': missing_counts.values,
        'Total_Rows': total_rows,
        'Missing_Percentage': missing_pct.values,
        'Non_Null_Count': (total_rows - missing_counts).values,
        'Dtype': [str(raw_df[c].dtype) for c in raw_df.columns]
    }).sort_values(by='Missing_Percentage', ascending=False).reset_index(drop=True)
    
    # Categorize column role
    def categorize_col(col):
        if col in ['PISCINA', 'COMUNIDAD', 'EMPLEADO', 'EMPLEADO.1', 'EMPLEADO.2']:
            return 'Identifier / Staff'
        elif col in ['FECHA', 'FECHA.1', 'FECHA.2']:
            return 'Timestamp'
        elif col in ['PH', 'TURBIDEZ', 'CLORO LIBRE']:
            return 'Water Quality (Core)'
        elif col in STATIC_POOL_COLS:
            return 'Pool Static Physical'
        elif col in OPERATIONAL_COLS:
            return 'Operational Controls'
        elif col in CHEMICAL_COLS:
            return 'Chemical Additions'
        else:
            return 'Other / Empty'

    summary['Category'] = summary['Column'].apply(categorize_col)
    return summary


def analyze_pool_level_completeness(raw_df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluates how complete static pool attributes are once aggregated per pool."""
    pool_profiles = extract_static_pool_profile(raw_df)
    total_pools = len(pool_profiles)
    
    completeness = {}
    for col in STATIC_POOL_COLS:
        if col in pool_profiles.columns:
            non_null = pool_profiles[col].notna().sum()
            completeness[col] = {
                'pools_with_data': int(non_null),
                'total_pools': int(total_pools),
                'coverage_pct': round((non_null / total_pools) * 100.0, 2)
            }
            
    # Key structural dimensions
    vol_cov = completeness.get('Volumen piscina', {}).get('coverage_pct', 0)
    sup_cov = completeness.get('Superficie piscina', {}).get('coverage_pct', 0)
    filter_cov = completeness.get('Diametro filtro', {}).get('coverage_pct', 0)
    
    return {
        'total_unique_pools': total_pools,
        'volume_coverage_pct': vol_cov,
        'surface_coverage_pct': sup_cov,
        'filter_coverage_pct': filter_cov,
        'detailed_completeness': completeness
    }


def analyze_temporal_missingness(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyzes measurement coverage across years, months, and sampling intervals."""
    df_valid = df.dropna(subset=['FECHA_DT']).copy()
    
    # Yearly counts
    yearly_counts = df_valid['ANIO'].value_counts().sort_index().to_dict()
    
    # Monthly density (Year x Month)
    monthly_table = pd.crosstab(df_valid['ANIO'], df_valid['MES'])
    
    # Sampling gap analysis per pool
    gaps = df_valid.groupby('PISCINA_CLEAN')['DIAS_DESDE_ULTIMA_MEDICION'].agg(['median', 'mean', 'max', 'count'])
    
    return {
        'total_measurements': int(len(df_valid)),
        'date_min': str(df_valid['FECHA_DT'].min()),
        'date_max': str(df_valid['FECHA_DT'].max()),
        'yearly_distribution': yearly_counts,
        'median_days_between_measurements': round(float(gaps['median'].median()), 2),
        'mean_days_between_measurements': round(float(gaps['mean'].mean()), 2),
        'monthly_cross_table': monthly_table.to_dict()
    }


def plot_missing_analysis(raw_summary: pd.DataFrame, df: pd.DataFrame, output_dir: str = "reports/figures"):
    """Generates visual figures for missing data analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Figure 1: Missing Percentage by Key Column Groups
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Filter out completely empty unnamed columns for cleaner visual
    plot_df = raw_summary[~raw_summary['Column'].str.contains('Unnamed')].copy()
    plot_df = plot_df.sort_values(by='Missing_Percentage', ascending=True)
    
    colors = {
        'Water Quality (Core)': '#1f77b4',
        'Pool Static Physical': '#2ca02c',
        'Operational Controls': '#ff7f0e',
        'Chemical Additions': '#d62728',
        'Timestamp': '#9467bd',
        'Identifier / Staff': '#8c564b',
        'Other / Empty': '#7f7f7f'
    }
    bar_colors = [colors.get(cat, '#333333') for cat in plot_df['Category']]
    
    bars = ax.barh(plot_df['Column'], plot_df['Missing_Percentage'], color=bar_colors, alpha=0.85, edgecolor='none')
    
    # Annotate bars
    for bar in bars:
        w = bar.get_width()
        if w > 0:
            ax.text(w + 1, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", 
                    va='center', ha='left', fontsize=8, color='#333333')
            
    ax.set_xlim(0, 115)
    ax.set_xlabel("Missing Percentage (%)", fontsize=11, fontweight='bold')
    ax.set_title("Missing Data Percentage by Column (Raw Merged File)\n*Note: Static attributes & logs appear high due to horizontal pasting structure", fontsize=13)
    
    # Custom Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=k) for k, c in colors.items() if k in plot_df['Category'].values]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "01_missing_percentage_by_column.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    # Figure 2: Temporal Logging Density Heatmap (Month x Year)
    fig, ax = plt.subplots(figsize=(10, 6))
    df_valid = df.dropna(subset=['FECHA_DT']).copy()
    density_matrix = pd.crosstab(df_valid['ANIO'], df_valid['MES'])
    
    # Rename months to Spanish/English abbreviations
    month_names = ['Ene/Jan', 'Feb', 'Mar', 'Abr/Apr', 'May', 'Jun', 'Jul', 'Ago/Aug', 'Sep', 'Oct', 'Nov', 'Dic/Dec']
    density_matrix.columns = [month_names[m-1] for m in density_matrix.columns]
    
    sns.heatmap(density_matrix, cmap="YlGnBu", annot=True, fmt="d", cbar_kws={'label': 'Measurement Count'}, ax=ax)
    ax.set_title("Temporal Measurement Logging Density (2023 - 2026)\nClear Summer Seasonality vs Winter Monitoring", fontsize=13)
    ax.set_xlabel("Month of Year", fontsize=11, fontweight='bold')
    ax.set_ylabel("Year", fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "02_temporal_data_density_heatmap.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()

    # Figure 3: Pool Logging Completeness Distribution & Target Availability
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pool measurement counts histogram
    pool_counts = df.groupby('PISCINA_CLEAN')['CLORO LIBRE'].count()
    axes[0].hist(pool_counts, bins=25, color='#1f77b4', edgecolor='black', alpha=0.7)
    axes[0].axvline(pool_counts.median(), color='red', linestyle='--', linewidth=2, label=f"Median: {pool_counts.median():.0f} logs/pool")
    axes[0].set_xlabel("Number of Measurements per Pool", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Number of Pools", fontsize=11, fontweight='bold')
    axes[0].set_title("Distribution of Measurement Logs Across Pools", fontsize=12)
    axes[0].legend()

    # Core Target & Features Missingness in Clean Merged Dataset
    key_features = [
        'CLORO LIBRE', 'PH', 'TURBIDEZ', 'Volumen piscina', 'Superficie piscina',
        'Temperatura agua', 'Horas filtracion diarias', 'Horas dosificación hipo',
        'TOTAL_CLORO_QUIMICO_DOSIS', 'CLORO_LIBRE_LAG1'
    ]
    present_keys = [k for k in key_features if k in df.columns]
    non_null_pct = (df[present_keys].notna().mean() * 100.0).sort_values()
    
    axes[1].barh(non_null_pct.index, non_null_pct.values, color='#2ca02c', alpha=0.8, edgecolor='black')
    for idx, val in enumerate(non_null_pct.values):
        axes[1].text(val + 1, idx, f"{val:.1f}%", va='center', ha='left', fontsize=9, fontweight='bold')
    axes[1].set_xlim(0, 115)
    axes[1].set_xlabel("Valid Data Availability (%)", fontsize=11, fontweight='bold')
    axes[1].set_title("Data Availability for Modeling in Unified Dataset", fontsize=12)
    
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "03_pool_logging_completeness.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()

    return [fig1_path, fig2_path, fig3_path]


def run_missing_data_pipeline(raw_filepath: str = "Merged_2023_2026.xlsx", output_dir: str = "reports") -> Dict[str, Any]:
    """Executes end-to-end missing data analysis."""
    print("="*60)
    print("RUNNING MISSING DATA ANALYSIS")
    print("="*60)
    
    raw_df = load_raw_data(raw_filepath)
    unified_df = build_unified_dataset(raw_filepath)
    
    raw_summary = analyze_raw_missingness(raw_df)
    pool_completeness = analyze_pool_level_completeness(raw_df)
    temporal_summary = analyze_temporal_missingness(unified_df)
    
    figures_dir = os.path.join(output_dir, "figures")
    fig_paths = plot_missing_analysis(raw_summary, unified_df, output_dir=figures_dir)
    
    # Save missing summary to CSV
    os.makedirs(output_dir, exist_ok=True)
    raw_summary_path = os.path.join(output_dir, "missing_data_summary.csv")
    raw_summary.to_csv(raw_summary_path, index=False)
    
    results = {
        'raw_total_rows': len(raw_df),
        'raw_total_cols': len(raw_df.columns),
        'unified_total_rows': len(unified_df),
        'target_valid_rows': int(unified_df['CLORO LIBRE'].notna().sum()),
        'target_missing_rows': int(unified_df['CLORO LIBRE'].isna().sum()),
        'target_missing_pct': round(float(unified_df['CLORO LIBRE'].isna().mean() * 100), 2),
        'pool_completeness': pool_completeness,
        'temporal_summary': temporal_summary,
        'generated_figures': fig_paths,
        'summary_csv': raw_summary_path
    }
    
    # Save results JSON
    json_path = os.path.join(output_dir, "missing_analysis_metrics.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
        
    print(f"Missing data analysis completed! Summary saved to: {raw_summary_path}")
    print(f"Metrics JSON saved to: {json_path}")
    print(f"Key Findings:")
    print(f" - Target 'CLORO LIBRE' Availability: {100 - results['target_missing_pct']:.2f}% ({results['target_valid_rows']} valid records)")
    print(f" - Total Unique Pools: {pool_completeness['total_unique_pools']}")
    print(f" - Pool Volume Coverage (propagated): {pool_completeness['volume_coverage_pct']}%")
    print(f" - Pool Surface Area Coverage (propagated): {pool_completeness['surface_coverage_pct']}%")
    
    return results


if __name__ == "__main__":
    run_missing_data_pipeline()
