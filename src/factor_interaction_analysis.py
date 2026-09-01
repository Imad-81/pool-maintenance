"""
Factor Interaction & Cross-Correlation Analysis Module.
Analyzes how various pool parameters, water quality variables, operational controls,
and chemical applications influence each other, with VIF, Mutual Information, and lag dynamics.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import mutual_info_regression

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import build_unified_dataset

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'


def select_analysis_features(df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    """Selects and cleans relevant continuous and operational features for interaction analysis."""
    candidate_cols = [
        'CLORO LIBRE', 'PH', 'TURBIDEZ', 'Temperatura agua',
        'Volumen piscina', 'Superficie piscina', 'PROFUNDIDAD_MEDIA_EST',
        'Horas filtracion diarias', 'Horas dosificación hipo',
        'Porcentaje dosificación hipoclorito', 'Tiempo lavado /enjuague filtro',
        'TOTAL_CLORO_QUIMICO_DOSIS', 'TOTAL_CLORO_QUIMICO_LAG1', 'TOTAL_CLORO_QUIMICO_SUM3D',
        'CLORO_LIBRE_LAG1', 'PH_LAG1', 'DIAS_DESDE_ULTIMA_MEDICION',
        'HORA_MEDICION', 'MES', 'ES_VERANO', 'ES_FIN_DE_SEMANA',
        'Caudal del motor', 'Diametro filtro'
    ]
    
    available_cols = [c for c in candidate_cols if c in df.columns]
    analysis_df = df[available_cols].copy()
    
    return available_cols, analysis_df


def compute_correlation_matrices(analysis_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Computes both Pearson (linear) and Spearman (rank monotonic) correlation matrices."""
    pearson_corr = analysis_df.corr(method='pearson')
    spearman_corr = analysis_df.corr(method='spearman')
    return pearson_corr, spearman_corr


def compute_vif(analysis_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Computes Variance Inflation Factor (VIF) to detect multicollinearity."""
    # Impute missing values with column median for VIF calculation
    vif_subset = analysis_df[feature_cols].dropna().copy()
    if len(vif_subset) < 100:
        # Fallback to median imputation if dropna loses too many rows
        vif_subset = analysis_df[feature_cols].copy()
        for col in feature_cols:
            vif_subset[col] = vif_subset[col].fillna(vif_subset[col].median())
            
    vif_data = []
    # Drop columns with zero variance
    valid_cols = [c for c in feature_cols if vif_subset[c].std() > 1e-6]
    X = vif_subset[valid_cols].values
    
    for i, col in enumerate(valid_cols):
        try:
            v = variance_inflation_factor(X, i)
            vif_data.append({'Feature': col, 'VIF': round(float(v), 2)})
        except Exception:
            vif_data.append({'Feature': col, 'VIF': np.nan})
            
    vif_df = pd.DataFrame(vif_data).sort_values(by='VIF', ascending=False).reset_index(drop=True)
    return vif_df


def compute_mutual_information(df: pd.DataFrame, target_col: str = 'CLORO LIBRE') -> pd.DataFrame:
    """Computes non-linear Mutual Information between features and target."""
    feature_cols = [
        'PH', 'TURBIDEZ', 'Temperatura agua', 'Volumen piscina', 'Superficie piscina',
        'PROFUNDIDAD_MEDIA_EST', 'Horas filtracion diarias', 'Horas dosificación hipo',
        'Porcentaje dosificación hipoclorito', 'TOTAL_CLORO_QUIMICO_DOSIS',
        'TOTAL_CLORO_QUIMICO_LAG1', 'TOTAL_CLORO_QUIMICO_SUM3D', 'CLORO_LIBRE_LAG1',
        'PH_LAG1', 'DIAS_DESDE_ULTIMA_MEDICION', 'HORA_MEDICION', 'MES', 'ES_VERANO'
    ]
    avail = [c for c in feature_cols if c in df.columns]
    
    # Prepare clean matrix for target
    sub = df[avail + [target_col]].dropna(subset=[target_col]).copy()
    for col in avail:
        sub[col] = sub[col].fillna(sub[col].median())
        
    X = sub[avail]
    y = sub[target_col]
    
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_df = pd.DataFrame({
        'Feature': avail,
        'Mutual_Information': np.round(mi_scores, 4)
    }).sort_values(by='Mutual_Information', ascending=False).reset_index(drop=True)
    
    return mi_df


def plot_factor_interactions(df: pd.DataFrame, pearson_corr: pd.DataFrame, vif_df: pd.DataFrame, mi_df: pd.DataFrame, output_dir: str = "reports/figures"):
    """Generates comprehensive interaction and correlation visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Full Correlation Heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        pearson_corr,
        mask=mask,
        cmap=cmap,
        vmax=0.8,
        vmin=-0.8,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8, "label": "Pearson Correlation (r)"},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7.5},
        ax=ax
    )
    ax.set_title("Cross-Factor Pearson Correlation Matrix", fontsize=14, pad=15)
    plt.tight_layout()
    fig4_path = os.path.join(output_dir, "04_full_correlation_matrix.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()

    # 2. Key Factor Relationships Multi-panel (4 panels)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Panel A: Water Temperature vs pH
    sub_a = df.dropna(subset=['Temperatura agua', 'PH'])
    if len(sub_a) > 0:
        sns.regplot(
            data=sub_a.sample(min(3000, len(sub_a)), random_state=42),
            x='Temperatura agua', y='PH',
            ax=axes[0, 0],
            scatter_kws={'alpha': 0.2, 'color': '#1f77b4', 's': 15},
            line_kws={'color': 'darkred', 'linewidth': 2}
        )
        r_val, p_val = stats.pearsonr(sub_a['Temperatura agua'], sub_a['PH'])
        axes[0, 0].set_title(f"A: Water Temp (°C) vs pH (r = {r_val:.2f})", fontsize=12)
        axes[0, 0].set_xlabel("Water Temperature (°C)")
        axes[0, 0].set_ylabel("pH")

    # Panel B: Turbidity vs Filtration Hours
    sub_b = df.dropna(subset=['Horas filtracion diarias', 'TURBIDEZ'])
    if len(sub_b) > 0:
        sns.boxplot(
            data=sub_b[sub_b['Horas filtracion diarias'].between(1, 16)],
            x='Horas filtracion diarias', y='TURBIDEZ',
            ax=axes[0, 1],
            color='#aec7e8', showfliers=False
        )
        axes[0, 1].set_title("B: Daily Filtration Hours vs Turbidity (NTU)", fontsize=12)
        axes[0, 1].set_xlabel("Daily Filtration Hours (h)")
        axes[0, 1].set_ylabel("Turbidity (NTU)")

    # Panel C: Hypochlorite Dosing Hours vs Free Chlorine
    sub_c = df.dropna(subset=['Horas dosificación hipo', 'CLORO LIBRE'])
    if len(sub_c) > 0:
        sns.boxplot(
            data=sub_c[sub_c['Horas dosificación hipo'].between(1, 14)],
            x='Horas dosificación hipo', y='CLORO LIBRE',
            ax=axes[1, 0],
            color='#2ca02c', showfliers=False
        )
        axes[1, 0].axhline(1.0, color='orange', linestyle='--', label='Min Target (1.0 ppm)')
        axes[1, 0].axhline(3.0, color='red', linestyle='--', label='Max Target (3.0 ppm)')
        axes[1, 0].set_title("C: Hypochlorite Dosing Hours vs Free Chlorine Level", fontsize=12)
        axes[1, 0].set_xlabel("Hypo Dosing Hours (h)")
        axes[1, 0].set_ylabel("Free Chlorine (ppm)")
        axes[1, 0].legend(loc='upper right', fontsize=8)

    # Panel D: Pool Volume vs Total Chlorine Added
    sub_d = df.dropna(subset=['Volumen piscina', 'TOTAL_CLORO_QUIMICO_DOSIS'])
    sub_d = sub_d[sub_d['TOTAL_CLORO_QUIMICO_DOSIS'] > 0]
    if len(sub_d) > 0:
        sns.regplot(
            data=sub_d.sample(min(2000, len(sub_d)), random_state=42),
            x='Volumen piscina', y='TOTAL_CLORO_QUIMICO_DOSIS',
            ax=axes[1, 1],
            scatter_kws={'alpha': 0.3, 'color': '#ff7f0e', 's': 18},
            line_kws={'color': 'black', 'linewidth': 2}
        )
        axes[1, 1].set_title("D: Pool Volume ($m^3$) vs Chemical Chlorine Dosage Quantity", fontsize=12)
        axes[1, 1].set_xlabel("Pool Volume ($m^3$)")
        axes[1, 1].set_ylabel("Chlorine Chemical Dose Quantity")

    plt.tight_layout()
    fig5_path = os.path.join(output_dir, "05_cross_factor_interactions.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()

    # 3. Multicollinearity VIF Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    clean_vif = vif_df.dropna().head(12)
    colors = ['#d62728' if v > 5.0 else '#2ca02c' for v in clean_vif['VIF']]
    
    bars = ax.barh(clean_vif['Feature'], clean_vif['VIF'], color=colors, alpha=0.85, edgecolor='black')
    ax.axvline(5.0, color='darkred', linestyle='--', linewidth=1.5, label='Collinearity Threshold (VIF = 5)')
    ax.axvline(10.0, color='black', linestyle=':', linewidth=1.5, label='Severe Collinearity (VIF = 10)')
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.1, bar.get_y() + bar.get_height()/2, f"{w:.2f}", va='center', ha='left', fontsize=9, fontweight='bold')
        
    ax.set_xlabel("Variance Inflation Factor (VIF)", fontsize=11, fontweight='bold')
    ax.set_title("Multicollinearity Diagnostics (VIF Ranking)", fontsize=13)
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig6_path = os.path.join(output_dir, "06_multicollinearity_vif_bars.png")
    plt.savefig(fig6_path, dpi=300)
    plt.close()

    # 4. Lagged & Rolling Dosage Correlation with Chlorine
    fig, ax = plt.subplots(figsize=(10, 5))
    lag_cols = [
        ('CLORO_LIBRE_LAG1', 'Prior Chlorine (t-1)'),
        ('CLORO_ROLLING_MEAN_3', 'Rolling 3-Measurement Cl Mean'),
        ('TOTAL_CLORO_QUIMICO_DOSIS', 'Chemical Dosed Today (t)'),
        ('TOTAL_CLORO_QUIMICO_LAG1', 'Chemical Dosed Yesterday (t-1)'),
        ('TOTAL_CLORO_QUIMICO_SUM3D', 'Chemical Dosed Past 3 Days'),
        ('Horas dosificación hipo', 'Hypo Dosing Hours (t)'),
        ('Temperatura agua', 'Water Temperature'),
        ('PH', 'Water pH'),
        ('TURBIDEZ', 'Turbidity'),
        ('Volumen piscina', 'Pool Volume ($m^3$)')
    ]
    
    corrs = []
    labels = []
    for col, lbl in lag_cols:
        if col in df.columns:
            valid = df.dropna(subset=[col, 'CLORO LIBRE'])
            if len(valid) > 10:
                r, _ = stats.pearsonr(valid[col], valid['CLORO LIBRE'])
                corrs.append(r)
                labels.append(lbl)
                
    lag_corr_df = pd.DataFrame({'Feature': labels, 'Correlation': corrs}).sort_values(by='Correlation', ascending=True)
    bar_cols = ['#1f77b4' if v >= 0 else '#d62728' for v in lag_corr_df['Correlation']]
    
    bars = ax.barh(lag_corr_df['Feature'], lag_corr_df['Correlation'], color=bar_cols, alpha=0.85, edgecolor='black')
    for bar in bars:
        w = bar.get_width()
        pos = w + (0.01 if w >= 0 else -0.04)
        ax.text(pos, bar.get_y() + bar.get_height()/2, f"{w:+.3f}", va='center', ha='left', fontsize=9, fontweight='bold')
        
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlim(-0.25, 0.75)
    ax.set_xlabel("Pearson Correlation (r) with Target: Free Chlorine (CLORO LIBRE)", fontsize=11, fontweight='bold')
    ax.set_title("Temporal Lag, Operational Controls & Environmental Factors vs Free Chlorine", fontsize=13)
    
    plt.tight_layout()
    fig7_path = os.path.join(output_dir, "07_lagged_dosage_correlation.png")
    plt.savefig(fig7_path, dpi=300)
    plt.close()

    return [fig4_path, fig5_path, fig6_path, fig7_path]


def run_factor_interaction_pipeline(filepath: str = "Merged_2023_2026.xlsx", output_dir: str = "reports") -> Dict[str, Any]:
    """Executes factor interaction, correlation, VIF, and MI analysis."""
    print("="*60)
    print("RUNNING FACTOR INTERACTION & CORRELATION ANALYSIS")
    print("="*60)
    
    df = build_unified_dataset(filepath)
    feature_cols, analysis_df = select_analysis_features(df)
    
    # 1. Correlations
    pearson_corr, spearman_corr = compute_correlation_matrices(analysis_df)
    
    # 2. VIF Multicollinearity
    vif_features = [
        'Volumen piscina', 'Superficie piscina', 'Temperatura agua',
        'Horas filtracion diarias', 'Horas dosificación hipo', 'PH', 'TURBIDEZ'
    ]
    vif_features = [f for f in vif_features if f in df.columns]
    vif_df = compute_vif(df, vif_features)
    
    # 3. Mutual Information with Chlorine
    mi_df = compute_mutual_information(df, target_col='CLORO LIBRE')
    
    # 4. Visualizations
    figures_dir = os.path.join(output_dir, "figures")
    fig_paths = plot_factor_interactions(df, pearson_corr, vif_df, mi_df, output_dir=figures_dir)
    
    # Save CSV outputs
    os.makedirs(output_dir, exist_ok=True)
    pearson_path = os.path.join(output_dir, "pearson_correlation_matrix.csv")
    spearman_path = os.path.join(output_dir, "spearman_correlation_matrix.csv")
    vif_path = os.path.join(output_dir, "vif_multicollinearity.csv")
    mi_path = os.path.join(output_dir, "mutual_information_ranking.csv")
    
    pearson_corr.to_csv(pearson_path)
    spearman_corr.to_csv(spearman_path)
    vif_df.to_csv(vif_path, index=False)
    mi_df.to_csv(mi_path, index=False)
    
    # Key correlations with Chlorine
    cl_pearson = pearson_corr['CLORO LIBRE'].drop('CLORO LIBRE').sort_values(ascending=False).to_dict()
    
    results = {
        'top_chlorine_correlations': {k: round(v, 4) for k, v in cl_pearson.items() if not np.isnan(v)},
        'top_mutual_information': mi_df.head(10).to_dict(orient='records'),
        'vif_multicollinearity': vif_df.to_dict(orient='records'),
        'generated_figures': fig_paths
    }
    
    json_path = os.path.join(output_dir, "factor_interaction_metrics.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
        
    print("Factor interaction analysis completed successfully!")
    print(f"Top 5 Positive Correlates with Free Chlorine:")
    for k, v in list(results['top_chlorine_correlations'].items())[:5]:
        print(f"  + {k}: r = {v:+.4f}")
    print(f"Top 5 Non-linear Drivers (Mutual Information):")
    for row in results['top_mutual_information'][:5]:
        print(f"  * {row['Feature']}: MI = {row['Mutual_Information']:.4f}")
        
    return results


if __name__ == "__main__":
    run_factor_interaction_pipeline()
