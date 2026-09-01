"""
Chlorine Target Predictive Analysis & Machine Learning Driver Modeling.
Performs in-depth target profiling, multi-algorithm feature importance ranking,
chlorine decay and response curve dynamics, and time-series baseline predictive modeling.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

from src.data_loader import build_unified_dataset

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'


def profile_chlorine_target(df: pd.DataFrame) -> Dict[str, Any]:
    """Profiles the distribution and regulatory compliance bands of CLORO LIBRE."""
    cl = df['CLORO LIBRE'].dropna()
    
    total = len(cl)
    under = (cl < 1.0).sum()
    optimal = ((cl >= 1.0) & (cl <= 3.0)).sum()
    over = (cl > 3.0).sum()
    extreme = (cl >= 4.5).sum()
    
    stats_dict = {
        'count': int(total),
        'mean': round(float(cl.mean()), 3),
        'std': round(float(cl.std()), 3),
        'median': round(float(cl.median()), 3),
        'min': round(float(cl.min()), 3),
        'max': round(float(cl.max()), 3),
        'q25': round(float(cl.quantile(0.25)), 3),
        'q75': round(float(cl.quantile(0.75)), 3),
        'skewness': round(float(cl.skew()), 3),
        'kurtosis': round(float(cl.kurtosis()), 3),
        'compliance': {
            'under_target_lt_1ppm': {'count': int(under), 'pct': round(float(under/total)*100, 2)},
            'optimal_range_1_to_3ppm': {'count': int(optimal), 'pct': round(float(optimal/total)*100, 2)},
            'over_target_gt_3ppm': {'count': int(over), 'pct': round(float(over/total)*100, 2)},
            'extreme_high_gte_4_5ppm': {'count': int(extreme), 'pct': round(float(extreme/total)*100, 2)}
        }
    }
    return stats_dict


def prepare_predictive_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Prepares features and target for predictive modeling."""
    feature_candidates = [
        # Lagged water quality
        'CLORO_LIBRE_LAG1', 'CLORO_LIBRE_LAG2', 'CLORO_ROLLING_MEAN_3', 'PH_LAG1',
        # Current water quality & environmental
        'PH', 'TURBIDEZ', 'Temperatura agua', 'DIAS_DESDE_ULTIMA_MEDICION',
        # Operations & Controls
        'Horas dosificación hipo', 'Porcentaje dosificación hipoclorito',
        'Horas filtracion diarias', 'Tiempo lavado /enjuague filtro',
        'TOTAL_CLORO_QUIMICO_DOSIS', 'TOTAL_CLORO_QUIMICO_LAG1', 'TOTAL_CLORO_QUIMICO_SUM3D',
        # Pool Physical Properties
        'Volumen piscina', 'Superficie piscina', 'PROFUNDIDAD_MEDIA_EST',
        'Caudal del motor', 'Diametro filtro', 'Numero de filtros', 'Número de motores',
        'PISCINA EXTERIOR', 'Piscina con skimmers', 'Piscina desbordante',
        # Temporal & Seasonality
        'MES', 'MES_SIN', 'MES_COS', 'HORA_MEDICION', 'HORA_SIN', 'HORA_COS',
        'DIA_SEMANA', 'ES_VERANO', 'ES_FIN_DE_SEMANA', 'ANIO'
    ]
    
    avail_features = [f for f in feature_candidates if f in df.columns]
    
    # Filter valid target
    valid_df = df.dropna(subset=['CLORO LIBRE']).copy()
    valid_df = valid_df.sort_values(by=['FECHA_DT']).reset_index(drop=True)
    
    X = valid_df[avail_features].copy()
    y = valid_df['CLORO LIBRE'].copy()
    
    return X, y, avail_features


def train_and_rank_feature_importances(X: pd.DataFrame, y: pd.Series, feature_names: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Trains multiple models (RandomForest, HistGradientBoosting, Ridge) to extract consensus feature rankings.
    """
    # Impute missing values with median for models that require dense data (RF, Ridge)
    X_imputed = X.fillna(X.median())
    
    # Standardize for Linear Models
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # 1. Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X_imputed, y)
    rf_importances = rf.feature_importances_
    
    # 2. HistGradientBoosting (handles missing values directly)
    hgb = HistGradientBoostingRegressor(max_iter=100, max_depth=8, min_samples_leaf=10, random_state=42)
    hgb.fit(X, y)
    # Permutation importance for HGB
    perm = permutation_importance(hgb, X, y, n_repeats=5, random_state=42, n_jobs=-1)
    hgb_importances = perm.importances_mean
    hgb_importances = np.maximum(0, hgb_importances)
    hgb_importances = hgb_importances / (hgb_importances.sum() + 1e-9)

    # 3. Ridge Regression (Standardized Coefficients)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)
    ridge_coefs = ridge.coef_
    
    # Combine into DataFrame
    rank_df = pd.DataFrame({
        'Feature': feature_names,
        'RF_Importance': rf_importances,
        'HGB_Importance': hgb_importances,
        'Ridge_Coefficient': ridge_coefs,
        'Abs_Ridge_Coef': np.abs(ridge_coefs)
    })
    
    # Normalized consensus score
    rank_df['Consensus_Score'] = (
        (rank_df['RF_Importance'] / rank_df['RF_Importance'].max()) +
        (rank_df['HGB_Importance'] / (rank_df['HGB_Importance'].max() + 1e-9)) +
        (rank_df['Abs_Ridge_Coef'] / rank_df['Abs_Ridge_Coef'].max())
    ) / 3.0
    
    rank_df = rank_df.sort_values(by='Consensus_Score', ascending=False).reset_index(drop=True)
    
    return rank_df, {'rf_model': rf, 'hgb_model': hgb, 'ridge_model': ridge}


def evaluate_baseline_time_series_models(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Evaluates baseline predictive accuracy using 5-fold TimeSeriesSplit CV and feature group ablations.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Define 3 feature sets:
    # Set A: Environmental & Static physical properties only (No lag, no dosing)
    set_a = [c for c in ['Volumen piscina', 'Superficie piscina', 'Temperatura agua', 'MES', 'HORA_MEDICION', 'ES_VERANO'] if c in X.columns]
    
    # Set B: Environmental + Physical + Operational Controls (Dosing hours, Filtration, Chemical doses)
    set_b = set_a + [c for c in ['Horas dosificación hipo', 'Porcentaje dosificación hipoclorito', 'Horas filtracion diarias', 'TOTAL_CLORO_QUIMICO_DOSIS', 'PH', 'TURBIDEZ'] if c in X.columns]
    
    # Set C: Full Model (including Lag1, Rolling mean, Lagged chemicals)
    set_c = list(X.columns)
    
    experiments = {
        'Model_A_Static_Environmental': set_a,
        'Model_B_Operations_and_Dosing': set_b,
        'Model_C_Full_Dynamic_with_Lags': set_c
    }
    
    results = {}
    for exp_name, feat_subset in experiments.items():
        sub_X = X[feat_subset]
        
        rmse_list = []
        mae_list = []
        r2_list = []
        acc_half_ppm_list = []
        
        for train_idx, test_idx in tscv.split(sub_X):
            X_train, X_test = sub_X.iloc[train_idx], sub_X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model = HistGradientBoostingRegressor(max_iter=100, max_depth=7, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            acc_half = np.mean(np.abs(y_test - preds) <= 0.5) * 100.0
            
            rmse_list.append(rmse)
            mae_list.append(mae)
            r2_list.append(r2)
            acc_half_ppm_list.append(acc_half)
            
        results[exp_name] = {
            'RMSE_mean': round(float(np.mean(rmse_list)), 3),
            'RMSE_std': round(float(np.std(rmse_list)), 3),
            'MAE_mean': round(float(np.mean(mae_list)), 3),
            'MAE_std': round(float(np.std(mae_list)), 3),
            'R2_mean': round(float(np.mean(r2_list)), 3),
            'Accuracy_within_0_5ppm_pct': round(float(np.mean(acc_half_ppm_list)), 2),
            'features_count': len(feat_subset)
        }
        
    return results


def plot_predictive_analysis(df: pd.DataFrame, stats_dict: Dict[str, Any], rank_df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, output_dir: str = "reports/figures"):
    """Generates all visualizations for chlorine target profiling and prediction."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Chlorine Distribution & Compliance Bands
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    cl = df['CLORO LIBRE'].dropna()
    sns.histplot(cl, bins=35, kde=True, ax=axes[0], color='#1f77b4', edgecolor='black', alpha=0.6)
    axes[0].axvline(1.0, color='orange', linestyle='--', linewidth=2, label='Min Sanitization Standard (1.0 ppm)')
    axes[0].axvline(3.0, color='red', linestyle='--', linewidth=2, label='Max Sanitization Standard (3.0 ppm)')
    axes[0].axvspan(1.0, 3.0, color='green', alpha=0.15, label='Safe Target Zone (1.0 - 3.0 ppm)')
    axes[0].set_title(f"Free Chlorine Distribution (Mean: {cl.mean():.2f} ± {cl.std():.2f} ppm)", fontsize=12)
    axes[0].set_xlabel("Free Chlorine - CLORO LIBRE (ppm)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Count of Records", fontsize=11, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=9)
    
    # Compliance Pie / Donut
    comp = stats_dict['compliance']
    labels = [
        f"Under Target (<1 ppm)\n{comp['under_target_lt_1ppm']['pct']}%",
        f"Optimal Zone (1-3 ppm)\n{comp['optimal_range_1_to_3ppm']['pct']}%",
        f"Over Target (>3 ppm)\n{comp['over_target_gt_3ppm']['pct']}%"
    ]
    sizes = [
        comp['under_target_lt_1ppm']['count'],
        comp['optimal_range_1_to_3ppm']['count'],
        comp['over_target_gt_3ppm']['count']
    ]
    colors = ['#ff9999', '#66b3ff', '#ffcc99']
    
    axes[1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, explode=(0.05, 0.05, 0.05),
                wedgeprops=dict(width=0.4, edgecolor='w', linewidth=2), textprops={'fontsize': 10, 'fontweight': 'bold'})
    axes[1].set_title("Pool Disinfection Regulatory Compliance Breakdown", fontsize=12)
    
    plt.tight_layout()
    fig8_path = os.path.join(output_dir, "08_chlorine_distribution_and_ranges.png")
    plt.savefig(fig8_path, dpi=300)
    plt.close()

    # 2. Multi-Model Feature Importance Comparison
    fig, ax = plt.subplots(figsize=(12, 7))
    top_ranks = rank_df.head(12).sort_values(by='Consensus_Score', ascending=True)
    
    y_pos = np.arange(len(top_ranks))
    height = 0.35
    
    ax.barh(y_pos - height/2, top_ranks['RF_Importance'], height=height, label='Random Forest (MDI)', color='#1f77b4', alpha=0.85)
    ax.barh(y_pos + height/2, top_ranks['HGB_Importance'], height=height, label='Gradient Boosting (Permutation)', color='#ff7f0e', alpha=0.85)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_ranks['Feature'], fontsize=10, fontweight='bold')
    ax.set_xlabel("Normalized Feature Importance Score", fontsize=11, fontweight='bold')
    ax.set_title("Top Predictive Drivers of Free Chlorine (Multi-Model Comparison)", fontsize=13)
    ax.legend(loc='lower right', frameon=True, facecolor='white')
    
    plt.tight_layout()
    fig9_path = os.path.join(output_dir, "09_chlorine_feature_importance_comparison.png")
    plt.savefig(fig9_path, dpi=300)
    plt.close()

    # 3. Chlorine vs Key Drivers (4 panels)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Panel A: Chlorine vs Prior Lag (Autoregressive persistence)
    sub1 = df.dropna(subset=['CLORO_LIBRE_LAG1', 'CLORO LIBRE'])
    if len(sub1) > 0:
        sns.regplot(
            data=sub1.sample(min(3000, len(sub1)), random_state=42),
            x='CLORO_LIBRE_LAG1', y='CLORO LIBRE',
            ax=axes[0, 0],
            scatter_kws={'alpha': 0.25, 'color': '#1f77b4', 's': 15},
            line_kws={'color': 'black', 'linewidth': 2}
        )
        axes[0, 0].set_title("A: Persistence: Prior Chlorine ($t-1$) vs Current ($t$)", fontsize=12)
        axes[0, 0].set_xlabel("Prior Chlorine Level (ppm)")
        axes[0, 0].set_ylabel("Current Chlorine Level (ppm)")

    # Panel B: Chlorine by Month / Summer Seasonality
    sns.boxplot(data=df.dropna(subset=['MES', 'CLORO LIBRE']), x='MES', y='CLORO LIBRE', hue='MES', ax=axes[0, 1], palette='Spectral', legend=False, showfliers=False)
    axes[0, 1].set_title("B: Seasonal Dynamics: Free Chlorine across Months", fontsize=12)
    axes[0, 1].set_xlabel("Month")
    axes[0, 1].set_ylabel("Free Chlorine (ppm)")
    axes[0, 1].axhline(1.0, color='orange', linestyle='--')
    axes[0, 1].axhline(3.0, color='red', linestyle='--')

    # Panel C: Chlorine vs Water Temperature
    sub3 = df.dropna(subset=['Temperatura agua', 'CLORO LIBRE'])
    if len(sub3) > 0:
        sns.regplot(
            data=sub3.sample(min(3000, len(sub3)), random_state=42),
            x='Temperatura agua', y='CLORO LIBRE',
            ax=axes[1, 0],
            scatter_kws={'alpha': 0.25, 'color': '#d62728', 's': 15},
            line_kws={'color': 'navy', 'linewidth': 2}
        )
        axes[1, 0].set_title("C: Water Temperature vs Free Chlorine Depletion", fontsize=12)
        axes[1, 0].set_xlabel("Water Temperature (°C)")
        axes[1, 0].set_ylabel("Free Chlorine (ppm)")

    # Panel D: Chlorine vs Hypo Dosing Intensity
    sub4 = df.dropna(subset=['Porcentaje dosificación hipoclorito', 'CLORO LIBRE'])
    if len(sub4) > 0:
        sns.regplot(
            data=sub4.sample(min(3000, len(sub4)), random_state=42),
            x='Porcentaje dosificación hipoclorito', y='CLORO LIBRE',
            ax=axes[1, 1],
            scatter_kws={'alpha': 0.25, 'color': '#2ca02c', 's': 15},
            line_kws={'color': 'darkgreen', 'linewidth': 2}
        )
        axes[1, 1].set_title("D: Hypochlorite Pump Dosing % vs Free Chlorine", fontsize=12)
        axes[1, 1].set_xlabel("Hypo Pump Dosing Rate (%)")
        axes[1, 1].set_ylabel("Free Chlorine (ppm)")

    plt.tight_layout()
    fig10_path = os.path.join(output_dir, "10_chlorine_vs_key_drivers_regression.png")
    plt.savefig(fig10_path, dpi=300)
    plt.close()

    # 4. Model Prediction Residuals & Fit
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Train full gradient boosting model for holdout visual
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    model = HistGradientBoostingRegressor(max_iter=150, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    
    # Actual vs Predicted
    axes[0].scatter(y_test, y_pred, alpha=0.3, color='#1f77b4', s=16)
    axes[0].plot([0, 5], [0, 5], color='red', linestyle='--', linewidth=2, label='Ideal Perfect Fit (y = x)')
    axes[0].set_xlabel("Actual Free Chlorine (ppm)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Predicted Free Chlorine (ppm)", fontsize=11, fontweight='bold')
    r2_val = r2_score(y_test, y_pred)
    rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))
    axes[0].set_title(f"Test Set: Actual vs Predicted Chlorine ($R^2 = {r2_val:.3f}$, RMSE = {rmse_val:.3f})", fontsize=12)
    axes[0].legend()

    # Residuals distribution
    sns.histplot(residuals, bins=35, kde=True, ax=axes[1], color='#2ca02c', edgecolor='black', alpha=0.6)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1].axvline(0.5, color='orange', linestyle=':', label='±0.5 ppm Error Margin')
    axes[1].axvline(-0.5, color='orange', linestyle=':')
    axes[1].set_xlabel("Prediction Error / Residual (ppm)", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Count", fontsize=11, fontweight='bold')
    axes[1].set_title("Model Residual Error Distribution (Zero-Centered)", fontsize=12)
    axes[1].legend()

    plt.tight_layout()
    fig11_path = os.path.join(output_dir, "11_chlorine_model_prediction_residuals.png")
    plt.savefig(fig11_path, dpi=300)
    plt.close()

    return [fig8_path, fig9_path, fig10_path, fig11_path]


def run_chlorine_predictive_pipeline(filepath: str = "Merged_2023_2026.xlsx", output_dir: str = "reports") -> Dict[str, Any]:
    """Runs complete chlorine predictive analysis and driver evaluation."""
    print("="*60)
    print("RUNNING CHLORINE PREDICTIVE DRIVER & ML ANALYSIS")
    print("="*60)
    
    df = build_unified_dataset(filepath)
    
    # 1. Target Profiling
    stats_dict = profile_chlorine_target(df)
    
    # 2. Predictive Dataset Preparation
    X, y, feature_names = prepare_predictive_dataset(df)
    
    # 3. Feature Importance Consensus Ranking
    rank_df, _ = train_and_rank_feature_importances(X, y, feature_names)
    
    # 4. Baseline Time-Series Evaluation
    cv_results = evaluate_baseline_time_series_models(X, y)
    
    # 5. Visualizations
    figures_dir = os.path.join(output_dir, "figures")
    fig_paths = plot_predictive_analysis(df, stats_dict, rank_df, X, y, output_dir=figures_dir)
    
    # Save CSV outputs
    os.makedirs(output_dir, exist_ok=True)
    rank_path = os.path.join(output_dir, "chlorine_feature_importance_ranking.csv")
    rank_df.to_csv(rank_path, index=False)
    
    results = {
        'target_summary': stats_dict,
        'top_predictive_features': rank_df.head(15).to_dict(orient='records'),
        'baseline_model_performance': cv_results,
        'generated_figures': fig_paths,
        'ranking_csv': rank_path
    }
    
    json_path = os.path.join(output_dir, "chlorine_predictive_metrics.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
        
    print("Chlorine predictive analysis completed successfully!")
    print(f"Top 5 Consensus Predictive Features for Free Chlorine:")
    for row in results['top_predictive_features'][:5]:
        print(f"  1. {row['Feature']} (Score: {row['Consensus_Score']:.3f}, RF: {row['RF_Importance']:.3f}, HGB: {row['HGB_Importance']:.3f})")
    print(f"\nModel Performance Benchmarks:")
    for m_name, m_res in cv_results.items():
        print(f"  * {m_name}: R² = {m_res['R2_mean']:.3f}, RMSE = {m_res['RMSE_mean']:.3f}, Accuracy within ±0.5ppm = {m_res['Accuracy_within_0_5ppm_pct']}%")
        
    return results


if __name__ == "__main__":
    run_chlorine_predictive_pipeline()
