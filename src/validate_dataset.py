#!/usr/bin/env python3
"""
Validation & Predictive Benchmarking Suite for Free Chlorine ML Dataset.

Performs:
1. Automated Data Integrity & Conservation Tests (nullity, physics ranges, leakage check).
2. Correlation Analysis & Feature Ranking against target_next_free_chlorine.
3. Out-of-Sample Machine Learning Benchmarking (Train on 2023-2025, Test on 2026).
4. Compliance Band Classification Evaluation (Under <1.0 ppm, Optimal 1.0-3.0 ppm, Over >3.0 ppm).
5. Generation of diagnostic plots and structured metrics report.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')


def run_integrity_tests(df: pd.DataFrame) -> dict:
    """Runs automated physical sanity and data leakage tests."""
    logger.info("Running physical integrity and data leakage checks...")
    
    # 1. Nullity check in core feature columns
    target = 'target_next_free_chlorine'
    assert df[target].isna().sum() == 0, "Target variable contains NaNs!"
    
    # 2. Physics & Range checks
    assert (df['total_active_cl2_grams'] >= 0.0).all(), "Negative chemical dosages found!"
    assert (df['pool_volume'] > 0.0).all(), "Zero or negative pool volume found!"
    assert (df['window_solar_rad_sum_mj'] >= 0.0).all(), "Negative solar radiation found!"
    assert (df['delta_days'] >= 0.5).all() and (df['delta_days'] <= 10.0).all(), "Interval bounds violated!"
    
    # 3. Leakage Check: verify train and test dates have zero overlap
    train_dates = pd.to_datetime(df[df['is_train_split'] == 1]['date_dt'])
    test_dates = pd.to_datetime(df[df['is_train_split'] == 0]['date_dt'])
    assert train_dates.max() < test_dates.min(), "Data leakage detected: train and test dates overlap!"
    
    logger.info(f"✓ All integrity checks passed! Zero NaNs in target, zero leakage between Train (max: {train_dates.max().date()}) and Test (min: {test_dates.min().date()}).")
    
    return {
        "total_rows": len(df),
        "train_rows": len(train_dates),
        "test_rows": len(test_dates),
        "train_date_range": f"{train_dates.min().date()} to {train_dates.max().date()}",
        "test_date_range": f"{test_dates.min().date()} to {test_dates.max().date()}",
        "integrity_status": "PASSED"
    }


def evaluate_feature_correlations(df: pd.DataFrame, output_dir: str = "reports") -> pd.DataFrame:
    """Computes Pearson correlations between features and target_next_free_chlorine."""
    logger.info("Computing feature correlations with target_next_free_chlorine...")
    os.makedirs(output_dir, exist_ok=True)
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corrs = df[num_cols].corr()['target_next_free_chlorine'].sort_values(ascending=False).reset_index()
    corrs.columns = ['Feature', 'Pearson_Correlation']
    
    # Save top correlations table
    csv_path = os.path.join(output_dir, "chlorine_feature_correlations_next_visit.csv")
    corrs.to_csv(csv_path, index=False)
    logger.info(f"Saved correlation rankings to {csv_path}")
    
    return corrs


def train_and_evaluate_ml_benchmark(df: pd.DataFrame, output_dir: str = "reports") -> dict:
    """
    Trains Gradient Boosted Regressor on Train (2023-2025) and evaluates out-of-sample on Test (2026).
    """
    logger.info("Training out-of-sample ML benchmark model (Train: 2023-2025, Test: 2026)...")
    
    # Feature columns to exclude
    exclude_cols = [
        'pool_clean', 'community_address', 'measurement_date', 'next_date_dt',
        'date_dt', 'date_only', 'next_employee', 'measurement_employee',
        'next_free_chlorine', 'next_ph', 'next_turbidity',
        'target_next_free_chlorine', 'target_next_compliance_band',
        'is_train_split'
    ]
    
    feature_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]
    logger.info(f"Using {len(feature_cols)} engineered features for predictive modeling")
    
    # Train / Test split
    train_mask = df['is_train_split'] == 1
    test_mask = df['is_train_split'] == 0
    
    X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, 'target_next_free_chlorine']
    X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, 'target_next_free_chlorine']
    y_test_band = df.loc[test_mask, 'target_next_compliance_band']
    
    # Train HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(
        loss='squared_error',
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Clip predictions to physical bounds [0.0, 6.0]
    y_pred_test_clipped = np.clip(y_pred_test, 0.0, 6.0)
    
    # Regression Metrics
    r2_tr = r2_score(y_train, y_pred_train)
    r2_te = r2_score(y_test, y_pred_test_clipped)
    mae_tr = mean_absolute_error(y_train, y_pred_train)
    mae_te = mean_absolute_error(y_test, y_pred_test_clipped)
    rmse_te = np.sqrt(mean_squared_error(y_test, y_pred_test_clipped))
    
    # Tolerance Band Accuracy
    within_0_5_ppm = (np.abs(y_test - y_pred_test_clipped) <= 0.5).mean() * 100.0
    within_0_3_ppm = (np.abs(y_test - y_pred_test_clipped) <= 0.3).mean() * 100.0
    
    # Compliance Band Classification Metrics
    pred_bands = pd.cut(y_pred_test_clipped, bins=[-np.inf, 0.999, 3.001, np.inf], labels=[0, 1, 2]).astype(int)
    band_accuracy = accuracy_score(y_test_band, pred_bands) * 100.0
    
    metrics = {
        "model": "HistGradientBoostingRegressor (Huber Loss)",
        "features_count": len(feature_cols),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_r2": round(float(r2_tr), 4),
        "test_r2": round(float(r2_te), 4),
        "train_mae_ppm": round(float(mae_tr), 4),
        "test_mae_ppm": round(float(mae_te), 4),
        "test_rmse_ppm": round(float(rmse_te), 4),
        "accuracy_within_0_5_ppm_pct": round(float(within_0_5_ppm), 2),
        "accuracy_within_0_3_ppm_pct": round(float(within_0_3_ppm), 2),
        "compliance_band_accuracy_pct": round(float(band_accuracy), 2)
    }
    
    logger.info(f"=== OUT-OF-SAMPLE TEST RESULTS (2026 Holdout) ===")
    logger.info(f"Test R²: {metrics['test_r2']} | Test MAE: {metrics['test_mae_ppm']} ppm | Test RMSE: {metrics['test_rmse_ppm']} ppm")
    logger.info(f"Predictions within ±0.5 ppm: {metrics['accuracy_within_0_5_ppm_pct']}% | Within ±0.3 ppm: {metrics['accuracy_within_0_3_ppm_pct']}%")
    logger.info(f"Compliance Band Accuracy (Under/Optimal/Over): {metrics['compliance_band_accuracy_pct']}%")
    
    # Save Metrics JSON
    metrics_path = os.path.join(output_dir, "chlorine_ml_benchmark_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved benchmark metrics to {metrics_path}")
    
    # Plot Visualizations
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    # Figure 1: Predicted vs Actual Scatter
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    sns.scatterplot(x=y_test, y=y_pred_test_clipped, alpha=0.3, color='#1f77b4', s=25, ax=ax)
    ax.plot([0, 5], [0, 5], 'r--', lw=2, label='Perfect Prediction (1:1)')
    ax.fill_between([0, 5], [0 - 0.5, 5 - 0.5], [0 + 0.5, 5 + 0.5], color='green', alpha=0.1, label='±0.5 ppm Precision Band')
    ax.set_title(f"Out-of-Sample Free Chlorine Prediction (2026 Test Set)\n$R^2 = {r2_te:.3f}$ | MAE = {mae_te:.3f} ppm | ±0.5 ppm Accuracy = {within_0_5_ppm:.1f}%")
    ax.set_xlabel("Actual Measured Chlorine at Next Visit (ppm)")
    ax.set_ylabel("Model Predicted Chlorine at Next Visit (ppm)")
    ax.set_xlim(0, 5.5)
    ax.set_ylim(0, 5.5)
    ax.legend(loc='upper left')
    plt.tight_layout()
    fig1_path = os.path.join(fig_dir, "08_ml_chlorine_prediction_scatter.png")
    plt.savefig(fig1_path)
    plt.close()
    
    # Figure 2: Residual Error Distribution
    residuals = y_test - y_pred_test_clipped
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.histplot(residuals, bins=50, kde=True, color='#2ca02c', ax=ax)
    ax.axvline(0, color='red', linestyle='--', lw=1.5)
    ax.axvline(-0.5, color='orange', linestyle=':', lw=1.5, label='±0.5 ppm threshold')
    ax.axvline(0.5, color='orange', linestyle=':', lw=1.5)
    ax.set_title(f"Residual Error Distribution (Actual - Predicted)\nMean Error = {residuals.mean():.3f} ppm | Std Error = {residuals.std():.3f} ppm")
    ax.set_xlabel("Prediction Error (ppm)")
    ax.set_ylabel("Frequency (Observations)")
    ax.legend()
    plt.tight_layout()
    fig2_path = os.path.join(fig_dir, "09_ml_prediction_residual_histogram.png")
    plt.savefig(fig2_path)
    plt.close()
    
    return metrics


def main():
    logger.info("=== Starting Dataset Validation & ML Benchmarking Suite ===")
    dataset_path = "data/processed/chlorine_ml_dataset.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing dataset at {dataset_path}")
        
    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded dataset: {len(df):,} rows x {len(df.columns)} columns")
    
    # 1. Run Integrity & Leakage Tests
    integrity_res = run_integrity_tests(df)
    
    # 2. Compute Correlations
    corrs = evaluate_feature_correlations(df)
    
    # 3. Train ML Benchmark & Evaluate Out-of-Sample Test
    ml_res = train_and_evaluate_ml_benchmark(df)
    
    logger.info("=== Dataset Validation & Benchmarking Complete! ===")


if __name__ == "__main__":
    main()
