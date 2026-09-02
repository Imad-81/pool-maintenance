#!/usr/bin/env python3
"""
High-Precision Free Chlorine Model Training & Optimization Pipeline.

Implements:
1. Multi-Target Formulations (Direct, Physics-Residual, Delta).
2. Modern Gradient Boosted Decision Trees (LightGBM, CatBoost, XGBoost, HistGBM).
3. Optuna Bayesian Hyperparameter Optimization with Time-Series Cross-Validation.
4. Out-of-Fold Stacking Ensemble with Non-Negative Bound Calibration.
5. Strict 2023-2025 Train vs. 2026 Out-of-Sample Test Evaluation.
"""

import os
import sys
import json
import logging
import joblib
import optuna
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Tuple

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(csv_path: str = "data/processed/chlorine_ml_dataset.csv") -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Loads processed dataset and extracts feature column names."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    
    logger.info(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Identify non-feature / leakage columns
    leakage_cols = {
        'pool_clean', 'date_only', 'community_address', 'measurement_date',
        'measurement_employee', 'date_dt', 'next_date_dt', 'next_free_chlorine',
        'next_ph', 'next_turbidity', 'next_employee', 'target_next_free_chlorine',
        'target_next_compliance_band', 'is_train_split', 'is_target_censored_at_5',
        'latent_unlogged_shock_flag'
    }
    
    features = [c for c in df.columns if c not in leakage_cols and np.issubdtype(df[c].dtype, np.number)]
    logger.info(f"Identified {len(features)} numeric modeling features")
    
    train_df = df[df['is_train_split'] == 1].copy().reset_index(drop=True)
    test_df = df[df['is_train_split'] == 0].copy().reset_index(drop=True)
    
    logger.info(f"Train split (2023-2025): {len(train_df):,} samples | Test split (2026): {len(test_df):,} samples")
    return train_df, test_df, features


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "") -> Dict[str, float]:
    """Calculates regression metrics and tolerance accuracy."""
    y_pred_clipped = np.clip(y_pred, 0.0, 5.5)
    mae = float(mean_absolute_error(y_true, y_pred_clipped))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_clipped)))
    r2 = float(r2_score(y_true, y_pred_clipped))
    
    abs_errors = np.abs(y_true - y_pred_clipped)
    acc_025 = float((abs_errors <= 0.25).mean() * 100.0)
    acc_050 = float((abs_errors <= 0.50).mean() * 100.0)
    acc_075 = float((abs_errors <= 0.75).mean() * 100.0)
    
    # Compliance band classification accuracy
    # 0 = Under (<1.0), 1 = Compliant (1.0-3.0), 2 = Over (>3.0)
    band_true = np.digitize(y_true, [1.0, 3.001])
    band_pred = np.digitize(y_pred_clipped, [1.0, 3.001])
    compliance_acc = float((band_true == band_pred).mean() * 100.0)
    
    metrics = {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "Within_0.25ppm_pct": round(acc_025, 2),
        "Within_0.50ppm_pct": round(acc_050, 2),
        "Within_0.75ppm_pct": round(acc_075, 2),
        "Compliance_Band_Acc_pct": round(compliance_acc, 2)
    }
    
    if model_name:
        logger.info(f"[{model_name}] Test MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f} | ±0.50ppm Acc: {acc_050:.1f}%")
        
    return metrics


def train_direct_models(X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, np.ndarray]:
    """Trains baseline and gradient boosted models directly on target chlorine."""
    logger.info("--- Training Direct Target Models ---")
    preds = {}
    
    # 1. LightGBM Direct
    logger.info("Training LightGBM Direct...")
    lgb_direct = lgb.LGBMRegressor(
        objective='regression_l1',
        n_estimators=700,
        learning_rate=0.02,
        num_leaves=45,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    lgb_direct.fit(X_train, y_train)
    preds['LightGBM_Direct'] = lgb_direct.predict(X_test)
    evaluate_predictions(y_test.values, preds['LightGBM_Direct'], "LightGBM Direct")
    
    # 2. CatBoost Direct
    logger.info("Training CatBoost Direct...")
    cb_direct = CatBoostRegressor(
        loss_function='MAE',
        iterations=800,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
        thread_count=-1
    )
    cb_direct.fit(X_train, y_train)
    preds['CatBoost_Direct'] = cb_direct.predict(X_test)
    evaluate_predictions(y_test.values, preds['CatBoost_Direct'], "CatBoost Direct")
    
    # 3. XGBoost Direct
    logger.info("Training XGBoost Direct...")
    xgb_direct = xgb.XGBRegressor(
        objective='reg:absoluteerror',
        n_estimators=700,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    xgb_direct.fit(X_train, y_train)
    preds['XGBoost_Direct'] = xgb_direct.predict(X_test)
    evaluate_predictions(y_test.values, preds['XGBoost_Direct'], "XGBoost Direct")
    
    return preds


def train_physics_residual_models(train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str]) -> Dict[str, np.ndarray]:
    """
    Physics-Guided Residual Modeling:
    Trains models on the residual delta from first-order physical decay baseline:
    Residual = C_actual - C_theoretical_decay
    """
    logger.info("--- Training Physics-Residual Models ---")
    preds = {}
    
    X_train = train_df[features]
    y_train_res = train_df['target_next_free_chlorine'] - train_df['theoretical_retained_chlorine']
    
    X_test = test_df[features]
    y_test = test_df['target_next_free_chlorine']
    c_phys_test = test_df['theoretical_retained_chlorine']
    
    # 1. LightGBM Physics Residual
    lgb_res = lgb.LGBMRegressor(
        objective='regression_l1',
        n_estimators=700,
        learning_rate=0.02,
        num_leaves=40,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    lgb_res.fit(X_train, y_train_res)
    pred_res_lgb = lgb_res.predict(X_test)
    preds['LightGBM_PhysicsResidual'] = np.clip(c_phys_test + pred_res_lgb, 0.0, 5.5)
    evaluate_predictions(y_test.values, preds['LightGBM_PhysicsResidual'], "LightGBM Physics Residual")
    
    # 2. CatBoost Physics Residual
    cb_res = CatBoostRegressor(
        loss_function='MAE',
        iterations=800,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
        thread_count=-1
    )
    cb_res.fit(X_train, y_train_res)
    pred_res_cb = cb_res.predict(X_test)
    preds['CatBoost_PhysicsResidual'] = np.clip(c_phys_test + pred_res_cb, 0.0, 5.5)
    evaluate_predictions(y_test.values, preds['CatBoost_PhysicsResidual'], "CatBoost Physics Residual")
    
    # 3. XGBoost Physics Residual
    xgb_res = xgb.XGBRegressor(
        objective='reg:absoluteerror',
        n_estimators=700,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    xgb_res.fit(X_train, y_train_res)
    pred_res_xgb = xgb_res.predict(X_test)
    preds['XGBoost_PhysicsResidual'] = np.clip(c_phys_test + pred_res_xgb, 0.0, 5.5)
    evaluate_predictions(y_test.values, preds['XGBoost_PhysicsResidual'], "XGBoost Physics Residual")
    
    return preds


def train_delta_models(train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str]) -> Dict[str, np.ndarray]:
    """
    Delta Modeling:
    Trains models on the change: Delta_C = C_next - C_now
    """
    logger.info("--- Training Delta (C_next - C_now) Models ---")
    preds = {}
    
    X_train = train_df[features]
    y_train_delta = train_df['target_next_free_chlorine'] - train_df['free_chlorine']
    
    X_test = test_df[features]
    y_test = test_df['target_next_free_chlorine']
    c_now_test = test_df['free_chlorine']
    
    # 1. LightGBM Delta
    lgb_delta = lgb.LGBMRegressor(
        objective='regression_l1',
        n_estimators=700,
        learning_rate=0.02,
        num_leaves=40,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    lgb_delta.fit(X_train, y_train_delta)
    pred_delta_lgb = lgb_delta.predict(X_test)
    preds['LightGBM_Delta'] = np.clip(c_now_test + pred_delta_lgb, 0.0, 5.5)
    evaluate_predictions(y_test.values, preds['LightGBM_Delta'], "LightGBM Delta")
    
    # 2. CatBoost Delta
    cb_delta = CatBoostRegressor(
        loss_function='MAE',
        iterations=800,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
        thread_count=-1
    )
    cb_delta.fit(X_train, y_train_delta)
    pred_delta_cb = cb_delta.predict(X_test)
    preds['CatBoost_Delta'] = np.clip(c_now_test + pred_delta_cb, 0.0, 5.5)
    evaluate_predictions(y_test.values, preds['CatBoost_Delta'], "CatBoost Delta")
    
    return preds


def run_optuna_tuning(train_df: pd.DataFrame, features: List[str], n_trials: int = 15) -> Dict[str, Any]:
    """Performs fast Bayesian Hyperparameter Optimization with TimeSeriesSplit CV on training data."""
    logger.info(f"--- Running Optuna Hyperparameter Optimization ({n_trials} trials) ---")
    
    # Subsample 15,000 recent rows for ultra-fast tuning
    sub_df = train_df.iloc[-16000:].copy() if len(train_df) > 16000 else train_df.copy()
    X = sub_df[features].values
    y = sub_df['target_next_free_chlorine'].values
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    def objective(trial):
        params = {
            'objective': 'regression_l1',
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.08, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 250, 500, step=50),
            'num_leaves': trial.suggest_int('num_leaves', 25, 55),
            'max_depth': trial.suggest_int('max_depth', 5, 9),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 50),
            'subsample': trial.suggest_float('subsample', 0.70, 0.90),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 0.90),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 5.0, log=True),
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        
        cv_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr)
            val_preds = np.clip(model.predict(X_va), 0.0, 5.5)
            cv_scores.append(mean_absolute_error(y_va, val_preds))
            
        return float(np.mean(cv_scores))
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    logger.info(f"Optuna Best CV MAE: {study.best_value:.4f}")
    logger.info(f"Optuna Best Hyperparameters: {study.best_params}")
    return study.best_params


def build_stacked_ensemble(train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str], best_lgb_params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Builds an Out-of-Fold (OOF) Stacked Ensemble combining:
    1. Tuned LightGBM
    2. Tuned CatBoost
    3. Tuned XGBoost
    4. Physics Residual Model
    5. Delta Model
    Meta-learner: Non-negative Linear Combiner.
    """
    logger.info("--- Building 5-Model Out-of-Fold Stacked Ensemble ---")
    
    tscv = TimeSeriesSplit(n_splits=3)
    X_train = train_df[features].reset_index(drop=True)
    y_train = train_df['target_next_free_chlorine'].reset_index(drop=True)
    c_phys_train = train_df['theoretical_retained_chlorine'].reset_index(drop=True)
    c_now_train = train_df['free_chlorine'].reset_index(drop=True)
    
    X_test = test_df[features].reset_index(drop=True)
    y_test = test_df['target_next_free_chlorine'].reset_index(drop=True)
    c_phys_test = test_df['theoretical_retained_chlorine'].reset_index(drop=True)
    c_now_test = test_df['free_chlorine'].reset_index(drop=True)
    
    model_names = ['LGB_Tuned', 'CatBoost', 'XGBoost', 'LGB_PhysRes', 'LGB_Delta']
    n_models = len(model_names)
    oof_preds = np.zeros((len(train_df), n_models))
    test_base_preds = np.zeros((len(test_df), n_models))
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        logger.info(f"Processing OOF Stacking Fold {fold+1}/3...")
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        # 1. LGB Tuned
        m1 = lgb.LGBMRegressor(objective='regression_l1', random_state=42, verbose=-1, n_jobs=-1, **best_lgb_params)
        m1.fit(X_tr, y_tr)
        oof_preds[val_idx, 0] = m1.predict(X_va)
        
        # 2. CatBoost
        m2 = CatBoostRegressor(loss_function='MAE', iterations=500, learning_rate=0.04, depth=6, random_seed=42, verbose=0, thread_count=-1)
        m2.fit(X_tr, y_tr)
        oof_preds[val_idx, 1] = m2.predict(X_va)
        
        # 3. XGBoost
        m3 = xgb.XGBRegressor(objective='reg:absoluteerror', n_estimators=500, learning_rate=0.03, max_depth=6, random_state=42, n_jobs=-1)
        m3.fit(X_tr, y_tr)
        oof_preds[val_idx, 2] = m3.predict(X_va)
        
        # 4. LGB Physics Residual
        y_tr_res = y_tr - c_phys_train.iloc[train_idx]
        m4 = lgb.LGBMRegressor(objective='regression_l1', n_estimators=450, learning_rate=0.03, num_leaves=35, random_state=42, verbose=-1, n_jobs=-1)
        m4.fit(X_tr, y_tr_res)
        oof_preds[val_idx, 3] = c_phys_train.iloc[val_idx] + m4.predict(X_va)
        
        # 5. LGB Delta
        y_tr_delta = y_tr - c_now_train.iloc[train_idx]
        m5 = lgb.LGBMRegressor(objective='regression_l1', n_estimators=450, learning_rate=0.03, num_leaves=35, random_state=42, verbose=-1, n_jobs=-1)
        m5.fit(X_tr, y_tr_delta)
        oof_preds[val_idx, 4] = c_now_train.iloc[val_idx] + m5.predict(X_va)
        
    # Fit full models on entire train set to predict on 2026 test set
    logger.info("Fitting full models on 100% of training data (2023-2025)...")
    
    # M1 Full
    m1_full = lgb.LGBMRegressor(objective='regression_l1', random_state=42, verbose=-1, n_jobs=-1, **best_lgb_params)
    m1_full.fit(X_train, y_train)
    test_base_preds[:, 0] = m1_full.predict(X_test)
    
    # M2 Full
    m2_full = CatBoostRegressor(loss_function='MAE', iterations=600, learning_rate=0.04, depth=6, random_seed=42, verbose=0, thread_count=-1)
    m2_full.fit(X_train, y_train)
    test_base_preds[:, 1] = m2_full.predict(X_test)
    
    # M3 Full
    m3_full = xgb.XGBRegressor(objective='reg:absoluteerror', n_estimators=600, learning_rate=0.03, max_depth=6, random_state=42, n_jobs=-1)
    m3_full.fit(X_train, y_train)
    test_base_preds[:, 2] = m3_full.predict(X_test)
    
    # M4 Full
    m4_full = lgb.LGBMRegressor(objective='regression_l1', n_estimators=500, learning_rate=0.03, num_leaves=35, random_state=42, verbose=-1, n_jobs=-1)
    m4_full.fit(X_train, y_train - c_phys_train)
    test_base_preds[:, 3] = c_phys_test + m4_full.predict(X_test)
    
    # M5 Full
    m5_full = lgb.LGBMRegressor(objective='regression_l1', n_estimators=500, learning_rate=0.03, num_leaves=35, random_state=42, verbose=-1, n_jobs=-1)
    m5_full.fit(X_train, y_train - c_now_train)
    test_base_preds[:, 4] = c_now_test + m5_full.predict(X_test)
    
    # Meta-learner on valid OOF rows (non-zero)
    valid_oof_mask = (oof_preds.sum(axis=1) > 0)
    meta_X = oof_preds[valid_oof_mask]
    meta_y = y_train[valid_oof_mask]
    
    meta_learner = LinearRegression(positive=True)
    meta_learner.fit(meta_X, meta_y)
    
    # Normalize weights
    raw_coefs = meta_learner.coef_
    if raw_coefs.sum() > 0:
        weights = raw_coefs / raw_coefs.sum()
    else:
        weights = np.ones(n_models) / n_models
        
    logger.info(f"Optimal Meta-Learner Weights: {dict(zip(model_names, [round(float(w), 4) for w in weights]))}")
    
    stacked_test_pred = np.zeros(len(test_df))
    for i, w in enumerate(weights):
        stacked_test_pred += w * test_base_preds[:, i]
        
    stacked_test_pred = np.clip(stacked_test_pred, 0.0, 5.5)
    
    # Save trained models
    os.makedirs("data/models", exist_ok=True)
    joblib.dump(m1_full, "data/models/lightgbm_tuned_model.pkl")
    joblib.dump(m2_full, "data/models/catboost_tuned_model.pkl")
    joblib.dump(m3_full, "data/models/xgboost_tuned_model.pkl")
    joblib.dump(m4_full, "data/models/lightgbm_physics_res_model.pkl")
    
    ensemble_info = {
        "model_names": model_names,
        "weights": dict(zip(model_names, [round(float(w), 4) for w in weights])),
        "intercept": float(meta_learner.intercept_)
    }
    
    return stacked_test_pred, ensemble_info


def main():
    logger.info("=== Starting High-Precision Model Training Pipeline ===")
    
    # 1. Load Data
    train_df, test_df, features = load_dataset()
    X_train = train_df[features]
    y_train = train_df['target_next_free_chlorine']
    X_test = test_df[features]
    y_test = test_df['target_next_free_chlorine']
    
    # 2. Run Direct Models
    direct_preds = train_direct_models(X_train, y_train, X_test, y_test)
    
    # 3. Run Physics-Residual Models
    res_preds = train_physics_residual_models(train_df, test_df, features)
    
    # 4. Run Delta Models
    delta_preds = train_delta_models(train_df, test_df, features)
    
    # 5. Optuna Bayesian Optimization
    best_lgb_params = run_optuna_tuning(train_df, features, n_trials=35)
    
    # 6. Stacked Ensemble
    stacked_pred, ensemble_info = build_stacked_ensemble(train_df, test_df, features, best_lgb_params)
    
    # 7. Collect All Predictions
    all_preds = {
        **direct_preds,
        **res_preds,
        **delta_preds,
        "Stacked_Ensemble_Optimal": stacked_pred
    }
    
    # 8. Evaluate and Benchmark All Models on 2026 Test Set
    logger.info("=== Final Benchmark on 2026 Out-of-Sample Test Set ===")
    metrics_summary = {}
    for name, pred in all_preds.items():
        metrics_summary[name] = evaluate_predictions(y_test.values, pred, name)
        
    # 9. Save Predictions to CSV
    export_pred_df = test_df[['pool_clean', 'date_dt', 'next_date_dt', 'delta_days', 'free_chlorine', 'target_next_free_chlorine', 'target_next_compliance_band']].copy()
    for name, pred in all_preds.items():
        export_pred_df[f"pred_{name}"] = pred
        
    os.makedirs("reports", exist_ok=True)
    pred_path = "reports/optimized_predictions_2026.csv"
    export_pred_df.to_csv(pred_path, index=False)
    logger.info(f"Saved all 2026 model predictions to {pred_path}")
    
    # 10. Save Metrics JSON
    metrics_path = "reports/model_comparison_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            "evaluation_date": datetime.now().isoformat(),
            "test_sample_count": len(test_df),
            "train_sample_count": len(train_df),
            "ensemble_weights": ensemble_info['weights'],
            "optuna_best_params": best_lgb_params,
            "models": metrics_summary
        }, f, indent=2)
    logger.info(f"Saved comparative metrics schema to {metrics_path}")
    
    logger.info("=== Model Optimization Pipeline Finished Successfully ===")


if __name__ == "__main__":
    main()
