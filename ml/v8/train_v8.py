"""
V8 Sparse-Visit Training Pipeline.

Usage:
    python -m ml.v8.train_v8                        # full run
    python -m ml.v8.train_v8 --dry-run             # validate data only
    python -m ml.v8.train_v8 --run-id custom-v8-id

The pipeline is intentionally isolated from ml.training — it imports only
ml.v8.* modules so the existing v6/v7 code paths are never touched.

Output artifacts (in models/v8-sparse-visit/):
    chlorine_next_v8.pkl
    ph_next_v8.pkl
    turbidity_next_v8.pkl
    inference_config_v8.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ml.v8.config_v8 import V8Config, DEFAULT_V8_CONFIG, REG_CHLORINE_MIN, REG_PH_MIN, REG_PH_MAX
from ml.v8.sparse_dataset import (
    load_sparse_visits,
    add_gap_columns,
    generate_gap_report,
)
from ml.v8.features_v8 import (
    build_features,
    all_feature_groups,
    compute_fill_values,
    apply_fill_values,
)

log = logging.getLogger("ml.v8.train_v8")


# ---------------------------------------------------------------------------
# STEP T1 — Build targets
# ---------------------------------------------------------------------------

def build_targets(df: pd.DataFrame, cfg: V8Config) -> pd.DataFrame:
    """
    Add target columns:
        target_cl_next_visit   — cl_post at the NEXT observed visit
        target_ph_next_visit   — ph_post at the NEXT observed visit
        target_turb_next_visit — turb_post at the NEXT observed visit

    Rows where gap_to > max_allowed_gap_for_target are excluded from training
    (the prediction horizon is too uncertain). Rows with no next visit (last
    visit per pool) are also excluded.
    """
    df = df.sort_values(["pool_clean", "date"]).copy()

    for col_post, target in [
        ("cl_post",   "target_cl_next_visit"),
        ("ph_post",   "target_ph_next_visit"),
        ("turb_post", "target_turb_next_visit"),
    ]:
        if col_post in df.columns:
            df[target] = df.groupby("pool_clean")[col_post].shift(-1)
        else:
            df[target] = np.nan

    # Also record breach status at next visit (useful for evaluation)
    if "target_cl_next_visit" in df.columns:
        df["target_cl_breach"] = df["target_cl_next_visit"] < REG_CHLORINE_MIN
    if "target_ph_next_visit" in df.columns:
        df["target_ph_breach"] = (
            (df["target_ph_next_visit"] < REG_PH_MIN) |
            (df["target_ph_next_visit"] > REG_PH_MAX)
        )

    before = len(df)
    # Drop last visit per pool (no next visit → target is NaN)
    df = df.dropna(subset=["target_cl_next_visit", "target_ph_next_visit"]).copy()
    # Drop rows with excessively long gap to next visit
    if "gap_to" in df.columns:
        df = df[
            df["gap_to"].isna() | (df["gap_to"] <= cfg.max_allowed_gap_for_target)
        ].copy()

    after = len(df)
    log.info(
        "build_targets: %d → %d rows (%d dropped: last visits or gap_to>%d)",
        before, after, before - after, cfg.max_allowed_gap_for_target,
    )
    return df


# ---------------------------------------------------------------------------
# STEP T2 — Temporal split
# ---------------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame,
    cfg: V8Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split on the temporal_split_quantile date cutoff.
    Returns (train, test).
    """
    cutoff = df["date"].quantile(cfg.temporal_split_quantile)
    train = df[df["date"] < cutoff].copy()
    test  = df[df["date"] >= cutoff].copy()
    log.info(
        "temporal_split: cutoff=%s  train=%d  test=%d",
        str(cutoff.date()), len(train), len(test),
    )
    return train, test


# ---------------------------------------------------------------------------
# STEP T3 — Prepare matrices
# ---------------------------------------------------------------------------

def prepare_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cfg: V8Config,
) -> Dict:
    """
    Fill NaN, assemble X_train / X_test.
    Returns dict with matrices, fill_values, feature_names.
    """
    feat_cols = all_feature_groups()
    # Only keep features that are actually present
    feat_cols = [c for c in feat_cols if c in train.columns]

    # Drop features with >50% null in train
    null_rates = train[feat_cols].isnull().mean()
    high_null = null_rates[null_rates > cfg.feature_null_drop_threshold].index.tolist()
    if high_null:
        log.info("Dropping high-null features (>50%%): %s", high_null)
        feat_cols = [c for c in feat_cols if c not in high_null]

    fill_values = compute_fill_values(train[feat_cols])

    train_filled = apply_fill_values(train[feat_cols], fill_values)
    test_filled  = apply_fill_values(test[feat_cols],  fill_values)

    X_train = train_filled.values
    X_test  = test_filled.values

    log.info("prepare_matrices: X_train=%s  X_test=%s  features=%d",
             X_train.shape, X_test.shape, len(feat_cols))

    return {
        "X_train":      X_train,
        "X_test":       X_test,
        "feature_names": feat_cols,
        "fill_values":  fill_values,
    }


# ---------------------------------------------------------------------------
# STEP T4 — Train one LightGBM model
# ---------------------------------------------------------------------------

def _train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sample_weight: np.ndarray,
    cfg: V8Config,
    target_name: str,
) -> Tuple[object, dict]:
    """Train one LightGBM regressor. Returns (model, metrics_dict)."""
    import lightgbm as lgb

    params = dict(cfg.lgbm_params)
    n_est   = params.pop("n_estimators", 800)
    early   = params.pop("early_stopping_rounds", 50)

    model = lgb.LGBMRegressor(n_estimators=n_est, **params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        sample_weight=sample_weight,
        callbacks=[
            lgb.early_stopping(early, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    preds = model.predict(X_test)
    mae   = float(np.mean(np.abs(preds - y_test)))
    rmse  = float(np.sqrt(np.mean((preds - y_test) ** 2)))
    ss_res = np.sum((y_test - preds) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2    = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    metrics = {
        "mae":        round(mae, 4),
        "rmse":       round(rmse, 4),
        "r2":         round(r2, 4),
        "best_iter":  int(model.best_iteration_),
        "n_train":    int(len(y_train)),
        "n_test":     int(len(y_test)),
    }
    log.info(
        "  %s — MAE=%.4f  RMSE=%.4f  R2=%.4f  best_iter=%d",
        target_name, mae, rmse, r2, model.best_iteration_,
    )
    return model, metrics


# ---------------------------------------------------------------------------
# STEP T5 — Save artifacts
# ---------------------------------------------------------------------------

def save_artifacts(
    cfg: V8Config,
    run_id: str,
    models: Dict[str, object],
    fill_values: dict,
    feature_names: list,
    metrics: dict,
) -> None:
    """Save model pickles + inference_config_v8.json to models/v8-sparse-visit/."""
    cfg.ensure_dirs()
    out = cfg.models_dir_path

    for name, model in models.items():
        pkl_path = out / f"{name}_v8.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f, protocol=5)
        log.info("  saved model → %s", pkl_path)

    inference_cfg = {
        "pipeline_version":   "v8-sparse-visit",
        "model_family":       "lightgbm_sparse",
        "run_id":             run_id,
        "generated_at":       datetime.now().isoformat(),
        "all_numeric_features": feature_names,
        "fill_values":        fill_values,
        "metrics":            metrics,
        "gap_config": {
            "max_allowed_gap_for_target": cfg.max_allowed_gap_for_target,
            "gap_weight_summer":          cfg.gap_weight_summer,
        },
    }
    cfg_path = out / "inference_config_v8.json"
    with open(cfg_path, "w") as f:
        json.dump(inference_cfg, f, indent=2, default=str)
    log.info("  saved inference config → %s", cfg_path)


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_v8_pipeline(
    cfg: V8Config = DEFAULT_V8_CONFIG,
    run_id: str | None = None,
    dry_run: bool = False,
) -> dict:
    """
    Execute the full v8 sparse training pipeline.

    Returns the metrics dict (or a dry-run summary on dry_run=True).
    """
    if run_id is None:
        run_id = datetime.now().strftime("v8-%Y%m%d-%H%M%S")

    log.info("=== V8 SPARSE-VISIT PIPELINE  run_id=%s ===", run_id)

    # T0 — Load sparse visits & gap report
    df = load_sparse_visits(cfg)
    df = add_gap_columns(df)
    per_pool, _, _ = generate_gap_report(df, cfg, save_csv=True)

    if dry_run:
        log.info("DRY-RUN: loaded %d visits across %d pools — stopping.", len(df), df["pool_clean"].nunique())
        return {
            "run_id":   run_id,
            "dry_run":  True,
            "visits":   int(len(df)),
            "pools":    int(df["pool_clean"].nunique()),
        }

    # T1 — Feature engineering
    df_feat = build_features(df, cfg)

    # T2 — Targets
    df_model = build_targets(df_feat, cfg)

    # T3 — Temporal split
    train, test = temporal_split(df_model, cfg)

    # T4 — Prepare matrices
    matrices = prepare_matrices(train, test, cfg)
    X_train       = matrices["X_train"]
    X_test        = matrices["X_test"]
    feat_names    = matrices["feature_names"]
    fill_values   = matrices["fill_values"]

    sw_train = train["sample_weight"].values if "sample_weight" in train.columns else np.ones(len(train))

    all_metrics: dict = {}
    trained_models: dict = {}

    # T5 — Train models
    target_specs = [
        ("target_cl_next_visit",   "chlorine_next"),
        ("target_ph_next_visit",   "ph_next"),
        ("target_turb_next_visit", "turbidity_next"),
    ]

    for target_col, model_name in target_specs:
        y_tr = train[target_col].values
        y_te = test[target_col].values
        # Drop rows where target is NaN (safety net)
        mask_tr = ~np.isnan(y_tr)
        mask_te = ~np.isnan(y_te)
        model, metrics = _train_lgbm(
            X_train[mask_tr], y_tr[mask_tr],
            X_test[mask_te],  y_te[mask_te],
            sw_train[mask_tr],
            cfg, model_name,
        )
        all_metrics[model_name] = metrics
        trained_models[model_name] = model

    # T6 — Save
    save_artifacts(cfg, run_id, trained_models, fill_values, feat_names, all_metrics)

    log.info("PIPELINE V8 COMPLETE  run_id=%s  metrics=%s", run_id, all_metrics)
    return {"run_id": run_id, "metrics": all_metrics, "dry_run": False}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pool V8 sparse-visit training pipeline")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Load & validate data only — skip training")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    try:
        result = run_v8_pipeline(DEFAULT_V8_CONFIG, run_id=args.run_id, dry_run=args.dry_run)
        if args.dry_run:
            log.info("DRY-RUN OK: %s", result)
        return 0
    except FileNotFoundError as e:
        log.error("DATA ERROR: %s", e)
        return 2
    except Exception as e:
        log.exception("v8 pipeline failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())

