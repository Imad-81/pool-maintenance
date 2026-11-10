"""
V8 Grouped Models — Training Pipeline.

Trains 6 LightGBM models total:
    Group 1 × {chlorine, pH, turbidity}  →  models/v8-grouped/group1/
    Group 2 × {chlorine, pH, turbidity}  →  models/v8-grouped/group2/

Each group uses its own temporal split and hyperparameters.

Usage:
    python -m ml.v8.grouped.train_grouped                # full run
    python -m ml.v8.grouped.train_grouped --dry-run     # validate data only
    python -m ml.v8.grouped.train_grouped --group G1    # train one group only
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.v8.config_v8 import V8Config, DEFAULT_V8_CONFIG, REG_CHLORINE_MIN, REG_PH_MIN, REG_PH_MAX
from ml.v8.sparse_dataset import load_sparse_visits, add_gap_columns, generate_gap_report
from ml.v8.grouped.grouper import (
    assign_groups, assert_group_counts, group_summary, seasonal_crosstab
)
from ml.v8.grouped.features_grouped import (
    build_features_for_group,
    g1_all_features, g2_all_features,
    compute_fill_values, apply_fill_values,
)

log = logging.getLogger("ml.v8.grouped.train_grouped")

# ---------------------------------------------------------------------------
# Per-group LightGBM hyper-parameters
# G1 (short-gap): high autocorrelation signal → less complexity needed
# G2 (long-gap):  decay dynamics → more capacity, careful regularisation
# ---------------------------------------------------------------------------
LGBM_PARAMS_G1 = {
    "objective":          "regression",
    "metric":             "mae",
    "num_leaves":         31,
    "learning_rate":      0.05,
    "feature_fraction":   0.75,
    "bagging_fraction":   0.8,
    "bagging_freq":       5,
    "min_child_samples":  25,
    "n_estimators":       600,
    "early_stopping_rounds": 50,
    "verbose":            -1,
    "n_jobs":             -1,
    "random_state":       42,
}

LGBM_PARAMS_G2 = {
    "objective":          "regression",
    "metric":             "mae",
    "num_leaves":         63,
    "learning_rate":      0.04,
    "feature_fraction":   0.80,
    "bagging_fraction":   0.8,
    "bagging_freq":       5,
    "min_child_samples":  15,   # smaller — G2 has less data
    "lambda_l1":          0.1,  # extra L1 regularisation for smaller dataset
    "lambda_l2":          0.1,
    "n_estimators":       1000,
    "early_stopping_rounds": 60,
    "verbose":            -1,
    "n_jobs":             -1,
    "random_state":       42,
}

GROUP_PARAMS = {"G1": LGBM_PARAMS_G1, "G2": LGBM_PARAMS_G2}


# ---------------------------------------------------------------------------
# STEP T1 — Build targets
# ---------------------------------------------------------------------------

def build_targets(df: pd.DataFrame, cfg: V8Config) -> pd.DataFrame:
    """
    Add target columns (next observed visit's post-treatment values).
    Excludes last visit per pool + visits where gap_to > max_allowed_gap_for_target.
    """
    df = df.sort_values(["pool_clean", "date"]).copy()

    target_map = {
        "cl_post":   "target_cl_next",
        "ph_post":   "target_ph_next",
        "turb_post": "target_turb_next",
    }
    for src, tgt in target_map.items():
        if src in df.columns:
            df[tgt] = df.groupby("pool_clean")[src].shift(-1)
        else:
            df[tgt] = np.nan

    # Breach flag at next visit (for evaluation)
    if "target_cl_next" in df.columns:
        df["target_cl_breach"] = df["target_cl_next"] < REG_CHLORINE_MIN
    if "target_ph_next" in df.columns:
        df["target_ph_breach"] = (
            (df["target_ph_next"] < REG_PH_MIN) | (df["target_ph_next"] > REG_PH_MAX)
        )

    before = len(df)
    # Drop last visit per pool (NaN target)
    df = df.dropna(subset=["target_cl_next", "target_ph_next"]).copy()
    # Drop rows with excessively long horizon
    if "gap_to" in df.columns:
        df = df[
            df["gap_to"].isna() | (df["gap_to"] <= cfg.max_allowed_gap_for_target)
        ].copy()

    after = len(df)
    log.info(
        "build_targets: %d → %d rows (%d dropped)",
        before, after, before - after,
    )
    return df


# ---------------------------------------------------------------------------
# STEP T2 — Temporal split (independent per group)
# ---------------------------------------------------------------------------

def temporal_split(df: pd.DataFrame, cfg: V8Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["date"].quantile(cfg.temporal_split_quantile)
    train  = df[df["date"] < cutoff].copy()
    test   = df[df["date"] >= cutoff].copy()
    log.info(
        "temporal_split: cutoff=%s  train=%d  test=%d",
        str(cutoff.date()), len(train), len(test),
    )
    return train, test


# ---------------------------------------------------------------------------
# STEP T3 — Feature matrix
# ---------------------------------------------------------------------------

def prepare_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    group_id: str,
    cfg: V8Config,
) -> Dict:
    feat_cols = g1_all_features() if group_id == "G1" else g2_all_features()
    feat_cols = [c for c in feat_cols if c in train.columns]

    # Stricter null threshold for G2 (smaller dataset → noisier rare features)
    null_threshold = cfg.feature_null_drop_threshold if group_id == "G1" else 0.3
    null_rates = train[feat_cols].isnull().mean()
    high_null = null_rates[null_rates > null_threshold].index.tolist()
    if high_null:
        log.info("Dropping high-null features (>%.0f%%): %s", null_threshold * 100, high_null)
        feat_cols = [c for c in feat_cols if c not in high_null]

    fill_values = compute_fill_values(train, feat_cols)
    X_train = apply_fill_values(train[feat_cols], fill_values).values
    X_test  = apply_fill_values(test[feat_cols],  fill_values).values

    log.info(
        "prepare_matrices(%s): X_train=%s  X_test=%s  features=%d",
        group_id, X_train.shape, X_test.shape, len(feat_cols),
    )
    return {
        "X_train": X_train, "X_test": X_test,
        "feature_names": feat_cols, "fill_values": fill_values,
    }


# ---------------------------------------------------------------------------
# STEP T4 — Train one LightGBM model
# ---------------------------------------------------------------------------

def _train_lgbm(
    X_train, y_train, X_test, y_test,
    sample_weight, params: dict, target_name: str,
) -> Tuple[object, dict]:
    import lightgbm as lgb

    p = dict(params)
    n_est = p.pop("n_estimators", 800)
    early = p.pop("early_stopping_rounds", 50)

    model = lgb.LGBMRegressor(n_estimators=n_est, **p)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        sample_weight=sample_weight,
        callbacks=[
            lgb.early_stopping(early, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    preds  = model.predict(X_test)
    mae    = float(np.mean(np.abs(preds - y_test)))
    rmse   = float(np.sqrt(np.mean((preds - y_test) ** 2)))
    ss_res = np.sum((y_test - preds) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    metrics = {
        "mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4),
        "best_iter": int(model.best_iteration_),
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
    }
    log.info(
        "    %s — MAE=%.4f  RMSE=%.4f  R2=%.4f  best_iter=%d",
        target_name, mae, rmse, r2, model.best_iteration_,
    )
    return model, metrics


# ---------------------------------------------------------------------------
# STEP T5 — Save artifacts
# ---------------------------------------------------------------------------

def save_group_artifacts(
    cfg: V8Config,
    group_id: str,
    run_id: str,
    models: dict,
    fill_values: dict,
    feature_names: List[str],
    metrics: dict,
    n_train: int,
    n_test: int,
) -> None:
    out_dir = cfg.project_root / "models" / "v8-grouped" / f"group{group_id[-1]}"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_g{group_id[-1]}"
    for name, model in models.items():
        pkl_path = out_dir / f"{name}{suffix}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f, protocol=5)
        log.info("    saved → %s", pkl_path.name)

    inf_cfg = {
        "pipeline_version":    "v8-grouped",
        "group_id":            group_id,
        "model_family":        "lightgbm_grouped_sparse",
        "run_id":              run_id,
        "generated_at":        datetime.now().isoformat(),
        "all_numeric_features": feature_names,
        "fill_values":         fill_values,
        "metrics":             metrics,
        "n_train":             n_train,
        "n_test":              n_test,
        "group_boundary": {
            "G1": "gap_since in [1, 6] days",
            "G2": "gap_since in [7, 30] days",
        },
    }
    cfg_path = out_dir / f"inference_config_g{group_id[-1]}.json"
    with open(cfg_path, "w") as f:
        json.dump(inf_cfg, f, indent=2, default=str)
    log.info("    saved config → %s", cfg_path.name)


# ---------------------------------------------------------------------------
# Train one group end-to-end
# ---------------------------------------------------------------------------

def train_group(
    df_full: pd.DataFrame,
    group_id: str,
    cfg: V8Config,
    run_id: str,
) -> dict:
    """Train all 3 models for one group. Returns metrics dict."""
    log.info("--- Training %s ---", group_id)

    sub = df_full[df_full["group"] == group_id].copy()
    sub = build_features_for_group(sub, group_id, cfg)
    sub = build_targets(sub, cfg)
    train, test = temporal_split(sub, cfg)

    matrices = prepare_matrices(train, test, group_id, cfg)
    X_train, X_test = matrices["X_train"], matrices["X_test"]
    feat_names  = matrices["feature_names"]
    fill_values = matrices["fill_values"]
    sw_train = train["sample_weight"].values if "sample_weight" in train.columns else np.ones(len(train))

    params = GROUP_PARAMS[group_id]
    all_metrics: dict = {}
    trained_models: dict = {}

    for target_col, model_name in [
        ("target_cl_next",   "chlorine_next"),
        ("target_ph_next",   "ph_next"),
        ("target_turb_next", "turbidity_next"),
    ]:
        y_tr = train[target_col].values
        y_te = test[target_col].values
        mask_tr = ~np.isnan(y_tr)
        mask_te = ~np.isnan(y_te)
        model, metrics = _train_lgbm(
            X_train[mask_tr], y_tr[mask_tr],
            X_test[mask_te],  y_te[mask_te],
            sw_train[mask_tr], params, f"{group_id}/{model_name}",
        )
        all_metrics[model_name]    = metrics
        trained_models[model_name] = model

    save_group_artifacts(
        cfg, group_id, run_id,
        trained_models, fill_values, feat_names, all_metrics,
        n_train=len(train), n_test=len(test),
    )
    return all_metrics


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_grouped_pipeline(
    cfg: V8Config = DEFAULT_V8_CONFIG,
    run_id: Optional[str] = None,
    dry_run: bool = False,
    groups: Optional[List[str]] = None,
) -> dict:
    if run_id is None:
        run_id = datetime.now().strftime("v8-grouped-%Y%m%d-%H%M%S")
    if groups is None:
        groups = ["G1", "G2"]

    log.info("=== V8 GROUPED PIPELINE  run_id=%s  groups=%s ===", run_id, groups)

    # Load + gap columns + group assignment
    df = load_sparse_visits(cfg)
    df = add_gap_columns(df)
    df = assign_groups(df)
    assert_group_counts(df)

    # Print group summary
    summ = group_summary(df)
    log.info("Group summary:\n%s", summ[["n_visits","gap_mean","cl_mean","cl_std","summer_pct"]].to_string())

    if dry_run:
        ct = seasonal_crosstab(df)
        log.info("DRY-RUN: groups ready\n%s", ct.to_string())
        return {
            "run_id": run_id, "dry_run": True,
            "G1_visits": int((df["group"] == "G1").sum()),
            "G2_visits": int((df["group"] == "G2").sum()),
            "pools":     int(df["pool_clean"].nunique()),
        }

    all_metrics: dict = {}
    for gid in groups:
        metrics = train_group(df, gid, cfg, run_id)
        all_metrics[gid] = metrics
        log.info("%s COMPLETE: %s", gid, metrics)

    log.info("GROUPED PIPELINE COMPLETE  run_id=%s", run_id)
    for gid, m in all_metrics.items():
        for tgt, vals in m.items():
            log.info("  %s/%s: MAE=%.4f  R2=%.4f", gid, tgt, vals["mae"], vals["r2"])

    return {"run_id": run_id, "dry_run": False, "metrics": all_metrics}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="V8 Grouped Pool Models — Training")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--group",  default=None, choices=["G1", "G2"],
                        help="Train only one group (default: both)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    groups = [args.group] if args.group else None
    try:
        result = run_grouped_pipeline(
            DEFAULT_V8_CONFIG, run_id=args.run_id,
            dry_run=args.dry_run, groups=groups,
        )
        if args.dry_run:
            log.info("DRY-RUN OK: %s", result)
        return 0
    except Exception as e:
        logging.exception("Grouped pipeline failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())

