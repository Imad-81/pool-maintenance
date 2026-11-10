"""
V8 Evaluation — gap-stratified metrics and cross-model comparison.

Provides:
  - evaluate_gap_stratified()  — metrics broken down by gap bucket & season
  - compare_v7_vs_v8()         — head-to-head on matched visit dates
  - print_eval_report()        — formatted summary to stdout

CLI:
    python -m ml.v8.evaluate_v8
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.v8.config_v8 import V8Config, DEFAULT_V8_CONFIG
from ml.v8.sparse_dataset import load_sparse_visits, add_gap_columns
from ml.v8.features_v8 import build_features, all_feature_groups, apply_fill_values
from ml.v8.train_v8 import build_targets, temporal_split

log = logging.getLogger("ml.v8.evaluate_v8")


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """Compute MAE, RMSE, R2 for a prediction array."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "label": label}
    mae  = float(np.mean(np.abs(yp - yt)))
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {
        "n":    int(len(yt)),
        "mae":  round(mae,  4),
        "rmse": round(rmse, 4),
        "r2":   round(r2,   4),
        "label": label,
    }


# ---------------------------------------------------------------------------
# Load V8 models
# ---------------------------------------------------------------------------

def load_v8_models(cfg: V8Config = DEFAULT_V8_CONFIG) -> Tuple[dict, dict, List[str]]:
    """
    Load all three v8 LightGBM models + fill values.

    Returns:
        (models_dict, fill_values, feature_names)
        models_dict keys: "chlorine_next", "ph_next", "turbidity_next"
    """
    out = cfg.models_dir_path
    cfg_path = out / "inference_config_v8.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"inference_config_v8.json not found at {cfg_path}. "
            "Run `python -m ml.v8.train_v8` first."
        )
    with open(cfg_path) as f:
        inf_cfg = json.load(f)

    fill_values   = inf_cfg["fill_values"]
    feature_names = inf_cfg["all_numeric_features"]

    models = {}
    for name in ("chlorine_next", "ph_next", "turbidity_next"):
        pkl = out / f"{name}_v8.pkl"
        with open(pkl, "rb") as f:
            models[name] = pickle.load(f)
        log.info("Loaded model: %s", pkl.name)

    return models, fill_values, feature_names


# ---------------------------------------------------------------------------
# Gap-stratified evaluation
# ---------------------------------------------------------------------------

_GAP_BUCKET_LABELS = {
    0: "first_visit",
    1: "same_day(1d)",
    2: "short(2-3d)",
    3: "weekly(4-7d)",
    4: "biweekly(8-14d)",
    5: "long(15d+)",
}

_SEASON_ORDER = ["Summer", "Spring", "Autumn", "Winter"]


def evaluate_gap_stratified(
    df_test: pd.DataFrame,
    models: dict,
    fill_values: dict,
    feature_names: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Run predictions on test set and compute metrics stratified by:
      - gap_category (1–5)
      - visit_season

    Returns dict:
        "overall"      — single-row overall metrics per target
        "by_gap"       — metrics per gap_category per target
        "by_season"    — metrics per season per target
    """
    # Align feature matrix
    feat_df = apply_fill_values(df_test[feature_names], fill_values)
    X = feat_df.values

    targets = {
        "chlorine_next": "target_cl_next_visit",
        "ph_next":       "target_ph_next_visit",
        "turbidity_next":"target_turb_next_visit",
    }

    preds = {}
    for name, model in models.items():
        preds[name] = model.predict(X)

    results = {"overall": [], "by_gap": [], "by_season": []}

    for model_name, target_col in targets.items():
        if target_col not in df_test.columns:
            continue
        y = df_test[target_col].values
        yp = preds[model_name]

        # Overall
        m = _metrics(y, yp, label=f"{model_name}/overall")
        results["overall"].append({"target": model_name, **m})

        # By gap category
        for cat_id, cat_label in _GAP_BUCKET_LABELS.items():
            mask = (df_test["gap_category"] == cat_id).values
            m = _metrics(y[mask], yp[mask], label=cat_label)
            results["by_gap"].append({"target": model_name, "gap_bucket": cat_label, **m})

        # By season
        for season in _SEASON_ORDER:
            if "visit_season" in df_test.columns:
                mask = (df_test["visit_season"] == season).values
            elif "is_summer" in df_test.columns:
                mask = (df_test["is_summer"] == (season == "Summer")).values
            else:
                continue
            m = _metrics(y[mask], yp[mask], label=season)
            results["by_season"].append({"target": model_name, "season": season, **m})

    return {k: pd.DataFrame(v) for k, v in results.items()}


# ---------------------------------------------------------------------------
# Cross-model comparison (v8 sparse vs v7 on matched visit dates)
# ---------------------------------------------------------------------------

def compare_v7_vs_v8(
    df_test: pd.DataFrame,
    models_v8: dict,
    fill_values_v8: dict,
    feature_names_v8: List[str],
    cfg: V8Config = DEFAULT_V8_CONFIG,
) -> Optional[pd.DataFrame]:
    """
    Compare v8 sparse vs v7 metrics on the same holdout visit dates.

    Loads v7 metrics from its inference_config_v7.json (read-only).
    Returns a comparison DataFrame, or None if v7 config not found.
    """
    v7_cfg_path = cfg.project_root / "models" / "v7-daily-delta" / "inference_config_v7.json"
    if not v7_cfg_path.exists():
        log.warning("v7 inference config not found at %s — skipping cross-model comparison", v7_cfg_path)
        return None

    with open(v7_cfg_path) as f:
        v7_inf = json.load(f)
    v7_metrics = v7_inf.get("metrics", {})

    # Compute v8 overall metrics
    feat_df = apply_fill_values(df_test[feature_names_v8], fill_values_v8)
    X = feat_df.values

    target_map = {
        "chlorine_next": "target_cl_next_visit",
        "ph_next":       "target_ph_next_visit",
        "turbidity_next":"target_turb_next_visit",
    }

    rows = []
    for model_name, target_col in target_map.items():
        if target_col not in df_test.columns:
            continue
        y  = df_test[target_col].values
        yp = models_v8[model_name].predict(X)
        m_v8 = _metrics(y, yp)

        # v7 reported metrics (from its own full timeseries holdout)
        m_v7 = v7_metrics.get(model_name, {})

        rows.append({
            "target":          model_name,
            "v8_mae":          m_v8["mae"],
            "v8_rmse":         m_v8["rmse"],
            "v8_r2":           m_v8["r2"],
            "v8_n_test":       m_v8["n"],
            "v7_mae":          m_v7.get("mae",  np.nan),
            "v7_rmse":         m_v7.get("rmse", np.nan),
            "v7_r2":           m_v7.get("r2",   np.nan),
            "mae_delta":       round(m_v8["mae"] - m_v7.get("mae", np.nan), 4),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full eval runner
# ---------------------------------------------------------------------------

def run_evaluation(cfg: V8Config = DEFAULT_V8_CONFIG) -> dict:
    """
    Load v8 models, run gap-stratified evaluation, compare with v7.
    Returns dict with all result DataFrames.
    """
    models, fill_values, feature_names = load_v8_models(cfg)

    # Rebuild test set
    df = load_sparse_visits(cfg)
    df = add_gap_columns(df)
    df_feat = build_features(df, cfg)
    df_model = build_targets(df_feat, cfg)
    _, test = temporal_split(df_model, cfg)

    # Only keep features present in feature_names
    missing_feats = [c for c in feature_names if c not in test.columns]
    for c in missing_feats:
        test[c] = fill_values.get(c, 0.0)

    strat = evaluate_gap_stratified(test, models, fill_values, feature_names)
    comp  = compare_v7_vs_v8(test, models, fill_values, feature_names, cfg)

    return {
        "overall":   strat["overall"],
        "by_gap":    strat["by_gap"],
        "by_season": strat["by_season"],
        "v7_vs_v8":  comp,
        "n_test":    len(test),
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def _fmt(val, decimals: int = 4) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "    N/A"
    return f"{val:>{decimals + 5}.{decimals}f}"


def print_eval_report(results: dict) -> None:
    sep = "=" * 68

    print(f"\n{sep}")
    print("V8 SPARSE-VISIT MODEL — EVALUATION REPORT")
    print(f"Test rows: {results['n_test']}")
    print(sep)

    print("\n── OVERALL METRICS ──")
    df = results["overall"]
    if df is not None and not df.empty:
        print(f"{'Target':<20} {'N':>6} {'MAE':>8} {'RMSE':>8} {'R2':>7}")
        print("-" * 54)
        for _, row in df.iterrows():
            print(f"{row['target']:<20} {row['n']:>6} {row['mae']:>8.4f} {row['rmse']:>8.4f} {row['r2']:>7.4f}")

    print("\n── METRICS BY GAP BUCKET ──")
    df = results["by_gap"]
    if df is not None and not df.empty:
        for target in df["target"].unique():
            print(f"\n  {target}:")
            sub = df[df["target"] == target].copy()
            print(f"  {'Gap Bucket':<20} {'N':>6} {'MAE':>8} {'RMSE':>8} {'R2':>7}")
            print("  " + "-" * 50)
            for _, row in sub.iterrows():
                n_str = str(row["n"]) if row["n"] > 0 else "  —"
                mae_s = f"{row['mae']:8.4f}" if row["n"] > 0 else "       —"
                r2_s  = f"{row['r2']:7.4f}"  if row["n"] > 0 else "      —"
                print(f"  {row['gap_bucket']:<20} {n_str:>6} {mae_s} {row['rmse']:>8.4f} {r2_s}")

    print("\n── METRICS BY SEASON ──")
    df = results["by_season"]
    if df is not None and not df.empty:
        for target in df["target"].unique():
            print(f"\n  {target}:")
            sub = df[df["target"] == target]
            print(f"  {'Season':<12} {'N':>6} {'MAE':>8} {'RMSE':>8} {'R2':>7}")
            print("  " + "-" * 44)
            for _, row in sub.iterrows():
                print(f"  {row['season']:<12} {row['n']:>6} {row['mae']:>8.4f} {row['rmse']:>8.4f} {row['r2']:>7.4f}")

    print("\n── V7 (DAILY) vs V8 (SPARSE) ──")
    df = results.get("v7_vs_v8")
    if df is not None and not df.empty:
        print(f"{'Target':<20} {'V8 MAE':>8} {'V7 MAE':>8} {'Delta':>8} {'V8 R2':>7} {'V7 R2':>7}")
        print("-" * 62)
        for _, row in df.iterrows():
            delta_sym = "▲" if row["mae_delta"] > 0 else "▼"
            print(
                f"{row['target']:<20} {row['v8_mae']:>8.4f} {row['v7_mae']:>8.4f}"
                f" {delta_sym}{abs(row['mae_delta']):>7.4f} {row['v8_r2']:>7.4f} {row['v7_r2']:>7.4f}"
            )
        print("  (▲ = v8 worse  ▼ = v8 better)")
    else:
        print("  (v7 config not found — skipping comparison)")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate V8 sparse-visit model")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    results = run_evaluation(DEFAULT_V8_CONFIG)
    print_eval_report(results)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

