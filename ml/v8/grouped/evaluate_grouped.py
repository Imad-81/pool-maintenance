"""
V8 Grouped Models — Evaluation Pipeline.

Loads trained G1 and G2 models, evaluates them on their own test sets,
runs cross-evaluation (G1 model on G2 data and vice versa), and
compares against the V8 baseline.

Usage:
    python -m ml.v8.grouped.evaluate_grouped
    python -m ml.v8.grouped.evaluate_grouped --group G1
    python -m ml.v8.grouped.evaluate_grouped --no-cross
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.v8.config_v8 import V8Config, DEFAULT_V8_CONFIG
from ml.v8.sparse_dataset import load_sparse_visits, add_gap_columns
from ml.v8.grouped.grouper import assign_groups
from ml.v8.grouped.features_grouped import (
    build_features_for_group,
    apply_fill_values,
    g1_all_features,
    g2_all_features,
)
from ml.v8.grouped.train_grouped import build_targets, temporal_split

log = logging.getLogger("ml.v8.grouped.evaluate_grouped")

TARGETS = ["chlorine_next", "ph_next", "turbidity_next"]
TARGET_COLS = {
    "chlorine_next": "target_cl_next",
    "ph_next": "target_ph_next",
    "turbidity_next": "target_turb_next",
}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_group_models(
    group_id: str,
    cfg: V8Config = DEFAULT_V8_CONFIG,
) -> Tuple[Dict, Dict, Dict]:
    """
    Returns (models, fill_values, inference_config) for the given group.
    models: {"chlorine_next": lgbm, "ph_next": lgbm, "turbidity_next": lgbm}
    """
    gid = group_id.lower()   # "g1" or "g2"
    gdir = cfg.project_root / "models" / "v8-grouped" / f"group{gid[1]}"

    cfg_path = gdir / f"inference_config_{gid}.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Inference config not found: {cfg_path}\n"
            "Run:  python -m ml.v8.grouped.train_grouped  first."
        )
    with cfg_path.open() as f:
        icfg = json.load(f)

    models: Dict = {}
    for tgt in TARGETS:
        suffix = "_g1" if group_id.upper() == "G1" else "_g2"
        pkl_path = gdir / f"{tgt}{suffix}.pkl"
        with pkl_path.open("rb") as f:
            models[tgt] = pickle.load(f)

    fill_values = icfg["fill_values"]
    return models, fill_values, icfg


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    return {
        "mae": round(_mae(y_true, y_pred), 4),
        "rmse": round(_rmse(y_true, y_pred), 4),
        "r2": round(_r2(y_true, y_pred), 4),
        "n": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

def evaluate_group_test(
    df_test: pd.DataFrame,
    group_id: str,
    models: Dict,
    fill_values: Dict,
    feature_names: List[str],
    *,
    gap_buckets: bool = True,
    seasonal: bool = True,
) -> Dict:
    """
    Evaluate a group's models on its held-out test dataframe.
    Returns nested dict: {target: {overall: metrics, by_gap: {...}, by_season: {...}}}
    """
    # Ensure all required features are present
    missing = [c for c in feature_names if c not in df_test.columns]
    for c in missing:
        df_test[c] = fill_values.get(c, 0.0)

    X = apply_fill_values(df_test[feature_names], fill_values).values
    gap = df_test["gap_since"].values if "gap_since" in df_test.columns else np.zeros(len(df_test))

    season = (
        df_test["visit_season"].values
        if "visit_season" in df_test.columns
        else np.array(["Unknown"] * len(df_test))
    )

    results: Dict = {}
    for tgt in TARGETS:
        target_col = TARGET_COLS[tgt]
        if target_col not in df_test.columns:
            continue

        y_true = df_test[target_col].values.astype(float)
        mask = ~np.isnan(y_true)
        if mask.sum() == 0:
            results[tgt] = {"overall": {}, "by_gap": {}, "by_season": {}}
            continue

        y_pred = models[tgt].predict(X[mask])
        yt = y_true[mask]
        g = gap[mask]
        s = season[mask]

        entry: Dict = {"overall": _metrics(yt, y_pred)}

        if gap_buckets:
            buckets = {
                "1-3d": (g >= 1) & (g <= 3),
                "4-6d": (g >= 4) & (g <= 6),
                "7-14d": (g >= 7) & (g <= 14),
                "15-30d": (g >= 15) & (g <= 30),
            }
            entry["by_gap"] = {}
            for label, bm in buckets.items():
                if bm.sum() >= 5:
                    entry["by_gap"][label] = _metrics(yt[bm], y_pred[bm])

        if seasonal:
            entry["by_season"] = {}
            for sn in ["Summer", "Spring", "Autumn", "Winter"]:
                sm = s == sn
                if sm.sum() >= 5:
                    entry["by_season"][sn] = _metrics(yt[sm], y_pred[sm])

        results[tgt] = entry

    return results


# ---------------------------------------------------------------------------
# Cross-evaluation
# ---------------------------------------------------------------------------

def cross_evaluate(
    df_grouped: pd.DataFrame,
    cfg: V8Config,
    models_g1: Dict,
    models_g2: Dict,
    fill_g1: Dict,
    fill_g2: Dict,
    feat_g1: List[str],
    feat_g2: List[str],
) -> Dict:
    """
    Cross evaluation:
      - G1 model scored on G2 test data (with G1 features extracted from G2 visits)
      - G2 model scored on G1 test data (with G2 features extracted from G1 visits)
    """
    # 1. G1 model on G2 data
    sub_g2 = df_grouped[df_grouped["group"] == "G2"].copy()
    feat_g2_for_g1 = build_features_for_group(sub_g2, "G1", cfg)
    model_g2_for_g1 = build_targets(feat_g2_for_g1, cfg)
    _, test_g2_for_g1 = temporal_split(model_g2_for_g1, cfg)

    for c in [f for f in feat_g1 if f not in test_g2_for_g1.columns]:
        test_g2_for_g1[c] = fill_g1.get(c, 0.0)

    X_g2_for_g1 = apply_fill_values(test_g2_for_g1[feat_g1], fill_g1).values
    res_g1_on_g2: Dict = {}
    for tgt in TARGETS:
        tcol = TARGET_COLS[tgt]
        if tcol in test_g2_for_g1.columns:
            y_true = test_g2_for_g1[tcol].values.astype(float)
            m = ~np.isnan(y_true)
            yp = models_g1[tgt].predict(X_g2_for_g1[m])
            res_g1_on_g2[tgt] = _metrics(y_true[m], yp)

    # 2. G2 model on G1 data
    sub_g1 = df_grouped[df_grouped["group"] == "G1"].copy()
    feat_g1_for_g2 = build_features_for_group(sub_g1, "G2", cfg)
    model_g1_for_g2 = build_targets(feat_g1_for_g2, cfg)
    _, test_g1_for_g2 = temporal_split(model_g1_for_g2, cfg)

    for c in [f for f in feat_g2 if f not in test_g1_for_g2.columns]:
        test_g1_for_g2[c] = fill_g2.get(c, 0.0)

    X_g1_for_g2 = apply_fill_values(test_g1_for_g2[feat_g2], fill_g2).values
    res_g2_on_g1: Dict = {}
    for tgt in TARGETS:
        tcol = TARGET_COLS[tgt]
        if tcol in test_g1_for_g2.columns:
            y_true = test_g1_for_g2[tcol].values.astype(float)
            m = ~np.isnan(y_true)
            yp = models_g2[tgt].predict(X_g1_for_g2[m])
            res_g2_on_g1[tgt] = _metrics(y_true[m], yp)

    return {
        "g1_model_on_g2_data": res_g1_on_g2,
        "g2_model_on_g1_data": res_g2_on_g1,
    }


# ---------------------------------------------------------------------------
# Comparison against V8 baseline
# ---------------------------------------------------------------------------

def load_v8_baseline_metrics(cfg: V8Config = DEFAULT_V8_CONFIG) -> Dict:
    """Load V8 baseline overall metrics from its inference_config."""
    baseline_path = cfg.models_dir_path / "inference_config_v8.json"
    if not baseline_path.exists():
        return {}
    with baseline_path.open() as f:
        data = json.load(f)
    return data.get("metrics", {})


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def _fmt_row(label: str, m: Dict) -> str:
    if not m or "mae" not in m:
        return f"  {label:<30}  {'N/A':>8}"
    return (
        f"  {label:<30}  MAE={m['mae']:6.4f}  RMSE={m['rmse']:6.4f}"
        f"  R²={m['r2']:+.4f}  n={m.get('n', '?')}"
    )


def print_full_report(
    g1_results: Dict,
    g2_results: Dict,
    cross_results: Dict,
    baseline_metrics: Dict,
) -> None:
    """Print a formatted evaluation report to stdout."""
    sep = "=" * 80
    thin = "-" * 80

    print(f"\n{sep}")
    print("  V8 GROUPED MODELS — HELD-OUT TEST EVALUATION REPORT")
    print(sep)

    for group_lbl, results in [
        ("GROUP 1: Short Gaps (gap < 7 days, n_test ≈ 5,627)", g1_results),
        ("GROUP 2: Long Gaps (gap 7–30 days, n_test ≈ 1,988)", g2_results),
    ]:
        print(f"\n{thin}")
        print(f"  {group_lbl}")
        print(f"{thin}")
        for tgt in TARGETS:
            if tgt not in results:
                continue
            entry = results[tgt]
            print(f"\n  Target: [{tgt}]")
            print(_fmt_row("Overall Test", entry.get("overall", {})))

            by_gap = entry.get("by_gap", {})
            if by_gap:
                print("  Stratified by Gap Duration:")
                for gb, gm in by_gap.items():
                    print(_fmt_row(f"    gap={gb}", gm))

            by_season = entry.get("by_season", {})
            if by_season:
                print("  Stratified by Season:")
                for sn, sm in by_season.items():
                    print(_fmt_row(f"    {sn}", sm))

    print(f"\n{sep}")
    print("  CROSS-EVALUATION (Specialisation Proof)")
    print(sep)
    print("  Hypothesis: G1 model will degrade on G2 data; G2 model will degrade on G1 data.\n")

    g1_on_g2 = cross_results.get("g1_model_on_g2_data", {})
    g2_on_g1 = cross_results.get("g2_model_on_g1_data", {})

    print("  Group 1 Model tested on Group 2 Data (expect WORSE than G2 native):")
    for tgt in TARGETS:
        native_mae = g2_results.get(tgt, {}).get("overall", {}).get("mae", float("nan"))
        cross_mae = g1_on_g2.get(tgt, {}).get("mae", float("nan"))
        diff = cross_mae - native_mae
        print(_fmt_row(f"    {tgt} (cross)", g1_on_g2.get(tgt, {})) + f"  [Δ vs native: {diff:+.4f}]")

    print("\n  Group 2 Model tested on Group 1 Data (expect WORSE than G1 native):")
    for tgt in TARGETS:
        native_mae = g1_results.get(tgt, {}).get("overall", {}).get("mae", float("nan"))
        cross_mae = g2_on_g1.get(tgt, {}).get("mae", float("nan"))
        diff = cross_mae - native_mae
        print(_fmt_row(f"    {tgt} (cross)", g2_on_g1.get(tgt, {})) + f"  [Δ vs native: {diff:+.4f}]")

    print(f"\n{sep}")
    print("  COMPARISON SUMMARY: V8 BASELINE vs V8 GROUPED (Test Sets)")
    print(sep)
    header = f"  {'Model / Group':<32} {'Chlorine MAE':<15} {'pH MAE':<12} {'Turbidity MAE':<15}"
    print(header)
    print("  " + "-" * 74)

    # Baseline V8
    b_cl = baseline_metrics.get("chlorine_next", {}).get("mae", float("nan"))
    b_ph = baseline_metrics.get("ph_next", {}).get("mae", float("nan"))
    b_tb = baseline_metrics.get("turbidity_next", {}).get("mae", float("nan"))
    print(f"  {'V8 Baseline (All Gaps)':<32} {b_cl:<15.4f} {b_ph:<12.4f} {b_tb:<15.4f}")

    # G1
    g1_cl = g1_results.get("chlorine_next", {}).get("overall", {}).get("mae", float("nan"))
    g1_ph = g1_results.get("ph_next", {}).get("overall", {}).get("mae", float("nan"))
    g1_tb = g1_results.get("turbidity_next", {}).get("overall", {}).get("mae", float("nan"))
    print(f"  {'V8 Group 1 (< 7 days)':<32} {g1_cl:<15.4f} {g1_ph:<12.4f} {g1_tb:<15.4f}")

    # G2
    g2_cl = g2_results.get("chlorine_next", {}).get("overall", {}).get("mae", float("nan"))
    g2_ph = g2_results.get("ph_next", {}).get("overall", {}).get("mae", float("nan"))
    g2_tb = g2_results.get("turbidity_next", {}).get("overall", {}).get("mae", float("nan"))
    print(f"  {'V8 Group 2 (7–30 days)':<32} {g2_cl:<15.4f} {g2_ph:<12.4f} {g2_tb:<15.4f}")

    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate V8 Grouped Models")
    parser.add_argument("--group", choices=["G1", "G2"], default=None,
                        help="Evaluate a single group (default: both)")
    parser.add_argument("--no-cross", action="store_true",
                        help="Skip cross-evaluation")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = DEFAULT_V8_CONFIG

    # Load data
    log.info("Loading sparse visits...")
    obs = load_sparse_visits(cfg)
    obs = add_gap_columns(obs)
    obs_grouped = assign_groups(obs)

    # Load models
    models_g1, fill_g1, icfg_g1 = load_group_models("G1", cfg)
    models_g2, fill_g2, icfg_g2 = load_group_models("G2", cfg)

    feat_names_g1 = icfg_g1["all_numeric_features"]
    feat_names_g2 = icfg_g2["all_numeric_features"]

    g1_results: Dict = {}
    g2_results: Dict = {}
    cross_results: Dict = {}

    if args.group in (None, "G1"):
        log.info("Preparing G1 test dataset...")
        sub_g1 = obs_grouped[obs_grouped["group"] == "G1"].copy()
        feat_g1 = build_features_for_group(sub_g1, "G1", cfg)
        model_g1 = build_targets(feat_g1, cfg)
        _, test_g1 = temporal_split(model_g1, cfg)
        log.info("Evaluating G1 models on G1 test data (n=%d)...", len(test_g1))
        g1_results = evaluate_group_test(test_g1, "G1", models_g1, fill_g1, feat_names_g1)

    if args.group in (None, "G2"):
        log.info("Preparing G2 test dataset...")
        sub_g2 = obs_grouped[obs_grouped["group"] == "G2"].copy()
        feat_g2 = build_features_for_group(sub_g2, "G2", cfg)
        model_g2 = build_targets(feat_g2, cfg)
        _, test_g2 = temporal_split(model_g2, cfg)
        log.info("Evaluating G2 models on G2 test data (n=%d)...", len(test_g2))
        g2_results = evaluate_group_test(test_g2, "G2", models_g2, fill_g2, feat_names_g2)

    if not args.no_cross and args.group is None:
        log.info("Running cross-evaluation...")
        cross_results = cross_evaluate(
            obs_grouped, cfg,
            models_g1, models_g2, fill_g1, fill_g2,
            feat_names_g1, feat_names_g2,
        )

    baseline_metrics = load_v8_baseline_metrics(cfg)

    print_full_report(g1_results, g2_results, cross_results, baseline_metrics)


if __name__ == "__main__":
    main()
