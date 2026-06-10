#!/usr/bin/env python3
"""
Pool Predictive Maintenance Pipeline - V4 Benchmark Suite
=========================================================

V4 keeps the V3 dataset preparation, feature engineering, target definitions,
preprocessing, and temporal split intact. Only the modeling and evaluation
stage is expanded into a fair multi-algorithm benchmark.

Primary outputs are written to outputs_v4/.
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import random
import re
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

warnings.filterwarnings("ignore")

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-swimming-pool-v4")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC, SVR
from sklearn.utils.class_weight import compute_sample_weight

# pyrefly: ignore [missing-import]
import joblib

# pyrefly: ignore [missing-import]
import matplotlib

matplotlib.use("Agg")
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback for minimal environments
    tqdm = None

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None

try:
    # pyrefly: ignore [missing-import]
    import shap
except Exception:  # pragma: no cover - optional dependency
    shap = None

try:
    # pyrefly: ignore [missing-import]
    import xgboost as xgb
except Exception:  # pragma: no cover - optional dependency
    xgb = None

try:
    # pyrefly: ignore [missing-import]
    import lightgbm as lgb
except Exception:  # pragma: no cover - optional dependency
    lgb = None

try:
    # pyrefly: ignore [missing-import]
    from catboost import CatBoostClassifier, CatBoostRegressor
except Exception:  # pragma: no cover - optional dependency
    CatBoostClassifier = None
    CatBoostRegressor = None


# ============================================================================
# REGULATORY CONSTANTS (Real Decreto 742/2013)
# ============================================================================

REG_CHLORINE_MIN = 0.5
REG_CHLORINE_IDEAL_MAX = 2.0
REG_CHLORINE_CLOSE = 5.0
REG_PH_MIN = 7.2
REG_PH_MAX = 8.0
REG_TURBIDITY_MAX = 5.0

PH_IDEAL = 7.2
CHLORINE_IDEAL = 1.25


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass(frozen=True)
class Config:
    raw_csv: str = "data/merged_pool_data_2017_2022.csv"
    output_dir: str = "outputs_v4"
    merge_tolerance_days: int = 14
    random_seed: int = 42
    stability_seeds: tuple[int, ...] = (42, 7, 21, 84, 2026)
    n_jobs: int = -1

    regression_targets: tuple[str, ...] = (
        "days_to_next_visit",
        "target_ph_next",
        "target_chlorine_next",
        "target_turbidity_next",
    )
    classification_targets: tuple[str, ...] = ("chlorine_breach_next",)

    # Keep model sizes practical while still representing each family.
    tree_n_estimators: int = 300
    boosting_n_estimators: int = 500
    gradient_boosting_n_estimators: int = 300
    catboost_iterations: int = 500
    svm_max_iter: int = 20_000

    # V3 used safety-aware weighting for visit timing. V4 applies it uniformly
    # to every regressor for this target to preserve fair comparison.
    use_visit_timing_sample_weight: bool = True
    use_balanced_classification_weights: bool = True

    shap_enabled: bool = True
    shap_sample_size: int = 1_000
    shap_max_display: int = 20
    plot_sample_size: int = 8_000
    stability_enabled: bool = True


CONFIG = Config()


POSITIONAL_RENAME = {
    0: "pool_id",
    1: "community_name",
    2: "reading_date",
    3: "technician",
    4: "ph",
    5: "turbidity",
    6: "free_chlorine",
    7: "_sep1",
    8: "sunscreen_abuse",
    9: "ph_pump_flow_rate",
    10: "hypochlorite_pump_flow_rate",
    11: "motor_flow_rate",
    12: "filter_diameter",
    13: "filter_count",
    14: "motor_count",
    15: "pool_heated",
    16: "pool_community",
    17: "pool_skimmer",
    18: "pool_overflow",
    19: "pool_outdoor",
    20: "pool_oval",
    21: "pool_private",
    22: "pool_public",
    23: "pool_rectangular_0714",
    24: "pool_rectangular_07",
    25: "pool_round",
    26: "pool_surface_m2",
    27: "vegetation_contamination",
    28: "pool_volume_m3",
    29: "deck_grass",
    30: "deck_mixed",
    31: "deck_paved",
    32: "_sep2",
    33: "ops_technician",
    34: "ops_date",
    35: "ph_dosing_hours",
    36: "daily_filtration_hours",
    37: "ph_dosing_pct",
    38: "filter_wash_rinse_time",
    39: "hypochlorite_dosing_hours",
    40: "hypochlorite_dosing_pct",
    41: "water_temperature",
    42: "_sep3",
    43: "prod_technician",
    44: "prod_date",
    45: "prod_t500_qp",
    46: "prod_alboral_tablets_250g",
    47: "prod_flovil_tablets",
    48: "prod_hypo_jugs_20kg",
    49: "prod_hypo_gr_chloryte",
    50: "prod_hypo_granular_xaka",
    51: "prod_hypo_sticks_bayrol",
    52: "prod_hypo_tab_ritocal",
    53: "prod_hypo_tablets_200g_qp",
    54: "prod_hypo_tablets_xaka",
    55: "prod_ph_minus_granular_6kg",
    56: "prod_ph_minus_liquid_13_5kg",
    57: "prod_ph_minus_liquid_27kg",
    58: "prod_protect_shine",
    59: "prod_sg_xaka_agonet",
    60: "prod_superklar",
}


@dataclass
class OutputPaths:
    root: Path
    plots: Path
    models: Path
    shap: Path
    errors: Path


@dataclass
class PreparedData:
    df_master: pd.DataFrame
    df_model: pd.DataFrame
    df_model_wq: pd.DataFrame
    df_train: pd.DataFrame
    df_test: pd.DataFrame
    df_train_wq: pd.DataFrame
    df_test_wq: pd.DataFrame
    X_train: np.ndarray
    X_test: np.ndarray
    X_train_wq: np.ndarray
    X_test_wq: np.ndarray
    feature_names: list[str]
    all_numeric_features: list[str]
    categorical_features: list[str]
    fill_values: dict[str, float]
    monthly_medians: dict[int, float]
    cutoff_date: pd.Timestamp
    preprocessor: ColumnTransformer
    backfill_summary: dict[str, dict[str, Any]]


@dataclass
class ModelSpec:
    name: str
    family: str
    factory: Callable[[int], Any]
    shap_eligible: bool = False


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def ensure_output_dirs(config: Config) -> OutputPaths:
    root = Path(config.output_dir)
    paths = OutputPaths(
        root=root,
        plots=root / "plots",
        models=root / "models",
        shap=root / "shap",
        errors=root / "errors",
    )
    for path in [paths.root, paths.plots, paths.models, paths.shap, paths.errors]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def setup_logging(paths: OutputPaths) -> logging.Logger:
    logger = logging.getLogger("pipeline_v4")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(paths.root / "pipeline_v4.log", mode="w")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def iter_progress(iterable: Any, **kwargs: Any) -> Any:
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def safe_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def clean_pool_id(value: Any) -> Any:
    if pd.isna(value):
        return np.nan
    return " ".join(str(value).strip().lower().split())


def parse_date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="mixed", dayfirst=True, errors="coerce")


def sanitize_filename(value: str) -> str:
    value = value.lower().replace("%", "pct")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def current_rss_mb() -> float:
    if psutil is None:
        return float("nan")
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)


def finite_or_nan(value: Any) -> float:
    try:
        value_float = float(value)
    except Exception:
        return float("nan")
    if math.isfinite(value_float):
        return value_float
    return float("nan")


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.maximum(np.abs(y_true), 1e-9)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)))


def percent_within(y_true: np.ndarray, y_pred: np.ndarray, pct: float) -> float:
    denominator = np.maximum(np.abs(y_true), 1e-9)
    relative_error = np.abs(y_true - y_pred) / denominator
    return float(np.mean(relative_error <= pct))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    abs_error = np.abs(residual)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "median_absolute_error": float(median_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "explained_variance": float(explained_variance_score(y_true, y_pred)),
        "mape": safe_mape(y_true, y_pred),
        "within_5_pct": percent_within(y_true, y_pred, 0.05),
        "within_10_pct": percent_within(y_true, y_pred, 0.10),
        "within_20_pct": percent_within(y_true, y_pred, 0.20),
        "error_mean": float(np.mean(residual)),
        "error_std": float(np.std(residual)),
        "absolute_error_p50": float(np.percentile(abs_error, 50)),
        "absolute_error_p90": float(np.percentile(abs_error, 90)),
        "absolute_error_p95": float(np.percentile(abs_error, 95)),
        "absolute_error_max": float(np.max(abs_error)),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    except Exception:
        metrics["pr_auc"] = float("nan")

    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return metrics


def fit_estimator(
    estimator: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> Any:
    if sample_weight is None:
        return estimator.fit(X_train, y_train)
    try:
        return estimator.fit(X_train, y_train, sample_weight=sample_weight)
    except TypeError:
        logging.getLogger("pipeline_v4").warning(
            "%s does not accept sample_weight; fitting without weights.",
            estimator.__class__.__name__,
        )
        return estimator.fit(X_train, y_train)


def predict_score(estimator: Any, X_test: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(X_test)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]
        return probabilities.ravel()
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X_test)).ravel()
    return np.asarray(estimator.predict(X_test)).ravel()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows available._"
    view = df.copy()
    if columns is not None:
        view = view[[col for col in columns if col in view.columns]]
    if max_rows is not None:
        view = view.head(max_rows)
    view = view.fillna("")
    header = "| " + " | ".join(view.columns.astype(str)) + " |"
    separator = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = []
    for _, row in view.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in view.columns) + " |")
    return "\n".join([header, separator] + rows)


# ============================================================================
# V3 DATASET PREPARATION - PRESERVED
# ============================================================================


def make_pool_type(row: pd.Series) -> str:
    parts: list[str] = []
    if row.get("pool_heated", 0):
        parts.append("heated")
    if row.get("pool_outdoor", 0):
        parts.append("outdoor")
    if row.get("pool_community", 0):
        parts.append("community")
    if row.get("pool_private", 0):
        parts.append("private")
    if row.get("pool_public", 0):
        parts.append("public")
    return "_".join(parts) if parts else "unknown"


def make_deck_type(row: pd.Series) -> str:
    grass = safe_float(pd.Series([row.get("deck_grass", 0)])).iloc[0] or 0
    paved = safe_float(pd.Series([row.get("deck_paved", 0)])).iloc[0] or 0
    mixed = safe_float(pd.Series([row.get("deck_mixed", 0)])).iloc[0] or 0
    if mixed > 0:
        return "mixed"
    if grass > 0 and paved > 0:
        return "mixed"
    if grass > 0:
        return "grass"
    if paved > 0:
        return "paved"
    return "unknown"


def merge_asof_by_pool(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_date: str,
    right_date: str,
    right_cols: list[str],
    tolerance: pd.Timedelta,
) -> pd.DataFrame:
    merged_parts: list[pd.DataFrame] = []
    for pool_id in df_left["pool_id"].unique():
        left_pool = df_left[df_left["pool_id"] == pool_id].copy()
        right_pool = df_right[df_right["pool_id"] == pool_id].copy()
        if len(right_pool) == 0:
            for col in right_cols:
                left_pool[col] = np.nan
            merged_parts.append(left_pool)
            continue
        left_pool = left_pool.sort_values(left_date)
        right_pool = right_pool.sort_values(right_date)
        merged = pd.merge_asof(
            left_pool,
            right_pool[right_cols + [right_date]],
            left_on=left_date,
            right_on=right_date,
            direction="backward",
            tolerance=tolerance,
        )
        merged_parts.append(merged)
    return pd.concat(merged_parts, ignore_index=True)


def consecutive_clean(series: np.ndarray) -> list[int]:
    result: list[int] = []
    count = 0
    for value in series:
        if value == 0:
            count += 1
        else:
            count = 0
        result.append(count)
    return result


def prepare_dataset(config: Config, paths: OutputPaths, logger: logging.Logger) -> PreparedData:
    logger.info("Loading raw data from %s", config.raw_csv)
    raw_path = Path(config.raw_csv)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {raw_path}")

    df = pd.read_csv(raw_path)
    df.columns = [POSITIONAL_RENAME.get(i, f"unknown_{i}") for i in range(len(df.columns))]
    df = df.drop(columns=[col for col in df.columns if col.startswith("_sep")])
    df["pool_id"] = df["pool_id"].apply(clean_pool_id)
    df["pool_id"] = df["pool_id"].ffill()
    df["community_name"] = df["community_name"].ffill()
    logger.info("Raw data loaded: %s rows x %s columns", df.shape[0], df.shape[1])

    reading_cols = [
        "pool_id",
        "community_name",
        "reading_date",
        "technician",
        "ph",
        "turbidity",
        "free_chlorine",
        "sunscreen_abuse",
        "ph_pump_flow_rate",
        "hypochlorite_pump_flow_rate",
        "motor_flow_rate",
        "filter_diameter",
        "filter_count",
        "motor_count",
        "pool_heated",
        "pool_community",
        "pool_skimmer",
        "pool_overflow",
        "pool_outdoor",
        "pool_oval",
        "pool_private",
        "pool_public",
        "pool_rectangular_0714",
        "pool_rectangular_07",
        "pool_round",
        "pool_surface_m2",
        "vegetation_contamination",
        "pool_volume_m3",
        "deck_grass",
        "deck_mixed",
        "deck_paved",
    ]

    ops_cols = [
        "ops_technician",
        "ops_date",
        "ph_dosing_hours",
        "daily_filtration_hours",
        "ph_dosing_pct",
        "filter_wash_rinse_time",
        "hypochlorite_dosing_hours",
        "hypochlorite_dosing_pct",
        "water_temperature",
    ]

    prod_cols = [
        "prod_technician",
        "prod_date",
        "prod_t500_qp",
        "prod_alboral_tablets_250g",
        "prod_flovil_tablets",
        "prod_hypo_jugs_20kg",
        "prod_hypo_gr_chloryte",
        "prod_hypo_granular_xaka",
        "prod_hypo_sticks_bayrol",
        "prod_hypo_tab_ritocal",
        "prod_hypo_tablets_200g_qp",
        "prod_hypo_tablets_xaka",
        "prod_ph_minus_granular_6kg",
        "prod_ph_minus_liquid_13_5kg",
        "prod_ph_minus_liquid_27kg",
        "prod_protect_shine",
        "prod_sg_xaka_agonet",
        "prod_superklar",
    ]

    df_readings = df[reading_cols].copy()
    df_readings = df_readings.dropna(subset=["reading_date"])

    df_operations = df[["pool_id"] + ops_cols].copy()
    df_operations = df_operations.dropna(subset=["ops_date"])
    df_operations = df_operations.dropna(
        subset=["ph_dosing_hours", "daily_filtration_hours", "water_temperature"],
        how="all",
    )

    df_products = df[["pool_id"] + prod_cols].copy()
    df_products = df_products.dropna(subset=["prod_date"])

    logger.info(
        "Separated tables: readings=%s, operations=%s, products=%s",
        df_readings.shape,
        df_operations.shape,
        df_products.shape,
    )

    # Readings cleaning and regulatory flags.
    df_readings["reading_date"] = parse_date_series(df_readings["reading_date"])
    for col in ["ph", "turbidity", "free_chlorine"]:
        df_readings[col] = safe_float(df_readings[col])

    df_readings["ph_outlier"] = ~df_readings["ph"].between(REG_PH_MIN, REG_PH_MAX) & df_readings["ph"].notna()
    df_readings["chlorine_outlier"] = (
        (df_readings["free_chlorine"] < REG_CHLORINE_MIN)
        | (df_readings["free_chlorine"] > REG_CHLORINE_CLOSE)
    ) & df_readings["free_chlorine"].notna()
    df_readings["turbidity_outlier"] = (
        df_readings["turbidity"] > REG_TURBIDITY_MAX
    ) & df_readings["turbidity"].notna()

    df_readings["ph_breach"] = ~df_readings["ph"].between(REG_PH_MIN, REG_PH_MAX) & df_readings["ph"].notna()
    df_readings["chlorine_breach"] = (
        (df_readings["free_chlorine"] < REG_CHLORINE_MIN)
        | (df_readings["free_chlorine"] > REG_CHLORINE_CLOSE)
    ) & df_readings["free_chlorine"].notna()
    df_readings["chlorine_low"] = (
        df_readings["free_chlorine"] < REG_CHLORINE_MIN
    ) & df_readings["free_chlorine"].notna()
    df_readings["chlorine_over_ideal"] = (
        df_readings["free_chlorine"] > REG_CHLORINE_IDEAL_MAX
    ) & df_readings["free_chlorine"].notna()
    df_readings["turbidity_breach"] = (
        df_readings["turbidity"] > REG_TURBIDITY_MAX
    ) & df_readings["turbidity"].notna()
    df_readings["any_breach"] = (
        df_readings["ph_breach"] | df_readings["chlorine_breach"] | df_readings["turbidity_breach"]
    )

    pool_type_flags = [
        "pool_heated",
        "pool_community",
        "pool_skimmer",
        "pool_overflow",
        "pool_outdoor",
        "pool_oval",
        "pool_private",
        "pool_public",
        "pool_rectangular_0714",
        "pool_rectangular_07",
        "pool_round",
    ]
    for col in pool_type_flags:
        df_readings[col] = safe_float(df_readings[col]).fillna(0).astype(int)

    for col in ["pool_surface_m2", "pool_volume_m3", "filter_diameter", "filter_count", "motor_count"]:
        df_readings[col] = safe_float(df_readings[col])

    df_readings["pool_type"] = df_readings.apply(make_pool_type, axis=1)
    df_readings["deck_type"] = df_readings.apply(make_deck_type, axis=1)

    before = len(df_readings)
    df_readings = df_readings.drop_duplicates(subset=["pool_id", "reading_date"])
    df_readings = df_readings.sort_values(["pool_id", "reading_date"]).reset_index(drop=True)
    logger.info("Readings deduplicated: %s -> %s", before, len(df_readings))

    # Operations cleaning.
    df_operations["ops_date"] = parse_date_series(df_operations["ops_date"])
    ops_numeric = [
        "ph_dosing_hours",
        "daily_filtration_hours",
        "ph_dosing_pct",
        "filter_wash_rinse_time",
        "hypochlorite_dosing_hours",
        "hypochlorite_dosing_pct",
        "water_temperature",
    ]
    for col in ops_numeric:
        df_operations[col] = safe_float(df_operations[col])
    df_operations = df_operations.groupby(["pool_id", "ops_date"], as_index=False)[ops_numeric].mean()
    df_operations = df_operations.sort_values(["pool_id", "ops_date"]).reset_index(drop=True)

    # Product cleaning.
    df_products["prod_date"] = parse_date_series(df_products["prod_date"])
    prod_value_cols = [col for col in prod_cols if col not in ["prod_technician", "prod_date"]]
    for col in prod_value_cols:
        df_products[col] = safe_float(df_products[col]).fillna(0)

    hypo_cols = [col for col in prod_value_cols if "hypo" in col.lower()]
    ph_minus_cols = [col for col in prod_value_cols if "ph_minus" in col.lower()]
    flocculant_cols = [
        col
        for col in [
            "prod_flovil_tablets",
            "prod_superklar",
            "prod_sg_xaka_agonet",
            "prod_alboral_tablets_250g",
        ]
        if col in df_products.columns
    ]
    df_products["total_chlorine_product"] = df_products[hypo_cols].sum(axis=1)
    df_products["total_ph_minus_product"] = df_products[ph_minus_cols].sum(axis=1)
    df_products["total_flocculant_product"] = df_products[flocculant_cols].sum(axis=1)

    agg_cols = prod_value_cols + [
        "total_chlorine_product",
        "total_ph_minus_product",
        "total_flocculant_product",
    ]
    df_products = df_products.groupby(["pool_id", "prod_date"], as_index=False)[agg_cols].max()
    df_products = df_products.sort_values(["pool_id", "prod_date"]).reset_index(drop=True)

    reading_pools = set(df_readings["pool_id"].dropna().unique())
    product_pools = set(df_products["pool_id"].dropna().unique())
    orphan_pools = product_pools - reading_pools
    if orphan_pools:
        logger.info("Dropping %s orphan product pools with no readings.", len(orphan_pools))
        df_products = df_products[df_products["pool_id"].isin(reading_pools)]

    # V3 static pool data backfill.
    backfill_summary: dict[str, dict[str, Any]] = {}
    numeric_static = ["pool_volume_m3", "pool_surface_m2", "filter_diameter", "filter_count", "motor_count"]
    for col in numeric_static:
        before_fill = int(df_readings[col].notna().sum())
        pool_vals = df_readings.groupby("pool_id")[col].apply(
            lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan
        )
        pool_vals_clean = pool_vals.dropna()
        fleet_median = pool_vals_clean.median() if len(pool_vals_clean) > 0 else 0.0
        pools_with_data = len(pool_vals_clean)
        pools_without = len(pool_vals) - pools_with_data
        df_readings[col] = df_readings["pool_id"].map(pool_vals).fillna(fleet_median)
        after_fill = int(df_readings[col].notna().sum())
        backfill_summary[col] = {
            "pools_with_original_data": int(pools_with_data),
            "pools_filled_with_median": int(pools_without),
            "fleet_median": round(float(fleet_median), 2),
            "rows_before": before_fill,
            "rows_after": after_fill,
        }

    flag_static = [
        "pool_heated",
        "pool_community",
        "pool_skimmer",
        "pool_overflow",
        "pool_outdoor",
        "pool_oval",
        "pool_private",
        "pool_public",
        "pool_rectangular_0714",
        "pool_rectangular_07",
        "pool_round",
        "vegetation_contamination",
    ]
    for col in flag_static:
        pool_vals = df_readings.groupby("pool_id")[col].apply(lambda x: x.max() if x.notna().any() else 0)
        df_readings[col] = df_readings["pool_id"].map(pool_vals).fillna(0).astype(int)

    for col in ["deck_grass", "deck_mixed", "deck_paved"]:
        if col in df_readings.columns:
            pool_vals = df_readings.groupby("pool_id")[col].apply(lambda x: x.max() if x.notna().any() else 0)
            df_readings[col] = df_readings["pool_id"].map(pool_vals).fillna(0)

    df_readings["pool_type"] = df_readings.apply(make_pool_type, axis=1)
    df_readings["deck_type"] = df_readings.apply(make_deck_type, axis=1)

    # Merge operations and products into master dataset using V3 merge-asof logic.
    tolerance = pd.Timedelta(f"{config.merge_tolerance_days}D")
    ops_merge_cols = [col for col in df_operations.columns if col not in ["pool_id"]]
    ops_value_cols = [col for col in ops_merge_cols if col != "ops_date"]
    df_master = merge_asof_by_pool(
        df_readings,
        df_operations,
        "reading_date",
        "ops_date",
        ops_value_cols,
        tolerance,
    )

    prod_merge_cols = [col for col in df_products.columns if col not in ["pool_id"]]
    prod_value_cols_merge = [col for col in prod_merge_cols if col != "prod_date"]
    df_master = merge_asof_by_pool(
        df_master,
        df_products,
        "reading_date",
        "prod_date",
        prod_value_cols_merge,
        tolerance,
    )
    df_master = df_master.sort_values(["pool_id", "reading_date"]).reset_index(drop=True)

    # V3 feature engineering - no changes.
    for col, prefix in [("ph", "ph"), ("free_chlorine", "chlorine"), ("turbidity", "turbidity")]:
        df_master[f"{prefix}_lag1"] = df_master.groupby("pool_id")[col].shift(1)
        df_master[f"{prefix}_lag2"] = df_master.groupby("pool_id")[col].shift(2)

    for col, prefix in [("ph", "ph"), ("free_chlorine", "chlorine"), ("turbidity", "turbidity")]:
        df_master[f"{prefix}_roll3_mean"] = df_master.groupby("pool_id")[col].transform(
            lambda x: x.rolling(window=3, min_periods=2).mean()
        )
        if prefix != "turbidity":
            df_master[f"{prefix}_roll3_std"] = df_master.groupby("pool_id")[col].transform(
                lambda x: x.rolling(window=3, min_periods=2).std()
            )

    df_master["days_since_last_visit"] = df_master.groupby("pool_id")["reading_date"].diff().dt.days
    df_master["visit_day_of_week"] = df_master["reading_date"].dt.dayofweek
    df_master["visit_month"] = df_master["reading_date"].dt.month
    df_master["visit_is_summer"] = df_master["visit_month"].isin([6, 7, 8, 9]).astype(int)

    df_master["ph_deviation"] = (df_master["ph"] - PH_IDEAL).abs()
    df_master["chlorine_deficit"] = (REG_CHLORINE_MIN - df_master["free_chlorine"]).clip(lower=0)
    df_master["last_total_chlorine_applied"] = df_master["total_chlorine_product"].fillna(0)

    df_master["chlorine_headroom_low"] = df_master["free_chlorine"] - REG_CHLORINE_MIN
    df_master["chlorine_headroom_high"] = REG_CHLORINE_CLOSE - df_master["free_chlorine"]
    df_master["ph_headroom_low"] = df_master["ph"] - REG_PH_MIN
    df_master["ph_headroom_high"] = REG_PH_MAX - df_master["ph"]
    df_master["turbidity_headroom"] = REG_TURBIDITY_MAX - df_master["turbidity"]
    df_master["min_headroom"] = df_master[
        [
            "chlorine_headroom_low",
            "chlorine_headroom_high",
            "ph_headroom_low",
            "ph_headroom_high",
            "turbidity_headroom",
        ]
    ].min(axis=1)

    df_master["ph_trend"] = df_master["ph"] - df_master["ph_lag1"]
    df_master["chlorine_trend"] = df_master["free_chlorine"] - df_master["chlorine_lag1"]
    df_master["turbidity_trend"] = df_master["turbidity"] - df_master["turbidity_lag1"]

    df_master["ph_rate_per_day"] = df_master["ph_trend"] / df_master["days_since_last_visit"].replace(0, np.nan)
    df_master["chlorine_rate_per_day"] = df_master["chlorine_trend"] / df_master[
        "days_since_last_visit"
    ].replace(0, np.nan)
    df_master["turbidity_rate_per_day"] = df_master["turbidity_trend"] / df_master[
        "days_since_last_visit"
    ].replace(0, np.nan)

    df_master["current_any_breach"] = df_master["any_breach"].astype(int)
    df_master["current_ph_breach"] = df_master["ph_breach"].astype(int)
    df_master["current_chlorine_breach"] = df_master["chlorine_breach"].astype(int)
    df_master["consecutive_clean_visits"] = df_master.groupby("pool_id")["current_any_breach"].transform(
        lambda x: consecutive_clean(x.values)
    )
    df_master["breach_rate_last5"] = df_master.groupby("pool_id")["current_any_breach"].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )

    lag_cols = [
        "ph_lag1",
        "ph_lag2",
        "chlorine_lag1",
        "chlorine_lag2",
        "turbidity_lag1",
        "turbidity_lag2",
    ]
    before_lag_drop = len(df_master)
    df_master = df_master.dropna(subset=lag_cols)
    logger.info("Dropped %s rows with null V3 lag features.", before_lag_drop - len(df_master))

    ph_correction = np.where(
        df_master["ph"] <= 7.5,
        1.0,
        1.0 - 0.5 * ((df_master["ph"] - 7.5) / 0.5),
    )
    df_master["cl_effectiveness_index"] = df_master["free_chlorine"] * np.clip(ph_correction, 0.1, 1.0)
    df_master["chlorine_dose_per_m3"] = df_master["last_total_chlorine_applied"] / df_master["pool_volume_m3"]
    df_master["ph_minus_dose_per_m3"] = df_master["total_ph_minus_product"] / df_master["pool_volume_m3"]
    df_master["chlorine_decay_per_m3"] = df_master["chlorine_rate_per_day"] / df_master["pool_volume_m3"]
    df_master["pool_visit_number"] = df_master.groupby("pool_id").cumcount()

    # V3 targets - no changes.
    df_master["next_reading_date"] = df_master.groupby("pool_id")["reading_date"].shift(-1)
    df_master["days_to_next_visit"] = (
        df_master["next_reading_date"] - df_master["reading_date"]
    ).dt.days
    df_master["target_ph_next"] = df_master.groupby("pool_id")["ph"].shift(-1)
    df_master["target_chlorine_next"] = df_master.groupby("pool_id")["free_chlorine"].shift(-1)
    df_master["target_turbidity_next"] = df_master.groupby("pool_id")["turbidity"].shift(-1)
    df_master["ph_breach_next"] = (
        (df_master["target_ph_next"] < REG_PH_MIN) | (df_master["target_ph_next"] > REG_PH_MAX)
    ) & df_master["target_ph_next"].notna()
    df_master["chlorine_breach_next"] = (
        (df_master["target_chlorine_next"] < REG_CHLORINE_MIN)
        | (df_master["target_chlorine_next"] > REG_CHLORINE_CLOSE)
    ) & df_master["target_chlorine_next"].notna()
    df_master["turbidity_breach_next"] = (
        df_master["target_turbidity_next"] > REG_TURBIDITY_MAX
    ) & df_master["target_turbidity_next"].notna()
    df_master["any_breach_next"] = (
        df_master["ph_breach_next"]
        | df_master["chlorine_breach_next"]
        | df_master["turbidity_breach_next"]
    )

    df_model = df_master.dropna(
        subset=["days_to_next_visit", "ph", "free_chlorine", "turbidity"]
    ).copy()
    before_interval_filter = len(df_model)
    df_model = df_model[df_model["days_to_next_visit"] <= 60].copy()
    logger.info(
        "Removed %s rows with >60 day intervals.",
        before_interval_filter - len(df_model),
    )

    monthly_medians = df_model.groupby("visit_month")["days_to_next_visit"].median()
    df_model["seasonal_baseline_days"] = df_model["visit_month"].map(monthly_medians)
    df_model["visit_deviation"] = df_model["days_to_next_visit"] - df_model["seasonal_baseline_days"]
    df_model_wq = df_model.dropna(
        subset=["target_ph_next", "target_chlorine_next", "target_turbidity_next"]
    ).copy()

    # V3 feature selection - no changes.
    static_features = ["pool_surface_m2", "pool_volume_m3", "filter_diameter", "filter_count", "motor_count"]
    categorical_features = ["pool_type", "deck_type"]
    lag_features = ["ph_lag1", "ph_lag2", "chlorine_lag1", "chlorine_lag2", "turbidity_lag1", "turbidity_lag2"]
    rolling_features = [
        "ph_roll3_mean",
        "ph_roll3_std",
        "chlorine_roll3_mean",
        "chlorine_roll3_std",
        "turbidity_roll3_mean",
    ]
    ops_features = ["daily_filtration_hours", "water_temperature", "ph_dosing_pct", "hypochlorite_dosing_pct"]
    product_features = ["last_total_chlorine_applied", "total_ph_minus_product"]
    temporal_features = ["days_since_last_visit", "visit_month", "visit_is_summer", "visit_day_of_week"]
    headroom_features = [
        "chlorine_headroom_low",
        "chlorine_headroom_high",
        "ph_headroom_low",
        "ph_headroom_high",
        "turbidity_headroom",
        "min_headroom",
    ]
    trend_features = [
        "ph_trend",
        "chlorine_trend",
        "turbidity_trend",
        "ph_rate_per_day",
        "chlorine_rate_per_day",
        "turbidity_rate_per_day",
    ]
    breach_history_features = [
        "consecutive_clean_visits",
        "breach_rate_last5",
        "current_any_breach",
        "current_ph_breach",
        "current_chlorine_breach",
    ]
    chemistry_features = ["ph_deviation", "chlorine_deficit"]
    v3_features = [
        "cl_effectiveness_index",
        "chlorine_dose_per_m3",
        "ph_minus_dose_per_m3",
        "chlorine_decay_per_m3",
        "pool_visit_number",
    ]

    all_numeric_features = (
        static_features
        + lag_features
        + rolling_features
        + ops_features
        + product_features
        + temporal_features
        + headroom_features
        + trend_features
        + breach_history_features
        + chemistry_features
        + v3_features
    )
    null_rates = df_model[all_numeric_features].isnull().mean()
    high_null = null_rates[null_rates > 0.5].index.tolist()
    if high_null:
        logger.info("Dropping V3 high-null features: %s", high_null)
        all_numeric_features = [feature for feature in all_numeric_features if feature not in high_null]

    cutoff_date = df_model["reading_date"].quantile(0.8)
    df_train = df_model[df_model["reading_date"] < cutoff_date].copy()
    df_test = df_model[df_model["reading_date"] >= cutoff_date].copy()
    df_train_wq = df_model_wq[df_model_wq["reading_date"] < cutoff_date].copy()
    df_test_wq = df_model_wq[df_model_wq["reading_date"] >= cutoff_date].copy()

    fill_values: dict[str, float] = {}
    for col in all_numeric_features:
        median_val = df_train[col].median()
        fill_values[col] = float(median_val) if pd.notna(median_val) else 0.0

    for dfx in [df_train, df_test, df_train_wq, df_test_wq]:
        dfx[all_numeric_features] = dfx[all_numeric_features].fillna(fill_values)
        for col in categorical_features:
            dfx[col] = dfx[col].fillna("unknown")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("num", "passthrough", all_numeric_features),
        ],
        remainder="drop",
    )

    model_features = categorical_features + all_numeric_features
    X_train = preprocessor.fit_transform(df_train[model_features])
    X_test = preprocessor.transform(df_test[model_features])
    X_train_wq = preprocessor.transform(df_train_wq[model_features])
    X_test_wq = preprocessor.transform(df_test_wq[model_features])
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features).tolist()
    feature_names = cat_names + all_numeric_features

    logger.info("Temporal cutoff: %s", cutoff_date)
    logger.info("Train/test rows: visit=%s/%s, water_quality=%s/%s", len(df_train), len(df_test), len(df_train_wq), len(df_test_wq))
    logger.info("Encoded feature matrix: X_train=%s, X_test=%s, features=%s", X_train.shape, X_test.shape, len(feature_names))

    with (paths.models / "preprocessor_v4.pkl").open("wb") as handle:
        pickle.dump(preprocessor, handle)

    monthly_medians_dict = {int(key): float(value) for key, value in monthly_medians.items()}
    save_json(
        paths.models / "inference_config_v4.json",
        {
            "fill_values": fill_values,
            "all_numeric_features": all_numeric_features,
            "categorical_features": categorical_features,
            "feature_names": feature_names,
            "monthly_medians": monthly_medians_dict,
            "cutoff_date": str(cutoff_date),
            "regulatory_thresholds": {
                "chlorine_min": REG_CHLORINE_MIN,
                "chlorine_ideal_max": REG_CHLORINE_IDEAL_MAX,
                "chlorine_close": REG_CHLORINE_CLOSE,
                "ph_min": REG_PH_MIN,
                "ph_max": REG_PH_MAX,
                "turbidity_max": REG_TURBIDITY_MAX,
            },
        },
    )

    df_master.to_csv(paths.root / "master_dataset_v4.csv", index=False)

    return PreparedData(
        df_master=df_master,
        df_model=df_model,
        df_model_wq=df_model_wq,
        df_train=df_train,
        df_test=df_test,
        df_train_wq=df_train_wq,
        df_test_wq=df_test_wq,
        X_train=np.asarray(X_train),
        X_test=np.asarray(X_test),
        X_train_wq=np.asarray(X_train_wq),
        X_test_wq=np.asarray(X_test_wq),
        feature_names=feature_names,
        all_numeric_features=all_numeric_features,
        categorical_features=categorical_features,
        fill_values=fill_values,
        monthly_medians=monthly_medians_dict,
        cutoff_date=cutoff_date,
        preprocessor=preprocessor,
        backfill_summary=backfill_summary,
    )


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================


def regression_model_specs(config: Config) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    if xgb is not None:
        specs.append(
            ModelSpec(
                "XGBoost Regressor",
                "gradient_boosted_trees",
                lambda seed: xgb.XGBRegressor(
                    n_estimators=config.boosting_n_estimators,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=seed,
                    n_jobs=config.n_jobs,
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    verbosity=0,
                ),
                shap_eligible=True,
            )
        )
    specs.extend(
        [
            ModelSpec(
                "Random Forest Regressor",
                "bagged_trees",
                lambda seed: RandomForestRegressor(
                    n_estimators=config.tree_n_estimators,
                    random_state=seed,
                    n_jobs=config.n_jobs,
                    min_samples_leaf=1,
                ),
                shap_eligible=True,
            ),
            ModelSpec(
                "Extra Trees Regressor",
                "extremely_randomized_trees",
                lambda seed: ExtraTreesRegressor(
                    n_estimators=config.tree_n_estimators,
                    random_state=seed,
                    n_jobs=config.n_jobs,
                    min_samples_leaf=1,
                ),
                shap_eligible=True,
            ),
            ModelSpec(
                "Gradient Boosting Regressor",
                "gradient_boosted_trees",
                lambda seed: GradientBoostingRegressor(
                    n_estimators=config.gradient_boosting_n_estimators,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=seed,
                ),
                shap_eligible=False,
            ),
        ]
    )
    if lgb is not None:
        specs.append(
            ModelSpec(
                "LightGBM Regressor",
                "histogram_gradient_boosted_trees",
                lambda seed: lgb.LGBMRegressor(
                    n_estimators=config.boosting_n_estimators,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=seed,
                    n_jobs=config.n_jobs,
                    verbose=-1,
                ),
                shap_eligible=True,
            )
        )
    if CatBoostRegressor is not None:
        specs.append(
            ModelSpec(
                "CatBoost Regressor",
                "ordered_boosted_trees",
                lambda seed: CatBoostRegressor(
                    iterations=config.catboost_iterations,
                    depth=6,
                    learning_rate=0.05,
                    loss_function="RMSE",
                    random_seed=seed,
                    verbose=False,
                    allow_writing_files=False,
                ),
                shap_eligible=True,
            )
        )
    specs.append(
        ModelSpec(
            "Support Vector Regressor",
            "kernel_method",
            lambda seed: SVR(
                kernel="rbf",
                C=10.0,
                epsilon=0.1,
                gamma="scale",
                cache_size=2048,
                max_iter=config.svm_max_iter,
            ),
            shap_eligible=False,
        )
    )
    return specs


def classification_model_specs(config: Config) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    if xgb is not None:
        specs.append(
            ModelSpec(
                "XGBoost Classifier",
                "gradient_boosted_trees",
                lambda seed: xgb.XGBClassifier(
                    n_estimators=config.boosting_n_estimators,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=seed,
                    n_jobs=config.n_jobs,
                    objective="binary:logistic",
                    eval_metric="aucpr",
                    verbosity=0,
                ),
                shap_eligible=True,
            )
        )
    specs.extend(
        [
            ModelSpec(
                "Random Forest Classifier",
                "bagged_trees",
                lambda seed: RandomForestClassifier(
                    n_estimators=config.tree_n_estimators,
                    random_state=seed,
                    n_jobs=config.n_jobs,
                    min_samples_leaf=1,
                ),
                shap_eligible=True,
            ),
            ModelSpec(
                "Extra Trees Classifier",
                "extremely_randomized_trees",
                lambda seed: ExtraTreesClassifier(
                    n_estimators=config.tree_n_estimators,
                    random_state=seed,
                    n_jobs=config.n_jobs,
                    min_samples_leaf=1,
                ),
                shap_eligible=True,
            ),
            ModelSpec(
                "Gradient Boosting Classifier",
                "gradient_boosted_trees",
                lambda seed: GradientBoostingClassifier(
                    n_estimators=config.gradient_boosting_n_estimators,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=seed,
                ),
                shap_eligible=False,
            ),
        ]
    )
    if lgb is not None:
        specs.append(
            ModelSpec(
                "LightGBM Classifier",
                "histogram_gradient_boosted_trees",
                lambda seed: lgb.LGBMClassifier(
                    n_estimators=config.boosting_n_estimators,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=seed,
                    n_jobs=config.n_jobs,
                    verbose=-1,
                ),
                shap_eligible=True,
            )
        )
    if CatBoostClassifier is not None:
        specs.append(
            ModelSpec(
                "CatBoost Classifier",
                "ordered_boosted_trees",
                lambda seed: CatBoostClassifier(
                    iterations=config.catboost_iterations,
                    depth=6,
                    learning_rate=0.05,
                    loss_function="Logloss",
                    eval_metric="PRAUC",
                    random_seed=seed,
                    verbose=False,
                    allow_writing_files=False,
                ),
                shap_eligible=True,
            )
        )
    specs.extend(
        [
            ModelSpec(
                "Support Vector Classifier",
                "kernel_method",
                lambda seed: SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    cache_size=2048,
                    max_iter=config.svm_max_iter,
                ),
                shap_eligible=False,
            ),
            ModelSpec(
                "Logistic Regression",
                "linear_model",
                lambda seed: LogisticRegression(
                    max_iter=2_000,
                    solver="lbfgs",
                    n_jobs=config.n_jobs,
                    random_state=seed,
                ),
                shap_eligible=False,
            ),
        ]
    )
    return specs


# ============================================================================
# PLOTS AND ANALYSIS
# ============================================================================


def sample_for_plot(y_true: np.ndarray, y_pred: np.ndarray, max_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(y_true) <= max_rows:
        return y_true, y_pred
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_true), size=max_rows, replace=False)
    return y_true[idx], y_pred[idx]


def save_regression_plots(
    paths: OutputPaths,
    target: str,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: Config,
) -> None:
    slug = f"{sanitize_filename(target)}__{sanitize_filename(model_name)}"
    y_plot, pred_plot = sample_for_plot(y_true, y_pred, config.plot_sample_size, config.random_seed)
    residuals = y_plot - pred_plot

    plt.figure(figsize=(8, 7))
    sns.scatterplot(x=y_plot, y=pred_plot, s=14, alpha=0.35, edgecolor=None)
    min_axis = float(min(np.min(y_plot), np.min(pred_plot)))
    max_axis = float(max(np.max(y_plot), np.max(pred_plot)))
    plt.plot([min_axis, max_axis], [min_axis, max_axis], color="black", linewidth=1.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Actual vs Predicted - {target} - {model_name}")
    plt.tight_layout()
    plt.savefig(paths.plots / f"actual_vs_predicted__{slug}.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, bins=60, kde=True)
    plt.axvline(0, color="black", linewidth=1.2)
    plt.xlabel("Residual (actual - predicted)")
    plt.ylabel("Count")
    plt.title(f"Residual Distribution - {target} - {model_name}")
    plt.tight_layout()
    plt.savefig(paths.plots / f"residual_distribution__{slug}.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pred_plot, y=residuals, s=14, alpha=0.35, edgecolor=None)
    plt.axhline(0, color="black", linewidth=1.2)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title(f"Residual Scatter - {target} - {model_name}")
    plt.tight_layout()
    plt.savefig(paths.plots / f"residual_scatter__{slug}.png", dpi=180)
    plt.close()

    denominator = np.maximum(np.abs(y_plot), 1e-9)
    pct_error = np.abs((y_plot - pred_plot) / denominator) * 100
    plt.figure(figsize=(8, 6))
    sns.histplot(np.clip(pct_error, 0, np.nanpercentile(pct_error, 99)), bins=60, kde=True)
    plt.xlabel("Absolute percentage error")
    plt.ylabel("Count")
    plt.title(f"Prediction Error Distribution - {target} - {model_name}")
    plt.tight_layout()
    plt.savefig(paths.plots / f"prediction_error_distribution__{slug}.png", dpi=180)
    plt.close()


def save_classification_plots(
    paths: OutputPaths,
    target: str,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> None:
    slug = f"{sanitize_filename(target)}__{sanitize_filename(model_name)}"

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {target} - {model_name}")
    plt.tight_layout()
    plt.savefig(paths.plots / f"confusion_matrix__{slug}.png", dpi=180)
    plt.close()

    try:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        plt.figure(figsize=(7, 6))
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {target} - {model_name}")
        plt.tight_layout()
        plt.savefig(paths.plots / f"precision_recall_curve__{slug}.png", dpi=180)
        plt.close()
    except Exception:
        logging.getLogger("pipeline_v4").exception("Could not plot precision-recall curve for %s", slug)

    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, linewidth=2)
        plt.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {target} - {model_name}")
        plt.tight_layout()
        plt.savefig(paths.plots / f"roc_curve__{slug}.png", dpi=180)
        plt.close()
    except Exception:
        logging.getLogger("pipeline_v4").exception("Could not plot ROC curve for %s", slug)


def extract_feature_importance(
    estimator: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    values: np.ndarray | None = None
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "get_feature_importance"):
        try:
            values = np.asarray(estimator.get_feature_importance(), dtype=float)
        except Exception:
            values = None
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        values = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)

    if values is None or len(values) != len(feature_names):
        return pd.DataFrame(columns=["feature", "importance"])

    importance = pd.DataFrame({"feature": feature_names, "importance": values})
    importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
    total = importance["importance"].sum()
    if total > 0:
        importance["importance_normalized"] = importance["importance"] / total
    else:
        importance["importance_normalized"] = 0.0
    return importance


def save_feature_importance_plot(
    paths: OutputPaths,
    target: str,
    model_name: str,
    importance: pd.DataFrame,
    top_n: int = 20,
) -> None:
    if importance.empty:
        return
    top = importance.head(top_n).sort_values("importance", ascending=True)
    slug = f"{sanitize_filename(target)}__{sanitize_filename(model_name)}"
    plt.figure(figsize=(9, 7))
    sns.barplot(data=top, x="importance", y="feature", color="#2d6cdf")
    plt.xlabel("Feature importance")
    plt.ylabel("")
    plt.title(f"Feature Importance - {target} - {model_name}")
    plt.tight_layout()
    plt.savefig(paths.plots / f"feature_importance__{slug}.png", dpi=180)
    plt.close()


def run_shap_analysis(
    paths: OutputPaths,
    estimator: Any,
    X_test: np.ndarray,
    feature_names: list[str],
    target: str,
    model_name: str,
    config: Config,
    logger: logging.Logger,
) -> pd.DataFrame:
    if not config.shap_enabled or shap is None:
        return pd.DataFrame()
    if len(X_test) == 0:
        return pd.DataFrame()

    sample_size = min(config.shap_sample_size, len(X_test))
    rng = np.random.default_rng(config.random_seed)
    sample_idx = rng.choice(len(X_test), size=sample_size, replace=False) if len(X_test) > sample_size else np.arange(len(X_test))
    X_sample = X_test[sample_idx]
    slug = f"{sanitize_filename(target)}__{sanitize_filename(model_name)}"

    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_array = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
        else:
            shap_array = np.asarray(shap_values)
        if shap_array.ndim == 3:
            shap_array = shap_array[:, :, 1] if shap_array.shape[2] > 1 else shap_array[:, :, 0]

        mean_abs = np.abs(shap_array).mean(axis=0)
        shap_importance = pd.DataFrame(
            {"feature": feature_names, "mean_abs_shap": mean_abs}
        ).sort_values("mean_abs_shap", ascending=False)
        shap_importance.to_csv(paths.shap / f"shap_importance__{slug}.csv", index=False)

        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_array,
            X_sample,
            feature_names=feature_names,
            max_display=config.shap_max_display,
            show=False,
        )
        plt.title(f"SHAP Summary - {target} - {model_name}")
        plt.tight_layout()
        plt.savefig(paths.shap / f"shap_summary__{slug}.png", dpi=180, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(9, 7))
        shap.summary_plot(
            shap_array,
            X_sample,
            feature_names=feature_names,
            plot_type="bar",
            max_display=config.shap_max_display,
            show=False,
        )
        plt.title(f"SHAP Feature Importance - {target} - {model_name}")
        plt.tight_layout()
        plt.savefig(paths.shap / f"shap_bar__{slug}.png", dpi=180, bbox_inches="tight")
        plt.close()
        return shap_importance
    except Exception as exc:
        logger.warning("SHAP failed for %s / %s: %s", target, model_name, exc)
        return pd.DataFrame()


def save_regression_error_analysis(
    paths: OutputPaths,
    target: str,
    model_name: str,
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    denominator = np.maximum(np.abs(y_true), 1e-9)
    errors = pd.DataFrame(
        {
            "pool_id": df_test["pool_id"].values,
            "date": df_test["reading_date"].values,
            "target": target,
            "model": model_name,
            "actual": y_true,
            "predicted": y_pred,
            "absolute_error": np.abs(y_true - y_pred),
            "percentage_error": np.abs((y_true - y_pred) / denominator) * 100,
        }
    )
    slug = f"{sanitize_filename(target)}__{sanitize_filename(model_name)}"
    errors.sort_values("absolute_error", ascending=True).head(25).to_csv(
        paths.errors / f"best_predictions__{slug}.csv", index=False
    )
    errors.sort_values("absolute_error", ascending=False).head(25).to_csv(
        paths.errors / f"worst_predictions__{slug}.csv", index=False
    )
    top100 = errors.sort_values("absolute_error", ascending=False).head(100)
    top100.to_csv(paths.errors / f"top_100_largest_errors__{slug}.csv", index=False)
    return top100


def save_model_comparison_plots(
    paths: OutputPaths,
    regression_df: pd.DataFrame,
    classification_df: pd.DataFrame,
) -> None:
    successful_reg = regression_df[regression_df["status"] == "ok"].copy()
    for target, group in successful_reg.groupby("target"):
        for metric, title in [
            ("rmse", "RMSE Comparison"),
            ("mae", "MAE Comparison"),
            ("r2", "R2 Comparison"),
            ("training_time_sec", "Training Time Comparison"),
            ("inference_time_sec", "Inference Time Comparison"),
        ]:
            plot_group = group.sort_values(metric, ascending=(metric != "r2"))
            plt.figure(figsize=(10, 6))
            sns.barplot(data=plot_group, x=metric, y="model", color="#2d6cdf")
            plt.ylabel("")
            plt.xlabel(metric)
            plt.title(f"{title} - {target}")
            plt.tight_layout()
            plt.savefig(paths.plots / f"{sanitize_filename(title)}__{sanitize_filename(target)}.png", dpi=180)
            plt.close()

    successful_clf = classification_df[classification_df["status"] == "ok"].copy()
    for target, group in successful_clf.groupby("target"):
        for metric, title in [
            ("recall", "Recall Comparison"),
            ("pr_auc", "PR AUC Comparison"),
            ("f1", "F1 Comparison"),
            ("training_time_sec", "Classification Training Time Comparison"),
            ("inference_time_sec", "Classification Inference Time Comparison"),
        ]:
            plot_group = group.sort_values(metric, ascending=metric.endswith("time_sec"))
            plt.figure(figsize=(10, 6))
            sns.barplot(data=plot_group, x=metric, y="model", color="#b25f00")
            plt.ylabel("")
            plt.xlabel(metric)
            plt.title(f"{title} - {target}")
            plt.tight_layout()
            plt.savefig(paths.plots / f"{sanitize_filename(title)}__{sanitize_filename(target)}.png", dpi=180)
            plt.close()


# ============================================================================
# BENCHMARK EXECUTION
# ============================================================================


def regression_target_arrays(
    prepared: PreparedData,
    target: str,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray | None]:
    if target == "days_to_next_visit":
        X_train = prepared.X_train
        X_test = prepared.X_test
        df_train = prepared.df_train
        df_test = prepared.df_test
        sample_weight = None
        if config.use_visit_timing_sample_weight:
            sample_weight = np.ones(len(df_train), dtype=float)
            sample_weight[df_train["any_breach_next"].values.astype(bool)] = 3.0
        y_train = df_train["visit_deviation"].astype(float).values
    else:
        X_train = prepared.X_train_wq
        X_test = prepared.X_test_wq
        df_train = prepared.df_train_wq
        df_test = prepared.df_test_wq
        sample_weight = None
        y_train = df_train[target].astype(float).values
    y_test = df_test[target].astype(float).values
    return X_train, X_test, y_train, y_test, df_test, sample_weight


def classification_target_arrays(
    prepared: PreparedData,
    target: str,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray | None]:
    X_train = prepared.X_train_wq
    X_test = prepared.X_test_wq
    df_train = prepared.df_train_wq
    df_test = prepared.df_test_wq
    y_train = df_train[target].astype(int).values
    y_test = df_test[target].astype(int).values
    sample_weight = None
    if config.use_balanced_classification_weights:
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    return X_train, X_test, y_train, y_test, df_test, sample_weight


def train_and_save_model(
    spec: ModelSpec,
    target: str,
    task_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None,
    paths: OutputPaths,
    seed: int,
) -> tuple[Any, float, float, float, float]:
    estimator = spec.factory(seed)
    rss_before = current_rss_mb()
    start = time.perf_counter()
    fit_estimator(estimator, X_train, y_train, sample_weight=sample_weight)
    training_time = time.perf_counter() - start
    rss_after = current_rss_mb()
    memory_delta = rss_after - rss_before if math.isfinite(rss_before) and math.isfinite(rss_after) else float("nan")

    model_path = paths.models / f"{sanitize_filename(task_type)}__{sanitize_filename(target)}__{sanitize_filename(spec.name)}.joblib"
    joblib.dump(estimator, model_path, compress=3)
    model_file_size_mb = model_path.stat().st_size / (1024**2)
    memory_estimate = max([value for value in [model_file_size_mb, memory_delta] if math.isfinite(value)] or [model_file_size_mb])
    return estimator, training_time, model_file_size_mb, memory_estimate, memory_delta


def run_regression_benchmarks(
    prepared: PreparedData,
    paths: OutputPaths,
    config: Config,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    specs = regression_model_specs(config)
    if not any(spec.name == "LightGBM Regressor" for spec in specs):
        logger.warning("LightGBM is not installed; LightGBM Regressor will be skipped.")
    if not any(spec.name == "CatBoost Regressor" for spec in specs):
        logger.warning("CatBoost is not installed; CatBoost Regressor will be skipped.")

    results: list[dict[str, Any]] = []
    feature_importance_rows: list[pd.DataFrame] = []
    predictions: dict[tuple[str, str], np.ndarray] = {}
    tasks = [(target, spec) for target in config.regression_targets for spec in specs]

    for target, spec in iter_progress(tasks, desc="Regression benchmark"):
        logger.info("Training regression model: target=%s model=%s", target, spec.name)
        X_train, X_test, y_train, y_test, df_test, sample_weight = regression_target_arrays(
            prepared,
            target,
            config,
        )
        base_row: dict[str, Any] = {
            "task_type": "regression",
            "target": target,
            "model": spec.name,
            "model_family": spec.family,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "status": "ok",
            "error": "",
        }
        try:
            estimator, train_time, file_size_mb, memory_estimate, memory_delta = train_and_save_model(
                spec,
                target,
                "regression",
                X_train,
                y_train,
                sample_weight,
                paths,
                config.random_seed,
            )
            infer_start = time.perf_counter()
            y_pred = np.asarray(estimator.predict(X_test), dtype=float).ravel()
            inference_time = time.perf_counter() - infer_start
            
            if target == "days_to_next_visit":
                y_pred = df_test["seasonal_baseline_days"].values + y_pred
                y_pred = np.clip(y_pred, 1, 60)

            predictions[(target, spec.name)] = y_pred

            row = {
                **base_row,
                **regression_metrics(y_test, y_pred),
                "training_time_sec": train_time,
                "inference_time_sec": inference_time,
                "inference_ms_per_row": (inference_time / max(len(y_test), 1)) * 1000,
                "model_file_size_mb": file_size_mb,
                "memory_usage_mb_estimate": memory_estimate,
                "memory_rss_delta_mb": memory_delta,
            }
            results.append(row)

            save_regression_plots(paths, target, spec.name, y_test, y_pred, config)
            save_regression_error_analysis(paths, target, spec.name, df_test, y_test, y_pred)

            importance = extract_feature_importance(estimator, prepared.feature_names)
            if not importance.empty:
                importance.insert(0, "target", target)
                importance.insert(1, "model", spec.name)
                feature_importance_rows.append(importance)
                save_feature_importance_plot(paths, target, spec.name, importance)

            if spec.shap_eligible:
                shap_importance = run_shap_analysis(
                    paths,
                    estimator,
                    X_test,
                    prepared.feature_names,
                    target,
                    spec.name,
                    config,
                    logger,
                )
                if not shap_importance.empty:
                    logger.info("Saved SHAP analysis for %s / %s", target, spec.name)
        except Exception as exc:
            logger.exception("Regression model failed: target=%s model=%s", target, spec.name)
            results.append({**base_row, "status": "failed", "error": str(exc)})

    regression_df = pd.DataFrame(results)
    feature_importance_df = (
        pd.concat(feature_importance_rows, ignore_index=True) if feature_importance_rows else pd.DataFrame()
    )
    return regression_df, feature_importance_df, predictions


def run_classification_benchmarks(
    prepared: PreparedData,
    paths: OutputPaths,
    config: Config,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = classification_model_specs(config)
    if not any(spec.name == "LightGBM Classifier" for spec in specs):
        logger.warning("LightGBM is not installed; LightGBM Classifier will be skipped.")
    if not any(spec.name == "CatBoost Classifier" for spec in specs):
        logger.warning("CatBoost is not installed; CatBoost Classifier will be skipped.")

    results: list[dict[str, Any]] = []
    feature_importance_rows: list[pd.DataFrame] = []
    tasks = [(target, spec) for target in config.classification_targets for spec in specs]

    for target, spec in iter_progress(tasks, desc="Classification benchmark"):
        logger.info("Training classification model: target=%s model=%s", target, spec.name)
        X_train, X_test, y_train, y_test, _, sample_weight = classification_target_arrays(
            prepared,
            target,
            config,
        )
        base_row: dict[str, Any] = {
            "task_type": "classification",
            "target": target,
            "model": spec.name,
            "model_family": spec.family,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "positive_rate_train": float(np.mean(y_train)),
            "positive_rate_test": float(np.mean(y_test)),
            "status": "ok",
            "error": "",
        }
        try:
            if len(np.unique(y_train)) < 2:
                raise ValueError(f"Training target {target} has only one class.")
            estimator, train_time, file_size_mb, memory_estimate, memory_delta = train_and_save_model(
                spec,
                target,
                "classification",
                X_train,
                y_train,
                sample_weight,
                paths,
                config.random_seed,
            )
            infer_start = time.perf_counter()
            y_pred = np.asarray(estimator.predict(X_test)).astype(int).ravel()
            y_score = predict_score(estimator, X_test)
            inference_time = time.perf_counter() - infer_start

            row = {
                **base_row,
                **classification_metrics(y_test, y_pred, y_score),
                "training_time_sec": train_time,
                "inference_time_sec": inference_time,
                "inference_ms_per_row": (inference_time / max(len(y_test), 1)) * 1000,
                "model_file_size_mb": file_size_mb,
                "memory_usage_mb_estimate": memory_estimate,
                "memory_rss_delta_mb": memory_delta,
            }
            results.append(row)
            save_classification_plots(paths, target, spec.name, y_test, y_pred, y_score)

            importance = extract_feature_importance(estimator, prepared.feature_names)
            if not importance.empty:
                importance.insert(0, "target", target)
                importance.insert(1, "model", spec.name)
                feature_importance_rows.append(importance)
                save_feature_importance_plot(paths, target, spec.name, importance)

            if spec.shap_eligible:
                run_shap_analysis(
                    paths,
                    estimator,
                    X_test,
                    prepared.feature_names,
                    target,
                    spec.name,
                    config,
                    logger,
                )
        except Exception as exc:
            logger.exception("Classification model failed: target=%s model=%s", target, spec.name)
            results.append({**base_row, "status": "failed", "error": str(exc)})

    classification_df = pd.DataFrame(results)
    feature_importance_df = (
        pd.concat(feature_importance_rows, ignore_index=True) if feature_importance_rows else pd.DataFrame()
    )
    return classification_df, feature_importance_df


def compute_rankings(regression_df: pd.DataFrame, classification_df: pd.DataFrame) -> pd.DataFrame:
    ranking_rows: list[pd.DataFrame] = []

    reg_ok = regression_df[regression_df["status"] == "ok"].copy()
    for target, group in reg_ok.groupby("target"):
        ranked = group.copy()
        ranked["rmse_rank"] = ranked["rmse"].rank(method="min", ascending=True)
        ranked["mae_rank"] = ranked["mae"].rank(method="min", ascending=True)
        ranked["r2_rank"] = ranked["r2"].rank(method="min", ascending=False)
        ranked["overall_weighted_score"] = (
            0.40 * ranked["rmse_rank"] + 0.30 * ranked["mae_rank"] + 0.30 * ranked["r2_rank"]
        )
        ranked["final_rank"] = ranked["overall_weighted_score"].rank(method="first", ascending=True).astype(int)
        ranked["ranking_basis"] = "40% RMSE rank, 30% MAE rank, 30% R2 rank"
        ranking_rows.append(
            ranked[
                [
                    "task_type",
                    "target",
                    "model",
                    "model_family",
                    "rmse",
                    "mae",
                    "r2",
                    "rmse_rank",
                    "mae_rank",
                    "r2_rank",
                    "overall_weighted_score",
                    "final_rank",
                    "ranking_basis",
                ]
            ]
        )

    clf_ok = classification_df[classification_df["status"] == "ok"].copy()
    for target, group in clf_ok.groupby("target"):
        ranked = group.copy()
        ranked["recall_rank"] = ranked["recall"].rank(method="min", ascending=False)
        ranked["pr_auc_rank"] = ranked["pr_auc"].rank(method="min", ascending=False)
        ranked["f1_rank"] = ranked["f1"].rank(method="min", ascending=False)
        ranked["overall_weighted_score"] = (
            0.40 * ranked["recall_rank"] + 0.35 * ranked["pr_auc_rank"] + 0.25 * ranked["f1_rank"]
        )
        ranked["final_rank"] = ranked["overall_weighted_score"].rank(method="first", ascending=True).astype(int)
        ranked["ranking_basis"] = "40% Recall rank, 35% PR-AUC rank, 25% F1 rank"
        ranking_rows.append(
            ranked[
                [
                    "task_type",
                    "target",
                    "model",
                    "model_family",
                    "recall",
                    "pr_auc",
                    "f1",
                    "recall_rank",
                    "pr_auc_rank",
                    "f1_rank",
                    "overall_weighted_score",
                    "final_rank",
                    "ranking_basis",
                ]
            ]
        )

    if not ranking_rows:
        return pd.DataFrame()
    return pd.concat(ranking_rows, ignore_index=True).sort_values(
        ["task_type", "target", "final_rank"]
    )


def run_stability_analysis(
    prepared: PreparedData,
    config: Config,
    logger: logging.Logger,
) -> pd.DataFrame:
    if not config.stability_enabled:
        return pd.DataFrame()

    specs = regression_model_specs(config)
    rows: list[dict[str, Any]] = []
    tasks = [(target, spec, seed) for target in config.regression_targets for spec in specs for seed in config.stability_seeds]

    for target, spec, seed in iter_progress(tasks, desc="Regression stability"):
        X_train, X_test, y_train, y_test, df_test, sample_weight = regression_target_arrays(prepared, target, config)
        base = {
            "task_type": "regression",
            "target": target,
            "model": spec.name,
            "seed": seed,
            "status": "ok",
            "error": "",
        }
        try:
            estimator = spec.factory(seed)
            fit_estimator(estimator, X_train, y_train, sample_weight=sample_weight)
            y_pred = np.asarray(estimator.predict(X_test), dtype=float).ravel()
            
            if target == "days_to_next_visit":
                y_pred = df_test["seasonal_baseline_days"].values + y_pred
                y_pred = np.clip(y_pred, 1, 60)

            metrics = regression_metrics(y_test, y_pred)
            rows.append(
                {
                    **base,
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                }
            )
        except Exception as exc:
            logger.warning("Stability run failed: target=%s model=%s seed=%s error=%s", target, spec.name, seed, exc)
            rows.append({**base, "status": "failed", "error": str(exc)})

    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    summary = (
        raw[raw["status"] == "ok"]
        .groupby(["task_type", "target", "model"], as_index=False)
        .agg(
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_r2=("r2", "mean"),
            std_r2=("r2", "std"),
            successful_runs=("seed", "count"),
        )
    )
    return summary


# ============================================================================
# REPORTING
# ============================================================================


def create_benchmark_summary(regression_df: pd.DataFrame, classification_df: pd.DataFrame) -> pd.DataFrame:
    reg_cols = [
        "task_type",
        "target",
        "model",
        "status",
        "rmse",
        "mae",
        "median_absolute_error",
        "r2",
        "explained_variance",
        "mape",
        "within_5_pct",
        "within_10_pct",
        "within_20_pct",
        "training_time_sec",
        "inference_ms_per_row",
        "model_file_size_mb",
        "memory_usage_mb_estimate",
    ]
    clf_cols = [
        "task_type",
        "target",
        "model",
        "status",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "mcc",
        "balanced_accuracy",
        "training_time_sec",
        "inference_ms_per_row",
        "model_file_size_mb",
        "memory_usage_mb_estimate",
    ]
    reg = regression_df[[col for col in reg_cols if col in regression_df.columns]].copy()
    clf = classification_df[[col for col in clf_cols if col in classification_df.columns]].copy()
    return pd.concat([reg, clf], ignore_index=True, sort=False)


def build_report(
    prepared: PreparedData,
    regression_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    rankings_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
    config: Config,
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    reg_ok = regression_df[regression_df["status"] == "ok"].copy()
    clf_ok = classification_df[classification_df["status"] == "ok"].copy()
    rankings_ok = rankings_df.copy()

    best_reg_by_target = (
        rankings_ok[rankings_ok["task_type"] == "regression"]
        .sort_values(["target", "final_rank"])
        .groupby("target")
        .head(1)
    )
    best_clf_by_target = (
        rankings_ok[rankings_ok["task_type"] == "classification"]
        .sort_values(["target", "final_rank"])
        .groupby("target")
        .head(1)
    )

    fastest_train = pd.concat([reg_ok, clf_ok], sort=False).sort_values("training_time_sec").head(1)
    fastest_predict = pd.concat([reg_ok, clf_ok], sort=False).sort_values("inference_ms_per_row").head(1)

    xgb_reg_wins = int((best_reg_by_target["model"].astype(str).str.contains("XGBoost")).sum()) if not best_reg_by_target.empty else 0
    total_reg_targets = len(best_reg_by_target)
    xgb_clf_wins = int((best_clf_by_target["model"].astype(str).str.contains("XGBoost")).sum()) if not best_clf_by_target.empty else 0
    total_clf_targets = len(best_clf_by_target)

    if not stability_df.empty:
        robust = stability_df.sort_values(["std_rmse", "std_mae"], ascending=True).head(1)
    else:
        robust = pd.DataFrame()

    failed_rows = pd.concat(
        [
            regression_df[regression_df["status"] != "ok"],
            classification_df[classification_df["status"] != "ok"],
        ],
        ignore_index=True,
        sort=False,
    )

    lines: list[str] = []
    lines.append("# Pool Predictive Maintenance V4 Benchmark Report")
    lines.append("")
    lines.append(f"Generated: {generated_at}")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        "V4 is a controlled model benchmark. It preserves the V3 data cleaning, static data backfill, "
        "feature engineering, target definitions, preprocessing, and temporal train/test split. The only "
        "changed layer is the estimator family used after the encoded train/test matrices are produced."
    )
    if not best_reg_by_target.empty:
        lines.append("")
        lines.append("Best regression models by target:")
        lines.append(markdown_table(best_reg_by_target, ["target", "model", "overall_weighted_score", "final_rank"]))
    if not best_clf_by_target.empty:
        lines.append("")
        lines.append("Best classification model by target:")
        lines.append(markdown_table(best_clf_by_target, ["target", "model", "overall_weighted_score", "final_rank"]))

    lines.append("")
    lines.append("## Dataset Overview")
    lines.append("")
    lines.append(f"- Raw source file: `{config.raw_csv}`")
    lines.append(f"- Master rows after V3 feature engineering: {len(prepared.df_master):,}")
    lines.append(f"- Model rows for visit timing: {len(prepared.df_model):,}")
    lines.append(f"- Model rows for water-quality and classification targets: {len(prepared.df_model_wq):,}")
    lines.append(f"- Unique pools: {prepared.df_master['pool_id'].nunique():,}")
    lines.append(f"- Date range: {prepared.df_master['reading_date'].min().date()} to {prepared.df_master['reading_date'].max().date()}")
    lines.append(f"- Temporal cutoff: {prepared.cutoff_date}")
    lines.append(f"- Train/test rows for visit timing: {len(prepared.df_train):,} / {len(prepared.df_test):,}")
    lines.append(f"- Train/test rows for water quality: {len(prepared.df_train_wq):,} / {len(prepared.df_test_wq):,}")
    lines.append("")
    lines.append("Static pool dimension backfill summary:")
    backfill_df = pd.DataFrame.from_dict(prepared.backfill_summary, orient="index").reset_index(names="feature")
    lines.append(markdown_table(backfill_df))

    lines.append("")
    lines.append("## Feature Overview")
    lines.append("")
    lines.append(f"- Numeric V3 features used: {len(prepared.all_numeric_features)}")
    lines.append(f"- Categorical V3 features used: {len(prepared.categorical_features)}")
    lines.append(f"- Encoded feature count after `ColumnTransformer`: {len(prepared.feature_names)}")
    lines.append(f"- Categorical features: {', '.join(prepared.categorical_features)}")
    lines.append("")
    lines.append("The benchmark uses the same V3 feature groups: static pool attributes, water-quality lags, rolling statistics, operations, product usage, visit timing fields, regulatory headroom, trend/rate features, breach history, chemistry metrics, and V3 volume-normalized dosing features.")

    lines.append("")
    lines.append("## Experiment Design")
    lines.append("")
    lines.append("- Temporal split: 80th percentile of `reading_date`, matching V3.")
    lines.append("- Numeric missing values: train-set medians, matching V3.")
    lines.append("- Categorical missing values: `unknown`, matching V3.")
    lines.append("- Encoding: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` for `pool_type` and `deck_type`, with numeric passthrough.")
    lines.append("- Regression ranking: 40% RMSE rank, 30% MAE rank, 30% R2 rank.")
    lines.append("- Classification ranking: 40% Recall rank, 35% PR-AUC rank, 25% F1 rank because missed chlorine breaches are more severe than false positives.")
    lines.append("- Visit timing sample weights: V3 safety weighting is applied uniformly across all regressors for `days_to_next_visit` when enabled.")
    lines.append("- Classification sample weights: balanced sample weights are applied uniformly across classifiers when enabled.")

    lines.append("")
    lines.append("## Model Descriptions")
    lines.append("")
    lines.append("- XGBoost: regularized gradient-boosted decision trees.")
    lines.append("- Random Forest: bagged decision trees with feature subsampling.")
    lines.append("- Extra Trees: highly randomized tree ensembles.")
    lines.append("- Gradient Boosting: sequential additive decision-tree boosting from scikit-learn.")
    lines.append("- LightGBM: histogram-based gradient boosting optimized for speed and scale.")
    lines.append("- CatBoost: ordered boosting with strong default regularization.")
    lines.append("- SVR/SVC: support-vector kernel methods using the same encoded feature matrix.")
    lines.append("- Logistic Regression: linear baseline for classification.")

    lines.append("")
    lines.append("## Regression Results")
    lines.append("")
    reg_view_cols = [
        "target",
        "model",
        "rmse",
        "mae",
        "median_absolute_error",
        "r2",
        "explained_variance",
        "mape",
        "within_5_pct",
        "within_10_pct",
        "within_20_pct",
        "training_time_sec",
        "inference_ms_per_row",
    ]
    reg_view = reg_ok[reg_view_cols].copy() if not reg_ok.empty else pd.DataFrame()
    for col in reg_view.select_dtypes(include=[float]).columns:
        reg_view[col] = reg_view[col].map(lambda x: round(float(x), 5))
    lines.append(markdown_table(reg_view.sort_values(["target", "rmse"]), max_rows=80))

    lines.append("")
    lines.append("## Classification Results")
    lines.append("")
    clf_view_cols = [
        "target",
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "mcc",
        "balanced_accuracy",
        "tn",
        "fp",
        "fn",
        "tp",
        "training_time_sec",
        "inference_ms_per_row",
    ]
    clf_view = clf_ok[clf_view_cols].copy() if not clf_ok.empty else pd.DataFrame()
    for col in clf_view.select_dtypes(include=[float]).columns:
        clf_view[col] = clf_view[col].map(lambda x: round(float(x), 5))
    lines.append(markdown_table(clf_view.sort_values(["target", "recall"], ascending=[True, False]), max_rows=80))

    lines.append("")
    lines.append("## Model Ranking Tables")
    lines.append("")
    rank_view = rankings_df.copy()
    for col in rank_view.select_dtypes(include=[float]).columns:
        rank_view[col] = rank_view[col].map(lambda x: round(float(x), 5))
    lines.append(markdown_table(rank_view, max_rows=120))

    lines.append("")
    lines.append("## Visual Analysis")
    lines.append("")
    lines.append("The `outputs_v4/plots/` directory contains actual-vs-predicted plots, residual distributions, residual scatter plots, prediction-error distributions, feature-importance plots, model comparison charts, confusion matrices, precision-recall curves, and ROC curves.")

    lines.append("")
    lines.append("## Feature Importance Analysis")
    lines.append("")
    if feature_importance_df.empty:
        lines.append("No model-native feature importance values were available.")
    else:
        top_importance = (
            feature_importance_df.sort_values("importance", ascending=False)
            .groupby(["target", "model"])
            .head(5)
            .copy()
        )
        for col in top_importance.select_dtypes(include=[float]).columns:
            top_importance[col] = top_importance[col].map(lambda x: round(float(x), 6))
        lines.append(markdown_table(top_importance[["target", "model", "feature", "importance"]], max_rows=120))

    lines.append("")
    lines.append("## SHAP Analysis")
    lines.append("")
    lines.append("SHAP summaries are generated for tree families where supported: XGBoost, LightGBM, CatBoost, Random Forest, and Extra Trees. Files are saved in `outputs_v4/shap/`. Kernel models and linear baselines are not assigned SHAP here to avoid introducing a different background approximation protocol.")

    lines.append("")
    lines.append("## Error Analysis")
    lines.append("")
    lines.append("For every regression target and model, V4 writes:")
    lines.append("- `best_predictions__*.csv`")
    lines.append("- `worst_predictions__*.csv`")
    lines.append("- `top_100_largest_errors__*.csv`")
    lines.append("")
    lines.append("Each file includes pool ID, date, actual value, predicted value, absolute error, and percentage error.")

    lines.append("")
    lines.append("## Stability Analysis")
    lines.append("")
    if stability_df.empty:
        lines.append("Stability analysis did not run or produced no successful rows.")
    else:
        stability_view = stability_df.copy()
        for col in stability_view.select_dtypes(include=[float]).columns:
            stability_view[col] = stability_view[col].map(lambda x: round(float(x), 5))
        lines.append(markdown_table(stability_view.sort_values(["target", "mean_rmse"]), max_rows=120))

    lines.append("")
    lines.append("## Performance vs Complexity Discussion")
    lines.append("")
    lines.append("Tree ensembles are expected to perform strongly on this tabular, mixed categorical/numeric operational dataset because they capture nonlinear thresholds, interactions, and regime changes without additional preprocessing. Kernel SVMs provide a useful nonlinear baseline but can be computationally expensive on large dense matrices. Logistic Regression is included as an interpretable linear safety classifier baseline.")

    lines.append("")
    lines.append("## Training Time Discussion")
    lines.append("")
    if not fastest_train.empty:
        ft = fastest_train.iloc[0]
        lines.append(f"The fastest successful training run was `{ft['model']}` on `{ft['target']}` at {ft['training_time_sec']:.4f} seconds.")
    lines.append("Training-time comparison charts are saved in `outputs_v4/plots/`.")

    lines.append("")
    lines.append("## Inference Time Discussion")
    lines.append("")
    if not fastest_predict.empty:
        fp = fastest_predict.iloc[0]
        lines.append(f"The fastest successful inference run was `{fp['model']}` on `{fp['target']}` at {fp['inference_ms_per_row']:.6f} ms per row.")
    lines.append("Inference-time comparison charts are saved in `outputs_v4/plots/`.")

    lines.append("")
    lines.append("## Memory Footprint Discussion")
    lines.append("")
    memory_cols = ["task_type", "target", "model", "model_file_size_mb", "memory_usage_mb_estimate"]
    memory_view = pd.concat([reg_ok, clf_ok], sort=False)[memory_cols].sort_values("memory_usage_mb_estimate")
    for col in memory_view.select_dtypes(include=[float]).columns:
        memory_view[col] = memory_view[col].map(lambda x: round(float(x), 5))
    lines.append(markdown_table(memory_view, max_rows=80))

    lines.append("")
    lines.append("## Model Recommendations")
    lines.append("")
    if not best_reg_by_target.empty:
        lines.append("Recommended regression model by target:")
        lines.append(markdown_table(best_reg_by_target[["target", "model", "overall_weighted_score", "final_rank"]]))
    if not best_clf_by_target.empty:
        lines.append("")
        lines.append("Recommended chlorine breach classifier:")
        lines.append(markdown_table(best_clf_by_target[["target", "model", "overall_weighted_score", "final_rank"]]))

    lines.append("")
    lines.append("## Final Recommendation Section")
    lines.append("")
    if not rankings_df.empty:
        best_accuracy = rankings_df.sort_values(["final_rank", "overall_weighted_score"]).head(1).iloc[0]
        lines.append(f"1. Best predictive accuracy: `{best_accuracy['model']}` for `{best_accuracy['target']}` under the configured weighted ranking.")
    if not robust.empty:
        rb = robust.iloc[0]
        lines.append(f"2. Most robust model: `{rb['model']}` for `{rb['target']}` based on the lowest RMSE/MAE standard deviation across seeds.")
    if not fastest_train.empty:
        ft = fastest_train.iloc[0]
        lines.append(f"3. Fastest training model: `{ft['model']}` on `{ft['target']}`.")
    if not fastest_predict.empty:
        fp = fastest_predict.iloc[0]
        lines.append(f"4. Fastest prediction model: `{fp['model']}` on `{fp['target']}`.")
    if not best_reg_by_target.empty or not best_clf_by_target.empty:
        lines.append("5. Production deployment should use the best-ranked model per target unless latency, file size, or operational support requirements override a small accuracy difference.")
    if total_reg_targets or total_clf_targets:
        total_wins = xgb_reg_wins + xgb_clf_wins
        total_targets = total_reg_targets + total_clf_targets
        if total_wins == total_targets:
            lines.append("6. XGBoost remains the best choice across all evaluated targets in this run.")
        elif total_wins == 0:
            lines.append("6. XGBoost is not the best overall choice in this run because other model families achieved better weighted ranks on the target benchmarks.")
        else:
            lines.append(f"6. XGBoost is partially competitive: it won {total_wins} of {total_targets} target rankings, but was not universally best.")
        lines.append("7. If XGBoost is not selected for a target, the reason should be read from that target's ranking table: lower RMSE/MAE or higher R2 for regression, or higher Recall/PR-AUC/F1 for chlorine safety classification.")

    if not failed_rows.empty:
        lines.append("")
        lines.append("## Failed or Skipped Runs")
        lines.append("")
        lines.append(markdown_table(failed_rows[["task_type", "target", "model", "status", "error"]], max_rows=80))

    return "\n".join(lines) + "\n"


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main() -> int:
    config = CONFIG
    paths = ensure_output_dirs(config)
    logger = setup_logging(paths)
    set_global_seed(config.random_seed)

    logger.info("Starting Pipeline V4 benchmark suite.")
    logger.info("Outputs will be written to %s", paths.root)

    prepared = prepare_dataset(config, paths, logger)

    regression_df, reg_importance_df, _ = run_regression_benchmarks(prepared, paths, config, logger)
    classification_df, clf_importance_df = run_classification_benchmarks(prepared, paths, config, logger)
    feature_importance_df = pd.concat(
        [reg_importance_df, clf_importance_df],
        ignore_index=True,
        sort=False,
    )

    rankings_df = compute_rankings(regression_df, classification_df)
    stability_df = run_stability_analysis(prepared, config, logger)

    benchmark_summary = create_benchmark_summary(regression_df, classification_df)

    regression_df.to_csv(paths.root / "regression_comparison.csv", index=False)
    classification_df.to_csv(paths.root / "classification_comparison.csv", index=False)
    benchmark_summary.to_csv(paths.root / "benchmark_summary.csv", index=False)
    rankings_df.to_csv(paths.root / "model_rankings.csv", index=False)
    stability_df.to_csv(paths.root / "stability_analysis.csv", index=False)
    if not feature_importance_df.empty:
        feature_importance_df.to_csv(paths.root / "feature_importance.csv", index=False)

    save_model_comparison_plots(paths, regression_df, classification_df)

    report_md = build_report(
        prepared,
        regression_df,
        classification_df,
        rankings_df,
        stability_df,
        feature_importance_df,
        config,
    )
    (paths.root / "benchmark_report.md").write_text(report_md, encoding="utf-8")
    (paths.root / "benchmark_report.txt").write_text(report_md, encoding="utf-8")

    logger.info("Pipeline V4 benchmark complete.")
    logger.info("Report: %s", paths.root / "benchmark_report.md")
    logger.info("Summary CSV: %s", paths.root / "benchmark_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
