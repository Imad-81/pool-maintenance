"""
Training pipeline composition + CLI.

Usage:
    python -m ml.training.train                       # full run, promote if better
    python -m ml.training.train --dry-run             # validate data load only
    python -m ml.training.train --run-id custom-2026w32

A run is reproducible: emit artifacts under models/<run_id>/ atomically, then
update models/latest.json so the backend hot-swaps. A crashed run leaves no
half-written state behind.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Heavy optional deps imported lazily inside run_pipeline so `--dry-run` and
# tests can import this module without paying the xgboost/shap load cost.
from ml.config import DEFAULT_CONFIG, PipelineConfig
from ml.training import steps as S
from ml.training.artifacts import ArtifactStore, build_inference_config
from ml.training.evaluate import compute_metrics, should_promote

log = logging.getLogger("ml.training")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_pipeline(cfg: PipelineConfig, run_id: str, dry_run: bool = False) -> dict:
    """Execute the full training pipeline. Returns the metrics dict.

    On `dry_run=True` we stop after STEP 1.5 (data load + pool filter) so the
    expensive weather/clean/train stages are skipped — used by CI smoke tests.
    """
    cfg.ensure_dirs()

    # STEP 1 — load + rename
    df_raw = S.load_and_rename(cfg)

    # STEP 1.5 — chlorine-pump pool filter
    df = S.filter_chlorine_pump_pools(df_raw, cfg)

    if dry_run:
        log.info("DRY-RUN  loaded %d rows across %d pools — stopping before weather/train",
                 len(df), df["pool_id"].dropna().nunique())
        return {"run_id": run_id, "dry_run": True, "rows": int(len(df)),
                "pools": int(df["pool_id"].dropna().nunique())}

    # STEP 2 — weather
    df_weather = S.load_or_fetch_weather(df, cfg)

    # STEP 3 — separate sub-tables
    df_readings, df_ops, df_products = S.separate_subtables(df)

    # STEP 4 — clean
    df_readings = S.clean_readings(df_readings)
    df_ops = S.clean_operations(df_ops)
    df_products = S.clean_products(df_products)

    # STEP 4.5 — backfill static
    df_readings, backfill_summary = S.backfill_static(df_readings)

    # STEP 5 — merge sub-tables
    df_master = S.merge_subtables(df_readings, df_ops, df_products, cfg)

    # STEP 6 — join weather (today + tomorrow)
    df_master, weather_today_cols, weather_tmrw_cols = S.join_weather(df_master, df_weather)

    # STEP 7 — feature engineering
    df_master = S.engineer_features(df_master, df_weather, cfg)

    # STEP 8 — targets
    df_master, df_model, df_model_wq = S.build_targets(df_master, cfg)

    # STEP 9 — feature selection + split
    split = S.select_features_and_split(df_model_wq, cfg, weather_tmrw_cols)
    preprocessor = split["preprocessor"]
    X_train, X_test = split["X_train"], split["X_test"]
    df_train_wq, df_test_wq = split["train"], split["test"]
    all_numeric = split["all_numeric"]
    categorical = split["categorical"]
    feature_names = split["feature_names"]
    fill_values = split["fill_values"]

    # save preprocessor (held in split["preprocessor"]; written via ArtifactStore below)

    # STEP 10 — train models
    import xgboost as xgb
    xgb_params = dict(cfg.xgb_params)
    early = cfg.early_stopping_rounds

    results: dict = {}

    # --- Cl ---
    y_cl = df_train_wq["target_cl_tomorrow"]
    y_cl_test = df_test_wq["target_cl_tomorrow"]
    sw_cl = np.ones(len(df_train_wq))
    breach = df_train_wq["any_breach_next"].values.astype(bool)
    sw_cl[breach] = 3.0
    model_cl = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=early, eval_metric="rmse")
    model_cl.fit(X_train, y_cl.values, eval_set=[(X_test, y_cl_test.values)],
                 sample_weight=sw_cl, verbose=False)
    pred_cl = model_cl.predict(X_test)
    res_cl = compute_metrics(y_cl_test.values, pred_cl, model_cl.best_iteration)
    results["chlorine_next"] = res_cl.to_dict()
    log.info("  chlorine_next: RMSE=%.4f MAE=%.4f R2=%.4f", res_cl.rmse, res_cl.mae, res_cl.r2)

    # --- pH ---
    y_ph = df_train_wq["target_ph_tomorrow"]
    y_ph_test = df_test_wq["target_ph_tomorrow"]
    model_ph = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=early, eval_metric="rmse")
    model_ph.fit(X_train, y_ph.values, eval_set=[(X_test, y_ph_test.values)], verbose=False)
    pred_ph = model_ph.predict(X_test)
    res_ph = compute_metrics(y_ph_test.values, pred_ph, model_ph.best_iteration)
    results["ph_next"] = res_ph.to_dict()
    log.info("  ph_next: RMSE=%.4f MAE=%.4f R2=%.4f", res_ph.rmse, res_ph.mae, res_ph.r2)

    # --- Turbidity (own subset, NaN-free for target_turb_tomorrow) ---
    df_train_t = df_train_wq.dropna(subset=["target_turb_tomorrow"])
    df_test_t  = df_test_wq.dropna(subset=["target_turb_tomorrow"])
    X_train_t = preprocessor.transform(df_train_t[categorical + all_numeric])
    X_test_t  = preprocessor.transform(df_test_t[categorical + all_numeric])
    model_turb = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=early, eval_metric="rmse")
    model_turb.fit(X_train_t, df_train_t["target_turb_tomorrow"].values,
                   eval_set=[(X_test_t, df_test_t["target_turb_tomorrow"].values)],
                   verbose=False)
    pred_turb = model_turb.predict(X_test_t)
    res_turb = compute_metrics(df_test_t["target_turb_tomorrow"].values, pred_turb, model_turb.best_iteration)
    results["turbidity_next"] = res_turb.to_dict()
    log.info("  turbidity_next: RMSE=%.4f MAE=%.4f R2=%.4f", res_turb.rmse, res_turb.mae, res_turb.r2)

    # STEP 11 — SHAP explainability (best-effort; skip if shap missing)
    shap_results: dict = {}
    try:
        import shap
        for name, model, X_shap in [
            ("chlorine_next", model_cl, X_test),
            ("ph_next", model_ph, X_test),
            ("turbidity_next", model_turb, X_test_t),
        ]:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_shap)
            top = pd.Series(np.abs(sv).mean(axis=0), index=feature_names).sort_values(ascending=False).head(15)
            shap_results[name] = top.to_dict()
            log.info("  SHAP %s top: %s", name, list(top.items())[:3])
    except Exception as e:  # pragma: no cover
        log.warning("  SHAP skipped: %s", e)

    # STEP 12 — write artifacts atomically
    from ml import features as F_mod
    control_features = F_mod.control_features()
    weather_current = F_mod.weather_current_features()
    weather_cumulative = F_mod.weather_cumulative_features()

    inference_config = build_inference_config(
        cfg=cfg, run_id=run_id, df_master=df_master, fill_values=fill_values,
        all_numeric_features=all_numeric, categorical_features=categorical,
        feature_names=feature_names, control_features=control_features,
        weather_current=weather_current, weather_cumulative=weather_cumulative,
        weather_tomorrow=weather_tmrw_cols, results=results, shap_results=shap_results,
    )

    store = ArtifactStore(cfg.models_dir_path, run_id)
    with store:
        store.write_model_xgb("chlorine_next", model_cl)
        store.write_model_xgb("ph_next", model_ph)
        store.write_model_xgb("turbidity_next", model_turb)
        store.write_pickle("preprocessor_v6.pkl", preprocessor)
        store.write_json("inference_config_v6.json", inference_config)

    # master dataset + report (legacy artifacts, kept for backward compatibility)
    master_path = cfg.output_dir_path / "master_dataset_v6.csv"
    df_master.to_csv(master_path, index=False)
    log.info("  saved master -> %s", master_path)

    # STEP 13 — promotion gate vs. current active run (read from latest.json)
    old_metrics = None
    old_id = ArtifactStore.read_latest_pointer(cfg.models_dir_path)
    if old_id is not None:
        try:
            import json
            with open(cfg.models_dir_path / old_id / "inference_config_v6.json") as f:
                old_cfg = json.load(f)
            old_metrics = old_cfg.get("metrics")
        except Exception as e:
            log.warning("  could not read prior metrics from %s: %s", old_id, e)
    else:
        log.info("  no prior active run — first-run promotion")

    promote, reason = should_promote(
        results, old_metrics,
        cfg.promotion_tolerance_cl, cfg.promotion_tolerance_ph, cfg.promotion_tolerance_turb)
    if promote:
        ArtifactStore.write_latest_pointer(cfg.models_dir_path, run_id)
        log.info("  PROMOTED %s — %s", run_id, reason)
    else:
        log.warning("  NOT promoted %s — %s", run_id, reason)
        # Archive the non-promoted run's artifacts so they don't clutter
        # models/ or confuse future runs. The _promote() in ArtifactStore
        # already moved .tmp -> final dir; here we move it to archive/.
        run_dir = cfg.models_dir_path / run_id
        if run_dir.exists():
            archive = cfg.models_dir_path / "archive"
            archive.mkdir(exist_ok=True)
            import shutil
            from datetime import datetime as _dt
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            shutil.move(str(run_dir), str(archive / f"{run_id}_{ts}"))
            log.info("  archived non-promoted run -> models/archive/%s_%s", run_id, ts)

    log.info("PIPELINE V6 COMPLETE  run_id=%s  metrics=%s", run_id,
             {k: {m: round(v, 4) if isinstance(v, float) else v for m, v in d.items()}
              for k, d in results.items()})
    return {"run_id": run_id, "metrics": results, "promoted": promote, "reason": reason,
            "shap_summary": shap_results}


def _list_prior_runs(cfg: PipelineConfig) -> list[str]:
    """Return prior run dirs in newest-first order (excluding the current .tmp)."""
    d = cfg.models_dir_path
    if not d.exists():
        return []
    runs = []
    for p in d.iterdir():
        if p.is_dir() and not p.name.endswith(".tmp") and p.name not in ("archive",):
            runs.append(p.name)
    runs.sort(reverse=True)
    return runs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pool V6 training pipeline")
    parser.add_argument("--run-id", default=None,
                        help="Unique run ID (default: auto-timestamped)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load data only — skip cleaning, weather and training")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)
    run_id = args.run_id or datetime.now().strftime("v6-%Y%m%d-%H%M%S")
    log.info("=== training run %s ===", run_id)
    try:
        run_pipeline(DEFAULT_CONFIG, run_id=run_id, dry_run=args.dry_run)
    except FileNotFoundError as e:
        log.error("DATA ERROR: %s", e)
        return 2
    except Exception as e:
        log.exception("training failed: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())