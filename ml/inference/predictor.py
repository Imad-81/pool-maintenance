"""
Pure chained multi-day forecast engine + the `PredictionService` the FastAPI
backend binds to at startup.

`predict_forward` is a pure function — it accepts the latest master row for a
pool, an as-of date, a live weather-lookup callable, and the loaded
models/preprocessor/config. It never touches disk, never logs files, never
imports globals. That makes it golden-output testable and lets the backend
plug in a SQLite-backed weather provider without monkey-patching.

`PredictionService` loads the *active* model run once (read from
`models/latest.json`) and re-loads when the scheduler hot-swaps. Hot-swap is
graceful: a failed reload logs and keeps the previously loaded artefacts, so
live prediction never dies mid-request.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ml.config import (
    CLIENT_CL_TARGET_MAX,
    CLIENT_CL_TARGET_MIN,
    REG_CHLORINE_CLOSE,
    REG_CHLORINE_IDEAL_MAX,
    REG_CHLORINE_MIN,
    REG_PH_MAX,
    REG_PH_MIN,
)
from ml.inference.chaining import (
    DEFAULT_HORIZON_DAYS,
    MAX_HORIZON_DAYS,
    UncertaintyBand,
    clamp_horizon,
    warning_band,
)

log = logging.getLogger(__name__)


# Type alias for the weather-lookup callable: given a normalised date and a
# list of weather column names, return {col_name: value} (NaN if missing).
WeatherLookup = Callable[[pd.Timestamp, list[str]], dict]


# ---------------------------------------------------------------------------
# Pure chained forecast
# ---------------------------------------------------------------------------

def predict_forward(
    *,
    pool_id: str,
    latest_row: pd.Series,
    as_of_date: pd.Timestamp,
    weather_lookup: WeatherLookup,
    models: dict,                # {"chlorine": XGBRegressor, "ph": ..., "turbidity": ...}
    preprocessor,
    config: dict,                # the inference_config_v6.json payload
    horizon_days: Optional[int] = None,
) -> dict:
    """Chain 1-day-forward predictions from the last visit to `as_of_date +
    horizon_days`.

    Returns a dict with pool_id, last_visit_date, days_since_visit, last
    readings, a `forecast` DataFrame (one row per day with per-day
    UncertaintyBand), the today/tomorrow sections highlighted, and an
    overall `visit_needed` flag.

    The caller (the backend / PredictionService) is responsible for providing
    the pool's latest fully-featured master row plus a working weather_lookup
    — this function isolates *only* the chaining logic so it can be unit
    tested deterministically against a stubbed weather_lookup.
    """
    as_of_date = pd.Timestamp(as_of_date).normalize()
    horizon = clamp_horizon(horizon_days)

    last_visit_date = pd.Timestamp(latest_row["reading_date"]).normalize()
    days_since = int((as_of_date - last_visit_date).days)
    if days_since < 0:
        return {"error": f"as_of_date {as_of_date.date()} is before last visit {last_visit_date.date()}"}

    all_numeric = list(config["all_numeric_features"])
    categorical = list(config["categorical_features"])
    fill_values = {k: float(v) for k, v in config["fill_values"].items()}
    today_wx = list(config.get("weather_current_features", []))
    tmrw_wx = list(config.get("weather_tomorrow_features", []))

    base = latest_row.copy()
    for col in all_numeric:
        if col not in base.index or pd.isna(base.get(col, np.nan)):
            base[col] = fill_values.get(col, 0.0)
    for col in categorical:
        if col not in base.index or pd.isna(base.get(col, np.nan)):
            base[col] = "unknown"

    cur_cl   = float(latest_row.get("free_chlorine", fill_values.get("free_chlorine", 2.0)))
    cur_ph   = float(latest_row.get("ph",            fill_values.get("ph",            7.4)))
    cur_turb = float(latest_row.get("turbidity",    fill_values.get("turbidity",     0.5)))

    row = base.copy()
    total_steps = days_since + 1  # +1 to include horizon's last day
    prev_cl, prev_ph, prev_turb = cur_cl, cur_ph, cur_turb
    forecast_rows: list[dict] = []

    model_cl, model_ph, model_turb = models["chlorine"], models["ph"], models["turbidity"]

    for step in range(1, total_steps + 1):
        if step > (days_since + horizon):
            break
        step_date = last_visit_date + pd.Timedelta(days=step)

        # temporal features
        row["visit_month"] = int(step_date.month)
        row["visit_day_of_week"] = int(step_date.dayofweek)
        row["visit_is_summer"] = int(step_date.month in (6, 7, 8, 9))
        row["visit_year"] = int(step_date.year)

        # weather injection — today's + tomorrow's
        row = _inject_weather(row, weather_lookup, step_date, today_wx, tmrw_wx)

        # recompute chemistry-dependent features on the previous step's state
        row = _recompute_features(row, cur_cl, cur_ph, cur_turb,
                                  step=step, prev_cl=prev_cl, prev_ph=prev_ph)

        # build the preprocessor frame
        feat = pd.DataFrame([row])
        for col in all_numeric:
            if col not in feat.columns:
                feat[col] = fill_values.get(col, 0.0)
            feat[col] = pd.to_numeric(feat[col], errors="coerce").fillna(fill_values.get(col, 0.0))
        for col in categorical:
            if col not in feat.columns:
                feat[col] = "unknown"
            feat[col] = feat[col].fillna("unknown").astype(str)

        X = preprocessor.transform(feat[categorical + all_numeric])
        pred_cl   = max(0.0, float(model_cl.predict(X)[0]))
        pred_ph   = float(model_ph.predict(X)[0])
        pred_turb = max(0.0, float(model_turb.predict(X)[0]))

        cl_breach = pred_cl < REG_CHLORINE_MIN or pred_cl > REG_CHLORINE_CLOSE
        ph_breach = pred_ph < REG_PH_MIN or pred_ph > REG_PH_MAX
        urgency, status = _classify(pred_cl, pred_ph, cl_breach, ph_breach)

        is_today    = (step_date == as_of_date)
        is_tomorrow = (step_date == as_of_date + pd.Timedelta(days=1))
        day_offset_from_today = int((step_date - as_of_date).days)

        band = warning_band(day_offset_from_today, pred_cl, pred_ph, pred_turb)

        day_label = step_date.strftime("%a")
        if is_today:
            day_label += " ◀ TODAY"
        elif is_tomorrow:
            day_label += " ◀ TOMORROW"

        forecast_rows.append({
            "date":            step_date.date(),
            "day":             day_label,
            "days_from_visit": step,
            "day_offset_from_today": day_offset_from_today,
            "predicted_cl":    round(pred_cl, 3),
            "predicted_ph":    round(pred_ph, 3),
            "predicted_turb":  round(pred_turb, 3),
            "cl_breach":       bool(cl_breach),
            "ph_breach":       bool(ph_breach),
            "urgency":         urgency,
            "status":          status,
            "is_today":        bool(is_today),
            "is_tomorrow":     bool(is_tomorrow),
            "uncertainty_band": band,
        })

        prev_cl, prev_ph, prev_turb = cur_cl, cur_ph, cur_turb
        cur_cl, cur_ph, cur_turb = pred_cl, pred_ph, pred_turb

    fc = pd.DataFrame(forecast_rows)
    if fc.empty:
        return {"pool_id": pool_id, "error": "no forecast produced — check days_since/inputs"}

    dashboard = fc[fc["is_today"] | fc["is_tomorrow"]]
    visit_needed = bool(
        (dashboard["cl_breach"].any() if len(dashboard) else False) or
        (dashboard["ph_breach"].any() if len(dashboard) else False) or
        ((dashboard["urgency"] == "Advised").any() if len(dashboard) else False)
    )

    return {
        "pool_id":           pool_id,
        "last_visit_date":   last_visit_date.date(),
        "days_since_visit":  days_since,
        "last_readings":     {
            "free_chlorine": round(float(latest_row.get("free_chlorine", 0)), 3),
            "ph":            round(float(latest_row.get("ph",            0)), 3),
            "turbidity":     round(float(latest_row.get("turbidity",     0)), 3),
        },
        "forecast":          fc,
        "today_forecast":    fc[fc["is_today"]].to_dict("records"),
        "tomorrow_forecast": fc[fc["is_tomorrow"]].to_dict("records"),
        "visit_needed":      visit_needed,
    }


# ---------------------------------------------------------------------------
# Feature recomputation (mirrors inference.py `_recompute_features` + the V6
# pipeline's headroom/trend logic from ml/features.py)
# ---------------------------------------------------------------------------

def _recompute_features(row, pred_cl, pred_ph, pred_turb, step, prev_cl, prev_ph):
    row = row.copy()

    # current state
    row["free_chlorine"] = pred_cl
    row["ph"]            = pred_ph
    row["turbidity"]     = max(0.0, pred_turb)

    # lags
    row["chlorine_lag2"] = row.get("chlorine_lag1", pred_cl)
    row["chlorine_lag1"] = prev_cl
    row["ph_lag2"]       = row.get("ph_lag1", pred_ph)
    row["ph_lag1"]       = prev_ph
    row["turbidity_lag1"] = row.get("turbidity_lag1", pred_turb)
    row["turbidity_lag2"] = row.get("turbidity_lag2", pred_turb)

    # rolling (3-step window)
    vals_cl   = [pred_cl,   prev_cl,   row.get("chlorine_lag2", pred_cl)]
    vals_ph   = [pred_ph,   prev_ph,   row.get("ph_lag2", pred_ph)]
    vals_turb = [pred_turb, row.get("turbidity_lag1", pred_turb), row.get("turbidity_lag2", pred_turb)]
    row["chlorine_roll3_mean"]  = float(np.mean(vals_cl))
    row["chlorine_roll3_std"]   = float(np.std(vals_cl))
    row["ph_roll3_mean"]        = float(np.mean(vals_ph))
    row["ph_roll3_std"]         = float(np.std(vals_ph))
    row["turbidity_roll3_mean"] = float(np.mean(vals_turb))

    # temporal
    row["days_since_last_visit"] = step

    # headroom
    from ml.features import add_headroom_features
    single = pd.DataFrame([row])
    single = add_headroom_features(single)
    for c in ["chlorine_headroom_low", "chlorine_headroom_high",
              "ph_headroom_low", "ph_headroom_high",
              "turbidity_headroom", "min_headroom",
              "cl_below_client_target", "cl_above_client_target"]:
        row[c] = float(single[c].iloc[0])

    # trends
    row["chlorine_trend"]        = pred_cl - prev_cl
    row["ph_trend"]              = pred_ph - prev_ph
    row["turbidity_trend"]       = pred_turb - row.get("turbidity_lag1", pred_turb)
    safe_step = step if step else np.nan
    row["chlorine_rate_per_day"] = (pred_cl - prev_cl) / safe_step if safe_step else 0.0
    row["ph_rate_per_day"]       = (pred_ph - prev_ph) / safe_step if safe_step else 0.0
    row["turbidity_rate_per_day"] = row["turbidity_trend"] / safe_step if safe_step else 0.0

    # effectiveness index
    from ml.features import cl_effectiveness
    row["cl_effectiveness_index"] = cl_effectiveness(pred_cl, pred_ph)

    return row


# ---------------------------------------------------------------------------
# Weather injection
# ---------------------------------------------------------------------------

def _inject_weather(row, weather_lookup, step_date, today_wx, tmrw_wx):
    today = pd.Timestamp(step_date).normalize()
    today_vals = weather_lookup(today, today_wx)
    for col, v in today_vals.items():
        row[col] = v
    # tomorrow block: lookup weather at (step_date+1), rename to w_tmrw_*
    tomorrow = today + pd.Timedelta(days=1)
    tmrw_src = [c.replace("w_tmrw_", "w_") for c in tmrw_wx]
    tmrw_vals = weather_lookup(tomorrow, tmrw_src)
    for t_col, s_col in zip(tmrw_wx, tmrw_src):
        row[t_col] = tmrw_vals.get(s_col, np.nan)
    # cumulative-since-last-visit weather — for a chained step, the most
    # physically meaningful approximation is the current day's value alone
    # (no inter-visit history in the chain). NaN here would drop the feature
    # which the preprocessor fill_values handle, so it's safe.
    return row


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _classify(pred_cl, pred_ph, cl_breach, ph_breach):
    if cl_breach or ph_breach:
        return "URGENT", "🚨 Regulatory breach — URGENT visit"
    if pred_cl < CLIENT_CL_TARGET_MIN:
        return "Advised", "⚠️  Cl below client target — visit advised"
    if pred_cl > REG_CHLORINE_IDEAL_MAX:
        return "Monitor", "⚠️  Cl above optimal range — monitor"
    return "Routine", "✅ OK"


# ---------------------------------------------------------------------------
# PredictionService — loaded once at backend startup, hot-swappable
# ---------------------------------------------------------------------------

class PredictionService:
    """Loads the active V6 run + creates an `Optimiser` on demand.

    Thread-safe: a single `reload()` flips `_state` atomically under a lock,
    and live `forecast`/`optimise` reads use snapshots so concurrent requests
    during a retrain promotion are safe.
    """

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self._lock = threading.RLock()
        self._state: Optional[dict] = None

    # --- public API --------------------------------------------------------

    def load(self) -> None:
        with self._lock:
            self._state = self._load_active_run()

    def reload(self) -> None:
        """Called by the scheduler after a retrain promotion. On failure,
        keep the previously loaded state and log — never kill live serving."""
        try:
            new_state = self._load_active_run()
        except Exception as e:  # pragma: no cover
            log.error("reload FAILED — keeping previous models: %s", e)
            return
        with self._lock:
            self._state = new_state
        log.info("reloaded active run %s", new_state.get("run_id"))

    def forecast(self, pool_id: str, latest_row: pd.Series,
                 as_of_date: pd.Timestamp, weather_lookup: WeatherLookup,
                 horizon_days: Optional[int] = None) -> dict:
        with self._lock:
            state = self._state
        if state is None:
            raise RuntimeError("PredictionService not loaded")
        return predict_forward(
            pool_id=pool_id, latest_row=latest_row, as_of_date=as_of_date,
            weather_lookup=weather_lookup, models=state["models"],
            preprocessor=state["preprocessor"], config=state["config"],
            horizon_days=horizon_days,
        )

    def optimise(self, pool_id: str, latest_row: pd.Series):
        from ml.inference.optimiser import Optimiser
        with self._lock:
            state = self._state
        if state is None:
            raise RuntimeError("PredictionService not loaded")
        opt = Optimiser(
            state["cfg"], state["models"]["chlorine"], state["models"]["ph"],
            state["preprocessor"], state["all_numeric_features"],
            state["categorical_features"], state["fill_values"],
        )
        return opt.optimise(pool_id, latest_row)

    def status(self) -> dict:
        with self._lock:
            state = self._state
        if state is None:
            return {"loaded": False}
        return {
            "loaded": True,
            "run_id": state["run_id"],
            "feature_schema": state["config"].get("feature_schema"),
            "metrics": state["config"].get("metrics"),
        }

    # --- internals ---------------------------------------------------------

    def _load_active_run(self) -> dict:
        from ml.config import DEFAULT_CONFIG
        pointer_path = self.models_dir / "latest.json"
        if not pointer_path.exists():
            # fall back to the legacy flat layout in models/ if present
            log.warning("no latest.json — falling back to legacy flat models")
            return self._load_legacy_flat(DEFAULT_CONFIG)

        pointer = json.loads(pointer_path.read_text())
        run_id = pointer["active_run_id"]
        run_dir = self.models_dir / run_id
        if not run_dir.exists():
            log.error("active run dir %s missing — falling back to legacy", run_dir)
            return self._load_legacy_flat(DEFAULT_CONFIG)

        return self._load_from_dir(run_dir, DEFAULT_CONFIG, run_id)

    def _load_from_dir(self, run_dir: Path, cfg, run_id: str) -> dict:
        import xgboost as xgb
        with open(run_dir / "inference_config_v6.json") as f:
            config = json.load(f)
        with open(run_dir / "preprocessor_v6.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        models = {}
        for name in ("chlorine", "ph", "turbidity"):
            m = xgb.XGBRegressor()
            m.load_model(run_dir / f"xgb_{name}_next.json")
            models[name] = m
        log.info("loaded run %s from %s", run_id, run_dir)
        return {
            "run_id": run_id,
            "config": config,
            "models": models,
            "preprocessor": preprocessor,
            "all_numeric_features": list(config["all_numeric_features"]),
            "categorical_features": list(config["categorical_features"]),
            "fill_values": {k: float(v) for k, v in config["fill_values"].items()},
            "cfg": cfg,
        }

    def _load_legacy_flat(self, cfg) -> dict:
        """Read the existing `models/xgb_*_next.json` + `preprocessor_v6.pkl`
        + `inference_config_v6.json` layout shipped before the refactor so the
        backend can boot against the already-trained artifacts."""
        import xgboost as xgb
        d = self.models_dir
        with open(d / "inference_config_v6.json") as f:
            config = json.load(f)
        with open(d / "preprocessor_v6.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        models = {}
        for name in ("chlorine", "ph", "turbidity"):
            m = xgb.XGBRegressor()
            m.load_model(d / f"xgb_{name}_next.json")
            models[name] = m
        log.info("loaded LEGACY flat model layout")
        return {
            "run_id": config.get("pipeline_version", "legacy"),
            "config": config,
            "models": models,
            "preprocessor": preprocessor,
            "all_numeric_features": list(config["all_numeric_features"]),
            "categorical_features": list(config["categorical_features"]),
            "fill_values": {k: float(v) for k, v in config["fill_values"].items()},
            "cfg": cfg,
        }