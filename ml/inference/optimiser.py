"""Dosing optimiser — grid-search over hypochlorite dosing % and pump hours to
find the minimal-effort configuration that keeps predicted chlorine inside the
client target [1.0–1.5 mg/L] and pH inside the regulatory range [7.2–8.0].

Ported from pipeline_v6.py STEP 12 (`optimise_dosing`) as a class so the
FastAPI backend can call it per-pool, with the live weather provider plugged
in so the recommendation reflects tomorrow's forecast.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from ml.config import (
    CLIENT_CL_TARGET_MAX,
    CLIENT_CL_TARGET_MIN,
    REG_PH_MAX,
    REG_PH_MIN,
    SETPOINT_FREE_CHLORINE,
    SETPOINT_PH,
    SETPOINT_TURBIDITY,
)
from ml.features import control_features

log = logging.getLogger(__name__)


@dataclass
class DosingResult:
    pool_id: str
    pool_volume_m3: float
    current_readings: dict
    recommended_dosing: dict
    predicted_tomorrow: dict
    feasible_configurations: int
    top_3_configs: list[dict]
    urgency: str
    reasons: list[str]


class Optimiser:
    """Wraps a preprocessor + chlorine & pH models and exposes a pure
    `optimise(pool_id, latest_row, horizon)` call.

    The grid (pct × hours) is the configured dosing grid from
    `PipelineConfig` — we hold the grid precomputed on the instance so
    repeated per-pool calls do not re-allocate.
    """

    def __init__(self, cfg, model_cl, model_ph, preprocessor,
                 all_numeric_features, categorical_features, fill_values,
                 is_delta: bool = False):
        self.cfg = cfg
        self.model_cl = model_cl
        self.model_ph = model_ph
        self.preprocessor = preprocessor
        self.all_numeric_features = all_numeric_features
        self.categorical_features = categorical_features
        self.fill_values = fill_values
        self.is_delta = is_delta
        self.pct_grid   = np.arange(0, 105, cfg.dosing_pct_step)
        self.hours_grid = np.arange(0, 25, cfg.dosing_hours_step)

    def optimise(self, pool_id: str, latest_row: pd.Series) -> DosingResult:
        env = _FeatureEnv.from_row(
            latest_row, self.all_numeric_features,
            self.categorical_features, self.fill_values,
            sp_cl=self.cfg.setpoint_free_chlorine,
            sp_ph=self.cfg.setpoint_ph,
            sp_turb=self.cfg.setpoint_turbidity,
        )
        pool_vol = env.pool_volume_m3 or 50.0

        # Vectorized 2D grid combinations (21 pct x 25 hours = 525 rows)
        pct_mesh, hours_mesh = np.meshgrid(self.pct_grid, self.hours_grid)
        pct_flat = pct_mesh.ravel().astype(float)
        hours_flat = hours_mesh.ravel().astype(float)
        n_grid = len(pct_flat)

        cur_cl = env.current_cl_raw if env.current_cl_raw is not None else env.current_cl
        cur_ph = env.current_ph_raw if env.current_ph_raw is not None else env.current_ph
        cur_turb = float(latest_row.get("turbidity") if pd.notna(latest_row.get("turbidity")) else 0.3)
        cur_cya = float(latest_row.get("cya") or latest_row.get("cya_cumulative_ppm") or 30.0)

        pump_flow = float(env.base.get("hypochlorite_pump_flow_rate") or 4.0)
        delivered_cl2 = np.clip((hours_flat * (pct_flat / 100.0) * pump_flow * 130.0) / pool_vol, 0.0, 0.40)

        if self.is_delta or self.preprocessor is None:
            from ml.inference.predictor import _build_v7_delta_feature_vector
            reading_date_raw = latest_row.get("reading_date")
            step_date = (
                pd.Timestamp(reading_date_raw) + pd.Timedelta(days=1)
                if pd.notna(reading_date_raw)
                else pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
            )
            base_v7 = _build_v7_delta_feature_vector(
                latest_row,
                step_date=step_date,
                anchor_cl=cur_cl,
                anchor_ph=cur_ph,
                anchor_turb=cur_turb,
                anchor_cya=cur_cya,
                cl_history=[cur_cl],
                ph_history=[cur_ph],
                turb_history=[cur_turb],
                step=1,
                feature_names=self.all_numeric_features,
                fill_values=self.fill_values,
            )
            grid_df = pd.DataFrame(np.repeat(base_v7.values, n_grid, axis=0), columns=base_v7.columns)
            grid_df["hypochlorite_dosing_pct"] = pct_flat
            grid_df["hypochlorite_dosing_hours"] = hours_flat
            grid_df["hypo_dosing_percentage"] = pct_flat
            grid_df["hypo_dosing_hours"] = hours_flat
            grid_df["daily_pump_cl2_delivered_ppm"] = delivered_cl2

            feat_df = grid_df[self.all_numeric_features]
            preds_delta_cl = np.asarray(self.model_cl.predict(feat_df), dtype=float)
            preds_delta_ph = np.asarray(self.model_ph.predict(feat_df), dtype=float)
            preds_cl = np.clip(cur_cl + preds_delta_cl, 0.05, 5.0)
            preds_ph = np.clip(cur_ph + preds_delta_ph, 6.8, 8.8)
        else:
            base_dict = env.base.to_dict()
            grid_data = {k: np.repeat(v, n_grid) for k, v in base_dict.items()}
            grid_data["hypochlorite_dosing_pct"] = pct_flat
            grid_data["hypochlorite_dosing_hours"] = hours_flat
            grid_data["hypo_dosing_percentage"] = pct_flat
            grid_data["hypo_dosing_hours"] = hours_flat
            grid_data["daily_pump_cl2_delivered_ppm"] = delivered_cl2

            grid_df = pd.DataFrame(grid_data)
            for col in self.all_numeric_features:
                if col not in grid_df.columns:
                    grid_df[col] = self.fill_values.get(col, 0.0)
                grid_df[col] = pd.to_numeric(grid_df[col], errors="coerce").fillna(self.fill_values.get(col, 0.0))
            for col in self.categorical_features:
                if col not in grid_df.columns:
                    grid_df[col] = "unknown"
                grid_df[col] = grid_df[col].fillna("unknown").astype(str)

            feat_df = grid_df[self.categorical_features + self.all_numeric_features]
            X = self.preprocessor.transform(feat_df)
            preds_cl = np.asarray(self.model_cl.predict(X), dtype=float)
            preds_ph = np.asarray(self.model_ph.predict(X), dtype=float)

        # Vectorized penalty and cost calculation
        cl_pen = np.maximum(0.0, CLIENT_CL_TARGET_MIN - preds_cl) + np.maximum(0.0, preds_cl - CLIENT_CL_TARGET_MAX)
        ph_pen = np.maximum(0.0, REG_PH_MIN - preds_ph) + np.maximum(0.0, preds_ph - REG_PH_MAX)
        total_pen = cl_pen + ph_pen
        cost = (pct_flat / 100.0) * hours_flat

        summary_df = pd.DataFrame({
            "hypochlorite_dosing_pct": pct_flat,
            "hypochlorite_dosing_hours": hours_flat,
            "pred_cl_next": np.round(preds_cl, 3),
            "pred_ph_next": np.round(preds_ph, 3),
            "cl_penalty": np.round(cl_pen, 4),
            "ph_penalty": np.round(ph_pen, 4),
            "total_penalty": np.round(total_pen, 4),
            "dosing_cost": np.round(cost, 3),
        }).sort_values(["total_penalty", "dosing_cost"])

        best = summary_df.iloc[0].to_dict()
        feasible = int((summary_df["total_penalty"] == 0).sum())

        urgency, reasons = _urgency(env.current_cl_raw, env.current_ph_raw)
        if best["total_penalty"] == 0:
            reasons.append(
                f"Optimal config found: {best['hypochlorite_dosing_pct']:.0f}% for "
                f"{best['hypochlorite_dosing_hours']:.1f}h → predicted "
                f"Cl={best['pred_cl_next']}, pH={best['pred_ph_next']}"
            )
        else:
            reasons.append(
                f"Best available: Cl penalty={best['cl_penalty']:.3f}, "
                f"pH penalty={best['ph_penalty']:.3f}"
            )

        return DosingResult(
            pool_id=pool_id,
            pool_volume_m3=pool_vol,
            current_readings={"ph": env.current_ph_raw, "free_chlorine": env.current_cl_raw},
            recommended_dosing={
                "hypochlorite_dosing_pct":   float(best["hypochlorite_dosing_pct"]),
                "hypochlorite_dosing_hours": float(best["hypochlorite_dosing_hours"]),
            },
            predicted_tomorrow={
                "free_chlorine": float(best["pred_cl_next"]), "ph": float(best["pred_ph_next"]),
            },
            feasible_configurations=feasible,
            top_3_configs=summary_df.head(3)[
                ["hypochlorite_dosing_pct", "hypochlorite_dosing_hours",
                 "pred_cl_next", "pred_ph_next", "total_penalty"]
            ].to_dict("records"),
            urgency=urgency,
            reasons=reasons,
        )



# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _urgency(current_cl, current_ph):
    from ml.config import REG_CHLORINE_MIN, REG_PH_MAX, REG_PH_MIN
    urgency = "Routine"
    reasons: list[str] = []
    if current_cl is not None and current_cl < REG_CHLORINE_MIN:
        urgency = "Immediate"
        reasons.append(f"⚠️ Current Cl ({current_cl:.2f}) BELOW {REG_CHLORINE_MIN} mg/L — pathogen risk")
    if current_ph is not None and (current_ph < REG_PH_MIN or current_ph > REG_PH_MAX):
        urgency = "Immediate"
        reasons.append(f"⚠️ Current pH ({current_ph:.2f}) outside [{REG_PH_MIN}–{REG_PH_MAX}]")
    if not reasons:
        reasons.append("Current readings within regulatory range")
    return urgency, reasons


class _FeatureEnv:
    """Tiny helper that holds a base row + the column lists the preprocessor
    expects, so the optimiser can cheaply mutate the control columns and
    re-transform for each grid point."""

    @classmethod
    def from_row(cls, row, all_numeric, categorical, fill_values,
                 sp_cl=SETPOINT_FREE_CHLORINE, sp_ph=SETPOINT_PH, sp_turb=SETPOINT_TURBIDITY):
        env = cls()
        env.all_numeric = all_numeric
        env.categorical = categorical
        env.fill_values = fill_values
        base = row.copy()
        for col in all_numeric:
            if col not in base.index or pd.isna(base.get(col, np.nan)):
                base[col] = fill_values.get(col, 0.0)
        for col in categorical:
            if col not in base.index or pd.isna(base.get(col, np.nan)):
                base[col] = "unknown"
        # Post-treatment setpoint features — setpoints are constant across
        # the dosing grid (the current reading is fixed); only the dosing
        # controls mutate per grid point. Deltas/rates are computed once
        # against the current reading and `days_since_last_visit`.
        sp_cl   = float(base.get("setpoint_free_chlorine", sp_cl))
        sp_ph   = float(base.get("setpoint_ph",            sp_ph))
        sp_turb = float(base.get("setpoint_turbidity",     sp_turb))
        cur_cl   = float(base.get("free_chlorine", fill_values.get("free_chlorine", sp_cl)))
        cur_ph   = float(base.get("ph",            fill_values.get("ph",            sp_ph)))
        cur_turb = float(base.get("turbidity",     fill_values.get("turbidity",     sp_turb)))
        gap = float(base.get("days_since_last_visit", fill_values.get("days_since_last_visit", 1.0)) or 1.0)
        gap = gap if gap > 0 else 1.0
        base["setpoint_free_chlorine"]  = sp_cl
        base["setpoint_ph"]             = sp_ph
        base["setpoint_turbidity"]      = sp_turb
        base["cl_degradation_from_setpoint"]   = sp_cl - cur_cl
        base["ph_drift_from_setpoint"]         = cur_ph - sp_ph
        base["turb_accumulation_from_setpoint"] = cur_turb - sp_turb
        base["cl_degradation_rate_from_setpoint"]    = (sp_cl - cur_cl) / gap
        base["ph_drift_rate_from_setpoint"]          = (cur_ph - sp_ph) / gap
        base["turb_accumulation_rate_from_setpoint"] = (cur_turb - sp_turb) / gap
        env.base = base
        env.current_cl_raw = float(row.get("free_chlorine", 0)) if "free_chlorine" in row and pd.notna(row.get("free_chlorine")) else None
        env.current_ph_raw = float(row.get("ph", 0)) if "ph" in row and pd.notna(row.get("ph")) else None
        env.current_cl = float(base.get("free_chlorine", fill_values.get("free_chlorine", 2.0)))
        env.current_ph = float(base.get("ph", fill_values.get("ph", 7.4)))
        env.pool_volume_m3 = float(base.get("pool_volume_m3", fill_values.get("pool_volume_m3", 225.0)))
        return env

    def base_row_copy(self):
        return self.base.copy()

    def frame(self, row):
        feat = pd.DataFrame([row])
        for col in self.all_numeric:
            if col not in feat.columns:
                feat[col] = self.fill_values.get(col, 0.0)
            feat[col] = pd.to_numeric(feat[col], errors="coerce").fillna(self.fill_values.get(col, 0.0))
        for col in self.categorical:
            if col not in feat.columns:
                feat[col] = "unknown"
            feat[col] = feat[col].fillna("unknown").astype(str)
        return feat[self.categorical + self.all_numeric]