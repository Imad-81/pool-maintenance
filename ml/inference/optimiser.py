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
                 all_numeric_features, categorical_features, fill_values):
        self.cfg = cfg
        self.model_cl = model_cl
        self.model_ph = model_ph
        self.preprocessor = preprocessor
        self.all_numeric_features = all_numeric_features
        self.categorical_features = categorical_features
        self.fill_values = fill_values
        self.pct_grid   = np.arange(0, 105, cfg.dosing_pct_step)
        self.hours_grid = np.arange(0, 25, cfg.dosing_hours_step)

    def optimise(self, pool_id: str, latest_row: pd.Series) -> DosingResult:
        env = _FeatureEnv.from_row(
            latest_row, self.all_numeric_features,
            self.categorical_features, self.fill_values,
        )
        current_cl = env.current_cl
        current_ph = env.current_cl and env.current_ph
        pool_vol = env.pool_volume_m3 or 50.0

        rows = []
        for pct in self.pct_grid:
            for hours in self.hours_grid:
                row = env.base_row_copy()
                if "hypochlorite_dosing_pct" in row:
                    row["hypochlorite_dosing_pct"] = float(pct)
                if "hypochlorite_dosing_hours" in row:
                    row["hypochlorite_dosing_hours"] = float(hours)
                X = self.preprocessor.transform(env.frame(row))
                pred_cl = float(self.model_cl.predict(X)[0])
                pred_ph = float(self.model_ph.predict(X)[0])
                cl_pen = max(0, CLIENT_CL_TARGET_MIN - pred_cl) + max(0, pred_cl - CLIENT_CL_TARGET_MAX)
                ph_pen = max(0, REG_PH_MIN - pred_ph) + max(0, pred_ph - REG_PH_MAX)
                total = cl_pen + ph_pen
                cost = (pct / 100.0) * float(hours)
                rows.append({
                    "hypochlorite_dosing_pct": float(pct),
                    "hypochlorite_dosing_hours": float(hours),
                    "pred_cl_next": round(pred_cl, 3),
                    "pred_ph_next": round(pred_ph, 3),
                    "cl_penalty": round(cl_pen, 4),
                    "ph_penalty": round(ph_pen, 4),
                    "total_penalty": round(total, 4),
                    "dosing_cost": round(cost, 3),
                })

        grid = pd.DataFrame(rows).sort_values(["total_penalty", "dosing_cost"])
        best = grid.iloc[0].to_dict()
        feasible = int((grid["total_penalty"] == 0).sum())

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
                "hypochlorite_dosing_pct":   best["hypochlorite_dosing_pct"],
                "hypochlorite_dosing_hours": best["hypochlorite_dosing_hours"],
            },
            predicted_tomorrow={
                "free_chlorine": best["pred_cl_next"], "ph": best["pred_ph_next"],
            },
            feasible_configurations=feasible,
            top_3_configs=grid.head(3)[
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
    def from_row(cls, row, all_numeric, categorical, fill_values):
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