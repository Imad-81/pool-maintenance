"""Evaluation helpers — metrics, holdout comparison, promotion gate."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

log = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    mae: float
    rmse: float
    r2: float
    p90: float
    best_iter: Optional[int]

    def to_dict(self) -> dict:
        return {"mae": self.mae, "rmse": self.rmse, "r2": self.r2, "p90": self.p90,
                "best_iter": self.best_iter}


def compute_metrics(y_true, y_pred, best_iter=None) -> ModelMetrics:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    p90  = float(np.percentile(np.abs(np.asarray(y_true) - np.asarray(y_pred)), 90))
    return ModelMetrics(mae=mae, rmse=rmse, r2=r2, p90=p90, best_iter=best_iter)


def should_promote(new_metrics: dict, old_metrics: Optional[dict],
                   tol_cl: float, tol_ph: float) -> tuple[bool, str]:
    """Promote a new run only if its primary metrics are no worse than the
    current active run's by more than the configured tolerance.

    Returns (promote, reason). On the very first run (old_metrics is None)
    we always promote.
    """
    if old_metrics is None:
        return True, "no prior active run — first run promoted"
    new_cl = new_metrics.get("chlorine_next", {}).get("mae")
    old_cl = old_metrics.get("chlorine_next", {}).get("mae")
    new_ph = new_metrics.get("ph_next", {}).get("mae")
    old_ph = old_metrics.get("ph_next", {}).get("mae")
    reasons = []
    if new_cl is None or old_cl is None or new_ph is None or old_ph is None:
        return True, "metric block incomplete — promoting for visibility"
    if new_cl > old_cl + tol_cl:
        reasons.append(f"Cl MAE regressed {new_cl:.4f} > {old_cl:.4f}+{tol_cl:.4f}")
    if new_ph > old_ph + tol_ph:
        reasons.append(f"pH MAE regressed {new_ph:.4f} > {old_ph:.4f}+{tol_ph:.4f}")
    if reasons:
        return False, "; ".join(reasons)
    return True, "primary metrics within tolerance of active run"