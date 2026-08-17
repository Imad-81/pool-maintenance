"""
Atomic artifact writer — never leaves a half-written model on disk.

A training run writes everything under `models/<run_id>/.tmp/`, then on
success renames that directory to `models/<run_id>/` and atomically rewrites
`models/latest.json` so the backend can hot-swap by re-reading one pointer
file. A crashed run is detected at next startup by the stale `.tmp` suffix
and is discarded.

The `inference_config.json` emitted here carries `run_id`, `feature_schema`
and `data_hash` — the backend (and `ml.inference`) refuse to load any
artifact whose `feature_schema` does not match the loaded preprocessor's
feature names, which catches silent feature drift introduced by a refactor.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


def _dir_hash(df_master: pd.DataFrame) -> str:
    """Stable SHA1 of the master dataset's content (shape + dtypes + first
    2k rows) — good enough to spot a dataset swap without hashing 35MB."""
    h = hashlib.sha1()
    h.update(str(df_master.shape).encode())
    h.update(str({c: str(dt) for c, dt in df_master.dtypes.items()}).encode())
    h.update(pd.util.hash_pandas_object(df_master.head(2000), index=False).values.tobytes())
    return h.hexdigest()[:16]


class ArtifactStore:
    """Owns the on-disk layout for one run's artifacts."""

    def __init__(self, models_dir: Path, run_id: str):
        self.models_dir = models_dir
        self.run_id = run_id
        self.final_dir = models_dir / run_id
        self.tmp_dir = models_dir / f"{run_id}.tmp"

    # --- staging ----------------------------------------------------------

    def __enter__(self) -> "ArtifactStore":
        if self.tmp_dir.exists():
            log.warning("Stale temp dir %s found from a crashed run — removing.", self.tmp_dir)
            shutil.rmtree(self.tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=False)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self._promote()
        else:
            log.error("Training failed — discarding staged artifacts at %s", self.tmp_dir)
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
        return False  # do not suppress the exception

    # --- writers (used during training) -----------------------------------

    def write_model_xgb(self, name: str, model) -> None:
        model.save_model(self.tmp_dir / f"xgb_{name}.json")

    def write_pickle(self, name: str, obj: Any) -> None:
        with open(self.tmp_dir / name, "wb") as f:
            pickle.dump(obj, f)

    def write_json(self, name: str, payload: dict) -> None:
        with open(self.tmp_dir / name, "w") as f:
            json.dump(payload, f, indent=2, default=str)

    def write_text(self, name: str, text: str) -> None:
        (self.tmp_dir / name).write_text(text)

    # --- promotion --------------------------------------------------------

    def _promote(self) -> None:
        if self.final_dir.exists():
            # Keep prior run: archive under models/ archive subdir.
            archive = self.models_dir / "archive"
            archive.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.move(str(self.final_dir), str(archive / f"{self.run_id}_{ts}"))
        os.replace(self.tmp_dir, self.final_dir)
        log.info("Promoted artifacts to %s", self.final_dir)

    # --- latest pointer (called by the trainer after success) --------------

    @staticmethod
    def write_latest_pointer(models_dir: Path, run_id: str) -> None:
        """Atomically update models/latest.json so the backend hot-swaps."""
        pointer = {"active_run_id": run_id, "updated_at": datetime.now().isoformat()}
        tmp = models_dir / "latest.json.tmp"
        final = models_dir / "latest.json"
        tmp.write_text(json.dumps(pointer, indent=2, default=str))
        os.replace(tmp, final)
        log.info("Updated latest pointer -> %s", run_id)

    @staticmethod
    def read_latest_pointer(models_dir: Path) -> str | None:
        p = models_dir / "latest.json"
        if not p.exists():
            return None
        return json.loads(p.read_text()).get("active_run_id")


def build_inference_config(
    cfg,               # PipelineConfig
    run_id: str,
    df_master: pd.DataFrame,
    fill_values: dict,
    all_numeric_features: list[str],
    categorical_features: list[str],
    feature_names: list[str],
    control_features: list[str],
    weather_current: list[str],
    weather_cumulative: list[str],
    weather_tomorrow: list[str],
    results: dict,
    shap_results: dict,
) -> dict:
    from ml.config import client_targets, reg_thresholds, treatment_setpoint

    return {
        "pipeline_version": "v6",
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(),
        "data_hash": _dir_hash(df_master),
        "feature_schema": feature_names,
        "fill_values": {k: float(v) if pd.notna(v) else 0.0 for k, v in fill_values.items()},
        "all_numeric_features": all_numeric_features,
        "categorical_features": categorical_features,
        "feature_names": feature_names,
        "control_features": control_features,
        "weather_current_features": weather_current,
        "weather_cumulative_features": weather_cumulative,
        "weather_tomorrow_features": weather_tomorrow,
        "alicante_coords": {
            "lat": cfg.alicante_lat, "lon": cfg.alicante_lon, "timezone": cfg.alicante_tz,
        },
        "regulatory_thresholds": reg_thresholds(),
        "client_targets": client_targets(),
        "treatment_setpoint": treatment_setpoint(cfg),
        "dosing_grid": {
            "pct_step": cfg.dosing_pct_step, "hours_step": cfg.dosing_hours_step,
        },
        "metrics": results,
        "shap_top_features": shap_results,
    }