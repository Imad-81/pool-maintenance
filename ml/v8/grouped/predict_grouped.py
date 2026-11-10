"""
V8 Grouped Models — Real-time Inference Engine & Routing Service.

Routes incoming pool predictions to either:
  - Group 1 (gaps < 7 days): Short-range summer & active turnover model (67 features)
  - Group 2 (gaps 7–30 days): Multi-week off-season kinetic decay model (70 features)

Usage (Python):
    from ml.v8.grouped.predict_grouped import V8GroupedPredictor
    predictor = V8GroupedPredictor()
    pred = predictor.predict_pool_next_visit("ALTAVISTA II (1027)", gap_days=3)
    print(pred)

Usage (CLI):
    python -m ml.v8.grouped.predict_grouped --pool "ALTAVISTA II (1027)" --gap 3
    python -m ml.v8.grouped.predict_grouped --pool "ALTAVISTA II (1027)" --gap 14
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.v8.config_v8 import (
    DEFAULT_V8_CONFIG,
    REG_CHLORINE_MIN,
    REG_PH_MAX,
    REG_PH_MIN,
    V8Config,
)
from ml.v8.grouped.evaluate_grouped import load_group_models
from ml.v8.grouped.features_grouped import (
    apply_fill_values,
    build_features_for_group,
)
from ml.v8.sparse_dataset import add_gap_columns, load_sparse_visits

log = logging.getLogger("ml.v8.grouped.predict_grouped")

REG_CHLORINE_MAX = 2.00
REG_TURBIDITY_MAX = 5.00


class V8GroupedPredictor:
    """
    Production inference service for V8 dual-regime grouped pool models.
    Automatically routes prediction queries to Group 1 or Group 2 based on
    the elapsed interval (gap_days).
    """

    def __init__(self, cfg: V8Config = DEFAULT_V8_CONFIG):
        self.cfg = cfg
        self.models_g1, self.fill_g1, self.cfg_g1 = load_group_models("G1", cfg)
        self.models_g2, self.fill_g2, self.cfg_g2 = load_group_models("G2", cfg)

        self.features_g1 = self.cfg_g1["all_numeric_features"]
        self.features_g2 = self.cfg_g2["all_numeric_features"]

        self._obs_cache: Optional[pd.DataFrame] = None
        log.info("V8GroupedPredictor initialized with G1 and G2 models.")

    def _get_obs(self) -> pd.DataFrame:
        if self._obs_cache is None:
            df = load_sparse_visits(self.cfg)
            self._obs_cache = add_gap_columns(df)
        return self._obs_cache

    def route_regime(self, gap_days: float) -> Tuple[str, str]:
        """
        Determine group routing and return (group_id, warning_message).
        """
        if gap_days < 7.0:
            return "G1", ""
        elif gap_days <= 30.0:
            return "G2", ""
        else:
            return (
                "G2",
                f"Warning: Interval ({gap_days:.1f} days) exceeds 30-day regular threshold. "
                "Defaulting to Group 2 kinetic decay model.",
            )

    def predict_from_features(
        self,
        feature_row: pd.Series,
        gap_days: float,
    ) -> Dict[str, Any]:
        """
        Run prediction given a single pre-calculated feature series and gap.
        """
        group_id, warning = self.route_regime(gap_days)

        if group_id == "G1":
            models = self.models_g1
            fill_values = self.fill_g1
            feature_names = self.features_g1
        else:
            models = self.models_g2
            fill_values = self.fill_g2
            feature_names = self.features_g2

        # Convert feature series to single-row DataFrame
        df_row = pd.DataFrame([feature_row])
        for c in feature_names:
            if c not in df_row.columns:
                df_row[c] = fill_values.get(c, 0.0)

        X = apply_fill_values(df_row[feature_names], fill_values).values

        cl_pred = float(models["chlorine_next"].predict(X)[0])
        ph_pred = float(models["ph_next"].predict(X)[0])
        tb_pred = float(models["turbidity_next"].predict(X)[0])

        # Floor predictions at physically meaningful boundaries
        cl_pred = max(0.0, round(cl_pred, 3))
        ph_pred = round(ph_pred, 3)
        tb_pred = max(0.01, round(tb_pred, 3))

        # Check regulatory compliance (RD 742/2013)
        cl_breach = cl_pred < REG_CHLORINE_MIN or cl_pred > REG_CHLORINE_MAX
        ph_breach = ph_pred < REG_PH_MIN or ph_pred > REG_PH_MAX
        tb_breach = tb_pred > REG_TURBIDITY_MAX

        status = "COMPLIANT"
        breaches = []
        if cl_breach:
            status = "ACTION_REQUIRED"
            breaches.append(f"Chlorine breach expected ({cl_pred:.2f} mg/L, legal: 0.50–2.00)")
        if ph_breach:
            status = "ACTION_REQUIRED"
            breaches.append(f"pH breach expected ({ph_pred:.2f}, legal: 7.20–8.00)")
        if tb_breach:
            status = "ACTION_REQUIRED"
            breaches.append(f"Turbidity breach expected ({tb_pred:.2f} NTU, legal: ≤5.00)")

        return {
            "routed_group": group_id,
            "gap_days": gap_days,
            "predicted_free_chlorine_ppm": cl_pred,
            "predicted_ph": ph_pred,
            "predicted_turbidity_ntu": tb_pred,
            "regulatory_status": status,
            "is_chlorine_breach": cl_breach,
            "is_ph_breach": ph_breach,
            "is_turbidity_breach": tb_breach,
            "breach_details": breaches,
            "warning": warning,
        }

    def predict_pool_next_visit(
        self,
        pool_name: str,
        gap_days: float = 3.0,
    ) -> Dict[str, Any]:
        """
        Retrieve latest recorded visit for pool_name from dataset,
        compute engineered features with the specified gap_days, and predict.
        """
        obs = self._get_obs()
        pool_data = obs[obs["pool_clean"] == pool_name].sort_values("date")

        if pool_data.empty:
            matches = [p for p in obs["pool_clean"].unique() if pool_name.lower() in p.lower()]
            if matches:
                raise ValueError(
                    f"Pool '{pool_name}' not found. Did you mean: '{matches[0]}'?"
                )
            raise ValueError(f"Pool '{pool_name}' not found in fleet database.")

        group_id, _ = self.route_regime(gap_days)

        # Build features for this pool's history
        pool_feat = build_features_for_group(pool_data, group_id, self.cfg)

        # Take the most recent visit
        latest_row = pool_feat.iloc[-1].copy()
        # Override gap_since with the queried gap
        latest_row["gap_since"] = gap_days
        latest_row["gap_log"] = float(np.log1p(gap_days))

        pred = self.predict_from_features(latest_row, gap_days)
        pred["pool_clean"] = pool_name
        pred["as_of_date"] = str(pool_data["date"].iloc[-1].date())
        pred["last_cl_post"] = float(latest_row.get("cl_post", np.nan))
        pred["last_ph_post"] = float(latest_row.get("ph_post", np.nan))
        pred["last_cya_post"] = float(latest_row.get("cya_post", np.nan))
        return pred


def main() -> None:
    parser = argparse.ArgumentParser(description="V8 Grouped Model Inference CLI")
    parser.add_argument("--pool", type=str, default="ALTAVISTA II (1027)",
                        help="Exact or partial pool name")
    parser.add_argument("--gap", type=float, default=3.0,
                        help="Projected days until next technician visit (e.g., 3.0 or 14.0)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    predictor = V8GroupedPredictor()

    print("\n" + "=" * 70)
    print(f"  V8 GROUPED PREDICTOR — INFERENCE QUERY")
    print("=" * 70)
    print(f"  Target Pool:    {args.pool}")
    print(f"  Projected Gap:  {args.gap:.1f} days")

    res = predictor.predict_pool_next_visit(args.pool, gap_days=args.gap)

    print("\n  [Routing & Previous Measurements]")
    print(f"    As of Date:         {res['as_of_date']}")
    print(f"    Routed Regime:      {res['routed_group']} ({'Short-range / Active' if res['routed_group'] == 'G1' else 'Multi-week / Decay'})")
    print(f"    Last Free Chlorine: {res['last_cl_post']:.2f} mg/L")
    print(f"    Last pH:            {res['last_ph_post']:.2f}")
    print(f"    Last Stabilizer:    {res['last_cya_post']:.1f} ppm CYA")

    print("\n  [Predicted Water Quality at Next Visit]")
    print(f"    Free Chlorine:      {res['predicted_free_chlorine_ppm']:.3f} mg/L")
    print(f"    pH:                 {res['predicted_ph']:.3f}")
    print(f"    Turbidity:          {res['predicted_turbidity_ntu']:.3f} NTU")

    print("\n  [Spanish Regulatory Status (RD 742/2013)]")
    status_str = f"    Status:             {res['regulatory_status']}"
    print(status_str)
    if res["breach_details"]:
        print("    Breach Alerts:")
        for b in res["breach_details"]:
            print(f"      • {b}")
    else:
        print("    No regulatory breaches expected across this interval.")

    if res.get("warning"):
        print(f"\n  Notice: {res['warning']}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

