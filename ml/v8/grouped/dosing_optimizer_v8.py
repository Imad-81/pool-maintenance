"""
V8 Chemical Dosing Optimizer.

Computes the optimal chemical dosage (shock chlorine grams, pump dosing ppm,
and acid grams) required to keep a swimming pool compliant with Spanish Health
Regulations (RD 742/2013) across an unmonitored interval (gap_days).

Usage (Python):
    from ml.v8.grouped.dosing_optimizer_v8 import V8DosingOptimizer
    optimizer = V8DosingOptimizer()
    rec = optimizer.optimize_pool("ALTAVISTA II (1027)", gap_days=3.0)
    print(rec)

Usage (CLI):
    python -m ml.v8.grouped.dosing_optimizer_v8 --pool "ALTAVISTA II (1027)" --gap 3.0
    python -m ml.v8.grouped.dosing_optimizer_v8 --pool "ALTAVISTA II (1027)" --gap 7.0
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

from ml.v8.config_v8 import (
    DEFAULT_V8_CONFIG,
    REG_CHLORINE_MIN,
    REG_PH_MAX,
    REG_PH_MIN,
    V8Config,
)
from ml.v8.grouped.features_grouped import build_features_for_group
from ml.v8.grouped.predict_grouped import REG_CHLORINE_MAX, V8GroupedPredictor
from ml.v8.sparse_dataset import add_gap_columns, load_sparse_visits

log = logging.getLogger("ml.v8.grouped.dosing_optimizer_v8")

# Target setpoints (Client target window inside legal bounds)
TARGET_CL_IDEAL = 1.30   # mg/L (client target 1.0–1.5 mg/L)
TARGET_PH_IDEAL = 7.40   # ideal neutral pH for bather comfort & chlorine activity


@dataclass
class DosingRecommendation:
    pool_name: str
    gap_days: float
    pool_volume_m3: float
    current_cl: float
    current_ph: float
    current_cya: float
    recommended_shock_ppm: float
    recommended_shock_grams: float
    recommended_acid_grams: float
    recommended_pump_cl_ppm: float
    projected_cl_at_next_visit: float
    projected_ph_at_next_visit: float
    projected_turbidity_at_next_visit: float
    compliance_status: str
    action_summary: str


class V8DosingOptimizer:
    """
    Evaluates candidate chemical additions through the V8 predictive models
    to discover the minimal chemical dosage required for sustained compliance.
    """

    def __init__(self, cfg: V8Config = DEFAULT_V8_CONFIG):
        self.cfg = cfg
        self.predictor = V8GroupedPredictor(cfg)
        self._obs_cache: Optional[pd.DataFrame] = None

    def _get_obs(self) -> pd.DataFrame:
        if self._obs_cache is None:
            df = load_sparse_visits(self.cfg)
            self._obs_cache = add_gap_columns(df)
        return self._obs_cache

    def optimize_pool(
        self,
        pool_name: str,
        gap_days: float = 3.0,
    ) -> DosingRecommendation:
        """
        Calculates optimal shock chlorine and acid required for pool_name across gap_days.
        """
        obs = self._get_obs()
        pool_data = obs[obs["pool_clean"] == pool_name].sort_values("date")

        if pool_data.empty:
            matches = [p for p in obs["pool_clean"].unique() if pool_name.lower() in p.lower()]
            if matches:
                raise ValueError(f"Pool '{pool_name}' not found. Did you mean '{matches[0]}'?")
            raise ValueError(f"Pool '{pool_name}' not found.")

        group_id, _ = self.predictor.route_regime(gap_days)
        pool_feat = build_features_for_group(pool_data, group_id, self.cfg)

        base_row = pool_feat.iloc[-1].copy()
        base_row["gap_since"] = gap_days
        base_row["gap_log"] = float(np.log1p(gap_days))

        vol_m3 = float(base_row.get("pool_volume", 50.0))
        if pd.isna(vol_m3) or vol_m3 <= 0:
            vol_m3 = 50.0

        cur_cl = float(base_row.get("cl_post", 1.5))
        cur_ph = float(base_row.get("ph_post", 7.4))
        cur_cya = float(base_row.get("cya_post", 35.0))

        # Candidate grid for chlorine boost (ppm) and acid addition (grams)
        # For pools already high in chlorine, 0.0 is candidate
        shock_candidates = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
        acid_candidates = [0.0, 150.0, 300.0, 500.0, 800.0, 1200.0]

        best_config = None
        best_cost = float("inf")

        for shock_ppm in shock_candidates:
            for acid_g in acid_candidates:
                candidate_row = base_row.copy()

                # Update candidate dosing features
                candidate_row["shock_ppm"] = shock_ppm
                candidate_row["acid_grams"] = acid_g
                candidate_row["is_dosed"] = 1.0 if (shock_ppm > 0 or acid_g > 0) else 0.0

                # Chemical physics adjustments to post-treatment state
                candidate_cl_post = cur_cl + shock_ppm
                # 100g of dry acid typically drops 50m³ pH by ~0.1
                ph_drop = (acid_g / 100.0) * 0.10 * (50.0 / vol_m3)
                candidate_ph_post = max(6.8, cur_ph - ph_drop)

                candidate_row["cl_post"] = candidate_cl_post
                candidate_row["ph_post"] = candidate_ph_post
                candidate_row["cl_dosage_boost"] = shock_ppm
                candidate_row["total_cl_input"] = float(candidate_row.get("pump_cl_ppm", 0.0)) + shock_ppm

                pred = self.predictor.predict_from_features(candidate_row, gap_days)
                pred_cl = pred["predicted_free_chlorine_ppm"]
                pred_ph = pred["predicted_ph"]

                # Cost function:
                # 1. Heavy penalty for Spanish regulatory breach (RD 742/2013)
                penalty = 0.0
                if pred_cl < REG_CHLORINE_MIN:
                    penalty += 500.0 * (REG_CHLORINE_MIN - pred_cl) ** 2
                elif pred_cl > REG_CHLORINE_MAX:
                    penalty += 300.0 * (pred_cl - REG_CHLORINE_MAX) ** 2

                if pred_ph < REG_PH_MIN:
                    penalty += 400.0 * (REG_PH_MIN - pred_ph) ** 2
                elif pred_ph > REG_PH_MAX:
                    penalty += 400.0 * (pred_ph - REG_PH_MAX) ** 2

                # 2. Closeness to ideal setpoint
                dist_cl = (pred_cl - TARGET_CL_IDEAL) ** 2
                dist_ph = (pred_ph - TARGET_PH_IDEAL) ** 2

                # 3. Parsimony: minimize chemical additions
                chem_cost = 0.05 * shock_ppm + 0.0001 * acid_g

                total_cost = penalty + dist_cl + dist_ph + chem_cost

                if total_cost < best_cost:
                    best_cost = total_cost
                    # Convert ppm shock to grams of commercial calcium hypochlorite (approx 70% available Cl)
                    grams_cl = (shock_ppm * vol_m3) / 0.70
                    best_config = {
                        "shock_ppm": shock_ppm,
                        "shock_grams": round(grams_cl, 1),
                        "acid_grams": acid_g,
                        "pred_cl": pred_cl,
                        "pred_ph": pred_ph,
                        "pred_turb": pred["predicted_turbidity_ntu"],
                        "status": "COMPLIANT" if penalty == 0 else "PARTIAL_COMPLIANCE",
                    }

        # Action summary
        actions = []
        if best_config["shock_grams"] > 0:
            actions.append(f"Add {best_config['shock_grams']:.0f}g Chlorine Shock ({best_config['shock_ppm']:.1f} ppm)")
        if best_config["acid_grams"] > 0:
            actions.append(f"Add {best_config['acid_grams']:.0f}g pH Minus (Dry Acid)")
        if not actions:
            actions.append("No chemical adjustment needed — pool is fully stable across this interval.")

        return DosingRecommendation(
            pool_name=pool_name,
            gap_days=gap_days,
            pool_volume_m3=vol_m3,
            current_cl=cur_cl,
            current_ph=cur_ph,
            current_cya=cur_cya,
            recommended_shock_ppm=best_config["shock_ppm"],
            recommended_shock_grams=best_config["shock_grams"],
            recommended_acid_grams=best_config["acid_grams"],
            recommended_pump_cl_ppm=float(base_row.get("pump_cl_ppm", 0.0)),
            projected_cl_at_next_visit=best_config["pred_cl"],
            projected_ph_at_next_visit=best_config["pred_ph"],
            projected_turbidity_at_next_visit=best_config["pred_turb"],
            compliance_status=best_config["status"],
            action_summary="; ".join(actions),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="V8 Chemical Dosing Optimizer CLI")
    parser.add_argument("--pool", type=str, default="ALTAVISTA II (1027)",
                        help="Exact or partial pool name")
    parser.add_argument("--gap", type=float, default=3.0,
                        help="Interval in days until next scheduled visit (default: 3.0)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    opt = V8DosingOptimizer()

    print("\n" + "=" * 76)
    print("  V8 CHEMICAL DOSING OPTIMIZER — RECOMMENDATION ENGINE")
    print("=" * 76)
    print(f"  Target Pool:       {args.pool}")
    print(f"  Forecast Horizon:  {args.gap:.1f} Days Until Next Visit\n")

    rec = opt.optimize_pool(args.pool, gap_days=args.gap)

    print("  [Pool Parameters & Current State]")
    print(f"    Pool Volume:     {rec.pool_volume_m3:.1f} m³")
    print(f"    Current Cl:      {rec.current_cl:.2f} mg/L")
    print(f"    Current pH:      {rec.current_ph:.2f}")
    print(f"    CYA Stabilizer:  {rec.current_cya:.1f} ppm")

    print("\n  [Optimized Chemical Dosage Recommendation]")
    print(f"    Shock Dosage:    {rec.recommended_shock_grams:.0f} grams ({rec.recommended_shock_ppm:.1f} ppm)")
    print(f"    pH Minus Acid:   {rec.recommended_acid_grams:.0f} grams")
    print(f"    Action Plan:     {rec.action_summary}")

    print("\n  [Projected Water Quality at Next Visit]")
    print(f"    Free Chlorine:   {rec.projected_cl_at_next_visit:.3f} mg/L  (Legal: 0.50 – 2.00 mg/L)")
    print(f"    pH:              {rec.projected_ph_at_next_visit:.3f}         (Legal: 7.20 – 8.00)")
    print(f"    Turbidity:       {rec.projected_turbidity_at_next_visit:.3f} NTU       (Legal: ≤ 5.00 NTU)")
    print(f"    Compliance:      {rec.compliance_status}")

    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
