#!/usr/bin/env python3
"""
Autoregressive Chained Maintenance Predictor & Proactive Dosing Optimizer.

Chains the 24-hour multi-chemical models (Free Chlorine, pH, Turbidity, Active HOCl)
forward across a 7-to-14 day weather forecast horizon to:
1. Forecast multi-chemical degradation trajectories.
2. Detect the exact calendar date of legal or operational breaches:
   - Free Chlorine < 1.0 mg/L (Spanish legal limit)
   - Free Chlorine < 1.2 mg/L (Client safe buffer)
   - pH outside [7.2, 7.8]
   - Turbidity > 1.0 NTU (Cloudy water threshold)
   - Active HOCl < 0.6 mg/L (Sanitizing threshold)
3. Recommend optimal maintenance dispatch dates (24h lead time).
4. Inverse-solve for required shock dose or automated pump runtime to maintain compliance.
"""

import os
import sys
import argparse
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10


class MaintenancePredictor:
    def __init__(self,
                 model_cl_path: str = "models/best_daily_chlorine_model_LightGBM_Delta.pkl",
                 model_ph_path: str = "models/best_daily_ph_model_LightGBM_Delta.pkl",
                 model_turb_path: str = "models/best_daily_turbidity_model_LightGBM_Delta.pkl",
                 data_path: str = "data/processed/pool_daily_ml_ready.csv"):
        """Loads trained multi-chemical models and historical pool baseline feature matrix."""
        logger.info("Initializing Multi-Chemical Maintenance Predictor...")
        if not os.path.exists(model_cl_path):
            raise FileNotFoundError(f"Missing chlorine model: {model_cl_path}")
        if not os.path.exists(model_ph_path):
            raise FileNotFoundError(f"Missing pH model: {model_ph_path}")
        if not os.path.exists(model_turb_path):
            raise FileNotFoundError(f"Missing turbidity model: {model_turb_path}")

        self.model_cl = joblib.load(model_cl_path)
        self.model_ph = joblib.load(model_ph_path)
        self.model_turb = joblib.load(model_turb_path)

        logger.info(f"Loading reference dataset from {data_path}...")
        self.df_all = pd.read_csv(data_path)

        # Feature column definition (must strictly match training)
        exclude_cols = {
            'pool_clean', 'date', 'imputation_method',
            'imputation_confidence_score', 'free_chlorine_pure_forward_ppm',
            'is_observed_measurement_day', 'is_chemical_dosed_day',
            'is_train_split',
            'target_next_day_free_chlorine', 'target_delta_free_chlorine',
            'target_next_day_ph', 'target_delta_ph',
            'target_next_day_turbidity', 'target_delta_turbidity',
            'target_next_day_compliance_band', 'target_is_observed'
        }
        self.feature_cols = [c for c in self.df_all.columns if c not in exclude_cols]
        logger.info(f"Predictor initialized with {len(self.feature_cols)} input features.")

    def get_available_pools(self) -> List[str]:
        """Returns sorted list of all unique pool names in the dataset."""
        return sorted(self.df_all['pool_clean'].unique().tolist())

    def simulate_rollout(self,
                          pool_name: str,
                          start_date: Optional[str] = None,
                          horizon_days: int = 14,
                          custom_initial_state: Optional[Dict[str, float]] = None,
                          override_shock_ppm: float = 0.0,
                          override_extra_pump_hours: float = 0.0,
                          override_erodible_tablets_g: float = 0.0) -> pd.DataFrame:
        """
        Executes a step-by-step autoregressive rollout across the multi-day forecast horizon.
        """
        pool_df = self.df_all[self.df_all['pool_clean'] == pool_name].sort_values('date').reset_index(drop=True)
        if len(pool_df) == 0:
            raise ValueError(f"Pool '{pool_name}' not found in dataset.")

        if start_date is not None:
            match = pool_df[pool_df['date'] == start_date]
            if len(match) > 0:
                start_idx = match.index[0]
            else:
                start_idx = len(pool_df) - horizon_days - 1
                logger.warning(f"Start date {start_date} not exact match. Using index {start_idx} ({pool_df.loc[start_idx, 'date']}).")
        else:
            # Default to the last observed visit day that has at least horizon_days following
            obs_indices = pool_df[pool_df['is_observed_measurement_day'] == 1].index
            valid_obs = [i for i in obs_indices if i + horizon_days < len(pool_df)]
            start_idx = valid_obs[-1] if len(valid_obs) > 0 else max(0, len(pool_df) - horizon_days - 1)

        base_row = pool_df.iloc[start_idx].copy()
        vol = float(base_row['pool_volume'])
        area = float(base_row['pool_surface_area'])
        spec_surface = area / vol
        is_outdoor = float(base_row['outdoor_pool'])
        is_community = float(base_row['community_pool'])

        # Starting state on departure day 0
        if custom_initial_state is not None:
            curr_cl = float(custom_initial_state.get('free_chlorine', base_row['free_chlorine_post_ppm']))
            curr_ph = float(custom_initial_state.get('ph', base_row['ph_post']))
            curr_turb = float(custom_initial_state.get('turbidity', base_row['turbidity_post']))
            curr_cya = float(custom_initial_state.get('cya', base_row['cya_post_ppm']))
        else:
            curr_cl = float(base_row['free_chlorine_post_ppm'])
            curr_ph = float(base_row['ph_post'])
            curr_turb = float(base_row['turbidity_post'])
            curr_cya = float(base_row['cya_post_ppm'])

        # Apply any override shock dosage or pump adjustment
        if override_shock_ppm > 0:
            curr_cl = min(curr_cl + override_shock_ppm, 5.0)

        dt_start = pd.to_datetime(base_row['date'])
        hocl_frac_0 = 1.0 / (1.0 + 10.0 ** (curr_ph - 7.53))

        records = [{
            'step': 0,
            'date': dt_start.strftime('%Y-%m-%d'),
            'day_of_week': dt_start.strftime('%A'),
            'free_chlorine_ppm': round(curr_cl, 3),
            'ph': round(curr_ph, 2),
            'turbidity_ntu': round(curr_turb, 2),
            'active_hocl_ppm': round(curr_cl * hocl_frac_0, 3),
            'cya_ppm': round(curr_cya, 1),
            'water_temp_c': float(base_row['water_temperature_c']),
            'solar_rad_mj': float(base_row['solar_radiation_mj']),
            'max_temp_c': float(base_row['temperature_ambient_max_c']),
            'is_legal_compliant': curr_cl >= 1.0 and 7.2 <= curr_ph <= 7.8 and curr_turb <= 1.0,
            'is_safety_compliant': curr_cl >= 1.2 and 7.2 <= curr_ph <= 7.8 and curr_turb <= 1.0,
            'breach_reason': 'None'
        }]

        cl_history = [curr_cl, curr_cl, curr_cl]
        remaining_tablets_g = override_erodible_tablets_g

        # Step forward day by day
        for h in range(1, horizon_days + 1):
            future_row_idx = start_idx + h
            if future_row_idx < len(pool_df):
                future_ref_row = pool_df.iloc[future_row_idx].copy()
            else:
                future_ref_row = pool_df.iloc[-1].copy()

            dt_current = dt_start + timedelta(days=h)

            # Weather for this step
            rad = float(future_ref_row['forecast_solar_radiation_mj'])
            water_temp = float(future_ref_row['water_temperature_c'])
            filt_hours = float(future_ref_row['daily_filtration_hours']) + override_extra_pump_hours
            pump_dose = float(future_ref_row['daily_pump_cl2_delivered_ppm'])
            if override_extra_pump_hours > 0:
                pump_flow = float(future_ref_row.get('hypochlorite_pump_flow_rate', 4.0))
                hypo_pct = float(future_ref_row.get('hypo_dosing_percentage', 10.0))
                extra_mass_g = pump_flow * override_extra_pump_hours * (hypo_pct / 100.0) * 130.0
                pump_dose = min(pump_dose + (extra_mass_g / vol), 0.60)

            # Erodible tablet dissolution in skimmer
            if remaining_tablets_g > 0:
                eroded_g = remaining_tablets_g * 0.15 * (filt_hours / 10.0)
                remaining_tablets_g = max(0.0, remaining_tablets_g - eroded_g)
                tablet_cl_ppm = min((eroded_g * 0.90) / vol, 0.50)
                pump_dose = min(pump_dose + tablet_cl_ppm, 0.70)

            # Dynamic Kinetic & Physics Calculations
            k_photo = 0.025 * (rad / 20.0) * np.clip(spec_surface, 0.5, 3.0) * is_outdoor
            k_temp = 0.015 * np.clip(water_temp / 25.0, 0.5, 2.0)
            k_turb = 0.020 * np.clip(curr_turb, 0.1, 5.0)
            k_cya = 0.015 * np.clip(curr_cya / 50.0, 0.0, 0.8)
            k_bather = 0.025 * (1.0 if dt_current.weekday() >= 5 else 0.0) * is_community
            decay_k = np.clip(0.04 + k_photo + k_temp + k_turb + k_bather - k_cya, 0.02, 0.80)

            hocl_fraction = 1.0 / (1.0 + 10.0 ** (curr_ph - 7.53))
            turnover = np.clip((15.0 * filt_hours) / vol, 0.0, 10.0)

            # Build feature vector for prediction
            feats = future_ref_row[self.feature_cols].copy()
            feats['free_chlorine_post_ppm'] = curr_cl
            feats['ph_post'] = curr_ph
            feats['ph'] = curr_ph
            feats['turbidity_post'] = curr_turb
            feats['turbidity'] = curr_turb
            feats['cya_post_ppm'] = curr_cya
            feats['active_hocl_ppm'] = curr_cl * hocl_fraction
            feats['free_chlorine_post_lag1'] = cl_history[-1]
            feats['free_chlorine_post_lag2'] = cl_history[-2]
            feats['free_chlorine_rolling_mean_3'] = np.mean(cl_history[-3:])
            feats['theoretical_decay_k'] = decay_k
            feats['theoretical_retained_chlorine'] = curr_cl * np.exp(-decay_k)
            feats['active_hocl_fraction'] = hocl_fraction
            feats['daily_turnover_ratio'] = turnover
            feats['hocl_demand_proxy'] = hocl_fraction * curr_cl * turnover * (1.0 + curr_turb)
            feats['daily_pump_cl2_delivered_ppm'] = pump_dose
            feats['shock_dosage_ppm'] = 0.0
            feats['clarifier_added_grams'] = 0.0
            feats['acid_added_grams'] = 0.0

            df_input = pd.DataFrame([feats])

            # Multi-Chemical Next-Day Predictions
            pred_delta_cl = float(self.model_cl.predict(df_input)[0])
            pred_delta_ph = float(self.model_ph.predict(df_input)[0])
            pred_delta_turb = float(self.model_turb.predict(df_input)[0])

            # Update chemistry
            next_cl = float(np.clip(curr_cl + pred_delta_cl, 0.0, 5.0))
            next_ph = float(np.clip(curr_ph + pred_delta_ph, 6.5, 8.8))
            next_turb = float(np.clip(curr_turb + pred_delta_turb, 0.05, 5.0))
            next_hocl = next_cl / (1.0 + 10.0 ** (next_ph - 7.53))

            # Breach evaluation
            breaches = []
            if next_cl < 1.0:
                breaches.append(f"Chlorine Legal Breach ({next_cl:.2f} < 1.00)")
            elif next_cl < 1.2:
                breaches.append(f"Chlorine Safe Buffer Breach ({next_cl:.2f} < 1.20)")
            if next_ph < 7.2 or next_ph > 7.8:
                breaches.append(f"pH Breach ({next_ph:.2f} outside [7.2, 7.8])")
            if next_turb > 1.0:
                breaches.append(f"Turbidity Breach ({next_turb:.2f} > 1.00 NTU)")
            if next_hocl < 0.6:
                breaches.append(f"Disinfectant HOCl Insufficient ({next_hocl:.2f} < 0.60)")

            is_legal = (next_cl >= 1.0) and (7.2 <= next_ph <= 7.8) and (next_turb <= 1.0)
            is_safe = (next_cl >= 1.2) and (7.2 <= next_ph <= 7.8) and (next_turb <= 1.0) and (next_hocl >= 0.6)

            records.append({
                'step': h,
                'date': dt_current.strftime('%Y-%m-%d'),
                'day_of_week': dt_current.strftime('%A'),
                'free_chlorine_ppm': round(next_cl, 3),
                'ph': round(next_ph, 2),
                'turbidity_ntu': round(next_turb, 2),
                'active_hocl_ppm': round(next_hocl, 3),
                'cya_ppm': round(curr_cya, 1),
                'water_temp_c': round(water_temp, 1),
                'solar_rad_mj': round(rad, 1),
                'max_temp_c': float(future_ref_row['temperature_ambient_max_c']),
                'is_legal_compliant': is_legal,
                'is_safety_compliant': is_safe,
                'breach_reason': "; ".join(breaches) if breaches else 'Compliant'
            })

            # Roll states forward
            curr_cl = next_cl
            curr_ph = next_ph
            curr_turb = next_turb
            cl_history.append(curr_cl)

        return pd.DataFrame(records)

    def analyze_maintenance_schedule(self, df_rollout: pd.DataFrame) -> Dict[str, Any]:
        """Identifies exact breach dates, failure mechanisms, and recommended visit schedule."""
        legal_breach_row = df_rollout[~df_rollout['is_legal_compliant']]
        safety_breach_row = df_rollout[~df_rollout['is_safety_compliant']]

        legal_breach_date = legal_breach_row.iloc[0]['date'] if len(legal_breach_row) > 0 else None
        legal_breach_day = legal_breach_row.iloc[0]['day_of_week'] if len(legal_breach_row) > 0 else None
        legal_breach_step = int(legal_breach_row.iloc[0]['step']) if len(legal_breach_row) > 0 else None

        safety_breach_date = safety_breach_row.iloc[0]['date'] if len(safety_breach_row) > 0 else None
        safety_breach_day = safety_breach_row.iloc[0]['day_of_week'] if len(safety_breach_row) > 0 else None
        safety_breach_step = int(safety_breach_row.iloc[0]['step']) if len(safety_breach_row) > 0 else None

        # Recommended maintenance visit: 24h prior to the earliest breach
        if safety_breach_step is not None and safety_breach_step > 1:
            rec_step = safety_breach_step - 1
            rec_date = df_rollout.loc[rec_step, 'date']
            rec_day = df_rollout.loc[rec_step, 'day_of_week']
            rec_status = "URGENT DISPATCH RECOMMENDED"
        elif safety_breach_step == 1:
            rec_date = df_rollout.loc[0, 'date']
            rec_day = df_rollout.loc[0, 'day_of_week']
            rec_status = "IMMEDIATE ATTENTION REQUIRED TODAY"
        else:
            rec_date = df_rollout.iloc[-1]['date']
            rec_day = df_rollout.iloc[-1]['day_of_week']
            rec_status = "POOL REMAINS FULLY COMPLIANT FOR 14 DAYS"

        return {
            "legal_breach_date": legal_breach_date,
            "legal_breach_day": legal_breach_day,
            "days_until_legal_breach": legal_breach_step,
            "safety_breach_date": safety_breach_date,
            "safety_breach_day": safety_breach_day,
            "days_until_safety_breach": safety_breach_step,
            "recommended_dispatch_date": rec_date,
            "recommended_dispatch_day": rec_day,
            "dispatch_urgency": rec_status,
            "initial_state": {
                "date": df_rollout.loc[0, 'date'],
                "free_chlorine": df_rollout.loc[0, 'free_chlorine_ppm'],
                "ph": df_rollout.loc[0, 'ph'],
                "turbidity": df_rollout.loc[0, 'turbidity_ntu'],
                "active_hocl": df_rollout.loc[0, 'active_hocl_ppm']
            }
        }

    def optimize_dosing(self, pool_name: str, start_date: Optional[str] = None, target_days: int = 7) -> Dict[str, Any]:
        """
        Inverse-solves for the exact shock dose or pump extension required
        to ensure zero breaches over the specified target days.
        """
        base_rollout = self.simulate_rollout(pool_name, start_date, horizon_days=target_days)
        base_analysis = self.analyze_maintenance_schedule(base_rollout)

        if base_analysis['days_until_safety_breach'] is None or base_analysis['days_until_safety_breach'] >= target_days:
            return {
                "dosing_needed": False,
                "recommended_shock_ppm": 0.0,
                "recommended_shock_grams": 0.0,
                "recommended_extra_pump_hours": 0.0,
                "explanation": f"Pool will remain compliant for all {target_days} days under current settings."
            }

        pool_row = self.df_all[self.df_all['pool_clean'] == pool_name].iloc[0]
        vol = float(pool_row['pool_volume'])

        # Check breach types
        reasons = " ".join(base_rollout['breach_reason'].tolist())
        cl_breach = "Chlorine" in reasons or "HOCl" in reasons
        turb_breach = "Turbidity" in reasons
        ph_breach = "pH" in reasons

        prescriptions = []

        # 1. Chlorine Optimization
        best_shock_ppm = 0.0
        shock_grams = 0.0
        best_extra_hours = 0.0

        if cl_breach:
            for test_shock in np.arange(0.2, 3.5, 0.2):
                test_rollout = self.simulate_rollout(pool_name, start_date, horizon_days=target_days, override_shock_ppm=test_shock)
                test_reasons = " ".join(test_rollout['breach_reason'].tolist())
                if "Chlorine" not in test_reasons and "HOCl" not in test_reasons:
                    best_shock_ppm = round(float(test_shock), 2)
                    break
            shock_grams = round((best_shock_ppm * vol) / 0.65, 1)

            for extra_h in np.arange(1.0, 8.0, 1.0):
                test_rollout_pump = self.simulate_rollout(pool_name, start_date, horizon_days=target_days, override_extra_pump_hours=extra_h)
                test_reasons_pump = " ".join(test_rollout_pump['breach_reason'].tolist())
                if "Chlorine" not in test_reasons_pump and "HOCl" not in test_reasons_pump:
                    best_extra_hours = round(float(extra_h), 1)
                    break

            best_tablets_g = 0.0
            for tabs_g in [200.0, 400.0, 600.0, 1000.0]:
                test_rollout_tabs = self.simulate_rollout(pool_name, start_date, horizon_days=target_days, override_erodible_tablets_g=tabs_g)
                test_reasons_tabs = " ".join(test_rollout_tabs['breach_reason'].tolist())
                if "Chlorine" not in test_reasons_tabs and "HOCl" not in test_reasons_tabs:
                    best_tablets_g = tabs_g
                    break

            cl_options = []
            if shock_grams > 0:
                cl_options.append(f"Add {shock_grams:,.0f}g Cal-Hypo shock granules (+{best_shock_ppm:.2f} ppm)")
            if best_tablets_g > 0:
                num_tabs = max(1, int(round(best_tablets_g / 200.0)))
                cl_options.append(f"Place {num_tabs} slow-release 200g Trichlor tablet(s) ({best_tablets_g:,.0f}g) in skimmer")
            if best_extra_hours > 0:
                cl_options.append(f"Extend pump timer by +{best_extra_hours}h/day")
            if not cl_options:
                cl_options.append("Place 1 slow-release 200g Trichlor tablet in skimmer")

            prescriptions.append("Chlorine: " + " OR ".join(cl_options))

        # 2. Turbidity Optimization
        clarifier_prescription = None
        if turb_breach:
            flovil_tabs = max(1, int(round(vol / 50.0)))
            superklar_ml = int(round(vol * 2.5))
            clarifier_prescription = f"Turbidity: Add {flovil_tabs} Flovil clarifier tablet(s) in skimmer or {superklar_ml} mL Bayrol Superklar coagulant"
            prescriptions.append(clarifier_prescription)

        # 3. pH Optimization
        ph_prescription = None
        if ph_breach:
            max_ph = base_rollout['ph'].max()
            min_ph = base_rollout['ph'].min()
            if max_ph > 7.8:
                acid_grams = round((max_ph - 7.4) * 75.0 * (vol / 10.0), 0)
                ph_prescription = f"pH High: Add {acid_grams:,.0f}g pH-minus (sodium bisulfate) to reduce pH from {max_ph:.2f} to 7.40"
                prescriptions.append(ph_prescription)
            elif min_ph < 7.2:
                ph_prescription = f"pH Low: Add pH-plus to raise pH from {min_ph:.2f} to 7.40"
                prescriptions.append(ph_prescription)

        return {
            "dosing_needed": True,
            "target_compliance_days": target_days,
            "recommended_shock_ppm": best_shock_ppm,
            "recommended_shock_grams_cal_hypo": shock_grams,
            "alternative_extra_pump_runtime_hours_per_day": best_extra_hours,
            "clarifier_prescription": clarifier_prescription,
            "ph_prescription": ph_prescription,
            "pool_volume_m3": vol,
            "prescription": " | ".join(prescriptions) if prescriptions else "Routine maintenance recommended."
        }


def plot_maintenance_forecast(df_rollout: pd.DataFrame, pool_name: str,
                              analysis: Dict[str, Any],
                              output_png: str = "reports/figures/20_maintenance_forecast_rollout.png"):
    """Generates an executive diagnostic visualization of the 14-day forecast and breach detection."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    dates = pd.to_datetime(df_rollout['date'])

    # 1. Free Chlorine & Active HOCl Trajectory
    ax1 = axes[0]
    ax1.plot(dates, df_rollout['free_chlorine_ppm'], color='#0284c7', marker='o', linewidth=2.5, label='Free Chlorine (ppm)')
    ax1.plot(dates, df_rollout['active_hocl_ppm'], color='#0d9488', linestyle='--', linewidth=2.0, label='Active HOCl Disinfectant (ppm)')
    ax1.axhline(1.0, color='#dc2626', linestyle='--', linewidth=1.5, label='Spanish Legal Minimum (1.0 ppm)')
    ax1.axhline(1.2, color='#f59e0b', linestyle=':', linewidth=1.5, label='Client Safe Operational Buffer (1.2 ppm)')
    ax1.axhspan(1.0, 3.0, alpha=0.08, color='green', label='Regulatory Ideal Band (1.0–3.0 ppm)')

    # Breach marker
    if analysis['safety_breach_date']:
        breach_dt = pd.to_datetime(analysis['safety_breach_date'])
        ax1.axvline(breach_dt, color='#dc2626', linestyle='-', linewidth=2.0, alpha=0.8)
        ax1.text(breach_dt, 3.5, f"  SERVICE REQUIRED\n  {analysis['safety_breach_day']} ({analysis['safety_breach_date']})",
                 color='#b91c1c', fontweight='bold', fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='#b91c1c'))

    ax1.set_ylabel('Chlorine (mg/L / ppm)', fontweight='bold')
    ax1.set_title(f'14-Day Proactive Maintenance Forecast: {pool_name} | Status: {analysis["dispatch_urgency"]}', fontweight='bold', fontsize=12)
    ax1.set_ylim(-0.1, 4.5)
    ax1.legend(loc='upper right', frameon=True, fontsize=9)

    # 2. pH Dynamics Trajectory
    ax2 = axes[1]
    ax2.plot(dates, df_rollout['ph'], color='#7c3aed', marker='s', linewidth=2.2, label='Predicted Water pH')
    ax2.axhline(7.2, color='#dc2626', linestyle='--', label='Regulatory Minimum (7.2)')
    ax2.axhline(7.8, color='#dc2626', linestyle='--', label='Regulatory Maximum (7.8)')
    ax2.axhspan(7.2, 7.8, alpha=0.08, color='green', label='Ideal pH Window (7.2–7.8)')
    ax2.set_ylabel('pH Units', fontweight='bold')
    ax2.set_ylim(6.8, 8.4)
    ax2.legend(loc='upper right', frameon=True, fontsize=9)

    # 3. Turbidity & Weather Solar Forcing
    ax3 = axes[2]
    ax3_twin = ax3.twinx()

    p_turb = ax3.plot(dates, df_rollout['turbidity_ntu'], color='#ea580c', marker='^', linewidth=2.0, label='Turbidity (NTU)')
    ax3.axhline(1.0, color='#dc2626', linestyle='--', label='Cloudy Water Threshold (1.0 NTU)')
    ax3.set_ylabel('Turbidity (NTU)', color='#ea580c', fontweight='bold')
    ax3.set_ylim(0.0, 2.5)

    p_rad = ax3_twin.plot(dates, df_rollout['solar_rad_mj'], color='#eab308', linestyle=':', linewidth=2.0, label='Forecast Solar Radiation (MJ/m²)')
    ax3_twin.set_ylabel('Solar Radiation (MJ/m²)', color='#ca8a04', fontweight='bold')
    ax3_twin.grid(False)

    lines = p_turb + p_rad
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper right', frameon=True, fontsize=9)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%a %d-%b'))
    ax3.set_xlabel('Forecast Calendar Date', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    logger.info(f"Saved maintenance forecast visualization to {output_png}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Chemical Proactive Maintenance Predictor & Dosing Optimizer")
    parser.add_argument("--pool", type=str, default="Arenales Playa Fase 6   (9930)", help="Pool identifier")
    parser.add_argument("--days", type=int, default=14, help="Forecast horizon in days (default: 14)")
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD), default latest visit")
    parser.add_argument("--optimize", action="store_true", default=True, help="Compute proactive chemical dosing prescription")
    parser.add_argument("--target-days", type=int, default=7, help="Target compliance duration for dosing optimizer")
    args = parser.parse_args()

    predictor = MaintenancePredictor()

    pool_name = args.pool
    logger.info(f"--- Running 14-Day Autoregressive Rollout for Pool: '{pool_name}' ---")

    df_rollout = predictor.simulate_rollout(pool_name, start_date=args.start_date, horizon_days=args.days)
    analysis = predictor.analyze_maintenance_schedule(df_rollout)

    print("\n" + "=" * 78)
    print(f" PROACTIVE MAINTENANCE FORECAST REPORT: {pool_name}")
    print("=" * 78)
    init = analysis['initial_state']
    print(f" Departure Date:      {init['date']} | Free Cl: {init['free_chlorine']:.2f} ppm | pH: {init['ph']:.2f} | Turbidity: {init['turbidity']:.2f} NTU")
    print(f" Active HOCl Power:   {init['active_hocl']:.2f} ppm (Target: 1.0–1.5 mg/L)")
    print("-" * 78)
    print(f" DISPATCH URGENCY:    {analysis['dispatch_urgency']}")
    print(f" Safety Breach Date:  {analysis['safety_breach_day']}, {analysis['safety_breach_date']} (in {analysis['days_until_safety_breach']} days)")
    print(f" Legal Breach Date:   {analysis['legal_breach_day']}, {analysis['legal_breach_date']} (in {analysis['days_until_legal_breach']} days)")
    print(f" RECOMMENDED VISIT:   {analysis['recommended_dispatch_day']}, {analysis['recommended_dispatch_date']}")
    print("-" * 78)
    print(" 14-DAY DAILY TRAJECTORY FORECAST:")
    print(df_rollout[['date', 'day_of_week', 'free_chlorine_ppm', 'active_hocl_ppm', 'ph', 'turbidity_ntu', 'breach_reason']].to_string(index=False))
    print("-" * 78)

    if args.optimize:
        dosing = predictor.optimize_dosing(pool_name, start_date=args.start_date, target_days=args.target_days)
        print(f" PROACTIVE DOSING PRESCRIPTION (for {args.target_days}-day compliance):")
        if dosing['dosing_needed']:
            if dosing.get('recommended_shock_ppm', 0) > 0:
                print(f"   * Chlorine Shock Dose:         +{dosing['recommended_shock_ppm']:.2f} ppm ({dosing['recommended_shock_grams_cal_hypo']:,.0f}g Cal-Hypo)")
                print(f"   * Alternative Pump Extension:  +{dosing['alternative_extra_pump_runtime_hours_per_day']:.1f} hours/day")
            if dosing.get('clarifier_prescription'):
                print(f"   * Clarifier Treatment:         {dosing['clarifier_prescription']}")
            if dosing.get('ph_prescription'):
                print(f"   * pH Acid/Base Treatment:      {dosing['ph_prescription']}")
            print(f"   * Comprehensive Action Plan:   {dosing['prescription']}")
        else:
            print(f"   * Status:                      {dosing['explanation']}")
        print("=" * 78 + "\n")

    plot_maintenance_forecast(df_rollout, pool_name, analysis)


if __name__ == "__main__":
    main()
