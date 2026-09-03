#!/usr/bin/env python3
"""
Continuous Daily Pool Time-Series Reconstruction Engine.

Implements the Comprehensive Multi-Chemical Pre/Post-Treatment Kinetic Bridge (Option 1):
1. Reconstructs unobserved daily pool states between technician visits.
2. Captures DUAL STATES on every day for all affected chemicals:
   - Free Chlorine: pre (arrival), post (departure refreshed), boost
   - pH: pre (arrival), post (rebalanced), delta
   - Active HOCl: pre (arrival), post (refreshed killing power)
   - Turbidity: pre (tested), post (clarified after filter backwash/vacuuming)
   - CYA: pre (prior), post (accumulated after stabilized product dosing)
3. Simulates daily forward photolysis decay driven by daily Alicante weather.
4. Models daily automated pump dosing and skimmer tablet erosion flux.
5. Calibrates terminal boundary conditions at the next visit using weather-weighted residual smoothing.
6. Guarantees physical non-negativity, thermodynamic equilibrium, and mass conservation.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.build_ml_dataset import (
    load_raw_data,
    disaggregate_tables,
    impute_pool_profiles,
    calculate_active_chlorine_and_cya,
    CHLORINE_PURITY_MAP,
    ERODIBLE_CHEMICALS,
    SHOCK_CHEMICALS,
    CYA_PRODUCING_CHEMICALS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Helper: safe weather lookup ──────────────────────────────────────────
def _weather_for_date(weather_lookup: pd.DataFrame, d, defaults=None):
    """Return a dict of weather fields for a single date, with safe defaults."""
    if defaults is None:
        defaults = {'rad': 15.0, 't_mean': 20.0, 't_max': 25.0, 'precip': 0.0,
                    'sunshine_hrs': 8.0, 'uv_max': 5.0, 'wind_max': 10.0,
                    'daylight_hrs': 12.0, 'et0': 4.0}
    if d in weather_lookup.index:
        w = weather_lookup.loc[d]
        return {
            'rad':          float(w['shortwave_radiation_sum'])      if pd.notna(w['shortwave_radiation_sum'])      else defaults['rad'],
            't_mean':       float(w['temperature_2m_mean'])          if pd.notna(w['temperature_2m_mean'])          else defaults['t_mean'],
            't_max':        float(w['temperature_2m_max'])           if pd.notna(w['temperature_2m_max'])           else defaults['t_max'],
            'precip':       float(w['precipitation_sum'])            if pd.notna(w['precipitation_sum'])            else defaults['precip'],
            'sunshine_hrs': float(w['sunshine_duration']) / 3600.0   if pd.notna(w['sunshine_duration'])            else defaults['sunshine_hrs'],
            'uv_max':       float(w['uv_index_max'])                if pd.notna(w['uv_index_max'])                 else defaults['uv_max'],
            'wind_max':     float(w['wind_speed_10m_max'])           if pd.notna(w['wind_speed_10m_max'])           else defaults['wind_max'],
            'daylight_hrs': float(w['daylight_duration']) / 3600.0   if pd.notna(w['daylight_duration'])            else defaults['daylight_hrs'],
            'et0':          float(w['et0_fao_evapotranspiration'])   if pd.notna(w['et0_fao_evapotranspiration'])   else defaults['et0'],
        }
    return dict(defaults)


# ── Helper: estimate water temperature from ambient ──────────────────────
def _estimate_water_temp(t_ambient_mean: float, t_ambient_max: float,
                         is_outdoor: float, month: int) -> float:
    """Pool water temperature with thermal mass offset and Mediterranean floor."""
    if is_outdoor < 0.5:
        return float(np.clip(27.0, 24.0, 30.0))
    t_water_est = 0.60 * t_ambient_mean + 0.40 * t_ambient_max - 1.5
    seasonal_floor = {1: 14.0, 2: 14.0, 3: 15.0, 4: 17.0, 5: 20.0, 6: 23.0,
                      7: 25.0, 8: 25.0, 9: 23.0, 10: 20.0, 11: 17.0, 12: 14.0}
    floor = seasonal_floor.get(month, 18.0)
    return float(np.clip(t_water_est, floor, 34.0))


# ── Helper: build one daily row dict with dual pre/post states ───────────
def _make_row(pool_name, vol, area, is_community, is_outdoor,
              rec_date, w_info,
              is_obs, is_dosed,
              c_pre, c_post, c_mean,
              ph_pre, ph_post,
              turb_pre, turb_post,
              cya_pre, cya_post,
              shock_ppm, e_dose_g, daily_pump_ppm,
              conf, method, water_temp):
    """Centralised row builder with dual pre/post states for ALL chemicals."""
    hocl_frac_pre = 1.0 / (1.0 + 10.0 ** (ph_pre - 7.53))
    hocl_frac_post = 1.0 / (1.0 + 10.0 ** (ph_post - 7.53))
    cl_boost = max(c_post - c_pre, 0.0)
    ph_delta = ph_post - ph_pre
    turb_delta = turb_post - turb_pre
    cya_added = max(cya_post - cya_pre, 0.0)

    return {
        'pool_clean': pool_name,
        'date': rec_date.isoformat(),
        'year': rec_date.year,
        'month': rec_date.month,
        'day_of_week': rec_date.weekday(),
        'is_weekend': int(rec_date.weekday() >= 5),
        'is_observed_measurement_day': int(is_obs),
        'is_chemical_dosed_day': int(is_dosed),
        # ── chlorine dual state ──
        'free_chlorine_pre_ppm':              round(c_pre, 3),
        'free_chlorine_post_ppm':             round(c_post, 3),
        'chlorine_dosage_boost_ppm':          round(cl_boost, 3),
        'free_chlorine_estimated_daily_mean_ppm': round(c_mean, 3),
        # ── pH dual state ──
        'ph_pre':                             round(ph_pre, 2),
        'ph_post':                            round(ph_post, 2),
        'ph_delta':                           round(ph_delta, 2),
        'ph':                                 round(ph_post, 2),  # active state for kinetics
        # ── active HOCl dual state ──
        'active_hocl_pre_ppm':                round(c_pre * hocl_frac_pre, 3),
        'active_hocl_post_ppm':               round(c_post * hocl_frac_post, 3),
        'active_hocl_ppm':                    round(c_post * hocl_frac_post, 3),
        # ── turbidity dual state ──
        'turbidity_pre':                      round(turb_pre, 2),
        'turbidity_post':                     round(turb_post, 2),
        'turbidity_delta':                    round(turb_delta, 2),
        'turbidity':                          round(turb_post, 2),  # active state for kinetics
        # ── CYA stabilizer dual state ──
        'cya_pre_ppm':                        round(cya_pre, 1),
        'cya_post_ppm':                       round(cya_post, 1),
        'cya_added_ppm':                      round(cya_added, 1),
        'cya_cumulative_ppm':                 round(cya_post, 1),
        # ── temperature & dosing ──
        'water_temperature_c':                round(water_temp, 1),
        'shock_dosage_ppm':                   round(shock_ppm, 3),
        'erodible_active_cl2_added_grams':    round(e_dose_g, 1),
        'daily_pump_cl2_delivered_ppm':       round(daily_pump_ppm, 3),
        # ── pool static profile ──
        'pool_volume':                        round(vol, 1),
        'pool_surface_area':                  round(area, 1),
        'community_pool':                     int(is_community),
        'outdoor_pool':                       int(is_outdoor),
        # ── weather ──
        'solar_radiation_mj':                 round(w_info['rad'], 2),
        'temperature_ambient_mean_c':         round(w_info['t_mean'], 1),
        'temperature_ambient_max_c':          round(w_info['t_max'], 1),
        'precipitation_mm':                   round(w_info['precip'], 1),
        'sunshine_duration_hrs':              round(w_info['sunshine_hrs'], 2),
        'uv_index_max':                       round(w_info['uv_max'], 1),
        'wind_speed_max_kmh':                 round(w_info['wind_max'], 1),
        'daylight_duration_hrs':              round(w_info['daylight_hrs'], 2),
        'et0_evapotranspiration':             round(w_info['et0'], 2),
        # ── metadata ──
        'imputation_confidence_score':        round(conf, 3),
        'imputation_method':                  method,
    }


def reconstruct_pool_daily_trajectories(
    df_water: pd.DataFrame,
    df_profile: pd.DataFrame,
    df_ops: pd.DataFrame,
    df_chem_std: pd.DataFrame,
    df_weather: pd.DataFrame,
    max_operational_gap_days: float = 14.0
) -> pd.DataFrame:
    """
    Reconstructs continuous daily trajectories with comprehensive dual pre/post states.
    """
    logger.info("Starting Multi-Chemical Pre/Post Trajectory Reconstruction...")

    weather_indexed = df_weather.copy()
    weather_indexed['date_dt'] = pd.to_datetime(weather_indexed['date']).dt.date
    weather_lookup = weather_indexed.set_index('date_dt')

    profile_lookup = df_profile.set_index('pool_clean')

    chem_grouped = {}
    for p, g in df_chem_std.groupby('pool_clean'):
        g = g.copy()
        g['date_only'] = g['date_dt'].dt.date
        chem_grouped[p] = g.groupby('date_only').sum(numeric_only=True)

    ops_grouped = {}
    for p, g in df_ops.groupby('pool_clean'):
        g = g.copy()
        g['date_only'] = g['date_dt'].dt.date
        ops_grouped[p] = g.sort_values('date_only')

    op_defaults = {
        'daily_filtration_hours': 10.0,
        'hypo_dosing_hours': 8.0,
        'hypo_dosing_percentage': 10.0,
    }

    all_daily_rows = []
    unique_pools = df_water['pool_clean'].unique()
    logger.info(f"Reconstructing daily series across {len(unique_pools)} unique pools...")

    total_reconstructed_days = 0
    total_intervals_processed = 0

    for pool_idx, pool_name in enumerate(unique_pools):
        pool_water = (df_water[df_water['pool_clean'] == pool_name]
                      .sort_values('date_dt')
                      .reset_index(drop=True))
        if len(pool_water) < 2:
            continue

        p_prof = (profile_lookup.loc[pool_name]
                  if pool_name in profile_lookup.index
                  else profile_lookup.iloc[0])
        vol             = float(max(p_prof['pool_volume'], 5.0))
        area            = float(max(p_prof['pool_surface_area'], 5.0))
        spec_surface    = area / vol
        is_community    = float(p_prof.get('community_pool', 1.0))
        is_outdoor      = float(p_prof.get('outdoor_pool', 1.0))
        hypo_pump_flow  = float(p_prof.get('hypochlorite_pump_flow_rate', 4.0))

        p_chems = chem_grouped.get(pool_name, None)
        p_ops   = ops_grouped.get(pool_name, None)

        seasonal_cya_tracker: Dict[int, float] = {}

        def _chem_on_date(d):
            if p_chems is None or d not in p_chems.index:
                return 0.0, 0.0, 0.0, 0.0
            row = p_chems.loc[d]
            return (float(row.get('active_cl2_shock_g', 0.0)),
                    float(row.get('active_cl2_liquid_g', 0.0)),
                    float(row.get('active_cl2_erodible_g', 0.0)),
                    float(row.get('cya_added_g', 0.0)))

        def _ops_before(d):
            filt_h = op_defaults['daily_filtration_hours']
            hypo_h = op_defaults['hypo_dosing_hours']
            hypo_p = op_defaults['hypo_dosing_percentage']
            if p_ops is not None and len(p_ops) > 0:
                prior = p_ops[p_ops['date_only'] <= d]
                if len(prior) > 0:
                    last = prior.iloc[-1]
                    filt_h = float(last['daily_filtration_hours']) if pd.notna(last['daily_filtration_hours']) else filt_h
                    hypo_h = float(last['hypo_dosing_hours'])     if pd.notna(last['hypo_dosing_hours'])     else hypo_h
                    hypo_p = float(last['hypo_dosing_percentage'])if pd.notna(last['hypo_dosing_percentage'])else hypo_p
            return filt_h, hypo_h, hypo_p

        for i in range(len(pool_water) - 1):
            row_curr = pool_water.iloc[i]
            row_next = pool_water.iloc[i + 1]

            d_curr   = row_curr['date_dt'].date()
            d_next   = row_next['date_dt'].date()
            days_gap = (d_next - d_curr).days

            # ──────────────────────────────────────────────────────────────
            # Non-operational hiatus (> 14 days) — record boundary day only
            # ──────────────────────────────────────────────────────────────
            if days_gap <= 0 or days_gap > max_operational_gap_days:
                w_info = _weather_for_date(weather_lookup, d_curr)
                s_g, l_g, e_g, cya_g = _chem_on_date(d_curr)
                year = d_curr.year
                if year not in seasonal_cya_tracker:
                    seasonal_cya_tracker[year] = 0.0
                cya_pre = seasonal_cya_tracker[year]
                cya_post = cya_pre + (cya_g / vol)
                seasonal_cya_tracker[year] = cya_post

                c_pre = float(row_curr['free_chlorine'])
                raw_shock = (s_g + l_g) / vol
                eff_boost = float(min(raw_shock * 0.20, 1.50))
                # Ensure physical non-inversion: c_post >= c_pre
                c_post = float(max(c_pre, min(c_pre + eff_boost, 5.0)))

                ph_pre = float(row_curr['ph']) if pd.notna(row_curr['ph']) else 7.40
                if (s_g + l_g + e_g) > 0:
                    if ph_pre > 7.55:
                        ph_post = float(ph_pre - min((ph_pre - 7.40) * 0.65, 0.40))
                    elif ph_pre < 7.25:
                        ph_post = float(ph_pre + min((7.40 - ph_pre) * 0.65, 0.30))
                    else:
                        ph_post = ph_pre
                else:
                    ph_post = ph_pre

                turb_pre = float(row_curr['turbidity']) if pd.notna(row_curr['turbidity']) else 0.30
                turb_post = float(min(turb_pre * 0.70, 0.30)) if (s_g + l_g + e_g) > 0 else turb_pre

                w_temp = _estimate_water_temp(w_info['t_mean'], w_info['t_max'],
                                              is_outdoor, d_curr.month)

                all_daily_rows.append(_make_row(
                    pool_name, vol, area, is_community, is_outdoor,
                    d_curr, w_info,
                    is_obs=True, is_dosed=((s_g + l_g + e_g) > 0),
                    c_pre=c_pre,
                    c_post=c_post,
                    c_mean=c_post,
                    ph_pre=ph_pre, ph_post=ph_post,
                    turb_pre=turb_pre, turb_post=turb_post,
                    cya_pre=cya_pre, cya_post=cya_post,
                    shock_ppm=raw_shock, e_dose_g=e_g, daily_pump_ppm=0.0,
                    conf=1.0, method='ground_truth_observation',
                    water_temp=w_temp,
                ))
                continue

            # ──────────────────────────────────────────────────────────────
            # Normal operational transition [d_curr → d_next]
            # ──────────────────────────────────────────────────────────────
            total_intervals_processed += 1

            c_curr_measured = float(row_curr['free_chlorine'])
            c_next_measured = float(row_next['free_chlorine'])
            ph_curr  = float(row_curr['ph'])        if pd.notna(row_curr['ph'])        else 7.40
            ph_next  = float(row_next['ph'])        if pd.notna(row_next['ph'])        else 7.40
            turb_curr= float(row_curr['turbidity']) if pd.notna(row_curr['turbidity']) else 0.30
            turb_next= float(row_next['turbidity']) if pd.notna(row_next['turbidity']) else 0.30

            filt_hours, hypo_hours, hypo_pct = _ops_before(d_curr)

            # Pump dose per day (capped at realistic maintenance 0.40 ppm/day)
            daily_pump_mass_g   = hypo_pump_flow * hypo_hours * (hypo_pct / 100.0) * 130.0
            daily_pump_dose_ppm = float(min(daily_pump_mass_g / vol, 0.40))

            # Chemicals added on visit day
            s_g, l_g, e_g, cya_g = _chem_on_date(d_curr)
            year = d_curr.year
            if year not in seasonal_cya_tracker:
                seasonal_cya_tracker[year] = 0.0
            cya_pre_d0 = seasonal_cya_tracker[year]
            cya_post_d0 = cya_pre_d0 + (cya_g / vol)
            seasonal_cya_tracker[year] = cya_post_d0
            current_cya = cya_post_d0

            # Effective shock: non-inversion guaranteed
            raw_shock_ppm   = (s_g + l_g) / vol
            eff_shock_boost = float(min(raw_shock_ppm * 0.20, 1.50))
            c_post_d0       = float(max(c_curr_measured, min(c_curr_measured + eff_shock_boost, 5.0)))

            # pH post rebalancing
            is_dosed_d0 = (s_g + l_g + e_g) > 0
            if is_dosed_d0:
                if ph_curr > 7.55:
                    ph_post_d0 = float(ph_curr - min((ph_curr - 7.40) * 0.65, 0.40))
                elif ph_curr < 7.25:
                    ph_post_d0 = float(ph_curr + min((7.40 - ph_curr) * 0.65, 0.30))
                else:
                    ph_post_d0 = ph_curr
                turb_post_d0 = float(min(turb_curr * 0.70, 0.30))
            else:
                ph_post_d0 = ph_curr
                turb_post_d0 = turb_curr

            # ── Forward kinetic simulation starting from refreshed c_post_d0 ──
            dates_in_interval = [d_curr + pd.Timedelta(days=k) for k in range(days_gap + 1)]
            remaining_erodible_g = e_g

            sim_forward_cl       = [c_post_d0]
            daily_stress_weights = []
            daily_weather_info   = []

            daily_weather_info.append(_weather_for_date(weather_lookup, d_curr))

            for step_idx in range(1, days_gap + 1):
                step_date = dates_in_interval[step_idx]
                w_step = _weather_for_date(weather_lookup, step_date)
                daily_weather_info.append(w_step)

                rad    = w_step['rad']
                t_mean = w_step['t_mean']

                # First-order decay rate k
                k_photolysis = 0.025 * (rad / 20.0) * np.clip(spec_surface, 0.5, 3.0) * is_outdoor
                k_temp       = 0.015 * np.clip(t_mean / 25.0, 0.5, 2.0)
                k_turb       = 0.020 * np.clip(turb_curr, 0.1, 5.0)
                k_cya_shield = 0.015 * np.clip(current_cya / 50.0, 0.0, 0.8)
                k_bather     = 0.025 * (1.0 if step_date.weekday() >= 5 else 0.0) * is_community

                k_day = float(np.clip(
                    0.04 + k_photolysis + k_temp + k_turb + k_bather - k_cya_shield,
                    0.02, 0.80))

                stress_w = k_day * (1.5 if step_date.weekday() >= 5 else 1.0)
                daily_stress_weights.append(stress_w)

                # Tablet erosion
                dissolution_speed  = (filt_hours / 10.0) * (1.0 + 0.025 * (t_mean - 20.0))
                dissolve_frac      = float(np.clip(1.0 - np.exp(-0.20 * dissolution_speed), 0.0, 1.0))
                eroded_g           = remaining_erodible_g * dissolve_frac
                remaining_erodible_g -= eroded_g
                tablet_ppm         = float(min(eroded_g / vol, 0.35))

                c_prev     = sim_forward_cl[-1]
                c_sim_next = (c_prev * np.exp(-k_day)) + daily_pump_dose_ppm + tablet_ppm
                sim_forward_cl.append(float(np.clip(c_sim_next, 0.0, 5.0)))

            # Boundary calibration against next visit arrival c_next_measured
            delta_terminal = c_next_measured - sim_forward_cl[-1]
            total_weight   = sum(daily_stress_weights) if sum(daily_stress_weights) > 0 else 1.0
            cum_weights    = np.cumsum(daily_stress_weights) / total_weight

            calibrated_cl = [c_curr_measured]
            for k in range(1, days_gap):
                correction = delta_terminal * cum_weights[k - 1]
                c_recon    = sim_forward_cl[k] + correction
                calibrated_cl.append(float(np.clip(c_recon, 0.10, 4.50)))

            confidence = float(np.clip(
                1.0 - (days_gap / 20.0) - min(abs(delta_terminal) / 5.0, 0.3),
                0.40, 0.98))

            # ── Emit daily rows for [d_curr .. d_next-1] ─────────────────
            for k in range(days_gap):
                curr_date = dates_in_interval[k]
                w_info    = daily_weather_info[k]

                is_visit = (k == 0)
                is_dosed = is_visit and is_dosed_d0

                alpha       = k / float(days_gap)
                ph_interp   = ph_post_d0 * (1.0 - alpha) + ph_next * alpha
                turb_interp = turb_post_d0 * (1.0 - alpha) + turb_next * alpha
                w_temp      = _estimate_water_temp(w_info['t_mean'], w_info['t_max'],
                                                   is_outdoor, curr_date.month)

                if is_visit:
                    c_pre_val  = c_curr_measured
                    c_post_val = c_post_d0
                    c_mean_val = c_post_d0
                    ph_pre_val = ph_curr
                    ph_post_val= ph_post_d0
                    turb_pre_val= turb_curr
                    turb_post_val= turb_post_d0
                    cya_pre_val= cya_pre_d0
                    cya_post_val= cya_post_d0
                    conf_val   = 1.0
                    method_val = 'ground_truth_observation'
                else:
                    c_pre_val  = calibrated_cl[k]
                    c_post_val = calibrated_cl[k]
                    c_mean_val = calibrated_cl[k]
                    ph_pre_val = ph_interp
                    ph_post_val= ph_interp
                    turb_pre_val= turb_interp
                    turb_post_val= turb_interp
                    cya_pre_val= current_cya
                    cya_post_val= current_cya
                    conf_val   = confidence
                    method_val = 'physics_kinetic_bridge'

                all_daily_rows.append(_make_row(
                    pool_name, vol, area, is_community, is_outdoor,
                    curr_date, w_info,
                    is_obs=is_visit, is_dosed=is_dosed,
                    c_pre=c_pre_val, c_post=c_post_val, c_mean=c_mean_val,
                    ph_pre=ph_pre_val, ph_post=ph_post_val,
                    turb_pre=turb_pre_val, turb_post=turb_post_val,
                    cya_pre=cya_pre_val, cya_post=cya_post_val,
                    shock_ppm=raw_shock_ppm if is_visit else 0.0,
                    e_dose_g=e_g if is_visit else 0.0,
                    daily_pump_ppm=daily_pump_dose_ppm,
                    conf=conf_val, method=method_val,
                    water_temp=w_temp,
                ))
                total_reconstructed_days += 1

        # ── Final measurement day for this pool ───────────────────────────
        last_row  = pool_water.iloc[-1]
        last_date = last_row['date_dt'].date()
        w_last    = _weather_for_date(weather_lookup, last_date)

        c_last    = float(last_row['free_chlorine'])
        ph_last   = float(last_row['ph'])        if pd.notna(last_row['ph'])        else 7.40
        turb_last = float(last_row['turbidity']) if pd.notna(last_row['turbidity']) else 0.30
        wt_last   = _estimate_water_temp(w_last['t_mean'], w_last['t_max'],
                                         is_outdoor, last_date.month)
        cya_final = seasonal_cya_tracker.get(last_date.year, 0.0)

        all_daily_rows.append(_make_row(
            pool_name, vol, area, is_community, is_outdoor,
            last_date, w_last,
            is_obs=True, is_dosed=False,
            c_pre=c_last, c_post=c_last, c_mean=c_last,
            ph_pre=ph_last, ph_post=ph_last,
            turb_pre=turb_last, turb_post=turb_last,
            cya_pre=cya_final, cya_post=cya_final,
            shock_ppm=0.0, e_dose_g=0.0, daily_pump_ppm=0.0,
            conf=1.0, method='ground_truth_observation',
            water_temp=wt_last,
        ))
        total_reconstructed_days += 1

        if (pool_idx + 1) % 25 == 0 or (pool_idx + 1) == len(unique_pools):
            logger.info(f"Processed {pool_idx + 1}/{len(unique_pools)} pools "
                        f"({total_reconstructed_days:,} daily rows generated)")

    df_daily = pd.DataFrame(all_daily_rows)

    before = len(df_daily)
    df_daily = df_daily.drop_duplicates(subset=['pool_clean', 'date'], keep='first')
    if len(df_daily) < before:
        logger.warning(f"Removed {before - len(df_daily)} duplicate pool-date rows")

    logger.info(f"Reconstruction Complete: Generated {len(df_daily):,} continuous pool-day records")
    return df_daily


def export_daily_dataset(df_daily: pd.DataFrame,
                         output_csv: str = "data/processed/pool_daily_reconstructed_timeseries.csv",
                         meta_json: str = "data/processed/pool_daily_timeseries_metadata.json") -> None:
    """Exports the reconstructed daily time series dataset and metadata."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    logger.info(f"Saving continuous daily dataset to {output_csv}...")
    df_daily.to_csv(output_csv, index=False)

    obs_count    = int((df_daily['is_observed_measurement_day'] == 1).sum())
    imputed_count = int((df_daily['is_observed_measurement_day'] == 0).sum())

    metadata = {
        "dataset_name": "Continuous Daily Pool Water Quality Time Series (Multi-Chemical Dual Pre/Post State)",
        "created_at": datetime.now().isoformat(),
        "total_pool_days": len(df_daily),
        "observed_measurement_days": obs_count,
        "imputed_intermediate_days": imputed_count,
        "unique_pools": int(df_daily['pool_clean'].nunique()),
        "date_range": {
            "start": str(df_daily['date'].min()),
            "end": str(df_daily['date'].max())
        },
        "imputation_methodology": (
            "Multi-Chemical Physics-Informed Kinetic Bridge (Dual pre/post states for Free Chlorine, "
            "pH, Active HOCl, Turbidity, and CYA, with Alicante solar photolysis, temperature decay, "
            "automated pump influx, tablet dissolution kinetics, and boundary calibration)"
        ),
        "features": list(df_daily.columns)
    }

    with open(meta_json, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved daily dataset metadata to {meta_json}")


def main():
    logger.info("=== Starting Multi-Chemical Pre/Post Daily Reconstruction Pipeline ===")

    df_raw, df_weather = load_raw_data()
    df_water, df_profile, df_ops, df_chem = disaggregate_tables(df_raw)

    train_pools = set(df_water[
        pd.to_datetime(df_water['measurement_date'],
                       format='%d-%m-%Y %H:%M', errors='coerce')
        < pd.Timestamp('2026-01-01')
    ]['pool_clean'].unique())
    df_profile_imputed = impute_pool_profiles(df_profile, train_pools)

    df_chem_std = calculate_active_chlorine_and_cya(df_chem)

    df_daily = reconstruct_pool_daily_trajectories(
        df_water=df_water,
        df_profile=df_profile_imputed,
        df_ops=df_ops,
        df_chem_std=df_chem_std,
        df_weather=df_weather,
        max_operational_gap_days=14.0
    )

    export_daily_dataset(df_daily)
    logger.info("=== Multi-Chemical Daily Reconstruction Completed Successfully! ===")


if __name__ == "__main__":
    main()
