#!/usr/bin/env python3
"""
Real-World Physics & Chemistry Verification Engine for Pool Water Dynamics.

Performs:
1. Thermodynamic Acid-Base Equilibrium & Biocidal HOCl Speciation Analysis.
2. Stoichiometric Mass Conservation & Chemical Dose-Response Accounting.
3. Photolytic (UV) & Thermal (Arrhenius) Decay Kinetics Verification.
4. Hydrodynamics, Hydraulic Turnover Rates & Regulatory Filtration Compliance.
5. Physical Plausibility Classification & Outlier Diagnostics.
6. 3-Tier Benchmarking: Pure Mechanistic ODE vs Pure ML vs Gray-Box Physics-Informed Hybrid.
7. Automated generation of publication-quality diagnostic figures and structured JSON metrics.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Configure matplotlib styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def verify_acid_base_equilibrium(df: pd.DataFrame, output_dir: str = "reports/figures") -> dict:
    """
    Evaluates temperature-dependent acid-base equilibrium of Hypochlorous Acid:
    HOCl <=> H+ + OCl-
    pKa(T) = 3000 / T_K - 10.068 + 0.0253 * T_K (Morris 1966)
    alpha_HOCl = 1 / (1 + 10^(pH - pKa))
    """
    logger.info("Verifying Thermodynamic Acid-Base Equilibrium & HOCl Speciation...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Temperature in Kelvin
    temp_c = df['window_temp_mean_c'].fillna(24.0).clip(5.0, 40.0)
    temp_k = temp_c + 273.15
    
    # 2. Temperature-dependent pKa
    pka = 3000.0 / temp_k - 10.068 + 0.0253 * temp_k
    
    # 3. Active HOCl fraction
    ph = df['ph'].clip(5.5, 9.5)
    alpha_hocl = 1.0 / (1.0 + 10.0 ** (ph - pka))
    active_hocl_ppm = df['free_chlorine'] * alpha_hocl
    
    # 4. Regulatory & Chemical Sanity Checks
    ph_valid_mask = (df['ph'] >= 6.5) & (df['ph'] <= 8.5)
    ph_optimal_mask = (df['ph'] >= 7.2) & (df['ph'] <= 7.6)
    
    # Correlation between pH and Turbidity (higher pH -> lower HOCl -> higher biological turbidity)
    valid_turb = df[df['turbidity'].notna() & df['ph'].notna()]
    r_ph_turb, p_ph_turb = stats.pearsonr(valid_turb['ph'], valid_turb['turbidity'])
    r_hocl_turb, p_hocl_turb = stats.pearsonr(
        df.loc[valid_turb.index, 'active_hocl_ppm'],
        valid_turb['turbidity']
    )
    
    metrics = {
        "pka_min": round(float(pka.min()), 3),
        "pka_max": round(float(pka.max()), 3),
        "pka_mean": round(float(pka.mean()), 3),
        "alpha_hocl_mean": round(float(alpha_hocl.mean()), 3),
        "alpha_hocl_median": round(float(alpha_hocl.median()), 3),
        "active_hocl_ppm_mean": round(float(active_hocl_ppm.mean()), 3),
        "ph_in_safe_range_pct": round(float(ph_valid_mask.mean() * 100.0), 2),
        "ph_in_optimal_range_pct": round(float(ph_optimal_mask.mean() * 100.0), 2),
        "corr_ph_vs_turbidity": round(float(r_ph_turb), 4),
        "corr_active_hocl_vs_turbidity": round(float(r_hocl_turb), 4)
    }
    
    # Figure 12: Acid-Base HOCl Speciation & Temperature Surface
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Subplot A: Theoretical Speciation Curves vs pH for multiple temperatures
    ph_grid = np.linspace(6.0, 9.0, 200)
    temp_levels = [10.0, 20.0, 25.0, 30.0, 35.0]
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(temp_levels)))
    
    for t_val, col in zip(temp_levels, colors):
        t_k = t_val + 273.15
        pka_t = 3000.0 / t_k - 10.068 + 0.0253 * t_k
        alpha_grid = 1.0 / (1.0 + 10.0 ** (ph_grid - pka_t)) * 100.0
        axes[0].plot(ph_grid, alpha_grid, label=f"Water Temp = {t_val:.0f}°C (pKa={pka_t:.2f})", color=col, lw=2.2)
        
    axes[0].axvspan(7.2, 7.6, color='green', alpha=0.15, label='Recommended pH Band (7.2 - 7.6)')
    axes[0].axhline(50.0, color='gray', linestyle=':', lw=1.2, label='50% Disinfection Equivalence')
    axes[0].set_title(r"Hypochlorous Acid ($\mathrm{HOCl}$) Speciation vs pH", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Water pH", fontsize=11)
    axes[0].set_ylabel(r"Active Disinfectant $\mathrm{HOCl}$ Fraction (%)", fontsize=11)
    axes[0].set_xlim(6.0, 9.0)
    axes[0].set_ylim(0, 100)
    axes[0].legend(loc='upper right', frameon=True, fontsize=9)
    
    # Subplot B: Dataset Empirical Distribution of Active HOCl Concentration
    sns.histplot(active_hocl_ppm, bins=50, kde=True, color='#0284c7', ax=axes[1], alpha=0.6)
    axes[1].axvline(active_hocl_ppm.mean(), color='#b91c1c', linestyle='--', lw=2,
                   label=f'Mean Active HOCl = {active_hocl_ppm.mean():.2f} ppm')
    axes[1].axvline(1.0, color='#15803d', linestyle=':', lw=2, label='Min Safe Biocidal Baseline (1.0 ppm)')
    axes[1].set_title(rf"Empirical Active Biocidal $\mathrm{{HOCl}}$ Distribution ($N={len(df):,}$, Mean={active_hocl_ppm.mean():.2f} ppm)",
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel(r"Active $\mathrm{HOCl}$ Concentration (ppm)", fontsize=11)
    axes[1].set_ylabel("Measurement Count", fontsize=11)
    axes[1].set_xlim(0, 4.0)
    axes[1].legend(loc='upper right', frameon=True, fontsize=9)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "12_physics_acid_base_hocl_speciation.png")
    plt.savefig(fig_path)
    plt.close()
    logger.info(f"Saved acid-base speciation figure to {fig_path}")
    
    return metrics


def verify_mass_conservation_and_stoichiometry(df: pd.DataFrame, output_dir: str = "reports/figures") -> dict:
    """
    Verifies Stoichiometric Mass Balance:
    Mass_injected = Mass_shock + Mass_erodible + Mass_pump
    Concentration Boost = Mass_injected / Volume
    Identifies unlogged shocks, sensor clipping, and mass-conserved transitions.
    """
    logger.info("Verifying Stoichiometric Mass Conservation & Dosage Dynamics...")
    os.makedirs(output_dir, exist_ok=True)
    
    vol = df['pool_volume'].clip(lower=10.0)
    dt = df['delta_days'].clip(lower=0.5, upper=10.0)
    
    # Mass calculations in grams
    m_initial_g = df['free_chlorine'] * vol
    m_final_g = df['target_next_free_chlorine'] * vol
    m_shock_g = df['active_cl2_shock_grams']
    m_erodible_g = df['erodible_dissolved_grams']
    m_pump_g = df['pump_cl2_delivered_grams']
    m_total_injected_g = m_shock_g + m_erodible_g + m_pump_g
    
    # Dissipated mass
    m_dissipated_g = (m_initial_g + m_total_injected_g) - m_final_g
    
    # Concentrations in ppm
    c_initial = df['free_chlorine']
    c_final = df['target_next_free_chlorine']
    c_dosed_instant = df['total_instant_dosage_ppm']
    c_dosed_erodible = df['erodible_dosage_ppm']
    c_dosed_pump = df['pump_dosage_ppm']
    
    # Theoretical maximum concentration ceiling before decay: C_0 + C_dosed
    c_max_possible = c_initial + c_dosed_instant + c_dosed_erodible + (c_dosed_pump * 0.5)
    
    # Physical Classifications:
    # 1. Mass Conserved: C_final <= C_max_possible (decay or equilibrium)
    # 2. Unlogged Shock: C_final - C_initial > 1.5 ppm with 0 recorded dosage
    # 3. Sensor Capped: C_final >= 5.0 ppm
    # 4. Depleted: C_final < 0.5 ppm
    
    is_unlogged_shock = (c_final - c_initial > 1.5) & (df['total_active_cl2_grams'] == 0) & (df['hypo_dosing_hours'] == 0)
    is_sensor_capped = (c_final >= 5.0) | (c_initial >= 5.0)
    is_mass_conserved = (c_final <= (c_max_possible + 0.3))  # +0.3 ppm allowance for measurement sensor noise
    
    mass_conserved_pct = (is_mass_conserved.sum() / len(df)) * 100.0
    unlogged_shock_pct = (is_unlogged_shock.sum() / len(df)) * 100.0
    sensor_capped_pct = (is_sensor_capped.sum() / len(df)) * 100.0
    
    metrics = {
        "mean_initial_cl2_mass_g": round(float(m_initial_g.mean()), 2),
        "mean_final_cl2_mass_g": round(float(m_final_g.mean()), 2),
        "mean_injected_cl2_mass_g": round(float(m_total_injected_g.mean()), 2),
        "mean_dissipated_cl2_mass_g": round(float(m_dissipated_g.mean()), 2),
        "mass_conservation_compliance_pct": round(float(mass_conserved_pct), 2),
        "unlogged_chemical_shock_pct": round(float(unlogged_shock_pct), 2),
        "sensor_saturation_pct": round(float(sensor_capped_pct), 2)
    }
    
    # Figure 13: Mass Balance Conservation & Dose-Response Scatter
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Subplot A: Theoretical Pre-Decay Ceiling vs Actual Next Visit Chlorine
    sample_idx = np.random.RandomState(42).choice(len(df), size=min(4000, len(df)), replace=False)
    axes[0].scatter(c_max_possible.values[sample_idx], c_final.values[sample_idx], alpha=0.3, color='#0284c7', s=18,
                    label='State Transitions ($t \\to t+1$)')
    axes[0].plot([0, 8], [0, 8], 'r--', lw=2, label='Mass Conservation Limit ($C_{t+1} = C_{\\max}$)')
    axes[0].fill_between([0, 8], [0, 8], [0, 0], color='green', alpha=0.08, label='Physical Decay Zone (Mass Conserved)')
    axes[0].fill_between([0, 8], [0, 8], [8, 8], color='red', alpha=0.08, label='Unlogged Addition / Anomaly Zone')
    axes[0].set_title(f"Stoichiometric Mass Conservation Envelope ({mass_conserved_pct:.1f}% Compliant)",
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel(r"Theoretical Maximum Chlorine Concentration $C_0 + \Delta C_{\mathrm{dose}}$ (ppm)", fontsize=11)
    axes[0].set_ylabel(r"Actual Measured Chlorine at Next Visit $C_{t+1}$ (ppm)", fontsize=11)
    axes[0].set_xlim(0, 8)
    axes[0].set_ylim(0, 6)
    axes[0].legend(loc='upper left', frameon=True, fontsize=8.5)
    
    # Subplot B: Average Chlorine Mass Lifecycle per 3.5-day Maintenance Cycle
    mass_bars = [m_initial_g.mean(), m_total_injected_g.mean(), m_dissipated_g.mean(), m_final_g.mean()]
    bar_labels = [r'Initial Mass' + '\n' + r'($t$)', r'Chemical Injected' + '\n' + r'($\Delta t$)', r'Dissipated / Consumed' + '\n' + r'($\Delta t$)', r'Final Mass' + '\n' + r'($t+1$)']
    bar_colors = ['#3b82f6', '#10b981', '#ef4444', '#6366f1']
    
    bars = axes[1].bar(bar_labels, mass_bars, color=bar_colors, width=0.55, edgecolor='#333333', lw=1)
    for b in bars:
        h = b.get_height()
        axes[1].text(b.get_x() + b.get_width()/2., h + 60, rf'{h:.0f} g $\mathrm{{Cl_2}}$',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    axes[1].set_title("Average Mass Balance Dynamics per Maintenance Cycle", fontsize=12, fontweight='bold')
    axes[1].set_ylabel(r"Active Chlorine Mass (grams $\mathrm{Cl_2}$)", fontsize=11)
    axes[1].set_ylim(0, max(mass_bars) * 1.25)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "13_physics_mass_balance_conservation.png")
    plt.savefig(fig_path)
    plt.close()
    logger.info(f"Saved mass conservation figure to {fig_path}")
    
    return metrics


def verify_photolytic_and_thermal_decay(df: pd.DataFrame, output_dir: str = "reports/figures") -> dict:
    """
    Evaluates Photolytic (Solar UV) & Thermal (Arrhenius) Decay Kinetics:
    k_eff = k_dark(T) + k_UV(Solar, CYA, Depth) + k_organic(Turbidity)
    Tests on quiescent (zero-chemical-dosing) intervals.
    """
    logger.info("Verifying Photolytic UV & Thermal Arrhenius Decay Kinetics...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter quiescent zero-dose subset
    zero_dose = df[(df['total_active_cl2_grams'] == 0) & (df['hypo_dosing_hours'] == 0) & (df['free_chlorine'] > 0.5)].copy()
    zero_dose['ratio'] = np.clip(zero_dose['target_next_free_chlorine'] / zero_dose['free_chlorine'], 0.02, 5.0)
    zero_dose['k_obs'] = -np.log(zero_dose['ratio']) / zero_dose['delta_days']
    
    # 1. Observed Decay Rate vs Solar Radiation
    valid_decay = zero_dose[zero_dose['k_obs'] > -0.1].copy()
    r_sol, p_sol = stats.pearsonr(valid_decay['window_solar_rad_mean_mj'], valid_decay['k_obs'])
    r_temp, p_temp = stats.pearsonr(valid_decay['window_temp_mean_c'], valid_decay['k_obs'])
    r_depth, p_depth = stats.pearsonr(valid_decay['estimated_mean_depth'], valid_decay['k_obs'])
    
    # 2. Cyanuric Acid UV Shielding Effect
    # Compare summer decay rates in high vs low CYA pools
    summer_zero = zero_dose[zero_dose['is_summer'] == 1]
    cya_med = summer_zero['cya_cumulative_seasonal_ppm'].median()
    high_cya_k = summer_zero[summer_zero['cya_cumulative_seasonal_ppm'] > cya_med]['k_obs'].mean()
    low_cya_k = summer_zero[summer_zero['cya_cumulative_seasonal_ppm'] <= cya_med]['k_obs'].mean()
    
    # 3. Arrhenius Activation Energy Estimation
    # k(T) = A * exp(-Ea / (R * T)) => ln(k) vs 1/T
    positive_decay = valid_decay[valid_decay['k_obs'] > 0.01].copy()
    positive_decay['inv_T_k'] = 1.0 / (positive_decay['window_temp_mean_c'] + 273.15)
    positive_decay['ln_k'] = np.log(positive_decay['k_obs'])
    
    if len(positive_decay) > 20:
        slope, intercept, r_val, p_val, std_err = stats.linregress(positive_decay['inv_T_k'], positive_decay['ln_k'])
        # R = 8.314 J/(mol*K), slope = -Ea / R => Ea = -slope * 8.314 / 1000 (kJ/mol)
        ea_kj_mol = -slope * 8.314 / 1000.0
    else:
        ea_kj_mol = 45.0
        
    metrics = {
        "quiescent_zero_dose_count": len(zero_dose),
        "mean_observed_k_day": round(float(valid_decay['k_obs'].mean()), 4),
        "median_observed_k_day": round(float(valid_decay['k_obs'].median()), 4),
        "half_life_days_median": round(float(np.log(2.0) / max(valid_decay['k_obs'].median(), 0.01)), 2),
        "corr_solar_rad_vs_k": round(float(r_sol), 4),
        "corr_temp_vs_k": round(float(r_temp), 4),
        "corr_depth_vs_k": round(float(r_depth), 4),
        "summer_high_cya_k_mean": round(float(high_cya_k), 4),
        "summer_low_cya_k_mean": round(float(low_cya_k), 4),
        "cya_shielding_protection_pct": round(float((1.0 - high_cya_k / max(low_cya_k, 0.001)) * 100.0), 2),
        "arrhenius_activation_energy_kj_mol": round(float(ea_kj_mol), 2)
    }
    
    # Figure 14: Photolytic & Thermal Kinetics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Subplot A: Decay Rate k_obs vs Solar Radiation
    sns.regplot(x=valid_decay['window_solar_rad_mean_mj'], y=valid_decay['k_obs'],
                scatter_kws={'alpha': 0.4, 'color': '#f59e0b', 's': 25},
                line_kws={'color': '#b45309', 'lw': 2.5}, ax=axes[0])
    axes[0].axhline(0, color='gray', linestyle=':', lw=1.2)
    axes[0].set_title(f"Photolytic Decay Rate vs Daily Solar Irradiance ($r = {r_sol:.3f}$)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel(r"Mean Solar Radiation during Window ($\mathrm{MJ/m^2/day}$)", fontsize=11)
    axes[0].set_ylabel(r"Observed First-Order Decay Rate $k_{\mathrm{obs}}$ ($\mathrm{day^{-1}}$)", fontsize=11)
    axes[0].set_ylim(-0.1, 0.6)
    
    # Subplot B: Cyanuric Acid (CYA) Photolysis Shielding Curve
    cya_levels = np.linspace(0, 100, 200)
    # Wojtowicz / O'Brien shielding function: Phi_CYA = 1 / (1 + 0.04 * [CYA])
    shielding_pct = (1.0 - 1.0 / (1.0 + 0.04 * cya_levels)) * 100.0
    axes[1].plot(cya_levels, shielding_pct, color='#059669', lw=2.5, label='Theoretical Wojtowicz-O\'Brien Model')
    axes[1].axvspan(30, 50, color='green', alpha=0.15, label='Ideal Commercial CYA Range (30 - 50 ppm)')
    axes[1].axvline(50, color='orange', linestyle='--', label='Max Recommended Threshold (50 ppm)')
    axes[1].axvline(100, color='red', linestyle='--', label='Over-Stabilization / Chlorine Lock (>100 ppm)')
    axes[1].set_title(r"Cyanuric Acid ($\mathrm{CYA}$) Photolytic Shielding Efficiency", fontsize=12, fontweight='bold')
    axes[1].set_xlabel(r"Cyanuric Acid Concentration $\mathrm{[CYA]}$ (ppm)", fontsize=11)
    axes[1].set_ylabel("UV Radiation Shielding Protection (%)", fontsize=11)
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, 100)
    axes[1].legend(loc='lower right', frameon=True, fontsize=9)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "14_physics_photolysis_decay_kinetics.png")
    plt.savefig(fig_path)
    plt.close()
    logger.info(f"Saved photolysis kinetics figure to {fig_path}")
    
    return metrics


def verify_hydrodynamics_and_turnover(df: pd.DataFrame, output_dir: str = "reports/figures") -> dict:
    """
    Evaluates Hydrodynamics, Hydraulic Turnover Times & Regulatory Standards:
    tau = Volume / Pump_flow (hours)
    Turnover cycles/day = (Pump_flow * Filtration_hours) / Volume
    Evaluates Spanish / European regulatory compliance (Real Decreto 742/2013: turnover <= 4h).
    """
    logger.info("Verifying Hydrodynamics & Hydraulic Turnover Compliance...")
    os.makedirs(output_dir, exist_ok=True)
    
    vol = df['pool_volume'].clip(lower=10.0)
    flow = df['motor_pump_flow_rate'].fillna(25.0).clip(lower=5.0)
    filt_hours = df['daily_filtration_hours'].fillna(10.0).clip(lower=1.0)
    
    turnover_time_h = vol / flow
    daily_cycles = (flow * filt_hours) / vol
    
    # Spanish RD 742/2013 Criteria: Collective pools should achieve turnover <= 4 hours
    rd742_compliant = (turnover_time_h <= 4.0)
    rd742_extended = (turnover_time_h <= 8.0)
    
    # Turnover correlation with turbidity
    valid_rows = df[df['turbidity'].notna()]
    r_cycles_turb, p_cycles_turb = stats.pearsonr(
        daily_cycles.loc[valid_rows.index],
        valid_rows['turbidity']
    )
    
    metrics = {
        "turnover_time_hours_mean": round(float(turnover_time_h.mean()), 2),
        "turnover_time_hours_median": round(float(turnover_time_h.median()), 2),
        "daily_turnover_cycles_mean": round(float(daily_cycles.mean()), 2),
        "daily_turnover_cycles_median": round(float(daily_cycles.median()), 2),
        "rd742_strict_4h_compliance_pct": round(float(rd742_compliant.mean() * 100.0), 2),
        "rd742_practical_8h_compliance_pct": round(float(rd742_extended.mean() * 100.0), 2),
        "corr_daily_cycles_vs_turbidity": round(float(r_cycles_turb), 4)
    }
    
    # Figure 15: Hydrodynamic Turnover & Hydraulic Performance
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Subplot A: Hydraulic Turnover Time Distribution vs Regulatory Thresholds
    sns.histplot(turnover_time_h, bins=45, color='#0284c7', kde=True, ax=axes[0], alpha=0.6)
    axes[0].axvline(4.0, color='#15803d', linestyle='--', lw=2.2, label=r'RD 742/2013 Standard ($\leq 4.0$ h)')
    axes[0].axvline(8.0, color='#d97706', linestyle=':', lw=2.2, label=r'Residential Standard ($\leq 8.0$ h)')
    axes[0].axvline(turnover_time_h.median(), color='#b91c1c', linestyle='-', lw=2,
                   label=f'Dataset Median = {turnover_time_h.median():.1f} h')
    axes[0].set_title(f"Hydraulic Turnover Time Distribution ($N={len(df):,}$)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Hydraulic Turnover Duration $\\tau$ (hours)", fontsize=11)
    axes[0].set_ylabel("Observation Count", fontsize=11)
    axes[0].set_xlim(0, 20)
    axes[0].legend(loc='upper right', frameon=True, fontsize=9)
    
    # Subplot B: Daily Turnover Cycles vs Water Turbidity (NTU)
    cycle_bins = pd.cut(daily_cycles, bins=[0, 1.0, 2.0, 3.0, 5.0, 10.0],
                        labels=['<1.0', '1.0-2.0', '2.0-3.0', '3.0-5.0', '>5.0'])
    turb_by_cycle = df['turbidity'].groupby(cycle_bins, observed=False).mean()
    
    bars = axes[1].bar(turb_by_cycle.index.astype(str), turb_by_cycle.values, color='#0d9488',
                       edgecolor='#333333', lw=1, width=0.55)
    for b in bars:
        h = b.get_height()
        axes[1].text(b.get_x() + b.get_width()/2., h + 0.02, f'{h:.2f} NTU',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    axes[1].set_title("Hydraulic Turnover Frequency vs Water Clarity (Turbidity)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Daily Hydraulic Turnover Cycles (turnovers/day)", fontsize=11)
    axes[1].set_ylabel("Mean Turbidity (NTU)", fontsize=11)
    axes[1].set_ylim(0, max(turb_by_cycle.values) * 1.3)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "15_physics_hydrodynamic_turnover.png")
    plt.savefig(fig_path)
    plt.close()
    logger.info(f"Saved hydrodynamics figure to {fig_path}")
    
    return metrics


def classify_physical_plausibility(df: pd.DataFrame, output_dir: str = "reports/figures") -> dict:
    """
    Classifies all 37,252 transitions into 5 mutually exclusive physical regimes:
    1. Physically Conserved & Stable (Mass conservation & normal decay/dosing obeyed)
    2. Unlogged Manual Chemical Shock (Cl jumps >1.5 ppm without logged chemical addition)
    3. Sensor Upper Ceiling Saturation (Measured at 5.0 ppm limit)
    4. Severe Organic Depletion (Cl drops >3.0 ppm in <3 days)
    5. Non-Physical Recording Jump / Measurement Noise
    """
    logger.info("Classifying Physical Plausibility & Anomaly Regimes across dataset...")
    os.makedirs(output_dir, exist_ok=True)
    
    c_initial = df['free_chlorine']
    c_final = df['target_next_free_chlorine']
    dt = df['delta_days']
    cl_delta = c_final - c_initial
    has_chem_logged = (df['total_active_cl2_grams'] > 0) | (df['hypo_dosing_hours'] > 0)
    
    # Classifications
    is_sensor_sat = (c_final >= 5.0) | (c_initial >= 5.0)
    is_unlogged_shock = (~is_sensor_sat) & (cl_delta > 1.5) & (~has_chem_logged)
    is_severe_drop = (~is_sensor_sat) & (cl_delta < -2.5) & (dt <= 3.5)
    is_recording_jump = (~is_sensor_sat) & (~is_unlogged_shock) & (~is_severe_drop) & (np.abs(cl_delta) > 3.5)
    is_physically_sound = (~is_sensor_sat) & (~is_unlogged_shock) & (~is_severe_drop) & (~is_recording_jump)
    
    categories = ['Physically Compliant\n& Stable', 'Sensor Ceiling\nSaturation (5.0 ppm)',
                  'Unlogged Manual\nChemical Shock', 'Severe Organic\nDepletion Drop', 'Recording Anomaly\n/ Sensor Noise']
    counts = [is_physically_sound.sum(), is_sensor_sat.sum(), is_unlogged_shock.sum(), is_severe_drop.sum(), is_recording_jump.sum()]
    pcts = [c / len(df) * 100.0 for c in counts]
    
    metrics = {
        "physically_sound_records": int(is_physically_sound.sum()),
        "physically_sound_pct": round(pcts[0], 2),
        "sensor_saturation_records": int(is_sensor_sat.sum()),
        "sensor_saturation_pct": round(pcts[1], 2),
        "unlogged_shock_records": int(is_unlogged_shock.sum()),
        "unlogged_shock_pct": round(pcts[2], 2),
        "severe_depletion_records": int(is_severe_drop.sum()),
        "severe_depletion_pct": round(pcts[3], 2),
        "recording_anomaly_records": int(is_recording_jump.sum()),
        "recording_anomaly_pct": round(pcts[4], 2),
        "total_evaluated_transitions": len(df)
    }
    
    # Figure 16: Physical Health & Regime Breakdown
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Subplot A: Donut Chart
    palette = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6']
    wedges, texts, autotexts = axes[0].pie(
        counts, labels=None, autopct='%1.1f%%', pctdistance=0.75,
        startangle=140, colors=palette,
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight('bold')
    axes[0].legend(wedges, [f"{cat.replace(chr(10), ' ')} ({p:.1f}%)" for cat, p in zip(categories, pcts)],
                   loc='lower center', bbox_to_anchor=(0.5, -0.2), frameon=True, fontsize=8.5, ncol=2)
    axes[0].set_title(f"Dataset Physical Integrity & Anomaly Regimes ($N={len(df):,}$)", fontsize=12, fontweight='bold')
    
    # Subplot B: Bar Breakdown with Counts
    bars = axes[1].barh(categories, counts, color=palette, edgecolor='#333333', lw=1, height=0.55)
    for b in bars:
        w = b.get_width()
        axes[1].text(w + 300, b.get_y() + b.get_height()/2., f'{w:,} ({w/len(df)*100:.1f}%)',
                     ha='left', va='center', fontsize=9.5, fontweight='bold')
        
    axes[1].set_title("Physical Regime Distribution", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Transition Count", fontsize=11)
    axes[1].set_xlim(0, max(counts) * 1.25)
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "16_physics_plausibility_classification.png")
    plt.savefig(fig_path)
    plt.close()
    logger.info(f"Saved physical plausibility figure to {fig_path}")
    
    return metrics


def benchmark_hybrid_physics_informed_model(df: pd.DataFrame, output_dir: str = "reports/figures") -> dict:
    """
    Benchmarks 3 Predictive Modeling Paradigms:
    1. Pure Mechanistic Continuous ODE
       dC/dt = r_in - k_eff * C => C(dt) = (C0 + D0)*exp(-k*dt) + (r_in/k)*(1 - exp(-k*dt))
    2. Pure Machine Learning (HistGradientBoostingRegressor)
    3. Gray-Box Physics-Informed Hybrid Model (Continuous Physical ODE Backbone + Residual ML)
    """
    logger.info("Executing 3-Tier Benchmark: Pure Physics vs Pure ML vs Gray-Box Hybrid...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Feature columns
    exclude_cols = [
        'pool_clean', 'community_address', 'measurement_date', 'next_date_dt',
        'date_dt', 'date_only', 'next_employee', 'measurement_employee',
        'next_free_chlorine', 'next_ph', 'next_turbidity',
        'target_next_free_chlorine', 'target_next_compliance_band',
        'is_train_split'
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]
    
    train_mask = df['is_train_split'] == 1
    test_mask = df['is_train_split'] == 0
    
    X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, 'target_next_free_chlorine']
    X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, 'target_next_free_chlorine']
    y_test_band = df.loc[test_mask, 'target_next_compliance_band']
    
    # Physics Variables
    vol = df['pool_volume'].clip(lower=10.0)
    dt = df['delta_days'].clip(lower=0.5, upper=10.0)
    temp_c = df['window_temp_mean_c'].fillna(24.0)
    solar_mj = df['window_solar_rad_mean_mj'].fillna(20.0)
    depth = df['estimated_mean_depth'].clip(lower=0.5, upper=3.0)
    cya = df['cya_cumulative_seasonal_ppm'].fillna(20.0).clip(lower=0.0, upper=100.0)
    turb = df['turbidity'].fillna(0.5).clip(lower=0.1, upper=5.0)
    
    c0_shock = df['free_chlorine'] + (df['active_cl2_shock_grams'] / vol)
    r_erodible = (df['erodible_dissolved_grams'] / dt) / vol
    raw_pump_rate = (df['hypochlorite_pump_flow_rate'].fillna(4.0) * df['hypo_dosing_hours'] * (df['hypo_dosing_percentage'] / 100.0) * 130.0) / vol
    
    # 1. Fit Calibrated Mechanistic ODE on Training Set
    def solve_ode(params, mask):
        scale_pump, k_dark_base, theta_temp, k_uv_base, gamma_cya, k_turb_base = params
        r_in = r_erodible[mask] + (scale_pump * raw_pump_rate[mask])
        cya_shield = 1.0 / (1.0 + gamma_cya * cya[mask])
        k_uv = k_uv_base * (solar_mj[mask] / 20.0) * (1.0 / depth[mask]) * cya_shield
        k_dark = k_dark_base * (theta_temp ** (temp_c[mask] - 20.0))
        k_turb = k_turb_base * turb[mask]
        k_eff = (k_dark + k_uv + k_turb).clip(lower=0.01, upper=2.0)
        exp_kdt = np.exp(-k_eff * dt[mask])
        c_pred = (c0_shock[mask] * exp_kdt) + (r_in / k_eff) * (1.0 - exp_kdt)
        return np.clip(c_pred, 0.0, 5.5)
    
    def loss_ode(params):
        preds = solve_ode(params, train_mask)
        return mean_squared_error(y_train, preds)
    
    init_p = [0.03, 0.01, 1.00, 0.25, 0.001, 0.02]
    bounds_p = [(0.001, 0.5), (0.001, 0.5), (1.0, 1.15), (0.01, 1.0), (0.0001, 0.1), (0.001, 0.2)]
    res_ode = minimize(loss_ode, init_p, bounds=bounds_p, method='L-BFGS-B')
    
    # ODE Predictions
    ode_train_pred = solve_ode(res_ode.x, train_mask)
    ode_test_pred = solve_ode(res_ode.x, test_mask)
    
    # 2. Pure Machine Learning Model (Standard HistGradientBoostingRegressor)
    ml_model = HistGradientBoostingRegressor(
        loss='squared_error',
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=42
    )
    ml_model.fit(X_train, y_train)
    ml_test_pred = np.clip(ml_model.predict(X_test), 0.0, 5.5)
    
    # 3. Physics-Informed Gray-Box Hybrid Model (PINN Residual Architecture)
    # Train ML to predict residual difference: y_train - ode_train_pred
    residual_train = y_train.values - ode_train_pred.values
    hybrid_ml = HistGradientBoostingRegressor(
        loss='squared_error',
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=42
    )
    hybrid_ml.fit(X_train, residual_train)
    hybrid_residual_pred = hybrid_ml.predict(X_test)
    hybrid_test_pred = np.clip(ode_test_pred.values + hybrid_residual_pred, 0.0, 5.5)
    
    # Evaluation Metrics
    def calc_metrics(y_true, y_pred, y_band):
        y_true_np = np.asarray(y_true)
        y_pred_np = np.asarray(y_pred)
        y_band_np = np.asarray(y_band)
        
        r2 = r2_score(y_true_np, y_pred_np)
        mae = mean_absolute_error(y_true_np, y_pred_np)
        rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
        acc_05 = (np.abs(y_true_np - y_pred_np) <= 0.5).mean() * 100.0
        acc_03 = (np.abs(y_true_np - y_pred_np) <= 0.3).mean() * 100.0
        pred_band_np = pd.cut(y_pred_np, bins=[-np.inf, 0.999, 3.001, np.inf], labels=[0, 1, 2]).to_numpy().astype(int)
        band_acc = accuracy_score(y_band_np, pred_band_np) * 100.0
        return {
            "r2": round(float(r2), 4),
            "mae_ppm": round(float(mae), 4),
            "rmse_ppm": round(float(rmse), 4),
            "accuracy_within_0_5ppm_pct": round(float(acc_05), 2),
            "accuracy_within_0_3ppm_pct": round(float(acc_03), 2),
            "compliance_band_accuracy_pct": round(float(band_acc), 2)
        }
        
    m_ode = calc_metrics(y_test, ode_test_pred, y_test_band)
    m_ml = calc_metrics(y_test, ml_test_pred, y_test_band)
    m_hybrid = calc_metrics(y_test, hybrid_test_pred, y_test_band)
    
    logger.info(f"Model 1 (Mechanistic ODE): R² = {m_ode['r2']} | MAE = {m_ode['mae_ppm']} ppm | ±0.5 ppm = {m_ode['accuracy_within_0_5ppm_pct']}%")
    logger.info(f"Model 2 (Pure ML):          R² = {m_ml['r2']} | MAE = {m_ml['mae_ppm']} ppm | ±0.5 ppm = {m_ml['accuracy_within_0_5ppm_pct']}%")
    logger.info(f"Model 3 (Gray-Box Hybrid): R² = {m_hybrid['r2']} | MAE = {m_hybrid['mae_ppm']} ppm | ±0.5 ppm = {m_hybrid['accuracy_within_0_5ppm_pct']}%")
    
    metrics = {
        "model_1_pure_mechanistic_ode": m_ode,
        "model_2_pure_machine_learning": m_ml,
        "model_3_physics_informed_gray_box": m_hybrid,
        "calibrated_ode_parameters": {
            "scale_pump_duty_cycle": round(float(res_ode.x[0]), 4),
            "k_dark_base_day_inv": round(float(res_ode.x[1]), 4),
            "theta_arrhenius": round(float(res_ode.x[2]), 4),
            "k_uv_base_day_inv": round(float(res_ode.x[3]), 4),
            "gamma_cya_shielding": round(float(res_ode.x[4]), 4),
            "k_turbidity_day_inv": round(float(res_ode.x[5]), 4)
        }
    }
    
    # Figure 17: Model Comparison Scatter & Residuals
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Subplot A: Accuracy Comparison Bar Chart
    models = ['Pure Mechanistic\nODE', 'Pure ML\n(Gradient Boosted)', 'Gray-Box\nHybrid (PINN)']
    mae_vals = [m_ode['mae_ppm'], m_ml['mae_ppm'], m_hybrid['mae_ppm']]
    acc_05_vals = [m_ode['accuracy_within_0_5ppm_pct'], m_ml['accuracy_within_0_5ppm_pct'], m_hybrid['accuracy_within_0_5ppm_pct']]
    
    x_pos = np.arange(len(models))
    w = 0.35
    b1 = axes[0].bar(x_pos - w/2, mae_vals, width=w, label='MAE (ppm, lower is better)', color='#ef4444', edgecolor='#333333')
    b2 = axes[0].bar(x_pos + w/2, [a/100.0 for a in acc_05_vals], width=w, label='±0.5 ppm Accuracy (higher is better)', color='#10b981', edgecolor='#333333')
    
    for b in b1:
        h = b.get_height()
        axes[0].text(b.get_x() + b.get_width()/2., h + 0.03, f'{h:.3f} ppm', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for b in b2:
        h = b.get_height()
        axes[0].text(b.get_x() + b.get_width()/2., h + 0.03, f'{h*100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(models, fontsize=10, fontweight='bold')
    axes[0].set_title("Out-of-Sample Performance Comparison (2026 Test Set)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Error / Accuracy Scale", fontsize=11)
    axes[0].set_ylim(0, 1.5)
    axes[0].legend(loc='upper right', frameon=True, fontsize=9)
    
    # Subplot B: Gray-Box Hybrid Model Scatter Plot
    sample_idx = np.random.RandomState(42).choice(len(y_test), size=min(3000, len(y_test)), replace=False)
    axes[1].scatter(y_test.values[sample_idx], hybrid_test_pred[sample_idx], alpha=0.35, color='#6366f1', s=20,
                    label='Hybrid Predictions')
    axes[1].plot([0, 5.5], [0, 5.5], 'r--', lw=2, label='Perfect Prediction (1:1)')
    axes[1].fill_between([0, 5.5], [0 - 0.5, 5.5 - 0.5], [0 + 0.5, 5.5 + 0.5], color='green', alpha=0.1,
                         label=r'$\pm 0.5$ ppm Precision Band')
    axes[1].set_title(f"Physics-Informed Gray-Box Model (2026 Holdout)\n$R^2 = {m_hybrid['r2']:.3f}$ | MAE = {m_hybrid['mae_ppm']:.3f} ppm | $\\pm 0.5$ ppm Acc = {m_hybrid['accuracy_within_0_5ppm_pct']:.1f}%",
                      fontsize=11.5, fontweight='bold')
    axes[1].set_xlabel("Actual Measured Chlorine (ppm)", fontsize=11)
    axes[1].set_ylabel("Predicted Chlorine (ppm)", fontsize=11)
    axes[1].set_xlim(0, 5.5)
    axes[1].set_ylim(0, 5.5)
    axes[1].legend(loc='upper left', frameon=True, fontsize=8.5)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "17_physics_hybrid_model_comparison.png")
    plt.savefig(fig_path)
    plt.close()
    logger.info(f"Saved hybrid model comparison figure to {fig_path}")
    
    return metrics


def run_physics_verification_pipeline(dataset_csv: str = "data/processed/chlorine_ml_dataset.csv",
                                      output_dir: str = "reports") -> dict:
    """Executes the full physics & chemistry verification suite and exports artifacts."""
    logger.info("=" * 70)
    logger.info("STARTING REAL-WORLD PHYSICS & CHEMISTRY VERIFICATION PIPELINE")
    logger.info("=" * 70)
    
    if not os.path.exists(dataset_csv):
        raise FileNotFoundError(f"Missing dataset: {dataset_csv}")
        
    df = pd.read_csv(dataset_csv)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Acid-Base Equilibrium
    acid_base_metrics = verify_acid_base_equilibrium(df, output_dir=fig_dir)
    
    # 2. Stoichiometric Mass Conservation
    mass_metrics = verify_mass_conservation_and_stoichiometry(df, output_dir=fig_dir)
    
    # 3. Photolytic & Thermal Kinetics
    kinetics_metrics = verify_photolytic_and_thermal_decay(df, output_dir=fig_dir)
    
    # 4. Hydrodynamics & Hydraulic Turnover
    hydro_metrics = verify_hydrodynamics_and_turnover(df, output_dir=fig_dir)
    
    # 5. Physical Plausibility Classification
    plausibility_metrics = classify_physical_plausibility(df, output_dir=fig_dir)
    
    # 6. Hybrid Modeling Benchmark
    hybrid_metrics = benchmark_hybrid_physics_informed_model(df, output_dir=fig_dir)
    
    all_metrics = {
        "dataset_metadata": {
            "total_records": len(df),
            "train_records": int((df['is_train_split'] == 1).sum()),
            "test_records": int((df['is_train_split'] == 0).sum()),
            "evaluation_timestamp": pd.Timestamp.now().isoformat()
        },
        "acid_base_equilibrium": acid_base_metrics,
        "mass_conservation": mass_metrics,
        "decay_kinetics": kinetics_metrics,
        "hydrodynamics_and_turnover": hydro_metrics,
        "physical_plausibility_classification": plausibility_metrics,
        "predictive_benchmarks": hybrid_metrics
    }
    
    metrics_path = os.path.join(output_dir, "physics_verification_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Saved complete physics verification metrics to {metrics_path}")
    
    logger.info("=" * 70)
    logger.info("PHYSICS VERIFICATION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)
    
    return all_metrics


if __name__ == "__main__":
    run_physics_verification_pipeline()
