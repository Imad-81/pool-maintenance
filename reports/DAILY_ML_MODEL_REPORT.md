# Multi-Chemical Daily Pool Machine Learning Suite Performance Report

**Dataset:** Multi-Chemical Daily Dataset (`pool_daily_ml_ready.csv`)  
**Total Samples:** 156,273 daily records across 138 pools (136 enriched columns)  
**Train Period:** 2023–2025 (129,860 pool-days)  
**Out-of-Sample Holdout Test:** 2026 (26,413 pool-days)  
**Trained Models:** Free Chlorine ($\Delta C$), pH ($\Delta 	ext{pH}$), Turbidity ($\Delta 	ext{Turb}$)

---

## 1. Executive Performance Summary: Chlorine Models

| Model & Formulation | Test MAE (ppm) | RMSE (ppm) | $R^2$ Score | $\pm 0.10$ ppm Acc | $\pm 0.25$ ppm Acc | $\pm 0.50$ ppm Acc | Compliance Band Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Chlorine ΔC)** | `0.0840` | `0.1630` | `0.9612` | **76.4%** | **93.0%** | **98.0%** | **94.9%** |
| **CatBoost (Chlorine ΔC)** | `0.0925` | `0.1716` | `0.9570` | **73.6%** | **92.0%** | **97.9%** | **94.8%** |
| **XGBoost (Chlorine ΔC)** | `0.0889` | `0.1677` | `0.9589` | **74.6%** | **92.3%** | **98.0%** | **94.9%** |
| **Ensemble Blend (Chlorine ΔC)** | `0.0865` | `0.1648` | `0.9603` | **75.7%** | **92.7%** | **98.0%** | **95.0%** |

---

## 2. Stratified Operational Benchmark (Audited Reality Check)

To ensure clinical integrity and prevent conflating synthetic kinetic smoothing with true laboratory tests, performance is evaluated across 4 distinct operational strata:

| Stratum | Evaluation Scope | Sample Size | MAE (ppm) | $R^2$ Score | Operational Meaning |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Stratum 1** | **Full 2026 Fleet Test Set** | 26,413 | **`0.0840`** | **`0.9612`** | Fleet-wide continuous tracking continuity |
| **Stratum 2** | **Real Lab Observations** | 6,165 | **`0.0864`** | **`0.9553`** | Model accuracy when predicting actual lab tests |
| **Stratum 3** | **Consecutive Real Days (Back-to-Back)** | 209 | **`0.5483`** | **`0.0910`** | True un-smoothed daily jump error ($\sigma = 1.02$) |
| **Stratum 4** | **14-Day Autoregressive Rollout** | 100 visits | **`0.2155` → `0.6629`** | N/A | True multi-day forward maintenance forecast error |

### Autoregressive Rollout Error Growth (Horizon $H = 1$ to $14$ Days):
| Horizon Ahead | Forecast Rollout MAE |
| :--- | :---: |
| **Day 1 Ahead** | `0.2155 ppm` |
| **Day 2 Ahead** | `0.2543 ppm` |
| **Day 3 Ahead** | `0.3226 ppm` |
| **Day 4 Ahead** | `0.3449 ppm` |
| **Day 5 Ahead** | `0.3961 ppm` |
| **Day 6 Ahead** | `0.4114 ppm` |
| **Day 7 Ahead** | `0.4859 ppm` |
| **Day 8 Ahead** | `0.4956 ppm` |
| **Day 9 Ahead** | `0.5457 ppm` |
| **Day 10 Ahead** | `0.5897 ppm` |
| **Day 11 Ahead** | `0.5994 ppm` |
| **Day 12 Ahead** | `0.6115 ppm` |
| **Day 13 Ahead** | `0.6164 ppm` |
| **Day 14 Ahead** | `0.6629 ppm` |

---

## 3. Secondary Chemical Suite: pH & Turbidity Models

To allow continuous multi-day autoregressive rollouts, separate gradient-boosted models were trained for pH and Turbidity:

| Parameter | Best Model | Test MAE | RMSE | $R^2$ Score | Key Target Precision |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **pH** | LightGBM ($\Delta 	ext{pH}$) | **`0.0115` pH** | `0.0247` | `0.9643` | **98.8%** within $\pm 0.10$ pH |
| **Turbidity** | LightGBM ($\Delta 	ext{Turb}$) | **`0.0148` NTU** | `0.0431` | `0.9153` | **99.8%** within $\pm 0.25$ NTU |

---

## 4. Key Drivers of Multi-Day Water Chemistry

The gradient-boosted models identified the following physical and operational features as the top drivers of chemistry evolution:
1. **Forecast Weather ($t+1$):** `forecast_solar_radiation_mj` (UV photolysis driving force), `forecast_temperature_max_c`.
2. **Thermal Inertia:** `water_temperature_c` (kinetic rate multiplier for Arrhenius decay).
3. **Chemical Additions & Automated Dosing:** `shock_dosage_ppm`, `daily_pump_cl2_delivered_ppm`, `clarifier_added_grams`, `acid_added_grams`, `daily_pump_ph_minus_ml`.
4. **Active Disinfectant Power:** `active_hocl_fraction` ($1 / (1 + 10^{	ext{pH} - 7.53})$), `hocl_demand_proxy`.
5. **Pool Geometry & Baseline Prior:** `specific_surface_ratio`, `pool_volume`, `pool_cl_hist_mean`.

---

## 5. Diagnostic Visualizations

![Model Diagnostic Evaluation](file:///Users/imadmac/projects/pool_project/reports/figures/19_daily_model_evaluation.png)
