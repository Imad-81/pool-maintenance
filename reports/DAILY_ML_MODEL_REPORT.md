# Daily Free Chlorine Machine Learning Model Performance Report

**Dataset:** Continuous Daily Pool Water Quality Dataset (`pool_daily_ml_ready.csv`)  
**Total Samples:** 156,273 daily records across 138 pools  
**Train Period:** 2023–2025 (129,860 pool-days)  
**Out-of-Sample Holdout Test:** 2026 (26,413 pool-days)  
**Primary Target:** `target_next_day_free_chlorine` (Tomorrow's Free Chlorine in mg/L)

---

## 1. Executive Performance Summary

The **Ensemble Blend (Delta Formulation)** achieved **ultra-high precision** on the 2026 out-of-sample holdout test set:

| Model & Formulation | Test MAE (ppm) | RMSE (ppm) | $R^2$ Score | $\pm 0.10$ ppm Acc | $\pm 0.25$ ppm Acc | $\pm 0.50$ ppm Acc | Compliance Band Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Direct)** | `0.0744` | `0.1484` | `0.9678` | **80.1%** | **94.5%** | **98.4%** | **95.1%** |
| **CatBoost (Direct)** | `0.0779` | `0.1532` | `0.9657` | **78.7%** | **94.1%** | **98.4%** | **95.1%** |
| **XGBoost (Direct)** | `0.0766` | `0.1511` | `0.9667` | **79.2%** | **94.3%** | **98.5%** | **95.1%** |
| **Ensemble Blend (Direct)** | `0.0738` | `0.1485` | `0.9678` | **80.1%** | **94.5%** | **98.5%** | **95.3%** |
| **LightGBM (Delta ΔC)** | `0.0705` | `0.1468` | `0.9685` | **81.4%** | **94.5%** | **98.5%** | **95.5%** |
| **CatBoost (Delta ΔC)** | `0.0745` | `0.1516` | `0.9664` | **79.9%** | **94.2%** | **98.4%** | **95.4%** |
| **XGBoost (Delta ΔC)** | `0.0716` | `0.1479` | `0.9681` | **80.9%** | **94.5%** | **98.5%** | **95.5%** |
| **Ensemble Blend (Delta ΔC)** | `0.0707` | `0.1468` | `0.9685` | **81.2%** | **94.6%** | **98.5%** | **95.6%** |

---

## 2. Key Accuracy Takeaways

1. **Mean Absolute Error (MAE):** The best model predicts tomorrow's chlorine with an error of just **`0.0705` ppm (mg/L)**.
2. **$\pm 0.25$ ppm Clinical Precision:** **94.53%** of all predictions are within a razor-thin **$\pm 0.25$ mg/L** of actual laboratory/sensor tests.
3. **$\pm 0.50$ ppm Operational Accuracy:** **98.49%** of predictions are within $\pm 0.50$ mg/L.
4. **Regulatory Band Classification:** **95.5%** accuracy in predicting whether tomorrow's pool will be Under-Target ($<1.0$ ppm), Compliant ($1.0–3.0$ ppm), or Over-Target ($>3.0$ ppm).
5. **Coefficient of Determination ($R^2$):** **`0.9685`**, confirming that **>95% of daily chlorine variance** is successfully explained by the feature set.

---

## 3. What Drives Tomorrow's Chlorine? (Top Model Features)

Based on gradient-boosted feature importance rankings, the top drivers of tomorrow's chlorine are:

1. **Today's Water State ($C_t$, pH, Turbidity):** `free_chlorine_estimated_daily_mean_ppm`, `active_hocl_ppm`, `ph`.
2. **Chemical Influx:** `shock_dosage_ppm`, `daily_pump_cl2_delivered_ppm`, `erodible_active_cl2_added_grams`.
3. **Physics Kinetics & Theoretical Decay:** `theoretical_retained_chlorine`, `theoretical_decay_k`, `active_hocl_fraction`.
4. **Alicante Weather Forcing:** `solar_radiation_mj`, `window_solar_rad_sum_mj` (3-day solar irradiance), `temperature_ambient_mean_c`.
5. **Pool Geometry & Baseline Prior:** `specific_surface_ratio`, `pool_volume`, `pool_cl_hist_mean`.

---

## 4. Diagnostic Visualizations

Below are the 2026 test evaluation diagnostics, error distributions, feature rankings, and sample time-series tracking:

![Model Diagnostic Evaluation](file:///Users/imadmac/projects/pool_project/reports/figures/19_daily_model_evaluation.png)
