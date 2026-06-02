# Pipeline V2 — Visit Timing + Chemical Dosage Prediction

## Goal Change

**V1**: Predicted *what* the water quality will be at the next visit.
**V2**: Predicts **when** the next visit should happen + what chemicals to bring. Grounded in Spanish pool regulations.

---

## Regulatory Framework

| Regulation | Scope |
|---|---|
| **Real Decreto 742/2013** | National — defines water quality parametric values for collective-use pools |
| **Decreto 85/2018** | Comunitat Valenciana — regional adaptation, requires daily autocontrol |

### Parametric Thresholds (RD 742/2013 Annex I)

| Parameter | Compliant Range | Pool Closure |
|---|---|---|
| Free chlorine | **0.5 – 2.0 mg/L** (ideal) | < 0.5 or > 5.0 |
| pH | **7.2 – 8.0** | — |
| Turbidity | **≤ 5 NTU** | — |

> [!IMPORTANT]
> **Critical finding**: 60% of readings show chlorine > 2.0 mg/L. This is standard Spanish practice (intentional overdosing), NOT a safety breach. Only chlorine < 0.5 (pathogen risk) or > 5.0 (chemical burn risk, mandatory closure) constitutes a safety breach. The pipeline uses corrected definitions.

---

## Key Design Decisions

### 1. Seasonal Baseline Deviation Model

Technicians follow a strongly seasonal schedule discovered from the data:

| Season | Median Visit Interval |
|---|---|
| Summer (Jun–Sep) | **2 days** |
| Spring/Fall (Mar–May, Oct) | **4–6 days** |
| Winter (Nov–Feb) | **6–7 days** |

Rather than predicting raw `days_to_next_visit` (which is dominated by season), the model predicts **deviation from the seasonal baseline**. This lets it focus on *which pools need earlier-than-normal visits* based on water quality features rather than just learning the calendar.

### 2. Safety Breach Weighting

Rows where a safety breach occurred at the next visit (pH outside 7.2–8.0 or chlorine < 0.5) are weighted 3× during training, biasing the model toward conservative (shorter) visit recommendations when conditions are risky.

### 3. Regulatory Headroom Features

New features measure how close each parameter is to its regulatory limit:
- `chlorine_headroom_low` = chlorine - 0.5 (distance from pathogen risk)
- `chlorine_headroom_high` = 5.0 - chlorine (distance from pool closure)
- `ph_headroom_low/high` = distance from pH limits
- `min_headroom` = closest distance to any limit

---

## Model Performance

| Model | Target | RMSE | MAE | R² | Best Trees |
|---|---|---|---|---|---|
| **Visit Timing** | days_to_next_visit | **3.15 days** | **1.58 days** | **0.15** | 4 |
| pH | next pH | 0.144 | 0.095 | 0.36 | 78 |
| Chlorine | next free chlorine | 0.747 | 0.518 | 0.33 | 117 |
| Turbidity | next turbidity | 0.213 | 0.116 | 0.43 | 31 |

> [!NOTE]
> The visit timing R² of 0.15 (on reconstructed days) is modest but honest — visit scheduling is primarily driven by season + company logistics, not water quality. The model's value is in identifying the **deviations**: pools that should be visited earlier or later than the seasonal default based on their chemistry.

---

## SHAP Feature Importance

### Visit Timing Model

![SHAP for visit timing](/Users/imadmac/.gemini/antigravity-ide/brain/ff338e04-327f-4c02-84ca-32e4574c14cf/shap_summary_visit_timing.png)

**Top drivers**: `days_since_last_visit` (operational inertia), `visit_day_of_week` (crew scheduling), `visit_month` (residual seasonality), `chlorine_lag1` (water quality signal!), `chlorine_roll3_std` (instability).

### Water Quality Models (for chemical dosage)

````carousel
![SHAP for pH model](/Users/imadmac/.gemini/antigravity-ide/brain/ff338e04-327f-4c02-84ca-32e4574c14cf/shap_summary_ph.png)
<!-- slide -->
![SHAP for chlorine model](/Users/imadmac/.gemini/antigravity-ide/brain/ff338e04-327f-4c02-84ca-32e4574c14cf/shap_summary_chlorine.png)
<!-- slide -->
![SHAP for turbidity model](/Users/imadmac/.gemini/antigravity-ide/brain/ff338e04-327f-4c02-84ca-32e4574c14cf/shap_summary_turbidity.png)
````

**Key insight**: The new V2 features (headroom, breach history) now appear in the top-5 for all water quality models:
- `chlorine_headroom_low` is #1 for chlorine prediction
- `ph_headroom_low` is #2 for pH prediction
- `breach_rate_last5` and `consecutive_clean_visits` add predictive value

---

## Example Prescriptions

The combined output provides visit timing + chemical dosage:

```
Pool: villamagna (1082)
  Current:   pH=7.2, Cl=1.0, Turb=0.2
  Predicted: pH=7.14, Cl=1.25, Turb=0.32
  ⏱  NEXT VISIT IN: 6 days — Soon
  📋 Reasons: Predicted pH (7.14) will breach range (7.2–8.0)
  💊 Chlorine: ✅ Chlorine within range (0.0 kg)
  💊 pH: Add pH plus — predicted pH (7.14) below 7.2 (RD 742/2013)
  💊 Turbidity: ✅ Turbidity within range
```

```
Pool: parque sport iv (22)
  Current:   pH=7.8, Cl=3.3, Turb=0.4
  Predicted: pH=7.54, Cl=3.15, Turb=0.74
  ⏱  NEXT VISIT IN: 8 days — Soon
  📋 Reasons: Headroom to nearest limit is only 0.20
```

---

## Output Files

All in [swimming_pool_eu/](file:///Users/imadmac/projects/swimming_pool_eu):

| File | Size | Description |
|---|---|---|
| [pipeline_v2.py](file:///Users/imadmac/projects/swimming_pool_eu/pipeline_v2.py) | 51 KB | Complete pipeline script |
| [master_dataset.csv](file:///Users/imadmac/projects/swimming_pool_eu/master_dataset.csv) | 2.2 MB | 3,397 rows × 114 columns |
| [xgb_visit_timing.json](file:///Users/imadmac/projects/swimming_pool_eu/xgb_visit_timing.json) | 174 KB | **Primary model** — visit timing |
| [xgb_ph.json](file:///Users/imadmac/projects/swimming_pool_eu/xgb_ph.json) | 390 KB | pH prediction |
| [xgb_chlorine.json](file:///Users/imadmac/projects/swimming_pool_eu/xgb_chlorine.json) | 505 KB | Chlorine prediction |
| [xgb_turbidity.json](file:///Users/imadmac/projects/swimming_pool_eu/xgb_turbidity.json) | 240 KB | Turbidity prediction |
| [shap_summary_*.png](file:///Users/imadmac/projects/swimming_pool_eu/shap_summary_visit_timing.png) | ~80 KB each | 4 SHAP plots |
| [evaluation_report.txt](file:///Users/imadmac/projects/swimming_pool_eu/evaluation_report.txt) | 5.7 KB | Full evaluation |
| [preprocessor.pkl](file:///Users/imadmac/projects/swimming_pool_eu/preprocessor.pkl) | 2.5 KB | Fitted ColumnTransformer |
| [inference_config.json](file:///Users/imadmac/projects/swimming_pool_eu/inference_config.json) | 3.5 KB | Feature lists, medians, regulatory thresholds, seasonal baselines |

---

## What Changed from V1

| Aspect | V1 | V2 |
|---|---|---|
| **Primary target** | next_ph, next_chlorine, next_turbidity | **days_to_next_visit** (seasonal deviation) |
| **Chemical dosage** | ❌ Only predicted values | ✅ Full prescription with kg needed |
| **Regulatory grounding** | Ad-hoc thresholds | **RD 742/2013 + D 85/2018** |
| **Chlorine breach** | > 2.0 mg/L (60% false positive) | **< 0.5 or > 5.0** (correct safety def) |
| **New features** | — | Headroom, trends, breach rate, consecutive clean visits |
| **Models** | 3 (water quality only) | **4** (visit timing + 3 water quality) |
| **Output** | kg of chemicals | Visit timing + urgency + chemicals + regulatory basis |
