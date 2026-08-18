# Pool Predictive Maintenance & Operations System — Technical Documentation (Version 6.0)

**Author:** Shaik Imaduddin  
**Date:** August 2026  
**Regulatory Framework:** Real Decreto 742/2013 (National) & Decreto 85/2018 (Comunitat Valenciana)  
**Target Region:** Alicante, Spain ($38.3452^\circ\text{ N}, -0.4815^\circ\text{ W}$)  

---

## 1. Executive Summary

The **Spain (Alicante) Pool Predictive Maintenance and Operations System (Version 6.0)** is an enterprise-grade AI and physical kinetics platform designed for collective-use swimming pools (*piscinas de uso colectivo*). 

The platform forecasts next-day water quality parameters (**Free Chlorine**, **pH**, and **Turbidity**), generates **chained multi-day rollouts** bridging irregular technician visits, determines **optimal next technician visit dates** via an intelligent visit recommender, computes **minimal chemical dosing pump configurations** via a vectorized $O(n)$ optimizer, and provides an end-to-end multi-language operations dashboard for fleet management.

### Key Performance Highlights (V6 System)

| Model | Target Variable | Unit | MAE | RMSE | $R^2$ Score | P90 Error | Operational Benchmark |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Model A** | **Free Chlorine Tomorrow** | $mg/L$ | **0.1972** | 0.3447 | **0.2571** | 0.4503 | Genuine setpoint-anchored degradation; eliminates target leakage |
| **Model C** | **pH Tomorrow** | pH units | **0.0332** | 0.0538 | **0.2974** | 0.0811 | Superior precision; well within standard $\pm 0.1$ probe tolerance |
| **Model D** | **Turbidity Tomorrow** | $NTU$ | **0.0420** | 0.0777 | **0.4013** | 0.0940 | Clean water clarity tracking; well within RD 742/2013 ($5.0\text{ NTU}$) |

* **Post-Treatment Setpoint Re-Anchor**: Targets and kinetics degrade from a client-confirmed post-treatment ideal ($\text{Cl}=2.5\text{ mg/L}$, $\text{pH}=7.4$, $\text{Turb}=0.5\text{ NTU}$), resolving the "measure $\rightarrow$ treat $\rightarrow$ degrade $\rightarrow$ re-measure" cycle.
* **Chained Multi-Day Inference Engine**: Simulates intermediate chemical evolution from the last technician visit to tomorrow ($T+1$) with dynamic feature recomputation and weather injection.
* **Visit Recommendation System**: Evaluates kinetic decay curves, regulatory risk, historical visit intervals, and weather severity to prescribe precise visit dates.
* **Vectorized Dosing Optimization**: Evaluates 525 dosing configurations in vectorized $O(n)$ time to determine minimal hypochlorite dosing and filtration hours.
* **Production Prisma ORM & PostgreSQL Architecture**: Unified relational schema supporting fleet telemetry, weather data, incident logs, cleaning schedules, and technician messaging.

---

## 2. Dataset & Data Ingestion Specifications

### 2.1 Dataset Overview
The V6 pipeline ingests `data/Merged_2023_2026.xlsx`, spanning **January 2, 2023 through August 5, 2026** (42,617 raw records across 61 columns).

### 2.2 Pool Scope Filtering
The modeling scope is restricted to community pools equipped with **liquid chlorine dosing pumps** by cross-referencing `data/Listado_piscinas_bomba_cloro.xlsx`:
- **Match Strategy**: 126 pools matched by exact numeric reference code (e.g., `Cabo Verde (19)` $\rightarrow$ `19`), and 9 compound pools matched via fuzzy community name reconciliation (e.g., `654-655`).
- **Result**: **135 active pools retained** (38,362 validated reading rows). Non-qualifying pools (~4,255 rows) are safely filtered out.

### 2.3 Preprocessing & Data Cleaning
1. **Header Normalization**: Spanish column headers are mapped via explicit string dictionaries, resilient to column reordering.
2. **Multi-Visit Handling**: When multiple visits occur on the same day (1,061 cases), the **last reading of the day** is preserved as the official end-of-day pool state, and a binary flag (`multi_visit_day = 1`) is recorded.
3. **Static Attribute Imputation**: Fleet medians are calculated across pools to fill missing physical metadata (volume $m^3$, surface area $m^2$, filter diameter, motor count), raising completeness from $1.1\%$ to **$100\%$**.

---

## 3. External Weather Intelligence (Alicante, Spain)

Chlorine photolysis and chemical decay are heavily governed by atmospheric conditions. The V6 system integrates high-resolution daily weather data for Alicante ($38.3452^\circ\text{ N}, -0.4815^\circ\text{ W}$) fetched from Open-Meteo and cached locally in `data/weather_alicante_2023_2026.csv` (1,312 days).

### 22 Engineered Weather Signals
1. **Current Day Weather (9 features)**: Max/mean temperature ($^\circ\text{C}$), max UV index, clear-sky UV index, solar radiation ($MJ/m^2$), sunshine duration (hours), precipitation ($mm$), max wind speed ($km/h$), and $ET_0$ evapotranspiration ($mm$).
2. **Cumulative Weather Since Last Visit (4 features)**: Accumulated UV, solar radiation, rainfall, and mean temperature over the inter-visit gap ($k$ days).
3. **Tomorrow's Weather Forecast (9 features)**: Prediction-day weather forecast (`w_tmrw_*`), providing forward-looking signals for next-day chemical consumption.

---

## 4. Feature Engineering Pipeline (87 Numeric Signals)

The feature extraction pipeline (`ml/features.py`) constructs **87 clean numeric features**:

1. **Autoregressive Lags & Rolling Statistics**:
   - 1-visit and 2-visit lags (`chlorine_lag1`, `ph_lag1`, `turbidity_lag1`, etc.).
   - 3-visit rolling averages and standard deviations (`chlorine_roll3_mean`, `chlorine_roll3_std`, `ph_roll3_mean`, `ph_roll3_std`).
2. **Regulatory Headrooms**:
   - `chlorine_headroom_low` ($\text{Cl} - 0.5$) and `chlorine_headroom_high` ($5.0 - \text{Cl}$).
   - `ph_headroom_low` ($\text{pH} - 7.2$) and `ph_headroom_high` ($8.0 - \text{pH}$).
   - `turbidity_headroom` ($5.0 - \text{Turb}$).
3. **Setpoint Drift & Accumulation Rates**:
   - `chlorine_decay_rate_from_setpoint` ($(\text{Cl}_{sp} - \text{Cl}) / \max(1, k)$).
   - `ph_drift_rate_from_setpoint` ($(\text{pH} - \text{pH}_{sp}) / \max(1, k)$).
   - `turb_accumulation_rate_from_setpoint` ($(\text{Turb} - \text{Turb}_{sp}) / \max(1, k)$).
4. **Temporal & Calendar Features**:
   - `visit_month`, `visit_day_of_week`, `visit_day_of_year`, `visit_is_summer`.
5. **Physical Pool Characteristics**:
   - `pool_volume_m3`, `surface_area_m2`, `filter_diameter_mm`, `motor_count`.

---

## 5. Machine Learning Architecture (`ml/training/`)

### 5.1 Next-Day Target Formulation
Technician visits are non-daily (average inter-visit gap $k \approx 3$ days). The dataset contains only **pre-treatment readings** (confirmed by Jesús Santana, IBERPISCINAS SLU): the technician measures, records, then adjusts. Degradation therefore evolves **from the assumed post-treatment setpoint**, not from the recorded reading:

$$\text{Target}_{\text{tomorrow}} = \text{Setpoint} + (C_{\text{next\_visit}} - \text{Setpoint}) \times \frac{1}{k}$$

- When $k = 1$ (consecutive day visit), the target is the exact next reading.
- When $k = 3$, the target represents 1 day of degradation from the setpoint toward the next pre-treatment reading.
- When no next visit exists, the target falls back to pure 1-day kinetic decay from the setpoint.

**Configurable Post-Treatment Setpoints**:
- Free Chlorine: `2.5 mg/L` (Alicante field practice; median reading 2.6 mg/L).
- pH: `7.4` (Midpoint of RD 742/2013 optimal range 7.2–8.0).
- Turbidity: `0.5 NTU` (Low ideal baseline, well within RD $\le 5.0$).

### 5.2 Temporal Train/Test Split
To prevent temporal data leakage, an 80/20 chronological split is enforced based on the 80th percentile reading date (**October 7, 2025**):
- **Training Set**: 29,526 rows ($2023\text{--}2025$).
- **Test Set**: 7,382 rows ($2025\text{--}2026$).

### 5.3 XGBoost Hyperparameters
```python
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "early_stopping_rounds": 50,
}
```

---

## 6. Physical Kinetics Rate Integration Engine

When technicians visit irregularly, a single 1-step prediction is insufficient. `ml/inference/predictor.py` executes a **Chained Multi-Step Rollout**:

$$\text{Last Visit }(T_0) \longrightarrow T_1 \longrightarrow \dots \longrightarrow T_{\text{today}} \longrightarrow T_{\text{tomorrow}}$$

At each intermediate step $t \rightarrow t+1$, dynamic feature state recomputation is coupled with **first-principles kinetic rate bounds**:

### 1. Chlorine Photolysis Kinetics
Under solar UV irradiation without active hypochlorite dosing, chlorine degrades via exponential first-order kinetics:
$$k_{\text{decay}} = 0.15 + 0.003 \times \max(0, \text{Solar Radiation} - 15.0)$$
$$\text{Cl}_{\text{anchor}} = \begin{cases} \text{Cl}_{sp} & \text{step} = 1 \\ \text{Cl}_t & \text{step} > 1 \end{cases}$$
$$\text{Cl}_{\text{kinetic}} = \text{Cl}_{\text{anchor}} \times \exp\left(-\frac{k_{\text{decay}}}{3.0}\right)$$
$$\text{Pred Cl}_{t+1} = \max\left(0.0, \min(\text{Raw ML Cl}, \text{Cl}_{\text{kinetic}})\right)$$

### 2. Carbonate Equilibrium & $CO_2$ Outgassing pH Drift
Water turbulence and atmospheric degassing steadily drive pH upward ($+0.035$ to $+0.06$ units/day):
$$\Delta\text{pH}_{\text{drift}} = 0.035 + 0.0015 \times \max(0, \text{Temp}_{\max} - 25.0)$$
$$\text{pH}_{\text{anchor}} = \begin{cases} \text{pH}_{sp} & \text{step} = 1 \\ \text{pH}_t & \text{step} > 1 \end{cases}$$
$$\text{Pred pH}_{t+1} = \min\left(8.6, \max(\text{Raw ML pH}, \text{pH}_{\text{anchor}} + \Delta\text{pH}_{\text{drift}})\right)$$

### 3. Wind-Borne Turbidity Accumulation
Environmental dust and particulate ingress increase turbidity ($+0.045$ to $+0.10$ NTU/day):
$$\Delta\text{Turb} = 0.045 + 0.002 \times \max(0, \text{Wind}_{\max} - 10.0)$$
$$\text{Turb}_{\text{anchor}} = \begin{cases} \text{Turb}_{sp} & \text{step} = 1 \\ \text{Turb}_t & \text{step} > 1 \end{cases}$$
$$\text{Pred Turb}_{t+1} = \min\left(5.0, \max(\text{Raw ML Turb}, \text{Turb}_{\text{anchor}} + \Delta\text{Turb})\right)$$

---

## 7. Intelligent Visit Recommendation Engine (`ml/inference/visit_recommender.py`)

The visit recommender determines the optimal date for the next technician visit by synthesizing four independent operational signals:

1. **Immediate Breach Detection**: If today's predicted chlorine is $<0.5\text{ mg/L}$ or $>5.0\text{ mg/L}$, or pH is outside $[7.2, 8.0]$, urgency is flagged as **URGENT** with a recommended visit date of **Today / Tomorrow**.
2. **Decay Curve Simulation**: Simulates day-by-day chemical evolution up to 14 days into the future. The projected breach date (when chlorine drops below client threshold $1.0\text{ mg/L}$) establishes the physical upper bound for visit timing.
3. **Seasonal Operational Cadence**: Blends physical degradation with Mediterranean seasonal standards:
   - Summer (June–September): Default 2-day cadence.
   - Shoulder (May, October): Default 4-day cadence.
   - Winter (November–April): Default 7-day cadence.
4. **Atmospheric Stress Factor**: Accelerates visit frequency when forecasted UV Index $> 8.0$ or ambient temperature $> 30^\circ\text{C}$.

---

## 8. Chemical Dosing Optimization Engine (`ml/inference/optimiser.py`)

The dosing optimizer utilizes vectorized numpy evaluation across 525 candidate configurations:
- **Search Space**: Hypochlorite dosing percentage $\in [0\%, 100\%]$ (step $5\%$) $\times$ Pump operating hours $\in [0\text{h}, 24\text{h}]$ (step $1\text{h}$) = **525 candidate configurations**.
- **Vectorized $O(n)$ Complexity**: Replaced legacy nested loops with array broadcasting, executing in $<2\text{ ms}$ per pool.
- **Objective Function**: Minimize dosing effort $(\text{Dosing\%} / 100) \times \text{Hours}$ subject to:
  $$\text{Pred Cl}_{\text{tomorrow}} \in [1.0, 1.5]\text{ mg/L} \quad\text{and}\quad \text{Pred pH}_{\text{tomorrow}} \in [7.2, 8.0]$$

---

## 9. Production Full-Stack Architecture

### 9.1 Database & Prisma ORM Schema (`prisma/schema.prisma`)
Relational PostgreSQL schema with models:
- `Pool`: Physical pool profiles, dimensions, and pump configurations.
- `Reading`: Historical and real-time water quality measurements.
- `WeatherDaily`: Historical archives and 7-day forecasts from Open-Meteo.
- `ModelRun`: Versioned model runs, evaluation metrics, and artifact references.
- `Incident`: Safety, mechanical, and water quality incident tracking.
- `CleaningLog`: Filter backwash, vacuuming, and pump maintenance logs.
- `Message`: Dispatch-to-technician communications and automated alerts.

### 9.2 Backend Repository & Caching (`backend/store/repo.py`)
- Fast in-memory caching for fleet summaries and pool profiles.
- Asynchronous APScheduler jobs: 4:00 AM daily weather sync and weekly Monday model retrain.
- Preflight data validation and fuzzy mapping for dataset ingestion.

### 9.3 Frontend Hubs (`frontend/src/`)
- Bilingual React 19 + TypeScript dashboard with zero-dependency i18n (`src/i18n.ts`).
- Dedicated operational hubs: Fleet Command, Pool Detail Analytics, Data Ingestion Studio, Incidents, Cleaning, Messaging, Fleet Analytics, and Admin Control.

---

## 10. Verification & Test Suite

The system includes an extensive 51-test automated suite executed via pytest:
```bash
pytest tests/ -v
```

Testing modules:
- `tests/api/`: REST endpoint contracts, authentication, upload validation, health probes.
- `tests/ml/`: Feature engineering parity, setpoint bounds, model training, dry-runs, optimizer vectorization.
- `tests/ml/test_visit_recommender.py`: Visit recommendation algorithms, decay curves, seasonal cadence.
- `tests/store/`: Database repository, caching layer, date parsing, fuzzy column mapping.
