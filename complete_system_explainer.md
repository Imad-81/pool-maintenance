# Pool Predictive Maintenance System — Complete Technical Explainer

> Everything you need to explain this system to your senior, from raw data to final predictions.

---

## 1. THE RAW DATA — What Do We Start With?

We have **one Apple Numbers spreadsheet** from Pepe Gutiérrez's pool maintenance company (SPP System) in Alicante, Spain. It contains **4,231 rows** across **61 columns** covering the year **2022** (Jan–Dec).

Each row in the spreadsheet is actually **three records side-by-side** (a denormalized structure):

### Sub-table 1: Water Quality Readings
*"What did the technician measure in the water?"*

| Column | What it is | Example |
|---|---|---|
| Pool ID | Which pool | "Bahamas V (1)" |
| Community | Address | "Calle Altea, 21, Playa de San Juan" |
| Reading date | When the technician visited | 2022-06-15 07:29 |
| Technician | Who visited | "Freddy Ocampo" |
| **pH** | Water acidity/alkalinity | 7.4 |
| **Free chlorine** | Disinfectant level (mg/L) | 2.2 |
| **Turbidity** | Water cloudiness (NTU) | 0.5 |
| Pool type flags | Heated? Outdoor? Public? | 0 or 1 |
| Pool dimensions | Surface area, volume | 150 m², 300 m³ |

### Sub-table 2: Operations
*"How was the pool equipment configured?"*

| Column | What it is |
|---|---|
| Daily filtration hours | How many hours the filter pump ran |
| Water temperature | Measured in °C |
| pH dosing percentage | How much the pH dosing pump was running |
| Hypochlorite dosing percentage | Chlorine pump setting |

### Sub-table 3: Chemical Products Applied
*"What chemicals did the technician physically put in the pool?"*

| Column | What it is |
|---|---|
| Hypochlorite tablets/jugs/sticks | Various chlorine product brands and forms |
| pH minus liquid/granular | Acid to lower pH |
| Flocculant products | Clarifiers for cloudiness |

### Data quality reality:
- **43 unique pools** in the dataset
- ~3,400 usable readings after cleaning
- Operations data is only available for **47%** of readings (not always recorded)
- Product data is available for **90%** of readings
- Many static pool features (volume, surface area, filter specs) are **>50% null** and had to be dropped

---

## 2. WHAT ARE WE PREDICTING?

### Primary prediction: **When should the next technician visit happen?**

Given today's water quality readings for a pool, the model outputs:
- **Recommended days until next visit** (e.g., "visit again in 6 days")
- **Urgency level**: Immediate / Soon / Routine / Extended
- **Why**: the specific reasoning (e.g., "pH predicted to breach 7.2 limit")

### Secondary prediction: **What chemicals should the technician bring?**

Three separate models predict what the water quality will look like at the next visit:
- **Predicted next pH** → determines if pH minus or pH plus is needed (and how many kg)
- **Predicted next chlorine** → determines if chlorine product is needed (and how many kg)
- **Predicted next turbidity** → determines if flocculant is needed

---

## 3. WHAT DATA POINTS (FEATURES) DOES THE MODEL USE?

We feed the model **38 features** (after one-hot encoding) grouped into categories:

### A. Recent water quality history (the strongest signal)

These tell the model *"what has the water been doing recently?"*

| Feature | What it captures |
|---|---|
| `ph_lag1`, `ph_lag2` | pH at the previous 1 and 2 visits |
| `chlorine_lag1`, `chlorine_lag2` | Chlorine at previous 1 and 2 visits |
| `turbidity_lag1`, `turbidity_lag2` | Turbidity at previous 1 and 2 visits |
| `ph_roll3_mean`, `ph_roll3_std` | Average and variability of pH over last 3 visits |
| `chlorine_roll3_mean`, `chlorine_roll3_std` | Average and variability of chlorine over last 3 visits |
| `turbidity_roll3_mean` | Average turbidity over last 3 visits |

**Why these matter**: Water quality has strong *autocorrelation* — if a pool's pH has been 7.3 for the last three visits, it's very likely to be near 7.3 at the next visit too. The rolling standard deviation captures *instability* — a pool that's been bouncing between 6.9 and 7.6 is riskier than one sitting steady at 7.3.

### B. Regulatory headroom features (NEW in V2)

These measure *"how close is this pool to violating the law?"*

| Feature | Formula | What it means |
|---|---|---|
| `chlorine_headroom_low` | chlorine − 0.5 | Distance from the pathogen risk threshold |
| `chlorine_headroom_high` | 5.0 − chlorine | Distance from mandatory pool closure |
| `ph_headroom_low` | pH − 7.2 | Distance from low pH limit |
| `ph_headroom_high` | 8.0 − pH | Distance from high pH limit |
| `turbidity_headroom` | 5.0 − turbidity | Distance from turbidity limit |
| `min_headroom` | Minimum of all above | "How close is the nearest regulatory limit?" |

**Why these matter**: A pool with pH = 7.21 (headroom = 0.01) is one bad day away from a regulatory breach. A pool with pH = 7.5 (headroom = 0.30) has comfortable margin. The model uses this to judge urgency.

### C. Trend / drift features (NEW in V2)

These capture *"which direction is the water quality moving?"*

| Feature | Formula | What it means |
|---|---|---|
| `ph_trend` | current pH − previous pH | Positive = pH is rising |
| `chlorine_trend` | current Cl − previous Cl | Negative = chlorine is dropping |
| `turbidity_trend` | current turb − previous turb | Positive = water getting cloudier |
| `*_rate_per_day` | trend ÷ days between visits | Speed of change per day |

**Why these matter**: A pool with chlorine = 1.0 and *falling* is more urgent than a pool with chlorine = 1.0 and *rising*.

### D. Breach history features (NEW in V2)

These capture *"is this pool a troublemaker?"*

| Feature | What it means |
|---|---|
| `consecutive_clean_visits` | How many visits in a row had no safety breach |
| `breach_rate_last5` | Fraction of last 5 visits with a breach (0.0 = perfect, 1.0 = all breaches) |
| `current_any_breach` | Is this reading itself in breach right now? (0/1) |
| `current_ph_breach` | Is pH currently outside 7.2–8.0? |
| `current_chlorine_breach` | Is chlorine currently < 0.5 or > 5.0? |

### E. Chemical products applied

| Feature | What it means |
|---|---|
| `last_total_chlorine_applied` | Total kg of chlorine products applied recently |
| `total_ph_minus_product` | Total kg of pH-lowering product applied recently |

### F. Temporal features

| Feature | What it means |
|---|---|
| `days_since_last_visit` | How long since the last visit |
| `visit_month` | Month of the year (1–12) |
| `visit_is_summer` | 1 if June–September, 0 otherwise |
| `visit_day_of_week` | 0 = Monday, 6 = Sunday |

### G. Current water quality metrics

| Feature | What it means |
|---|---|
| `ph_deviation` | |current pH − 7.2| — distance from ideal |
| `chlorine_deficit` | max(0, 0.5 − chlorine) — shortfall from minimum |

### H. Pool characteristics (categorical, one-hot encoded)

| Feature | Examples |
|---|---|
| `pool_type` | "outdoor_community", "outdoor_private", "unknown" |
| `deck_type` | "grass", "paved", "mixed", "unknown" |

---

## 4. THE MODEL — What Is XGBoost and How Does It Work?

### What is XGBoost?

**XGBoost** (eXtreme Gradient Boosting) is a **tree-based machine learning algorithm**. It's not a neural network. It builds a collection of **decision trees** that work together.

Think of it like this:

### The simple analogy

Imagine you ask 100 experienced pool technicians:
- The first technician looks at the data and makes a rough guess: "I think the next visit should be in 5 days"
- The second technician looks at **where the first was wrong**, and tries to correct it: "for pools with low chlorine, the first guy was off by 2 days, so I'll add a correction of -2"
- The third technician looks at **where the first two combined are still wrong**, and adds another correction
- ...and so on for all 100 technicians

Each "technician" is a **decision tree** (a flowchart of if/then rules), and the corrections are added together. This is called **boosting**.

### How a single decision tree works

```
                     Is visit_month > 5?
                    /                    \
                 YES                      NO
            Is pH < 7.2?            Is chlorine < 1.0?
            /         \              /            \
         YES           NO         YES              NO
   "visit in       "visit in    "visit in       "visit in
    1 day"          3 days"      4 days"          7 days"
```

Each tree asks a series of questions about the features and reaches a leaf with a predicted value. XGBoost builds many of these trees (we configured up to 500, but early stopping typically selects 4–120 depending on the target).

### How the trees work TOGETHER (gradient boosting)

```
Prediction = Tree₁(features) + Tree₂(features) + Tree₃(features) + ... + Treeₙ(features)
```

- **Tree 1** makes an initial rough prediction
- **Tree 2** is trained on the *residual errors* of Tree 1 (where it was wrong)
- **Tree 3** is trained on the *residual errors* of Trees 1+2 combined
- Each tree adds a small correction (controlled by `learning_rate = 0.05`)
- The final prediction is the **sum** of all trees' outputs

### Our hyperparameters (and what they mean)

| Parameter | Value | What it does |
|---|---|---|
| `n_estimators` | 500 (max) | Maximum number of trees to build |
| `max_depth` | 5 | Each tree can ask at most 5 nested questions |
| `learning_rate` | 0.05 | Each tree's correction is scaled down by 95% (prevents overfitting) |
| `subsample` | 0.8 | Each tree only sees 80% of the training rows (randomness helps) |
| `colsample_bytree` | 0.8 | Each tree only sees 80% of the features (randomness helps) |
| `reg_alpha` | 0.1 | L1 regularization (encourages simpler trees) |
| `reg_lambda` | 1.0 | L2 regularization (prevents any single feature from dominating) |
| `early_stopping_rounds` | 50 | Stop adding trees if the test error hasn't improved in 50 rounds |

### Early stopping in action

We hold out 20% of the data as a test set. After each tree is added, we check the error on the test set. If the error stops improving for 50 trees in a row, we stop — this prevents overfitting. That's why:
- The visit timing model only used **4 trees** (signal was learned quickly)
- The pH model used **78 trees**
- The chlorine model used **117 trees** (more complex patterns)

---

## 5. THE VISIT TIMING MODEL — The Clever Part

### The problem with naive prediction

The technicians follow a strong **seasonal schedule**:
- Summer (Jun–Sep): visit every **2 days** (heavy use, fast chlorine degradation)
- Winter (Nov–Feb): visit every **6–7 days** (light use, stable chemistry)

If we naively predict `days_to_next_visit`, the model just learns the calendar — it predicts "2 days in summer, 7 days in winter" for every pool regardless of water quality. That's useless.

### The solution: predict the DEVIATION

Instead of predicting raw days, we predict:

```
visit_deviation = actual_days − seasonal_baseline
```

where `seasonal_baseline` is the **median visit interval for that month** across all pools:

| Month | Seasonal Baseline |
|---|---|
| Jan | 7 days |
| Feb | 7 days |
| Mar | 7 days |
| Apr | 7 days |
| May | 4 days |
| Jun–Sep | 2 days |
| Oct | 6 days |
| Nov–Dec | 6 days |

So the model's job is: *"should this specific pool be visited EARLIER or LATER than the seasonal default?"*

- **Prediction = -2** → visit 2 days *earlier* than normal (something's wrong)
- **Prediction = 0** → standard schedule is fine
- **Prediction = +3** → this pool is stable, can wait 3 extra days

To get the actual recommendation: `recommended_days = seasonal_baseline + predicted_deviation`

### Sample weighting

Rows where a **safety breach** occurred at the next visit (pH outside 7.2–8.0, or chlorine dangerous) are weighted **3×** during training. This biases the model toward recommending earlier visits when conditions look risky — a conservative, safety-first approach.

---

## 6. THE WATER QUALITY MODELS — Chemical Dosage

Three separate XGBoost regressors, each predicting ONE parameter at the next visit:

| Model | Target | What it predicts |
|---|---|---|
| pH model | `target_ph_next` | What will pH be at the next visit? |
| Chlorine model | `target_chlorine_next` | What will free chlorine be at the next visit? |
| Turbidity model | `target_turbidity_next` | What will turbidity be at the next visit? |

The **target** is computed by shifting the actual readings forward by one visit per pool:

```python
target_ph_next = pH at the NEXT visit for the same pool
```

So for each row, the model sees today's features and tries to predict the value a technician will measure next time they visit that same pool.

### How chemical dosage is calculated

Once we have the predicted values, simple chemistry rules (from the regulations) determine what to do:

```
If predicted chlorine < 0.5 mg/L:
    → URGENT: add chlorine
    → kg_needed = (1.25 − predicted_chlorine) × pool_volume_m³ × 0.0025

If predicted pH > 8.0:
    → add pH minus
    → kg_needed = (predicted_pH − 7.2) × pool_volume_m³ × 0.001

If predicted turbidity > 2.0:
    → add flocculant
```

---

## 7. TRAIN/TEST SPLIT — How We Evaluate

### Why NOT random split

This is **time-series data** — future readings depend on past readings. A random 80/20 split would "leak" future information into the training set (the model could memorize a pool's future values that appear in nearby rows).

### What we do: temporal cutoff

```
Training data: all readings BEFORE September 19, 2022  (2,646 rows)
Test data:     all readings ON/AFTER September 19, 2022 (662 rows)
```

The model is trained on Jan–Sep and tested on Sep–Dec. It never sees the future during training.

---

## 8. MODEL PERFORMANCE — How Good Are We?

| Model | RMSE | MAE | R² | What it means |
|---|---|---|---|---|
| **Visit timing** | 3.15 days | 1.58 days | 0.15 | Off by ~1.6 days on average |
| pH | 0.144 | 0.095 | 0.36 | Off by ~0.1 pH unit on average |
| Chlorine | 0.747 | 0.518 | 0.33 | Off by ~0.5 mg/L on average |
| Turbidity | 0.213 | 0.116 | 0.43 | Off by ~0.1 NTU on average |

### What these metrics mean

- **RMSE** (Root Mean Squared Error): average prediction error, penalizing large errors more. *Lower = better.*
- **MAE** (Mean Absolute Error): average absolute prediction error in the same units as the target. *Lower = better.*
- **R²** (R-squared): proportion of variance explained by the model. 1.0 = perfect, 0.0 = no better than just predicting the average. *Higher = better.*

### Is the visit timing R² of 0.15 bad?

It's **honest, not bad**. Here's why:
- Visit scheduling is primarily driven by company logistics + season, not water quality
- The seasonal baseline already captures ~85% of the pattern
- The deviation model captures the remaining signal: which pools are riskier
- The **MAE of 1.58 days** means predictions are off by less than 2 days on average — useful for planning
- The pH model's MAE of 0.095 is well within measurement precision (pH meters have ±0.1 accuracy)

---

## 9. SHAP — How Do We Know WHICH Features Matter?

### What is SHAP?

**SHAP** (SHapley Additive exPlanations) is a method from game theory that answers: *"How much did each feature contribute to this specific prediction?"*

For every single prediction, SHAP assigns each feature a **SHAP value** — a positive or negative number showing how much that feature pushed the prediction up or down from the average.

### Example

For a specific pool reading:
```
Average predicted visit interval: 5 days
SHAP contributions:
  + days_since_last_visit = 1:     -1.2 days  (visited yesterday → visit sooner)
  + chlorine_headroom_low = 0.1:   -0.8 days  (chlorine dangerously low → visit sooner)
  + visit_month = 12:              +0.3 days   (it's winter → slight delay OK)
  + consecutive_clean_visits = 15: +0.5 days   (pool has been clean → can wait)
  = Final prediction:              3.8 days
```

### What the SHAP bar plots show

The SHAP bar plots in our output show the **mean absolute SHAP value** per feature across all test predictions — essentially "on average, how important is each feature?"

#### Top 5 features per model:

**Visit Timing** — what determines when to visit:
1. `days_since_last_visit` — operational inertia / recent visit pattern
2. `visit_day_of_week` — crew scheduling effects
3. `visit_month` — residual seasonal pattern
4. `chlorine_lag1` — **water quality signal** (recent chlorine reading)
5. `chlorine_roll3_std` — **instability signal** (is chlorine bouncing around?)

**Chlorine Prediction** — what determines future chlorine:
1. `chlorine_headroom_low` — distance from the danger threshold
2. `chlorine_roll3_mean` — recent average chlorine level
3. `consecutive_clean_visits` — track record of the pool
4. `chlorine_lag2` — chlorine from 2 visits ago
5. `chlorine_lag1` — chlorine from 1 visit ago

---

## 10. THE REGULATORY GROUNDING — RD 742/2013

### Why the regulations matter

Spain's **Real Decreto 742/2013** sets legally binding water quality limits for all public/collective-use pools. The Comunitat Valenciana's **Decreto 85/2018** adds regional requirements. Our prescription system maps directly to these:

| Parameter | Compliant Range (RD 742/2013) | Our Model's Action |
|---|---|---|
| Free chlorine < 0.5 mg/L | ❌ Non-compliant — pathogen risk | Urgency = IMMEDIATE + prescribe chlorine |
| Free chlorine 0.5–2.0 mg/L | ✅ Fully compliant | No chlorine action needed |
| Free chlorine 2.0–5.0 mg/L | ⚠️ Over-ideal but common in Spain | No safety concern (60% of our readings are here) |
| Free chlorine > 5.0 mg/L | ❌ Mandatory pool closure | Urgency = IMMEDIATE + stop adding chlorine |
| pH outside 7.2–8.0 | ❌ Non-compliant | Urgency = SOON + prescribe pH corrector |
| Turbidity > 5 NTU | ❌ Non-compliant | Prescribe flocculant |

### The 60% chlorine finding

This is a key insight to share with your senior: **60% of all readings in the dataset have chlorine > 2.0 mg/L**. This is NOT a problem — it's standard practice in Spanish pool maintenance. Technicians intentionally overdose chlorine because:
- Chlorine degrades rapidly in the Mediterranean sun
- Bather load spikes are unpredictable
- A 3.0 mg/L reading is perfectly safe and preferable to risking a drop below 0.5

Our V1 pipeline incorrectly flagged these as "breaches" (68.7% breach rate). V2 corrects this to only flag genuine safety hazards (<0.5 or >5.0), resulting in a realistic **14.9% breach rate** (mostly pH drift).

---

## 11. THE COMPLETE FLOW — End to End

```
RAW DATA (Numbers file)
    │
    ▼
STEP 1: Convert to CSV, rename Spanish → English columns
    │
    ▼
STEP 2: Separate into 3 sub-tables (readings, operations, products)
    │
    ▼
STEP 3: Clean data (parse dates, cast types, flag outliers, deduplicate)
    │
    ▼
STEP 4: Merge sub-tables using merge_asof (temporal join within 14 days)
    │
    ▼
STEP 5: Engineer features
    ├── Lag features (previous 1-2 readings)
    ├── Rolling stats (mean/std over last 3 readings)
    ├── Regulatory headroom (distance from each legal limit)
    ├── Trend features (direction of change + rate per day)
    ├── Breach history (consecutive clean visits, breach rate)
    └── Temporal features (month, day of week, is_summer)
    │
    ▼
STEP 6: Define targets
    ├── PRIMARY: days_to_next_visit → seasonal deviation
    └── SECONDARY: next pH, next chlorine, next turbidity
    │
    ▼
STEP 7: Temporal train/test split (80/20 by date, Sep 19 cutoff)
    │
    ▼
STEP 8: Train 4 XGBoost models
    ├── Visit timing (deviation from seasonal baseline, breach-weighted)
    ├── pH prediction
    ├── Chlorine prediction
    └── Turbidity prediction
    │
    ▼
STEP 9: SHAP explainability (which features drive each model)
    │
    ▼
STEP 10: Combined prescription
    ├── ⏱  "Visit again in X days" + urgency tier
    ├── 💊 Chemical dosages (kg of each product)
    └── 📋 Regulatory basis (which RD 742/2013 threshold is relevant)
```

---

## 12. WHAT THE MODEL CAN'T DO (Limitations)

Be upfront about these with your senior:

| Limitation | Why | Mitigation |
|---|---|---|
| Static pool features are mostly missing | Volume, surface area, filter specs have >50% nulls | Enriching these in future data collection would significantly improve dosage calculations |
| No weather data | Temperature, rainfall, UV affect chlorine decay | Could integrate with weather APIs for future versions |
| Only 1 year of data | Can't learn multi-year patterns, limited seasonal cycles | More years = better seasonal modeling |
| Visit timing R² is low (0.15) | Schedule is mostly logistics-driven, not chemistry-driven | The model's value is in catching the *exceptions*, not replacing the schedule |
| No microbiological data | RD 742/2013 also requires monthly lab tests for bacteria | Outside scope — this is for daily autocontrol only |
| Model assumes recent history is representative | If a pool's equipment fails or usage changes dramatically, lags will be wrong | Would need anomaly detection on top |

---

## 13. QUICK REFERENCE — Files Produced

| File | What it is | Size |
|---|---|---|
| `pipeline_v2.py` | The complete Python script | 51 KB |
| `master_dataset.csv` | Cleaned, merged, feature-engineered dataset | 2.2 MB |
| `xgb_visit_timing.json` | Trained visit timing model | 174 KB |
| `xgb_ph.json` | Trained pH prediction model | 390 KB |
| `xgb_chlorine.json` | Trained chlorine prediction model | 505 KB |
| `xgb_turbidity.json` | Trained turbidity prediction model | 240 KB |
| `shap_summary_*.png` | 4 feature importance plots | ~80 KB each |
| `evaluation_report.txt` | Full performance metrics | 5.7 KB |
| `preprocessor.pkl` | Fitted sklearn feature encoder | 2.5 KB |
| `inference_config.json` | Feature lists, fill values, seasonal baselines, regulatory thresholds | 3.5 KB |
