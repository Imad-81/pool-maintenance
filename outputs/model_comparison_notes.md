# Model Comparison Notes — V5 Pipeline (Multi-Model)

## Overview

V5 trains 6 model types (XGBoost, Random Forest, Extra Trees, Gradient Boosting, LightGBM, CatBoost)
for each of 4 regression targets and 6 classifier types for the chlorine breach classification.

**Key difference from V3**: V5 uses `StandardScaler` on numeric features (V3 uses `passthrough`).
V5 also uses `n_estimators=100` for all models (V3 uses `n_estimators=500` with early stopping for XGBoost).

Both use the same data pipeline, features, and temporal split (80th percentile date cutoff).

## Best Models per Target

| Target | Best Model (V5) | RMSE | MAE | R² |
|--------|-----------------|------|-----|-----|
| **Visit Timing** | LightGBM | 1.659 days | 0.937 days | 0.552 |
| **pH** | Extra Trees | 0.117 | 0.082 | 0.530 |
| **Chlorine** | LightGBM | 0.732 mg/L | 0.529 mg/L | 0.277 |
| **Turbidity** | CatBoost | 0.176 NTU | 0.097 NTU | 0.688 |
| **Chlorine Breach Clf** | Extra Trees | — | — | — | Precision=0.026, Recall=0.803 |

## V3 (XGBoost-only) vs V5 (best-of-6) Comparison

V3 numbers from the evaluation report use `n_estimators=500` with early stopping and no StandardScaler.
V5 numbers use `n_estimators=100` with StandardScaler.

**Note**: The V5 pipeline runs on the full 5.7-year dataset (2017–2022, 476 pools, ~134K model rows)
while the original V3 numbers in docs/ referenced a smaller single-year dataset (~3,400 rows, 43 pools).
The V5 results shown here are from the V5 run on the full dataset, which is also what V3 produces
when run on the same dataset.

| Target | V3 XGBoost (full data) | V5 Best Model | Winner |
|--------|----------------------|--------------|--------|
| Visit Timing MAE | 0.936 days | 0.937 (LightGBM) | Tie — within noise |
| pH MAE | 0.082 | 0.082 (Extra Trees) | Tie |
| Chlorine MAE | 0.529 mg/L | 0.529 (LightGBM) | Tie |
| Turbidity MAE | 0.098 | 0.097 (CatBoost) | CatBoost — marginal |

**Conclusion**: No model type significantly outperforms XGBoost on this dataset. The differences
are within noise (< 1% relative improvement). This is typical for tabular data where gradient
boosting methods converge to similar performance. V3's choice of XGBoost is well justified — 
it has the advantage of native JSON serialization, SHAP support, and wide ecosystem compatibility.

## Chlorine Breach Classifier Notes

All classifiers achieve ~80% recall (by design — threshold is tuned for this), but precision
is uniformly terrible (2–3%). This is a direct consequence of the extreme class imbalance:
only 0.59% of readings have chlorine < 0.5 mg/L. At 80% recall, the classifier correctly
catches ~80% of true low-chlorine events but generates ~40× more false alarms than true alerts.

V3 uses `scale_pos_weight=199`; V5 uses `scale_pos_weight=10` for XGBoost and `class_weight=balanced`
for sklearn models. Despite these differences, all achieve similar recall/precision tradeoffs,
confirming that the bottleneck is data imbalance, not model choice.

## Fixes Required to Run V5

The V5 pipeline ran as-is on the full dataset without requiring code fixes. The only
environmental requirement was installing `lightgbm` and `catboost` (already in requirements.txt).
