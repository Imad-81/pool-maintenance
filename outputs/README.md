# outputs/ — Generated Datasets & SHAP Explainability Visualizations

Contains processed datasets, feature importance plots, and operational outputs generated during ML training runs.

---

## Directory Contents

| File | Type | Description |
|:---|:---|:---|
| **`master_dataset_v6.csv`** | CSV | Cleaned, deduplicated, feature-engineered master dataset (38,362 rows × 87 numeric features) produced by `ml.training.steps.load_and_clean_data` and `engineer_features`. |
| **`shap_summary_chlorine_next.png`** | Image (PNG) | SHAP feature importance summary plot for the Free Chlorine Next-Day XGBoost model. |
| **`shap_summary_ph_next.png`** | Image (PNG) | SHAP feature importance summary plot for the pH Next-Day XGBoost model. |
| **`shap_summary_turbidity_next.png`** | Image (PNG) | SHAP feature importance summary plot for the Turbidity Next-Day XGBoost model. |

---

## Regeneration

These artifacts are automatically regenerated during a full training run:
```bash
python pipeline_v6.py
# or
python -m ml.training.train
```

Historical or superseded run outputs are automatically replaced or archived to maintain repository cleanliness.
