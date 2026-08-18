# models/ — Model Artifacts & Run Versioning

Contains trained XGBoost model artifacts, preprocessors, and configuration files organized as versioned run directories.

---

## Directory Layout

```
models/
├── latest.json                 # Active model pointer: {"active_run_id": "<run-id>"}
├── <run-id>/                   # Directory per training run (e.g., v6-setpoint-v2/)
│   ├── xgb_chlorine_next.json  # Free Chlorine Next-Day XGBoost Regressor
│   ├── xgb_ph_next.json        # pH Next-Day XGBoost Regressor
│   ├── xgb_turbidity_next.json # Turbidity Next-Day XGBoost Regressor
│   ├── preprocessor_v6.pkl     # Fitted scikit-learn ColumnTransformer (87 features)
│   └── inference_config_v6.json# Feature names, fill values, thresholds, and post-treatment setpoints
└── archive/                    # Archived / superseded model runs
```

---

## Model Artifact Descriptions

| Artifact | Format | Description |
|:---|:---|:---|
| `xgb_chlorine_next.json` | JSON | XGBoost regressor predicting next-day Free Chlorine ($mg/L$). Formatted with `PipelineConfig` hyperparameters and early stopping. |
| `xgb_ph_next.json` | JSON | XGBoost regressor predicting next-day pH. Formatted with `PipelineConfig` hyperparameters and early stopping. |
| `xgb_turbidity_next.json` | JSON | XGBoost regressor predicting next-day Turbidity ($NTU$). |
| `preprocessor_v6.pkl` | Pickle | Serialized `ColumnTransformer` handling one-hot encoding for categorical variables and passthrough of 87 engineered numeric features. |
| `inference_config_v6.json` | JSON | Serialized run configuration containing feature ordering, imputation fill values, Spanish regulatory limits (RD 742/2013), and post-treatment setpoints (`Cl=2.5`, `pH=7.4`, `Turb=0.5`). |
| `latest.json` | JSON | Active run reference loaded by FastAPI's `PredictionService` (`backend/deps.py`) allowing zero-downtime hot-swapping upon automated retraining. |

---

## Retraining & Promotion Criteria

Models are trained using:
```bash
python pipeline_v6.py
# or
python -m ml.training.train
```

A newly trained model run is automatically promoted to `latest.json` if and only if it satisfies all quality gates:
1. **Free Chlorine MAE** $\le 0.40\text{ mg/L}$
2. **pH MAE** $\le 0.10\text{ pH units}$
3. **Turbidity MAE** $\le 0.15\text{ NTU}$
4. Non-negative $R^2$ scores across all targets.

If a run fails these criteria, the promotion is aborted and previous models in `latest.json` remain active.
