import sys
import os

with open('pipeline_v5.py', 'r') as f:
    content = f.read()

# Replace Imports
imports_old = """# ML / Stats
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_recall_curve, classification_report
# pyrefly: ignore [missing-import]
import xgboost as xgb

# Explainability
# pyrefly: ignore [missing-import]
import shap"""

imports_new = """# ML / Stats
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_recall_curve, classification_report
# pyrefly: ignore [missing-import]
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
# pyrefly: ignore [missing-import]
import lightgbm as lgb
# pyrefly: ignore [missing-import]
import catboost as cb

# Explainability
# pyrefly: ignore [missing-import]
import shap"""

content = content.replace(imports_old, imports_new)

# Replace Preprocessor
prep_old = """# --- Preprocessor ---
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', 'passthrough', all_numeric_features),
    ],
    remainder='drop'
)"""

prep_new = """# --- Preprocessor ---
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', StandardScaler(), all_numeric_features),
    ],
    remainder='drop'
)"""
content = content.replace(prep_old, prep_new)

# Replace Step 8 and 9
start_idx = content.find("# STEP 8 — TRAIN MODELS")
end_idx = content.find("# ============================================================================\n# STEP 10 — COMBINED PRESCRIPTION")

step8_9_new = """# STEP 8 — TRAIN MULTIPLE MODELS
# ============================================================================
print_step(8, "TRAIN MULTIPLE MODELS (XGB, RF, ET, GB, LGBM, CB, SVM)")

models_dict = {}
results = {}

def get_regressors():
    return {
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'Extra Trees': ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1),
        'CatBoost': cb.CatBoostRegressor(iterations=100, depth=5, learning_rate=0.05, random_state=42, verbose=0, thread_count=-1),
        'SVM': SVR(C=1.0, epsilon=0.1)
    }

def get_classifiers():
    return {
        'XGBoost': xgb.XGBClassifier(scale_pos_weight=10, n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1),
        'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'Extra Trees': ExtraTreesClassifier(class_weight='balanced', n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42),
        'LightGBM': lgb.LGBMClassifier(class_weight='balanced', n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1),
        'CatBoost': cb.CatBoostClassifier(auto_class_weights='Balanced', iterations=100, depth=5, learning_rate=0.05, random_state=42, verbose=0, thread_count=-1),
        'SVM': SVC(probability=True, class_weight='balanced', random_state=42)
    }

# --- 8A: PRIMARY MODEL — Visit Timing ---
print("=" * 50)
print("  PRIMARY MODEL: days_to_next_visit")
print("=" * 50)

y_train_dev = df_train['visit_deviation'].values
y_test_dev = df_test['visit_deviation'].values
y_test_actual = df_test['days_to_next_visit'].values

sample_weights = np.ones(len(df_train))
breach_mask = df_train['any_breach_next'].values.astype(bool)
sample_weights[breach_mask] = 3.0

best_visit_model = None
best_visit_rmse = float('inf')
visit_results = {}

for name, model in get_regressors().items():
    print(f"  Training {name}...")
    if name == 'XGBoost':
        model.fit(X_train, y_train_dev, sample_weight=sample_weights)
    else:
        # Pass sample weights only if supported easily
        try:
            model.fit(X_train, y_train_dev, sample_weight=sample_weights)
        except:
            model.fit(X_train, y_train_dev)

    y_pred_dev = model.predict(X_test)
    y_pred_days = df_test['seasonal_baseline_days'].values + y_pred_dev
    y_pred_days = np.clip(y_pred_days, 1, 60)
    
    rmse_days = np.sqrt(mean_squared_error(y_test_actual, y_pred_days))
    mae_days = mean_absolute_error(y_test_actual, y_pred_days)
    r2_days = r2_score(y_test_actual, y_pred_days)
    p90_days = np.percentile(np.abs(y_test_actual - y_pred_days), 90)
    
    visit_results[name] = {'rmse': rmse_days, 'mae': mae_days, 'r2': r2_days, 'p90': p90_days, 'rmse_dev': np.sqrt(mean_squared_error(y_test_dev, y_pred_dev))}
    print(f"    RMSE: {rmse_days:.2f} | MAE: {mae_days:.2f} | R²: {r2_days:.4f}")
    
    if rmse_days < best_visit_rmse:
        best_visit_rmse = rmse_days
        best_visit_model = model

print(f"  --> Best Visit Timing Model: {best_visit_model.__class__.__name__} (RMSE: {best_visit_rmse:.2f})")
models_dict['visit_timing'] = best_visit_model
results['visit_timing'] = visit_results[best_visit_model.__class__.__name__] if best_visit_model.__class__.__name__ not in ['XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor', 'RandomForestRegressor', 'ExtraTreesRegressor', 'GradientBoostingRegressor', 'SVR'] else visit_results[[k for k, v in get_regressors().items() if v.__class__ == best_visit_model.__class__][0]]
results['visit_timing_all'] = visit_results

with open(os.path.join(MODELS_DIR, 'best_visit_timing.pkl'), 'wb') as f:
    pickle.dump(best_visit_model, f)

# --- 8B: SECONDARY MODELS ---
print(f"\\n{'='*50}")
print("  SECONDARY MODELS: Water Quality (pH, Chlorine, Turbidity)")
print("=" * 50)

for target_name, target_col in [
    ('ph', 'target_ph_next'),
    ('chlorine', 'target_chlorine_next'),
    ('turbidity', 'target_turbidity_next'),
]:
    print(f"\\n--- {target_name} model ---")
    y_train_t = df_train_wq[target_col].values
    y_test_t = df_test_wq[target_col].values
    
    best_wq_model = None
    best_wq_rmse = float('inf')
    wq_results = {}
    
    for name, model in get_regressors().items():
        print(f"  Training {name}...")
        model.fit(X_train_wq, y_train_t)
        y_pred_t = model.predict(X_test_wq)
        
        rmse_t = np.sqrt(mean_squared_error(y_test_t, y_pred_t))
        mae_t = mean_absolute_error(y_test_t, y_pred_t)
        r2_t = r2_score(y_test_t, y_pred_t)
        p90_t = np.percentile(np.abs(y_test_t - y_pred_t), 90)
        
        wq_results[name] = {'rmse': rmse_t, 'mae': mae_t, 'r2': r2_t, 'p90': p90_t}
        print(f"    RMSE: {rmse_t:.4f} | MAE: {mae_t:.4f} | R²: {r2_t:.4f}")
        
        if rmse_t < best_wq_rmse:
            best_wq_rmse = rmse_t
            best_wq_model = model
            
    print(f"  --> Best {target_name} Model: {best_wq_model.__class__.__name__} (RMSE: {best_wq_rmse:.4f})")
    models_dict[target_name] = best_wq_model
    results[target_name] = wq_results[[k for k, v in get_regressors().items() if v.__class__ == best_wq_model.__class__][0]]
    results[f'{target_name}_all'] = wq_results
    
    with open(os.path.join(MODELS_DIR, f'best_{target_name}.pkl'), 'wb') as f:
        pickle.dump(best_wq_model, f)
        
    if target_name == 'chlorine':
        print(f"\\n  --- Training dedicated Chlorine Breach Classifier (< {REG_CHLORINE_MIN} mg/L) ---")
        y_train_breach = (y_train_t < REG_CHLORINE_MIN).astype(int)
        y_test_breach = (y_test_t < REG_CHLORINE_MIN).astype(int)
        
        best_clf_model = None
        best_clf_score = -1
        clf_results = {}
        best_thresh = 0.5
        best_clf_report = ""
        
        for name, model_clf in get_classifiers().items():
            print(f"  Training {name}...")
            model_clf.fit(X_train_wq, y_train_breach)
            y_pred_proba = model_clf.predict_proba(X_test_wq)[:, 1]
            
            precisions, recalls, thresholds = precision_recall_curve(y_test_breach, y_pred_proba)
            valid_idx = np.where(recalls >= 0.80)[0]
            optimal_idx = valid_idx[-1] if len(valid_idx) > 0 else 0
            if optimal_idx >= len(thresholds):
                optimal_idx = len(thresholds) - 1
            
            opt_thresh = thresholds[optimal_idx]
            opt_prec = precisions[optimal_idx]
            opt_rec = recalls[optimal_idx]
            
            score = opt_prec if opt_rec >= 0.8 else 0
            
            clf_results[name] = {'threshold': opt_thresh, 'precision': opt_prec, 'recall': opt_rec}
            print(f"    Threshold: {opt_thresh:.4f} -> Precision: {opt_prec:.3f}, Recall: {opt_rec:.3f}")
            
            if score > best_clf_score:
                best_clf_score = score
                best_clf_model = model_clf
                best_thresh = opt_thresh
                best_clf_report = classification_report(y_test_breach, (y_pred_proba >= opt_thresh).astype(int))
                
        print(f"  --> Best Classifier: {best_clf_model.__class__.__name__} (Precision: {best_clf_score:.3f})")
        models_dict['chlorine_clf'] = best_clf_model
        results['chlorine']['clf_threshold'] = best_thresh
        results['chlorine']['clf_report'] = best_clf_report
        results['chlorine_clf_all'] = clf_results
        
        with open(os.path.join(MODELS_DIR, 'best_chlorine_clf.pkl'), 'wb') as f:
            pickle.dump(best_clf_model, f)
            
        config_path = os.path.join(MODELS_DIR, 'inference_config.json')
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        cfg['chlorine_breach_threshold'] = float(best_thresh)
        with open(config_path, 'w') as f:
            json.dump(cfg, f, indent=2, default=str)


# ============================================================================
# STEP 9 — SHAP EXPLAINABILITY (Only for Best Models)
# ============================================================================
print_step(9, "SHAP EXPLAINABILITY")

shap_results = {}

for model_name, model in models_dict.items():
    print(f"\\n--- SHAP: {model_name} ---")
    if model_name == 'visit_timing':
        X_shap = X_test
    else:
        X_shap = X_test_wq
        
    try:
        if isinstance(model, SVR) or isinstance(model, SVC):
            X_train_summary = shap.kmeans(X_train_wq if model_name != 'visit_timing' else X_train, 50)
            explainer = shap.KernelExplainer(model.predict if not isinstance(model, SVC) else model.predict_proba, X_train_summary)
            shap_values = explainer.shap_values(X_shap[:100])
            if isinstance(model, SVC):
                shap_values = shap_values[1]
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap)
            
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)
        top15 = feature_importance.head(15)
        
        print(f"  Top 15 features:")
        for feat, val in top15.items():
            print(f"    {feat}: {val:.4f}")
        shap_results[model_name] = top15.to_dict()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        X_shap_plot = X_shap[:100] if isinstance(model, (SVR, SVC)) else X_shap
        shap.summary_plot(shap_values, X_shap_plot, feature_names=feature_names, plot_type='bar', max_display=15, show=False)
        title = f'SHAP Feature Importance — {model_name.upper().replace("_", " ")} Model'
        plt.title(title, fontsize=14)
        plt.tight_layout()
        
        plot_path = os.path.join(OUTPUT_DIR, f'shap_summary_{model_name}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {plot_path}")
    except Exception as e:
        print(f"  Could not compute SHAP for {model_name}: {e}")

"""

content = content[:start_idx] + step8_9_new + "\n" + content[end_idx:]

start_report_idx = content.find('report.append("\\n\\n3. DELIVERABLE 1 — CHLORINE SAFETY FORECASTING & ALERTS")')
end_report_idx = content.find('report.append("\\n\\n6. EXAMPLE PRESCRIPTIONS & ALERTS (REAL-WORLD OUTPUTS)")')

report_new = """report.append("\\n\\n3. MODEL COMPARISON (ALL MODELS)")
report.append("-" * 70)
for tgt in ['visit_timing', 'ph', 'chlorine', 'turbidity']:
    report.append(f"\\n--- {tgt.upper()} REGRESSION MODELS ---")
    for m_name, m_res in results[f'{tgt}_all'].items():
        report.append(f"  {m_name:<20}: RMSE = {m_res['rmse']:.4f} | MAE = {m_res['mae']:.4f} | R² = {m_res['r2']:.4f}")

report.append(f"\\n--- CHLORINE BREACH CLASSIFIERS ---")
for m_name, m_res in results['chlorine_clf_all'].items():
    report.append(f"  {m_name:<20}: Precision = {m_res['precision']:.3f} | Recall = {m_res['recall']:.3f} | Threshold = {m_res['threshold']:.4f}")

report.append("\\n\\n4. DELIVERABLE 1 — CHLORINE SAFETY FORECASTING & ALERTS")
report.append("-" * 70)
c_res = results['chlorine']
best_cl_model_name = models_dict['chlorine'].__class__.__name__
best_clf_model_name = models_dict['chlorine_clf'].__class__.__name__

report.append(f"Target: Predict free chlorine levels and identify regulatory breaches (< {REG_CHLORINE_MIN} mg/L)")
report.append(f"Best Regression Model: {best_cl_model_name}")
report.append(f"Performance:")
report.append(f"  - Mean Absolute Error (MAE): {c_res['mae']:.3f} mg/L")
report.append(f"  - Root Mean Squared Error (RMSE): {c_res['rmse']:.3f} mg/L")
report.append(f"  - R-squared (R²): {c_res['r2']:.3f}")
report.append(f"  - 90th Percentile Error: {c_res['p90']:.3f} mg/L")
report.append(f"")
report.append(f"Safety Alert Classification (< {REG_CHLORINE_MIN} mg/L):")
report.append(f"  Best Classifier Model: {best_clf_model_name}")
report.append(f"  - Optimal Threshold: {results['chlorine'].get('clf_threshold', 0.5):.4f}")
report.append(f"\\n{results['chlorine'].get('clf_report', '')}")
report.append(f"\\nTop Drivers (SHAP) for Chlorine:")
if 'chlorine' in shap_results:
    for i, (feat, val) in enumerate(list(shap_results['chlorine'].items())[:15], 1):
        report.append(f"  {i}. {feat}: {val:.4f}")

report.append("\\n\\n5. DELIVERABLE 2 — WATER CHEMISTRY FORECASTING")
report.append("-" * 70)
report.append("Target: Forecast pH and Turbidity trajectories to maintain ideal water balance.")
ph_res = results['ph']
turb_res = results['turbidity']
best_ph_model_name = models_dict['ph'].__class__.__name__
best_turb_model_name = models_dict['turbidity'].__class__.__name__

report.append(f"pH Best Model: {best_ph_model_name}")
report.append(f"  - RMSE: {ph_res['rmse']:.3f} | MAE: {ph_res['mae']:.3f} | R²: {ph_res['r2']:.3f} | p90: {ph_res['p90']:.3f}")

report.append(f"\\nTurbidity Best Model: {best_turb_model_name}")
report.append(f"  - RMSE: {turb_res['rmse']:.3f} | MAE: {turb_res['mae']:.3f} | R²: {turb_res['r2']:.3f} | p90: {turb_res['p90']:.3f}")

report.append("\\n\\n6. DELIVERABLE 3 — CHEMICAL DEMAND & CONSUMPTION FORECASTING")
report.append("-" * 70)
vt_res = results['visit_timing']
best_vt_model_name = models_dict['visit_timing'].__class__.__name__

report.append(f"Visit Timing Best Model: {best_vt_model_name}")
report.append(f"  - RMSE: {vt_res['rmse']:.2f} days | MAE: {vt_res['mae']:.2f} days | R²: {vt_res['r2']:.3f} | p90: {vt_res['p90']:.2f} days")

"""

content = content[:start_report_idx] + report_new + "\n" + content[end_report_idx:]

content = content.replace("models['visit_timing'], models['ph'], models['chlorine'], models['turbidity']", "models_dict['visit_timing'], models_dict['ph'], models_dict['chlorine'], models_dict['turbidity']")
content = content.replace("models.get('chlorine_clf')", "models_dict.get('chlorine_clf')")

content = content.replace("'xgb_visit_timing.json': MODELS_DIR", "'best_visit_timing.pkl': MODELS_DIR")
content = content.replace("'xgb_ph.json': MODELS_DIR", "'best_ph.pkl': MODELS_DIR")
content = content.replace("'xgb_chlorine.json': MODELS_DIR", "'best_chlorine.pkl': MODELS_DIR")
content = content.replace("'xgb_turbidity.json': MODELS_DIR", "'best_turbidity.pkl': MODELS_DIR")

with open('pipeline_v5.py', 'w') as f:
    f.write(content)
print("Updated pipeline_v5.py successfully.")
