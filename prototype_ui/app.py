#!/usr/bin/env python3
"""
Pool Predictive Maintenance — Prototype UI Server (with live ML inference)
Serves the demo dashboard on http://localhost:8050

Usage:
    cd prototype_ui && python app.py
    Open http://localhost:8050 in your browser.

Dependencies: flask, pandas, openpyxl, xgboost, scikit-learn
"""

import json, os, sys, csv, io, traceback, pickle, warnings
from datetime import datetime
from difflib import SequenceMatcher

# Silence "X does not have valid feature names" from LightGBM models (harmless — predictions still work)
warnings.filterwarnings('ignore', message='X does not have valid feature names', category=UserWarning)

# pyrefly: ignore [missing-import]
from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import xgboost as xgb

try:
    import openpyxl  # noqa
except ImportError:
    import subprocess; subprocess.check_call([sys.executable,'-m','pip','install','openpyxl'])

# Feature engineering — mirrors pipeline_v3.py Step 5
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_pipeline import build_features, MIN_READINGS_FOR_MODEL

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PORT = 8050
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
DATA_DIR      = os.path.join(_PROJECT_ROOT, 'data')
OUTPUTS_DIR   = os.path.join(_PROJECT_ROOT, 'outputs')
MODELS_DIR    = os.path.join(_PROJECT_ROOT, 'models')

# Regulatory constants (mirror pipeline_v3.py)
REG_CHLORINE_MIN   = 0.5
REG_CHLORINE_CLOSE = 5.0
REG_PH_MIN         = 7.2
REG_PH_MAX         = 8.0
REG_TURBIDITY_MAX  = 5.0
CHLORINE_IDEAL     = 1.25
PH_IDEAL           = 7.2

app = Flask(__name__, static_folder=_THIS_DIR)

# ---------------------------------------------------------------------------
# Model globals (loaded at startup)
# ---------------------------------------------------------------------------
inference_cfg          = {}
preprocessor           = None
models                 = {}
monthly_medians_dict   = {}
fill_values            = {}
all_numeric_features   = []
categorical_features   = []
chlorine_breach_threshold = 0.5
models_loaded          = False
models_error           = None

def load_models():
    global inference_cfg, preprocessor, models, monthly_medians_dict
    global fill_values, all_numeric_features, categorical_features
    global chlorine_breach_threshold, models_loaded, models_error

    print("\nLoading trained models...")
    try:
        with open(os.path.join(MODELS_DIR, 'inference_config.json')) as f:
            inference_cfg = json.load(f)

        fill_values           = inference_cfg.get('fill_values', {})
        all_numeric_features  = inference_cfg.get('all_numeric_features', [])
        categorical_features  = inference_cfg.get('categorical_features', ['pool_type','deck_type'])
        chlorine_breach_threshold = float(inference_cfg.get('chlorine_breach_threshold', 0.5))
        monthly_medians_dict  = {int(k): float(v)
                                 for k,v in inference_cfg.get('monthly_medians',{}).items()}

        with open(os.path.join(MODELS_DIR, 'preprocessor.pkl'), 'rb') as f:
            preprocessor = pickle.load(f)
        print("  ✓ preprocessor.pkl")

        specs = [
            ('ph',           'best_ph.pkl',           'xgb_ph.json',           xgb.XGBRegressor),
            ('chlorine',     'best_chlorine.pkl',     'xgb_chlorine.json',     xgb.XGBRegressor),
            ('turbidity',    'best_turbidity.pkl',    'xgb_turbidity.json',    xgb.XGBRegressor),
            ('chlorine_clf', 'best_chlorine_clf.pkl', 'xgb_chlorine_clf.json', xgb.XGBClassifier),
        ]
        for name, pkl_name, json_name, cls in specs:
            pkl_p  = os.path.join(MODELS_DIR, pkl_name)
            json_p = os.path.join(MODELS_DIR, json_name)
            if os.path.exists(pkl_p):
                with open(pkl_p, 'rb') as f:
                    models[name] = pickle.load(f)
                print(f"  ✓ {pkl_name}  ({name})")
            elif os.path.exists(json_p):
                m = cls(); m.load_model(json_p)
                models[name] = m
                print(f"  ✓ {json_name} ({name})")
            else:
                print(f"  ⚠  {name}: not found")

        models_loaded = True
        print(f"  ✓ {len(models)} models loaded | breach threshold={chlorine_breach_threshold:.4f}")

    except Exception as e:
        models_error  = str(e)
        models_loaded = False
        traceback.print_exc()
        print(f"  ✗ Model load failed: {e}\n  Dashboard will use rule-based fallback for all pools.")

# ---------------------------------------------------------------------------
# Column auto-detection for file uploads
# ---------------------------------------------------------------------------
COLUMN_PATTERNS = {
    'pool_id':        ['pool_id','pool id','poolid','pool','id','piscina','nombre'],
    'reading_date':   ['reading_date','date','fecha','datetime','timestamp','fecha_lectura'],
    'ph':             ['ph','p.h.','p.h'],
    'free_chlorine':  ['free_chlorine','chlorine','cl','cloro','free_cl','cloro_libre','free chlorine'],
    'turbidity':      ['turbidity','turb','ntu','turbidez'],
    'pool_volume_m3': ['pool_volume_m3','volume','vol','volumen','m3','pool_volume'],
    'community_name': ['community_name','community','comunidad','urbanizacion','urbanización','location'],
}

def auto_detect_mapping(columns):
    cols_lower = {c: c.lower().strip() for c in columns}
    mapping, used = {}, set()
    for internal, patterns in COLUMN_PATTERNS.items():
        best_col, best_score = None, 0.0
        for col, col_low in cols_lower.items():
            if col in used: continue
            for pat in patterns:
                if pat in col_low or col_low in pat:
                    score = 0.9 + len(pat)/100
                    if score > best_score: best_score, best_col = score, col
                ratio = SequenceMatcher(None, pat, col_low).ratio()
                if ratio > 0.7 and ratio > best_score:
                    best_score, best_col = ratio, col
        if best_col and best_score > 0.5:
            mapping[internal] = best_col
            used.add(best_col)
    return mapping

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_float(val):
    if val is None: return None
    if isinstance(val, (int, float)): return float(val) if pd.notna(val) else None
    s = str(val).strip().replace(',','.')
    if not s: return None
    try: return float(s)
    except: return None

def parse_date_flexible(val):
    if val is None: return None
    if isinstance(val, (pd.Timestamp, datetime)): return val.isoformat()
    s = str(val).strip()
    if not s: return None
    for fmt in ('%Y-%m-%dT%H:%M','%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S','%Y-%m-%d %H:%M','%Y-%m-%d',
                '%d/%m/%Y %H:%M:%S','%d/%m/%Y %H:%M','%d/%m/%Y',
                '%d-%m-%Y %H:%M:%S','%d-%m-%Y','%m/%d/%Y %H:%M:%S',
                '%m/%d/%Y','%Y/%m/%d','%d.%m.%Y'):
        try: return datetime.strptime(s, fmt).isoformat()
        except: pass
    try:
        dt = pd.to_datetime(s, dayfirst=True)
        if pd.notna(dt): return dt.isoformat()
    except: pass
    return None

# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------
def predict_pool(pool_id: str, history: list) -> dict:
    """Route a pool through model inference or rule-based fallback."""
    if not history:
        return _rule_based({}, "No readings available.")

    latest  = history[-1]
    ph      = latest.get('ph')
    cl      = latest.get('free_chlorine')
    turb    = latest.get('turbidity')
    pool_vol = latest.get('pool_volume_m3') or fill_values.get('pool_volume_m3', 220.0)
    has_vol  = latest.get('pool_volume_m3') is not None
    n        = len(history)

    def fallback(reason):
        return _rule_based(latest, reason, pool_vol=pool_vol, has_vol=has_vol)

    if n < MIN_READINGS_FOR_MODEL:
        return fallback(
            f"Only {n} reading{'s' if n!=1 else ''} for this pool — "
            f"need at least {MIN_READINGS_FOR_MODEL} to generate trend-based predictions. "
            "Showing rule-based estimate instead."
        )

    if not models_loaded:
        return fallback(f"Models unavailable ({models_error or 'load failed'}).")

    try:
        df_hist  = pd.DataFrame(history)
        df_hist['reading_date'] = pd.to_datetime(df_hist['reading_date'], errors='coerce')
        df_hist  = df_hist.sort_values('reading_date').reset_index(drop=True)

        pv_default = latest.get('pool_volume_m3') or fill_values.get('pool_volume_m3', 220.0)
        df_feat  = build_features(df_hist, fill_values=fill_values, pool_volume_default=pv_default)

        uses_defaults = []
        if not has_vol:               uses_defaults.append('pool_volume_m3')
        if df_feat['pool_type'].iloc[-1] == 'unknown': uses_defaults.append('pool_type')
        if df_feat['deck_type'].iloc[-1] == 'unknown': uses_defaults.append('deck_type')

        row = df_feat.iloc[-1:].copy()
        for col in all_numeric_features:
            if col not in row.columns: row[col] = fill_values.get(col, 0.0)
            row[col] = row[col].fillna(fill_values.get(col, 0.0))
        for col in categorical_features:
            if col not in row.columns: row[col] = 'unknown'
            row[col] = row[col].fillna('unknown')

        X = preprocessor.transform(row[categorical_features + all_numeric_features])

        # Water quality
        pred_ph   = float(models['ph'].predict(X)[0])
        pred_cl   = float(models['chlorine'].predict(X)[0])
        pred_turb = float(models['turbidity'].predict(X)[0])

        # Breach probability
        breach_proba = 0.0
        if 'chlorine_clf' in models:
            breach_proba = float(models['chlorine_clf'].predict_proba(X)[0][1])

        # Urgency based purely on chemical state (no visit timing ML)
        reasons, urgency = [], 'Extended'
        
        # Immediate Urgency: Active regulatory breaches right now
        if cl is not None and cl < REG_CHLORINE_MIN:
            urgency = 'Immediate'
            reasons.append(f"⚠️ Current chlorine ({cl:.1f}) BELOW {REG_CHLORINE_MIN} mg/L — pathogen risk (RD 742/2013)")
        if ph is not None and (ph < REG_PH_MIN or ph > REG_PH_MAX):
            urgency = 'Immediate'
            reasons.append(f"⚠️ Current pH ({ph:.1f}) OUTSIDE {REG_PH_MIN}–{REG_PH_MAX} (RD 742/2013)")
            
        # Soon Urgency: Predicted breaches, high probability alarms, or low headroom
        if breach_proba >= chlorine_breach_threshold:
            if urgency != 'Immediate': urgency = 'Soon'
            reasons.append(f"🚨 Preventive alert: {breach_proba:.1%} probability chlorine drops below {REG_CHLORINE_MIN} mg/L")
        if pred_cl < REG_CHLORINE_MIN:
            if urgency != 'Immediate': urgency = 'Soon'
            reasons.append(f"Predicted chlorine ({pred_cl:.2f}) will breach minimum ({REG_CHLORINE_MIN})")
        if pred_ph < REG_PH_MIN or pred_ph > REG_PH_MAX:
            if urgency != 'Immediate': urgency = 'Soon'
            reasons.append(f"Predicted pH ({pred_ph:.2f}) will breach range ({REG_PH_MIN}–{REG_PH_MAX})")
        if not reasons:
            min_hd = row['min_headroom'].iloc[0]
            if pd.notna(min_hd) and min_hd < 0.3:
                urgency = 'Soon'
                reasons.append(f"Headroom to nearest limit is only {min_hd:.2f}")
            else:
                reasons.append("Parameters stable, within regulatory range")

        if urgency == 'Immediate':
            pred_days = 1
        elif urgency == 'Soon':
            pred_days = 3
        else:
            pred_days = 30  # Standard long-term routine schedule when no chemical intervention is needed

        return {
            'source': 'model',
            'urgency': urgency,
            'reasons': reasons,
            'recommended_days': pred_days,
            'prescriptions': _prescriptions_from_predicted(pred_ph, pred_cl, pred_turb, pool_vol),
            'predicted_next': {'ph': round(pred_ph,2), 'free_chlorine': round(pred_cl,2), 'turbidity': round(pred_turb,2)},
            'breach_proba': breach_proba,
            'uses_defaults': uses_defaults,
            'pool_volume_m3': pool_vol,
            'has_volume': has_vol,
        }

    except Exception as e:
        traceback.print_exc()
        return fallback(f"Feature error ({str(e)[:80]}) — rule-based estimate shown.")


def _rule_based(latest, reason, pool_vol=None, has_vol=False):
    ph   = latest.get('ph')   if latest else None
    cl   = latest.get('free_chlorine') if latest else None
    turb = latest.get('turbidity') if latest else None
    pool_vol = pool_vol or fill_values.get('pool_volume_m3', 220.0)

    urgency = 'Immediate' if (
        (cl is not None and cl < REG_CHLORINE_MIN) or
        (ph is not None and (ph < REG_PH_MIN or ph > REG_PH_MAX)) or
        (turb is not None and turb > REG_TURBIDITY_MAX)
    ) else 'Extended'

    days = 1 if urgency == 'Immediate' else 30

    return {
        'source': 'rule_based',
        'reason': reason,
        'urgency': urgency,
        'recommended_days': days,
        'prescriptions': _prescriptions_from_current(ph, cl, turb, pool_vol, basis='rules'),
        'predicted_next': {'ph': None, 'free_chlorine': None, 'turbidity': None},
        'breach_proba': (1.0 if cl is not None and cl < REG_CHLORINE_MIN else
                         0.4 if cl is not None and cl < 1.0 else 0.0),
        'uses_defaults': [],
        'pool_volume_m3': pool_vol,
        'has_volume': has_vol,
    }


def _prescriptions_from_predicted(pred_ph, pred_cl, pred_turb, vol):
    vol = vol or 220.0
    rx = {}
    if pred_cl < REG_CHLORINE_MIN:
        rx['chlorine'] = {'action': f'⚠️ URGENT — Add Sodium Hypochlorite 15% (predicted below {REG_CHLORINE_MIN} mg/L)',
                          'kg': round(max(0,(CHLORINE_IDEAL-pred_cl)*vol*0.00667),2), 'basis':'model'}
    elif pred_cl < 1.0:
        rx['chlorine'] = {'action': 'Add Sodium Hypochlorite 15% (maintenance)',
                          'kg': round(max(0,(CHLORINE_IDEAL-pred_cl)*vol*0.00667),2), 'basis':'model'}
    else:
        rx['chlorine'] = {'action': '✅ Chlorine within range', 'kg': 0, 'basis':'model'}

    if pred_ph > REG_PH_MAX:
        rx['ph'] = {'action': f'Add Sodium Bisulfate (pH minus) — exceeds {REG_PH_MAX}',
                    'kg': round(((pred_ph-PH_IDEAL)/0.1)*vol*0.0075,2), 'basis':'model'}
    elif pred_ph > 7.6:
        rx['ph'] = {'action': 'Add Sodium Bisulfate (pH minus) — approaching upper limit',
                    'kg': round(((pred_ph-PH_IDEAL)/0.1)*vol*0.0075,2), 'basis':'model'}
    elif pred_ph < REG_PH_MIN:
        rx['ph'] = {'action': f'Add Sodium Carbonate (pH plus) — below {REG_PH_MIN}',
                    'kg': round(((PH_IDEAL-pred_ph)/0.1)*vol*0.01,2), 'basis':'model'}
    else:
        rx['ph'] = {'action': '✅ pH within range', 'kg': 0, 'basis':'model'}

    if pred_turb > REG_TURBIDITY_MAX:
        rx['turbidity'] = {'action': f'⚠️ Add Flocculant — exceeds {REG_TURBIDITY_MAX} NTU', 'kg': None, 'basis':'model'}
    elif pred_turb > 2.0:
        rx['turbidity'] = {'action': 'Add Flocculant (preventive)', 'kg': None, 'basis':'model'}
    else:
        rx['turbidity'] = {'action': '✅ Turbidity within range', 'kg': None, 'basis':'model'}
    return rx


def _prescriptions_from_current(ph, cl, turb, vol, basis='rules'):
    vol = vol or 220.0
    rx = {}
    if cl is not None and cl < REG_CHLORINE_MIN:
        rx['chlorine'] = {'action': '⚠️ URGENT — Add Sodium Hypochlorite 15%',
                          'kg': round(max(0,(CHLORINE_IDEAL-cl)*vol*0.00667),2), 'basis':basis}
    elif cl is not None and cl < 1.0:
        rx['chlorine'] = {'action': 'Add Sodium Hypochlorite 15% (maintenance)',
                          'kg': round(max(0,(CHLORINE_IDEAL-cl)*vol*0.00667),2), 'basis':basis}
    else:
        rx['chlorine'] = {'action': '✅ Within range', 'kg': 0, 'basis':basis}

    if ph is not None and ph > REG_PH_MAX:
        rx['ph'] = {'action': 'Add Sodium Bisulfate (pH minus)',
                    'kg': round(((ph-PH_IDEAL)/0.1)*vol*0.0075,2), 'basis':basis}
    elif ph is not None and ph < REG_PH_MIN:
        rx['ph'] = {'action': 'Add Sodium Carbonate (pH plus)',
                    'kg': round(((PH_IDEAL-ph)/0.1)*vol*0.01,2), 'basis':basis}
    else:
        rx['ph'] = {'action': '✅ Within range', 'kg': 0, 'basis':basis}

    if turb is not None and turb > REG_TURBIDITY_MAX:
        rx['turbidity'] = {'action': '⚠️ Add Flocculant', 'kg': None, 'basis':basis}
    elif turb is not None and turb > 2.0:
        rx['turbidity'] = {'action': 'Add Flocculant (preventive)', 'kg': None, 'basis':basis}
    else:
        rx['turbidity'] = {'action': '✅ Within range', 'kg': None, 'basis':basis}
    return rx

# ---------------------------------------------------------------------------
# DataStore
# ---------------------------------------------------------------------------
class DataStore:
    def __init__(self):
        self.pool_history = {}; self.pool_latest = {}; self.fleet_data = []
        self.source = 'demo'; self.source_name = ''
        self.total_rows = 0;   self.total_pools = 0
        self._snap = {}

    def rebuild_fleet(self):
        self.fleet_data = []
        for pid, latest in self.pool_latest.items():
            hist = self.pool_history.get(pid, [latest])
            r    = predict_pool(pid, hist)
            self.fleet_data.append({
                'pool_id': pid,
                'community_name': latest.get('community_name',''),
                'reading_date':   latest.get('reading_date',''),
                'ph':             latest.get('ph'),
                'free_chlorine':  latest.get('free_chlorine'),
                'turbidity':      latest.get('turbidity'),
                'urgency':         r['urgency'],
                'num_readings':    len(hist),
                'prediction_source': r['source'],
                'breach_proba':    r.get('breach_proba', 0.0),
            })
        self.fleet_data.sort(key=lambda x: {'Immediate':0,'Soon':1,'Routine':2,'Extended':3}.get(x['urgency'],4))
        self.total_pools = len(self.pool_latest)
        self.total_rows  = sum(len(v) for v in self.pool_history.values())

    def save_snapshot(self):
        import copy
        self._snap = {
            'pool_history': copy.deepcopy(self.pool_history),
            'pool_latest':  copy.deepcopy(self.pool_latest),
            'fleet_data':   list(self.fleet_data),
            'total_rows':   self.total_rows,
            'total_pools':  self.total_pools,
            'source_name':  self.source_name,
        }

    def reset_to_demo(self):
        import copy
        s = self._snap
        self.pool_history = copy.deepcopy(s['pool_history'])
        self.pool_latest  = copy.deepcopy(s['pool_latest'])
        self.fleet_data   = list(s['fleet_data'])
        self.total_rows   = s['total_rows']
        self.total_pools  = s['total_pools']
        self.source = 'demo'; self.source_name = s['source_name']

    def insert_reading(self, entry):
        pid = entry['pool_id']
        if pid not in self.pool_history: self.pool_history[pid] = []
        self.pool_history[pid].append(entry)
        self.pool_history[pid].sort(key=lambda x: x.get('reading_date',''))
        if (pid not in self.pool_latest or
                entry.get('reading_date','') >= self.pool_latest[pid].get('reading_date','')):
            self.pool_latest[pid] = entry

        # Surgical update: only recompute the one affected pool, not all 475
        hist    = self.pool_history[pid]
        latest  = self.pool_latest[pid]
        r       = predict_pool(pid, hist)
        new_row = {
            'pool_id':            pid,
            'community_name':     latest.get('community_name',''),
            'reading_date':       latest.get('reading_date',''),
            'ph':                 latest.get('ph'),
            'free_chlorine':      latest.get('free_chlorine'),
            'turbidity':          latest.get('turbidity'),
            'urgency':            r['urgency'],
            'num_readings':       len(hist),
            'prediction_source':  r['source'],
            'breach_proba':       r.get('breach_proba', 0.0),
        }
        # Replace existing entry or append
        existing_idx = next((i for i,p in enumerate(self.fleet_data) if p['pool_id'] == pid), None)
        if existing_idx is not None:
            self.fleet_data[existing_idx] = new_row
        else:
            self.fleet_data.append(new_row)
        # Re-sort by urgency
        self.fleet_data.sort(key=lambda x: {'Immediate':0,'Soon':1,'Routine':2,'Extended':3}.get(x['urgency'],4))
        self.total_pools = len(self.pool_latest)
        self.total_rows  = sum(len(v) for v in self.pool_history.values())

    def load_from_rows(self, rows, source='uploaded', source_name=''):
        self.pool_history = {}; self.pool_latest = {}
        for entry in rows:
            pid = entry.get('pool_id','').strip()
            if not pid: continue
            rd  = entry.get('reading_date','')
            if not rd: continue
            if entry.get('ph') is None and entry.get('free_chlorine') is None: continue
            if pid not in self.pool_history: self.pool_history[pid] = []
            self.pool_history[pid].append(entry)
            if pid not in self.pool_latest or rd > self.pool_latest[pid].get('reading_date',''):
                self.pool_latest[pid] = entry
        for pid in self.pool_history:
            self.pool_history[pid].sort(key=lambda x: x.get('reading_date',''))
        self.source = source; self.source_name = source_name
        self.rebuild_fleet()


store = DataStore()

# ---------------------------------------------------------------------------
# Load demo data
# ---------------------------------------------------------------------------
def load_demo_data():
    path = os.path.join(OUTPUTS_DIR, 'master_dataset.csv')
    if not os.path.exists(path):
        path = os.path.join(DATA_DIR, 'merged_pool_data_2017_2022.csv')

    print(f"Loading demo data from {path}...")
    rows = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pool_id = row.get('pool_id','').strip()
            if not pool_id: continue
            rd = row.get('reading_date','')
            if not rd: continue
            try:
                ph   = float(row['ph'])           if row.get('ph','').strip()           else None
                cl   = float(row['free_chlorine']) if row.get('free_chlorine','').strip() else None
                turb = float(row['turbidity'])     if row.get('turbidity','').strip()     else None
            except: ph = cl = turb = None
            if ph is None and cl is None: continue

            entry = {'pool_id': pool_id, 'community_name': row.get('community_name',''),
                     'reading_date': rd, 'ph': ph, 'free_chlorine': cl, 'turbidity': turb,
                     'source': 'demo'}

            for fn in ['pool_volume_m3','pool_surface_m2','min_headroom',
                       'pool_heated','pool_outdoor','pool_community','pool_private','pool_public',
                       'deck_grass','deck_mixed','deck_paved',
                       'daily_filtration_hours','water_temperature','ph_dosing_pct',
                       'hypochlorite_dosing_pct','last_total_chlorine_applied',
                       'total_ph_minus_product','total_chlorine_product',
                       'filter_diameter','filter_count','motor_count',
                       'target_ph_next','target_chlorine_next','target_turbidity_next',
                       'days_to_next_visit']:
                try:   v = row.get(fn,''); entry[fn] = float(v) if v and v.strip() else None
                except: entry[fn] = None
            rows.append(entry)

    store.load_from_rows(rows, source='demo', source_name=os.path.basename(path))
    store.save_snapshot()
    print(f"Demo data: {store.total_rows} readings, {store.total_pools} pools")


# ---------------------------------------------------------------------------
# Pending upload state
# ---------------------------------------------------------------------------
_pending = {'df': None, 'filename': '', 'columns': [], 'suggested_mapping': {}}

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
@app.route('/index.html')
def serve_index(): return send_from_directory(_THIS_DIR, 'index.html')


@app.route('/api/status')
def api_status():
    return jsonify({'source': store.source, 'source_name': store.source_name,
                    'total_rows': store.total_rows, 'total_pools': store.total_pools,
                    'models_loaded': models_loaded, 'models_error': models_error,
                    'min_readings_for_model': MIN_READINGS_FOR_MODEL})

@app.route('/api/fleet')
def api_fleet():
    fd = request.args.get('date')
    if fd:
        out = []
        for pid, history in store.pool_history.items():
            slc = [r for r in history if r.get('reading_date','')[:10] <= fd]
            if not slc: continue
            closest = slc[-1]
            res = predict_pool(pid, slc)
            out.append({'pool_id': pid,
                        'community_name': closest.get('community_name',''),
                        'reading_date':   closest.get('reading_date',''),
                        'ph':             closest.get('ph'),
                        'free_chlorine':  closest.get('free_chlorine'),
                        'turbidity':      closest.get('turbidity'),
                        'urgency':         res['urgency'],
                        'num_readings':    len(slc),
                        'prediction_source': res['source'],
                        'breach_proba':    res.get('breach_proba',0.0)})
        out.sort(key=lambda x: {'Immediate':0,'Soon':1,'Routine':2,'Extended':3}.get(x['urgency'],4))
        return jsonify(out)
    return jsonify(store.fleet_data)

@app.route('/api/pool')
def api_pool():
    pid = request.args.get('id','')
    if pid not in store.pool_history:
        return jsonify({'error': 'Pool not found'}), 404
    history = store.pool_history[pid]
    latest  = store.pool_latest[pid]
    res     = predict_pool(pid, history)
    return jsonify({'pool_id': pid,
                    'community_name': latest.get('community_name',''),
                    'latest': latest,
                    'urgency': res['urgency'],
                    'recommended_days': res['recommended_days'],
                    'prescriptions':    res['prescriptions'],
                    'pool_volume_m3':   res['pool_volume_m3'],
                    'has_volume':       res['has_volume'],
                    'history':          history[-200:],
                    'prediction': {'source':       res['source'],
                                   'reason':       res.get('reason',''),
                                   'predicted_next': res['predicted_next'],
                                   'breach_proba': res['breach_proba'],
                                   'uses_defaults': res.get('uses_defaults',[]),
                                   'min_readings_for_model': MIN_READINGS_FOR_MODEL,
                                   'history_len':  len(history)}})

@app.route('/api/dates')
def api_dates():
    all_d = sorted({r.get('reading_date','')[:10]
                    for h in store.pool_history.values() for r in h if r.get('reading_date')})
    return jsonify({'min': all_d[0] if all_d else '', 'max': all_d[-1] if all_d else '', 'count': len(all_d)})

@app.route('/api/pool-ids')
def api_pool_ids(): return jsonify(sorted(store.pool_history.keys()))

@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files: return jsonify({'error':'No file uploaded'}), 400
    f = request.files['file']
    if not f.filename: return jsonify({'error':'Empty filename'}), 400
    fname = f.filename.lower()
    try:
        if fname.endswith(('.xlsx','.xls')):
            df = pd.read_excel(f, engine='openpyxl')
        elif fname.endswith('.csv'):
            raw = f.read()
            for enc in ('utf-8','latin-1','cp1252'):
                try: text = raw.decode(enc); break
                except: pass
            else: text = raw.decode('utf-8', errors='replace')
            first = text.split('\n')[0]
            sep = '\t' if '\t' in first else (';' if ';' in first else ',')
            df  = pd.read_csv(io.StringIO(text), sep=sep)
        else:
            return jsonify({'error': f'Unsupported type: {os.path.splitext(fname)[1]}. Use .csv or .xlsx'}), 400
        if df.empty or len(df.columns) < 2:
            return jsonify({'error':'File appears empty or has too few columns.'}), 400
        _pending.update({'df': df, 'filename': f.filename,
                         'columns': list(df.columns.astype(str)),
                         'suggested_mapping': auto_detect_mapping(list(df.columns.astype(str)))})
        return jsonify({'columns': _pending['columns'],
                        'suggested_mapping': _pending['suggested_mapping'],
                        'filename': f.filename, 'total_rows': len(df),
                        'preview': df.head(5).fillna('').astype(str).to_dict(orient='records')})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Could not parse file: {e}'}), 400

@app.route('/api/map-columns', methods=['POST'])
def api_map_columns():
    body    = request.get_json(force=True)
    mapping = body.get('mapping', {})
    df      = _pending.get('df')
    if df is None: return jsonify({'error':'No file uploaded yet.'}), 400
    if 'pool_id' not in mapping or 'reading_date' not in mapping:
        return jsonify({'error':'Pool ID and Date columns are required.'}), 400
    if not any(k in mapping for k in ('ph','free_chlorine','turbidity')):
        return jsonify({'error':'At least one measurement column must be mapped.'}), 400

    rows, skipped = [], []
    for idx, raw in df.iterrows():
        entry = {'source':'uploaded'}
        pid = str(raw.get(mapping['pool_id'],'')).strip()
        if not pid or pid=='nan':
            skipped.append({'row': int(idx)+2, 'reason':'Missing pool ID'}); continue
        entry['pool_id'] = pid.lower()
        rd = parse_date_flexible(raw.get(mapping['reading_date']))
        if not rd:
            skipped.append({'row': int(idx)+2, 'reason':'Invalid date'}); continue
        entry['reading_date'] = rd
        for f2 in ('ph','free_chlorine','turbidity'):
            entry[f2] = safe_float(raw.get(mapping[f2])) if f2 in mapping else None
        if 'pool_volume_m3' in mapping: entry['pool_volume_m3'] = safe_float(raw.get(mapping['pool_volume_m3']))
        if 'community_name' in mapping:
            cn = str(raw.get(mapping['community_name'],'')).strip()
            entry['community_name'] = cn if cn!='nan' else ''
        if all(entry.get(k) is None for k in ('ph','free_chlorine','turbidity')):
            skipped.append({'row': int(idx)+2, 'reason':'No valid measurements'}); continue
        rows.append(entry)

    if not rows: return jsonify({'error':'No valid rows parsed.', 'skipped': skipped[:50]}), 400
    store.load_from_rows(rows, source='uploaded', source_name=_pending.get('filename','uploaded'))
    _pending['df'] = None
    return jsonify({'success': True, 'loaded_rows': len(rows), 'loaded_pools': store.total_pools,
                    'skipped_count': len(skipped), 'skipped': skipped[:50]})

@app.route('/api/add-reading', methods=['POST'])
def api_add_reading():
    body    = request.get_json(force=True)
    pool_id = str(body.get('pool_id','')).strip()
    if not pool_id: return jsonify({'error':'Pool ID is required.'}), 400
    rd = parse_date_flexible(body.get('reading_date',''))
    if not rd: return jsonify({'error':'Valid reading date is required.'}), 400
    ph   = safe_float(body.get('ph'))
    cl   = safe_float(body.get('free_chlorine'))
    turb = safe_float(body.get('turbidity'))
    vol  = safe_float(body.get('pool_volume_m3'))
    errs = []
    if ph   is not None and (ph < 0 or ph > 14):  errs.append('pH must be 0–14.')
    if cl   is not None and cl < 0:                errs.append('Chlorine cannot be negative.')
    if turb is not None and turb < 0:              errs.append('Turbidity cannot be negative.')
    if vol  is not None and vol <= 0:              errs.append('Volume must be positive.')
    if ph is None and cl is None and turb is None: errs.append('At least one measurement required.')
    if errs: return jsonify({'error': ' '.join(errs)}), 400

    entry = {'pool_id': pool_id.lower(), 'reading_date': rd,
             'ph': ph, 'free_chlorine': cl, 'turbidity': turb,
             'community_name': str(body.get('community_name','')).strip(),
             'pool_volume_m3': vol, 'source': 'manual'}
    store.insert_reading(entry)

    hist = store.pool_history[pool_id.lower()]
    res  = predict_pool(pool_id.lower(), hist)
    return jsonify({'success': True, 'pool_id': pool_id.lower(), 'reading_date': rd,
                    'prediction_source': res['source'], 'history_len': len(hist),
                    'min_readings_for_model': MIN_READINGS_FOR_MODEL,
                    'reason': res.get('reason','')})

@app.route('/api/reset', methods=['POST'])
def api_reset():
    store.reset_to_demo(); _pending['df'] = None
    return jsonify({'success': True, 'total_rows': store.total_rows,
                    'total_pools': store.total_pools, 'source_name': store.source_name})

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    load_models()
    load_demo_data()
    print(f"\n{'='*50}")
    print(f"  Pool Predictive Maintenance — Demo Dashboard")
    print(f"  http://localhost:{PORT}")
    print(f"  {store.total_pools} pools | Models: {'✓' if models_loaded else '✗ fallback'}")
    print(f"{'='*50}\n")
    app.run(host='0.0.0.0', port=PORT, debug=False)
