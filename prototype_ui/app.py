#!/usr/bin/env python3
"""
Pool Predictive Maintenance — Prototype UI Server
Serves the demo dashboard on http://localhost:8050

Usage:
    cd prototype_ui
    python app.py
    Then open http://localhost:8050 in your browser.

Dependencies: flask, pandas, openpyxl (for .xlsx support)
"""

import json
import os
import sys
import csv
import io
import re
import traceback
from datetime import datetime
from difflib import SequenceMatcher

try:
    from flask import Flask, request, jsonify, send_from_directory
except ImportError:
    print("Flask not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'flask'])
    from flask import Flask, request, jsonify, send_from_directory

try:
    import pandas as pd
except ImportError:
    print("pandas not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas'])
    import pandas as pd

# Ensure openpyxl is available for xlsx
try:
    import openpyxl  # noqa: F401
except ImportError:
    print("openpyxl not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'openpyxl'])

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PORT = 8050
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
OUTPUTS_DIR = os.path.join(_PROJECT_ROOT, 'outputs')
MODELS_DIR = os.path.join(_PROJECT_ROOT, 'models')

# Regulatory constants
REG_CHLORINE_MIN = 0.5
REG_CHLORINE_CLOSE = 5.0
REG_PH_MIN = 7.2
REG_PH_MAX = 8.0
REG_TURBIDITY_MAX = 5.0
CHLORINE_IDEAL = 1.25
PH_IDEAL = 7.2

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=_THIS_DIR)

# ---------------------------------------------------------------------------
# In-memory data store
# ---------------------------------------------------------------------------
class DataStore:
    """Holds the active dataset in memory. Supports demo data and user uploads."""

    def __init__(self):
        self.pool_history = {}   # pool_id -> list of reading dicts
        self.pool_latest = {}    # pool_id -> latest reading dict
        self.fleet_data = []     # pre-built fleet overview list
        self.source = 'demo'     # 'demo' | 'uploaded'
        self.source_name = ''    # filename for uploaded data
        self.total_rows = 0
        self.total_pools = 0

        # Keep a copy of demo data so we can reset
        self._demo_pool_history = {}
        self._demo_pool_latest = {}
        self._demo_fleet_data = []
        self._demo_total_rows = 0
        self._demo_total_pools = 0
        self._demo_source_name = ''

    # ------------------------------------------------------------------
    def rebuild_fleet(self):
        """Rebuild fleet_data from pool_latest."""
        self.fleet_data = []
        for pid, latest in self.pool_latest.items():
            urgency = compute_urgency(
                latest.get('ph'), latest.get('free_chlorine'),
                latest.get('turbidity'), latest.get('min_headroom'))
            self.fleet_data.append({
                'pool_id': pid,
                'community_name': latest.get('community_name', ''),
                'reading_date': latest.get('reading_date', ''),
                'ph': latest.get('ph'),
                'free_chlorine': latest.get('free_chlorine'),
                'turbidity': latest.get('turbidity'),
                'urgency': urgency,
                'num_readings': len(self.pool_history.get(pid, [])),
            })
        self.fleet_data.sort(key=lambda x: {
            'Immediate': 0, 'Soon': 1, 'Routine': 2, 'Extended': 3
        }.get(x['urgency'], 4))
        self.total_pools = len(self.pool_latest)
        self.total_rows = sum(len(v) for v in self.pool_history.values())

    # ------------------------------------------------------------------
    def save_demo_snapshot(self):
        """Save current state as the demo baseline (called once at startup)."""
        import copy
        self._demo_pool_history = copy.deepcopy(self.pool_history)
        self._demo_pool_latest = copy.deepcopy(self.pool_latest)
        self._demo_fleet_data = list(self.fleet_data)
        self._demo_total_rows = self.total_rows
        self._demo_total_pools = self.total_pools
        self._demo_source_name = self.source_name

    def reset_to_demo(self):
        """Restore demo data."""
        import copy
        self.pool_history = copy.deepcopy(self._demo_pool_history)
        self.pool_latest = copy.deepcopy(self._demo_pool_latest)
        self.fleet_data = list(self._demo_fleet_data)
        self.total_rows = self._demo_total_rows
        self.total_pools = self._demo_total_pools
        self.source = 'demo'
        self.source_name = self._demo_source_name

    # ------------------------------------------------------------------
    def insert_reading(self, entry):
        """Insert a single reading into the store and update latest/fleet."""
        pid = entry['pool_id']
        if pid not in self.pool_history:
            self.pool_history[pid] = []
        self.pool_history[pid].append(entry)
        self.pool_history[pid].sort(key=lambda x: x.get('reading_date', ''))

        # Update latest
        if (pid not in self.pool_latest or
                entry.get('reading_date', '') >= self.pool_latest[pid].get('reading_date', '')):
            self.pool_latest[pid] = entry

        self.rebuild_fleet()

    # ------------------------------------------------------------------
    def load_from_rows(self, rows, source='uploaded', source_name=''):
        """Replace current data with parsed rows (list of dicts)."""
        self.pool_history = {}
        self.pool_latest = {}
        for entry in rows:
            pid = entry.get('pool_id', '').strip()
            if not pid:
                continue
            rd = entry.get('reading_date', '')
            if not rd:
                continue
            if entry.get('ph') is None and entry.get('free_chlorine') is None:
                continue

            if pid not in self.pool_history:
                self.pool_history[pid] = []
            self.pool_history[pid].append(entry)

            if pid not in self.pool_latest or rd > self.pool_latest[pid].get('reading_date', ''):
                self.pool_latest[pid] = entry

        # Sort histories
        for pid in self.pool_history:
            self.pool_history[pid].sort(key=lambda x: x.get('reading_date', ''))

        self.source = source
        self.source_name = source_name
        self.rebuild_fleet()


store = DataStore()

# ---------------------------------------------------------------------------
# Load seasonal baselines
# ---------------------------------------------------------------------------
seasonal_baselines = {}
config_path = os.path.join(MODELS_DIR, 'inference_config.json')
if os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)
    seasonal_baselines = {int(k): v for k, v in config.get('monthly_medians', {}).items()}

# ---------------------------------------------------------------------------
# Utility functions (same logic as before)
# ---------------------------------------------------------------------------
def compute_urgency(ph, cl, turb, min_headroom=None):
    if cl is not None and cl < REG_CHLORINE_MIN:
        return 'Immediate'
    if ph is not None and (ph < REG_PH_MIN or ph > REG_PH_MAX):
        return 'Immediate'
    if turb is not None and turb > REG_TURBIDITY_MAX:
        return 'Immediate'
    if min_headroom is not None and min_headroom < 0.3:
        return 'Soon'
    if min_headroom is not None and min_headroom < 0.5:
        return 'Routine'
    return 'Extended'


def prescribe_chemicals(ph, cl, turb, pool_vol=None):
    pool_vol = pool_vol or 50.0
    prescriptions = {}
    if cl is not None and cl < REG_CHLORINE_MIN:
        cl_kg = max(0, (CHLORINE_IDEAL - cl) * pool_vol * 0.00667)
        prescriptions['chlorine'] = {'action': '⚠️ URGENT — Add Sodium Hypochlorite 15%', 'kg': round(cl_kg, 2)}
    elif cl is not None and cl < 1.0:
        cl_kg = max(0, (CHLORINE_IDEAL - cl) * pool_vol * 0.00667)
        prescriptions['chlorine'] = {'action': 'Add Sodium Hypochlorite 15% (maintenance)', 'kg': round(cl_kg, 2)}
    else:
        prescriptions['chlorine'] = {'action': '✅ Within range', 'kg': 0}

    if ph is not None and ph > REG_PH_MAX:
        ph_kg = ((ph - PH_IDEAL) / 0.1) * pool_vol * 0.0075
        prescriptions['ph'] = {'action': 'Add Sodium Bisulfate (pH minus)', 'kg': round(ph_kg, 2)}
    elif ph is not None and ph < REG_PH_MIN:
        ph_kg = ((PH_IDEAL - ph) / 0.1) * pool_vol * 0.01
        prescriptions['ph'] = {'action': 'Add Sodium Carbonate (pH plus)', 'kg': round(ph_kg, 2)}
    else:
        prescriptions['ph'] = {'action': '✅ Within range', 'kg': 0}

    if turb is not None and turb > REG_TURBIDITY_MAX:
        prescriptions['turbidity'] = {'action': '⚠️ Add Flocculant', 'kg': None}
    elif turb is not None and turb > 2.0:
        prescriptions['turbidity'] = {'action': 'Add Flocculant (preventive)', 'kg': None}
    else:
        prescriptions['turbidity'] = {'action': '✅ Within range', 'kg': None}
    return prescriptions


def safe_float(val):
    """Convert a value to float, return None on failure."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val) if pd.notna(val) else None
    s = str(val).strip().replace(',', '.')
    if not s:
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def parse_date_flexible(val):
    """Try multiple date formats; return ISO string or None."""
    if val is None:
        return None
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.isoformat()
    s = str(val).strip()
    if not s:
        return None
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d',
                '%d/%m/%Y %H:%M:%S', '%d/%m/%Y %H:%M', '%d/%m/%Y',
                '%d-%m-%Y %H:%M:%S', '%d-%m-%Y', '%m/%d/%Y %H:%M:%S',
                '%m/%d/%Y', '%Y/%m/%d', '%d.%m.%Y'):
        try:
            return datetime.strptime(s, fmt).isoformat()
        except ValueError:
            continue
    # Last resort: pandas
    try:
        dt = pd.to_datetime(s, dayfirst=True)
        if pd.notna(dt):
            return dt.isoformat()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Column auto-detection
# ---------------------------------------------------------------------------
# Patterns for fuzzy matching upload column headers to internal names.
COLUMN_PATTERNS = {
    'pool_id': ['pool_id', 'pool id', 'poolid', 'pool', 'id', 'piscina', 'nombre'],
    'reading_date': ['reading_date', 'date', 'fecha', 'datetime', 'timestamp', 'fecha_lectura'],
    'ph': ['ph', 'p.h.', 'p.h'],
    'free_chlorine': ['free_chlorine', 'chlorine', 'cl', 'cloro', 'free_cl', 'cloro_libre', 'free chlorine'],
    'turbidity': ['turbidity', 'turb', 'ntu', 'turbidez'],
    'pool_volume_m3': ['pool_volume_m3', 'volume', 'vol', 'volumen', 'm3', 'pool_volume'],
    'community_name': ['community_name', 'community', 'comunidad', 'urbanizacion', 'urbanización', 'location'],
}


def auto_detect_mapping(columns):
    """Given a list of column names, return best-guess mapping {internal_name: source_col}."""
    cols_lower = {c: c.lower().strip() for c in columns}
    mapping = {}
    used = set()

    for internal_name, patterns in COLUMN_PATTERNS.items():
        best_col = None
        best_score = 0.0
        for col, col_low in cols_lower.items():
            if col in used:
                continue
            for pattern in patterns:
                # Exact substring match
                if pattern in col_low or col_low in pattern:
                    score = 0.9 + (len(pattern) / 100.0)
                    if score > best_score:
                        best_score = score
                        best_col = col
                # Fuzzy match
                ratio = SequenceMatcher(None, pattern, col_low).ratio()
                if ratio > 0.7 and ratio > best_score:
                    best_score = ratio
                    best_col = col
        if best_col and best_score > 0.5:
            mapping[internal_name] = best_col
            used.add(best_col)

    return mapping


# ---------------------------------------------------------------------------
# Load demo data at startup
# ---------------------------------------------------------------------------
def load_demo_data():
    data_path = os.path.join(OUTPUTS_DIR, 'master_dataset.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(DATA_DIR, 'merged_pool_data_2017_2022.csv')

    print(f"Loading demo data from {data_path}...")
    rows = []
    with open(data_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pool_id = row.get('pool_id', '').strip()
            if not pool_id:
                continue
            reading_date = row.get('reading_date', '')
            if not reading_date:
                continue
            try:
                ph = float(row['ph']) if row.get('ph', '').strip() else None
                cl = float(row['free_chlorine']) if row.get('free_chlorine', '').strip() else None
                turb = float(row['turbidity']) if row.get('turbidity', '').strip() else None
            except (ValueError, TypeError):
                ph = cl = turb = None
            if ph is None and cl is None:
                continue

            entry = {
                'pool_id': pool_id,
                'community_name': row.get('community_name', ''),
                'reading_date': reading_date,
                'ph': ph,
                'free_chlorine': cl,
                'turbidity': turb,
                'source': 'demo',
            }
            for f_name in ['pool_volume_m3', 'pool_surface_m2', 'min_headroom',
                           'target_ph_next', 'target_chlorine_next',
                           'target_turbidity_next', 'days_to_next_visit']:
                try:
                    entry[f_name] = float(row[f_name]) if row.get(f_name, '').strip() else None
                except (ValueError, TypeError):
                    entry[f_name] = None
            rows.append(entry)

    store.load_from_rows(rows, source='demo',
                         source_name=os.path.basename(data_path))
    store.save_demo_snapshot()
    print(f"Demo data: {store.total_rows} readings, {store.total_pools} pools")


# ---------------------------------------------------------------------------
# Pending upload state (column headers waiting for mapping confirmation)
# ---------------------------------------------------------------------------
_pending_upload = {
    'df': None,          # pandas DataFrame
    'filename': '',
    'columns': [],
    'suggested_mapping': {},
}


# =========================================================================
# Routes
# =========================================================================

@app.route('/')
@app.route('/index.html')
def serve_index():
    return send_from_directory(_THIS_DIR, 'index.html')


# -- Data source status ---------------------------------------------------
@app.route('/api/status')
def api_status():
    return jsonify({
        'source': store.source,
        'source_name': store.source_name,
        'total_rows': store.total_rows,
        'total_pools': store.total_pools,
    })


# -- Fleet overview --------------------------------------------------------
@app.route('/api/fleet')
def api_fleet():
    filter_date = request.args.get('date')
    if filter_date:
        filtered = []
        for pid, history in store.pool_history.items():
            closest = None
            for r in history:
                if r.get('reading_date', '')[:10] <= filter_date:
                    closest = r
            if closest:
                urgency = compute_urgency(
                    closest.get('ph'), closest.get('free_chlorine'),
                    closest.get('turbidity'), closest.get('min_headroom'))
                filtered.append({
                    'pool_id': pid,
                    'community_name': closest.get('community_name', ''),
                    'reading_date': closest.get('reading_date', ''),
                    'ph': closest.get('ph'),
                    'free_chlorine': closest.get('free_chlorine'),
                    'turbidity': closest.get('turbidity'),
                    'urgency': urgency,
                    'num_readings': len([
                        r for r in store.pool_history[pid]
                        if r.get('reading_date', '')[:10] <= filter_date]),
                })
        filtered.sort(key=lambda x: {
            'Immediate': 0, 'Soon': 1, 'Routine': 2, 'Extended': 3
        }.get(x['urgency'], 4))
        return jsonify(filtered)
    return jsonify(store.fleet_data)


# -- Pool detail -----------------------------------------------------------
@app.route('/api/pool')
def api_pool():
    pool_id = request.args.get('id', '')
    if pool_id not in store.pool_history:
        return jsonify({'error': 'Pool not found'}), 404

    history = store.pool_history[pool_id]
    latest = store.pool_latest[pool_id]
    urgency = compute_urgency(
        latest.get('ph'), latest.get('free_chlorine'),
        latest.get('turbidity'), latest.get('min_headroom'))

    try:
        month = int(latest.get('reading_date', '2022-06-01')[5:7])
    except Exception:
        month = 6
    baseline = seasonal_baselines.get(month, 5)
    recommended_days = max(1, round(baseline))
    if urgency == 'Immediate':
        recommended_days = 1
    elif urgency == 'Soon':
        recommended_days = max(1, recommended_days - 2)

    prescriptions = prescribe_chemicals(
        latest.get('ph'), latest.get('free_chlorine'),
        latest.get('turbidity'), latest.get('pool_volume_m3'))

    has_volume = latest.get('pool_volume_m3') is not None
    return jsonify({
        'pool_id': pool_id,
        'community_name': latest.get('community_name', ''),
        'latest': latest,
        'urgency': urgency,
        'recommended_days': recommended_days,
        'prescriptions': prescriptions,
        'pool_volume_m3': latest.get('pool_volume_m3'),
        'has_volume': has_volume,
        'history': history[-200:],
    })


# -- Date range for time-travel slider ------------------------------------
@app.route('/api/dates')
def api_dates():
    all_dates = set()
    for history in store.pool_history.values():
        for r in history:
            d = r.get('reading_date', '')[:10]
            if d:
                all_dates.add(d)
    dates = sorted(all_dates)
    return jsonify({
        'min': dates[0] if dates else '',
        'max': dates[-1] if dates else '',
        'count': len(dates),
    })


# -- Pool IDs (for autocomplete) ------------------------------------------
@app.route('/api/pool-ids')
def api_pool_ids():
    return jsonify(sorted(store.pool_history.keys()))


# =========================================================================
# File upload
# =========================================================================
@app.route('/api/upload', methods=['POST'])
def api_upload():
    """Accept a CSV or XLSX file, parse columns, return headers for mapping."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    f = request.files['file']
    if not f.filename:
        return jsonify({'error': 'Empty filename'}), 400

    fname = f.filename.lower()
    try:
        if fname.endswith('.xlsx') or fname.endswith('.xls'):
            df = pd.read_excel(f, engine='openpyxl')
        elif fname.endswith('.csv'):
            raw = f.read()
            # Try utf-8, then latin-1
            for enc in ('utf-8', 'latin-1', 'cp1252'):
                try:
                    text = raw.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                text = raw.decode('utf-8', errors='replace')
            # Detect delimiter
            first_line = text.split('\n')[0]
            if '\t' in first_line:
                sep = '\t'
            elif ';' in first_line:
                sep = ';'
            else:
                sep = ','
            df = pd.read_csv(io.StringIO(text), sep=sep)
        else:
            return jsonify({
                'error': f'Unsupported file type: {os.path.splitext(fname)[1]}. Please upload a .csv or .xlsx file.'
            }), 400

        if df.empty or len(df.columns) < 2:
            return jsonify({'error': 'The file appears to be empty or has too few columns.'}), 400

        # Save pending upload
        _pending_upload['df'] = df
        _pending_upload['filename'] = f.filename
        _pending_upload['columns'] = list(df.columns.astype(str))
        _pending_upload['suggested_mapping'] = auto_detect_mapping(_pending_upload['columns'])

        # Build preview (first 5 rows)
        preview = df.head(5).fillna('').astype(str).to_dict(orient='records')

        return jsonify({
            'columns': _pending_upload['columns'],
            'suggested_mapping': _pending_upload['suggested_mapping'],
            'filename': f.filename,
            'total_rows': len(df),
            'preview': preview,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Could not parse file: {str(e)}'}), 400


# =========================================================================
# Column mapping confirmation
# =========================================================================
@app.route('/api/map-columns', methods=['POST'])
def api_map_columns():
    """Receive confirmed column mapping, parse data, rebuild store."""
    body = request.get_json(force=True)
    mapping = body.get('mapping', {})  # {internal_name: source_col_name}

    df = _pending_upload.get('df')
    if df is None:
        return jsonify({'error': 'No file has been uploaded yet. Please upload a file first.'}), 400

    # Validate required mappings
    if 'pool_id' not in mapping or 'reading_date' not in mapping:
        return jsonify({'error': 'Pool ID and Date columns are required.'}), 400

    has_measurement = any(k in mapping for k in ('ph', 'free_chlorine', 'turbidity'))
    if not has_measurement:
        return jsonify({
            'error': 'At least one measurement column (pH, Chlorine, or Turbidity) must be mapped.'
        }), 400

    # Parse the dataframe using the mapping
    rows = []
    skipped = []
    for idx, raw_row in df.iterrows():
        entry = {'source': 'uploaded'}
        # Pool ID
        pid = str(raw_row.get(mapping['pool_id'], '')).strip()
        if not pid or pid == 'nan':
            skipped.append({'row': int(idx) + 2, 'reason': 'Missing pool ID'})
            continue
        entry['pool_id'] = pid.lower()

        # Reading date
        rd = parse_date_flexible(raw_row.get(mapping['reading_date']))
        if not rd:
            skipped.append({'row': int(idx) + 2, 'reason': 'Invalid or missing date'})
            continue
        entry['reading_date'] = rd

        # Measurements
        for internal_name in ('ph', 'free_chlorine', 'turbidity'):
            if internal_name in mapping:
                entry[internal_name] = safe_float(raw_row.get(mapping[internal_name]))
            else:
                entry[internal_name] = None

        # Optional fields
        if 'pool_volume_m3' in mapping:
            entry['pool_volume_m3'] = safe_float(raw_row.get(mapping['pool_volume_m3']))
        if 'community_name' in mapping:
            cn = str(raw_row.get(mapping['community_name'], '')).strip()
            entry['community_name'] = cn if cn != 'nan' else ''

        # Must have at least one measurement
        if entry.get('ph') is None and entry.get('free_chlorine') is None and entry.get('turbidity') is None:
            skipped.append({'row': int(idx) + 2, 'reason': 'No valid measurement values'})
            continue

        rows.append(entry)

    if not rows:
        return jsonify({
            'error': 'No valid rows could be parsed from the file. All rows were skipped.',
            'skipped': skipped[:50],
        }), 400

    store.load_from_rows(rows, source='uploaded',
                         source_name=_pending_upload.get('filename', 'uploaded'))

    # Clear pending
    _pending_upload['df'] = None

    return jsonify({
        'success': True,
        'loaded_rows': len(rows),
        'loaded_pools': store.total_pools,
        'skipped_count': len(skipped),
        'skipped': skipped[:50],
    })


# =========================================================================
# Manual data entry
# =========================================================================
@app.route('/api/add-reading', methods=['POST'])
def api_add_reading():
    body = request.get_json(force=True)

    # Validate pool_id
    pool_id = str(body.get('pool_id', '')).strip()
    if not pool_id:
        return jsonify({'error': 'Pool ID is required.'}), 400

    # Validate date
    rd_raw = body.get('reading_date', '')
    rd = parse_date_flexible(rd_raw)
    if not rd:
        return jsonify({'error': 'A valid reading date is required.'}), 400

    # Validate measurements
    ph = safe_float(body.get('ph'))
    cl = safe_float(body.get('free_chlorine'))
    turb = safe_float(body.get('turbidity'))
    vol = safe_float(body.get('pool_volume_m3'))
    community = str(body.get('community_name', '')).strip()

    errors = []
    if ph is not None and (ph < 0 or ph > 14):
        errors.append('pH must be between 0 and 14.')
    if cl is not None and cl < 0:
        errors.append('Chlorine cannot be negative.')
    if turb is not None and turb < 0:
        errors.append('Turbidity cannot be negative.')
    if vol is not None and vol <= 0:
        errors.append('Pool volume must be positive.')
    if ph is None and cl is None and turb is None:
        errors.append('At least one measurement (pH, Chlorine, or Turbidity) is required.')
    if errors:
        return jsonify({'error': ' '.join(errors)}), 400

    entry = {
        'pool_id': pool_id.lower(),
        'reading_date': rd,
        'ph': ph,
        'free_chlorine': cl,
        'turbidity': turb,
        'community_name': community,
        'pool_volume_m3': vol,
        'source': 'manual',
    }
    store.insert_reading(entry)

    return jsonify({
        'success': True,
        'pool_id': entry['pool_id'],
        'reading_date': rd,
    })


# =========================================================================
# Reset to demo
# =========================================================================
@app.route('/api/reset', methods=['POST'])
def api_reset():
    store.reset_to_demo()
    _pending_upload['df'] = None
    return jsonify({
        'success': True,
        'total_rows': store.total_rows,
        'total_pools': store.total_pools,
        'source_name': store.source_name,
    })


# =========================================================================
# Main
# =========================================================================
if __name__ == '__main__':
    load_demo_data()
    print(f"\n{'=' * 50}")
    print(f"  Pool Predictive Maintenance — Demo Dashboard")
    print(f"  Starting on http://localhost:{PORT}")
    print(f"  {store.total_pools} pools loaded")
    print(f"{'=' * 50}\n")
    app.run(host='0.0.0.0', port=PORT, debug=False)
