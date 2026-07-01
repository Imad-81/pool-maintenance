#!/usr/bin/env python3
"""
Pool Predictive Maintenance — Prototype UI Server
Serves the demo dashboard on http://localhost:8050

Usage:
    python app.py
    Then open http://localhost:8050 in your browser.
"""

import http.server
import json
import os
import sys
import csv
from urllib.parse import urlparse, parse_qs
from datetime import datetime

PORT = 8050
_THIS_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.path.join(os.getcwd(), 'prototype_ui')
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')
OUTPUTS_DIR = os.path.join(_PROJECT_ROOT, 'outputs')
MODELS_DIR = os.path.join(_PROJECT_ROOT, 'models')
STATIC_DIR = _THIS_DIR

# Regulatory constants
REG_CHLORINE_MIN = 0.5
REG_CHLORINE_CLOSE = 5.0
REG_PH_MIN = 7.2
REG_PH_MAX = 8.0
REG_TURBIDITY_MAX = 5.0
CHLORINE_IDEAL = 1.25
PH_IDEAL = 7.2

# Pre-load data
print("Loading data...")
data_path = os.path.join(OUTPUTS_DIR, 'master_dataset.csv')
if not os.path.exists(data_path):
    data_path = os.path.join(DATA_DIR, 'merged_pool_data_2017_2022.csv')

# Load inference config for seasonal baselines
config_path = os.path.join(MODELS_DIR, 'inference_config.json')
seasonal_baselines = {}
if os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)
    seasonal_baselines = {int(k): v for k, v in config.get('monthly_medians', {}).items()}

# Read the master dataset efficiently
print(f"Reading {data_path}...")
rows = []
pool_latest = {}  # pool_id -> latest row data
pool_history = {}  # pool_id -> list of readings

needed_cols = {
    'pool_id', 'community_name', 'reading_date', 'ph', 'turbidity', 'free_chlorine',
    'pool_volume_m3', 'pool_surface_m2', 'pool_type', 'deck_type',
    'target_ph_next', 'target_chlorine_next', 'target_turbidity_next',
    'days_to_next_visit', 'min_headroom', 'any_breach',
    'chlorine_headroom_low', 'chlorine_headroom_high',
    'ph_headroom_low', 'ph_headroom_high', 'turbidity_headroom',
}

with open(data_path, 'r', encoding='utf-8', errors='replace') as f:
    reader = csv.DictReader(f)
    all_cols = set(reader.fieldnames)
    use_cols = needed_cols & all_cols
    
    for row in reader:
        pool_id = row.get('pool_id', '').strip()
        if not pool_id:
            continue
        
        reading_date = row.get('reading_date', '')
        if not reading_date:
            continue
            
        try:
            ph = float(row.get('ph', '')) if row.get('ph', '').strip() else None
            cl = float(row.get('free_chlorine', '')) if row.get('free_chlorine', '').strip() else None
            turb = float(row.get('turbidity', '')) if row.get('turbidity', '').strip() else None
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
        }
        
        # Optional fields
        for f_name in ['pool_volume_m3', 'pool_surface_m2', 'min_headroom',
                      'target_ph_next', 'target_chlorine_next', 'target_turbidity_next',
                      'days_to_next_visit']:
            try:
                entry[f_name] = float(row.get(f_name, '')) if row.get(f_name, '').strip() else None
            except (ValueError, TypeError):
                entry[f_name] = None
        
        if pool_id not in pool_history:
            pool_history[pool_id] = []
        pool_history[pool_id].append(entry)
        
        # Track latest
        if pool_id not in pool_latest or reading_date > pool_latest[pool_id]['reading_date']:
            pool_latest[pool_id] = entry

print(f"Loaded {sum(len(v) for v in pool_history.values())} readings for {len(pool_history)} pools")

# Sort histories by date
for pid in pool_history:
    pool_history[pid].sort(key=lambda x: x['reading_date'])


def compute_urgency(ph, cl, turb, min_headroom=None):
    """Determine urgency level based on current readings."""
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
    """Simple prescription logic from pipeline_v3."""
    pool_vol = pool_vol or 50.0
    prescriptions = {}
    
    if cl is not None and cl < REG_CHLORINE_MIN:
        cl_kg = max(0, (CHLORINE_IDEAL - cl) * pool_vol * 0.00667)
        prescriptions['chlorine'] = {'action': f'⚠️ URGENT — Add Sodium Hypochlorite 15%', 'kg': round(cl_kg, 2)}
    elif cl is not None and cl < 1.0:
        cl_kg = max(0, (CHLORINE_IDEAL - cl) * pool_vol * 0.00667)
        prescriptions['chlorine'] = {'action': 'Add Sodium Hypochlorite 15% (maintenance)', 'kg': round(cl_kg, 2)}
    else:
        prescriptions['chlorine'] = {'action': '✅ Within range', 'kg': 0}
    
    if ph is not None and ph > REG_PH_MAX:
        ph_kg = ((ph - PH_IDEAL) / 0.1) * pool_vol * 0.0075
        prescriptions['ph'] = {'action': f'Add Sodium Bisulfate (pH minus)', 'kg': round(ph_kg, 2)}
    elif ph is not None and ph < REG_PH_MIN:
        ph_kg = ((PH_IDEAL - ph) / 0.1) * pool_vol * 0.01
        prescriptions['ph'] = {'action': f'Add Sodium Carbonate (pH plus)', 'kg': round(ph_kg, 2)}
    else:
        prescriptions['ph'] = {'action': '✅ Within range', 'kg': 0}
    
    if turb is not None and turb > REG_TURBIDITY_MAX:
        prescriptions['turbidity'] = {'action': '⚠️ Add Flocculant', 'kg': None}
    elif turb is not None and turb > 2.0:
        prescriptions['turbidity'] = {'action': 'Add Flocculant (preventive)', 'kg': None}
    else:
        prescriptions['turbidity'] = {'action': '✅ Within range', 'kg': None}
    
    return prescriptions


# Build fleet overview
fleet_data = []
for pid, latest in pool_latest.items():
    urgency = compute_urgency(latest['ph'], latest['free_chlorine'], latest['turbidity'], latest.get('min_headroom'))
    fleet_data.append({
        'pool_id': pid,
        'community_name': latest.get('community_name', ''),
        'reading_date': latest['reading_date'],
        'ph': latest['ph'],
        'free_chlorine': latest['free_chlorine'],
        'turbidity': latest['turbidity'],
        'urgency': urgency,
        'num_readings': len(pool_history[pid]),
    })

fleet_data.sort(key=lambda x: {'Immediate': 0, 'Soon': 1, 'Routine': 2, 'Extended': 3}.get(x['urgency'], 4))


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)
        
        if path == '/' or path == '/index.html':
            self.serve_file('index.html', 'text/html')
        elif path == '/api/fleet':
            # Optional date filter
            filter_date = params.get('date', [None])[0]
            if filter_date:
                filtered = []
                for pid, history in pool_history.items():
                    # Find latest reading on or before filter_date
                    closest = None
                    for r in history:
                        if r['reading_date'][:10] <= filter_date:
                            closest = r
                    if closest:
                        urgency = compute_urgency(closest['ph'], closest['free_chlorine'], closest['turbidity'], closest.get('min_headroom'))
                        filtered.append({
                            'pool_id': pid,
                            'community_name': closest.get('community_name', ''),
                            'reading_date': closest['reading_date'],
                            'ph': closest['ph'],
                            'free_chlorine': closest['free_chlorine'],
                            'turbidity': closest['turbidity'],
                            'urgency': urgency,
                            'num_readings': len([r for r in pool_history[pid] if r['reading_date'][:10] <= filter_date]),
                        })
                filtered.sort(key=lambda x: {'Immediate': 0, 'Soon': 1, 'Routine': 2, 'Extended': 3}.get(x['urgency'], 4))
                self.send_json(filtered)
            else:
                self.send_json(fleet_data)
        elif path == '/api/pool':
            pool_id = params.get('id', [''])[0]
            if pool_id in pool_history:
                history = pool_history[pool_id]
                latest = pool_latest[pool_id]
                urgency = compute_urgency(latest['ph'], latest['free_chlorine'], latest['turbidity'], latest.get('min_headroom'))
                
                # Determine next visit recommendation
                month = None
                try:
                    month = int(latest['reading_date'][5:7])
                except:
                    month = 6
                baseline = seasonal_baselines.get(month, 5)
                recommended_days = max(1, round(baseline))
                if urgency == 'Immediate':
                    recommended_days = 1
                elif urgency == 'Soon':
                    recommended_days = max(1, recommended_days - 2)
                
                prescriptions = prescribe_chemicals(
                    latest['ph'], latest['free_chlorine'], latest['turbidity'],
                    latest.get('pool_volume_m3')
                )
                
                self.send_json({
                    'pool_id': pool_id,
                    'community_name': latest.get('community_name', ''),
                    'latest': latest,
                    'urgency': urgency,
                    'recommended_days': recommended_days,
                    'prescriptions': prescriptions,
                    'pool_volume_m3': latest.get('pool_volume_m3'),
                    'history': history[-200:],  # Last 200 readings
                })
            else:
                self.send_json({'error': 'Pool not found'}, 404)
        elif path == '/api/dates':
            # Get date range for time travel slider
            all_dates = set()
            for history in pool_history.values():
                for r in history:
                    all_dates.add(r['reading_date'][:10])
            dates = sorted(all_dates)
            self.send_json({'min': dates[0] if dates else '', 'max': dates[-1] if dates else '', 'count': len(dates)})
        else:
            self.send_error(404)
    
    def serve_file(self, filename, content_type):
        filepath = os.path.join(STATIC_DIR, filename)
        if os.path.exists(filepath):
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.end_headers()
            with open(filepath, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404)
    
    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())
    
    def log_message(self, format, *args):
        # Suppress request logs except errors
        if '404' in str(args) or '500' in str(args):
            super().log_message(format, *args)


if __name__ == '__main__':
    print(f"\n{'='*50}")
    print(f"  Pool Predictive Maintenance — Demo Dashboard")
    print(f"  Starting on http://localhost:{PORT}")
    print(f"  {len(pool_history)} pools loaded")
    print(f"{'='*50}\n")
    
    server = http.server.HTTPServer(('', PORT), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
