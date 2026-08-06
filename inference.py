#!/usr/bin/env python3
"""
Backward-compatibility shim — delegates to the refactored ml.inference engine.

This file exists so that existing automation and the dashboard docs that
reference `python inference.py` continue to work unchanged.

Usage:
    python inference.py                      # all active pools, today + tomorrow
    python inference.py --pool 461           # single pool
    python inference.py --date 2026-08-10    # specific query date

For all new use-cases, the FastAPI backend is preferred.
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from ml.inference.predictor import PredictionService, predict_forward
from ml.inference.chaining import clamp_horizon

# ---------------------------------------------------------------------------
# Weather lookup adapter for the legacy CLI
# ---------------------------------------------------------------------------
def _make_wx_lookup():
    wx_path = os.path.join(SCRIPT_DIR, "data", "weather_alicante_2023_2026.csv")
    if not os.path.exists(wx_path):
        def _missing(_date, _cols):
            return {c: np.nan for c in _cols}
        return _missing
    df = pd.read_csv(wx_path, parse_dates=["date"])
    rename = {
        "temperature_2m_max": "w_temp_max",
        "temperature_2m_mean": "w_temp_mean",
        "uv_index_max": "w_uv_max",
        "uv_index_clear_sky_max": "w_uv_clear_sky_max",
        "shortwave_radiation_sum": "w_solar_radiation",
        "sunshine_duration": "w_sunshine_hours",
        "precipitation_sum": "w_precipitation_mm",
        "wind_speed_10m_max": "w_wind_max_kmh",
        "et0_fao_evapotranspiration": "w_et0",
    }
    df = df.rename(columns=rename).set_index("date")

    def lookup(date, cols):
        d = pd.Timestamp(date).normalize()
        if d not in df.index:
            return {c: np.nan for c in cols}
        row = df.loc[d]
        return {c: (float(row[c]) if c in row.index and pd.notna(row[c]) else np.nan) for c in cols}

    return lookup


def print_pool_forecast(result):
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return
    lr = result["last_readings"]
    fc = result["forecast"]
    print(f"\n{'─' * 72}")
    print(f"  Pool : {result['pool_id']}")
    print(f"  Last visit: {result['last_visit_date']} "
          f"({result['days_since_visit']} days ago)  |  "
          f"Cl={lr['free_chlorine']:.2f}  pH={lr['ph']:.2f}  Turb={lr['turbidity']:.2f}")
    print()
    print(f"  {'Date':<12}{'Day':<22}{'Cl (mg/L)':<12}{'pH':<8}{'Turb (NTU)':<12}Status")
    print(f"  {'─' * 12}{'─' * 22}{'─' * 12}{'─' * 8}{'─' * 12}{'─' * 30}")
    for _, row in fc.iterrows():
        print(f"  {str(row['date']):<12}{row['day']:<22}"
              f"{row['predicted_cl']:<12.3f}{row['predicted_ph']:<8.3f}"
              f"{row['predicted_turb']:<12.3f}{row['status']}")
    vn = result["visit_needed"]
    print()
    print(f"  📋 {'⚠️  VISIT NEEDED in next 2 days' if vn else '✅ No visit needed in next 2 days'}")


def main():
    parser = argparse.ArgumentParser(description="Pool V6 Inference — chained daily forecast (legacy CLI)")
    parser.add_argument("--pool", type=str, default=None)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    as_of = pd.Timestamp(args.date).normalize() if args.date else pd.Timestamp.now().normalize()

    print(f"\n{'=' * 72}")
    print(f"  POOL PREDICTIVE MAINTENANCE V6 — DASHBOARD FORECAST")
    print(f"  Query date (today): {as_of.date()}")
    print(f"{'=' * 72}")

    svc = PredictionService(os.path.join(SCRIPT_DIR, "models"))
    svc.load()
    wx = _make_wx_lookup()

    df_m = pd.read_csv(os.path.join(SCRIPT_DIR, "outputs", "master_dataset_v6.csv"),
                       parse_dates=["reading_date"])

    if args.pool:
        pool_ids = [args.pool]
    else:
        latest = df_m.groupby("pool_id")["reading_date"].max()
        cutoff = as_of - pd.Timedelta(days=30)
        pool_ids = latest[latest >= cutoff].index.tolist()[: args.top]
        print(f"\n  Active pools (reading within past 30 days): {len(pool_ids)}")

    urgent, advised, routine = [], [], []
    for pid in pool_ids:
        pool_rows = df_m[df_m["pool_id"] == pid].sort_values("reading_date")
        if len(pool_rows) == 0:
            continue
        latest_row = pool_rows.iloc[-1]
        try:
            res = svc.forecast(pid, latest_row, as_of, wx, horizon_days=2)
        except Exception as e:
            print(f"  ⚠ {pid}: {e}")
            continue
        print_pool_forecast(res)
        dt = res["forecast"]
        dt_rows = dt[dt["is_today"] | dt["is_tomorrow"]]
        if dt_rows["cl_breach"].any() or dt_rows["ph_breach"].any():
            urgent.append(pid)
        elif (dt_rows["urgency"] == "Advised").any():
            advised.append(pid)
        else:
            routine.append(pid)

    print(f"\n{'=' * 72}")
    print(f"  SUMMARY — {as_of.date()}")
    print(f"{'=' * 72}")
    print(f"  🚨 URGENT  ({len(urgent)}): {', '.join(str(p) for p in urgent) or 'None'}")
    print(f"  ⚠️  ADVISED ({len(advised)}): {', '.join(str(p) for p in advised) or 'None'}")
    print(f"  ✅ ROUTINE ({len(routine)}): {len(routine)} pools OK")
    print()


if __name__ == "__main__":
    main()