"""
Create the SQLite schema + import master_dataset_v6.csv on first run.

Usage:
    python -m backend.store.migrate
    python -m backend.store.migrate --force  (drop + recreate)

The script imports static pool attributes, readings, and optionally weather
from the cache, so the backend can boot without re-running the full pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

log = logging.getLogger("migrate")


def migrate(force: bool = False) -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    master_path = project_root / "outputs" / "master_dataset_v6.csv"
    weather_path = project_root / "data" / "weather_alicante_2023_2026.csv"

    # Defer store imports until after we've confirmed data exists
    from backend.store.schema import engine, create_all, enable_wal
    from backend.store import repo

    if not master_path.exists():
        log.error("master dataset not found: %s — run `python -m ml.training.train` first", master_path)
        sys.exit(1)

    # Schema
    import sqlmodel
    if force:
        # import sqlmodel inline avoids lazy import
        sqlmodel.SQLModel.metadata.drop_all(engine)
        log.info("dropped all tables (--force)")
    create_all(); enable_wal()
    log.info("schema created (WAL on)")

    # --- import pool metadata ---
    df = pd.read_csv(master_path, low_memory=False)
    pool_cols = [
        "pool_id", "pool_type", "deck_type",
        "pool_volume_m3", "pool_surface_m2", "filter_diameter",
        "filter_count", "motor_count",
        "pool_heated", "pool_community", "pool_outdoor", "pool_private", "pool_public",
        "pool_skimmer", "pool_overflow", "pool_oval", "pool_round",
        "pool_rectangular_0714", "pool_rectangular_07",
        "vegetation_contamination",
        "deck_grass", "deck_mixed", "deck_paved",
    ]
    pool_cols = [c for c in pool_cols if c in df.columns]
    pool_df = df[pool_cols + ["community_name"]].drop_duplicates(subset="pool_id")
    with repo.Session(engine) as session:
        for _, row in pool_df.iterrows():
            d = {c: (None if pd.isna(row[c]) else (float(row[c]) if isinstance(row[c], (int, float)) else row[c]))
                 for c in pool_cols if c in row}
            d["pool_id"] = str(row["pool_id"])
            d["community_name"] = str(row.get("community_name", "")) if pd.notna(row.get("community_name")) else None
            for flag_col in ["pool_heated","pool_community","pool_outdoor","pool_private","pool_public",
                             "pool_skimmer","pool_overflow","pool_oval","pool_round",
                             "pool_rectangular_0714","pool_rectangular_07","vegetation_contamination"]:
                if flag_col in d and isinstance(d[flag_col], float):
                    d[flag_col] = int(d[flag_col])
            repo.upsert_pool(session, d)
        pools_count = len(pool_df)
        session.commit()
    log.info("imported %d pools", pools_count)

    # --- import readings (keep last reading per pool-day, same dedup logic) ---
    reading_cols = [
        "pool_id", "community_name", "reading_date",
        "ph", "turbidity", "free_chlorine",
        "hypochlorite_dosing_pct", "hypochlorite_dosing_hours",
        "ph_dosing_pct", "ph_dosing_hours",
        "daily_filtration_hours", "water_temperature",
    ]
    reading_cols = [c for c in reading_cols if c in df.columns]
    df_r = df[reading_cols].copy()
    df_r["reading_date"] = pd.to_datetime(df_r["reading_date"])
    # dedup: keep last per pool-day
    df_r["date_only"] = df_r["reading_date"].dt.normalize()
    df_r = df_r.sort_values(["pool_id", "reading_date"]).drop_duplicates(
        subset=["pool_id", "date_only"], keep="last"
    ).drop(columns=["date_only"])
    batch, n = [], 0
    with repo.Session(engine) as session:
        for _, row in df_r.iterrows():
            r = {c: (None if pd.isna(row[c]) else (float(row[c]) if isinstance(row[c], (int, float)) else row[c]))
                 for c in reading_cols}
            r["pool_id"] = str(row["pool_id"])
            r["reading_date"] = pd.Timestamp(row["reading_date"]).to_pydatetime()
            r["source"] = "master"
            batch.append(r)
            if len(batch) >= 2000:
                n += repo.upsert_readings_batch(session, batch, source="master")
                batch = []; session.commit()
        if batch:
            n += repo.upsert_readings_batch(session, batch, source="master"); session.commit()
    log.info("imported %d readings (deduplicated)", n)

# --- import weather cache (if present) ---
    if weather_path.exists():
        WX_RENAME = {
            "temperature_2m_max":      "w_temp_max",
            "temperature_2m_mean":     "w_temp_mean",
            "uv_index_max":            "w_uv_max",
            "uv_index_clear_sky_max":  "w_uv_clear_sky_max",
            "shortwave_radiation_sum": "w_solar_radiation",
            "sunshine_duration":       "w_sunshine_hours",
            "precipitation_sum":       "w_precipitation_mm",
            "wind_speed_10m_max":      "w_wind_max_kmh",
            "et0_fao_evapotranspiration": "w_et0",
        }
        w_cols = [
            "date", "w_temp_max", "w_temp_mean", "w_uv_max", "w_uv_clear_sky_max",
            "w_solar_radiation", "w_sunshine_hours", "w_precipitation_mm",
            "w_wind_max_kmh", "w_et0",
        ]
        df_w = pd.read_csv(weather_path, parse_dates=["date"])
        if any(c in df_w.columns for c in WX_RENAME):
            df_w = df_w.rename(columns=WX_RENAME)
        w_cols = [c for c in w_cols if c in df_w.columns]  # filter AFTER rename
        df_w = df_w[w_cols]
        batch, n_w = [], 0
        with repo.Session(engine) as session:
            for _, row in df_w.iterrows():
                r = {c: (None if pd.isna(row[c]) else (float(row[c]) if isinstance(row[c], (int, float)) else row[c]))
                     for c in w_cols}
                r["date"] = pd.Timestamp(row["date"]).to_pydatetime()
                batch.append(r)
                if len(batch) >= 2000:
                    n_w += repo.upsert_weather_batch(session, batch)
                    batch = []; session.commit()
            if batch:
                n_w += repo.upsert_weather_batch(session, batch); session.commit()
        log.info("imported %d weather days", n_w)

    # --- seed model run from active artifact ---
    from ml.training.artifacts import ArtifactStore
    latest_id = ArtifactStore.read_latest_pointer(project_root / "models")
    if latest_id:
        run_dir = project_root / "models" / latest_id
        cfg_path = run_dir / "inference_config_v6.json"
        if cfg_path.exists():
            cfg_data = json.loads(cfg_path.read_text())
            metrics_str = json.dumps(cfg_data.get("metrics", {}))
            schema_str = json.dumps(cfg_data.get("feature_schema", []))
            with repo.Session(engine) as session:
                existing = session.get(repo.ModelRun, latest_id)
                if existing is None:
                    repo.add_model_run(session, latest_id, str(run_dir),
                                       metrics_json=metrics_str,
                                       feature_schema_json=schema_str,
                                       is_active=1)
                    session.commit()
                    log.info("seeded model run %s (active)", latest_id)

    log.info("migration complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    migrate(force=args.force)