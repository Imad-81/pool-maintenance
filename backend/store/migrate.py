"""
Prisma database migration & dataset seeder for PostgreSQL.

Usage:
    python -m backend.store.migrate
    python -m backend.store.migrate --force  (clears data and re-seeds)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from backend.store.client import db, connect_db, disconnect_db
from backend.store import repo

log = logging.getLogger("migrate")


async def run_db_push(force: bool = False) -> None:
    """Run `prisma db push` via subprocess to ensure PostgreSQL schema is synchronized."""
    log.info("Synchronizing PostgreSQL schema with Prisma...")
    project_root = Path(__file__).resolve().parents[2]
    schema_path = project_root / "prisma" / "schema.prisma"

    cmd = ["prisma", "db", "push", "--accept-data-loss", "--skip-generate"]
    if schema_path.exists():
        cmd.extend(["--schema", str(schema_path)])

    env = os.environ.copy()
    # Ensure local venv binaries are on PATH
    venv_bin = str(project_root / "venv" / "bin")
    if venv_bin not in env.get("PATH", ""):
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"

    proc = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        log.error("Prisma db push failed (exit code %d):\nSTDOUT: %s\nSTDERR: %s", proc.returncode, proc.stdout, proc.stderr)
        raise RuntimeError(f"Prisma db push failed: {proc.stderr or proc.stdout}")
    log.info("Prisma schema synchronized: %s", proc.stdout.strip())


async def migrate_data(force: bool = False) -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    master_path = project_root / "outputs" / "master_dataset_v6.csv"
    weather_path = project_root / "data" / "weather_alicante_2023_2026.csv"

    if not master_path.exists():
        log.error("Master dataset not found: %s", master_path)
        sys.exit(1)

    # Sync schema
    await run_db_push(force=force)


    await connect_db()

    try:
        if force:
            log.info("Clearing existing data (--force)...")
            await db.reading.delete_many()
            await db.pool.delete_many()
            await db.weatherdaily.delete_many()
            await db.modelrun.delete_many()
            await db.ingestlog.delete_many()
            log.info("Existing tables cleared.")
        else:
            existing_pools = await db.pool.count()
            if existing_pools > 0:
                existing_readings = await db.reading.count()
                log.info("Database already initialized (%d pools, %d readings). Skipping seeder (use --force to re-seed).", existing_pools, existing_readings)
                return

        # --- 1. Import Pool Metadata ---
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

        log.info("Importing %d unique pools...", len(pool_df))
        pool_records = []
        for _, row in pool_df.iterrows():
            d = {
                c: (None if pd.isna(row[c]) else (float(row[c]) if isinstance(row[c], (int, float)) else row[c]))
                for c in pool_cols if c in row
            }
            d["pool_id"] = str(row["pool_id"]).strip()
            d["community_name"] = str(row.get("community_name", "")).strip() if pd.notna(row.get("community_name")) else None
            for flag in [
                "pool_heated", "pool_community", "pool_outdoor", "pool_private", "pool_public",
                "pool_skimmer", "pool_overflow", "pool_oval", "pool_round",
                "pool_rectangular_0714", "pool_rectangular_07", "vegetation_contamination"
            ]:
                if flag in d and d[flag] is not None:
                    d[flag] = int(d[flag])
            pool_records.append(d)
        await db.pool.create_many(data=pool_records, skip_duplicates=True)
        log.info("Imported %d pools successfully.", len(pool_records))

        # --- 2. Import Readings ---
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
        df_r["date_only"] = df_r["reading_date"].dt.normalize()
        # Deduplicate: keep last reading per pool-day
        df_r = df_r.sort_values(["pool_id", "reading_date"]).drop_duplicates(
            subset=["pool_id", "date_only"], keep="last"
        ).drop(columns=["date_only"])

        log.info("Importing %d deduplicated readings...", len(df_r))
        batch = []
        n_readings = 0
        for _, row in df_r.iterrows():
            r = {
                c: (None if pd.isna(row[c]) else (float(row[c]) if isinstance(row[c], (int, float)) else row[c]))
                for c in reading_cols
            }
            r["pool_id"] = str(row["pool_id"]).strip()
            r["reading_date"] = pd.Timestamp(row["reading_date"]).to_pydatetime()
            r["source"] = "master"
            batch.append(r)
            if len(batch) >= 2000:
                n = await db.reading.create_many(data=batch, skip_duplicates=True)
                n_readings += n
                batch = []
                log.info("...imported %d readings", n_readings)
        if batch:
            n = await db.reading.create_many(data=batch, skip_duplicates=True)
            n_readings += n
        log.info("Imported %d total readings.", n_readings)

        # --- 3. Import Weather Daily Cache ---
        if weather_path.exists():
            log.info("Importing weather cache from %s...", weather_path)
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
            w_cols = [c for c in w_cols if c in df_w.columns]
            df_w = df_w[w_cols]

            w_batch = []
            n_wx = 0
            for _, row in df_w.iterrows():
                r = {
                    c: (None if pd.isna(row[c]) else (float(row[c]) if isinstance(row[c], (int, float)) else row[c]))
                    for c in w_cols
                }
                r["date"] = pd.Timestamp(row["date"]).normalize().to_pydatetime()
                w_batch.append(r)
                if len(w_batch) >= 1000:
                    n = await db.weatherdaily.create_many(data=w_batch, skip_duplicates=True)
                    n_wx += n
                    w_batch = []
            if w_batch:
                n = await db.weatherdaily.create_many(data=w_batch, skip_duplicates=True)
                n_wx += n
            log.info("Imported %d weather days.", n_wx)

        # --- 4. Seed Active Model Run ---
        from ml.training.artifacts import ArtifactStore
        latest_id = ArtifactStore.read_latest_pointer(project_root / "models")
        if latest_id:
            run_dir = project_root / "models" / latest_id
            cfg_path = run_dir / "inference_config_v6.json"
            if cfg_path.exists():
                cfg_data = json.loads(cfg_path.read_text())
                metrics_str = json.dumps(cfg_data.get("metrics", {}))
                schema_str = json.dumps(cfg_data.get("feature_schema", []))
                existing = await db.modelrun.find_unique(where={"run_id": latest_id})
                if existing is None:
                    await repo.add_model_run(
                        run_id=latest_id,
                        artifact_dir=str(run_dir),
                        metrics_json=metrics_str,
                        feature_schema_json=schema_str,
                        is_active=1,
                        client=db,
                    )
                    log.info("Seeded active model run: %s", latest_id)

        log.info("PostgreSQL migration & dataset seeding completed successfully!")
    finally:
        await disconnect_db()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Prisma database migration and seeding")
    parser.add_argument("--force", action="store_true", help="Clear tables and reseed")
    args = parser.parse_args()
    asyncio.run(migrate_data(force=args.force))


if __name__ == "__main__":
    main()