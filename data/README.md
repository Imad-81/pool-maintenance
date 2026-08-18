# data/ — Raw Telemetry, Fleet References & Weather Archives

Contains raw historical data files, fleet reference lists, and cached atmospheric intelligence used by the training pipeline and production backend.

---

## Dataset Files

| File | Type | Description |
|:---|:---|:---|
| **`Merged_2023_2026.xlsx`** | Excel (.xlsx) | **Master Telemetry Dataset** — 42,617 rows across 61 columns spanning January 2, 2023 through August 5, 2026. Contains water quality measurements (pH, Free Chlorine, Turbidity), operational logs, and chemical product applications. |
| **`Listado_piscinas_bomba_cloro.xlsx`** | Excel (.xlsx) | **Fleet Reference List** — Official registry of 138 community pools equipped with automated liquid chlorine dosing pumps in Alicante, Spain. Used to isolate the 135 qualifying active pools. |
| **`weather_alicante_2023_2026.csv`** | CSV | **Open-Meteo Weather Cache** — 1,312 days of high-resolution daily weather for Alicante ($38.3452^\circ\text{ N}, -0.4815^\circ\text{ W}$). Includes temperature, UV index, solar shortwave radiation, precipitation, wind speed, sunshine duration, and $ET_0$ evapotranspiration. |

---

## Database Runtime Storage

- **`store.db`**: Local SQLite database generated during development by `python -m backend.store.migrate`. (Excluded from git via `.gitignore`).
- **PostgreSQL**: In production / Docker Compose, tables are managed via **Prisma ORM** (`prisma/schema.prisma`) connected to PostgreSQL 16.

---

## Updating Weather Data

To refresh the historical and forecast weather cache from Open-Meteo:
```bash
python fetch_weather.py --lat 38.3452 --lon -0.4815 --start 2023-01-01 --end 2026-08-10 -o data/weather_alicante_2023_2026.csv
```
