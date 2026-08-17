# data/

Contains all raw input data files used by the V6 training pipeline.

| File | Description |
|---|---|
| `Merged_2023_2026.xlsx` | **Primary dataset** — 42,617 rows across 61 columns (Jan 2023–Aug 2026). |
| `Listado_piscinas_bomba_cloro.xlsx` | Reference list of communities with liquid-chlorine dosing pumps. |
| `weather_alicante_2023_2026.csv` | Cached Open-Meteo daily weather for Alicante (1,312 days). |

`store.db` is the backend's local SQLite database (generated at runtime by
`python -m backend.store.migrate`); it is excluded from git via `.gitignore`.
