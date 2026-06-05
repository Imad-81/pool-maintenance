# data/

Contains all raw and processed input data files.

| File | Description |
|---|---|
| `merged_pool_data_2017_2022.csv` | **Primary input** — 212,850 rows, 522 pools, Apr 2017–Dec 2022. Used by `pipeline_v2.py`. |
| `raw_data.csv` | Original 2022-only export (4,231 rows, 46 pools). Superseded by the merged dataset. |
| `registros_piscinas_generado-2.numbers` | Original Apple Numbers source file (2022 data). |
| `Pool_data/` | Raw batch `.xlsx` files (batch_001 to batch_012, years 2012–2022) from SPP System. |

> **Note:** Large files (CSVs, Numbers files, Pool_data/) are excluded from git via `.gitignore`.
> Contact the project lead to obtain these files for local development.
