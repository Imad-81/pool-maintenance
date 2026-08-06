"""
Single source of truth for regulatory thresholds, client targets, weather
coordinates, XGBoost hyper-parameters and on-disk artifact paths.

Importing `PipelineConfig` has no side-effects (no directories created, no
warnings suppressed, no eager execution) so it is safe to import from tests
and from the FastAPI backend startup path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


# ---------------------------------------------------------------------------
# Regulatory basis: Real Decreto 742/2013, Annexe I
# ---------------------------------------------------------------------------
REG_CHLORINE_MIN       = 0.5   # mg/L — pathogen risk below this
REG_CHLORINE_IDEAL_MAX = 2.0   # mg/L — ideal upper (common practice goes above)
REG_CHLORINE_CLOSE     = 5.0   # mg/L — mandatory closure above this
REG_PH_MIN             = 7.2
REG_PH_MAX             = 8.0
REG_TURBIDITY_MAX      = 5.0   # NTU

# Client optimal target (Jesús Santana brief)
CLIENT_CL_TARGET_MIN   = 1.0   # mg/L
CLIENT_CL_TARGET_MAX   = 1.5   # mg/L
CLIENT_CL_IDEAL        = 1.25  # midpoint
PH_IDEAL               = 7.4   # midpoint of 7.2–8.0 (common Spanish practice)


# ---------------------------------------------------------------------------
# Spanish → snake_case column rename map (robust to Excel column reordering)
# ---------------------------------------------------------------------------
RENAME_MAP: Dict[str, str] = {
    "PISCINA":                                "pool_id",
    "COMUNIDAD":                              "community_name",
    "FECHA":                                  "reading_date",
    "EMPLEADO":                               "technician",
    "PH":                                     "ph",
    "TURBIDEZ":                               "turbidity",
    "CLORO LIBRE":                            "free_chlorine",
    "ABUSO CREMAS PROTECCION":                "sunscreen_abuse",
    "Caudal bomba de PH":                     "ph_pump_flow_rate",
    "Caudal bomba hipoclorito":               "hypochlorite_pump_flow_rate",
    "Caudal del motor":                       "motor_flow_rate",
    "Diametro filtro":                        "filter_diameter",
    "Numero de filtros":                      "filter_count",
    "Número de motores":                      "motor_count",
    "PISCINA CLIMATIZADA":                    "pool_heated",
    "PISCINA COMUNITARIA":                    "pool_community",
    "Piscina con skimmers":                   "pool_skimmer",
    "Piscina desbordante":                    "pool_overflow",
    "PISCINA EXTERIOR":                       "pool_outdoor",
    "Piscina ovalada":                        "pool_oval",
    "PISCINA PARTICULAR":                     "pool_private",
    "PISCINA PUBLICA":                        "pool_public",
    "(0714) Piscina rectangular":             "pool_rectangular_0714",
    "(07) Piscina rectangular":               "pool_rectangular_07",
    "Piscina redonda":                        "pool_round",
    "Superficie piscina":                     "pool_surface_m2",
    "VEGETACION CONTAMINANTE":                "vegetation_contamination",
    "Volumen piscina":                        "pool_volume_m3",
    "Zona playa césped":                      "deck_grass",
    "Zona PLAYA mixta":                       "deck_mixed",
    "Zona PLAYA pavimentada":                 "deck_paved",
    "EMPLEADO.1":                             "ops_technician",
    "FECHA.1":                                "ops_date",
    "Horas dosificacion PH":                  "ph_dosing_hours",
    "Horas filtracion diarias":               "daily_filtration_hours",
    "Porcentaje dosificación PH":             "ph_dosing_pct",
    "Tiempo lavado /enjuague filtro":         "filter_wash_rinse_time",
    "Horas dosificación hipo":                "hypochlorite_dosing_hours",
    "Porcentaje dosificación hipoclorito":    "hypochlorite_dosing_pct",
    "Temperatura agua":                       "water_temperature",
    "EMPLEADO.2":                             "prod_technician",
    "FECHA.2":                                "prod_date",
    "T-500 (GRUPO QP)":                       "prod_t500_qp",
    "ALBORAL TABLETAS 250 GRS RF. 201710":    "prod_alboral_tablets_250g",
    "FLOVIL PASTILLAS":                       "prod_flovil_tablets",
    "HIPO GARRAFAS 20KG.":                    "prod_hypo_jugs_20kg",
    "HIPO GR CHLORYTE":                       "prod_hypo_gr_chloryte",
    "HIPO GRANULADO XAKA":                    "prod_hypo_granular_xaka",
    "HIPO STICKS BAYROL":                     "prod_hypo_sticks_bayrol",
    "HIPO TAB. RITOCAL":                      "prod_hypo_tab_ritocal",
    "HIPO TABLETAS 200Gr. QP":               "prod_hypo_tablets_200g_qp",
    "HIPO TABLETAS XAKA":                     "prod_hypo_tablets_xaka",
    "PH MINUS GRANULADO 6kg":                "prod_ph_minus_granular_6kg",
    "PH MINUS LIQUIDO 13.5 KG":              "prod_ph_minus_liquid_13_5kg",
    "PH MINUS LIQUIDO 27 KG.":              "prod_ph_minus_liquid_27kg",
    "PROTECT & SHINE":                        "prod_protect_shine",
    "SG XAKA (AGONET GR90)":                 "prod_sg_xaka_agonet",
    "SUPERKLAR":                              "prod_superklar",
}


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration for one training run.

    All on-disk paths are resolved relative to `project_root` (the repository
    root) so the same code runs from a developer laptop, the Docker container
    and the test-runner without monkey-patching globals.
    """

    # --- Paths (resolved relative to project_root at materialise time) -------
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    raw_excel: str          = "data/Merged_2023_2026.xlsx"
    chlorine_pump_list: str = "data/Listado_piscinas_bomba_cloro.xlsx"
    weather_csv: str        = "data/weather_alicante_2023_2026.csv"
    output_dir: str         = "outputs"
    models_dir: str         = "models"

    # --- Weather (Alicante, per client brief) -------------------------------
    alicante_lat: float = 38.3452
    alicante_lon: float = -0.4815
    alicante_tz: str    = "Europe/Madrid"

    # --- Weather features kept from the Open-Meteo response -----------------
    weather_cols_keep: tuple = (
        "date",
        "temperature_2m_max",
        "temperature_2m_mean",
        "uv_index_max",
        "uv_index_clear_sky_max",
        "shortwave_radiation_sum",
        "sunshine_duration",
        "precipitation_sum",
        "wind_speed_10m_max",
        "et0_fao_evapotranspiration",
        "weather_code",
    )

    # --- XGBoost hyper-parameters (shared by the three regressors) -----------
    xgb_params: dict = field(default_factory=lambda: dict(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    ))
    early_stopping_rounds: int = 50

    # --- Dosing optimiser grid ----------------------------------------------
    dosing_pct_step: int   = 5
    dosing_hours_step: float = 1.0

    # --- Temporal split quantile (0.8 = 80th-percentile date cutoff) --------
    temporal_split_quantile: float = 0.8

    # --- Null-rate tolerance for feature dropping ---------------------------
    feature_null_drop_threshold: float = 0.5

    # --- Merge tolerance (days) for asof joins of ops / products ------------
    merge_tolerance_days: int = 14

    # --- Promotion gate (Phase 5 retrain) -----------------------------------
    # A new run is promoted only if, on the same holdout, each primary metric
    # (Cl MAE, pH MAE) is no worse than the active run's by more than `tolerance`.
    promotion_tolerance_cl:  float = 0.02   # mg/L  slack
    promotion_tolerance_ph:  float = 0.005  # pH    slack

    # --- Convenience absolute paths (computed, not stored) ------------------
    @property
    def raw_excel_path(self) -> Path:        return self.project_root / self.raw_excel
    @property
    def chlorine_pump_list_path(self) -> Path: return self.project_root / self.chlorine_pump_list
    @property
    def weather_csv_path(self) -> Path:      return self.project_root / self.weather_csv
    @property
    def output_dir_path(self) -> Path:       return self.project_root / self.output_dir
    @property
    def models_dir_path(self) -> Path:       return self.project_root / self.models_dir

    def ensure_dirs(self) -> None:
        """Create output/models directories on demand (idempotent)."""
        self.output_dir_path.mkdir(parents=True, exist_ok=True)
        self.models_dir_path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Module-level singleton for backward-compat with code that imported the
# constants directly from pipeline_v6.py before the refactor.
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = PipelineConfig()


def reg_thresholds() -> dict:
    """Return the regulatory thresholds as a serialisable dict (mirrors the
    inference_config_v6.json `regulatory_thresholds` block)."""
    return {
        "chlorine_min":       REG_CHLORINE_MIN,
        "chlorine_ideal_max": REG_CHLORINE_IDEAL_MAX,
        "chlorine_close":     REG_CHLORINE_CLOSE,
        "ph_min":             REG_PH_MIN,
        "ph_max":             REG_PH_MAX,
        "turbidity_max":      REG_TURBIDITY_MAX,
    }


def client_targets() -> dict:
    return {
        "chlorine_min":  CLIENT_CL_TARGET_MIN,
        "chlorine_max":  CLIENT_CL_TARGET_MAX,
        "chlorine_ideal": CLIENT_CL_IDEAL,
        "ph_ideal":      PH_IDEAL,
    }