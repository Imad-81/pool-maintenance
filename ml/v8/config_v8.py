"""
V8 Sparse-Visit pipeline configuration.

This config is intentionally self-contained — it does NOT inherit from
ml.config.PipelineConfig so that v8 can evolve independently and cannot
accidentally break the existing v6/v7 pipelines.

Key design decisions vs v7:
  - use_sparse_only=True  → only rows where is_observed_measurement_day==1
  - No physics-kinetic bridging, no imputed rows
  - Target = next OBSERVED visit's post-treatment chemistry values
  - Rows with gap_to_next > max_allowed_gap_for_target are excluded from
    training (the gap is too long to build a reliable target)
  - Summer visits are down-weighted (gap_weight_summer < 1.0) because
    summer accounts for 60.8% of all visits but only ~25% of the calendar
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Regulatory basis: Real Decreto 742/2013, Annexe I  (same as v6/v7)
# ---------------------------------------------------------------------------
REG_CHLORINE_MIN       = 0.5
REG_CHLORINE_CLOSE     = 5.0
REG_PH_MIN             = 7.2
REG_PH_MAX             = 8.0
REG_TURBIDITY_MAX      = 5.0

CLIENT_CL_TARGET_MIN   = 1.0
CLIENT_CL_TARGET_MAX   = 1.5


# ---------------------------------------------------------------------------
# V8 gap bucket boundaries (days)
# ---------------------------------------------------------------------------
GAP_BUCKET_SAME_DAY   = 1          # gap == 1  (back-to-back calendar days)
GAP_BUCKET_SHORT      = (2, 3)     # typical pool-every-2-3-days
GAP_BUCKET_WEEKLY     = (4, 7)     # weekly rounds
GAP_BUCKET_BIWEEKLY   = (8, 14)   # fortnightly
GAP_BUCKET_LONG       = 15         # >= 15 days (closure / winter break)

SUMMER_MONTHS = {6, 7, 8, 9}


@dataclass
class V8Config:
    """All paths and hyper-parameters for the v8 sparse-visit pipeline."""

    # --- Paths (relative to project_root) ----------------------------------
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )
    # Input: the daily reconstructed timeseries used by v7 (we filter it)
    daily_timeseries_csv: str = "data/pool_daily_reconstructed_timeseries.csv"
    # Input: raw Excel (fallback if timeseries CSV unavailable)
    raw_excel: str = "data/Merged_2023_2026.xlsx"
    # Cached weather
    weather_csv: str = "data/weather_alicante_daily.csv"
    # Output model artifacts
    models_dir: str = "models/v8-sparse-visit"
    # Output gap analysis report
    gap_report_csv: str = "outputs/v8_gap_analysis_report.csv"

    # --- Sparse filtering ---------------------------------------------------
    use_sparse_only: bool = True
    # Observed-day column name in the timeseries CSV
    observed_flag_col: str = "is_observed_measurement_day"
    # Imputed-row method values to strip (defensive — sparse filter is primary)
    imputed_methods: tuple = ("physics_kinetic_bridge",)

    # --- Target building ----------------------------------------------------
    # Visits with more than this many days until the NEXT visit are excluded
    # from the training target (the future is too uncertain over long gaps)
    max_allowed_gap_for_target: int = 60

    # --- Sample weighting ---------------------------------------------------
    # Summer visits are over-represented (~61% of all visits, ~25% of calendar)
    # We down-weight them to avoid a summer-biased model
    gap_weight_summer: float = 0.7      # applied to summer visit rows
    gap_weight_non_summer: float = 1.0

    # --- LightGBM hyper-parameters -----------------------------------------
    lgbm_params: dict = field(default_factory=lambda: {
        "objective":        "regression",
        "metric":           "mae",
        "num_leaves":       63,
        "learning_rate":    0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "min_child_samples": 20,
        "n_estimators":     800,
        "early_stopping_rounds": 50,
        "verbose":          -1,
        "n_jobs":           -1,
        "random_state":     42,
    })

    # --- Training split -----------------------------------------------------
    temporal_split_quantile: float = 0.80  # 80th-percentile date as cutoff
    feature_null_drop_threshold: float = 0.5

    # --- Convenience properties ---------------------------------------------
    @property
    def daily_timeseries_path(self) -> Path:
        return self.project_root / self.daily_timeseries_csv

    @property
    def raw_excel_path(self) -> Path:
        return self.project_root / self.raw_excel

    @property
    def weather_csv_path(self) -> Path:
        return self.project_root / self.weather_csv

    @property
    def models_dir_path(self) -> Path:
        return self.project_root / self.models_dir

    @property
    def gap_report_path(self) -> Path:
        return self.project_root / self.gap_report_csv

    def ensure_dirs(self) -> None:
        """Create output directories on demand (idempotent)."""
        self.models_dir_path.mkdir(parents=True, exist_ok=True)
        self.gap_report_path.parent.mkdir(parents=True, exist_ok=True)

    def __post_init__(self) -> None:
        # Coerce project_root to Path in case a string is passed
        if not isinstance(self.project_root, Path):
            self.project_root = Path(self.project_root)
        if not (0.0 <= self.gap_weight_summer <= 1.0):
            raise ValueError("gap_weight_summer must be in [0, 1]")
        if self.max_allowed_gap_for_target < 1:
            raise ValueError("max_allowed_gap_for_target must be >= 1")


# Module-level default singleton
DEFAULT_V8_CONFIG = V8Config()
