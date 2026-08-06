"""
ML core for the Pool Predictive Maintenance System (V6).

This package contains the pure-Python training and inference logic with no
HTTP/IO coupling, so it can be exercised from a CLI, a test, or a service.

Layout
------
ml.config        : PipelineConfig dataclass (single source of truth for
                   regulatory thresholds, client targets, coordinates,
                   XGBoost hyper-parameters and on-disk paths).
ml.features      : shared feature engineering used by both training and
                   inference — guarantees parity across the two phases.
ml.training      : reproducible training pipeline (CLI + step functions +
                   evaluation + atomic artifact writer).
ml.inference     : live prediction engine (chained multi-day forecast +
                   dosing optimiser) consumed by the FastAPI backend.
"""

from ml.config import PipelineConfig

__all__ = ["PipelineConfig"]