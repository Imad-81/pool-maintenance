"""Reproducible training pipeline for the V6 pool predictive models.

Public entry point: `python -m ml.training.train`.
"""

from ml.training.train import run_pipeline, main

__all__ = ["run_pipeline", "main"]