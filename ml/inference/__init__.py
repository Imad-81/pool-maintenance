"""Live prediction engine for the V6 pool predictive maintenance system.

Composed of three pieces:

* `predictor.predict_forward`  — pure chained multi-day forecast function
* `chaining`                   — horizon + uncertainty escalation helper
* `optimiser.Optimiser`        — dosing grid-search aware of live weather

The single public entry point the backend uses is `PredictionService`, which
loads the active model run once and hot-swaps when the scheduler promotes a
new run.
"""

from ml.inference.predictor import predict_forward, PredictionService

__all__ = ["predict_forward", "PredictionService"]