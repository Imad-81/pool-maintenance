"""Horizon + uncertainty escalation helper for chained forecasts.

The V6 pipeline is trained to predict the *next calendar day*. Chaining beyond
that compounds prediction error — each step feeds the previous predicted
state back in as if it were ground truth, so crystalised drift accumulates.

We expose `default_horizon_days = 2` (today + tomorrow) as the headline UI
value, and `max_horizon_days = 7` as the hard cap. Forecasts past tomorrow
carry a `warning_band` (linearly widening) so the UI can render an explicit
"higher uncertainty past tomorrow" disclosure.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_HORIZON_DAYS = 2   # today + tomorrow (matches inference.py)
MAX_HORIZON_DAYS     = 7   # hard cap for the optional "show more days" toggle
UNCERTAINTY_RAMP_DAY = 2   # from day index 2 onwards (i.e. day after tomorrow)

# Per-step empirical error floors — derived from the V6 test-set MAE.
# Used to grow a forecast warning band one chaining step at a time.
STEP_ERROR_CL   = 0.20   # mg/L per chained step
STEP_ERROR_PH   = 0.034   # pH units per chained step
STEP_ERROR_TURB = 0.04    # NTU per chained step


@dataclass(frozen=True)
class UncertaintyBand:
    day_offset: int          # 1 = tomorrow, 2 = day after tomorrow, ...
    cl_low:  float
    cl_high: float
    ph_low:  float
    ph_high: float
    turb_low: float
    turb_high: float


def warning_band(day_offset: int, pred_cl: float, pred_ph: float, pred_turb: float) -> UncertaintyBand:
    """Compute the ± warning band for a chained forecast at `day_offset` from
    today. Bands are zero-width for today/tomorrow (interpolation target is
    well-defined) and grow linearly past day index 2 — surfaced as the explicit
    "chained — higher uncertainty past tomorrow" UI caveat."""
    if day_offset <= 1:
        n = 0.0
    else:
        n = float(day_offset - 1)
    return UncertaintyBand(
        day_offset=day_offset,
        cl_low=pred_cl  - n * STEP_ERROR_CL,   cl_high=pred_cl  + n * STEP_ERROR_CL,
        ph_low=pred_ph  - n * STEP_ERROR_PH,   ph_high=pred_ph  + n * STEP_ERROR_PH,
        turb_low=pred_turb - n * STEP_ERROR_TURB, turb_high=pred_turb + n * STEP_ERROR_TURB,
    )


def clamp_horizon(days: int | None) -> int:
    """Clamp an invoker-supplied horizon into [1, MAX_HORIZON_DAYS].
    None -> DEFAULT_HORIZON_DAYS."""
    if days is None:
        return DEFAULT_HORIZON_DAYS
    if days < 1:
        return 1
    if days > MAX_HORIZON_DAYS:
        return MAX_HORIZON_DAYS
    return days