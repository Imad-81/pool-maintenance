"""
Visit recommendation engine for swimming pools.

Determines the optimal next visit date and projected water chemistry values
based on:
1. RD 742/2013 regulatory safety limits (Free Chlorine 0.5–2.0 / 5.0 mg/L, pH 7.2–8.0, Turbidity <= 5.0 NTU).
2. Jesús Santana client optimal target range (Free Chlorine 1.0–1.5 mg/L).
3. Alicante seasonal baseline maintenance cadence (Summer: 2d, May: 4d, Apr/Oct: 6d, Winter: 7d).
"""

from __future__ import annotations

import dataclasses
from datetime import date, datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ml.config import (
    CLIENT_CL_TARGET_MAX,
    CLIENT_CL_TARGET_MIN,
    REG_CHLORINE_CLOSE,
    REG_CHLORINE_MIN,
    REG_PH_MAX,
    REG_PH_MIN,
    REG_TURBIDITY_MAX,
)


def get_seasonal_baseline_days(month: int) -> int:
    """Return median visit interval in days for a given calendar month."""
    if month in (6, 7, 8, 9):
        return 2  # Summer peak bather load
    if month == 5:
        return 4  # Spring pre-season
    if month in (4, 10):
        return 6  # Shoulder months
    return 7  # Winter / low activity (Nov–Mar)


def _format_uncertainty_band(band) -> Optional[dict]:
    if band is None:
        return None
    if dataclasses.is_dataclass(band):
        return dataclasses.asdict(band)
    if hasattr(band, "_asdict"):
        return {k: float(v) for k, v in band._asdict().items()}
    if isinstance(band, dict):
        return {k: float(v) for k, v in band.items() if v is not None}
    return None


def compute_recommended_visit(
    forecast_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    last_visit_date: pd.Timestamp,
) -> dict:
    """Compute when the pool will need a visit and what the values will be on
    that date.

    Returns a structured dictionary with:
      - date: ISO date string (YYYY-MM-DD)
      - day_label: Localized short day label (e.g. "Jue, 21 Ago" / "Today")
      - day_offset_from_today: int (0 for today, 1 for tomorrow, etc.)
      - days_since_last_visit: int
      - urgency: "Immediate" | "Advised" | "Routine"
      - trigger: "regulatory_breach" | "target_decay" | "seasonal_routine"
      - reason: Localized explanatory string
      - predicted_cl: float (mg/L)
      - predicted_ph: float
      - predicted_turb: float (NTU)
      - uncertainty_band: dict with error margins
      - is_breach: bool
    """
    if forecast_df.empty:
        as_of_d = pd.Timestamp(as_of_date).date()
        return {
            "date": str(as_of_d),
            "day_label": as_of_date.strftime("%a %d %b"),
            "day_offset_from_today": 0,
            "days_since_last_visit": int((as_of_date - last_visit_date).days),
            "urgency": "Routine",
            "trigger": "seasonal_routine",
            "reason": "No forecast data available",
            "predicted_cl": None,
            "predicted_ph": None,
            "predicted_turb": None,
            "uncertainty_band": None,
            "is_breach": False,
        }

    as_of_ts = pd.Timestamp(as_of_date).tz_localize(None).normalize()
    last_visit_ts = pd.Timestamp(last_visit_date).tz_localize(None).normalize()
    month = as_of_ts.month
    baseline_interval = get_seasonal_baseline_days(month)

    # Filter forecast rows on or after today
    upcoming = forecast_df[forecast_df["day_offset_from_today"] >= 0].copy()
    if upcoming.empty:
        upcoming = forecast_df.tail(1).copy()

    # -----------------------------------------------------------------------
    # 1. Check for Active / Imminent Regulatory Breach (Today or Tomorrow)
    # -----------------------------------------------------------------------
    today_row = upcoming[upcoming["day_offset_from_today"] == 0]
    tomorrow_row = upcoming[upcoming["day_offset_from_today"] == 1]

    if not today_row.empty and (
        today_row.iloc[0].get("cl_breach", False)
        or today_row.iloc[0].get("ph_breach", False)
        or float(today_row.iloc[0].get("predicted_cl", 1.2)) < REG_CHLORINE_MIN
        or float(today_row.iloc[0].get("predicted_cl", 1.2)) > REG_CHLORINE_CLOSE
        or float(today_row.iloc[0].get("predicted_ph", 7.4)) < REG_PH_MIN
        or float(today_row.iloc[0].get("predicted_ph", 7.4)) > REG_PH_MAX
        or float(today_row.iloc[0].get("predicted_turb", 0.5)) > REG_TURBIDITY_MAX
    ):
        target = today_row.iloc[0]
        cl_val = float(target["predicted_cl"])
        ph_val = float(target["predicted_ph"])
        reasons = []
        if cl_val < REG_CHLORINE_MIN:
            reasons.append(f"Cloro libre ({cl_val:.2f} mg/L) por debajo del mínimo legal (0.50 mg/L)")
        elif cl_val > REG_CHLORINE_CLOSE:
            reasons.append(f"Cloro libre ({cl_val:.2f} mg/L) supera límite crítico (5.00 mg/L)")
        if ph_val < REG_PH_MIN or ph_val > REG_PH_MAX:
            reasons.append(f"pH ({ph_val:.2f}) fuera de rango normativo (7.20–8.00)")

        reason_text = " • ".join(reasons) if reasons else "Infracción normativa inminente detectada hoy"
        return _build_recommendation(
            target=target,
            urgency="Immediate",
            trigger="regulatory_breach",
            reason=reason_text,
            is_breach=True,
        )

    if not tomorrow_row.empty and (
        tomorrow_row.iloc[0].get("cl_breach", False)
        or tomorrow_row.iloc[0].get("ph_breach", False)
    ):
        target = tomorrow_row.iloc[0]
        cl_val = float(target["predicted_cl"])
        ph_val = float(target["predicted_ph"])
        reasons = []
        if cl_val < REG_CHLORINE_MIN:
            reasons.append(f"Cloro libre caerá a {cl_val:.2f} mg/L mañana (< 0.50 mg/L)")
        elif cl_val > REG_CHLORINE_CLOSE:
            reasons.append(f"Cloro libre alcanzará {cl_val:.2f} mg/L mañana (> 5.00 mg/L)")
        if ph_val < REG_PH_MIN or ph_val > REG_PH_MAX:
            reasons.append(f"pH alcanzará {ph_val:.2f} mañana (fuera de 7.20–8.00)")

        reason_text = " • ".join(reasons) if reasons else "Infracción normativa prevista para mañana"
        return _build_recommendation(
            target=target,
            urgency="Immediate",
            trigger="regulatory_breach",
            reason=reason_text,
            is_breach=True,
        )

    # -----------------------------------------------------------------------
    # 2. Check for Future Regulatory Breach across Extended Lookahead (Day +2..+N)
    # -----------------------------------------------------------------------
    future_breaches = upcoming[
        (upcoming["day_offset_from_today"] > 1)
        & (upcoming["cl_breach"] | upcoming["ph_breach"])
    ]
    if not future_breaches.empty:
        breach_row = future_breaches.iloc[0]
        breach_offset = int(breach_row["day_offset_from_today"])
        # Recommend visiting 1 day prior to the breach if possible, so technician arrives before violation
        rec_offset = max(0, breach_offset - 1)
        rec_match = upcoming[upcoming["day_offset_from_today"] == rec_offset]
        target = rec_match.iloc[0] if not rec_match.empty else breach_row

        cl_val = float(breach_row["predicted_cl"])
        ph_val = float(breach_row["predicted_ph"])
        offset_val = int(target["day_offset_from_today"])
        urg = "Immediate" if offset_val <= 1 else "Advised"

        reason_text = (
            f"Visita preventiva: el cloro ({cl_val:.2f} mg/L) o pH ({ph_val:.2f}) "
            f"infringirá la normativa el {breach_row['date']} (en {breach_offset} días)"
        )
        return _build_recommendation(
            target=target,
            urgency=urg,
            trigger="regulatory_breach",
            reason=reason_text,
            is_breach=bool(target.get("cl_breach", False) or target.get("ph_breach", False)),
        )

    # -----------------------------------------------------------------------
    # 3. Check for Client Target Deviation (Free Chlorine < 1.0 or > 1.5 mg/L)
    # -----------------------------------------------------------------------
    target_deviations = upcoming[
        (upcoming["predicted_cl"] < CLIENT_CL_TARGET_MIN)
        | (upcoming["predicted_cl"] > CLIENT_CL_TARGET_MAX)
    ]
    if not target_deviations.empty:
        target = target_deviations.iloc[0]
        cl_val = float(target["predicted_cl"])
        offset_val = int(target["day_offset_from_today"])
        urg = "Immediate" if offset_val <= 0 else "Advised"

        if cl_val < CLIENT_CL_TARGET_MIN:
            reason_text = (
                f"El cloro libre descenderá a {cl_val:.2f} mg/L "
                f"(por debajo del objetivo óptimo {CLIENT_CL_TARGET_MIN:.1f}–{CLIENT_CL_TARGET_MAX:.1f} mg/L)"
            )
        else:
            reason_text = (
                f"El cloro libre alcanzará {cl_val:.2f} mg/L "
                f"(superior al objetivo óptimo {CLIENT_CL_TARGET_MIN:.1f}–{CLIENT_CL_TARGET_MAX:.1f} mg/L)"
            )

        return _build_recommendation(
            target=target,
            urgency=urg,
            trigger="target_decay",
            reason=reason_text,
            is_breach=False,
        )

    # -----------------------------------------------------------------------
    # 4. Fallback: Seasonal Routine Maintenance Interval
    # -----------------------------------------------------------------------
    days_since = int((as_of_ts - last_visit_ts).days)
    days_until_routine = baseline_interval - days_since

    if days_until_routine <= 0:
        # Overdue for seasonal maintenance
        target = upcoming.iloc[0]
        reason_text = (
            f"Mantenimiento rutinario estacional ({baseline_interval}d pauta Alicante) — "
            f"{days_since} días transcurridos desde la última visita"
        )
        return _build_recommendation(
            target=target,
            urgency="Advised" if days_since > baseline_interval + 2 else "Routine",
            trigger="seasonal_routine",
            reason=reason_text,
            is_breach=False,
        )

    # Find the forecast row corresponding to the seasonal target day
    match_routine = upcoming[upcoming["day_offset_from_today"] == days_until_routine]
    if not match_routine.empty:
        target = match_routine.iloc[0]
    else:
        target = upcoming.iloc[-1]

    season_name = "Verano" if month in (6, 7, 8, 9) else "Invierno" if month in (11, 12, 1, 2, 3) else "Temporada media"
    reason_text = (
        f"Mantenimiento rutinario programado ({season_name}: cada {baseline_interval} días; "
        f"agua en rango óptimo 1.0–1.5 mg/L)"
    )

    return _build_recommendation(
        target=target,
        urgency="Routine",
        trigger="seasonal_routine",
        reason=reason_text,
        is_breach=False,
    )


def _build_recommendation(
    target: pd.Series,
    urgency: str,
    trigger: str,
    reason: str,
    is_breach: bool,
) -> dict:
    """Helper to assemble standard recommendation dictionary."""
    d_val = target["date"]
    if isinstance(d_val, (pd.Timestamp, datetime, date)):
        date_str = str(d_val.strftime("%Y-%m-%d") if hasattr(d_val, "strftime") else d_val)
        day_label = d_val.strftime("%a %d %b") if hasattr(d_val, "strftime") else str(d_val)
    else:
        date_str = str(d_val)
        day_label = str(target.get("day", d_val))

    offset = int(target.get("day_offset_from_today", 0))
    days_from_visit = int(target.get("days_from_visit", offset))
    cl_pred = round(float(target["predicted_cl"]), 2)
    ph_pred = round(float(target["predicted_ph"]), 2)
    turb_pred = round(float(target["predicted_turb"]), 2)

    band = _format_uncertainty_band(target.get("uncertainty_band"))

    return {
        "date": date_str,
        "day_label": day_label,
        "day_offset_from_today": offset,
        "days_since_last_visit": days_from_visit,
        "urgency": urgency,
        "trigger": trigger,
        "reason": reason,
        "predicted_cl": cl_pred,
        "predicted_ph": ph_pred,
        "predicted_turb": turb_pred,
        "uncertainty_band": band,
        "is_breach": is_breach,
    }
