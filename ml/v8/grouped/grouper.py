"""
V8 Grouped Models — Visit-Level Group Assignment.

Groups are defined by the inter-visit gap (gap_since, days since last visit):

    Group 1 (G1) — Short Gap   : gap_since in [1, 6]  days
    Group 2 (G2) — Long Gap    : gap_since in [7, 30] days
    Excluded                   : gap_since > 30 OR first visit (gap_since is NaN)

The same pool appears in both groups depending on the season:
  • Summer visits → almost always G1 (visits every 2–3 days)
  • Off-season visits → mostly G2 (weekly/fortnightly maintenance)

This module provides:
    assign_groups(df)        — adds `group` column
    group_summary(df)        — per-group statistics
    assert_group_counts(df)  — validation guard
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("ml.v8.grouped.grouper")

# ---------------------------------------------------------------------------
# Constants — group boundary (days)
# ---------------------------------------------------------------------------
G1_MAX_GAP   = 6    # gap_since <= G1_MAX_GAP  → Group 1
G2_MIN_GAP   = 7    # gap_since >= G2_MIN_GAP  → Group 2
G2_MAX_GAP   = 30   # gap_since > G2_MAX_GAP   → excluded

GROUP_LABELS = {
    "G1": f"Short-gap (1–{G1_MAX_GAP} days)",
    "G2": f"Long-gap ({G2_MIN_GAP}–{G2_MAX_GAP} days)",
    "excluded": f"Excluded (>30 days or first visit)",
}

SUMMER_MONTHS = {6, 7, 8, 9}

# ---------------------------------------------------------------------------
# Core: assign_groups
# ---------------------------------------------------------------------------

def assign_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a `group` column to a sparse-visit DataFrame that already has
    `gap_since` (from ml.v8.sparse_dataset.add_gap_columns).

    Returns a copy of df with:
        group  — "G1", "G2", or "excluded"
    """
    if "gap_since" not in df.columns:
        raise ValueError(
            "DataFrame must have a 'gap_since' column. "
            "Call ml.v8.sparse_dataset.add_gap_columns(df) first."
        )

    df = df.copy()

    conditions = [
        df["gap_since"].isna(),                               # first visit
        df["gap_since"] > G2_MAX_GAP,                         # very long gap
        df["gap_since"] <= G1_MAX_GAP,                        # short gap → G1
        df["gap_since"].between(G2_MIN_GAP, G2_MAX_GAP),      # long gap → G2
    ]
    choices = ["excluded", "excluded", "G1", "G2"]
    df["group"] = np.select(conditions, choices, default="excluded")

    n_g1  = (df["group"] == "G1").sum()
    n_g2  = (df["group"] == "G2").sum()
    n_exc = (df["group"] == "excluded").sum()
    total = len(df)

    log.info(
        "assign_groups: G1=%d (%.1f%%)  G2=%d (%.1f%%)  excluded=%d (%.1f%%)",
        n_g1,  100 * n_g1 / total,
        n_g2,  100 * n_g2 / total,
        n_exc, 100 * n_exc / total,
    )
    return df


# ---------------------------------------------------------------------------
# Validation: assert_group_counts
# ---------------------------------------------------------------------------

def assert_group_counts(
    df: pd.DataFrame,
    min_g1: int = 25_000,
    min_g2: int = 8_000,
) -> None:
    """
    Raise AssertionError if either group is too small to be useful.
    Default thresholds are conservative — real data has ~28K G1 and ~10K G2.
    """
    if "group" not in df.columns:
        df = assign_groups(df)

    n_g1 = (df["group"] == "G1").sum()
    n_g2 = (df["group"] == "G2").sum()

    assert n_g1 >= min_g1, (
        f"Group 1 too small: {n_g1} < {min_g1}. "
        "Check that gap_since is computed correctly."
    )
    assert n_g2 >= min_g2, (
        f"Group 2 too small: {n_g2} < {min_g2}. "
        "Check that gap_since is computed correctly and covers off-season visits."
    )
    # Groups must be mutually exclusive
    overlap = df[df["group"].isin(["G1", "G2"])].groupby("group")["gap_since"].agg(["min", "max"])
    if "G1" in overlap.index and "G2" in overlap.index:
        g1_max = overlap.loc["G1", "max"]
        g2_min = overlap.loc["G2", "min"]
        assert g1_max < g2_min, (
            f"Groups overlap! G1 max gap = {g1_max}, G2 min gap = {g2_min}"
        )


# ---------------------------------------------------------------------------
# Statistics: group_summary
# ---------------------------------------------------------------------------

def group_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a per-group summary DataFrame with chemistry stats, seasonal
    breakdown, and gap statistics.
    """
    if "group" not in df.columns:
        df = assign_groups(df)

    rows = []
    for grp in ["G1", "G2"]:
        sub = df[df["group"] == grp]
        if len(sub) == 0:
            continue

        gaps = sub["gap_since"].dropna()

        # seasonal composition
        if "visit_season" in sub.columns:
            season_col = "visit_season"
        elif "date" in sub.columns:
            month = pd.to_datetime(sub["date"]).dt.month
            season_map = {
                12: "Winter", 1: "Winter", 2: "Winter",
                 3: "Spring", 4: "Spring", 5: "Spring",
                 6: "Summer", 7: "Summer", 8: "Summer", 9: "Summer",
                10: "Autumn", 11: "Autumn",
            }
            sub = sub.copy()
            sub["_season"] = month.map(season_map)
            season_col = "_season"
        else:
            season_col = None

        summer_pct = (
            100.0 * (sub[season_col] == "Summer").mean()
            if season_col else np.nan
        )

        cl_col   = next((c for c in ["cl_post", "free_chlorine_post_ppm"] if c in sub.columns), None)
        ph_col   = next((c for c in ["ph_post"] if c in sub.columns), None)
        turb_col = next((c for c in ["turb_post", "turbidity_post"] if c in sub.columns), None)

        rows.append({
            "group":              grp,
            "label":              GROUP_LABELS[grp],
            "n_visits":           len(sub),
            "n_pools":            sub["pool_clean"].nunique() if "pool_clean" in sub.columns else np.nan,
            "gap_mean":           round(gaps.mean(), 2),
            "gap_median":         gaps.median(),
            "gap_std":            round(gaps.std(), 2),
            "gap_min":            gaps.min(),
            "gap_max":            gaps.max(),
            "summer_pct":         round(summer_pct, 1),
            "cl_mean":            round(sub[cl_col].mean(), 3)   if cl_col   else np.nan,
            "cl_std":             round(sub[cl_col].std(), 3)    if cl_col   else np.nan,
            "ph_mean":            round(sub[ph_col].mean(), 3)   if ph_col   else np.nan,
            "ph_std":             round(sub[ph_col].std(), 3)    if ph_col   else np.nan,
            "turb_mean":          round(sub[turb_col].mean(), 4) if turb_col else np.nan,
            "turb_std":           round(sub[turb_col].std(), 4)  if turb_col else np.nan,
        })

    return pd.DataFrame(rows).set_index("group")


def seasonal_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    """Return a season × group crosstab of visit counts."""
    if "group" not in df.columns:
        df = assign_groups(df)
    if "visit_season" not in df.columns and "date" in df.columns:
        month = pd.to_datetime(df["date"]).dt.month
        season_map = {
            12: "Winter", 1: "Winter", 2: "Winter",
             3: "Spring", 4: "Spring", 5: "Spring",
             6: "Summer", 7: "Summer", 8: "Summer", 9: "Summer",
            10: "Autumn", 11: "Autumn",
        }
        df = df.copy()
        df["visit_season"] = month.map(season_map)

    active = df[df["group"].isin(["G1", "G2"])]
    return pd.crosstab(active["visit_season"], active["group"])


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def print_group_summary(df: pd.DataFrame) -> None:
    """Pretty-print group summary to stdout."""
    if "group" not in df.columns:
        df = assign_groups(df)

    summary = group_summary(df)
    ct = seasonal_crosstab(df)

    sep = "=" * 60
    print(f"\n{sep}")
    print("V8 GROUPED MODEL — VISIT GROUP SUMMARY")
    print(sep)
    print(f"\n{'Field':<25} {'Group 1 (Short)':<20} {'Group 2 (Long)'}")
    print("-" * 60)
    for field in ["n_visits", "n_pools", "gap_mean", "gap_median", "gap_std",
                  "gap_min", "gap_max", "summer_pct",
                  "cl_mean", "cl_std", "ph_mean", "ph_std", "turb_mean", "turb_std"]:
        g1 = summary.loc["G1", field] if "G1" in summary.index else "—"
        g2 = summary.loc["G2", field] if "G2" in summary.index else "—"
        print(f"  {field:<23} {str(g1):<20} {g2}")

    print(f"\nSeasonal crosstab:")
    print(ct.to_string())
    print(sep)

