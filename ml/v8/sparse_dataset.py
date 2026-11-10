"""
V8 Sparse Dataset Builder.

Loads the daily-reconstructed timeseries CSV produced by v7 and strips out
all imputed rows, keeping only actual technician visits
(is_observed_measurement_day == 1).

Also provides:
  - compute_gap_stats()       — per-pool gap statistics
  - classify_pools_by_gap()  — gap-bucket classification
  - generate_gap_report()    — full fleet gap report DataFrame (saved to CSV)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ml.v8.config_v8 import (
    V8Config,
    DEFAULT_V8_CONFIG,
    SUMMER_MONTHS,
    GAP_BUCKET_SAME_DAY,
    GAP_BUCKET_SHORT,
    GAP_BUCKET_WEEKLY,
    GAP_BUCKET_BIWEEKLY,
    GAP_BUCKET_LONG,
)

log = logging.getLogger("ml.v8.sparse_dataset")


# ---------------------------------------------------------------------------
# Column aliases — the timeseries CSV uses these names
# ---------------------------------------------------------------------------
COL_POOL       = "pool_clean"
COL_DATE       = "date"
COL_OBS_FLAG   = "is_observed_measurement_day"
COL_IMPUTE_MTD = "imputation_method"
COL_CL_POST    = "free_chlorine_post_ppm"
COL_CL_PRE     = "free_chlorine_pre_ppm"
COL_PH_POST    = "ph_post"
COL_PH_PRE     = "ph_pre"
COL_TURB_POST  = "turbidity_post"
COL_TURB_PRE   = "turbidity_pre"


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------

def load_sparse_visits(cfg: V8Config = DEFAULT_V8_CONFIG) -> pd.DataFrame:
    """
    Load the daily timeseries and return only real observed visits.

    Tries the daily timeseries CSV first; falls back to raw Excel if not found.
    Returns a DataFrame with one row per technician visit, sorted by
    (pool_clean, date).
    """
    ts_path = cfg.project_root / "data" / "pool_daily_reconstructed_timeseries.csv"
    if not ts_path.exists():
        raise FileNotFoundError(
            f"Daily timeseries CSV not found at {ts_path}. "
            "Run the v7 reconstruction pipeline first, or point cfg.daily_timeseries_csv "
            "at the correct file."
        )

    log.info("Loading daily timeseries from %s", ts_path)
    df = pd.read_csv(ts_path, low_memory=False)
    total_rows = len(df)
    log.info("  Total rows in timeseries: %d", total_rows)

    # --- Primary sparse filter ---
    if COL_OBS_FLAG in df.columns:
        df_sparse = df[df[COL_OBS_FLAG] == 1].copy()
    else:
        log.warning("Column '%s' not found — falling back to imputation_method filter", COL_OBS_FLAG)
        df_sparse = df[df[COL_IMPUTE_MTD] == "ground_truth_observation"].copy()

    # --- Defensive secondary strip of any remaining imputed rows ---
    if COL_IMPUTE_MTD in df_sparse.columns:
        before = len(df_sparse)
        df_sparse = df_sparse[
            ~df_sparse[COL_IMPUTE_MTD].isin(cfg.imputed_methods)
        ].copy()
        dropped = before - len(df_sparse)
        if dropped:
            log.info("  Secondary imputation strip removed %d extra rows", dropped)

    df_sparse[COL_DATE] = pd.to_datetime(df_sparse[COL_DATE])
    df_sparse = (
        df_sparse
        .sort_values([COL_POOL, COL_DATE])
        .reset_index(drop=True)
    )

    n_pools  = df_sparse[COL_POOL].nunique()
    n_visits = len(df_sparse)
    pct      = 100.0 * n_visits / total_rows
    log.info(
        "  Sparse visits: %d rows (%d pools, %.1f%% of timeseries)",
        n_visits, n_pools, pct,
    )
    return df_sparse


# ---------------------------------------------------------------------------
# GAP COMPUTATION
# ---------------------------------------------------------------------------

def _gap_category(days: float) -> int:
    """Map a gap in days to an ordinal bucket 1–5."""
    if pd.isna(days):
        return 0   # first visit
    d = int(days)
    if d <= GAP_BUCKET_SAME_DAY:
        return 1
    if d <= GAP_BUCKET_SHORT[1]:
        return 2
    if d <= GAP_BUCKET_WEEKLY[1]:
        return 3
    if d <= GAP_BUCKET_BIWEEKLY[1]:
        return 4
    return 5


def add_gap_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add gap_since (days since last visit) and gap_to (days to next visit)
    to a sparse-visit DataFrame.  Also adds gap_category and is_summer.
    """
    df = df.copy()
    df = df.sort_values([COL_POOL, COL_DATE]).reset_index(drop=True)

    df["gap_since"] = (
        df.groupby(COL_POOL)[COL_DATE]
          .diff()
          .dt.days
    )
    df["gap_to"] = (
        df.groupby(COL_POOL)[COL_DATE]
          .diff(-1)      # negative diff = look forward
          .dt.days
          .abs()
    )
    df["gap_category"]  = df["gap_since"].apply(_gap_category)
    df["is_summer"]     = df[COL_DATE].dt.month.isin(SUMMER_MONTHS).astype(int)
    df["visit_month"]   = df[COL_DATE].dt.month
    df["visit_season"]  = df["visit_month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
         3: "Spring", 4: "Spring", 5: "Spring",
         6: "Summer", 7: "Summer", 8: "Summer", 9: "Summer",
        10: "Autumn", 11: "Autumn",
    })
    return df


# ---------------------------------------------------------------------------
# GAP STATISTICS
# ---------------------------------------------------------------------------

def compute_gap_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a per-pool gap statistics DataFrame.

    Columns:
        pool_clean, n_visits, min_gap, max_gap, mean_gap, median_gap,
        p25_gap, p75_gap, pct_gaps_gt7, pct_gaps_gt14, pct_gaps_gt30,
        gap_bucket, summer_visit_pct, date_start, date_end, span_days
    """
    if "gap_since" not in df.columns:
        df = add_gap_columns(df)

    records = []
    for pool, grp in df.groupby(COL_POOL):
        gaps = grp["gap_since"].dropna()
        n_visits = len(grp)
        date_start = grp[COL_DATE].min()
        date_end   = grp[COL_DATE].max()
        span_days  = (date_end - date_start).days

        if len(gaps) == 0:
            records.append({
                "pool_clean": pool, "n_visits": n_visits,
                "min_gap": np.nan, "max_gap": np.nan,
                "mean_gap": np.nan, "median_gap": np.nan,
                "p25_gap": np.nan, "p75_gap": np.nan,
                "pct_gaps_gt7": np.nan, "pct_gaps_gt14": np.nan,
                "pct_gaps_gt30": np.nan,
                "gap_bucket": "insufficient",
                "summer_visit_pct": np.nan,
                "date_start": date_start, "date_end": date_end,
                "span_days": span_days,
            })
            continue

        max_gap   = gaps.max()
        mean_gap  = gaps.mean()
        # Classify pool by its dominant visit cadence
        if max_gap <= 3:
            bucket = "near_daily"
        elif max_gap <= 7:
            bucket = "weekly"
        elif max_gap <= 14:
            bucket = "fortnightly"
        elif max_gap <= 30:
            bucket = "monthly"
        else:
            bucket = "long_gaps"

        summer_pct = grp["is_summer"].mean() if "is_summer" in grp.columns else np.nan

        records.append({
            "pool_clean":       pool,
            "n_visits":         n_visits,
            "min_gap":          gaps.min(),
            "max_gap":          max_gap,
            "mean_gap":         round(mean_gap, 2),
            "median_gap":       gaps.median(),
            "p25_gap":          gaps.quantile(0.25),
            "p75_gap":          gaps.quantile(0.75),
            "pct_gaps_gt7":     round(100.0 * (gaps > 7).mean(), 1),
            "pct_gaps_gt14":    round(100.0 * (gaps > 14).mean(), 1),
            "pct_gaps_gt30":    round(100.0 * (gaps > 30).mean(), 1),
            "gap_bucket":       bucket,
            "summer_visit_pct": round(100.0 * summer_pct, 1),
            "date_start":       date_start,
            "date_end":         date_end,
            "span_days":        span_days,
        })

    stats = pd.DataFrame(records).sort_values("pool_clean").reset_index(drop=True)
    return stats


def classify_pools_by_gap(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Return pools grouped into gap-cadence buckets.

    Buckets:
        near_daily   — max_gap ≤ 3 d
        weekly       — max_gap 4–7 d
        fortnightly  — max_gap 8–14 d
        monthly      — max_gap 15–30 d
        long_gaps    — max_gap > 30 d
    """
    stats = compute_gap_stats(df)
    result: Dict[str, List[str]] = {
        "near_daily":   [],
        "weekly":       [],
        "fortnightly":  [],
        "monthly":      [],
        "long_gaps":    [],
    }
    for _, row in stats.iterrows():
        bucket = row["gap_bucket"]
        if bucket in result:
            result[bucket].append(row["pool_clean"])
    return result


# ---------------------------------------------------------------------------
# FLEET-LEVEL GAP REPORT
# ---------------------------------------------------------------------------

def _seasonal_gap_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean/median gap per season across the whole fleet."""
    if "gap_since" not in df.columns:
        df = add_gap_columns(df)
    return (
        df.dropna(subset=["gap_since"])
          .groupby("visit_season")["gap_since"]
          .agg(mean_gap="mean", median_gap="median", n_gaps="count")
          .round(2)
          .reset_index()
    )


def _gap_distribution_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Fleet-wide gap bucket distribution."""
    if "gap_since" not in df.columns:
        df = add_gap_columns(df)
    gaps = df["gap_since"].dropna()
    n = len(gaps)
    buckets = [
        ("<=3d",   (gaps <= 3).sum()),
        ("4-7d",   gaps.between(4, 7).sum()),
        ("8-14d",  gaps.between(8, 14).sum()),
        ("15-30d", gaps.between(15, 30).sum()),
        (">30d",   (gaps > 30).sum()),
    ]
    rows = [{"gap_bucket": lbl, "count": int(cnt), "pct": round(100.0 * cnt / n, 1)}
            for lbl, cnt in buckets]
    return pd.DataFrame(rows)


def generate_gap_report(
    df: pd.DataFrame,
    cfg: V8Config = DEFAULT_V8_CONFIG,
    save_csv: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate and optionally save the full gap analysis report.

    Returns:
        (per_pool_stats, seasonal_summary, gap_distribution)
    """
    if "gap_since" not in df.columns:
        df = add_gap_columns(df)

    per_pool    = compute_gap_stats(df)
    seasonal    = _seasonal_gap_summary(df)
    dist        = _gap_distribution_summary(df)

    # --- Console summary ---
    log.info("=== V8 GAP REPORT ===")
    log.info("Fleet: %d pools, %d total visits", per_pool["pool_clean"].nunique(), len(df))
    log.info("Pools with 100%% daily coverage (max_gap<=1):  %d", (per_pool["max_gap"] <= 1).sum())
    log.info("Pools with max_gap > 7 days :  %d", (per_pool["max_gap"] > 7).sum())
    log.info("Pools with max_gap > 14 days:  %d", (per_pool["max_gap"] > 14).sum())
    log.info("Pools with max_gap > 30 days:  %d", (per_pool["max_gap"] > 30).sum())
    log.info("Pools with max_gap > 60 days:  %d", (per_pool["max_gap"] > 60).sum())
    log.info("Gap bucket distribution:\n%s", dist.to_string(index=False))
    log.info("Seasonal gaps:\n%s", seasonal.to_string(index=False))

    # --- Save ---
    if save_csv:
        cfg.ensure_dirs()
        per_pool.to_csv(cfg.gap_report_path, index=False)
        log.info("Gap report saved → %s", cfg.gap_report_path)

    return per_pool, seasonal, dist


# ---------------------------------------------------------------------------
# Convenience: pretty-print summary to stdout
# ---------------------------------------------------------------------------

def print_gap_summary(cfg: V8Config = DEFAULT_V8_CONFIG) -> None:
    """Load data, build gap report, print to stdout. Useful for quick CLI check."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    df = load_sparse_visits(cfg)
    df = add_gap_columns(df)
    per_pool, seasonal, dist = generate_gap_report(df, cfg, save_csv=True)

    print("\n" + "=" * 60)
    print("POOL GAP ANALYSIS — V8 Sparse Dataset")
    print("=" * 60)
    print(f"Total pools  : {per_pool['pool_clean'].nunique()}")
    print(f"Total visits : {len(df)}")
    print(f"\nPools with ZERO gaps (every day observed)  : "
          f"{(per_pool['max_gap'] <= 1).sum()}")
    print(f"Pools with max gap >  7 days : {(per_pool['max_gap'] > 7).sum()}")
    print(f"Pools with max gap > 14 days : {(per_pool['max_gap'] > 14).sum()}")
    print(f"Pools with max gap > 30 days : {(per_pool['max_gap'] > 30).sum()}")
    print(f"Pools with max gap > 60 days : {(per_pool['max_gap'] > 60).sum()}")
    print(f"\nGap bucket distribution (fleet-wide):")
    print(dist.to_string(index=False))
    print(f"\nSeasonal visit gaps:")
    print(seasonal.to_string(index=False))
    print(f"\nGap bucket classification of pools:")
    buckets = classify_pools_by_gap(df)
    for bkt, pools in buckets.items():
        print(f"  {bkt:15s}: {len(pools):3d} pools")
    print(f"\nPer-pool stats sample (worst 5 by max_gap):")
    print(per_pool.nlargest(5, "max_gap")[
        ["pool_clean", "n_visits", "max_gap", "mean_gap", "median_gap", "gap_bucket"]
    ].to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    print_gap_summary()

