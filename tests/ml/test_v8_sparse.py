"""
V8 Sparse-Visit — isolated test suite.

These tests verify:
  1. Gap report covers all 138 pools
  2. No imputed rows exist in the sparse dataset
  3. Gap features are free of NaN (after first visit per pool)
  4. The dry-run pipeline loads the expected number of visits
  5. Gap category distribution matches the fleet-level analysis

Run with:
    python -m pytest tests/ml/test_v8_sparse.py -v

The existing test suite (test_training.py, test_v7_delta_integration.py, etc.)
is NOT imported or modified here.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on the path (same pattern used by test_training.py)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from ml.v8.config_v8 import V8Config
from ml.v8.sparse_dataset import (
    load_sparse_visits,
    add_gap_columns,
    compute_gap_stats,
    classify_pools_by_gap,
    generate_gap_report,
)
from ml.v8.features_v8 import build_features, all_feature_groups
from ml.v8.train_v8 import run_v8_pipeline


# ---------------------------------------------------------------------------
# Shared fixture — load once for the whole module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg():
    return V8Config()


@pytest.fixture(scope="module")
def sparse_df(cfg):
    """Load sparse visits once per test module (avoids repeated CSV reads)."""
    df = load_sparse_visits(cfg)
    return add_gap_columns(df)


@pytest.fixture(scope="module")
def gap_stats(sparse_df):
    return compute_gap_stats(sparse_df)


# ---------------------------------------------------------------------------
# TEST 1 — Gap report covers all pools
# ---------------------------------------------------------------------------

def test_gap_report_completeness(cfg, sparse_df, gap_stats):
    """All 138 pools must appear in the per-pool gap statistics."""
    n_pools_in_data  = sparse_df["pool_clean"].nunique()
    n_pools_in_stats = len(gap_stats)
    assert n_pools_in_stats == n_pools_in_data, (
        f"Gap report has {n_pools_in_stats} rows but dataset has {n_pools_in_data} pools."
    )
    assert n_pools_in_stats == 138, (
        f"Expected 138 pools, found {n_pools_in_stats}."
    )


# ---------------------------------------------------------------------------
# TEST 2 — No imputed rows in sparse dataset
# ---------------------------------------------------------------------------

def test_no_imputed_rows_in_sparse_dataset(sparse_df):
    """
    The sparse dataset must contain ONLY ground_truth_observation rows —
    zero physics_kinetic_bridge rows.
    """
    if "imputation_method" in sparse_df.columns:
        imputed = (sparse_df["imputation_method"] == "physics_kinetic_bridge").sum()
        assert imputed == 0, (
            f"Found {imputed} imputed rows in sparse dataset — expected 0."
        )
    if "is_observed_measurement_day" in sparse_df.columns:
        non_obs = (sparse_df["is_observed_measurement_day"] != 1).sum()
        assert non_obs == 0, (
            f"Found {non_obs} rows where is_observed_measurement_day != 1."
        )


# ---------------------------------------------------------------------------
# TEST 3 — Gap features have no unexpected NaN
# ---------------------------------------------------------------------------

def test_gap_features_no_nan(cfg, sparse_df):
    """
    After build_features(), gap_log / gap_category / is_long_gap_lag must
    have no NaN values on non-first visits (gap_since is populated).
    """
    df_feat = build_features(sparse_df, cfg)

    # Non-first visits: gap_since is not NaN
    non_first = df_feat[df_feat["gap_since"].notna()]
    assert len(non_first) > 0, "No non-first visits found — something is wrong."

    gap_check_cols = ["gap_log", "gap_category", "is_long_gap_lag"]
    nan_counts = non_first[gap_check_cols].isna().sum()
    for col, count in nan_counts.items():
        assert count == 0, f"Gap feature '{col}' has {count} unexpected NaN values."


# ---------------------------------------------------------------------------
# TEST 4 — Dry-run pipeline loads ~38K visits
# ---------------------------------------------------------------------------

def test_sparse_dry_run(cfg):
    """
    Dry-run should complete quickly and report ~38 285 visits across 138 pools.
    We allow ±500 visits of tolerance for data updates.
    """
    result = run_v8_pipeline(cfg, run_id="test-dry-run-v8", dry_run=True)
    assert result["dry_run"] is True, "dry_run flag not set in result"
    assert result["pools"] == 138, f"Expected 138 pools, got {result['pools']}"
    assert 37_000 <= result["visits"] <= 40_000, (
        f"Expected ~38 285 visits, got {result['visits']}."
    )


# ---------------------------------------------------------------------------
# TEST 5 — Gap category distribution matches fleet analysis
# ---------------------------------------------------------------------------

def test_gap_categories_distribution(sparse_df):
    """
    Verify the fleet-wide gap distribution matches the figures from the
    plan analysis (within ±2% tolerance):
      <=3 days : ~59.6%
      4-7 days : ~28.2%
      8-14 days: ~11.4%
    """
    gaps = sparse_df["gap_since"].dropna()
    n = len(gaps)

    pct_le3  = 100.0 * (gaps <= 3).sum() / n
    pct_4_7  = 100.0 * gaps.between(4, 7).sum() / n
    pct_8_14 = 100.0 * gaps.between(8, 14).sum() / n

    assert abs(pct_le3  - 59.6) < 2.0, f"<=3d bucket: expected ~59.6%, got {pct_le3:.1f}%"
    assert abs(pct_4_7  - 28.2) < 2.0, f"4-7d bucket: expected ~28.2%, got {pct_4_7:.1f}%"
    assert abs(pct_8_14 - 11.4) < 2.0, f"8-14d bucket: expected ~11.4%, got {pct_8_14:.1f}%"


# ---------------------------------------------------------------------------
# TEST 6 — No near-daily pools (confirms all pools have genuine gaps)
# ---------------------------------------------------------------------------

def test_no_near_daily_pools(sparse_df):
    """
    No pool should be classified as 'near_daily' (max_gap ≤ 3).
    Every pool in the fleet has at least one gap > 7 days.
    """
    buckets = classify_pools_by_gap(sparse_df)
    assert buckets["near_daily"] == [], (
        f"Expected 0 near-daily pools, found: {buckets['near_daily']}"
    )
    stats = compute_gap_stats(sparse_df)
    pools_with_gt7 = (stats["max_gap"] > 7).sum()
    assert pools_with_gt7 == 138, (
        f"Expected all 138 pools to have max_gap > 7, got {pools_with_gt7}"
    )


# ---------------------------------------------------------------------------
# TEST 7 — Sample weights are sane
# ---------------------------------------------------------------------------

def test_sample_weights_sane(cfg, sparse_df):
    """Sample weights must be positive, finite, and normalised to mean ≈ 1.0."""
    df_feat = build_features(sparse_df, cfg)
    assert "sample_weight" in df_feat.columns
    sw = df_feat["sample_weight"]
    assert sw.isna().sum() == 0,       "sample_weight has NaN values"
    assert (sw <= 0).sum() == 0,       "sample_weight has non-positive values"
    assert np.isfinite(sw).all(),      "sample_weight has non-finite values"
    assert abs(sw.mean() - 1.0) < 0.1, f"sample_weight mean deviates from 1: {sw.mean():.3f}"


# ---------------------------------------------------------------------------
# TEST 8 — Feature column count is consistent
# ---------------------------------------------------------------------------

def test_feature_column_count(cfg, sparse_df):
    """build_features() must produce exactly the declared feature columns."""
    df_feat = build_features(sparse_df, cfg)
    declared = all_feature_groups()
    missing = [c for c in declared if c not in df_feat.columns]
    assert missing == [], f"Features declared but not built: {missing}"
    assert len(declared) == 46, (
        f"Expected 46 feature columns, got {len(declared)}. "
        "Update this test if the feature set intentionally changes."
    )

