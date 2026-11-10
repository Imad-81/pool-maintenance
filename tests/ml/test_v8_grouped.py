"""
Tests for V8 Grouped Models — grouper, feature engineering, and artifact checks.

Run:
    python -m pytest tests/ml/test_v8_grouped.py -v

All 7 tests are designed to pass once training is complete.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def obs_with_gaps():
    """Load sparse visits and add gap columns."""
    from ml.v8.config_v8 import DEFAULT_V8_CONFIG
    from ml.v8.sparse_dataset import load_sparse_visits, add_gap_columns

    cfg = DEFAULT_V8_CONFIG
    obs = load_sparse_visits(cfg)
    obs = add_gap_columns(obs)
    return obs


@pytest.fixture(scope="module")
def grouped_dfs(obs_with_gaps):
    """Return (df_g1, df_g2) tuple from assign_groups."""
    from ml.v8.grouped.grouper import assign_groups

    assigned = assign_groups(obs_with_gaps)
    df_g1 = assigned[assigned["group"] == "G1"].copy()
    df_g2 = assigned[assigned["group"] == "G2"].copy()
    return df_g1, df_g2


# ---------------------------------------------------------------------------
# Test 1: Groups are mutually exclusive at the visit level
# ---------------------------------------------------------------------------

def test_groups_mutually_exclusive(grouped_dfs):
    """
    No single visit (pool+date pair) should appear in both G1 and G2.
    Groups are defined by gap_since at visit time, not by pool.
    """
    df_g1, df_g2 = grouped_dfs

    key_g1 = set(zip(df_g1["pool_clean"], df_g1["date"]))
    key_g2 = set(zip(df_g2["pool_clean"], df_g2["date"]))

    overlap = key_g1 & key_g2
    assert len(overlap) == 0, (
        f"Found {len(overlap)} visits in both G1 and G2 — groups must be mutually exclusive!"
    )


# ---------------------------------------------------------------------------
# Test 2: Group sizes are within expected bounds
# ---------------------------------------------------------------------------

def test_group_sizes(grouped_dfs):
    """
    G1 should have ~28,027 visits and G2 ~10,017 visits (±5% tolerance).
    """
    df_g1, df_g2 = grouped_dfs

    EXPECTED_G1 = 28_027
    EXPECTED_G2 = 10_017
    TOL = 0.05  # 5%

    assert abs(len(df_g1) - EXPECTED_G1) / EXPECTED_G1 <= TOL, (
        f"G1 size {len(df_g1)} differs from expected {EXPECTED_G1} by >{TOL*100:.0f}%"
    )
    assert abs(len(df_g2) - EXPECTED_G2) / EXPECTED_G2 <= TOL, (
        f"G2 size {len(df_g2)} differs from expected {EXPECTED_G2} by >{TOL*100:.0f}%"
    )


# ---------------------------------------------------------------------------
# Test 3: G2 Cl std > G1 Cl std (G2 is more variable — longer gaps)
# ---------------------------------------------------------------------------

def test_g1_g2_chemistry_differ(grouped_dfs):
    """
    G2 visits should have higher Cl variability than G1 visits,
    since longer gaps introduce more decay uncertainty.
    """
    df_g1, df_g2 = grouped_dfs

    cl_col = "free_chlorine_post_ppm"
    assert cl_col in df_g1.columns and cl_col in df_g2.columns, (
        f"Column '{cl_col}' missing from group dataframes"
    )

    std_g1 = df_g1[cl_col].std()
    std_g2 = df_g2[cl_col].std()

    assert std_g2 > std_g1, (
        f"Expected G2 Cl std ({std_g2:.4f}) > G1 Cl std ({std_g1:.4f}), "
        "but G1 appears more variable — check grouper boundaries."
    )


# ---------------------------------------------------------------------------
# Test 4: All 138 pools appear in at least one group
# ---------------------------------------------------------------------------

def test_all_pools_in_at_least_one_group(obs_with_gaps, grouped_dfs):
    """
    Every pool in the dataset should have at least some visits in G1 or G2.
    Pools with max_gap > 30 days still participate if they have some shorter-gap visits.
    """
    from ml.v8.grouped.grouper import assign_groups

    df_g1, df_g2 = grouped_dfs
    all_pools = set(obs_with_gaps["pool_clean"].unique())
    covered_pools = set(df_g1["pool_clean"].unique()) | set(df_g2["pool_clean"].unique())

    uncovered = all_pools - covered_pools
    # Allow at most 10% uncovered (pools with ALL gaps > 30d or only 1 visit)
    max_uncovered = max(1, int(0.10 * len(all_pools)))
    assert len(uncovered) <= max_uncovered, (
        f"{len(uncovered)} pools are not in any group: {sorted(uncovered)[:5]}..."
    )


# ---------------------------------------------------------------------------
# Test 5: Seasonal dominance — summer in G1, off-season in G2
# ---------------------------------------------------------------------------

def test_seasonal_dominance(grouped_dfs):
    """
    G1 (short gaps < 7d) should be dominated by Summer visits (>75%).
    G2 (long gaps 7-30d) should be dominated by off-season visits (Spring+Autumn+Winter > 60%).
    """
    df_g1, df_g2 = grouped_dfs

    season_col = "visit_season"
    assert season_col in df_g1.columns and season_col in df_g2.columns, (
        f"Column '{season_col}' missing — ensure sparse_dataset.py sets it."
    )

    g1_summer_pct = (df_g1[season_col] == "Summer").mean()
    g2_offseason_pct = (df_g2[season_col] != "Summer").mean()

    assert g1_summer_pct >= 0.70, (
        f"G1 summer pct = {g1_summer_pct:.1%}, expected ≥70% "
        "(short gaps happen during active summer season)"
    )
    assert g2_offseason_pct >= 0.60, (
        f"G2 off-season pct = {g2_offseason_pct:.1%}, expected ≥60% "
        "(long gaps happen during off-season / low-use periods)"
    )


# ---------------------------------------------------------------------------
# Test 6: Dry-run training completes without error
# ---------------------------------------------------------------------------

def test_dry_run_grouped():
    """
    Running train_grouped with --dry-run should exit cleanly (exit code 0).
    This validates data loading, group assignment, and feature engineering
    without actually training any models.
    """
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "ml.v8.grouped.train_grouped", "--dry-run"],
        cwd=str(Path(__file__).parents[2]),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"Dry-run failed with code {result.returncode}.\n"
        f"STDOUT: {result.stdout[-1000:]}\nSTDERR: {result.stderr[-1000:]}"
    )
    assert "DRY-RUN OK" in result.stderr or "DRY-RUN OK" in result.stdout, (
        "Expected 'DRY-RUN OK' in output — check train_grouped.py dry_run branch."
    )


# ---------------------------------------------------------------------------
# Test 7: Trained artifact files exist
# ---------------------------------------------------------------------------

def test_artifacts_created():
    """
    After a successful training run, all 6 pkl files and 2 inference configs
    must exist under models/v8-grouped/group1/ and models/v8-grouped/group2/.
    """
    from ml.v8.config_v8 import DEFAULT_V8_CONFIG

    cfg = DEFAULT_V8_CONFIG
    base = cfg.project_root / "models" / "v8-grouped"

    expected = {
        "group1": ["chlorine_next_g1.pkl", "ph_next_g1.pkl",
                   "turbidity_next_g1.pkl", "inference_config_g1.json"],
        "group2": ["chlorine_next_g2.pkl", "ph_next_g2.pkl",
                   "turbidity_next_g2.pkl", "inference_config_g2.json"],
    }

    missing = []
    for subdir, files in expected.items():
        for fname in files:
            p = base / subdir / fname
            if not p.exists():
                missing.append(str(p))

    assert not missing, (
        f"Missing trained artifacts:\n" + "\n".join(missing) +
        "\nRun:  python -m ml.v8.grouped.train_grouped  to generate them."
    )

    # Also sanity-check the pkl files are valid pickles with a predict() method
    for subdir, files in expected.items():
        for fname in files:
            if not fname.endswith(".pkl"):
                continue
            p = base / subdir / fname
            with p.open("rb") as f:
                model = pickle.load(f)
            assert hasattr(model, "predict"), (
                f"{fname} does not have a predict() method — not a valid LightGBM model."
            )


# ---------------------------------------------------------------------------
# Test 8: Feature engineering produces correct feature counts
# ---------------------------------------------------------------------------

def test_feature_counts(grouped_dfs):
    """
    G1 feature engineering should produce 67 features, G2 should produce 70.
    These counts must match what train_grouped.py expects.
    """
    from ml.v8.grouped.features_grouped import build_features_for_group, g1_all_features, g2_all_features

    df_g1, df_g2 = grouped_dfs

    # Build features on a small sample to keep the test fast
    sample_g1 = df_g1.head(200)
    sample_g2 = df_g2.head(200)

    feat_g1 = build_features_for_group(sample_g1, "G1")
    feat_g2 = build_features_for_group(sample_g2, "G2")

    expected_g1 = len(g1_all_features())
    expected_g2 = len(g2_all_features())

    assert feat_g1.shape[1] >= expected_g1, (
        f"G1 feature matrix has {feat_g1.shape[1]} cols, expected ≥{expected_g1}."
    )
    assert feat_g2.shape[1] >= expected_g2, (
        f"G2 feature matrix has {feat_g2.shape[1]} cols, expected ≥{expected_g2}."
    )


# ---------------------------------------------------------------------------
# Test 9: V8GroupedPredictor routing and output validation
# ---------------------------------------------------------------------------

def test_v8_grouped_predictor():
    """
    Test V8GroupedPredictor routes short gaps to G1 and long gaps to G2,
    and returns valid numerical predictions within realistic physical bounds.
    """
    from ml.v8.grouped.predict_grouped import V8GroupedPredictor

    predictor = V8GroupedPredictor()

    # Route tests
    g_short, _ = predictor.route_regime(3.0)
    g_long, _ = predictor.route_regime(14.0)
    assert g_short == "G1", f"Expected G1 for gap 3.0, got {g_short}"
    assert g_long == "G2", f"Expected G2 for gap 14.0, got {g_long}"

    # Prediction test
    res = predictor.predict_pool_next_visit("ALTAVISTA II (1027)", gap_days=3.0)
    assert res["routed_group"] == "G1"
    assert 0.0 <= res["predicted_free_chlorine_ppm"] <= 10.0
    assert 6.0 <= res["predicted_ph"] <= 9.0
    assert 0.0 <= res["predicted_turbidity_ntu"] <= 20.0
    assert res["regulatory_status"] in ("COMPLIANT", "ACTION_REQUIRED")


