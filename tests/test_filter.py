"""
Minimal sanity tests for the filter core — not the full synthetic-recovery
and identifiability suite described in plans/KalmanScale_v1.md Section 3,
just enough to catch obvious breakage before the MVP goes live.
"""

from datetime import date, timedelta

import numpy as np
import pytest

from kalmanscale.filter import FilterParams, run_filter

np.random.seed(0)


def _dates(n, start=date(2026, 1, 1)):
    return [start + timedelta(days=i) for i in range(n)]


def test_recovers_known_trend_and_bias_with_varying_calorie_balance():
    # Generate data from the filter's own generative model (see
    # plans/KalmanScale_v1.md Section 3): true_beta is *residual* drift not
    # explained by tracked calories (e.g. metabolic adaptation), separate
    # from true_b, a fixed tracking bias. I-E must vary day to day for the
    # two to be separable at all — see the identifiability note there.
    n = 150
    true_beta = -0.02   # lb/day of unexplained drift
    true_b = 80.0        # kcal/day net tracking bias
    x0 = 200.0
    dates = _dates(n)

    rng = np.random.default_rng(1)
    cal_out = 2800 + rng.normal(0, 250, n)
    cal_in = 2800 + rng.normal(0, 250, n)

    x = x0
    weights = []
    for i in range(n):
        x += true_beta + (cal_in[i] - cal_out[i] - true_b) / 3500.0 + rng.normal(0, 0.05)
        weights.append(x + rng.normal(0, 0.3))

    entries = [
        {"date": dates[i], "weight": weights[i], "cal_in": cal_in[i], "cal_out": cal_out[i]}
        for i in range(n)
    ]

    results = run_filter(entries)
    final = results[-1]

    # beta and b are NOT individually identifiable here (constant-mean I-E
    # noise gives the filter no way to separate a constant beta pull from a
    # constant b pull — this reproduces the identifiability note in
    # plans/KalmanScale_v1.md Section 3 empirically). What *is* identifiable
    # is their combined effect on daily weight change.
    true_combined = true_beta - true_b / 3500.0
    final_combined = final["beta"] - final["b"] / 3500.0
    assert final_combined == pytest.approx(true_combined, abs=0.01)


def test_missing_calorie_data_does_not_crash():
    entries = [
        {"date": date(2026, 1, 1), "weight": 200.0, "cal_in": None, "cal_out": None},
        {"date": date(2026, 1, 2), "weight": 199.5, "cal_in": 2200.0, "cal_out": None},
        {"date": date(2026, 1, 3), "weight": 199.8, "cal_in": None, "cal_out": 2600.0},
        {"date": date(2026, 1, 4), "weight": 199.2, "cal_in": 2100.0, "cal_out": 2700.0},
    ]
    results = run_filter(entries)
    assert len(results) == 4
    assert all(np.isfinite(r["x"]) for r in results)


def test_multi_day_gap_decays_e_and_does_not_crash():
    entries = [
        {"date": date(2026, 1, 1), "weight": 200.0, "cal_in": 2200.0, "cal_out": 2600.0},
        {"date": date(2026, 1, 8), "weight": 198.0, "cal_in": 2100.0, "cal_out": 2700.0},
    ]
    results = run_filter(entries)
    assert len(results) == 2
    assert np.isfinite(results[-1]["x"])


def test_single_data_point_just_initializes():
    entries = [{"date": date(2026, 1, 1), "weight": 200.0, "cal_in": None, "cal_out": None}]
    results = run_filter(entries)
    assert len(results) == 1
    assert results[0]["x"] == pytest.approx(200.0)
    params = FilterParams()
    assert results[0]["se_x"] == pytest.approx(params.p0_x**0.5)
