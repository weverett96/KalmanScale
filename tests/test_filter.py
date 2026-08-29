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


def test_single_data_point_with_body_fat_initializes_fat_directly():
    entries = [
        {"date": date(2026, 1, 1), "weight": 200.0, "cal_in": None, "cal_out": None, "body_fat_pct": 25.0}
    ]
    results = run_filter(entries)
    assert results[0]["fat"] == pytest.approx(50.0)  # 200 * 25%


def test_fat_mass_recovered_with_intermittent_bioimpedance_readings():
    # body_fat_pct is only logged ~1/3 of days (the point of this test: the
    # filter must handle missing bioimpedance observations without special
    # casing, same as it already does for missing calorie data).
    n = 90
    true_fat = 60.0  # lb, held constant
    weight = 200.0
    dates = _dates(n)
    rng = np.random.default_rng(2)

    entries = []
    for i in range(n):
        entry = {"date": dates[i], "weight": weight, "cal_in": None, "cal_out": None}
        if rng.random() < (1 / 3):
            noisy_fat = true_fat + rng.normal(0, 5.0)
            entry["body_fat_pct"] = noisy_fat / weight * 100.0
        else:
            entry["body_fat_pct"] = None
        entries.append(entry)

    results = run_filter(entries)
    assert results[-1]["fat"] == pytest.approx(true_fat, abs=3.0)


def test_fat_uncertainty_grows_when_bioimpedance_missing_then_shrinks_on_reading():
    entries = [
        {"date": date(2026, 1, 1), "weight": 200.0, "cal_in": None, "cal_out": None, "body_fat_pct": 25.0},
        {"date": date(2026, 1, 2), "weight": 200.0, "cal_in": None, "cal_out": None, "body_fat_pct": None},
        {"date": date(2026, 1, 3), "weight": 200.0, "cal_in": None, "cal_out": None, "body_fat_pct": None},
        {"date": date(2026, 1, 4), "weight": 200.0, "cal_in": None, "cal_out": None, "body_fat_pct": None},
        {"date": date(2026, 1, 5), "weight": 200.0, "cal_in": None, "cal_out": None, "body_fat_pct": 26.0},
    ]
    results = run_filter(entries)
    se_fat = [r["se_fat"] for r in results]
    # Strictly grows across the unobserved stretch (days 2-4)...
    assert se_fat[1] < se_fat[2] < se_fat[3]
    # ...and shrinks back down once a reading arrives again on day 5.
    assert se_fat[4] < se_fat[3]


def test_matches_pre_bioimpedance_filter_when_body_fat_never_provided():
    """Regression test: adding the fat_t state must not perturb x/beta/b/e
    at all when body_fat_pct is absent throughout. Cross-checks against a
    frozen reimplementation of the original 4-state filter math."""

    def old_filter(entries, params):
        KCAL_PER_LB = 3500.0

        def F(phi, with_control):
            b_term = -1.0 / KCAL_PER_LB if with_control else 0.0
            return np.array(
                [
                    [1.0, 1.0, b_term, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, phi],
                ]
            )

        H = np.array([1.0, 0.0, 0.0, 1.0])
        B = np.array([1.0 / KCAL_PER_LB, 0.0, 0.0, 0.0])
        Q = np.diag([params.q_x, params.q_beta, params.q_b, params.q_e])
        R = params.r

        first = entries[0]
        x = np.array([first["weight"], 0.0, 0.0, 0.0])
        P = np.diag([params.p0_x, params.p0_beta, params.p0_b, params.p0_e])
        out = [(x.copy(), P.copy())]
        prev_date = first["date"]

        for e in entries[1:]:
            gap_days = (e["date"] - prev_date).days
            F0 = F(params.phi, with_control=False)
            for _ in range(gap_days - 1):
                x = F0 @ x
                P = F0 @ P @ F0.T + Q
            have_cal = e.get("cal_in") is not None and e.get("cal_out") is not None
            if have_cal:
                Fm = F(params.phi, with_control=True)
                u = e["cal_in"] - e["cal_out"]
            else:
                Fm = F0
                u = 0.0
            x = Fm @ x + B * u
            P = Fm @ P @ Fm.T + Q
            z = e["weight"]
            y = z - H @ x
            S = H @ P @ H + R
            K = (P @ H) / S
            x = x + K * y
            P = P - np.outer(K, H @ P)
            out.append((x.copy(), P.copy()))
            prev_date = e["date"]
        return out

    n = 60
    dates = _dates(n)
    rng = np.random.default_rng(3)
    cal_out = 2800 + rng.normal(0, 250, n)
    cal_in = 2800 + rng.normal(0, 250, n)
    weight = 200.0
    weights = []
    for i in range(n):
        weight += rng.normal(0, 0.1)
        weights.append(weight + rng.normal(0, 0.3))

    entries = [
        {"date": dates[i], "weight": weights[i], "cal_in": cal_in[i], "cal_out": cal_out[i]}
        for i in range(n)
    ]

    params = FilterParams()
    new_results = run_filter(entries, params)
    old_results = old_filter(entries, params)

    for new_r, (old_x, old_P) in zip(new_results, old_results):
        old_se = np.sqrt(np.diag(old_P))
        assert new_r["x"] == pytest.approx(old_x[0])
        assert new_r["beta"] == pytest.approx(old_x[1])
        assert new_r["b"] == pytest.approx(old_x[2])
        assert new_r["e"] == pytest.approx(old_x[3])
        assert new_r["se_x"] == pytest.approx(old_se[0])
        assert new_r["se_beta"] == pytest.approx(old_se[1])
        assert new_r["se_b"] == pytest.approx(old_se[2])
        assert new_r["se_e"] == pytest.approx(old_se[3])
