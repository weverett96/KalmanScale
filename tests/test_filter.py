"""
Sanity tests for the filter core — synthetic recovery of beta/kappa,
identifiability, ride timing, gaps, and the fat-mass bolt-on.
"""

from datetime import date, timedelta

import numpy as np
import pytest

from kalmanscale.filter import FilterParams, forecast_ride_kcal, projected_trend, run_filter

np.random.seed(0)


def _dates(n, start=date(2026, 1, 1)):
    return [start + timedelta(days=i) for i in range(n)]


def _simulate(ride_kcal, true_beta, true_kappa, seed, x0=200.0, phi=0.7):
    """Generate weights from the filter's own generative model: the step
    into day i uses rides from day i-1, plus AR(1) water noise and white
    scale noise."""
    rng = np.random.default_rng(seed)
    x, e = x0, 0.0
    weights = [x0]
    for i in range(1, len(ride_kcal)):
        x += true_beta - true_kappa * ride_kcal[i - 1] / 3500.0 + rng.normal(0, 0.05)
        e = phi * e + rng.normal(0, 0.2)
        weights.append(x + e + rng.normal(0, 0.3))
    return weights


def _varying_rides(n, seed):
    """Weekday/weekend pattern with alternating 3-week build and 1-week rest
    blocks — realistic variation in ride volume."""
    rng = np.random.default_rng(seed)
    rides = np.zeros(n)
    for i in range(n):
        block_scale = 0.4 if (i // 7) % 4 == 3 else 1.0
        dow = i % 7
        if dow in (5, 6):
            base = 1800.0  # long weekend rides
        elif dow in (1, 3):
            base = 800.0
        else:
            base = 0.0
        rides[i] = max(0.0, base * block_scale + rng.normal(0, 150) if base else 0.0)
    return rides


def test_recovers_beta_and_kappa_with_varying_ride_volume():
    n = 240
    true_beta = 0.05    # lb/day baseline gain on a no-ride day
    true_kappa = 0.6    # 40% of ride kcal eaten back
    dates = _dates(n)
    rides = _varying_rides(n, seed=1)
    weights = _simulate(rides, true_beta, true_kappa, seed=2)

    entries = [{"date": dates[i], "weight": weights[i]} for i in range(n)]
    ride_map = {dates[i]: rides[i] for i in range(n)}
    final = run_filter(entries, ride_map)[-1]

    assert final["kappa"] == pytest.approx(true_kappa, abs=3 * final["se_kappa"])
    assert final["beta"] == pytest.approx(true_beta, abs=3 * final["se_beta"])
    # And the SEs are actually informative, not just wide enough to pass.
    assert final["se_kappa"] < 0.2


def test_kappa_not_identifiable_with_constant_rides():
    # Identifiability check: with the same ride every day, -kappa*A/3500 is
    # just another constant per-day offset, indistinguishable from beta.
    # Only their combined effect on daily weight change is recoverable, and
    # kappa's posterior should stay close to its prior width.
    n = 240
    true_beta, true_kappa, A = 0.05, 0.6, 700.0
    dates = _dates(n)
    rides = np.full(n, A)
    weights = _simulate(rides, true_beta, true_kappa, seed=3)

    entries = [{"date": dates[i], "weight": weights[i]} for i in range(n)]
    final = run_filter(entries, {d: A for d in dates})[-1]

    combined = final["beta"] - final["kappa"] * A / 3500.0
    assert combined == pytest.approx(true_beta - true_kappa * A / 3500.0, abs=0.01)
    assert final["se_kappa"] > 0.5 * FilterParams().p0_kappa**0.5


def test_no_rides_leaves_kappa_at_prior():
    entries = [{"date": d, "weight": 200.0 - 0.01 * i} for i, d in enumerate(_dates(30))]
    params = FilterParams()
    final = run_filter(entries)[-1]
    assert final["kappa"] == pytest.approx(params.kappa0)


def test_ride_enters_step_to_next_day_not_same_day():
    params = FilterParams()
    d0, d1, d2 = _dates(3)
    entries = [{"date": d0, "weight": 200.0}, {"date": d1, "weight": 200.0}, {"date": d2, "weight": 200.0}]

    # A ride on d1 must not affect the d1 estimate (weigh-in precedes it)...
    base = run_filter(entries, {}, params)
    ride_d1 = run_filter(entries, {d1: 3500.0}, params)
    assert ride_d1[1]["x"] == pytest.approx(base[1]["x"])
    # ...but must pull the d2 prediction down.
    assert ride_d1[2]["x"] < base[2]["x"]


def test_ride_on_gap_day_counts_toward_next_weigh_in():
    d = _dates(5)
    entries = [{"date": d[0], "weight": 200.0}, {"date": d[4], "weight": 200.0}]
    base = run_filter(entries)
    ride_gap = run_filter(entries, {d[2]: 3500.0})
    assert ride_gap[1]["x"] < base[1]["x"]
    assert ride_gap[1]["kappa"] != pytest.approx(base[1]["kappa"])


def test_multi_day_gap_decays_e_and_does_not_crash():
    entries = [
        {"date": date(2026, 1, 1), "weight": 200.0},
        {"date": date(2026, 1, 8), "weight": 198.0},
    ]
    results = run_filter(entries, {date(2026, 1, 3): 1200.0})
    assert len(results) == 2
    assert np.isfinite(results[-1]["x"])


def test_unsorted_entries_raise():
    entries = [
        {"date": date(2026, 1, 2), "weight": 200.0},
        {"date": date(2026, 1, 1), "weight": 200.0},
    ]
    with pytest.raises(ValueError):
        run_filter(entries)


def test_single_data_point_just_initializes():
    entries = [{"date": date(2026, 1, 1), "weight": 200.0}]
    results = run_filter(entries)
    assert len(results) == 1
    assert results[0]["x"] == pytest.approx(200.0)
    params = FilterParams()
    assert results[0]["se_x"] == pytest.approx(params.p0_x**0.5)


def test_single_data_point_with_body_fat_initializes_fat_directly():
    entries = [{"date": date(2026, 1, 1), "weight": 200.0, "body_fat_pct": 25.0}]
    results = run_filter(entries)
    assert results[0]["fat"] == pytest.approx(50.0)  # 200 * 25%


def test_fat_mass_recovered_with_intermittent_bioimpedance_readings():
    # body_fat_pct is only logged ~1/3 of days: the filter must handle
    # missing bioimpedance observations without special casing.
    n = 90
    true_fat = 60.0  # lb, held constant
    weight = 200.0
    dates = _dates(n)
    rng = np.random.default_rng(2)

    entries = []
    for i in range(n):
        entry = {"date": dates[i], "weight": weight}
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
        {"date": date(2026, 1, 1), "weight": 200.0, "body_fat_pct": 25.0},
        {"date": date(2026, 1, 2), "weight": 200.0, "body_fat_pct": None},
        {"date": date(2026, 1, 3), "weight": 200.0, "body_fat_pct": None},
        {"date": date(2026, 1, 4), "weight": 200.0, "body_fat_pct": None},
        {"date": date(2026, 1, 5), "weight": 200.0, "body_fat_pct": 26.0},
    ]
    results = run_filter(entries)
    se_fat = [r["se_fat"] for r in results]
    # Strictly grows across the unobserved stretch (days 2-4)...
    assert se_fat[1] < se_fat[2] < se_fat[3]
    # ...and shrinks back down once a reading arrives again on day 5.
    assert se_fat[4] < se_fat[3]


def test_matches_reference_implementation():
    """Cross-checks run_filter (weight-only days, with rides and a gap)
    against a frozen, independently written 4-state [x, beta, kappa, e]
    filter — fat is uncoupled, so it must not perturb the other states."""

    def reference(entries, rides, params):
        H = np.array([1.0, 0.0, 0.0, 1.0])
        Q = np.diag([params.q_x, params.q_beta, params.q_kappa, params.q_e])
        first = entries[0]
        x = np.array([first["weight"], 0.0, params.kappa0, 0.0])
        P = np.diag([params.p0_x, params.p0_beta, params.p0_kappa, params.p0_e])
        out = [(x.copy(), P.copy())]
        day = first["date"]
        for e in entries[1:]:
            while day < e["date"]:
                A = rides.get(day, 0.0)
                F = np.array(
                    [
                        [1.0, 1.0, -A / 3500.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, params.phi],
                    ]
                )
                x = F @ x
                P = F @ P @ F.T + Q
                day += timedelta(days=1)
            S = H @ P @ H + params.r
            K = (P @ H) / S
            x = x + K * (e["weight"] - H @ x)
            P = P - np.outer(K, H @ P)
            out.append((x.copy(), P.copy()))
        return out

    n = 60
    dates = [d for i, d in enumerate(_dates(n)) if i not in (20, 21, 40)]
    rides = {d: r for d, r in zip(_dates(n), _varying_rides(n, seed=4))}
    rng = np.random.default_rng(3)
    entries = [{"date": d, "weight": 200.0 + rng.normal(0, 0.5)} for d in dates]

    params = FilterParams()
    new_results = run_filter(entries, rides, params)
    ref_results = reference(entries, rides, params)

    for new_r, (ref_x, ref_P) in zip(new_results, ref_results):
        ref_se = np.sqrt(np.diag(ref_P))
        for i, key in enumerate(["x", "beta", "kappa", "e"]):
            assert new_r[key] == pytest.approx(ref_x[i])
            assert new_r[f"se_{key}"] == pytest.approx(ref_se[i])


def test_forecast_ride_kcal_weights_recent_weeks_more():
    end = date(2026, 3, 14)
    rides = {}
    # Older week (Feb 29..Mar 7 region): 1400 kcal/day avg; latest week: 700.
    for i in range(7):
        rides[end - timedelta(days=i)] = 700.0
        rides[end - timedelta(days=7 + i)] = 1400.0
    f = forecast_ride_kcal(rides, end - timedelta(days=13), end, decay=0.5)
    assert f == pytest.approx((700.0 + 0.5 * 1400.0) / 1.5)


def test_forecast_ride_kcal_partial_history_not_dragged_to_zero():
    end = date(2026, 3, 10)
    rides = {end - timedelta(days=1): 2100.0}
    # Only 3 days of history: a single partial block, mean over those days.
    f = forecast_ride_kcal(rides, end - timedelta(days=2), end)
    assert f == pytest.approx(700.0)
    assert forecast_ride_kcal(rides, end + timedelta(days=1), end) is None


def test_projected_trend_combines_beta_and_kappa():
    latest = {"beta": 0.05, "kappa": 0.5, "se_beta": 0.02, "se_kappa": 0.1, "cov_beta_kappa": 0.0}
    trend, se = projected_trend(latest, 700.0)
    assert trend == pytest.approx(0.05 - 0.5 * 0.2)
    assert se == pytest.approx(np.hypot(0.02, 0.2 * 0.1))
    # With no riding, trend is just beta.
    assert projected_trend(latest, 0.0) == (pytest.approx(0.05), pytest.approx(0.02))
