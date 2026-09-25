"""
Sanity tests for the filter core — synthetic recovery of beta/kappa,
identifiability, ride timing, gaps, and the fat-mass bolt-on.
"""

from datetime import date, timedelta

import numpy as np
import pytest

from kalmanscale.filter import (
    FilterParams,
    forecast_ride_kcal,
    navy_body_fat_pct,
    projected_trend,
    run_filter,
)

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
    """Reduction test: with no fat readings, the Option B fat/lean split must
    reproduce a frozen, independently written unsplit [x, beta, kappa, e]
    filter exactly (x = F + L, q_x = q_fat + q_lean) — weight alone can't
    see the split."""

    def reference(entries, rides, params):
        H = np.array([1.0, 0.0, 0.0, 1.0])
        Q = np.diag([params.q_fat + params.q_lean, params.q_beta, params.q_kappa, params.q_e])
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


def test_navy_body_fat_pct_hand_computed():
    # 86.010*log10(24) - 70.041*log10(64.5) + 36.76
    expected = 86.010 * 1.380211 - 70.041 * 1.809560 + 36.76
    assert navy_body_fat_pct(40.5, 16.5, 64.5) == pytest.approx(expected, abs=1e-3)
    assert navy_body_fat_pct(38.5, 16.5, 64.5) < navy_body_fat_pct(40.5, 16.5, 64.5)


def _fat_series(n, true_fat, offset, tape_every, seed, weight=200.0):
    """Constant true fat; daily noisy Index readings (SD 5 lb) and, every
    `tape_every` days, a precise tape reading (SD 1 lb) offset by `offset`."""
    rng = np.random.default_rng(seed)
    entries = []
    for i, d in enumerate(_dates(n)):
        entry = {"date": d, "weight": weight}
        entry["body_fat_pct"] = (true_fat + rng.normal(0, 5.0)) / weight * 100.0
        if tape_every and i % tape_every == 0:
            entry["tape_bf_pct"] = (true_fat + offset + rng.normal(0, 1.0)) / weight * 100.0
        entries.append(entry)
    return entries


def test_tape_offset_recovered():
    final = run_filter(_fat_series(120, 45.0, 14.0, tape_every=7, seed=5))[-1]
    assert final["btape"] == pytest.approx(14.0, abs=3 * final["se_btape"])
    assert final["fat"] == pytest.approx(45.0, abs=3 * final["se_fat"])


def test_tape_readings_tighten_fat_estimate():
    # Short window where Index noise dominates: weekly tape shouldn't hurt,
    # and must shrink fat uncertainty once the offset is learned.
    bia_only = run_filter(_fat_series(120, 45.0, 14.0, tape_every=0, seed=6))[-1]
    with_tape = run_filter(_fat_series(120, 45.0, 14.0, tape_every=7, seed=6))[-1]
    assert with_tape["se_fat"] < bia_only["se_fat"]


def test_fat_readings_inform_beta():
    # Option B: fat observations constrain delta = beta - kappa*A/3500 via
    # p, so adding weekly tape readings must tighten beta.
    base = [{"date": d, "weight": 200.0 - 0.1 * i} for i, d in enumerate(_dates(60))]
    with_tape = [dict(e) for e in base]
    for i in range(0, 60, 7):
        # Fat falling at p * 0.1 lb/day, on a 14 lb Navy offset.
        with_tape[i]["tape_bf_pct"] = (50.0 + 14.0 - 0.075 * i) / base[i]["weight"] * 100.0
    for i in range(0, 60, 1):
        with_tape[i]["body_fat_pct"] = (50.0 - 0.075 * i) / base[i]["weight"] * 100.0
    a, b = run_filter(base)[-1], run_filter(with_tape)[-1]
    assert b["se_beta"] < a["se_beta"]


def _simulate_split(n, p, true_beta, fat0, lean0, offset, seed, c_water=0.0, spike_day=None):
    rng = np.random.default_rng(seed)
    F, L, e = fat0, lean0, 0.0
    entries, truth = [], []
    for i, d in enumerate(_dates(n)):
        if i:
            F += p * true_beta + rng.normal(0, 0.02)
            L += (1 - p) * true_beta + rng.normal(0, 0.1)
            e = 0.7 * e + rng.normal(0, 0.2)
        if spike_day is not None and i == spike_day:
            e += 3.0  # sodium/glycogen spike
        weight = F + L + e + rng.normal(0, 0.3)
        entry = {"date": d, "weight": weight,
                 "body_fat_pct": (F + c_water * e + rng.normal(0, 5.0)) / weight * 100.0}
        if i % 7 == 0:
            entry["tape_bf_pct"] = (F + offset + rng.normal(0, 1.0)) / weight * 100.0
        entries.append(entry)
        truth.append((F, L))
    return entries, truth


def test_recovers_fat_and_lean_split():
    entries, truth = _simulate_split(180, 0.75, -0.08, 50.0, 150.0, 14.0, seed=8)
    final = run_filter(entries)[-1]
    F, L = truth[-1]
    assert final["fat"] == pytest.approx(F, abs=3 * final["se_fat"])
    assert final["lean"] == pytest.approx(L, abs=3 * final["se_lean"])
    assert final["btape"] == pytest.approx(14.0, abs=3 * final["se_btape"])
    assert final["se_fat"] < 2.0


def test_water_spike_does_not_move_fat_when_loading_modeled():
    # Index reads a water spike as fat loss (c = -2 lb fat per lb water).
    # With c modeled correctly, the fat estimate should stay closer to the
    # truth across the spike than with c = 0.
    entries, truth = _simulate_split(
        60, 0.75, -0.05, 50.0, 150.0, 14.0, seed=9, c_water=-2.0, spike_day=40
    )
    right = run_filter(entries, params=FilterParams(c_water=-2.0))
    wrong = run_filter(entries, params=FilterParams(c_water=0.0))
    err = lambda res: max(abs(res[i]["fat"] - truth[i][0]) for i in range(40, 45))
    assert err(right) < err(wrong)


def _fixed_p_reference(entries, rides, params):
    """Frozen, independently written fixed-p Option B filter, state
    [F, L, beta, kappa, e, btape]: F += p*delta, L += (1-p)*delta."""
    p, n = params.p_fat, 6
    Q = np.diag([params.q_fat, params.q_lean, params.q_beta, params.q_kappa, params.q_e, params.q_btape])
    first = entries[0]
    w0, bf0 = first["weight"], first.get("body_fat_pct")
    f0 = w0 * (bf0 / 100.0 if bf0 is not None else params.fat_frac0)
    x = np.array([f0, w0 - f0, 0.0, params.kappa0, 0.0, 0.0])
    P = np.diag([params.p0_fat, params.p0_x + params.p0_fat, params.p0_beta, params.p0_kappa, params.p0_e, params.p0_btape])
    P[0, 1] = P[1, 0] = -params.p0_fat
    out = [x.copy()]
    day = first["date"]
    for e in entries[1:]:
        while day < e["date"]:
            a = rides.get(day, 0.0) / 3500.0
            F = np.eye(n)
            F[0, 2], F[0, 3] = p, -p * a
            F[1, 2], F[1, 3] = 1 - p, -(1 - p) * a
            F[4, 4] = params.phi
            x, P = F @ x, F @ P @ F.T + Q
            day += timedelta(days=1)
        H, R, z = [[1, 1, 0, 0, 1, 0]], [params.r], [e["weight"]]
        if e.get("body_fat_pct") is not None:
            H.append([1, 0, 0, 0, params.c_water, 0]); R.append(params.r_fat)
            z.append(e["weight"] * e["body_fat_pct"] / 100.0)
        if e.get("tape_bf_pct") is not None:
            H.append([1, 0, 0, 0, 0, 1]); R.append(params.r_tape)
            z.append(e["weight"] * e["tape_bf_pct"] / 100.0)
        H, R, z = np.array(H, float), np.diag(R), np.array(z)
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        x, P = x + K @ (z - H @ x), P - K @ H @ P
        out.append(x.copy())
    return out


def test_zero_split_deviation_is_exactly_fixed_p_model():
    # With no room for the fat/lean split to deviate from p, the 8-state
    # filter must reproduce the fixed-p model, fat readings included.
    entries, _ = _simulate_split(90, 0.75, -0.08, 50.0, 150.0, 14.0, seed=10)
    rides = {d: r for d, r in zip(_dates(90), _varying_rides(90, seed=11))}
    params = FilterParams(p0_dbeta=0.0, q_dbeta=0.0, p0_dkappa=0.0, q_dkappa=0.0)
    for new, ref in zip(run_filter(entries, rides, params), _fixed_p_reference(entries, rides, params)):
        assert new["fat"] == pytest.approx(ref[0])
        assert new["lean"] == pytest.approx(ref[1])
        assert new["beta"] == pytest.approx(ref[2])
        assert new["kappa"] == pytest.approx(ref[3])
        assert new["beta_fat"] == pytest.approx(0.75 * ref[2])
        assert new["btape"] == pytest.approx(ref[5])


def test_learns_fat_lean_split_that_differs_from_prior():
    # Recomposition: losing fat while gaining lean — a split far from the
    # 75/25 prior. Weekly tape + daily Index readings should reveal it.
    n = 240
    beta_fat, beta_lean = -0.08, 0.03
    rng = np.random.default_rng(12)
    F, L, e = 50.0, 150.0, 0.0
    entries = []
    for i, d in enumerate(_dates(n)):
        if i:
            F += beta_fat + rng.normal(0, 0.02)
            L += beta_lean + rng.normal(0, 0.1)
            e = 0.7 * e + rng.normal(0, 0.2)
        w = F + L + e + rng.normal(0, 0.3)
        entry = {"date": d, "weight": w, "body_fat_pct": (F + rng.normal(0, 5.0)) / w * 100.0}
        if i % 7 == 0:
            entry["tape_bf_pct"] = (F + 14.0 + rng.normal(0, 1.0)) / w * 100.0
        entries.append(entry)

    final = run_filter(entries)[-1]
    assert final["beta_fat"] == pytest.approx(beta_fat, abs=3 * final["se_beta_fat"])
    assert final["beta_lean"] == pytest.approx(beta_lean, abs=3 * final["se_beta_lean"])
    # And it's actually learned, not just wide error bars: lean drift comes
    # out positive although total drift is negative (a fixed 75/25 split
    # would force it to -0.0125), and far tighter than its prior.
    params = FilterParams()
    prior_se_lean = np.sqrt(0.25**2 * params.p0_beta + params.p0_dbeta)
    assert final["beta_lean"] > 0
    assert final["se_beta_lean"] < 0.5 * prior_se_lean
