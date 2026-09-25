from datetime import date, timedelta

import numpy as np
import pytest

from kalmanscale.filter import (
    _TOTAL,
    FilterParams,
    _F,
    _Q,
    filter_states,
    forecast_ride_kcal,
    sample_future_rides,
    simulate_forecast,
)


def _dates(n, start=date(2026, 3, 1)):
    return [start + timedelta(days=i) for i in range(n)]


def _history(n, seed=0):
    rng = np.random.default_rng(seed)
    dates = _dates(n)
    entries = [
        {"date": d, "weight": 200.0 - 0.05 * i + rng.normal(0, 0.4), "body_fat_pct": 22.0}
        for i, d in enumerate(dates)
    ]
    rides = {d: float(rng.choice([0.0, 600.0, 1500.0])) for d in dates}
    return entries, rides


def test_matches_exact_gaussian_forecast_with_fixed_ride_schedule():
    # 8 days of history -> exactly one full 7-day block before the latest
    # day, so resampled weeks are deterministic. The model is then linear-
    # Gaussian, so the Monte Carlo median and IQR must match the exact
    # Kalman predict step.
    entries, rides = _history(8)
    params = FilterParams()
    latest, x, P = list(filter_states(entries, rides, params))[-1]
    horizon, n = 30, 20000
    fc = simulate_forecast(x, P, latest, rides, entries[0]["date"], params, horizon, n, seed=1)

    block = {d.weekday(): rides[d] for d in _dates(7)}
    Q = _Q(params)
    for d in range(horizon):
        prev = latest + timedelta(days=d)
        ride = rides[latest] if d == 0 else block[prev.weekday()]
        F = _F(params, ride)
        x, P = F @ x, F @ P @ F.T + Q
        mean, sd = _TOTAL @ x, np.sqrt(_TOTAL @ P @ _TOTAL)
        assert fc["q50"][d] == pytest.approx(mean, abs=0.03 * sd)
        assert fc["q75"][d] - fc["q25"][d] == pytest.approx(1.349 * sd, rel=0.03)


def test_resampled_rides_average_to_ewma_forecast():
    # 17 days: two full blocks plus a 3-day partial oldest block.
    rides = {d: float((i * 37) % 11) * 150.0 for i, d in enumerate(_dates(17))}
    start, end = _dates(17)[0], _dates(17)[-1]
    sampled = sample_future_rides(rides, start, end, end + timedelta(days=1), 28, 20000, np.random.default_rng(2))
    assert sampled.mean() == pytest.approx(forecast_ride_kcal(rides, start, end), rel=0.02)


def test_resampling_keeps_weekday_pattern():
    # One week of history: every simulated week replays it, weekday-aligned.
    week = _dates(7)
    rides = {d: 100.0 * (d.weekday() + 1) for d in week}
    first = week[-1] + timedelta(days=1)
    sampled = sample_future_rides(rides, week[0], week[-1], first, 21, 50, np.random.default_rng(3))
    expected = [100.0 * ((first + timedelta(days=i)).weekday() + 1) for i in range(21)]
    assert np.allclose(sampled, expected)


def test_forecast_is_deterministic_and_widens():
    entries, rides = _history(20)
    latest, x, P = list(filter_states(entries, rides))[-1]
    a = simulate_forecast(x, P, latest, rides, entries[0]["date"])
    b = simulate_forecast(x, P, latest, rides, entries[0]["date"])
    assert a == b
    width = [hi - lo for lo, hi in zip(a["q25"], a["q75"])]
    assert width[-1] > 2 * width[0]
    assert len(a["dates"]) == 30 and a["dates"][0] == (latest + timedelta(days=1)).isoformat()


def test_unrecorded_ride_on_latest_day_is_simulated_not_zero():
    # Weigh-in before the ride: a missing ride on the latest day shouldn't
    # count as a rest day. With every past day a 1000 kcal ride, the first
    # forecast step should include one.
    dates = _dates(15)
    entries = [{"date": d, "weight": 200.0} for d in dates]
    rides = {d: 1000.0 for d in dates[:-1]}
    latest, x, P = list(filter_states(entries, rides))[-1]
    simulated = simulate_forecast(x, P, latest, rides, dates[0], n=4000, seed=4)
    rest_day = simulate_forecast(x, P, latest, {**rides, latest: 0.0}, dates[0], n=4000, seed=4)
    assert simulated["q50"][0] < rest_day["q50"][0]


def test_band_includes_ride_volume_uncertainty():
    # Guard against "simplifying" the forecast to a fixed average ride
    # schedule: with uneven past weeks and rides that really drive weight
    # (kappa ~ 1), resampling whole weeks must widen the band well beyond
    # the exact Gaussian forecast under the EWMA-expected weekday schedule.
    rng = np.random.default_rng(1)
    dates = _dates(84)
    scale = lambda i: 3.0 if (i // 7) % 3 == 0 else 0.2
    rides = {d: scale(i) * float(rng.choice([0.0, 600.0, 1500.0])) for i, d in enumerate(dates)}
    x, entries = 200.0, []
    for i, d in enumerate(dates):
        if i:
            x += 0.1 - rides[dates[i - 1]] / 3500.0 + rng.normal(0, 0.1)
        entries.append({"date": d, "weight": x + rng.normal(0, 0.4), "body_fat_pct": 22.0})

    params = FilterParams()
    latest, xs, P = list(filter_states(entries, rides, params))[-1]
    fc = simulate_forecast(xs, P, latest, rides, dates[0], params, n=20000, seed=5)

    # Expected ride kcal per weekday: the average schedule the resampler
    # draws around (every weekday gets the same EWMA mix of past weeks).
    expected = sample_future_rides(
        rides, dates[0], latest - timedelta(days=1), latest, 30, 20000, np.random.default_rng(6)
    ).mean(axis=0)
    Q = _Q(params)
    for d in range(30):
        F = _F(params, expected[d])
        xs, P = F @ xs, F @ P @ F.T + Q
    fixed_iqr = 1.349 * np.sqrt(_TOTAL @ P @ _TOTAL)
    assert fc["q75"][-1] - fc["q25"][-1] > 1.5 * fixed_iqr
