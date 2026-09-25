"""
5-state Kalman filter for weight trend estimation.

State vector s_t = [x_t, beta_t, kappa_t, e_t, fat_t]:
  x_t      true weight (lb)
  beta_t   baseline drift (lb/day): average intake minus non-ride
           expenditure, i.e. what weight does on a day with no riding
  kappa_t  fraction of ride kcal that actually shows up as a deficit
           (i.e. not eaten back); slow random walk
  e_t      AR(1) autocorrelated scale-noise component (sodium/glycogen/GI)
  fat_t    true fat mass (lb), from Garmin Index bioimpedance

Transition, with A_{t-1} = the previous calendar day's ride kcal (a morning
weigh-in reflects yesterday's ride, not today's):

  x_t = x_{t-1} + beta_{t-1} - kappa_{t-1} * A_{t-1} / 3500

A_{t-1} is a known input, so this is still linear in the state — F just
varies per day. A day with no ride is A = 0, not missing data.

fat_t is currently a bolt-on, uncoupled random walk (Option A) — it has no
interaction with x/beta/kappa/e. Coupling it into the weight dynamics
(Option B, e.g. splitting x_t into fat + lean/water) is a deliberate future
step, not done here; see plans/KalmanScale_v1.md Section 3.

See plans/KalmanScale_v1.md Section 3 for the core derivation. Default
params below are placeholders (Milestone 6: tune against real data).
"""

from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta

import numpy as np

KCAL_PER_LB = 3500.0


@dataclass
class FilterParams:
    q_x: float = 0.02      # process noise var, weight (lb^2/day)
    q_beta: float = 1e-5   # process noise var, trend ((lb/day)^2/day)
    q_kappa: float = 1e-4  # process noise var, ride-kcal retention (1/day)
    q_e: float = 0.05      # process noise var, AR(1) transient (lb^2)
    q_fat: float = 0.01    # process noise var, fat mass (lb^2/day)
    phi: float = 0.7       # AR(1) persistence of water-weight component
    r: float = 0.09        # measurement noise var, white residual (lb^2)
    r_fat: float = 25.0    # measurement noise var, bioimpedance fat mass (lb^2)
    kappa0: float = 0.5    # prior mean for kappa (weakly informative)
    p0_x: float = 1.0      # initial covariance
    p0_beta: float = 0.01
    p0_kappa: float = 0.25
    p0_e: float = 1.0
    p0_fat: float = 100.0


# State order: [x, beta, kappa, e, fat]
_H_WEIGHT = np.array([1.0, 0.0, 0.0, 1.0, 0.0])
_H_FAT = np.array([0.0, 0.0, 0.0, 0.0, 1.0])


def _F(phi: float, ride_kcal: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 1.0, -ride_kcal / KCAL_PER_LB, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, phi, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )


def _state_result(day, x: np.ndarray, P: np.ndarray) -> dict:
    se = np.sqrt(np.diag(P))
    beta, se_beta = float(x[1]), float(se[1])
    return {
        "date": day.isoformat() if isinstance(day, Date) else day,
        "x": float(x[0]),
        "beta": beta,
        "kappa": float(x[2]),
        "e": float(x[3]),
        "fat": float(x[4]),
        "se_x": float(se[0]),
        "se_beta": se_beta,
        "se_kappa": float(se[2]),
        "se_e": float(se[3]),
        "se_fat": float(se[4]),
        "beta_z": beta / se_beta if se_beta > 0 else 0.0,
        "cov_beta_kappa": float(P[1, 2]),
    }


def forecast_ride_kcal(
    ride_kcal_by_date: dict[Date, float],
    start: Date,
    end: Date,
    decay: float = 0.7,
) -> float | None:
    """
    Forward-looking daily ride kcal, for projecting the trend. Splits
    [start, end] into consecutive 7-day blocks ending at `end` (matching a
    7-day training calendar), takes each block's mean daily kcal, and
    combines them with an EWMA: the most recent block gets weight 1, the
    one before `decay`, then decay^2, and so on (decay 0.7 ≈ 2-week
    half-life). A partial oldest block uses the days it has, and weights
    are normalized over the blocks that exist, so a short history isn't
    dragged toward zero. Returns None if start > end.
    """
    if start > end:
        return None
    num = den = 0.0
    block_end, k = end, 0
    while block_end >= start:
        block_start = max(start, block_end - timedelta(days=6))
        n_days = (block_end - block_start).days + 1
        total = sum(
            ride_kcal_by_date.get(block_start + timedelta(days=i), 0.0) for i in range(n_days)
        )
        w = decay**k
        num += w * total / n_days
        den += w
        block_end, k = block_start - timedelta(days=1), k + 1
    return num / den


def projected_trend(latest: dict, ride_kcal_per_day: float) -> tuple[float, float]:
    """(trend, se) in lb/day: beta plus the ride term at the forecast ride
    volume, beta - kappa * A / 3500, with the beta/kappa covariance in the
    SE."""
    a = ride_kcal_per_day / KCAL_PER_LB
    trend = latest["beta"] - latest["kappa"] * a
    var = (
        latest["se_beta"] ** 2
        + (a * latest["se_kappa"]) ** 2
        - 2 * a * latest["cov_beta_kappa"]
    )
    return trend, float(np.sqrt(max(var, 0.0)))


def _update(x: np.ndarray, P: np.ndarray, H: np.ndarray, R: np.ndarray, z: np.ndarray):
    """General KF update for a 1- or 2-row measurement (H is (n,5), R is (n,n))."""
    y = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    P = P - K @ H @ P
    return x, P


def run_filter(
    entries: list[dict],
    ride_kcal_by_date: dict[Date, float] | None = None,
    params: FilterParams | None = None,
) -> list[dict]:
    """
    entries: list of dicts sorted by strictly increasing 'date' (a date
    object), each with 'weight' (float) and optional 'body_fat_pct' (float
    or None). body_fat_pct, if present, is converted to a derived fat-mass
    measurement (weight * body_fat_pct / 100) and used to update fat_t; if
    absent, fat_t is predict-only for that day.

    ride_kcal_by_date: {date: total ride kcal that day}. Days not present
    count as 0 kcal (no ride). The step into day d uses rides from day
    d - 1, including across gap days with no weigh-in — a ride on a day
    you didn't weigh in still counts toward the next weigh-in.

    Returns one result dict per input entry (same order), plus synthesizes
    predict-only steps internally for any gap days (not included in the
    output, since there's no entry for them).
    """
    if params is None:
        params = FilterParams()
    if ride_kcal_by_date is None:
        ride_kcal_by_date = {}
    if not entries:
        return []

    Q = np.diag([params.q_x, params.q_beta, params.q_kappa, params.q_e, params.q_fat])

    # First entry: initialize directly from the measurement (no KF update —
    # a single data point should initialize, not update; see the "single
    # data point" edge case in plans/KalmanScale_v1.md Section 3).
    first = entries[0]
    first_bf_pct = first.get("body_fat_pct")
    fat0 = first["weight"] * first_bf_pct / 100.0 if first_bf_pct is not None else 0.0
    x = np.array([first["weight"], 0.0, params.kappa0, 0.0, fat0])
    P = np.diag(
        [params.p0_x, params.p0_beta, params.p0_kappa, params.p0_e, params.p0_fat]
    )

    results = [_state_result(first["date"], x, P)]
    prev_date = first["date"]

    for e in entries[1:]:
        gap_days = (e["date"] - prev_date).days
        if gap_days < 1:
            raise ValueError(
                f"entries must be sorted by strictly increasing date; "
                f"got {e['date']} after {prev_date}"
            )

        # One predict step per calendar day up to and including e's date;
        # skipped days are predict-only (e decays, fat randomly walks).
        for k in range(gap_days):
            ride_kcal = ride_kcal_by_date.get(prev_date + timedelta(days=k), 0.0)
            F = _F(params.phi, ride_kcal)
            x = F @ x
            P = F @ P @ F.T + Q

        x, P = _apply_measurement(x, P, e, params)

        results.append(_state_result(e["date"], x, P))
        prev_date = e["date"]

    return results


def _apply_measurement(x, P, e: dict, params: FilterParams):
    """Builds the per-day H/R/z (1 row if only weight observed, 2 rows if
    body_fat_pct is also present) and applies the update."""
    weight = e["weight"]
    bf_pct = e.get("body_fat_pct")

    if bf_pct is not None:
        H = np.vstack([_H_WEIGHT, _H_FAT])
        R = np.diag([params.r, params.r_fat])
        z = np.array([weight, weight * bf_pct / 100.0])
    else:
        H = _H_WEIGHT.reshape(1, -1)
        R = np.array([[params.r]])
        z = np.array([weight])

    return _update(x, P, H, R, z)
