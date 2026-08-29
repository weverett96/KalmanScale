"""
5-state bias-augmented Kalman filter for weight trend estimation.

State vector s_t = [x_t, beta_t, b_t, e_t, fat_t]:
  x_t    true weight (lb)
  beta_t trend (lb/day)
  b_t    net tracking bias, intake - Whoop-output (kcal/day)
  e_t    AR(1) autocorrelated scale-noise component (sodium/glycogen/GI)
  fat_t  true fat mass (lb), from Garmin Index bioimpedance

fat_t is currently a bolt-on, uncoupled random walk (Option A) — it has no
interaction with x/beta/b/e. Coupling it into the weight dynamics (Option
B, e.g. splitting x_t into fat + lean/water) is a deliberate future step,
not done here; see plans/KalmanScale_v1.md Section 3.

See plans/KalmanScale_v1.md Section 3 for the core derivation. Default
params below are placeholders (Milestone 6: tune against real data).
"""

from dataclasses import dataclass
from datetime import date as Date

import numpy as np

KCAL_PER_LB = 3500.0


@dataclass
class FilterParams:
    q_x: float = 0.02      # process noise var, weight (lb^2/day)
    q_beta: float = 1e-5   # process noise var, trend ((lb/day)^2/day)
    q_b: float = 4.0       # process noise var, bias ((kcal/day)^2/day)
    q_e: float = 0.05      # process noise var, AR(1) transient (lb^2)
    q_fat: float = 0.01    # process noise var, fat mass (lb^2/day)
    phi: float = 0.7       # AR(1) persistence of water-weight component
    r: float = 0.09        # measurement noise var, white residual (lb^2)
    r_fat: float = 25.0    # measurement noise var, bioimpedance fat mass (lb^2)
    p0_x: float = 1.0      # initial covariance
    p0_beta: float = 0.01
    p0_b: float = 400.0
    p0_e: float = 1.0
    p0_fat: float = 100.0


# State order: [x, beta, b, e, fat]
_H_WEIGHT = np.array([1.0, 0.0, 0.0, 1.0, 0.0])
_H_FAT = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
_B = np.array([1.0 / KCAL_PER_LB, 0.0, 0.0, 0.0, 0.0])


def _F(phi: float, with_control: bool) -> np.ndarray:
    b_term = -1.0 / KCAL_PER_LB if with_control else 0.0
    return np.array(
        [
            [1.0, 1.0, b_term, 0.0, 0.0],
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
        "b": float(x[2]),
        "e": float(x[3]),
        "fat": float(x[4]),
        "se_x": float(se[0]),
        "se_beta": se_beta,
        "se_b": float(se[2]),
        "se_e": float(se[3]),
        "se_fat": float(se[4]),
        "beta_z": beta / se_beta if se_beta > 0 else 0.0,
    }


def _update(x: np.ndarray, P: np.ndarray, H: np.ndarray, R: np.ndarray, z: np.ndarray):
    """General KF update for a 1- or 2-row measurement (H is (n,5), R is (n,n))."""
    y = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    P = P - K @ H @ P
    return x, P


def run_filter(entries: list[dict], params: FilterParams | None = None) -> list[dict]:
    """
    entries: list of dicts sorted by strictly increasing 'date' (a date
    object), each with 'weight' (float) and optional 'cal_in'/'cal_out'/
    'body_fat_pct' (float or None). A day counts as having calorie data
    only if both cal_in and cal_out are present; otherwise it's treated as
    a zero-control step (per the "no calorie data" decision in
    plans/KalmanScale_v1.md Section 8). body_fat_pct, if present, is
    converted to a derived fat-mass measurement (weight * body_fat_pct /
    100) and used to update fat_t; if absent, fat_t is predict-only for
    that day (its uncertainty just grows, like e_t across a gap day).

    Returns one result dict per input entry (same order), plus synthesizes
    zero-control predict-only steps internally for any gap days (not
    included in the output, since there's no entry for them).
    """
    if params is None:
        params = FilterParams()
    if not entries:
        return []

    Q = np.diag([params.q_x, params.q_beta, params.q_b, params.q_e, params.q_fat])

    # First entry: initialize directly from the measurement (no KF update —
    # a single data point should initialize, not update; see the "single
    # data point" edge case in plans/KalmanScale_v1.md Section 3).
    first = entries[0]
    first_bf_pct = first.get("body_fat_pct")
    fat0 = first["weight"] * first_bf_pct / 100.0 if first_bf_pct is not None else 0.0
    x = np.array([first["weight"], 0.0, 0.0, 0.0, fat0])
    P = np.diag(
        [params.p0_x, params.p0_beta, params.p0_b, params.p0_e, params.p0_fat]
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

        # Predict-only for any skipped days in between (e and fat still
        # evolve — e decays, fat randomly walks — with no measurement).
        F0 = _F(params.phi, with_control=False)
        for _ in range(gap_days - 1):
            x = F0 @ x
            P = F0 @ P @ F0.T + Q

        have_calories = e.get("cal_in") is not None and e.get("cal_out") is not None
        if have_calories:
            F = _F(params.phi, with_control=True)
            u = e["cal_in"] - e["cal_out"]
        else:
            F = F0
            u = 0.0

        x = F @ x + _B * u
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
