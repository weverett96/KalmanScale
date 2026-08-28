"""
4-state bias-augmented Kalman filter for weight trend estimation.

State vector s_t = [x_t, beta_t, b_t, e_t]:
  x_t    true weight (lb)
  beta_t trend (lb/day)
  b_t    net tracking bias, intake - Whoop-output (kcal/day)
  e_t    AR(1) autocorrelated scale-noise component (sodium/glycogen/GI)

See plans/KalmanScale_v1.md Section 3 for the derivation. Default params
below are placeholders (Milestone 6: tune against real data).
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
    phi: float = 0.7       # AR(1) persistence of water-weight component
    r: float = 0.09        # measurement noise var, white residual (lb^2)
    p0_x: float = 1.0      # initial covariance
    p0_beta: float = 0.01
    p0_b: float = 400.0
    p0_e: float = 1.0


_H = np.array([1.0, 0.0, 0.0, 1.0])
_B = np.array([1.0 / KCAL_PER_LB, 0.0, 0.0, 0.0])


def _F(phi: float, with_control: bool) -> np.ndarray:
    b_term = -1.0 / KCAL_PER_LB if with_control else 0.0
    return np.array(
        [
            [1.0, 1.0, b_term, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, phi],
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
        "se_x": float(se[0]),
        "se_beta": se_beta,
        "se_b": float(se[2]),
        "se_e": float(se[3]),
        "beta_z": beta / se_beta if se_beta > 0 else 0.0,
    }


def run_filter(entries: list[dict], params: FilterParams | None = None) -> list[dict]:
    """
    entries: list of dicts sorted by strictly increasing 'date' (a date
    object), each with 'weight' (float) and optional 'cal_in'/'cal_out'
    (float or None). A day counts as having calorie data only if both are
    present; otherwise it's treated as a zero-control step (per the "no
    calorie data" decision in plans/KalmanScale_v1.md Section 8).

    Returns one result dict per input entry (same order), plus synthesizes
    zero-control predict-only steps internally for any gap days (not
    included in the output, since there's no entry for them).
    """
    if params is None:
        params = FilterParams()
    if not entries:
        return []

    Q = np.diag([params.q_x, params.q_beta, params.q_b, params.q_e])
    R = params.r

    first = entries[0]
    x = np.array([first["weight"], 0.0, 0.0, 0.0])
    P = np.diag([params.p0_x, params.p0_beta, params.p0_b, params.p0_e])

    results = [_state_result(first["date"], x, P)]
    prev_date = first["date"]

    for e in entries[1:]:
        gap_days = (e["date"] - prev_date).days
        if gap_days < 1:
            raise ValueError(
                f"entries must be sorted by strictly increasing date; "
                f"got {e['date']} after {prev_date}"
            )

        # Predict-only for any skipped days in between (e still decays).
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

        z = e["weight"]
        y = z - _H @ x
        S = _H @ P @ _H + R
        K = (P @ _H) / S
        x = x + K * y
        P = P - np.outer(K, _H @ P)

        results.append(_state_result(e["date"], x, P))
        prev_date = e["date"]

    return results
