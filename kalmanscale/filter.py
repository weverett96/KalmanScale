"""
9-state Kalman filter for weight trend estimation, with weight split into
fat and lean mass, each with its own drift and ride response.

State vector s_t = [F, L, beta_F, beta_L, kappa_F, kappa_L, e, btape, g]:
  F, L        fat and lean mass (lb). Fat is on the Garmin Index
              bioimpedance scale; lean is everything else, excluding the
              transient water in e. True weight is F + L.
  beta_F/L    baseline fat / lean drift (lb/day) on a day with no riding.
              beta = beta_F + beta_L is the total baseline drift.
  kappa_F/L   fat / lean share of ride kcal that shows up as a deficit
              (i.e. not eaten back). kappa = kappa_F + kappa_L.
  e           AR(1) autocorrelated scale-noise component (sodium/glycogen/GI)
  btape       constant offset (lb of fat mass) between tape-measure (US Navy
              formula) body fat and the Index — the Index is the reference,
              so this absorbs the method gap; slow random walk
  g           glycogen-bound water (lb): tracks recent energy balance, so
              starting a deficit gives a one-time drop that then holds

Transition, with a = A_{t-1} / 3500 and A_{t-1} = the previous calendar
day's ride kcal (a morning weigh-in reflects yesterday's ride):

  D = (beta_F + beta_L) - (kappa_F + kappa_L) * a     (tissue change, lb)

  F_t      = F_{t-1} + beta_F - kappa_F * a
  L_t      = L_{t-1} + beta_L - kappa_L * a
  beta_F_t = beta_F - p * lam * D
  beta_L_t = beta_L - (1 - p) * lam * D
  g_t      = phi_g * g + (1 - phi_g) * eta * (beta_F + beta_L)

Metabolic adaptation: maintenance burn falls ~10 kcal/day per lb lost
(Hall), so each lb of tissue change moves the no-ride drift by
lam = 10 / 3500 lb/day in the other direction. Integrated, beta_t =
beta_0 - lam * (W_t - W_0) (plus its random walk): weight decays
exponentially toward the equilibrium where beta = 0 instead of falling
in a straight line. Only the deterministic part of D feeds it — process
noise in F and L doesn't, which is negligible at lam ~ 0.003.

Glycogen water: g is an EWMA (phi_g) of eta * beta, i.e. water held with
glycogen settles at eta lb per lb/day of baseline drift. A new deficit
first drops weight faster (g falling to its new level over ~1-2 weeks),
then g holds steady and only F + L keep moving. Without g the filter
reads that one-time drop as trend. Driven by beta, not rides: a ride's
glycogen is mostly refilled within a day or two, and a per-ride water
dip would compete with kappa for the same next-day weight response.

Linear in the state (A is a known input; F just varies per day). The fat
share of change is never a state — it emerges as beta_F / beta and
kappa_F / kappa, so it can differ between baseline and ride-driven change
and drift over time.

Priors and process noise are built in (total, deviation) coordinates:
  beta_F = p * beta + d_beta,   beta_L = (1 - p) * beta - d_beta
(same for kappa), with independent variances on the total and on the
deviation d. The totals get exactly the old unsplit model's variances, and
the deviation starts small, so the filter begins as a fixed-p split and
only moves off p as fat readings demand it. With the deviation variances
set to 0 it *is* the fixed-p model.

Measurements:
  scale   z = F + L + e + g
  Index   z = F + c * (e + g)   (c: fixed water loading of bioimpedance)
  tape    z = F + btape

With no fat readings, F + L, beta and kappa evolve exactly like the old
unsplit filter (q_fat + q_lean = q_x) — weight alone can't see the split —
once lam, eta and g's variances are 0.

See plans/KalmanScale_v1.md Section 3 for the derivation. Default params
below are placeholders (Milestone 6: tune against real data).
"""

from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta

import numpy as np

KCAL_PER_LB = 3500.0


@dataclass
class FilterParams:
    q_fat: float = 0.002   # process noise var, fat mass (lb^2/day)
    q_lean: float = 0.018  # process noise var, lean mass (lb^2/day); q_fat + q_lean = old q_x
    q_beta: float = 1e-4   # process noise var, total drift ((lb/day)^2/day); ~2 wk to register a diet change
    q_dbeta: float = 1e-7  # process noise var, fat/lean drift split deviation
    q_kappa: float = 1e-4  # process noise var, total ride-kcal retention (1/day)
    q_dkappa: float = 1e-5 # process noise var, fat/lean ride split deviation
    q_e: float = 0.05      # process noise var, AR(1) transient (lb^2)
    q_btape: float = 1e-3  # process noise var, tape-vs-Index offset (lb^2/day)
    phi: float = 0.7       # AR(1) persistence of water-weight component
    lam: float = 10.0 / KCAL_PER_LB  # metabolic adaptation (1/day): drift change per lb of tissue change
    eta: float = 10.0      # glycogen water at equilibrium, lb per (lb/day) of tissue balance
    phi_g: float = 0.85    # daily persistence of glycogen water (~1-2 wk to settle)
    q_g: float = 0.005     # process noise var, glycogen water (lb^2/day)
    p_fat: float = 0.75    # prior fat share of weight change
    c_water: float = 0.0   # bioimpedance fat-mass loading on e (unestimated)
    r: float = 0.09        # measurement noise var, white residual (lb^2)
    r_fat: float = 25.0    # measurement noise var, bioimpedance fat mass (lb^2)
    r_tape: float = 1.0    # measurement noise var, tape fat mass (lb^2): ±0.25 in ≈ ±1 lb
    kappa0: float = 0.5    # prior mean for total kappa (weakly informative)
    fat_frac0: float = 0.25  # initial fat fraction when day 1 has no Index reading
    p0_x: float = 1.0      # initial covariance: total weight
    p0_fat: float = 100.0  # fat (total-weight and fat priors are independent)
    p0_beta: float = 0.01  # total drift
    p0_dbeta: float = 4e-4 # split deviation, SD 0.02 lb/day (~0.14 lb/wk)
    p0_kappa: float = 0.25 # total kappa
    p0_dkappa: float = 0.01  # split deviation, SD 0.1
    p0_e: float = 1.0
    p0_btape: float = 225.0  # SD 15 lb: Navy vs BIA can differ by several % BF
    p0_g: float = 0.5      # glycogen water off its equilibrium eta * beta at the start


# State order: [F, L, beta_F, beta_L, kappa_F, kappa_L, e, btape, g]
N_STATES = 9
_H_WEIGHT = np.array([1.0, 1.0, 0, 0, 0, 0, 1.0, 0, 1.0])
_H_TAPE = np.array([1.0, 0, 0, 0, 0, 0, 0, 1.0, 0])
_TOTAL = np.array([1.0, 1.0, 0, 0, 0, 0, 0, 0, 0])


def _H_fat(c_water: float) -> np.ndarray:
    return np.array([1.0, 0, 0, 0, 0, 0, c_water, 0, c_water])


def _split(p: float) -> np.ndarray:
    """Maps (total, deviation) -> (fat part, lean part)."""
    return np.array([[p, 1.0], [1.0 - p, -1.0]])


def _split_cov(p: float, var_total: float, var_dev: float) -> np.ndarray:
    M = _split(p)
    return M @ np.diag([var_total, var_dev]) @ M.T


def _F(params: "FilterParams", ride_kcal: float) -> np.ndarray:
    a = ride_kcal / KCAL_PER_LB
    F = np.eye(N_STATES)
    F[0, 2], F[0, 4] = 1.0, -a
    F[1, 3], F[1, 5] = 1.0, -a
    F[6, 6] = params.phi
    D = np.array([0, 0, 1.0, 1.0, -a, -a, 0, 0, 0])  # tissue change
    F[2] -= params.p_fat * params.lam * D
    F[3] -= (1.0 - params.p_fat) * params.lam * D
    F[8, 8] = params.phi_g
    F[8, 2] = F[8, 3] = (1.0 - params.phi_g) * params.eta
    return F


def _Q(params: "FilterParams") -> np.ndarray:
    Q = np.zeros((N_STATES, N_STATES))
    Q[0, 0], Q[1, 1] = params.q_fat, params.q_lean
    Q[2:4, 2:4] = _split_cov(params.p_fat, params.q_beta, params.q_dbeta)
    Q[4:6, 4:6] = _split_cov(params.p_fat, params.q_kappa, params.q_dkappa)
    Q[6, 6], Q[7, 7], Q[8, 8] = params.q_e, params.q_btape, params.q_g
    return Q


def _vec(*idx) -> np.ndarray:
    v = np.zeros(N_STATES)
    v[list(idx)] = 1.0
    return v


# Linear readouts: (value vector) for reported quantities.
_READOUTS = {
    "fat": _vec(0),
    "lean": _vec(1),
    "x": _TOTAL,
    "beta": _vec(2, 3),
    "beta_fat": _vec(2),
    "beta_lean": _vec(3),
    "kappa": _vec(4, 5),
    "kappa_fat": _vec(4),
    "kappa_lean": _vec(5),
    "e": _vec(6),
    "btape": _vec(7),
    "g": _vec(8),
}


def state_result(day, x: np.ndarray, P: np.ndarray) -> dict:
    out = {"date": day.isoformat() if isinstance(day, Date) else day}
    for name, v in _READOUTS.items():
        out[name] = float(v @ x)
        out[f"se_{name}"] = float(np.sqrt(max(v @ P @ v, 0.0)))
    for part in ("", "_fat", "_lean"):
        out[f"cov_beta_kappa{part}"] = float(
            _READOUTS[f"beta{part}"] @ P @ _READOUTS[f"kappa{part}"]
        )
    out["beta_z"] = out["beta"] / out["se_beta"] if out["se_beta"] > 0 else 0.0
    return out


def navy_body_fat_pct(abdomen_in: float, neck_in: float, height_in: float) -> float:
    """US Navy circumference body-fat estimate (men), all inputs in inches.
    Abdomen is measured horizontally at the navel."""
    return (
        86.010 * np.log10(abdomen_in - neck_in)
        - 70.041 * np.log10(height_in)
        + 36.76
    )


def _ride_blocks(
    ride_kcal_by_date: dict[Date, float], start: Date, end: Date, decay: float
) -> list[tuple[Date, list[float], float]]:
    """Consecutive, non-overlapping 7-day blocks of [start, end], newest
    first, ending at `end` (matching a 7-day training calendar). Each is
    (block_start, daily kcal, EWMA weight): the newest block gets weight 1,
    the one before `decay`, then decay^2, and so on. The oldest block may
    be partial and holds only the days it has."""
    blocks = []
    block_end, k = end, 0
    while block_end >= start:
        block_start = max(start, block_end - timedelta(days=6))
        n_days = (block_end - block_start).days + 1
        kcal = [ride_kcal_by_date.get(block_start + timedelta(days=i), 0.0) for i in range(n_days)]
        blocks.append((block_start, kcal, decay**k))
        block_end, k = block_start - timedelta(days=1), k + 1
    return blocks


def forecast_ride_kcal(
    ride_kcal_by_date: dict[Date, float],
    start: Date,
    end: Date,
    decay: float = 0.7,
) -> float | None:
    """
    Forward-looking daily ride kcal, for projecting the trend: an EWMA of
    each 7-day block's mean daily kcal (see _ride_blocks; decay 0.7 ≈
    2-week half-life). A partial oldest block uses the days it has, and
    weights are normalized over the blocks that exist, so a short history
    isn't dragged toward zero. Returns None if start > end.
    """
    blocks = _ride_blocks(ride_kcal_by_date, start, end, decay)
    if not blocks:
        return None
    den = sum(w for _, _, w in blocks)
    return sum(w * np.mean(kcal) for _, kcal, w in blocks) / den


def sample_future_rides(
    ride_kcal_by_date: dict[Date, float],
    start: Date,
    end: Date,
    first_day: Date,
    horizon: int,
    n: int,
    rng: np.random.Generator,
    decay: float = 0.7,
) -> np.ndarray:
    """
    (n, horizon) ride kcal for days first_day .. first_day + horizon - 1.
    Each simulated 7-day week (starting at first_day) of each sample
    replays one past block from [start, end], picked with the EWMA weights,
    aligned by weekday so e.g. a long Saturday ride stays on Saturday. A
    partial block's missing weekdays are filled with its mean, so the
    expected daily kcal equals forecast_ride_kcal exactly.
    """
    blocks = _ride_blocks(ride_kcal_by_date, start, end, decay)
    if not blocks:
        return np.zeros((n, horizon))
    weights = np.array([w for _, _, w in blocks])
    weights /= weights.sum()

    # by_weekday[b, d] = block b's kcal on weekday d (date.weekday()).
    by_weekday = np.empty((len(blocks), 7))
    for b, (block_start, kcal, _) in enumerate(blocks):
        by_weekday[b, :] = np.mean(kcal)
        for i, k in enumerate(kcal):
            by_weekday[b, (block_start + timedelta(days=i)).weekday()] = k

    n_weeks = -(-horizon // 7)
    picks = rng.choice(len(blocks), size=(n, n_weeks), p=weights)
    days = np.arange(horizon)
    weekdays = np.array([(first_day + timedelta(days=int(d))).weekday() for d in days])
    return by_weekday[picks[:, days // 7], weekdays]


def _sqrt_psd(M: np.ndarray) -> np.ndarray:
    """A matrix S with S @ S.T == M, for a symmetric PSD M (tolerates
    tiny negative eigenvalues from rounding, and exact zeros)."""
    w, V = np.linalg.eigh((M + M.T) / 2)
    return V * np.sqrt(np.clip(w, 0.0, None))


def simulate_forecast(
    x: np.ndarray,
    P: np.ndarray,
    latest: Date,
    ride_kcal_by_date: dict[Date, float],
    history_start: Date,
    params: FilterParams | None = None,
    horizon: int = 30,
    n: int = 2000,
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
    seed: int | None = None,
) -> dict:
    """
    Monte Carlo forecast of true weight (F + L) for the `horizon` days
    after `latest`, starting from the filter's posterior (x, P) there.

    Each of `n` samples draws a starting state from N(x, P), then steps
    forward day by day with the filter's own dynamics and process noise.
    The step into day d uses day d-1's rides: the latest day's recorded
    rides if it has any (else it's simulated too — the weigh-in usually
    precedes the ride), then resampled past weeks (sample_future_rides,
    using blocks from history_start through the day before latest).

    Weight-dependent dynamics (metabolic adaptation, glycogen water) are
    in _F, so the median curves rather than following a straight line.

    Returns {"dates": [...], "q25": [...], "q50": [...], ...} (one key per
    quantile). seed defaults to latest's ordinal, so the band is stable
    across reloads and only changes when new data arrives.
    """
    if params is None:
        params = FilterParams()
    rng = np.random.default_rng(latest.toordinal() if seed is None else seed)

    X = x + rng.standard_normal((n, N_STATES)) @ _sqrt_psd(P).T
    Q_sqrt = _sqrt_psd(_Q(params))
    p = params.p_fat

    # Ride inputs for days latest .. latest + horizon - 1 (step d uses d-1).
    known_today = ride_kcal_by_date.get(latest)
    first_sim_day = latest + timedelta(days=1) if known_today is not None else latest
    sim_len = horizon - 1 if known_today is not None else horizon
    sampled = sample_future_rides(
        ride_kcal_by_date, history_start, latest - timedelta(days=1),
        first_sim_day, sim_len, n, rng,
    )
    rides = sampled if known_today is None else np.hstack([np.full((n, 1), known_today), sampled])

    out = {f"q{int(round(q * 100))}": [] for q in quantiles}
    dates = []
    for d in range(horizon):
        a = rides[:, d] / KCAL_PER_LB
        D = X[:, 2] + X[:, 3] - (X[:, 4] + X[:, 5]) * a
        X_next = X.copy()
        X_next[:, 0] = X[:, 0] + X[:, 2] - X[:, 4] * a
        X_next[:, 1] = X[:, 1] + X[:, 3] - X[:, 5] * a
        X_next[:, 2] = X[:, 2] - p * params.lam * D
        X_next[:, 3] = X[:, 3] - (1.0 - p) * params.lam * D
        X_next[:, 6] = params.phi * X[:, 6]
        X_next[:, 8] = params.phi_g * X[:, 8] + (1.0 - params.phi_g) * params.eta * (X[:, 2] + X[:, 3])
        X = X_next + rng.standard_normal((n, N_STATES)) @ Q_sqrt.T

        total = X[:, 0] + X[:, 1]
        for q, v in zip(quantiles, np.quantile(total, quantiles)):
            out[f"q{int(round(q * 100))}"].append(float(v))
        dates.append((latest + timedelta(days=d + 1)).isoformat())
    return {"dates": dates, **out}


def projected_trend(
    latest: dict, ride_kcal_per_day: float, part: str = ""
) -> tuple[float, float]:
    """(trend, se) in lb/day of tissue change as of today: beta plus the
    ride term at the forecast ride volume, beta - kappa * A / 3500, with
    the beta/kappa covariance in the SE. Metabolic adaptation slows it from
    here on (simulate_forecast accounts for that). part is "" for total
    weight, "_fat" or "_lean" for one component."""
    a = ride_kcal_per_day / KCAL_PER_LB
    trend = latest[f"beta{part}"] - latest[f"kappa{part}"] * a
    var = (
        latest[f"se_beta{part}"] ** 2
        + (a * latest[f"se_kappa{part}"]) ** 2
        - 2 * a * latest[f"cov_beta_kappa{part}"]
    )
    return trend, float(np.sqrt(max(var, 0.0)))


def _update(x: np.ndarray, P: np.ndarray, H: np.ndarray, R: np.ndarray, z: np.ndarray):
    """General KF update for an n-row measurement (H is (n,8), R is (n,n))."""
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
    or None) and 'tape_bf_pct' (Navy-formula body fat %, or None).
    Either body-fat % is converted to a derived fat-mass measurement
    (weight * pct / 100) using that day's weight — so a tape reading only
    counts on a day with a weigh-in. body_fat_pct measures F (+ c*e),
    tape_bf_pct measures F + btape; with neither, the F/L split is
    predict-only for that day.

    ride_kcal_by_date: {date: total ride kcal that day}. Days not present
    count as 0 kcal (no ride). The step into day d uses rides from day
    d - 1, including across gap days with no weigh-in — a ride on a day
    you didn't weigh in still counts toward the next weigh-in.

    Returns one result dict per input entry (same order), plus synthesizes
    predict-only steps internally for any gap days (not included in the
    output, since there's no entry for them).
    """
    return [state_result(day, x, P) for day, x, P in filter_states(entries, ride_kcal_by_date, params)]


def filter_states(
    entries: list[dict],
    ride_kcal_by_date: dict[Date, float] | None = None,
    params: FilterParams | None = None,
):
    """Yields (date, x, P) — the raw posterior mean and covariance — after
    each entry. See run_filter for the input contract."""
    if params is None:
        params = FilterParams()
    if ride_kcal_by_date is None:
        ride_kcal_by_date = {}
    if not entries:
        return

    Q = _Q(params)

    # First entry: initialize directly from the measurement (no KF update —
    # a single data point should initialize, not update; see the "single
    # data point" edge case in plans/KalmanScale_v1.md Section 3).
    first = entries[0]
    weight0 = first["weight"]
    bf0 = first.get("body_fat_pct")
    fat0 = weight0 * (bf0 / 100.0 if bf0 is not None else params.fat_frac0)
    p = params.p_fat
    x = np.array(
        [fat0, weight0 - fat0, 0.0, 0.0, p * params.kappa0, (1 - p) * params.kappa0, 0.0, 0.0, 0.0]
    )
    P = np.zeros((N_STATES, N_STATES))
    # Independent priors on total weight (p0_x) and fat (p0_fat), mapped
    # into [F, L] = [F, total - F]: var(L) = p0_x + p0_fat, cov(F, L) = -p0_fat.
    # This keeps F + L's prior at exactly p0_x, matching the unsplit model.
    P[0, 0] = params.p0_fat
    P[1, 1] = params.p0_x + params.p0_fat
    P[0, 1] = P[1, 0] = -params.p0_fat
    P[2:4, 2:4] = _split_cov(p, params.p0_beta, params.p0_dbeta)
    P[4:6, 4:6] = _split_cov(p, params.p0_kappa, params.p0_dkappa)
    P[6, 6], P[7, 7], P[8, 8] = params.p0_e, params.p0_btape, params.p0_g
    # Glycogen water starts near its equilibrium for the (unknown) drift,
    # g = eta * beta, give or take p0_g — no assumed diet change at day 1.
    T = np.eye(N_STATES)
    T[8, 2] = T[8, 3] = params.eta
    P = T @ P @ T.T

    yield first["date"], x, P
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
            F = _F(params, ride_kcal)
            x = F @ x
            P = F @ P @ F.T + Q

        x, P = _apply_measurement(x, P, e, params)

        yield e["date"], x, P
        prev_date = e["date"]


def _apply_measurement(x, P, e: dict, params: FilterParams):
    """Builds the per-day H/R/z — a weight row, plus a row each for an
    Index and a tape body-fat reading if present — and applies the update."""
    weight = e["weight"]
    rows, noise, z = [_H_WEIGHT], [params.r], [weight]

    bf_pct = e.get("body_fat_pct")
    if bf_pct is not None:
        rows.append(_H_fat(params.c_water))
        noise.append(params.r_fat)
        z.append(weight * bf_pct / 100.0)

    tape_pct = e.get("tape_bf_pct")
    if tape_pct is not None:
        rows.append(_H_TAPE)
        noise.append(params.r_tape)
        z.append(weight * tape_pct / 100.0)

    return _update(x, P, np.vstack(rows), np.diag(noise), np.array(z))
