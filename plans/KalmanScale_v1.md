# Trend Filter — Project Plan

A local web app that fuses daily weight, logged intake, and Whoop expenditure through a bias-augmented Kalman filter, per the model derived in chat.

## 1. Stack

- **Backend:** Python + FastAPI. Reuses your existing numerical stack (numpy/scipy); easiest place to keep the filter correct and testable.
- **Storage:** SQLite (single-user, local, zero config). One table is enough at this scale.
- **Frontend:** Static HTML/JS served by FastAPI, Chart.js for the trend chart. No build step, no framework — this is a personal instrument, not a product.
- **Deployment:** `uvicorn` on a Raspberry Pi on your home network, always-on. No auth, no multi-user concerns — reachable from phone/laptop over LAN, and doesn't depend on a laptop being open. (A plain localhost deployment on your laptop still works fine for initial development.)

Swap any of this out freely — the only real constraint is that the filter core (Section 3) should be a standalone, unit-testable module independent of the API/UI, since that's the part whose correctness actually matters.

## 2. Data model

```sql
CREATE TABLE entries (
    date TEXT PRIMARY KEY,      -- ISO 8601
    weight REAL NOT NULL,       -- lb
    cal_in REAL,                -- nullable
    cal_out REAL                -- nullable, Whoop
);
```

No separate "filter runs" table — the filter is cheap enough (O(n), n = days logged) to rerun on every request rather than cache incrementally. Revisit only if n grows into the thousands.

## 3. Filter core (`filter.py`)

State vector: $s_t = [x_t, \beta_t, b_t, e_t]^\top$
- $x_t$ — true weight (lb)
- $\beta_t$ — trend (lb/day)
- $b_t$ — net tracking bias, intake − Whoop-output (kcal/day)
- $e_t$ — AR(1) autocorrelated component of scale noise (sodium/glycogen/GI persistence)

Transition (control input $u_t = (I_t - E_t)/3500$):

$$x_t = x_{t-1} + \beta_{t-1} + \frac{I_t - E_t - b_{t-1}}{3500}, \quad \beta_t = \beta_{t-1}, \quad b_t = b_{t-1}, \quad e_t = \phi e_{t-1} + \xi_t$$

where $\xi_t \sim N(0, q_e)$ and $\phi \in (0,1)$ is the day-to-day persistence of the water-weight component (start around $\phi \approx 0.6$–$0.8$, i.e. a 2–4 day half-life, and treat as a tunable parameter, not a fixed constant).

Measurement: $z_t = x_t + e_t + \eta_t$, with $\eta_t \sim N(0, r)$ now the *genuinely* white residual (scale precision, timing jitter) — smaller than the old single-noise $r$, since the autocorrelated part has been pulled out into $e_t$.

Still linear in the state (4x4 $F$ now, with $\phi$ in the $(4,4)$ entry), so a standard KF still applies — no EKF needed. $Q = \text{diag}(q_x, q_\beta, q_b, q_e)$, $r$, and $\phi$ are configuration, not hardcoded. Consider using a tested library (e.g. `filterpy`) for the predict/update linear algebra rather than hand-rolling the covariance update — that's the highest-risk place for a subtle bug (asymmetric or non-PSD $P$) to hide; per-day $F$/$Q$ construction for gaps and missing calorie data still has to be bespoke either way.

**Identifiability risk — $\beta$ vs. $b$:** both states add a constant per-day offset to $x_t$ ($\beta_{t-1}$ directly, $b_{t-1}$ scaled by $\div 3500$), and are structurally interchangeable from the measurement's point of view — an increase in $\beta$ and a decrease in $b$ predict the same $x_t$. They're only separable if $I_t - E_t$ varies enough over time to excite the difference; if intake/expenditure tracking is roughly steady day to day, $\hat\beta$ and $\hat b$ can trade off against each other even though $\hat x$ stays fine. Worth resolving with a synthetic test (see below) before trusting the individual values, and worth being clear that $\beta$ may end up representing residual/unexplained drift rather than "the trend" once $u_t$ already accounts for most calorie-driven change — see the caveat on the stat panel in Section 6.

**Garmin Index bioimpedance (implemented 2026-08-29):** a 5th state $fat_t$ (true fat mass, lb) was added, sourced from the Garmin Index scale's body-fat % reading. Currently **Option A**: $fat_t = fat_{t-1} + \zeta_t$, $\zeta_t \sim N(0, q_{fat})$ — a bolt-on random walk, uncoupled from $x/\beta/b/e$, updated from a derived measurement $z_{fat} = \text{weight} \times \text{body\_fat\_pct}/100$ only on days a reading exists (missing days are predict-only, same pattern as $e_t$ across gaps — verified in `tests/test_filter.py`). Raw BMI was considered and rejected as a data source since it's a fixed multiple of weight already logged, carrying no new information.

**Option B (not yet implemented, the actual point of adding this):** couple $fat_t$ into the weight dynamics — e.g. split $x_t$ into fat + lean/water components so bioimpedance helps de-confound the $\beta$/$b$ identifiability problem above, since fat mass tracks caloric balance more directly than raw scale weight does. Deferred because it requires restructuring the state relationships, not just appending a state, and BIA readings are themselves hydration-sensitive (arguably more so than the scale) — that correlation with $e_t$ needs deliberate handling or the two noisy signals will just confuse each other. Fat mass (not raw body-fat %) was chosen as the unit specifically so this extension is incremental rather than a rewrite.

**Why this matters:** with only white $\eta_t$, the filter's posterior SE on $\beta$ reads more confident than it should, because it can't distinguish "true trend" from "still riding out Tuesday's high-sodium dinner." Separating $e_t$ out lets the filter correctly attribute a multi-day run of readings in one direction to a decaying transient rather than folding it into $\beta$ or $b$.

**Testing approach for this module specifically:**
- Generate synthetic data with known $\beta^*$, $b^*$, $\phi^*$, and injected AR(1) + white noise; assert the filter recovers $\beta^*$ and $b^*$ within a few SEs, and recovers something close to $\phi^*$/$q_e$ if you're estimating them rather than fixing them.
- Specifically test that a simulated 3-day sodium spike (large $\xi_t$ then decay) does *not* get absorbed into $\hat\beta$ — this is the whole point of the extra state.
- Identifiability check: run a synthetic series with $I_t - E_t$ held nearly constant and confirm whether $\hat\beta$ and $\hat b$ converge to their individual true values, or only their sum does. If they don't separate cleanly, revisit whether both need to be free states, or whether a tighter $q_b$ (bias changes slowly) relative to $q_\beta$ resolves it.
- Edge cases: missing `cal_in`/`cal_out` on some days (net should fall back to 0 control input, not crash), gaps of >1 day between entries (multi-step predict, remembering $e_t$ decays each skipped day too), a single data point (should not update, just initialize).

## 4. Data ingestion

**Whoop — automated.** Whoop has a self-serve developer platform (free, requires a Whoop membership). Register an app in the Developer Dashboard, run the OAuth flow once against your own account to get an access + refresh token (request the `offline` scope so the refresh token doesn't expire), then:
- A daily cron job on the Pi refreshes the access token and pulls the latest cycle/recovery/workout data, computing `cal_out` and upserting it into `entries`.
- Store the refresh token securely on the Pi (env var or a gitignored file, not in the repo).
- Log failures loudly (Whoop's API does change; a silent failure here just looks like a flat calorie line and could go unnoticed for a while).

**Garmin (weight) — manual, by design.** Garmin's Connect Developer Program only grants access to vetted organizations, not personal use — there's no legitimate self-serve path. Unofficial libraries exist but rely on reverse-engineered auth that broke once already in 2026 and can break again without warning. Given that fragility relative to the payoff (one number a day), skip automating this:
- Enter weight manually each morning through the web UI, or
- Periodically export weight history as CSV from Garmin Connect and bulk-import.
- Revisit only if the manual entry actually becomes a compliance problem in practice.

**Intake (`cal_in`) — manual, by design.** No app integration (e.g. MyFitnessPal, Cronometer) for v1 — enter an estimate manually through the same form each day. This is a deliberate scope decision, not a gap; see Section 8.

Cronometer specifically was investigated (2026-08-28) and rejected on the same grounds as Garmin: no official self-serve API (enterprise-partner access only); the only integration path is an unofficial, reverse-engineered client hitting Cronometer's mobile app backend, authenticating with your actual account **username/password** (not a revocable OAuth token) — worse exposure than the Garmin case, for the same "one number a day" payoff. Revisit only if manual entry becomes a real compliance problem, same as Garmin.

## 5. API

- `POST /entries` — upsert one day (date, weight, cal_in?, cal_out?)
- `DELETE /entries/{date}`
- `GET /entries` — list, sorted by date
- `GET /filter` — run filter over all entries with current params, return per-day filtered state + full posterior covariance for the latest day
- `GET/PUT /params` — get/set $q_x, q_\beta, q_b, q_e, \phi, r$ (persist alongside the DB, e.g. a `params` table with one row)

## 6. Frontend

- Entry form (date, weight, cal in, cal out)
- Chart: raw scatter + filtered trend line (Chart.js)
- Stat panel: filtered weight, $\beta$ (lb/wk) ± SE, implied kcal/day balance, estimated bias $b$ ± SE, current water-weight transient $e_t$, and a plain z-score readout of whether $\beta$ is distinguishable from zero. Caveat: since $u_t$ already accounts for most calorie-driven weight change, $\beta$ may read more as "residual/unexplained trend" than "the trend" — worth labeling accordingly rather than as a plain rate (see the identifiability note in Section 3).
- Advanced/collapsed panel: $Q$ (including $q_e$), $\phi$, $r$ inputs
- Log table with delete

This is close to a working reference implementation of the above already sitting in this chat as a single-file HTML artifact (client-side only, no persistence backend, and built on the earlier 3-state model without the AR(1) term) — worth pulling from as a frontend starting point once the API contract is in place, but the filter logic itself needs the 4-state update.

## 7. Milestones

1. **Filter core + tests** — 4-state model including the AR(1) term from the start; get this right in isolation before anything touches a UI. No historical backfill for v1 (see Section 8), so synthetic tests are the *sole* pre-launch correctness check — there's no real dataset to sanity-check against before go-live, so synthetic coverage needs to stand on its own. Also expect $\hat\beta$'s posterior SE to be wide for the first 1–2 weeks of real use (little data yet) — that's correct filter behavior, not a bug, worth remembering when the app first goes live.
2. **API + SQLite** — CRUD on entries, wire the filter endpoint.
3. **Frontend** — form, chart, stats, talking to the API.
4. **Deploy to Pi** — get the app running persistently on the Pi before adding automation on top of it, so you can tell ingestion bugs apart from deployment bugs.
5. **Whoop cron job** — OAuth setup, token refresh, daily pull into `entries`, failure logging.
6. **Param tuning pass** — sanity-check default $Q$, $\phi$, and $r$ against a few weeks of your actual data; adjust rather than trust the values used in the earlier prototype.
7. **v2 (optional):** units toggle (kg/lb); Garmin CSV bulk-import helper; historical backfill of past Whoop/weight data to seed the filter on setup.

## 8. Open decisions to make while building

- ~~MyFitnessPal/Cronometer integration for `cal_in`~~ — resolved: manual entry only, explicit non-goal for v1 (Section 4).
- ~~Historical backfill on first launch~~ — resolved: start fresh, no backfill for v1; deferred to v2 (Section 7).
- Units: lb-only vs. toggle. Simplify to one unit for v1.
- What happens on a day with a weight entry but no calorie data — treat as a pure random-walk step on $x,\beta$ with $b$ carried forward? (Recommended: yes, exactly that — control input just drops to 0.)
- Whether `GET /filter` returns the full trajectory (needed for the chart) or just the latest state (needed for the stat panel) — probably both, single response.
- SQLite backup strategy off the Pi's SD card (`sqlite3 .backup` on a cron schedule, synced somewhere off-device) — SD card failure is the most likely single point of data loss in this whole setup.
- Whether the Pi is reachable only on your home LAN or you want remote access (e.g. Tailscale) — LAN-only is simpler and probably sufficient.