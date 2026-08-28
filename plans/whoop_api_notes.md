# Whoop API notes

Research notes on the Whoop Developer Platform, gathered ahead of Milestone 5
(automated `cal_out` ingestion) in `KalmanScale_v1.md`. Not an implementation
doc — just what's actually available and how the integration will work.

## 1. Platform overview

- Register a team + app at `developer-dashboard.whoop.com`. Requires your own
  Whoop account to authenticate into the dashboard.
- Up to 5 apps per team by default; more available on request.
- An app approval step exists before production launch — details weren't
  fully surfaced by search. **Verify this during actual Milestone 5 setup**;
  it may add lead time before the cron job can go live against a real token.

## 2. OAuth2 flow

Standard authorization-code flow:

- **Authorization endpoint:** `https://api.prod.whoop.com/oauth/oauth2/auth`
- **Token endpoint:** `https://api.prod.whoop.com/oauth/oauth2/token`
  (also used for refresh)
- Params: `client_id` / `client_secret` (from the dashboard), a pre-registered
  `redirect_uri`, a `state` value (CSRF protection, min 8 chars), and `scope`.
- **`offline` scope is required to get a refresh token** — confirms the
  assumption already in `KalmanScale_v1.md` Section 4.
- Using a refresh token invalidates the previous access token (so don't
  refresh speculatively/concurrently).
- Access tokens are short-lived (`expires_in` seconds in the token response);
  an expired token returns `401`.
- **Refresh tokens rotate and are single-use**: every refresh call returns a
  *new* refresh token alongside the new access token, and immediately
  invalidates the one that was just used. There's no fixed expiry on a
  refresh token by itself, but using it burns it. Confirmed working
  end-to-end via `scripts/whoop_auth.py` on 2026-08-27 (successful
  `GET /v2/cycle` call with the resulting access token).
  **Design implication for Milestone 5**: the cron job must persist the new
  `refresh_token` from every refresh response back to storage, not just
  reuse the original — and must never run concurrent refresh calls against
  the same token (the second of two racing requests fails, since the first
  invalidates it).

## 3. Scopes

Six member-data scopes: `read:cycles`, `read:recovery`, `read:sleep`,
`read:workout`, `read:profile`, `read:body_measurement`.

KalmanScale only needs `read:cycles` for `cal_out`, but a typical/default
scope request is `offline read:cycles read:sleep read:recovery read:workout
read:body_measurement read:profile` — the others are free to request now
even if unused today (see Section 5 note on recovery data).

## 4. Rate limits

100 requests/minute, 10,000/day per client (`X-RateLimit-*` response
headers); increases available on request. Not a concern at single-user
scale — a daily cron pulling a handful of endpoints is nowhere close.

## 5. Relevant endpoints (v2)

- **`GET /v2/cycle`** — paginated physiological cycles, sorted by start time
  descending, filterable by time range. **This is the one KalmanScale
  needs.** The `score` object includes:
  - `kilojoule` — energy expenditure. Convert to kcal: `kcal = kilojoule / 4.184`.
  - `strain`, `average_heart_rate`, `max_heart_rate` — not used by the
    filter, come along for free.
  - `score_state` — see the important note below.
- `GET /v2/cycle/{id}` — single cycle by ID.
- `GET /v2/recovery`, `GET /v2/cycle/{id}/recovery` — `recovery_score`,
  `resting_heart_rate`, `hrv_rmssd_milli`, `spo2_percentage`,
  `skin_temp_celsius`, `user_calibrating` flag. Not used by the filter
  today, but worth storing alongside `cal_out` in case it's useful later
  (e.g. as a covariate for the AR(1) water-weight term — recovery/HRV
  correlates with hydration and inflammation).
- `GET /v2/activity/sleep`, `GET /v2/activity/sleep/{id}` — sleep stage
  breakdown.
- `GET /v2/activity/workout`, `GET /v2/activity/workout/{id}` — per-workout
  `kilojoule`/`strain`/heart rate, `distance_meter`, `altitude_gain_meter`,
  `zone_durations`.
- `GET /v2/user/profile/basic` — name, email, user ID.
- `GET /v2/user/measurement/body` — height, weight, max heart rate. **Note:**
  this is Whoop's own recorded body weight, not the scale value the plan
  gets from Garmin — don't conflate the two as a data source for `entries.weight`.
- `DELETE /v2/user/access` — revoke access (useful for a future "disconnect
  Whoop" feature; not needed for v1).

**Important — `score_state`:** every score-bearing object (`cycle`,
`recovery`, `workout`) has a `score_state` field, seen values include
`SCORED` and `PENDING_SCORE`. A cycle can be returned by the API before
Whoop finishes scoring it (e.g. if the cron runs early in the morning
before the previous day's cycle closes out) — `kilojoule` shouldn't be
trusted unless `score_state == SCORED`. This is a concrete case to add to
the "log failures loudly" handling in `KalmanScale_v1.md` Section 4: a
`PENDING_SCORE` cycle should be skipped and retried on the next run, not
upserted as zero/null.

## 6. Webhooks (v2) — considered, not recommended for v1

Whoop supports webhooks: register an HTTPS endpoint, and Whoop POSTs an
event notification when a user's cycle/recovery/sleep/workout data updates.
The payload only says *what* changed — you still call the API to fetch the
actual data. Requires webhook signature validation on the receiving end.
(v1 webhooks are deprecated/removed; v2 changed recovery webhook and
activity-ID handling.)

For KalmanScale: the existing daily-cron-poll design (Milestone 5) is
simpler and sufficient for single-user personal use, and avoids needing a
publicly reachable HTTPS endpoint on the home Pi (port-forwarding or a
tunnel). Worth revisiting only if the Pi ends up exposed to the internet
for some other reason anyway.

## 7. Implications for KalmanScale_v1.md (not yet applied)

- Confirms the cron + OAuth2 + `offline` scope design in Section 4 is
  workable as written.
- New concrete detail to fold in when Milestone 5 is implemented: check
  `score_state == SCORED` before upserting `cal_out`, and convert energy
  with `kcal = kilojoule / 4.184`.
- Terminology note: don't use Whoop's `GET /v2/user/measurement/body`
  weight as a data source — `entries.weight` comes from Garmin per the
  existing plan.

## Sources

- [WHOOP Developer Platform](https://developer.whoop.com/docs/introduction/)
- [Getting Started](https://developer.whoop.com/docs/developing/getting-started/)
- [OAuth 2.0](https://developer.whoop.com/docs/developing/oauth/)
- [WHOOP API Docs](https://developer.whoop.com/api/)
- [Webhooks](https://developer.whoop.com/docs/developing/webhooks/)
- [v1 to v2 Migration Guide](https://developer.whoop.com/docs/developing/v1-v2-migration/)
