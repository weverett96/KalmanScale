# KalmanScale

Estimates true weight trends by fusing daily Garmin Index weigh-ins with cycling
ride energy (power-meter kJ) through a Kalman filter. All data comes from
[intervals.icu](https://intervals.icu).

## Idea

Daily scale weight is noisy (water retention, timing, etc.). Ride energy is measured
precisely by a power meter, but how much of it becomes a real deficit depends on how
much gets eaten back. A Kalman filter combines:

- **Process model**: baseline drift β (lb/day on a no-ride day) minus κ × yesterday's
  ride kcal / 3500, where κ — the fraction of ride kcal *not* eaten back — is learned
  from how weight responds to changes in ride volume
- **Measurements**: daily scale weight, plus Garmin Index body-fat % (as fat mass)

No intake logging required.

## Setup

1. In intervals.icu, connect Garmin and enable **Weight** and **Body Fat** under
   Settings → Integrations → Garmin Connect (wellness fields).
2. Copy your API key from intervals.icu Settings → Developer Settings into `.env`:
   `INTERVALS_API_KEY=...`
3. `uvicorn kalmanscale.app:app`, then **Sync from intervals.icu**.

intervals.icu only has data from when Garmin was connected (2026-09-20), so the
filter starts fresh from there. Expect κ to sit near its 50% prior with wide
uncertainty for the first few months — it needs varied ride volume to pin down.
