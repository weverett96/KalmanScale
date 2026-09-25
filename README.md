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
- **Curvature**: maintenance burn falls ~10 kcal/day per lb lost, so β shifts by
  λ = 10/3500 per lb of tissue change and loss slows toward an equilibrium. Water
  held with glycogen settles at η lb per lb/day of β over ~1–2 weeks, so the
  one-time drop when a deficit starts isn't read as trend

No intake logging required.

The chart extends 30 days ahead with a Monte Carlo forecast: 2,000 draws from the
filter's current posterior, stepped forward with its own dynamics, with future rides
resampled from your past 7-day training weeks (recent weeks favored). Because β adapts as weight
changes, the median curves toward equilibrium rather than running in a straight
line. It shows the median true weight and a 50% interval.

## Setup

1. In intervals.icu, connect Garmin and enable **Weight** and **Body Fat** under
   Settings → Integrations → Garmin Connect (wellness fields).
2. Copy your API key from intervals.icu Settings → Developer Settings into `.env`:
   `INTERVALS_API_KEY=...`
3. `uvicorn kalmanscale.app:app`, then **Sync from intervals.icu**.

intervals.icu only has data from when Garmin was connected (2026-09-20), so the
filter starts fresh from there. Expect κ to sit near its 50% prior with wide
uncertainty for the first few months — it needs varied ride volume to pin down.
