# KalmanScale

Estimates true weight trends by fusing daily weight measurements with energy-balance data
(Whoop calorie expenditure and estimated calorie intake) through a Kalman filter.

## Idea

Daily scale weight is noisy (water retention, timing, etc.), but energy balance
(intake − expenditure) predicts the *direction and rate* of weight change. A Kalman
filter combines:

- **Process model**: predicted weight change from estimated daily caloric surplus/deficit
  (using Whoop kcal expenditure and estimated kcal intake)
- **Measurement**: daily scale weight readings

to produce a smoothed, more accurate estimate of underlying weight trend than either
signal alone.

## Status

Early scaffolding.
