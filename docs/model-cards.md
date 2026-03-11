# Model Cards

## Baseline Pace Model

Purpose:

- estimate clean-air expected lap pace before race noise, traffic, and large tyre penalties

Current algorithm:

- gradient boosting regressor

Training inputs:

- driver
- constructor
- compound
- event/circuit
- season
- lap number
- air/track temperature
- humidity
- qualifying pace proxy
- team strength proxy
- fuel-load proxy
- stint progress
- wet-track flag
- green-flag indicator

Default target:

- `clean_air_baseline_seconds`

Metrics logged:

- MAE
- RMSE

Current limitation:

- uncertainty is residual-based rather than fully probabilistic

## Tyre Degradation Model

Purpose:

- estimate lap delta relative to baseline as tyres age in race conditions

Current algorithm:

- gradient boosting regressor

Training inputs:

- driver
- constructor
- compound
- event/circuit
- season
- lap number
- tyre age
- traffic density
- air/track temperature
- qualifying/team priors
- stint progress
- degradation class
- wet-track flag
- green-flag indicator

Default target:

- `lap_time_delta_to_baseline`

Metrics logged:

- MAE
- RMSE

Current limitation:

- the target is derived from baseline rather than a dedicated calibrated degradation curve model

