# Simulator Notes

StrategyLab uses a Monte Carlo lap-by-lap simulator instead of a single end-state predictor.

## Current V1 Mechanics

Each lap time is built from:

- baseline pace
- compound factor
- fuel effect proxy
- degradation effect
- traffic penalty
- weather penalty
- safety car / VSC penalty
- pace noise
- out-lap penalty

Pit stop handling includes:

- pit lane transit loss
- stationary time noise
- reduced effective loss under VSC/SC
- compound reset
- out-lap warmup effect

Race interruptions:

- one sampled SC or VSC window can occur in a run
- strategy windows can flex under SC/VSC if the stop definition allows it

Traffic and overtakes:

- local traffic penalty is applied when gaps are small
- pass probability depends on pace delta, overtake skill, and track overtaking difficulty

## What the Result Means

Monte Carlo outputs are not exact predictions. They are decision support outputs:

- expected finish position
- expected race time
- finish probability distribution
- pit-window range
- traffic risk
- undercut/overcut heuristics

## Known V1 Simplifications

- only one interruption regime is sampled per race run
- rivals use simple default strategies rather than optimized responses
- weather is intentionally simple
- traffic is local rather than fully telemetry-derived
- tyre cliff behavior is implicit through degradation, not a dedicated learned hazard model yet

