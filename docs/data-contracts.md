# Data Contracts

The public contracts live in `src/strategylab/contracts.py`.

## Core Records

### `RaceKey`

Identifies a race/session with:

- `season`
- `event_name`
- `circuit`
- `session_type`

It also exposes a slug used for storage and manifest naming.

### `DriverLapRecord`

The canonical per-driver, per-lap row used for processed data and feature generation.

Important fields:

- lap time and sector times
- tyre compound and tyre age
- stint number
- pit flags
- track status
- weather state
- position/gap state
- qualifying/team priors
- clean-air baseline target

### `TrackProfile`

Carries circuit-level simulator priors:

- lap length
- total laps
- pit lane loss
- overtaking difficulty
- degradation score
- DRS zone count
- safety-car prior

### `SessionBundle`

Returned by source adapters and used by ingestion:

- `race_key`
- `track_profile`
- `laps`
- `stints`
- `pit_events`
- `weather_samples`
- `timeline`
- `source_metadata`

## Model Contracts

### `DatasetManifest`

Tracks:

- dataset version
- layer
- description
- record count
- source sessions
- feature columns
- target columns
- file path

### `EvaluationReport`

Tracks:

- model name/version
- dataset version
- metric list
- notes

### `ModelPrediction`

Tracks:

- model name/version
- dataset version
- predicted values
- confidence intervals

## Strategy Contracts

### `SimulationParticipant`

Represents a car/driver prior entering a race simulation:

- baseline pace prior
- pace variance
- degradation slope
- traffic sensitivity
- overtake skill
- starting compound and tyre age

### `PitStopPlan`

Defines a planned stop:

- fixed lap
- target compound
- optional flexible safety-car window

### `StrategyCandidate`

Defines a strategy for the target driver:

- name
- target driver
- pit-stop sequence
- SC/VSC adjustment flag

### `SimulationInput`

Defines the simulator payload:

- race key
- track profile
- participants
- strategy candidates
- optional weather samples
- Monte Carlo run count
- event probabilities
- target driver
- seed

### `StrategyResult` and `StrategyRecommendation`

These are the top-level decision outputs:

- expected race time
- expected finish position
- probability distribution fields
- pit window
- traffic/undercut/overcut indicators
- risks and explanation
- primary/backup recommendation pairing

