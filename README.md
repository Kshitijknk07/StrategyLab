# StrategyLab

StrategyLab is a backend-first Formula 1 strategy intelligence platform. It ingests historical race/session data, materializes a unified driver-lap dataset, trains local tabular models for baseline pace and tyre degradation, runs Monte Carlo race simulations, and serves strategy comparison and recommendation APIs.

This repository is intentionally **backend + ML only**. There is no frontend in V1.

## Current V1 Shape

The codebase implements a serious scaffold for a historical-only strategy platform:

- Layered storage for `raw`, `processed`, `features`, and `simulation_inputs`
- Typed contracts for race/session data, model outputs, and simulation requests
- Ingestion service with fixture support and FastF1/Jolpica adapters
- Unified feature dataset generation from canonical driver-lap records
- Local-only tabular training flow for:
  - baseline clean-air pace
  - tyre degradation / lap-time delta to baseline
- Monte Carlo lap-by-lap race simulator with:
  - tyre ageing
  - pit stop loss
  - traffic penalty
  - overtake probability
  - safety car / VSC sampling
  - pit-window flexibility under interruptions
- FastAPI endpoints for ingestion, model training/evaluation, simulation, and strategy recommendation
- CLI entrypoints for worker-style ingestion jobs and trainer workflows
- Pytest coverage with an end-to-end fixture-based flow

V1 is designed for **2018+ historical workflows**, with fixture-based testing and a clean path toward fuller FastF1-backed production ingestion.

## Architecture

Core modules:

- `src/strategylab/contracts.py`
  - canonical public contracts for race data, model I/O, and simulation payloads
- `src/strategylab/data/`
  - source adapters, normalization, feature engineering, layered storage, ingestion service
- `src/strategylab/models/`
  - local tabular model base, pace model, tyre degradation model, registry/catalog
- `src/strategylab/domain/`
  - race-state structures for simulation
- `src/strategylab/services/`
  - Monte Carlo simulator, optimizer, and recommendation layer
- `src/strategylab/apps/api/`
  - FastAPI application
- `src/strategylab/apps/worker/`
  - ingestion CLI
- `src/strategylab/apps/trainer/`
  - train/evaluate CLI
- `tests/`
  - fixture-driven ingestion, model, simulation, and API tests

Data flow:

1. Ingest a session from `fixture`, `fastf1`, or `jolpica`
2. Persist untouched source data to `raw`
3. Normalize processed artifacts into `processed`
4. Build the unified driver-lap feature dataset into `features`
5. Train local models against a versioned dataset
6. Simulate strategy candidates with Monte Carlo sampling
7. Expose ranked strategies and a primary/backup recommendation via API

## Local-Only ML Policy

StrategyLab does **not** use hosted ML or LLM APIs.

- Training is local
- Inference is local
- Model artifacts are stored locally
- Experiment/report metadata is stored locally

The current V1 stack uses classical tabular ML because the data is structured, temporal, and sparse enough that strong feature engineering beats prematurely complex deep learning.

## Storage Layers

StrategyLab keeps layers separated on purpose:

- `raw/`
  - source snapshots and source metadata
- `processed/`
  - normalized laps, stints, weather, timeline, track profile
- `features/`
  - versioned Parquet datasets for training
- `simulation_inputs/`
  - reproducible manifests used by the simulator and recommendation layer
- `manifests/`
  - dataset manifests and evaluation reports
- `artifacts/models/`
  - local trained model artifacts and metadata

## Unified Driver-Lap Dataset

The most important table in the system is the per-driver, per-lap dataset. It carries the structured state needed by both ML and simulation:

- race context: season, event, circuit, session
- competitor context: driver, constructor
- pace context: lap time, sector times, clean-air baseline, qualifying/team proxies
- tyre context: compound, tyre age, stint number
- race state: positions, gaps, pit flags, track status
- condition context: air temp, track temp, humidity, wet/dry
- engineered context: rolling pace, stint progress, gap features, position gain, pit flags

This is the contract that should grow as realism increases.

## API Surface

Main endpoints:

- `POST /ingestion/sessions:refresh`
- `GET /races/{season}/{event_name}/timeline`
- `GET /races/{season}/{event_name}/driver-laps`
- `GET /drivers/{driver}/stints`
- `GET /circuits/{circuit}/profile`
- `POST /models/{model_name}/train`
- `POST /models/{model_name}/evaluate`
- `POST /simulation/strategy`
- `POST /simulation/compare`
- `POST /simulation/monte-carlo`
- `POST /recommendations/strategy`

## Quick Start

### 1. Create the environment

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install '.[dev]'
cp .env.example .env
```

### 2. Run the tests

```bash
.venv/bin/pytest
```

### 3. Ingest a fixture session

```bash
.venv/bin/strategylab-worker ingest fixture 2024 Bahrain_GP sakhir --fixture-path tests/fixtures/bahrain_2024_fixture.json
```

### 4. Train local models

Use the dataset version returned by ingestion:

```bash
.venv/bin/strategylab-trainer train pace <dataset_version>
.venv/bin/strategylab-trainer train tyre <dataset_version>
```

### 5. Start the API

```bash
.venv/bin/strategylab-api
```

### 6. Example simulation request

```json
{
  "race_key": {
    "season": 2024,
    "event_name": "Bahrain_GP",
    "circuit": "sakhir",
    "session_type": "race"
  },
  "track_profile": {
    "circuit": "sakhir",
    "country": "Bahrain",
    "lap_length_km": 5.412,
    "total_laps": 57,
    "pit_lane_loss_seconds": 21.8,
    "overtaking_difficulty": 0.42,
    "degradation_score": 0.68,
    "drs_zones": 3,
    "typical_safety_car_rate": 0.28
  },
  "participants": [
    {
      "driver": "VER",
      "constructor": "Red Bull",
      "grid_position": 1,
      "baseline_pace_seconds": 94.8,
      "starting_compound": "soft"
    }
  ],
  "strategies": [
    {
      "name": "one_stop",
      "target_driver": "VER",
      "pit_stops": [
        {
          "lap": 18,
          "compound": "hard",
          "flexible_window_start": 16,
          "flexible_window_end": 20
        }
      ]
    }
  ],
  "monte_carlo_runs": 250,
  "target_driver": "VER",
  "seed": 42
}
```

## Docker

The repository includes a lean local Docker setup for API + Postgres + Redis. The current codebase primarily uses local filesystem storage, but the compose stack gives you the service shape needed for a more production-like local environment.

```bash
docker compose up --build
```

## Testing and Backtesting Philosophy

V1 evaluation is split by concern:

- data checks
  - schema validity, processed artifacts, feature columns
- model checks
  - MAE/RMSE for pace and tyre layers
- simulation checks
  - deterministic seeded Monte Carlo, ranked strategy output, recommendation shape
- API checks
  - ingestion, training, and strategy workflows

Backtesting against real historical races should be expanded next. The current fixture-driven tests verify reproducibility and interface behavior, not final motorsport-grade predictive accuracy.

## Current Realism Boundaries

What is already modeled:

- baseline pace prior per driver/car
- tyre degradation slope
- pit-lane loss and stationary noise
- traffic penalty
- overtake probability
- safety car / VSC timing uncertainty
- stochastic lap noise

What is intentionally deferred:

- live session ingestion and mid-race replanning
- richer weather transitions
- red flags
- full telemetry-derived traffic states
- calibrated track-specific overtaking maps
- reinforcement-learning style policy search

## Roadmap

### V1

- historical ingestion and typed storage layers
- unified driver-lap dataset
- local pace + tyre models
- Monte Carlo strategy comparison API
- primary/backup recommendation engine

### V2

- richer FastF1 extraction
- explicit traffic/overtake model training
- track-specific pit-loss priors
- better safety-car hazard modeling
- MLflow-backed experiment and registry depth

### V3

- live-race state updates
- dynamic contingency recommendations
- probabilistic weather regime shifts
- richer explainability and backtesting dashboards

## Supporting Docs

- [Architecture](docs/architecture.md)
- [Data Contracts](docs/data-contracts.md)
- [Model Cards](docs/model-cards.md)
- [Simulator Notes](docs/simulator.md)

