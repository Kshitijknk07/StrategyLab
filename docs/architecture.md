# Architecture

StrategyLab is built as a backend-first racing intelligence stack with clean boundaries between data, ML, simulation, and serving.

## Subsystems

- `contracts`
  - stable typed interfaces for datasets, predictions, and strategy requests
- `data`
  - ingestion adapters, normalization, layered storage, feature materialization
- `models`
  - local tabular training/evaluation/prediction
- `domain`
  - race-state primitives
- `services`
  - simulator, optimizer, recommender
- `apps`
  - API, worker, trainer entrypoints

## Request Path

1. Ingestion receives a fixture or source request.
2. Source client returns a canonical session bundle.
3. Processors normalize laps, stints, weather, timeline, and track profile.
4. Feature store builds the unified driver-lap dataset and writes a manifest.
5. Local models train against a dataset version and register artifacts.
6. Simulation receives race-state priors and candidate strategies.
7. Optimizer ranks candidates through Monte Carlo.
8. Recommender returns primary and backup strategies plus assumptions and confidence notes.

## Storage Model

- Filesystem-backed storage is the current default.
- Parquet is used for processed/features layers.
- JSON is used for manifests and source payloads.
- Docker services for Postgres and Redis are provided for production-style expansion, but are not yet the primary persistence path in V1.

## Why This Shape

- It keeps raw data isolated from logic.
- It lets models and simulator share one canonical driver-lap contract.
- It supports reproducibility through dataset and model versioning.
- It preserves a clear path toward more realistic event modeling without a rewrite.

