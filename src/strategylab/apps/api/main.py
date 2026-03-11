from __future__ import annotations

from typing import Any

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException

from strategylab.contracts import IngestionRefreshRequest, ModelEvaluateRequest, SimulationInput, TrainRequest
from strategylab.data.service import IngestionService
from strategylab.infra.config import get_settings
from strategylab.infra.logging import configure_logging
from strategylab.models.catalog import get_model
from strategylab.services.optimizer import StrategyOptimizer
from strategylab.services.recommender import StrategyRecommender

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(
    title="StrategyLab API",
    version="0.1.0",
    description="Backend-first Formula 1 strategy intelligence platform.",
)

ingestion_service = IngestionService()
optimizer = StrategyOptimizer()
recommender = StrategyRecommender(optimizer)


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok", "env": settings.env}


@app.post("/ingestion/sessions:refresh")
async def refresh_session(request: IngestionRefreshRequest) -> dict[str, str]:
    try:
        return ingestion_service.refresh(request)
    except Exception as exc:  # pragma: no cover - FastAPI error translation
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/races/{season}/{event_name}/timeline")
async def race_timeline(season: int, event_name: str) -> list[dict[str, Any]]:
    return _read_latest_processed("timeline", season, event_name)


@app.get("/races/{season}/{event_name}/driver-laps")
async def driver_laps(season: int, event_name: str) -> list[dict[str, Any]]:
    return _read_latest_processed("laps", season, event_name)


@app.get("/drivers/{driver}/stints")
async def driver_stints(driver: str) -> list[dict[str, Any]]:
    rows = _read_latest_processed("stints")
    return [row for row in rows if row.get("driver") == driver]


@app.get("/circuits/{circuit}/profile")
async def circuit_profile(circuit: str) -> dict[str, Any]:
    rows = _read_latest_processed("track_profile")
    for row in rows:
        if row.get("circuit") == circuit:
            return row
    raise HTTPException(status_code=404, detail=f"Circuit not found: {circuit}")


@app.post("/models/{model_name}/train")
async def train_model(model_name: str, request: TrainRequest) -> dict[str, Any]:
    try:
        model = get_model(model_name)
        return model.train(request.dataset_version, request.target_column, request.training_config)
    except Exception as exc:  # pragma: no cover - FastAPI error translation
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/models/{model_name}/evaluate")
async def evaluate_model(model_name: str, request: ModelEvaluateRequest) -> dict[str, Any]:
    try:
        model = get_model(model_name)
        report = model.evaluate(request.dataset_version, request.model_version)
        return report.model_dump(mode="json")
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/simulation/strategy")
async def simulate_strategy(simulation_input: SimulationInput) -> dict[str, Any]:
    ranked = optimizer.compare(simulation_input)
    if not ranked:
        raise HTTPException(status_code=400, detail="No strategies supplied.")
    return ranked[0].model_dump(mode="json")


@app.post("/simulation/compare")
async def compare_strategies(simulation_input: SimulationInput) -> list[dict[str, Any]]:
    return [result.model_dump(mode="json") for result in optimizer.compare(simulation_input)]


@app.post("/simulation/monte-carlo")
async def monte_carlo_forecast(simulation_input: SimulationInput) -> dict[str, Any]:
    results = optimizer.compare(simulation_input)
    return {
        "race_key": simulation_input.race_key.model_dump(mode="json"),
        "runs": simulation_input.monte_carlo_runs,
        "results": [result.model_dump(mode="json") for result in results],
    }


@app.post("/recommendations/strategy")
async def recommend_strategy(simulation_input: SimulationInput) -> dict[str, Any]:
    try:
        recommendation = recommender.recommend(simulation_input)
        return recommendation.model_dump(mode="json")
    except Exception as exc:  # pragma: no cover - FastAPI error translation
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def run() -> None:
    uvicorn.run("strategylab.apps.api.main:app", host="0.0.0.0", port=8000, reload=False)


def _read_latest_processed(
    table_name: str,
    season: int | None = None,
    event_name: str | None = None,
) -> list[dict[str, Any]]:
    files = sorted(settings.processed_path.glob(f"{table_name}_*.parquet"))
    if not files:
        raise HTTPException(status_code=404, detail=f"No processed data found for {table_name}")
    frame = pd.read_parquet(files[-1])
    if season is not None and "season" in frame.columns:
        frame = frame[frame["season"] == season]
    if event_name is not None and "event_name" in frame.columns:
        frame = frame[frame["event_name"] == event_name]
    if frame.empty:
        raise HTTPException(status_code=404, detail="No matching records found.")
    return frame.to_dict(orient="records")
