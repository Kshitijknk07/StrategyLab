from __future__ import annotations

from fastapi.testclient import TestClient

from strategylab.contracts import (
    Compound,
    PitStopPlan,
    RaceKey,
    SessionType,
    SimulationInput,
    SimulationParticipant,
    StrategyCandidate,
    TrackProfile,
)


def test_api_ingestion_and_simulation_flow(api_app, fixture_path) -> None:
    client = TestClient(api_app)

    ingest_response = client.post(
        "/ingestion/sessions:refresh",
        json={
            "source": "fixture",
            "season": 2024,
            "event_name": "Bahrain_GP",
            "circuit": "sakhir",
            "fixture_path": str(fixture_path),
        },
    )
    assert ingest_response.status_code == 200
    dataset_version = ingest_response.json()["dataset_version"]

    timeline_response = client.get("/races/2024/Bahrain_GP/timeline")
    assert timeline_response.status_code == 200
    assert len(timeline_response.json()) >= 1

    train_response = client.post("/models/pace/train", json={"dataset_version": dataset_version})
    assert train_response.status_code == 200
    assert train_response.json()["model_name"] == "baseline_pace"

    simulation_payload = SimulationInput(
        race_key=RaceKey(season=2024, event_name="Bahrain_GP", circuit="sakhir", session_type=SessionType.RACE),
        track_profile=TrackProfile(
            circuit="sakhir",
            country="Bahrain",
            lap_length_km=5.412,
            total_laps=57,
            pit_lane_loss_seconds=21.8,
            overtaking_difficulty=0.42,
            degradation_score=0.68,
            drs_zones=3,
            typical_safety_car_rate=0.28,
        ),
        participants=[
            SimulationParticipant(
                driver="VER",
                constructor="Red Bull",
                grid_position=1,
                baseline_pace_seconds=94.8,
                degradation_per_lap=0.055,
                starting_compound=Compound.SOFT,
            ),
            SimulationParticipant(
                driver="NOR",
                constructor="McLaren",
                grid_position=2,
                baseline_pace_seconds=95.2,
                degradation_per_lap=0.060,
                starting_compound=Compound.SOFT,
            ),
            SimulationParticipant(
                driver="LEC",
                constructor="Ferrari",
                grid_position=3,
                baseline_pace_seconds=95.4,
                degradation_per_lap=0.058,
                starting_compound=Compound.SOFT,
            ),
        ],
        strategies=[
            StrategyCandidate(
                name="one_stop",
                target_driver="NOR",
                pit_stops=[PitStopPlan(lap=19, compound=Compound.HARD, flexible_window_start=17, flexible_window_end=21)],
            ),
            StrategyCandidate(
                name="two_stop",
                target_driver="NOR",
                pit_stops=[
                    PitStopPlan(lap=15, compound=Compound.MEDIUM, flexible_window_start=13, flexible_window_end=17),
                    PitStopPlan(lap=38, compound=Compound.HARD, flexible_window_start=36, flexible_window_end=40),
                ],
            ),
        ],
        monte_carlo_runs=40,
        target_driver="NOR",
        seed=5,
    ).model_dump(mode="json")

    recommendation_response = client.post("/recommendations/strategy", json=simulation_payload)
    assert recommendation_response.status_code == 200
    body = recommendation_response.json()
    assert body["primary_strategy"]["strategy_name"] in {"one_stop", "two_stop"}
    assert body["backup_strategy"]["strategy_name"] in {"one_stop", "two_stop"}
