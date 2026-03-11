from __future__ import annotations

import importlib
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from strategylab.contracts import (
    Compound,
    DriverLapRecord,
    EventType,
    PitStopPlan,
    RaceKey,
    RaceTimelineEvent,
    SimulationInput,
    SimulationParticipant,
    StintRecord,
    StrategyCandidate,
    TrackProfile,
    TrackStatus,
    WeatherSample,
)
from strategylab.data.ingestion.base import SessionBundle
from strategylab.infra.config import get_settings


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("STRATEGYLAB_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("STRATEGYLAB_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("STRATEGYLAB_DATASET_ROOT", str(tmp_path / "artifacts" / "datasets"))
    monkeypatch.setenv("STRATEGYLAB_MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
    monkeypatch.setenv("STRATEGYLAB_DEFAULT_MONTE_CARLO_RUNS", "40")
    monkeypatch.setenv("STRATEGYLAB_RANDOM_SEED", "11")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def race_key() -> RaceKey:
    return RaceKey(season=2024, event_name="italian_gp", circuit="monza")


@pytest.fixture
def track_profile() -> TrackProfile:
    return TrackProfile(
        circuit="monza",
        country="Italy",
        lap_length_km=5.793,
        total_laps=18,
        pit_lane_loss_seconds=20.5,
        overtaking_difficulty=0.35,
        degradation_score=0.45,
        drs_zones=2,
        typical_safety_car_rate=0.28,
    )


@pytest.fixture
def sample_bundle(race_key: RaceKey, track_profile: TrackProfile) -> SessionBundle:
    drivers = [
        ("VER", "Red Bull", 89.8, 1, 0.042),
        ("HAM", "Mercedes", 90.1, 2, 0.045),
        ("NOR", "McLaren", 90.3, 3, 0.048),
    ]
    laps: list[DriverLapRecord] = []
    stints: list[StintRecord] = []
    weather_samples: list[WeatherSample] = []
    timeline: list[RaceTimelineEvent] = []
    start_time = datetime(2024, 9, 1, 13, 0, 0)

    for lap_number in range(1, 19):
        if lap_number in {1, 6, 12, 18}:
            weather_samples.append(
                WeatherSample(
                    race_key=race_key,
                    lap_number=lap_number,
                    timestamp_utc=start_time + timedelta(minutes=lap_number * 2),
                    air_temp_c=28.0,
                    track_temp_c=39.0 - (lap_number * 0.2),
                    humidity_pct=42.0,
                    rain_intensity=0.0,
                    wind_speed_kph=7.0,
                )
            )
        if lap_number == 9:
            timeline.append(
                RaceTimelineEvent(
                    race_key=race_key,
                    lap_number=lap_number,
                    event_type=EventType.TRACK_STATUS,
                    message="Virtual safety car deployed",
                    track_status=TrackStatus.VSC,
                )
            )

        for driver, constructor, baseline, grid, degradation in drivers:
            first_stint = lap_number <= 9
            tyre_age = lap_number - 1 if first_stint else lap_number - 10
            compound = Compound.MEDIUM if first_stint else Compound.HARD
            fuel_proxy = max(0.2, 1.0 - (lap_number / track_profile.total_laps))
            base_clean_air = baseline + (fuel_proxy * 0.8)
            lap_time = base_clean_air + degradation * tyre_age * (1.0 + track_profile.degradation_score)
            if driver == "HAM" and lap_number <= 4:
                lap_time += 0.12
            track_status = TrackStatus.VSC if lap_number == 9 else TrackStatus.GREEN
            laps.append(
                DriverLapRecord(
                    race_key=race_key,
                    driver=driver,
                    constructor=constructor,
                    lap_number=lap_number,
                    lap_time_seconds=lap_time,
                    sector_1_seconds=lap_time * 0.31,
                    sector_2_seconds=lap_time * 0.34,
                    sector_3_seconds=lap_time * 0.35,
                    compound=compound,
                    tyre_age_laps=tyre_age,
                    stint_number=1 if first_stint else 2,
                    pit_in=lap_number == 9,
                    pit_out=lap_number == 10,
                    track_status=track_status,
                    air_temp_c=28.0,
                    track_temp_c=39.0 - (lap_number * 0.2),
                    humidity_pct=42.0,
                    wet_track=False,
                    position_start=grid,
                    position_end=grid,
                    gap_to_ahead_seconds=0.0 if grid == 1 else (1.2 if grid == 2 else 1.9),
                    gap_to_leader_seconds=0.0 if grid == 1 else (3.1 if grid == 2 else 5.8),
                    drs_available=grid > 1,
                    traffic_density=0.15 if grid > 1 else 0.0,
                    qualifying_pace_proxy=-0.12 * grid,
                    team_strength_proxy=1.0 - (grid * 0.05),
                    circuit_degradation_class=track_profile.degradation_score,
                    overtaking_difficulty_score=track_profile.overtaking_difficulty,
                    fuel_load_proxy=fuel_proxy,
                    clean_air_baseline_seconds=base_clean_air,
                )
            )

    for driver, _, _, _, _ in drivers:
        stints.append(
            StintRecord(
                race_key=race_key,
                driver=driver,
                stint_number=1,
                compound=Compound.MEDIUM,
                start_lap=1,
                end_lap=9,
                laps=9,
            )
        )
        stints.append(
            StintRecord(
                race_key=race_key,
                driver=driver,
                stint_number=2,
                compound=Compound.HARD,
                start_lap=10,
                end_lap=18,
                laps=9,
            )
        )

    return SessionBundle(
        race_key=race_key,
        track_profile=track_profile,
        laps=laps,
        stints=stints,
        weather_samples=weather_samples,
        timeline=timeline,
        source_metadata={"source": "fixture"},
    )


@pytest.fixture
def simulation_input(race_key: RaceKey, track_profile: TrackProfile) -> SimulationInput:
    return SimulationInput(
        race_key=race_key,
        track_profile=track_profile,
        target_driver="HAM",
        monte_carlo_runs=40,
        seed=5,
        participants=[
            SimulationParticipant(
                driver="VER",
                constructor="Red Bull",
                grid_position=1,
                baseline_pace_seconds=89.8,
                degradation_per_lap=0.042,
                pace_variance_seconds=0.18,
                traffic_sensitivity=0.35,
                overtake_skill=0.7,
                starting_compound=Compound.MEDIUM,
            ),
            SimulationParticipant(
                driver="HAM",
                constructor="Mercedes",
                grid_position=2,
                baseline_pace_seconds=90.1,
                degradation_per_lap=0.047,
                pace_variance_seconds=0.2,
                traffic_sensitivity=0.45,
                overtake_skill=0.62,
                starting_compound=Compound.MEDIUM,
            ),
            SimulationParticipant(
                driver="NOR",
                constructor="McLaren",
                grid_position=3,
                baseline_pace_seconds=90.3,
                degradation_per_lap=0.049,
                pace_variance_seconds=0.19,
                traffic_sensitivity=0.5,
                overtake_skill=0.6,
                starting_compound=Compound.SOFT,
            ),
        ],
        strategies=[
            StrategyCandidate(
                name="medium-hard_early",
                target_driver="HAM",
                pit_stops=[PitStopPlan(lap=8, compound=Compound.HARD, flexible_window_start=7, flexible_window_end=10)],
            ),
            StrategyCandidate(
                name="medium-hard_late",
                target_driver="HAM",
                pit_stops=[
                    PitStopPlan(
                        lap=11,
                        compound=Compound.HARD,
                        flexible_window_start=10,
                        flexible_window_end=13,
                    )
                ],
            ),
        ],
    )


@pytest.fixture
def api_module():
    from strategylab.apps.api import main as api_main

    return importlib.reload(api_main)


@pytest.fixture
def fixture_path() -> Path:
    return Path(__file__).parent / "fixtures" / "bahrain_2024_fixture.json"
