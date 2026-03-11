"""Canonical schemas shared across ingestion, models, and APIs."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DataLayer(str, Enum):
    RAW = "raw"
    PROCESSED = "processed"
    FEATURES = "features"
    SIMULATION_INPUTS = "simulation_inputs"


class SessionType(str, Enum):
    RACE = "R"
    QUALIFYING = "Q"
    PRACTICE = "P"
    SPRINT = "S"


class TrackStatus(str, Enum):
    GREEN = "green"
    VSC = "vsc"
    SAFETY_CAR = "safety_car"


class RaceSessionKey(BaseModel):
    season: int
    event_slug: str
    circuit: str
    session_type: SessionType = SessionType.RACE


class DriverLapRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    race: RaceSessionKey
    driver: str
    constructor: str
    lap_number: int
    lap_time_seconds: float
    sector_1_seconds: float | None = None
    sector_2_seconds: float | None = None
    sector_3_seconds: float | None = None
    tyre_compound: str
    tyre_age_laps: int
    stint_number: int
    pit_in: bool = False
    pit_out: bool = False
    track_temperature_c: float | None = None
    air_temperature_c: float | None = None
    humidity_pct: float | None = None
    is_wet: bool = False
    track_status: TrackStatus = TrackStatus.GREEN
    position_start: int
    position_end: int
    gap_to_car_ahead_seconds: float | None = None
    gap_to_leader_seconds: float | None = None
    drs_available_proxy: float = 0.0
    traffic_density_proxy: float = 0.0
    qualifying_pace_proxy: float = 0.0
    team_strength_proxy: float = 0.0
    circuit_degradation_class: float = 0.0
    overtaking_difficulty_score: float = 0.0


class StintRecord(BaseModel):
    race: RaceSessionKey
    driver: str
    stint_number: int
    compound: str
    start_lap: int
    end_lap: int
    laps: int


class PitEvent(BaseModel):
    race: RaceSessionKey
    driver: str
    lap_number: int
    stationary_time_seconds: float | None = None
    lane_time_loss_seconds: float | None = None
    compound_out: str | None = None


class WeatherSample(BaseModel):
    race: RaceSessionKey
    lap_number: int | None = None
    timestamp: datetime | None = None
    air_temperature_c: float | None = None
    track_temperature_c: float | None = None
    humidity_pct: float | None = None
    rain_intensity: float = 0.0


class RaceTimelineEvent(BaseModel):
    race: RaceSessionKey
    lap_number: int | None = None
    timestamp: datetime | None = None
    category: str
    message: str
    track_status: TrackStatus | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrackProfile(BaseModel):
    circuit: str
    country: str | None = None
    length_km: float
    total_laps: int
    pit_lane_loss_seconds: float
    degradation_score: float = 0.5
    overtaking_difficulty: float = 0.5
    safety_car_risk: float = 0.25
    drs_zones: int = 1


class PredictionInterval(BaseModel):
    lower: float
    upper: float
    confidence: float = 0.95


class ModelPrediction(BaseModel):
    value: float
    interval: PredictionInterval | None = None
    model_version: str
    dataset_version: str
    feature_version: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    model_name: str
    model_version: str
    dataset_version: str
    metrics: dict[str, float]
    generated_at: datetime
    slices: dict[str, dict[str, float]] = Field(default_factory=dict)


class SimulationParticipant(BaseModel):
    driver: str
    constructor: str
    grid_position: int
    starting_compound: str
    baseline_pace_seconds: float
    degradation_per_lap_seconds: float
    tyre_age_laps: int = 0
    pit_crew_delta_seconds: float = 0.0
    traffic_sensitivity: float = 0.5
    overtake_skill: float = 0.5
    qualifying_delta_seconds: float = 0.0
    incumbent_strategy: str = "one_stop"


class PitStopPlan(BaseModel):
    lap: int
    compound_out: str


class StrategyCandidate(BaseModel):
    name: str
    target_driver: str
    starting_compound: str | None = None
    pit_plan: list[PitStopPlan]
    allow_safety_car_opportunism: bool = True
    window_slack_laps: int = 2


class SimulationAssumptions(BaseModel):
    monte_carlo_runs: int = 500
    pace_noise_sigma_seconds: float = 0.35
    pit_stop_sigma_seconds: float = 0.4
    safety_car_window_probability: float = 0.25
    weather_shift_probability: float = 0.05
    seed: int = 42


class SimulationInput(BaseModel):
    race: RaceSessionKey
    track: TrackProfile
    target_driver: str
    participants: list[SimulationParticipant]
    candidate_strategies: list[StrategyCandidate]
    weather_baseline: Literal["dry", "mixed", "wet"] = "dry"
    assumptions: SimulationAssumptions = Field(default_factory=SimulationAssumptions)


class FinishingDistribution(BaseModel):
    mean_finish_position: float
    podium_probability: float
    win_probability: float
    top_10_probability: float


class StrategySimulationResult(BaseModel):
    strategy_name: str
    target_driver: str
    expected_race_time_seconds: float
    expected_finish_position: float
    distribution: FinishingDistribution
    recommended_pit_window: tuple[int, int]
    traffic_risk_score: float
    safety_car_dependency: float
    explanation: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


class StrategyRecommendation(BaseModel):
    race: RaceSessionKey
    primary_strategy: StrategySimulationResult
    backup_strategy: StrategySimulationResult
    key_risks: list[str]
    assumptions: list[str]

