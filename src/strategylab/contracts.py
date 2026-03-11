from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, computed_field


class SessionType(StrEnum):
    RACE = "race"
    QUALIFYING = "qualifying"
    PRACTICE = "practice"
    SPRINT = "sprint"


class TrackStatus(StrEnum):
    GREEN = "green"
    VSC = "virtual_safety_car"
    SAFETY_CAR = "safety_car"
    RED_FLAG = "red_flag"


class Compound(StrEnum):
    SOFT = "soft"
    MEDIUM = "medium"
    HARD = "hard"
    INTERMEDIATE = "intermediate"
    WET = "wet"


class DatasetLayer(StrEnum):
    RAW = "raw"
    PROCESSED = "processed"
    FEATURES = "features"
    SIMULATION_INPUTS = "simulation_inputs"


class EventType(StrEnum):
    LAP_COMPLETION = "lap_completion"
    PIT_IN = "pit_in"
    PIT_OUT = "pit_out"
    STINT_START = "stint_start"
    STINT_END = "stint_end"
    WEATHER = "weather"
    TRACK_STATUS = "track_status"
    RACE_CONTROL = "race_control"


class RaceKey(BaseModel):
    season: int = Field(ge=1950)
    event_name: str
    circuit: str
    session_type: SessionType = SessionType.RACE

    @computed_field
    @property
    def slug(self) -> str:
        event = self.event_name.lower().replace(" ", "_")
        circuit = self.circuit.lower().replace(" ", "_")
        return f"{self.season}_{event}_{circuit}_{self.session_type.value}"


class PredictionInterval(BaseModel):
    low: float
    high: float
    confidence: float = Field(default=0.95, gt=0.0, le=1.0)


class DatasetManifest(BaseModel):
    dataset_version: str
    layer: DatasetLayer
    description: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    record_count: int = Field(default=0, ge=0)
    source_sessions: list[RaceKey] = Field(default_factory=list)
    feature_columns: list[str] = Field(default_factory=list)
    target_columns: list[str] = Field(default_factory=list)
    file_path: str


class EvaluationMetric(BaseModel):
    name: str
    value: float


class EvaluationReport(BaseModel):
    model_name: str
    model_version: str
    dataset_version: str
    metrics: list[EvaluationMetric]
    notes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ModelPrediction(BaseModel):
    model_name: str
    model_version: str
    dataset_version: str
    values: list[float]
    intervals: list[PredictionInterval] = Field(default_factory=list)
    feature_window_start: datetime | None = None
    feature_window_end: datetime | None = None


class TrackProfile(BaseModel):
    circuit: str
    country: str | None = None
    lap_length_km: float = Field(gt=0.0)
    total_laps: int = Field(gt=0)
    pit_lane_loss_seconds: float = Field(gt=0.0)
    overtaking_difficulty: float = Field(default=0.5, ge=0.0, le=1.0)
    degradation_score: float = Field(default=0.5, ge=0.0, le=1.0)
    drs_zones: int = Field(default=1, ge=0)
    typical_safety_car_rate: float = Field(default=0.3, ge=0.0, le=1.0)


class WeatherSample(BaseModel):
    race_key: RaceKey
    lap_number: int = Field(ge=1)
    timestamp_utc: datetime
    air_temp_c: float
    track_temp_c: float
    humidity_pct: float = Field(ge=0.0, le=100.0)
    rain_intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    wind_speed_kph: float = Field(default=0.0, ge=0.0)


class PitEvent(BaseModel):
    race_key: RaceKey
    driver: str
    lap_number: int = Field(ge=1)
    event_type: Literal["pit_in", "pit_out"]
    stationary_seconds: float | None = Field(default=None, ge=0.0)
    lane_loss_seconds: float | None = Field(default=None, ge=0.0)
    compound_after: Compound | None = None


class StintRecord(BaseModel):
    race_key: RaceKey
    driver: str
    stint_number: int = Field(ge=1)
    compound: Compound
    start_lap: int = Field(ge=1)
    end_lap: int = Field(ge=1)
    laps: int = Field(ge=1)


class RaceTimelineEvent(BaseModel):
    race_key: RaceKey
    lap_number: int = Field(ge=1)
    event_type: EventType
    message: str
    track_status: TrackStatus | None = None
    driver: str | None = None
    payload: dict[str, str | int | float | None] = Field(default_factory=dict)


class DriverLapRecord(BaseModel):
    race_key: RaceKey
    driver: str
    constructor: str
    lap_number: int = Field(ge=1)
    lap_time_seconds: float = Field(gt=0.0)
    sector_1_seconds: float | None = Field(default=None, gt=0.0)
    sector_2_seconds: float | None = Field(default=None, gt=0.0)
    sector_3_seconds: float | None = Field(default=None, gt=0.0)
    compound: Compound
    tyre_age_laps: int = Field(ge=0)
    stint_number: int = Field(ge=1)
    pit_in: bool = False
    pit_out: bool = False
    track_status: TrackStatus = TrackStatus.GREEN
    air_temp_c: float = 25.0
    track_temp_c: float = 35.0
    humidity_pct: float = 45.0
    wet_track: bool = False
    position_start: int = Field(ge=1)
    position_end: int = Field(ge=1)
    gap_to_ahead_seconds: float | None = Field(default=None, ge=0.0)
    gap_to_leader_seconds: float | None = Field(default=None, ge=0.0)
    drs_available: bool = False
    traffic_density: float = Field(default=0.0, ge=0.0, le=1.0)
    qualifying_pace_proxy: float = Field(default=0.0)
    team_strength_proxy: float = Field(default=0.0)
    circuit_degradation_class: float = Field(default=0.5, ge=0.0, le=1.0)
    overtaking_difficulty_score: float = Field(default=0.5, ge=0.0, le=1.0)
    fuel_load_proxy: float = Field(default=1.0, ge=0.0)
    clean_air_baseline_seconds: float | None = Field(default=None, gt=0.0)


class SimulationParticipant(BaseModel):
    driver: str
    constructor: str
    grid_position: int = Field(ge=1)
    baseline_pace_seconds: float = Field(gt=0.0)
    pace_variance_seconds: float = Field(default=0.25, ge=0.0)
    degradation_per_lap: float = Field(default=0.05, ge=0.0)
    traffic_sensitivity: float = Field(default=0.5, ge=0.0, le=1.0)
    overtake_skill: float = Field(default=0.5, ge=0.0, le=1.0)
    starting_compound: Compound
    starting_tyre_age: int = Field(default=0, ge=0)
    default_pit_loss_seconds: float | None = Field(default=None, ge=0.0)


class PitStopPlan(BaseModel):
    lap: int = Field(ge=1)
    compound: Compound
    flexible_window_start: int | None = Field(default=None, ge=1)
    flexible_window_end: int | None = Field(default=None, ge=1)


class StrategyCandidate(BaseModel):
    name: str
    target_driver: str
    pit_stops: list[PitStopPlan]
    allow_sc_adjustment: bool = True
    notes: list[str] = Field(default_factory=list)


class StrategyResult(BaseModel):
    strategy_name: str
    expected_race_time_seconds: float
    expected_finish_position: float
    win_probability: float = Field(ge=0.0, le=1.0)
    podium_probability: float = Field(ge=0.0, le=1.0)
    top_ten_probability: float = Field(ge=0.0, le=1.0)
    finish_distribution: dict[str, float]
    recommended_pit_window: PredictionInterval
    traffic_risk: float = Field(ge=0.0, le=1.0)
    undercut_gain_seconds: float
    overcut_gain_seconds: float
    key_risks: list[str] = Field(default_factory=list)
    explanation: list[str] = Field(default_factory=list)


class StrategyRecommendation(BaseModel):
    race_key: RaceKey
    primary_strategy: StrategyResult
    backup_strategy: StrategyResult
    assumptions: list[str] = Field(default_factory=list)
    confidence_notes: list[str] = Field(default_factory=list)


class SimulationInput(BaseModel):
    race_key: RaceKey
    track_profile: TrackProfile
    participants: list[SimulationParticipant]
    strategies: list[StrategyCandidate]
    weather_samples: list[WeatherSample] = Field(default_factory=list)
    monte_carlo_runs: int = Field(default=500, ge=10, le=10000)
    target_driver: str
    safety_car_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    vsc_probability: float = Field(default=0.2, ge=0.0, le=1.0)
    seed: int | None = None


class IngestionRefreshRequest(BaseModel):
    source: Literal["fixture", "fastf1", "jolpica"]
    season: int = Field(ge=1950)
    event_name: str
    circuit: str
    fixture_path: str | None = None


class TrainRequest(BaseModel):
    dataset_version: str
    target_column: str | None = None
    model_config: dict[str, str | float | int | bool] = Field(default_factory=dict)


class ModelEvaluateRequest(BaseModel):
    dataset_version: str
    model_version: str

