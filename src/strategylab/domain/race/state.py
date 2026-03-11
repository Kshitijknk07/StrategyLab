from __future__ import annotations

from dataclasses import dataclass, field

from strategylab.contracts import Compound, SimulationParticipant, StrategyCandidate, TrackProfile, WeatherSample


@dataclass(slots=True)
class DriverRaceState:
    driver: str
    constructor: str
    grid_position: int
    baseline_pace_seconds: float
    pace_variance_seconds: float
    degradation_per_lap: float
    traffic_sensitivity: float
    overtake_skill: float
    compound: Compound
    tyre_age_laps: int
    position: int
    cumulative_time_seconds: float = 0.0
    pit_count: int = 0
    completed_stops: list[int] = field(default_factory=list)
    current_stint_laps: int = 0
    traffic_loss_seconds: float = 0.0
    out_lap_penalty_seconds: float = 0.0
    last_lap_time_seconds: float = 0.0


@dataclass(slots=True)
class RaceSimulationContext:
    track: TrackProfile
    weather_by_lap: dict[int, WeatherSample]
    target_driver: str
    target_strategy: StrategyCandidate
    rival_strategies: dict[str, list[tuple[int, Compound]]]

    @classmethod
    def from_inputs(
        cls,
        track: TrackProfile,
        weather_samples: list[WeatherSample],
        target_driver: str,
        target_strategy: StrategyCandidate,
        participants: list[SimulationParticipant],
    ) -> RaceSimulationContext:
        weather_map = {sample.lap_number: sample for sample in weather_samples}
        rival_strategies = {
            participant.driver: _default_strategy(track.total_laps, participant.starting_compound)
            for participant in participants
            if participant.driver != target_driver
        }
        return cls(
            track=track,
            weather_by_lap=weather_map,
            target_driver=target_driver,
            target_strategy=target_strategy,
            rival_strategies=rival_strategies,
        )


def build_driver_states(participants: list[SimulationParticipant]) -> list[DriverRaceState]:
    ordered = sorted(participants, key=lambda item: item.grid_position)
    return [
        DriverRaceState(
            driver=participant.driver,
            constructor=participant.constructor,
            grid_position=participant.grid_position,
            baseline_pace_seconds=participant.baseline_pace_seconds,
            pace_variance_seconds=participant.pace_variance_seconds,
            degradation_per_lap=participant.degradation_per_lap,
            traffic_sensitivity=participant.traffic_sensitivity,
            overtake_skill=participant.overtake_skill,
            compound=participant.starting_compound,
            tyre_age_laps=participant.starting_tyre_age,
            position=index,
        )
        for index, participant in enumerate(ordered, start=1)
    ]


def _default_strategy(total_laps: int, starting_compound: Compound) -> list[tuple[int, Compound]]:
    if total_laps <= 40:
        pit_lap = max(total_laps // 2, 10)
    else:
        pit_lap = max(min(total_laps // 2, 25), 14)
    next_compound = Compound.HARD if starting_compound is not Compound.HARD else Compound.MEDIUM
    return [(pit_lap, next_compound)]
