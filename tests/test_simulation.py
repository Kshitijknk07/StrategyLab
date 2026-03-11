from __future__ import annotations

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
from strategylab.services.optimizer import StrategyOptimizer
from strategylab.services.recommender import StrategyRecommender


def test_strategy_optimizer_and_recommender() -> None:
    race_key = RaceKey(season=2024, event_name="Bahrain_GP", circuit="sakhir", session_type=SessionType.RACE)
    track = TrackProfile(
        circuit="sakhir",
        country="Bahrain",
        lap_length_km=5.412,
        total_laps=57,
        pit_lane_loss_seconds=21.8,
        overtaking_difficulty=0.42,
        degradation_score=0.68,
        drs_zones=3,
        typical_safety_car_rate=0.28,
    )
    participants = [
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
            degradation_per_lap=0.06,
            starting_compound=Compound.SOFT,
        ),
        SimulationParticipant(
            driver="LEC",
            constructor="Ferrari",
            grid_position=3,
            baseline_pace_seconds=95.3,
            degradation_per_lap=0.058,
            starting_compound=Compound.SOFT,
        ),
    ]
    strategies = [
        StrategyCandidate(
            name="one_stop",
            target_driver="NOR",
            pit_stops=[PitStopPlan(lap=18, compound=Compound.HARD, flexible_window_start=16, flexible_window_end=20)],
        ),
        StrategyCandidate(
            name="two_stop",
            target_driver="NOR",
            pit_stops=[
                PitStopPlan(lap=14, compound=Compound.MEDIUM, flexible_window_start=12, flexible_window_end=16),
                PitStopPlan(lap=36, compound=Compound.HARD, flexible_window_start=34, flexible_window_end=38),
            ],
        ),
    ]
    simulation_input = SimulationInput(
        race_key=race_key,
        track_profile=track,
        participants=participants,
        strategies=strategies,
        monte_carlo_runs=50,
        target_driver="NOR",
        seed=11,
    )

    ranked = StrategyOptimizer().compare(simulation_input)
    assert len(ranked) == 2
    assert ranked[0].expected_finish_position <= ranked[1].expected_finish_position

    recommendation = StrategyRecommender().recommend(simulation_input)
    assert recommendation.primary_strategy.strategy_name in {"one_stop", "two_stop"}
    assert recommendation.backup_strategy.strategy_name in {"one_stop", "two_stop"}
    assert recommendation.primary_strategy.strategy_name != recommendation.backup_strategy.strategy_name

