from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from math import exp
from statistics import mean, pstdev

import numpy as np

from strategylab.contracts import (
    Compound,
    PredictionInterval,
    SimulationInput,
    StrategyCandidate,
    StrategyResult,
    TrackStatus,
)
from strategylab.domain.race.state import RaceSimulationContext, build_driver_states


class RaceSimulator:
    def compare_strategies(self, simulation_input: SimulationInput) -> list[StrategyResult]:
        rng = np.random.default_rng(simulation_input.seed)
        results: list[StrategyResult] = []
        for candidate in simulation_input.strategies:
            runs = [
                self._run_single(simulation_input, candidate, int(rng.integers(0, 2**31 - 1)))
                for _ in range(simulation_input.monte_carlo_runs)
            ]
            results.append(self._aggregate(candidate, runs, simulation_input.track_profile.total_laps))
        return sorted(results, key=lambda item: (item.expected_finish_position, item.expected_race_time_seconds))

    def _run_single(
        self,
        simulation_input: SimulationInput,
        target_strategy: StrategyCandidate,
        seed: int,
    ) -> dict[str, float | list[int] | Counter[int]]:
        rng = np.random.default_rng(seed)
        context = RaceSimulationContext.from_inputs(
            track=simulation_input.track_profile,
            weather_samples=simulation_input.weather_samples,
            target_driver=simulation_input.target_driver,
            target_strategy=target_strategy,
            participants=simulation_input.participants,
        )
        states = build_driver_states(simulation_input.participants)
        scheduled_events = self._sample_track_events(
            simulation_input.track_profile.total_laps,
            simulation_input.safety_car_probability * simulation_input.track_profile.typical_safety_car_rate,
            simulation_input.vsc_probability,
            rng,
        )
        target_pits: list[int] = []
        for lap in range(1, simulation_input.track_profile.total_laps + 1):
            states.sort(key=lambda item: (item.cumulative_time_seconds, item.grid_position))
            for position, state in enumerate(states, start=1):
                state.position = position

            lap_status = scheduled_events.get(lap, TrackStatus.GREEN)
            gaps = _compute_gaps(states)
            for state in states:
                weather = context.weather_by_lap.get(lap)
                pit_this_lap, next_compound = self._pit_decision(state, lap, lap_status, context)
                lap_time = self._lap_time(
                    state=state,
                    gap_to_ahead=gaps.get(state.driver, 99.0),
                    track_status=lap_status,
                    weather=weather.rain_intensity if weather else 0.0,
                    rng=rng,
                    track=context.track,
                )
                if pit_this_lap:
                    lap_time += self._pit_loss(context.track.pit_lane_loss_seconds, lap_status, rng)
                    state.tyre_age_laps = 0
                    state.current_stint_laps = 0
                    state.compound = next_compound
                    state.pit_count += 1
                    state.completed_stops.append(lap)
                    state.out_lap_penalty_seconds = 0.9
                    if state.driver == context.target_driver:
                        target_pits.append(lap)
                state.cumulative_time_seconds += lap_time
                state.last_lap_time_seconds = lap_time
                state.tyre_age_laps += 1
                state.current_stint_laps += 1
            self._resolve_overtakes(states, context.track.overtaking_difficulty, rng)
        states.sort(key=lambda item: (item.cumulative_time_seconds, item.grid_position))
        finish_distribution = Counter({position: 1 for position, _ in enumerate(states, start=1)})
        target = next(state for state in states if state.driver == context.target_driver)
        target_index = next(index for index, state in enumerate(states, start=1) if state.driver == target.driver)
        return {
            "race_time": target.cumulative_time_seconds,
            "finish_position": float(target_index),
            "pit_laps": target_pits,
            "traffic_loss": target.traffic_loss_seconds,
            "finish_distribution": finish_distribution,
        }

    def _aggregate(
        self,
        candidate: StrategyCandidate,
        runs: Iterable[dict[str, float | list[int] | Counter[int]]],
        total_laps: int,
    ) -> StrategyResult:
        runs_list = list(runs)
        race_times = [float(item["race_time"]) for item in runs_list]
        finish_positions = [float(item["finish_position"]) for item in runs_list]
        traffic_losses = [float(item["traffic_loss"]) for item in runs_list]
        pit_laps = [item["pit_laps"][0] for item in runs_list if item["pit_laps"]]
        pit_mean = mean(pit_laps) if pit_laps else max(total_laps / 2, 1.0)
        pit_std = pstdev(pit_laps) if len(pit_laps) > 1 else 1.0
        distribution_counter: Counter[int] = Counter()
        for item in runs_list:
            distribution_counter.update(item["finish_distribution"])
        total_runs = len(runs_list)
        finish_distribution = {
            f"P{position}": count / total_runs for position, count in sorted(distribution_counter.items())
        }
        win_probability = sum(1 for pos in finish_positions if pos <= 1.0) / total_runs
        podium_probability = sum(1 for pos in finish_positions if pos <= 3.0) / total_runs
        top_ten_probability = sum(1 for pos in finish_positions if pos <= 10.0) / total_runs
        undercut_gain = _undercut_gain(candidate)
        overcut_gain = max(0.0, 0.6 - undercut_gain / 2.0)
        traffic_risk = min(1.0, float(mean(traffic_losses) / 8.0))
        explanation = [
            f"Average race time {mean(race_times):.2f}s across {total_runs} Monte Carlo runs.",
            f"Average finish position {mean(finish_positions):.2f} with traffic risk {traffic_risk:.2f}.",
        ]
        if candidate.pit_stops:
            explanation.append(
                f"First stop centers on lap {pit_mean:.1f} with about +/- {max(pit_std, 1.0):.1f} laps of spread."
            )
        key_risks = []
        if traffic_risk > 0.55:
            key_risks.append("High rejoin traffic risk could erase the tyre offset.")
        if pit_std > 4.0:
            key_risks.append("Pit window is sensitive to race interruptions and pace variance.")
        if not key_risks:
            key_risks.append("Recommendation remains sensitive to clean-air assumptions and event timing.")
        return StrategyResult(
            strategy_name=candidate.name,
            expected_race_time_seconds=float(mean(race_times)),
            expected_finish_position=float(mean(finish_positions)),
            win_probability=win_probability,
            podium_probability=podium_probability,
            top_ten_probability=top_ten_probability,
            finish_distribution=finish_distribution,
            recommended_pit_window=PredictionInterval(
                low=max(1.0, pit_mean - max(1.0, pit_std)),
                high=pit_mean + max(1.0, pit_std),
                confidence=0.68,
            ),
            traffic_risk=traffic_risk,
            undercut_gain_seconds=undercut_gain,
            overcut_gain_seconds=overcut_gain,
            key_risks=key_risks,
            explanation=explanation,
        )

    def _sample_track_events(
        self,
        total_laps: int,
        safety_car_probability: float,
        vsc_probability: float,
        rng: np.random.Generator,
    ) -> dict[int, TrackStatus]:
        events: dict[int, TrackStatus] = {}
        if rng.random() < safety_car_probability:
            start = int(rng.integers(low=max(3, total_laps // 6), high=max(4, total_laps - 8)))
            duration = int(rng.integers(2, 5))
            for lap in range(start, min(total_laps + 1, start + duration)):
                events[lap] = TrackStatus.SAFETY_CAR
        elif rng.random() < vsc_probability:
            start = int(rng.integers(low=max(3, total_laps // 5), high=max(4, total_laps - 6)))
            duration = int(rng.integers(1, 3))
            for lap in range(start, min(total_laps + 1, start + duration)):
                events[lap] = TrackStatus.VSC
        return events

    def _pit_decision(
        self,
        state,
        lap: int,
        lap_status: TrackStatus,
        context: RaceSimulationContext,
    ) -> tuple[bool, Compound]:
        driver = state.driver
        if driver == context.target_driver:
            for stop_index, stop in enumerate(context.target_strategy.pit_stops, start=1):
                if stop_index <= state.pit_count:
                    continue
                if lap == stop.lap:
                    return True, stop.compound
                if (
                    context.target_strategy.allow_sc_adjustment
                    and lap_status in {TrackStatus.SAFETY_CAR, TrackStatus.VSC}
                    and stop.flexible_window_start is not None
                    and stop.flexible_window_end is not None
                    and stop.flexible_window_start <= lap <= stop.flexible_window_end
                ):
                    return True, stop.compound
            return False, Compound.HARD
        for stop_index, (stop_lap, compound) in enumerate(context.rival_strategies.get(driver, []), start=1):
            if stop_index <= state.pit_count:
                continue
            if lap == stop_lap:
                return True, compound
        return False, Compound.HARD

    def _lap_time(
        self,
        state,
        gap_to_ahead: float,
        track_status: TrackStatus,
        weather: float,
        rng: np.random.Generator,
        track,
    ) -> float:
        compound_factor = {
            Compound.SOFT: 0.96,
            Compound.MEDIUM: 1.0,
            Compound.HARD: 1.03,
            Compound.INTERMEDIATE: 1.08,
            Compound.WET: 1.14,
        }[state.compound]
        base = state.baseline_pace_seconds * compound_factor
        fuel_effect = max(0.1, (track.total_laps - state.current_stint_laps) / track.total_laps) * 0.8
        degradation = state.degradation_per_lap * state.tyre_age_laps * (1.0 + track.degradation_score)
        traffic_penalty = 0.0
        if gap_to_ahead < 1.8 and state.position > 1 and track_status is TrackStatus.GREEN:
            traffic_penalty = (1.8 - gap_to_ahead) * state.traffic_sensitivity * (0.8 + track.overtaking_difficulty)
            state.traffic_loss_seconds += traffic_penalty
        weather_penalty = weather * (2.4 if state.compound in {Compound.SOFT, Compound.MEDIUM, Compound.HARD} else 0.9)
        status_penalty = 0.0
        if track_status is TrackStatus.VSC:
            status_penalty = 12.0
        elif track_status is TrackStatus.SAFETY_CAR:
            status_penalty = 22.0
        noise = float(rng.normal(0.0, state.pace_variance_seconds if track_status is TrackStatus.GREEN else 0.08))
        out_lap = state.out_lap_penalty_seconds
        state.out_lap_penalty_seconds = max(0.0, state.out_lap_penalty_seconds - 0.5)
        return base + fuel_effect + degradation + traffic_penalty + weather_penalty + status_penalty + noise + out_lap

    def _pit_loss(
        self,
        pit_lane_loss_seconds: float,
        track_status: TrackStatus,
        rng: np.random.Generator,
    ) -> float:
        multiplier = 1.0
        if track_status is TrackStatus.SAFETY_CAR:
            multiplier = 0.62
        elif track_status is TrackStatus.VSC:
            multiplier = 0.8
        stationary = float(rng.normal(2.4, 0.25))
        in_out_lap = float(rng.normal(1.2, 0.15))
        return (pit_lane_loss_seconds * multiplier) + stationary + in_out_lap

    def _resolve_overtakes(
        self,
        states,
        overtaking_difficulty: float,
        rng: np.random.Generator,
    ) -> None:
        states.sort(key=lambda item: (item.cumulative_time_seconds, item.grid_position))
        for index in range(1, len(states)):
            ahead = states[index - 1]
            behind = states[index]
            gap = behind.cumulative_time_seconds - ahead.cumulative_time_seconds
            pace_delta = ahead.baseline_pace_seconds - behind.baseline_pace_seconds
            if gap > 1.0 or pace_delta <= 0:
                continue
            pass_probability = 1.0 / (1.0 + exp(-(pace_delta * 3.0 + behind.overtake_skill - overtaking_difficulty)))
            if rng.random() < pass_probability:
                behind.cumulative_time_seconds = ahead.cumulative_time_seconds - 0.15
                ahead.cumulative_time_seconds += 0.05


def _compute_gaps(states) -> dict[str, float]:
    ordered = sorted(states, key=lambda item: (item.cumulative_time_seconds, item.grid_position))
    gaps: dict[str, float] = {}
    for index, state in enumerate(ordered):
        if index == 0:
            gaps[state.driver] = 99.0
            continue
        gaps[state.driver] = state.cumulative_time_seconds - ordered[index - 1].cumulative_time_seconds
    return gaps


def _undercut_gain(candidate: StrategyCandidate) -> float:
    if not candidate.pit_stops:
        return 0.0
    first = candidate.pit_stops[0].lap
    if first <= 15:
        return 1.3
    if first <= 22:
        return 0.8
    return 0.4
