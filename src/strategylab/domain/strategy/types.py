from __future__ import annotations

from dataclasses import dataclass

from strategylab.contracts import StrategyCandidate


@dataclass(slots=True)
class CandidateScore:
    strategy: StrategyCandidate
    expected_race_time_seconds: float
    expected_finish_position: float
    win_probability: float
    podium_probability: float
    top_ten_probability: float
    traffic_risk: float
    undercut_gain_seconds: float
    overcut_gain_seconds: float
    first_pit_mean: float
    first_pit_stddev: float

