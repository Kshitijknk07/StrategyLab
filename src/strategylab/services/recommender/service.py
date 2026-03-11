from __future__ import annotations

from strategylab.contracts import SimulationInput, StrategyRecommendation
from strategylab.services.optimizer.service import StrategyOptimizer


class StrategyRecommender:
    def __init__(self, optimizer: StrategyOptimizer | None = None) -> None:
        self.optimizer = optimizer or StrategyOptimizer()

    def recommend(self, simulation_input: SimulationInput) -> StrategyRecommendation:
        ranked = self.optimizer.compare(simulation_input)
        if len(ranked) < 2:
            raise ValueError("At least two strategies are required to produce a primary and backup recommendation.")
        primary, backup = ranked[0], ranked[1]
        assumptions = [
            "Historical-only V1 assumptions are used for traffic, pit execution, and interruption priors.",
            "All predictive layers are local models or deterministic heuristics; no hosted inference is used.",
        ]
        confidence_notes = [
            "Primary recommendation is ranked by expected finish position, then expected race time.",
            "Finish distribution is Monte Carlo based and sensitive to baseline pace and degradation priors.",
        ]
        return StrategyRecommendation(
            race_key=simulation_input.race_key,
            primary_strategy=primary,
            backup_strategy=backup,
            assumptions=assumptions,
            confidence_notes=confidence_notes,
        )

