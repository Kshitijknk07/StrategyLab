from __future__ import annotations

from strategylab.contracts import SimulationInput, StrategyResult
from strategylab.services.simulator.service import RaceSimulator


class StrategyOptimizer:
    def __init__(self, simulator: RaceSimulator | None = None) -> None:
        self.simulator = simulator or RaceSimulator()

    def compare(self, simulation_input: SimulationInput) -> list[StrategyResult]:
        return self.simulator.compare_strategies(simulation_input)

