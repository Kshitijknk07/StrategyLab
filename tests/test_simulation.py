from __future__ import annotations

from strategylab.services.optimizer import StrategyOptimizer
from strategylab.services.recommender import StrategyRecommender


def test_simulation_and_recommendation(simulation_input):
    optimizer = StrategyOptimizer()
    recommender = StrategyRecommender(optimizer)

    ranked = optimizer.compare(simulation_input)
    recommendation = recommender.recommend(simulation_input)

    assert len(ranked) == 2
    assert ranked[0].strategy_name in {"medium-hard_early", "medium-hard_late"}
    assert 1.0 <= ranked[0].expected_finish_position <= 20.0
    assert recommendation.primary_strategy.strategy_name != ""
    assert recommendation.backup_strategy.strategy_name != ""
    assert recommendation.primary_strategy.strategy_name != recommendation.backup_strategy.strategy_name
