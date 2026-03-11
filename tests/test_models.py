from __future__ import annotations

from strategylab.data.feature_store import FeatureStore
from strategylab.models.pace.model import BaselinePaceModel
from strategylab.models.tyre_deg.model import TyreDegModel


def test_models_train_and_evaluate(sample_bundle):
    feature_store = FeatureStore()
    feature_store.build_driver_lap_dataset(
        sample_bundle.laps,
        description="synthetic model dataset",
        dataset_version="model_train_v1",
    )

    pace_model = BaselinePaceModel(feature_store=feature_store)
    tyre_model = TyreDegModel(feature_store=feature_store)

    pace_result = pace_model.train("model_train_v1")
    tyre_result = tyre_model.train("model_train_v1")

    assert pace_result["model_version"]
    assert tyre_result["model_version"]
    assert len(pace_result["metrics"]) == 2
    assert len(tyre_result["metrics"]) == 2

    pace_report = pace_model.evaluate("model_train_v1", pace_result["model_version"])
    tyre_report = tyre_model.evaluate("model_train_v1", tyre_result["model_version"])

    assert pace_report.model_name == "baseline_pace"
    assert tyre_report.model_name == "tyre_degradation"
    assert all(metric.value >= 0.0 for metric in pace_report.metrics)
    assert all(metric.value >= 0.0 for metric in tyre_report.metrics)
