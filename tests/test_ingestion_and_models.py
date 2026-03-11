from __future__ import annotations

from strategylab.contracts import IngestionRefreshRequest
from strategylab.data.feature_store import FeatureStore
from strategylab.data.service import IngestionService
from strategylab.models.catalog import get_model


def test_ingestion_refresh_builds_feature_dataset(fixture_path) -> None:
    service = IngestionService()
    result = service.refresh(
        IngestionRefreshRequest(
            source="fixture",
            season=2024,
            event_name="Bahrain_GP",
            circuit="sakhir",
            fixture_path=str(fixture_path),
        )
    )

    frame, manifest = FeatureStore().load_dataset(result["dataset_version"])
    assert manifest.record_count == 12
    assert "lap_time_delta_to_baseline" in frame.columns
    assert "rolling_driver_pace" in frame.columns
    assert result["race_key"] == "2024_bahrain_gp_sakhir_race"


def test_train_and_evaluate_local_models(fixture_path) -> None:
    service = IngestionService()
    result = service.refresh(
        IngestionRefreshRequest(
            source="fixture",
            season=2024,
            event_name="Bahrain_GP",
            circuit="sakhir",
            fixture_path=str(fixture_path),
        )
    )

    pace_model = get_model("pace")
    pace_result = pace_model.train(result["dataset_version"])
    assert pace_result["model_name"] == "baseline_pace"
    assert pace_result["dataset_version"] == result["dataset_version"]

    pace_report = pace_model.evaluate(result["dataset_version"], pace_result["model_version"])
    assert {metric.name for metric in pace_report.metrics} == {"mae", "rmse"}

    tyre_model = get_model("tyre")
    tyre_result = tyre_model.train(result["dataset_version"])
    tyre_report = tyre_model.evaluate(result["dataset_version"], tyre_result["model_version"])
    assert tyre_result["model_name"] == "tyre_degradation"
    assert len(tyre_report.metrics) == 2

