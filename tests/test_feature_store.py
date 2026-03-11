from __future__ import annotations

from strategylab.data.feature_store import FeatureStore


def test_feature_store_builds_versioned_dataset(sample_bundle):
    store = FeatureStore()
    manifest = store.build_driver_lap_dataset(
        sample_bundle.laps,
        description="synthetic training dataset",
        dataset_version="test_feature_v1",
    )
    frame, loaded_manifest = store.load_dataset("test_feature_v1")

    assert manifest.layer.value == "features"
    assert loaded_manifest.dataset_version == "test_feature_v1"
    assert len(frame) == len(sample_bundle.laps)
    assert "lap_time_delta_to_baseline" in frame.columns
    assert "stint_progress" in frame.columns
    assert "track_status_is_green" in frame.columns
