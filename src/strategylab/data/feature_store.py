from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from strategylab.contracts import DatasetLayer, DatasetManifest, DriverLapRecord, RaceKey
from strategylab.data.processors import RaceSessionNormalizer, build_gap_features
from strategylab.data.storage import LayeredStorage


class FeatureStore:
    def __init__(self, storage: LayeredStorage | None = None) -> None:
        self.storage = storage or LayeredStorage()
        self.normalizer = RaceSessionNormalizer()

    def build_driver_lap_dataset(
        self,
        records: list[DriverLapRecord],
        description: str,
        dataset_version: str | None = None,
    ) -> DatasetManifest:
        version = dataset_version or datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        frame = self.normalizer.driver_lap_frame(records)
        features = build_gap_features(frame)
        source_sessions = _extract_race_keys(records)
        return self.storage.write_dataframe(
            layer=DatasetLayer.FEATURES,
            dataset_version=version,
            table_name="driver_laps",
            frame=features,
            description=description,
            source_sessions=source_sessions,
            target_columns=["lap_time_delta_to_baseline"],
        )

    def load_dataset(self, dataset_version: str) -> tuple[pd.DataFrame, DatasetManifest]:
        manifest = self.storage.manifests.load_dataset_manifest(dataset_version)
        return self.storage.read_dataframe(manifest), manifest


def _extract_race_keys(records: list[DriverLapRecord]) -> list[RaceKey]:
    seen: dict[str, RaceKey] = {}
    for record in records:
        seen.setdefault(record.race_key.slug, record.race_key)
    return list(seen.values())

