from __future__ import annotations

from datetime import UTC, datetime

from strategylab.contracts import DatasetLayer, IngestionRefreshRequest
from strategylab.data.feature_store import FeatureStore
from strategylab.data.ingestion.base import BaseSourceClient, SessionBundle
from strategylab.data.ingestion.fastf1_client import FastF1SourceClient
from strategylab.data.ingestion.fixture_client import FixtureSourceClient
from strategylab.data.ingestion.jolpica_client import JolpicaSourceClient
from strategylab.data.processors import RaceSessionNormalizer
from strategylab.data.storage import LayeredStorage


class IngestionService:
    def __init__(self, storage: LayeredStorage | None = None) -> None:
        self.storage = storage or LayeredStorage()
        self.normalizer = RaceSessionNormalizer()
        self.feature_store = FeatureStore(self.storage)

    def refresh(self, request: IngestionRefreshRequest) -> dict[str, str]:
        client = self._build_client(request)
        bundle = client.fetch_session_bundle(request.season, request.event_name, request.circuit)
        version = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        self._persist_raw(bundle, version)
        self._persist_processed(bundle, version)
        feature_manifest = self.feature_store.build_driver_lap_dataset(
            bundle.laps,
            description=f"Unified driver lap dataset for {bundle.race_key.slug}",
            dataset_version=version,
        )
        return {
            "race_key": bundle.race_key.slug,
            "dataset_version": version,
            "feature_manifest": feature_manifest.file_path,
        }

    def _build_client(self, request: IngestionRefreshRequest) -> BaseSourceClient:
        if request.source == "fixture":
            if not request.fixture_path:
                raise ValueError("fixture_path is required when source='fixture'")
            return FixtureSourceClient(request.fixture_path)
        if request.source == "fastf1":
            return FastF1SourceClient()
        return JolpicaSourceClient()

    def _persist_raw(self, bundle: SessionBundle, version: str) -> None:
        self.storage.write_json(
            DatasetLayer.RAW,
            f"{bundle.race_key.slug}_{version}",
            bundle.model_dump(mode="json"),
        )

    def _persist_processed(self, bundle: SessionBundle, version: str) -> None:
        source_sessions = [bundle.race_key]
        self.storage.write_dataframe(
            DatasetLayer.PROCESSED,
            version,
            "laps",
            self.normalizer.driver_lap_frame(bundle.laps),
            f"Processed laps for {bundle.race_key.slug}",
            source_sessions,
        )
        self.storage.write_dataframe(
            DatasetLayer.PROCESSED,
            version,
            "stints",
            self.normalizer.stint_frame(bundle),
            f"Processed stints for {bundle.race_key.slug}",
            source_sessions,
        )
        self.storage.write_dataframe(
            DatasetLayer.PROCESSED,
            version,
            "weather",
            self.normalizer.weather_frame(bundle),
            f"Processed weather for {bundle.race_key.slug}",
            source_sessions,
        )
        self.storage.write_dataframe(
            DatasetLayer.PROCESSED,
            version,
            "timeline",
            self.normalizer.timeline_frame(bundle),
            f"Unified event timeline for {bundle.race_key.slug}",
            source_sessions,
        )
        self.storage.write_dataframe(
            DatasetLayer.PROCESSED,
            version,
            "track_profile",
            self.normalizer.track_profile_frame(bundle),
            f"Track profile for {bundle.race_key.slug}",
            source_sessions,
        )
