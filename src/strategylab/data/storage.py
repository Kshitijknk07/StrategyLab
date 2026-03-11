from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from strategylab.contracts import DatasetLayer, DatasetManifest, RaceKey
from strategylab.infra.config import get_settings
from strategylab.infra.registry import JsonManifestStore


class LayeredStorage:
    def __init__(self) -> None:
        settings = get_settings()
        self.settings = settings
        self.manifests = JsonManifestStore(settings.manifests_path)

    def _layer_path(self, layer: DatasetLayer) -> Path:
        if layer is DatasetLayer.RAW:
            return self.settings.raw_path
        if layer is DatasetLayer.PROCESSED:
            return self.settings.processed_path
        if layer is DatasetLayer.FEATURES:
            return self.settings.features_path
        return self.settings.simulation_inputs_path

    def write_json(self, layer: DatasetLayer, name: str, payload: dict[str, Any]) -> Path:
        path = self._layer_path(layer) / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def write_dataframe(
        self,
        layer: DatasetLayer,
        dataset_version: str,
        table_name: str,
        frame: pd.DataFrame,
        description: str,
        source_sessions: list[RaceKey],
        target_columns: list[str] | None = None,
    ) -> DatasetManifest:
        path = self._layer_path(layer) / f"{table_name}_{dataset_version}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        manifest = DatasetManifest(
            dataset_version=dataset_version,
            layer=layer,
            description=description,
            record_count=len(frame),
            source_sessions=source_sessions,
            feature_columns=list(frame.columns),
            target_columns=target_columns or [],
            file_path=str(path),
        )
        self.manifests.save_dataset_manifest(manifest)
        return manifest

    def read_dataframe(self, manifest: DatasetManifest) -> pd.DataFrame:
        return pd.read_parquet(manifest.file_path)
