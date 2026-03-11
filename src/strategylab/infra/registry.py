"""Simple local registries for models, datasets, and evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from strategylab.contracts import DatasetManifest, EvaluationReport
from strategylab.infra.config import get_settings


class JsonManifestStore:
    """Persist dataset manifests and evaluation reports as JSON."""

    def __init__(self, root: Path | None = None) -> None:
        settings = get_settings()
        self.root = root or settings.manifests_path
        self.root.mkdir(parents=True, exist_ok=True)
        self.dataset_root = self.root / "datasets"
        self.report_root = self.root / "reports"
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self.report_root.mkdir(parents=True, exist_ok=True)

    def write_manifest(self, relative_path: str, payload: dict[str, Any]) -> Path:
        manifest_path = self.root / relative_path
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return manifest_path

    def read_manifest(self, relative_path: str) -> dict[str, Any]:
        manifest_path = self.root / relative_path
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def save_dataset_manifest(self, manifest: DatasetManifest) -> Path:
        filename = f"{manifest.layer.value}_{manifest.dataset_version}.json"
        return self.write_manifest(f"datasets/{filename}", manifest.model_dump(mode="json"))

    def load_dataset_manifest(self, dataset_version: str, layer: str | None = None) -> DatasetManifest:
        if layer is None:
            matches = sorted(self.dataset_root.glob(f"*_{dataset_version}.json"))
            if not matches:
                raise FileNotFoundError(f"Dataset manifest not found: {dataset_version}")
            return DatasetManifest.model_validate_json(matches[-1].read_text(encoding="utf-8"))
        candidate = self.dataset_root / f"{layer}_{dataset_version}.json"
        if candidate.exists():
            manifest = DatasetManifest.model_validate_json(candidate.read_text(encoding="utf-8"))
            if manifest.layer.value == layer:
                return manifest
        for path in sorted(self.dataset_root.glob("*.json")):
            manifest = DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
            if manifest.dataset_version == dataset_version and manifest.layer.value == layer:
                return manifest
        raise FileNotFoundError(f"Dataset manifest not found: {dataset_version} ({layer})")

    def save_evaluation_report(self, report: EvaluationReport) -> Path:
        safe_name = f"{report.model_name}_{report.model_version}_{report.dataset_version}.json"
        return self.write_manifest(f"reports/{safe_name}", report.model_dump(mode="json"))


class LocalModelRegistry:
    """Persist local model artifacts and companion metadata."""

    def __init__(self, root: Path | None = None) -> None:
        settings = get_settings()
        self.root = root or settings.model_root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, model_name: str, model_version: str, payload: dict[str, Any]) -> Path:
        artifact_dir = self.root / model_name / model_version
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "artifact.joblib"
        joblib.dump(payload, artifact_path)
        return artifact_path

    def load(self, model_name: str, model_version: str) -> dict[str, Any]:
        artifact_path = self.root / model_name / model_version / "artifact.joblib"
        return joblib.load(artifact_path)

    def save_metadata(self, model_name: str, model_version: str, payload: dict[str, Any]) -> Path:
        artifact_dir = self.root / model_name / model_version
        artifact_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = artifact_dir / "metadata.json"
        metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return metadata_path
