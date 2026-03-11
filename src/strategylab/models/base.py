from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from math import sqrt
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from strategylab.contracts import EvaluationMetric, EvaluationReport, ModelPrediction, PredictionInterval
from strategylab.data.feature_store import FeatureStore
from strategylab.infra.config import get_settings
from strategylab.infra.registry import JsonManifestStore, LocalModelRegistry


class BaseTabularModel(ABC):
    def __init__(
        self,
        feature_store: FeatureStore | None = None,
        registry: LocalModelRegistry | None = None,
        manifest_store: JsonManifestStore | None = None,
    ) -> None:
        self.feature_store = feature_store or FeatureStore()
        self.registry = registry or LocalModelRegistry()
        self.manifest_store = manifest_store or JsonManifestStore()
        self.settings = get_settings()

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def default_target(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def categorical_columns(self) -> list[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def numeric_columns(self) -> list[str]:
        raise NotImplementedError

    def train(
        self,
        dataset_version: str,
        target_column: str | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        frame, manifest = self.feature_store.load_dataset(dataset_version)
        target = target_column or self.default_target
        prepared = self.prepare_training_frame(frame, target)
        if prepared.empty:
            raise ValueError(f"No usable training rows available for {self.model_name}.")
        train_frame, validation_frame = self._time_split(prepared)
        pipeline = self._build_pipeline(model_config or {})
        pipeline.fit(train_frame[self.feature_columns], train_frame[target])
        validation_predictions = pipeline.predict(validation_frame[self.feature_columns])
        residual_std = float(np.std(validation_frame[target] - validation_predictions))
        report = EvaluationReport(
            model_name=self.model_name,
            model_version=self._new_version(),
            dataset_version=dataset_version,
            metrics=[
                EvaluationMetric(
                    name="mae",
                    value=float(mean_absolute_error(validation_frame[target], validation_predictions)),
                ),
                EvaluationMetric(
                    name="rmse",
                    value=float(sqrt(mean_squared_error(validation_frame[target], validation_predictions))),
                ),
            ],
            notes=[f"Trained on {len(train_frame)} rows; validated on {len(validation_frame)} rows."],
        )
        payload = {
            "model_name": self.model_name,
            "model_version": report.model_version,
            "dataset_version": dataset_version,
            "target_column": target,
            "feature_columns": self.feature_columns,
            "pipeline": pipeline,
            "residual_std": residual_std,
            "trained_at": datetime.now(UTC).isoformat(),
            "metrics": [metric.model_dump() for metric in report.metrics],
        }
        artifact_path = self.registry.save(self.model_name, report.model_version, payload)
        self.registry.save_metadata(
            self.model_name,
            report.model_version,
            {
                "artifact_path": str(artifact_path),
                "dataset_version": dataset_version,
                "target_column": target,
                "metrics": [metric.model_dump() for metric in report.metrics],
            },
        )
        self.manifest_store.save_evaluation_report(report)
        return {
            "model_name": self.model_name,
            "model_version": report.model_version,
            "dataset_version": dataset_version,
            "artifact_path": str(artifact_path),
            "metrics": report.metrics,
            "target_column": target,
            "source_manifest": manifest.file_path,
        }

    def predict(self, model_version: str, frame: pd.DataFrame, dataset_version: str | None = None) -> ModelPrediction:
        payload = self.registry.load(self.model_name, model_version)
        prepared = self.prepare_inference_frame(frame)
        values = payload["pipeline"].predict(prepared[self.feature_columns])
        residual_std = float(payload.get("residual_std", 0.0))
        intervals = [
            PredictionInterval(low=float(value - 1.96 * residual_std), high=float(value + 1.96 * residual_std))
            for value in values
        ]
        return ModelPrediction(
            model_name=self.model_name,
            model_version=model_version,
            dataset_version=dataset_version or payload["dataset_version"],
            values=[float(value) for value in values],
            intervals=intervals,
        )

    def evaluate(self, dataset_version: str, model_version: str) -> EvaluationReport:
        frame, _ = self.feature_store.load_dataset(dataset_version)
        payload = self.registry.load(self.model_name, model_version)
        target = payload["target_column"]
        prepared = self.prepare_training_frame(frame, target)
        predictions = payload["pipeline"].predict(prepared[self.feature_columns])
        report = EvaluationReport(
            model_name=self.model_name,
            model_version=model_version,
            dataset_version=dataset_version,
            metrics=[
                EvaluationMetric(name="mae", value=float(mean_absolute_error(prepared[target], predictions))),
                EvaluationMetric(name="rmse", value=float(sqrt(mean_squared_error(prepared[target], predictions)))),
            ],
            notes=["Evaluation run against the full dataset version."],
        )
        self.manifest_store.save_evaluation_report(report)
        return report

    @property
    def feature_columns(self) -> list[str]:
        return [*self.categorical_columns, *self.numeric_columns]

    def prepare_training_frame(self, frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
        working = frame.copy()
        for feature in self.feature_columns:
            if feature not in working.columns:
                working[feature] = np.nan
        if target_column not in working.columns:
            raise ValueError(f"Target column '{target_column}' not present in dataset.")
        working = working.dropna(subset=[target_column])
        working = working.sort_values(self._sort_columns(working)).reset_index(drop=True)
        return working

    def prepare_inference_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        working = frame.copy()
        for feature in self.feature_columns:
            if feature not in working.columns:
                working[feature] = np.nan
        return working

    def _build_pipeline(self, model_config: dict[str, Any]) -> Pipeline:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("categorical", categorical_pipeline, self.categorical_columns),
                ("numeric", numeric_pipeline, self.numeric_columns),
            ]
        )
        estimator = GradientBoostingRegressor(
            random_state=int(model_config.get("random_state", self.settings.random_seed)),
            learning_rate=float(model_config.get("learning_rate", 0.05)),
            n_estimators=int(model_config.get("n_estimators", 250)),
            max_depth=int(model_config.get("max_depth", 3)),
        )
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("estimator", estimator),
            ]
        )

    def _time_split(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if len(frame) < 10:
            return frame, frame
        split_index = max(int(len(frame) * 0.8), 1)
        train = frame.iloc[:split_index].copy()
        validation = frame.iloc[split_index:].copy()
        return train, validation

    def _sort_columns(self, frame: pd.DataFrame) -> list[str]:
        candidates = [column for column in ("season", "event_name", "lap_number", "driver") if column in frame.columns]
        return candidates or list(frame.columns[:1])

    def _new_version(self) -> str:
        return datetime.now(UTC).strftime("%Y%m%d%H%M%S") + "_" + uuid4().hex[:8]
