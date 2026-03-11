"""Shared base classes for local tabular models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from strategylab.data.schemas import EvaluationReport, ModelPrediction, PredictionInterval
from strategylab.infra.config import get_settings


@dataclass(slots=True)
class TrainingArtifact:
    model_name: str
    model_version: str
    dataset_version: str
    feature_version: str
    model_path: Path
    metrics: dict[str, float]


class BatchPrediction(BaseModel):
    predictions: list[ModelPrediction]


class LocalTabularModel(ABC):
    """Train and serve local gradient-boosting tabular models."""

    model_name = "base_model"

    def __init__(self) -> None:
        self.settings = get_settings()
        self.pipeline: Pipeline | None = None
        self.model_version: str | None = None
        self.dataset_version: str | None = None
        self.feature_version: str | None = None
        self.residual_sigma: float = 0.0

    @property
    @abstractmethod
    def categorical_features(self) -> list[str]:
        """Categorical features used by the model."""

    @property
    @abstractmethod
    def numeric_features(self) -> list[str]:
        """Numeric features used by the model."""

    @property
    @abstractmethod
    def target_column(self) -> str:
        """Training target column."""

    @property
    def feature_columns(self) -> list[str]:
        return [*self.categorical_features, *self.numeric_features]

    def train(
        self,
        frame: pd.DataFrame,
        *,
        dataset_version: str,
        feature_version: str,
    ) -> TrainingArtifact:
        features = frame[self.feature_columns]
        target = frame[self.target_column]
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(features, target)

        fitted = self.pipeline.predict(features)
        self.residual_sigma = float(np.std(target - fitted))
        self.dataset_version = dataset_version
        self.feature_version = feature_version
        self.model_version = self._build_model_version(dataset_version, feature_version, len(frame))

        model_dir = self.settings.model_root / self.model_name / self.model_version
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.joblib"
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "model_version": self.model_version,
                "dataset_version": dataset_version,
                "feature_version": feature_version,
                "residual_sigma": self.residual_sigma,
            },
            model_path,
        )

        metrics = self._metrics(target.to_numpy(), fitted)
        return TrainingArtifact(
            model_name=self.model_name,
            model_version=self.model_version,
            dataset_version=dataset_version,
            feature_version=feature_version,
            model_path=model_path,
            metrics=metrics,
        )

    def load(self, model_path: str | Path) -> None:
        payload = joblib.load(model_path)
        self.pipeline = payload["pipeline"]
        self.model_version = payload["model_version"]
        self.dataset_version = payload["dataset_version"]
        self.feature_version = payload["feature_version"]
        self.residual_sigma = payload["residual_sigma"]

    def predict(self, frame: pd.DataFrame) -> BatchPrediction:
        if self.pipeline is None or self.model_version is None or self.dataset_version is None or self.feature_version is None:
            raise RuntimeError("Model is not loaded or trained.")
        values = self.pipeline.predict(frame[self.feature_columns])
        interval = PredictionInterval(
            lower=float(-1.96 * self.residual_sigma),
            upper=float(1.96 * self.residual_sigma),
            confidence=0.95,
        )
        predictions = [
            ModelPrediction(
                value=float(value),
                interval=PredictionInterval(
                    lower=float(value + interval.lower),
                    upper=float(value + interval.upper),
                    confidence=interval.confidence,
                ),
                model_version=self.model_version,
                dataset_version=self.dataset_version,
                feature_version=self.feature_version,
            )
            for value in values
        ]
        return BatchPrediction(predictions=predictions)

    def evaluate(self, frame: pd.DataFrame) -> EvaluationReport:
        if self.pipeline is None or self.model_version is None or self.dataset_version is None:
            raise RuntimeError("Model is not loaded or trained.")
        features = frame[self.feature_columns]
        target = frame[self.target_column]
        predicted = self.pipeline.predict(features)
        metrics = self._metrics(target.to_numpy(), predicted)
        return EvaluationReport(
            model_name=self.model_name,
            model_version=self.model_version,
            dataset_version=self.dataset_version,
            metrics=metrics,
            generated_at=datetime.now(tz=timezone.utc),
        )

    def _build_pipeline(self) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    self.categorical_features,
                ),
                (
                    "numeric",
                    Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                    self.numeric_features,
                ),
            ]
        )
        regressor = GradientBoostingRegressor(random_state=self.settings.random_seed)
        return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])

    def _build_model_version(self, dataset_version: str, feature_version: str, row_count: int) -> str:
        digest = sha1(
            f"{self.model_name}-{dataset_version}-{feature_version}-{row_count}".encode("utf-8")
        ).hexdigest()
        return digest[:12]

    @staticmethod
    def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
        rmse = mean_squared_error(actual, predicted) ** 0.5
        mae = mean_absolute_error(actual, predicted)
        return {"rmse": float(rmse), "mae": float(mae)}

