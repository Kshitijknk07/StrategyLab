from __future__ import annotations

from strategylab.contracts import IngestionRefreshRequest
from strategylab.data.service import IngestionService
from strategylab.models.catalog import get_model


class IngestionPipeline:
    def __init__(self, service: IngestionService | None = None) -> None:
        self.service = service or IngestionService()

    def ingest(self, request: IngestionRefreshRequest) -> dict[str, str]:
        return self.service.refresh(request)


class TrainingPipeline:
    def train_model(self, model_name: str, dataset_version: str, target_column: str | None = None):
        model = get_model(model_name)
        return model.train(dataset_version=dataset_version, target_column=target_column)

    def evaluate_model(self, model_name: str, dataset_version: str, model_version: str):
        model = get_model(model_name)
        return model.evaluate(dataset_version=dataset_version, model_version=model_version)
