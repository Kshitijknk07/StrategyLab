from __future__ import annotations

from strategylab.models.base import BaseTabularModel
from strategylab.models.pace.model import BaselinePaceModel
from strategylab.models.tyre_deg.model import TyreDegModel


def get_model(model_name: str) -> BaseTabularModel:
    normalized = model_name.strip().lower()
    if normalized in {"pace", "baseline_pace"}:
        return BaselinePaceModel()
    if normalized in {"tyre", "tyre_deg", "tyre_degradation"}:
        return TyreDegModel()
    raise KeyError(f"Unknown model: {model_name}")
