from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from strategylab.infra.config import get_settings


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("STRATEGYLAB_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("STRATEGYLAB_MODEL_ROOT", str(tmp_path / "artifacts" / "models"))
    monkeypatch.setenv("STRATEGYLAB_DATASET_ROOT", str(tmp_path / "artifacts" / "datasets"))
    monkeypatch.setenv("STRATEGYLAB_MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
    monkeypatch.setenv("STRATEGYLAB_RANDOM_SEED", "17")
    get_settings.cache_clear()


@pytest.fixture
def fixture_path() -> Path:
    return Path(__file__).parent / "fixtures" / "bahrain_2024_fixture.json"


@pytest.fixture
def api_app():
    import strategylab.apps.api.main as api_main

    get_settings.cache_clear()
    api_main = importlib.reload(api_main)
    return api_main.app

