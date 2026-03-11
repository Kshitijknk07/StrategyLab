"""Runtime configuration for StrategyLab."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="STRATEGYLAB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = "development"
    log_level: str = "INFO"
    storage_root: Path = Field(default=Path("./storage"))
    artifact_root: Path = Field(default=Path("./artifacts"))
    dataset_root: Path = Field(default=Path("./artifacts/datasets"))
    database_url: str = "postgresql+psycopg://strategylab:strategylab@localhost:5432/strategylab"
    redis_url: str = "redis://localhost:6379/0"
    mlflow_tracking_uri: str = "file:./.mlruns"
    default_monte_carlo_runs: int = 500
    random_seed: int = 44

    @property
    def raw_path(self) -> Path:
        return self.storage_root / "raw"

    @property
    def processed_path(self) -> Path:
        return self.storage_root / "processed"

    @property
    def features_path(self) -> Path:
        return self.storage_root / "features"

    @property
    def simulation_inputs_path(self) -> Path:
        return self.storage_root / "simulation_inputs"

    @property
    def manifests_path(self) -> Path:
        return self.storage_root / "manifests"

    @property
    def model_registry_path(self) -> Path:
        return self.artifact_root / "models"

    @property
    def model_root(self) -> Path:
        return self.model_registry_path

    def ensure_directories(self) -> None:
        """Create all local filesystem roots."""
        for path in [
            self.storage_root,
            self.raw_path,
            self.processed_path,
            self.features_path,
            self.simulation_inputs_path,
            self.manifests_path,
            self.artifact_root,
            self.model_root,
            self.dataset_root,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
