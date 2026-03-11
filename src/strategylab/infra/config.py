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

    env: str = "local"
    log_level: str = "INFO"
    storage_root: Path = Field(default=Path("./storage"))
    model_root: Path = Field(default=Path("./artifacts/models"))
    dataset_root: Path = Field(default=Path("./artifacts/datasets"))
    metadata_db_url: str = "postgresql+psycopg://strategylab:strategylab@localhost:5432/strategylab"
    redis_url: str = "redis://localhost:6379/0"
    mlflow_tracking_uri: str = "./mlruns"
    default_monte_carlo_runs: int = 500
    random_seed: int = 42

    def ensure_directories(self) -> None:
        """Create all local filesystem roots."""
        for path in [
            self.storage_root,
            self.storage_root / "raw",
            self.storage_root / "processed",
            self.storage_root / "features",
            self.storage_root / "simulation_inputs",
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

