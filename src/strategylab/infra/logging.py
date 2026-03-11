"""Structured logging setup."""

from __future__ import annotations

import logging
from typing import Final

from strategylab.infra.config import get_settings

_FORMAT: Final[str] = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure_logging(level: str | None = None) -> None:
    """Configure root logging once for the process."""
    settings = get_settings()
    logging.basicConfig(level=(level or settings.log_level).upper(), format=_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """Return a configured module logger."""
    configure_logging()
    return logging.getLogger(name)
