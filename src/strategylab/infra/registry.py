"""Simple local artifact registry for models and datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from strategylab.infra.config import get_settings


class LocalRegistry:
    """Persist reproducibility metadata alongside artifacts."""

    def __init__(self, root: Path | None = None) -> None:
        settings = get_settings()
        self.root = root or settings.model_root
        self.root.mkdir(parents=True, exist_ok=True)

    def write_manifest(self, relative_path: str, payload: dict[str, Any]) -> Path:
        manifest_path = self.root / relative_path
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return manifest_path

    def read_manifest(self, relative_path: str) -> dict[str, Any]:
        manifest_path = self.root / relative_path
        return json.loads(manifest_path.read_text(encoding="utf-8"))

