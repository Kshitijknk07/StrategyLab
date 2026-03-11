from __future__ import annotations

import json
from pathlib import Path

from strategylab.data.ingestion.base import BaseSourceClient, SessionBundle


class FixtureSourceClient(BaseSourceClient):
    def __init__(self, fixture_path: str) -> None:
        self.fixture_path = Path(fixture_path)

    def fetch_session_bundle(self, season: int, event_name: str, circuit: str) -> SessionBundle:
        payload = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        bundle = SessionBundle.model_validate(payload)
        if bundle.race_key.season != season or bundle.race_key.event_name != event_name:
            raise ValueError("Fixture contents do not match requested race key.")
        return bundle

