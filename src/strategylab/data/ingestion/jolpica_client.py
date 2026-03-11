from __future__ import annotations

import httpx

from strategylab.contracts import RaceKey, SessionType, TrackProfile
from strategylab.data.ingestion.base import BaseSourceClient, SessionBundle


class JolpicaSourceClient(BaseSourceClient):
    def __init__(self, base_url: str = "https://api.jolpi.ca/ergast/f1") -> None:
        self.base_url = base_url.rstrip("/")

    def fetch_session_bundle(self, season: int, event_name: str, circuit: str) -> SessionBundle:
        response = httpx.get(f"{self.base_url}/{season}.json", timeout=20.0)
        response.raise_for_status()
        payload = response.json()
        race_key = RaceKey(
            season=season,
            event_name=event_name,
            circuit=circuit,
            session_type=SessionType.RACE,
        )
        return SessionBundle(
            race_key=race_key,
            track_profile=TrackProfile(
                circuit=circuit,
                lap_length_km=5.0,
                total_laps=57,
                pit_lane_loss_seconds=21.0,
                overtaking_difficulty=0.5,
                degradation_score=0.5,
                drs_zones=2,
                typical_safety_car_rate=0.3,
            ),
            source_metadata={
                "source": "jolpica",
                "records_available": len(payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])),
            },
        )

