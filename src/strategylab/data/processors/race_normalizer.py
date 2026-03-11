"""Normalize raw payloads into canonical StrategyLab records."""

from __future__ import annotations

import json
from typing import Any

from strategylab.data.ingestion.base import IngestedWeekend
from strategylab.data.schemas import (
    DriverLapRecord,
    PitEvent,
    RaceSessionKey,
    RaceTimelineEvent,
    SessionType,
    StintRecord,
    TrackProfile,
    TrackStatus,
    WeatherSample,
)


class RaceNormalizer:
    """Transform fixture or source-specific payloads into typed canonical records."""

    def from_fixture_json(self, payload: str) -> IngestedWeekend:
        data = json.loads(payload)
        race = RaceSessionKey(
            season=data["race"]["season"],
            event_slug=data["race"]["event_slug"],
            circuit=data["race"]["circuit"],
            session_type=SessionType(data["race"].get("session_type", "R")),
        )
        track = TrackProfile.model_validate(data["track"])
        laps = [DriverLapRecord.model_validate({**item, "race": race}) for item in data.get("laps", [])]
        stints = [StintRecord.model_validate({**item, "race": race}) for item in data.get("stints", [])]
        pit_events = [PitEvent.model_validate({**item, "race": race}) for item in data.get("pit_events", [])]
        weather = [WeatherSample.model_validate({**item, "race": race}) for item in data.get("weather", [])]
        timeline = [
            RaceTimelineEvent.model_validate(
                {
                    **item,
                    "race": race,
                    "track_status": item.get("track_status", TrackStatus.GREEN.value),
                }
            )
            for item in data.get("timeline", [])
        ]
        return IngestedWeekend(
            race=race,
            track=track,
            raw_payload=data,
            laps=laps,
            stints=stints,
            pit_events=pit_events,
            weather=weather,
            timeline=timeline,
        )

    def build_event_timeline(
        self,
        race: RaceSessionKey,
        laps: list[DriverLapRecord],
        pit_events: list[PitEvent],
        weather: list[WeatherSample],
        timeline: list[RaceTimelineEvent],
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for lap in laps:
            events.append(
                {
                    "lap_number": lap.lap_number,
                    "category": "lap",
                    "driver": lap.driver,
                    "message": f"{lap.driver} completed lap {lap.lap_number}",
                }
            )
        for event in pit_events:
            events.append(
                {
                    "lap_number": event.lap_number,
                    "category": "pit",
                    "driver": event.driver,
                    "message": f"{event.driver} pit stop",
                }
            )
        for sample in weather:
            if sample.lap_number is None:
                continue
            events.append(
                {
                    "lap_number": sample.lap_number,
                    "category": "weather",
                    "message": f"Weather sample at lap {sample.lap_number}",
                }
            )
        for event in timeline:
            events.append(
                {
                    "lap_number": event.lap_number,
                    "category": event.category,
                    "message": event.message,
                    "track_status": event.track_status.value if event.track_status else None,
                }
            )
        return sorted(events, key=lambda item: (item.get("lap_number") or 0, item["category"]))

