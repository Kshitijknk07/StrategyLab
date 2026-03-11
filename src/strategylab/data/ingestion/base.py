from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from strategylab.contracts import DriverLapRecord, PitEvent, RaceKey, RaceTimelineEvent, StintRecord, TrackProfile, WeatherSample


class SessionBundle(BaseModel):
    race_key: RaceKey
    track_profile: TrackProfile
    laps: list[DriverLapRecord] = Field(default_factory=list)
    stints: list[StintRecord] = Field(default_factory=list)
    pit_events: list[PitEvent] = Field(default_factory=list)
    weather_samples: list[WeatherSample] = Field(default_factory=list)
    timeline: list[RaceTimelineEvent] = Field(default_factory=list)
    source_metadata: dict[str, str | int | float | None] = Field(default_factory=dict)


class BaseSourceClient(ABC):
    @abstractmethod
    def fetch_session_bundle(
        self,
        season: int,
        event_name: str,
        circuit: str,
    ) -> SessionBundle:
        raise NotImplementedError

