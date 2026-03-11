from __future__ import annotations

from datetime import UTC, datetime
from importlib.util import find_spec
from typing import Any

import pandas as pd

from strategylab.contracts import (
    Compound,
    DriverLapRecord,
    EventType,
    RaceKey,
    RaceTimelineEvent,
    SessionType,
    StintRecord,
    TrackProfile,
    TrackStatus,
    WeatherSample,
)
from strategylab.data.ingestion.base import BaseSourceClient, SessionBundle


class FastF1SourceClient(BaseSourceClient):
    def __init__(self) -> None:
        if find_spec("fastf1") is None:
            raise ImportError("fastf1 is not installed. Install StrategyLab with the 'sources' extra.")

    def fetch_session_bundle(self, season: int, event_name: str, circuit: str) -> SessionBundle:
        import fastf1

        session = fastf1.get_session(season, event_name, "R")
        session.load(telemetry=False, weather=True, messages=True)

        race_key = RaceKey(
            season=season,
            event_name=str(session.event["EventName"]),
            circuit=circuit,
            session_type=SessionType.RACE,
        )
        track_profile = TrackProfile(
            circuit=circuit,
            country=str(session.event.get("Country", "")) or None,
            lap_length_km=max(float(session.event.get("CircuitLength", 5.0) or 5.0), 1.0),
            total_laps=int(session.total_laps or session.laps["LapNumber"].max()),
            pit_lane_loss_seconds=21.0,
            overtaking_difficulty=0.55,
            degradation_score=0.5,
            drs_zones=2,
            typical_safety_car_rate=0.3,
        )

        laps_frame = session.laps.copy()
        laps = self._build_lap_records(race_key, laps_frame)
        stints = self._build_stints(race_key, laps)
        weather_samples = self._build_weather(race_key, session.weather_data)
        timeline = self._build_timeline(race_key, laps)

        return SessionBundle(
            race_key=race_key,
            track_profile=track_profile,
            laps=laps,
            stints=stints,
            weather_samples=weather_samples,
            timeline=timeline,
            source_metadata={"source": "fastf1", "session_name": str(session.name)},
        )

    def _build_lap_records(self, race_key: RaceKey, frame: pd.DataFrame) -> list[DriverLapRecord]:
        results: list[DriverLapRecord] = []
        position_maps = {
            int(lap): subset.sort_values("Time").reset_index(drop=True)
            for lap, subset in frame.groupby("LapNumber", dropna=True)
        }
        for _, row in frame.iterrows():
            lap_number = int(row["LapNumber"])
            ordered = position_maps.get(lap_number)
            driver = str(row["Driver"])
            position_end = 1
            if ordered is not None:
                match = ordered.index[ordered["Driver"] == driver]
                if len(match) > 0:
                    position_end = int(match[0]) + 1
            lap_time = row.get("LapTime")
            if pd.isna(lap_time):
                continue
            tyre_life = int(row.get("TyreLife") or 0)
            stint_number = int(row.get("Stint") or 1)
            compound = _normalize_compound(row.get("Compound"))
            status = _normalize_track_status(row.get("TrackStatus"))
            position_start = max(position_end + int(row.get("PositionChange", 0) or 0), 1)
            results.append(
                DriverLapRecord(
                    race_key=race_key,
                    driver=driver,
                    constructor=str(row.get("Team", "Unknown")),
                    lap_number=lap_number,
                    lap_time_seconds=float(lap_time.total_seconds()),
                    sector_1_seconds=_seconds(row.get("Sector1Time")),
                    sector_2_seconds=_seconds(row.get("Sector2Time")),
                    sector_3_seconds=_seconds(row.get("Sector3Time")),
                    compound=compound,
                    tyre_age_laps=tyre_life,
                    stint_number=stint_number,
                    pit_in=bool(pd.notna(row.get("PitInTime"))),
                    pit_out=bool(pd.notna(row.get("PitOutTime"))),
                    track_status=status,
                    air_temp_c=float(row.get("AirTemp") or 25.0),
                    track_temp_c=float(row.get("TrackTemp") or 35.0),
                    humidity_pct=float(row.get("Humidity") or 45.0),
                    wet_track=compound in {Compound.INTERMEDIATE, Compound.WET},
                    position_start=position_start,
                    position_end=position_end,
                    drs_available=track_profile_drs_proxy(status, driver),
                    traffic_density=0.0,
                )
            )
        return results

    def _build_stints(self, race_key: RaceKey, laps: list[DriverLapRecord]) -> list[StintRecord]:
        stints: list[StintRecord] = []
        by_driver_stint: dict[tuple[str, int], list[DriverLapRecord]] = {}
        for lap in laps:
            by_driver_stint.setdefault((lap.driver, lap.stint_number), []).append(lap)
        for (driver, stint_number), rows in by_driver_stint.items():
            rows = sorted(rows, key=lambda item: item.lap_number)
            stints.append(
                StintRecord(
                    race_key=race_key,
                    driver=driver,
                    stint_number=stint_number,
                    compound=rows[0].compound,
                    start_lap=rows[0].lap_number,
                    end_lap=rows[-1].lap_number,
                    laps=len(rows),
                )
            )
        return stints

    def _build_weather(self, race_key: RaceKey, frame: pd.DataFrame | None) -> list[WeatherSample]:
        if frame is None or frame.empty:
            return []
        weather: list[WeatherSample] = []
        for idx, row in frame.reset_index(drop=True).iterrows():
            weather.append(
                WeatherSample(
                    race_key=race_key,
                    lap_number=idx + 1,
                    timestamp_utc=row.get("Time", pd.Timestamp(datetime.now(UTC))).to_pydatetime(),
                    air_temp_c=float(row.get("AirTemp") or 25.0),
                    track_temp_c=float(row.get("TrackTemp") or 35.0),
                    humidity_pct=float(row.get("Humidity") or 45.0),
                    rain_intensity=1.0 if bool(row.get("Rainfall")) else 0.0,
                    wind_speed_kph=float(row.get("WindSpeed") or 0.0),
                )
            )
        return weather

    def _build_timeline(self, race_key: RaceKey, laps: list[DriverLapRecord]) -> list[RaceTimelineEvent]:
        events: list[RaceTimelineEvent] = []
        for lap in laps:
            if lap.pit_in:
                events.append(
                    RaceTimelineEvent(
                        race_key=race_key,
                        lap_number=lap.lap_number,
                        event_type=EventType.PIT_IN,
                        message=f"{lap.driver} pits in",
                        driver=lap.driver,
                    )
                )
            if lap.track_status is not TrackStatus.GREEN:
                events.append(
                    RaceTimelineEvent(
                        race_key=race_key,
                        lap_number=lap.lap_number,
                        event_type=EventType.TRACK_STATUS,
                        message=f"Track status changed to {lap.track_status.value}",
                        track_status=lap.track_status,
                    )
                )
        return events


def _normalize_compound(raw: Any) -> Compound:
    text = str(raw or "medium").strip().lower()
    mapping = {
        "soft": Compound.SOFT,
        "s": Compound.SOFT,
        "medium": Compound.MEDIUM,
        "m": Compound.MEDIUM,
        "hard": Compound.HARD,
        "h": Compound.HARD,
        "intermediate": Compound.INTERMEDIATE,
        "wet": Compound.WET,
    }
    return mapping.get(text, Compound.MEDIUM)


def _normalize_track_status(raw: Any) -> TrackStatus:
    text = str(raw or "").lower()
    if "4" in text or "safety" in text:
        return TrackStatus.SAFETY_CAR
    if "6" in text or "vsc" in text or "virtual" in text:
        return TrackStatus.VSC
    if "red" in text:
        return TrackStatus.RED_FLAG
    return TrackStatus.GREEN


def _seconds(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value.total_seconds())


def track_profile_drs_proxy(track_status: TrackStatus, driver: str) -> bool:
    return track_status is TrackStatus.GREEN and bool(driver)
