from __future__ import annotations

from collections import defaultdict

import pandas as pd

from strategylab.contracts import DriverLapRecord, EventType, RaceTimelineEvent, TrackStatus
from strategylab.data.ingestion.base import SessionBundle


class RaceSessionNormalizer:
    def build_timeline(self, bundle: SessionBundle) -> list[RaceTimelineEvent]:
        timeline = list(bundle.timeline)
        by_lap_status: dict[int, TrackStatus] = {}
        for lap in bundle.laps:
            if lap.track_status is not TrackStatus.GREEN and by_lap_status.get(lap.lap_number) != lap.track_status:
                timeline.append(
                    RaceTimelineEvent(
                        race_key=bundle.race_key,
                        lap_number=lap.lap_number,
                        event_type=EventType.TRACK_STATUS,
                        message=f"Track status {lap.track_status.value}",
                        track_status=lap.track_status,
                    )
                )
                by_lap_status[lap.lap_number] = lap.track_status
        timeline.extend(
            RaceTimelineEvent(
                race_key=bundle.race_key,
                lap_number=sample.lap_number,
                event_type=EventType.WEATHER,
                message="Weather sample",
                payload={
                    "air_temp_c": sample.air_temp_c,
                    "track_temp_c": sample.track_temp_c,
                    "humidity_pct": sample.humidity_pct,
                    "rain_intensity": sample.rain_intensity,
                },
            )
            for sample in bundle.weather_samples
        )
        return sorted(timeline, key=lambda event: (event.lap_number, event.event_type.value))

    def driver_lap_frame(self, records: list[DriverLapRecord]) -> pd.DataFrame:
        frame = pd.DataFrame(record.model_dump(mode="json") for record in records)
        if frame.empty:
            return frame
        frame["track_status"] = frame["track_status"].astype("string")
        frame["compound"] = frame["compound"].astype("string")
        frame["session_type"] = frame["race_key"].map(lambda rk: rk["session_type"])
        frame["event_name"] = frame["race_key"].map(lambda rk: rk["event_name"])
        frame["circuit"] = frame["race_key"].map(lambda rk: rk["circuit"])
        frame["season"] = frame["race_key"].map(lambda rk: rk["season"])
        frame = frame.drop(columns=["race_key"])
        return frame

    def stint_frame(self, bundle: SessionBundle) -> pd.DataFrame:
        return self._flatten_race_frame(pd.DataFrame(stint.model_dump(mode="json") for stint in bundle.stints))

    def pit_event_frame(self, bundle: SessionBundle) -> pd.DataFrame:
        return self._flatten_race_frame(pd.DataFrame(event.model_dump(mode="json") for event in bundle.pit_events))

    def weather_frame(self, bundle: SessionBundle) -> pd.DataFrame:
        return self._flatten_race_frame(
            pd.DataFrame(sample.model_dump(mode="json") for sample in bundle.weather_samples)
        )

    def timeline_frame(self, bundle: SessionBundle) -> pd.DataFrame:
        timeline = self.build_timeline(bundle)
        return self._flatten_race_frame(pd.DataFrame(event.model_dump(mode="json") for event in timeline))

    def track_profile_frame(self, bundle: SessionBundle) -> pd.DataFrame:
        return pd.DataFrame([bundle.track_profile.model_dump(mode="json")])

    def _flatten_race_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or "race_key" not in frame.columns:
            return frame
        frame["session_type"] = frame["race_key"].map(lambda rk: rk["session_type"])
        frame["event_name"] = frame["race_key"].map(lambda rk: rk["event_name"])
        frame["circuit"] = frame["race_key"].map(lambda rk: rk["circuit"])
        frame["season"] = frame["race_key"].map(lambda rk: rk["season"])
        return frame.drop(columns=["race_key"])


def build_gap_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    ordered = frame.sort_values(["driver", "lap_number"]).copy()
    ordered["lap_time_delta_to_baseline"] = (
        ordered["lap_time_seconds"] - ordered["clean_air_baseline_seconds"].fillna(ordered["lap_time_seconds"])
    )
    ordered["position_gain"] = ordered["position_start"] - ordered["position_end"]
    ordered["track_status_is_green"] = (ordered["track_status"] == TrackStatus.GREEN.value).astype(int)
    ordered["wet_track_flag"] = ordered["wet_track"].astype(int)
    ordered["pit_flag"] = (ordered["pit_in"] | ordered["pit_out"]).astype(int)
    ordered["stint_progress"] = ordered.groupby(["driver", "stint_number"]).cumcount() + 1
    ordered["rolling_driver_pace"] = (
        ordered.groupby("driver")["lap_time_seconds"].transform(lambda series: series.rolling(3, min_periods=1).mean())
    )
    ordered["rolling_team_pace"] = (
        ordered.groupby("constructor")["lap_time_seconds"].transform(lambda series: series.rolling(3, min_periods=1).mean())
    )
    gap_map = defaultdict(float)
    for idx, row in ordered.iterrows():
        key = (row["lap_number"], row["position_end"])
        ahead_gap = gap_map.get((row["lap_number"], row["position_end"] - 1), row["gap_to_ahead_seconds"] or 0.0)
        ordered.at[idx, "gap_to_ahead_seconds"] = ahead_gap
        gap_map[key] = row["gap_to_leader_seconds"] or 0.0
    return ordered
