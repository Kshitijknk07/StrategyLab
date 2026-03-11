from __future__ import annotations

from strategylab.models.base import BaseTabularModel


class BaselinePaceModel(BaseTabularModel):
    @property
    def model_name(self) -> str:
        return "baseline_pace"

    @property
    def default_target(self) -> str:
        return "clean_air_baseline_seconds"

    @property
    def categorical_columns(self) -> list[str]:
        return ["driver", "constructor", "compound", "event_name", "circuit"]

    @property
    def numeric_columns(self) -> list[str]:
        return [
            "season",
            "lap_number",
            "air_temp_c",
            "track_temp_c",
            "humidity_pct",
            "qualifying_pace_proxy",
            "team_strength_proxy",
            "fuel_load_proxy",
            "stint_progress",
            "wet_track_flag",
            "track_status_is_green",
        ]
