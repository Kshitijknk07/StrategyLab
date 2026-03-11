from __future__ import annotations

from strategylab.models.base import BaseTabularModel


class TyreDegModel(BaseTabularModel):
    @property
    def model_name(self) -> str:
        return "tyre_degradation"

    @property
    def default_target(self) -> str:
        return "lap_time_delta_to_baseline"

    @property
    def categorical_columns(self) -> list[str]:
        return ["driver", "constructor", "compound", "event_name", "circuit"]

    @property
    def numeric_columns(self) -> list[str]:
        return [
            "season",
            "lap_number",
            "tyre_age_laps",
            "traffic_density",
            "track_temp_c",
            "air_temp_c",
            "qualifying_pace_proxy",
            "team_strength_proxy",
            "stint_progress",
            "circuit_degradation_class",
            "wet_track_flag",
            "track_status_is_green",
        ]
