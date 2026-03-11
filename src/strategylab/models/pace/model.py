"""Baseline clean-air pace model."""

from __future__ import annotations

from strategylab.models.base import LocalTabularModel


class BaselinePaceModel(LocalTabularModel):
    model_name = "baseline_pace"

    @property
    def categorical_features(self) -> list[str]:
        return ["driver", "constructor", "race_circuit", "tyre_compound", "track_status"]

    @property
    def numeric_features(self) -> list[str]:
        return [
            "lap_number",
            "qualifying_pace_proxy",
            "team_strength_proxy",
            "track_temperature_c",
            "air_temperature_c",
            "humidity_pct",
            "fuel_phase_proxy",
            "traffic_density_proxy",
        ]

    @property
    def target_column(self) -> str:
        return "baseline_target_seconds"

