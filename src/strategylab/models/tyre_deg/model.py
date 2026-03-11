"""Tyre degradation model."""

from __future__ import annotations

from strategylab.models.base import LocalTabularModel


class TyreDegradationModel(LocalTabularModel):
    model_name = "tyre_degradation"

    @property
    def categorical_features(self) -> list[str]:
        return ["driver", "constructor", "race_circuit", "tyre_compound", "track_status"]

    @property
    def numeric_features(self) -> list[str]:
        return [
            "lap_number",
            "tyre_age_laps",
            "stint_number",
            "stint_phase_ratio",
            "track_temperature_c",
            "air_temperature_c",
            "humidity_pct",
            "traffic_density_proxy",
            "circuit_degradation_class",
            "fuel_phase_proxy",
        ]

    @property
    def target_column(self) -> str:
        return "degradation_target_seconds"
