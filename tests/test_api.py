from __future__ import annotations

import httpx
import pytest


@pytest.mark.asyncio
async def test_api_end_to_end(sample_bundle, simulation_input, tmp_path, api_module):
    fixture_path = tmp_path / "bundle.json"
    fixture_path.write_text(sample_bundle.model_dump_json(indent=2), encoding="utf-8")
    transport = httpx.ASGITransport(app=api_module.app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        ingest_response = await client.post(
            "/ingestion/sessions:refresh",
            json={
                "source": "fixture",
                "season": 2024,
                "event_name": "italian_gp",
                "circuit": "monza",
                "fixture_path": str(fixture_path),
            },
        )
        assert ingest_response.status_code == 200
        dataset_version = ingest_response.json()["dataset_version"]

        laps_response = await client.get("/races/2024/italian_gp/driver-laps")
        assert laps_response.status_code == 200
        assert len(laps_response.json()) == len(sample_bundle.laps)

        train_response = await client.post(
            "/models/pace/train",
            json={"dataset_version": dataset_version},
        )
        assert train_response.status_code == 200
        model_version = train_response.json()["model_version"]

        evaluate_response = await client.post(
            "/models/pace/evaluate",
            json={"dataset_version": dataset_version, "model_version": model_version},
        )
        assert evaluate_response.status_code == 200
        assert evaluate_response.json()["model_name"] == "baseline_pace"

        compare_response = await client.post(
            "/simulation/compare",
            json=simulation_input.model_dump(mode="json"),
        )
        assert compare_response.status_code == 200
        assert len(compare_response.json()) == 2

        recommendation_response = await client.post(
            "/recommendations/strategy",
            json=simulation_input.model_dump(mode="json"),
        )
        assert recommendation_response.status_code == 200
        body = recommendation_response.json()
        assert body["primary_strategy"]["strategy_name"]
        assert body["backup_strategy"]["strategy_name"]
