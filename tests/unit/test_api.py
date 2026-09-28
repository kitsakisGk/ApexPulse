"""Tests for the HTTP API.

An API is a contract, so these assertions cover the shape of every response, the
status codes a caller branches on, and the behaviour when a dependency is
missing — which is the case a dashboard actually has to handle.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from httpx import ASGITransport, AsyncClient

from apexpulse.api import create_app
from apexpulse.config import Settings
from apexpulse.ml import save_model, train_model
from apexpulse.producer import MatchSimulator
from apexpulse.schemas.events import TickEvent
from apexpulse.storage import DuckDBSink
from apexpulse.stream import MatchTracker

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

BASE_TIME = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
async def checkpoint(tmp_path_factory):
    """Train a model once for the module to serve."""
    directory = tmp_path_factory.mktemp("api-model")

    async with DuckDBSink(path=":memory:", settings=Settings(), batch_size=10_000) as sink:
        for seed in range(8):
            simulator = MatchSimulator(
                match_id=f"m-{seed:02d}", seed=seed, tick_rate_hz=2.0, start_time=BASE_TIME
            )
            for event in simulator.run():
                await sink.handle(event)
        await sink.flush()
        frame = sink.training_frame()

    booster, calibrator, result = train_model(frame, num_rounds=120, seed=7)
    save_model(booster, result, calibrator, directory=directory)
    return directory


@pytest.fixture
async def client(checkpoint) -> AsyncIterator[AsyncClient]:
    """A client against an app with a loaded model and a populated store."""
    settings = Settings(state_backend="memory", model_dir=checkpoint)
    app = create_app(settings)

    async with app.router.lifespan_context(app):
        tracker = MatchTracker(store=app.state.apex.store, settings=settings)

        ticks = 0
        for event in MatchSimulator(seed=55, tick_rate_hz=4.0, start_time=BASE_TIME).run():
            await tracker.handle(event)
            if isinstance(event, TickEvent):
                ticks += 1
            # Stop on a live tick so the prediction routes have something to score.
            if (
                ticks >= 400
                and isinstance(event, TickEvent)
                and event.state.round_state.phase.value in {"live", "bomb_planted"}
            ):
                break

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as active:
            yield active


@pytest.fixture
async def bare_client(tmp_path) -> AsyncIterator[AsyncClient]:
    """A client against an app with no model and no match data.

    Points the model directory at an empty path, which is the state a fresh
    deployment is in before anyone has trained a checkpoint.
    """
    settings = Settings(state_backend="memory", model_dir=tmp_path / "no-checkpoint")
    app = create_app(settings)

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as active:
            yield active


# -- Health -------------------------------------------------------------------


async def test_health_reports_every_component(client: AsyncClient) -> None:
    response = await client.get("/health")
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "ok"
    assert {component["name"] for component in body["components"]} >= {"model"}
    assert all(component["healthy"] for component in body["components"])


async def test_health_degrades_when_the_model_is_missing(bare_client: AsyncClient) -> None:
    """A missing checkpoint is visible rather than silent."""
    response = await bare_client.get("/health")
    body = response.json()

    assert response.status_code == 503
    assert body["status"] == "degraded"
    model = next(item for item in body["components"] if item["name"] == "model")
    assert model["healthy"] is False
    assert model["detail"]


async def test_readiness_does_not_require_a_model(bare_client: AsyncClient) -> None:
    """The API serves live state without a model, so it is still ready."""
    response = await bare_client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"ready": True, "reason": None}


async def test_metrics_report_counters_and_latency(client: AsyncClient) -> None:
    await client.get("/matches/apex-001")

    response = await client.get("/metrics")
    body = response.json()

    assert response.status_code == 200
    assert body["model_loaded"] is True
    assert body["live_matches"] >= 1
    assert body["predictions_served"] >= 1
    assert body["uptime_seconds"] > 0.0


async def test_metrics_work_without_a_model(bare_client: AsyncClient) -> None:
    response = await bare_client.get("/metrics")
    body = response.json()

    assert response.status_code == 200
    assert body["model_loaded"] is False
    assert body["inference_p99_ms"] == 0.0


# -- Match listing ------------------------------------------------------------


async def test_listing_returns_live_matches(client: AsyncClient) -> None:
    response = await client.get("/matches")
    body = response.json()

    assert response.status_code == 200
    assert body["count"] >= 1
    assert body["matches"][0]["match_id"] == "apex-001"
    assert body["matches"][0]["map_name"] == "de_mirage"


async def test_listing_is_empty_with_no_matches(bare_client: AsyncClient) -> None:
    response = await bare_client.get("/matches")

    assert response.status_code == 200
    assert response.json() == {"count": 0, "matches": []}


# -- Match snapshot -----------------------------------------------------------


async def test_a_snapshot_carries_the_full_match_state(client: AsyncClient) -> None:
    response = await client.get("/matches/apex-001")
    body = response.json()

    assert response.status_code == 200
    assert body["match_id"] == "apex-001"
    assert len(body["players"]) == 10
    assert body["ct"]["alive"] + body["t"]["alive"] <= 10
    assert body["round"]["number"] >= 1
    assert "momentum" in body


async def test_a_snapshot_includes_a_prediction_when_a_model_is_loaded(
    client: AsyncClient,
) -> None:
    response = await client.get("/matches/apex-001")
    prediction = response.json()["prediction"]

    assert prediction is not None
    assert 0.0 <= prediction["ct_win_probability"] <= 1.0
    assert prediction["ct_win_probability"] + prediction["t_win_probability"] == pytest.approx(1.0)
    assert prediction["favoured_side"] in {"CT", "T", "even"}


async def test_an_unknown_match_returns_404(client: AsyncClient) -> None:
    response = await client.get("/matches/never-existed")

    assert response.status_code == 404
    assert "never-existed" in response.json()["detail"]


# -- History ------------------------------------------------------------------


async def test_history_returns_completed_rounds(client: AsyncClient) -> None:
    response = await client.get("/matches/apex-001/history")
    body = response.json()

    assert response.status_code == 200
    assert body["rounds_played"] >= 1
    assert body["ct_wins"] + body["t_wins"] == body["rounds_played"]
    assert len(body["rounds"]) == body["rounds_played"]


async def test_history_rounds_are_ordered(client: AsyncClient) -> None:
    rounds = (await client.get("/matches/apex-001/history")).json()["rounds"]

    numbers = [item["round_number"] for item in rounds]
    assert numbers == sorted(numbers)


async def test_history_for_an_unknown_match_is_empty_not_404(client: AsyncClient) -> None:
    """'No rounds yet' and 'never heard of it' are the same answer to a chart."""
    response = await client.get("/matches/never-existed/history")
    body = response.json()

    assert response.status_code == 200
    assert body["rounds_played"] == 0
    assert body["rounds"] == []


# -- Predictions --------------------------------------------------------------


async def test_a_prediction_carries_its_features(client: AsyncClient) -> None:
    from apexpulse.features import FEATURE_COUNT

    response = await client.get("/matches/apex-001/prediction")
    body = response.json()

    assert response.status_code == 200
    assert len(body["features"]) == FEATURE_COUNT
    assert 0.0 <= body["ct_win_probability"] <= 1.0
    assert body["latency_ms"] >= 0.0


async def test_features_are_ordered_by_training_importance(client: AsyncClient) -> None:
    """The most influential feature should be the first thing a reader sees."""
    features = (await client.get("/matches/apex-001/prediction")).json()["features"]

    importances = [feature["importance"] for feature in features]
    assert importances == sorted(importances, reverse=True)


async def test_predicting_an_unknown_match_returns_404(client: AsyncClient) -> None:
    response = await client.get("/matches/never-existed/prediction")

    assert response.status_code == 404


async def test_predicting_without_a_model_returns_503(bare_client: AsyncClient) -> None:
    """A missing model is a service condition, not a bad request."""
    response = await bare_client.get("/matches/anything/prediction")

    assert response.status_code == 503
    assert "apexpulse train" in response.json()["detail"]


# -- Model introspection ------------------------------------------------------


async def test_model_info_reports_the_checkpoint(client: AsyncClient) -> None:
    from apexpulse.features import FEATURE_NAMES

    response = await client.get("/model")
    body = response.json()

    assert response.status_code == 200
    assert body["loaded"] is True
    assert tuple(body["features"]) == FEATURE_NAMES
    assert body["trees"] is not None
    assert body["roc_auc"] is not None
    assert len(body["top_features"]) <= 5


async def test_model_info_reports_absence_without_erroring(bare_client: AsyncClient) -> None:
    """A dashboard should render the absence, not handle an exception."""
    response = await bare_client.get("/model")
    body = response.json()

    assert response.status_code == 200
    assert body["loaded"] is False
    assert body["features"] == []


# -- Contract -----------------------------------------------------------------


async def test_the_openapi_schema_documents_every_route(client: AsyncClient) -> None:
    spec = (await client.get("/openapi.json")).json()

    assert set(spec["paths"]) >= {
        "/health",
        "/ready",
        "/metrics",
        "/matches",
        "/matches/{match_id}",
        "/matches/{match_id}/history",
        "/matches/{match_id}/prediction",
        "/model",
    }


async def test_unknown_fields_are_rejected_by_the_response_models(client: AsyncClient) -> None:
    """extra='forbid' means a stale field cannot silently ride along."""
    from apexpulse.api.schemas import ScoreLine

    with pytest.raises(Exception, match="extra"):
        ScoreLine(ct=1, t=0, unexpected=True)  # type: ignore[call-arg]


async def test_responses_are_json(client: AsyncClient) -> None:
    for path in ("/health", "/ready", "/metrics", "/matches", "/model"):
        response = await client.get(path)
        assert response.headers["content-type"].startswith("application/json"), path
