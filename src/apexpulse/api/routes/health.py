"""Health, readiness, and metrics routes.

Health and readiness answer different questions and an orchestrator needs both:
health says the process is alive and its dependencies respond, readiness says it
can serve traffic right now. A process can be healthy but not ready — still
loading a model, say — and routing to it would return errors.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from apexpulse import __version__
from apexpulse.api.dependencies import get_state
from apexpulse.api.schemas import (
    ComponentHealth,
    HealthResponse,
    MetricsResponse,
    ReadinessResponse,
)

router = APIRouter(tags=["operations"])


@router.get("/health", response_model=HealthResponse, summary="Liveness and dependencies")
async def health(request: Request, response: Response) -> HealthResponse:
    """Report the status of each dependency.

    Returns 503 when any component is unhealthy, so a monitor can act on the
    status code without parsing the body.
    """
    state = get_state(request)
    components: list[ComponentHealth] = []

    store_healthy = False
    detail: str | None = None
    if state.store is None:
        detail = "state store is not started"
    else:
        try:
            store_healthy = await state.store.ping()
            if not store_healthy:
                detail = "state store did not respond to ping"
        except Exception as exc:  # report the failure rather than 500 the probe
            detail = f"{type(exc).__name__}: {exc}"

    components.append(
        ComponentHealth(
            name=f"state store ({state.settings.state_backend})",
            healthy=store_healthy,
            detail=detail,
        )
    )
    components.append(
        ComponentHealth(
            name="model",
            healthy=state.model_loaded,
            detail=None if state.model_loaded else "no checkpoint loaded",
        )
    )

    payload = HealthResponse(
        status="ok" if all(item.healthy for item in components) else "degraded",
        version=__version__,
        environment=state.settings.environment,
        components=tuple(components),
    )

    if not payload.is_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return payload


@router.get("/ready", response_model=ReadinessResponse, summary="Can this instance serve traffic")
async def ready(request: Request, response: Response) -> ReadinessResponse:
    """Report whether the instance can answer requests.

    The model is deliberately not required. The API serves live match state
    without one, which is useful on its own; only a missing state store makes
    the service unable to answer at all.
    """
    state = get_state(request)

    if state.manager is None or state.store is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(ready=False, reason="state store is not started")

    try:
        reachable = await state.store.ping()
    except Exception as exc:  # an unreachable store means unable to serve
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(ready=False, reason=f"{type(exc).__name__}: {exc}")

    if not reachable:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(ready=False, reason="state store is unreachable")

    return ReadinessResponse(ready=True)


@router.get("/metrics", response_model=MetricsResponse, summary="Operational counters")
async def metrics(request: Request) -> MetricsResponse:
    """Report counters and inference latency percentiles."""
    state = get_state(request)

    live_matches = 0
    if state.manager is not None:
        live_matches = len(await state.manager.live_match_ids())

    latency = state.engine.stats if state.engine is not None else None
    metadata = state.engine.metadata if state.engine is not None else {}

    return MetricsResponse(
        live_matches=live_matches,
        predictions_served=state.predictions_served,
        model_loaded=state.model_loaded,
        model_trained_at=metadata.get("trained_at"),
        inference_p50_ms=latency.p50_ms if latency else 0.0,
        inference_p95_ms=latency.p95_ms if latency else 0.0,
        inference_p99_ms=latency.p99_ms if latency else 0.0,
        uptime_seconds=state.uptime_seconds,
    )
