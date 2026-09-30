"""FastAPI application factory.

The app is built by a function rather than created at import time, so tests can
construct an isolated instance with their own settings instead of sharing one
global object.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apexpulse import __version__
from apexpulse.api.dependencies import AppState
from apexpulse.config import get_settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from apexpulse.config import Settings

DESCRIPTION = """
Real-time CS2 telemetry and live win-probability.

Ingests match telemetry, scores a win-probability model on every tick, and serves
the result to a dashboard. See `/docs` for the interactive schema.
""".strip()


def create_app(
    settings: Settings | None = None,
    *,
    simulate: bool = False,
    simulate_seed: int = 42,
) -> FastAPI:
    """Build the API application.

    Args:
        settings: Runtime configuration; defaults to the process settings.
        simulate: Drive a match inside this process, so the API has live data
            without a separate producer. Useful for a demo; a real deployment
            consumes telemetry from the broker instead.
        simulate_seed: RNG seed for the simulated match.
    """
    resolved = settings or get_settings()
    state = AppState(settings=resolved)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
        """Open shared resources for the life of the process."""
        await state.startup()

        simulation: asyncio.Task[None] | None = None
        if simulate:
            simulation = asyncio.create_task(_run_simulation(state, simulate_seed))

        try:
            yield
        finally:
            if simulation is not None:
                simulation.cancel()
                with suppress(asyncio.CancelledError):
                    await simulation
            await state.shutdown()

    app = FastAPI(
        title="ApexPulse",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.state.apex = state

    # The dashboard is served from a different origin in development, so the
    # browser needs permission to call this API from it.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    from apexpulse.api.routes import health, matches, predictions, stream

    app.include_router(health.router)
    app.include_router(matches.router)
    app.include_router(predictions.router)
    app.include_router(stream.router)

    return app


async def _run_simulation(state: AppState, seed: int) -> None:
    """Replay simulated matches into the live state store and the broadcaster.

    Paced to real time so a viewer sees the probability move as a match unfolds
    rather than a whole match flashing past in a second. Loops, so a demo left
    open does not run dry.
    """
    from apexpulse.api.broadcaster import StreamBroadcaster
    from apexpulse.producer import MatchSimulator
    from apexpulse.schemas.events import TickEvent
    from apexpulse.stream import MatchTracker

    if state.store is None:
        return

    tracker = MatchTracker(store=state.store, settings=state.settings)
    broadcaster = StreamBroadcaster(state=state)
    tick_rate = 4.0
    interval = 1.0 / tick_rate

    offset = 0
    while True:
        simulator = MatchSimulator(match_id="apex-demo", seed=seed + offset, tick_rate_hz=tick_rate)
        for event in simulator.run():
            await tracker.handle(event)
            await broadcaster.handle(event)
            if isinstance(event, TickEvent):
                await asyncio.sleep(interval)
        offset += 1
