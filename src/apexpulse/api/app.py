"""FastAPI application factory.

The app is built by a function rather than created at import time, so tests can
construct an isolated instance with their own settings instead of sharing one
global object.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
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


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the API application.

    Args:
        settings: Runtime configuration; defaults to the process settings.
    """
    resolved = settings or get_settings()
    state = AppState(settings=resolved)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
        """Open shared resources for the life of the process."""
        await state.startup()
        try:
            yield
        finally:
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

    from apexpulse.api.routes import health, matches

    app.include_router(health.router)
    app.include_router(matches.router)

    return app
