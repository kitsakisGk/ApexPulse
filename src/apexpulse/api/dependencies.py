"""Shared application state and request dependencies.

Resources that are expensive to create — the state store connection, the loaded
model — are opened once at startup and reused. Opening them per request would
add tens of milliseconds to a path that exists to be fast, and would reload a
300KB booster on every call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from apexpulse.config import get_settings
from apexpulse.logging import get_logger

if TYPE_CHECKING:
    from apexpulse.config import Settings
    from apexpulse.inference.engine import InferenceEngine
    from apexpulse.storage.match_state import MatchStateManager
    from apexpulse.storage.state import StateStore

logger = get_logger(__name__)


@dataclass
class AppState:
    """Resources shared across requests for the lifetime of the process."""

    settings: Settings = field(default_factory=get_settings)
    store: StateStore | None = None
    manager: MatchStateManager | None = None
    engine: InferenceEngine | None = None
    started_at: float = field(default_factory=time.monotonic)
    predictions_served: int = 0

    @property
    def uptime_seconds(self) -> float:
        """Seconds since startup."""
        return time.monotonic() - self.started_at

    @property
    def model_loaded(self) -> bool:
        """Whether a checkpoint was found and loaded."""
        return self.engine is not None

    async def startup(self) -> None:
        """Open the state store and load the model, if one exists.

        A missing checkpoint is not fatal. The API still serves live match state,
        which is useful on its own, and reports the absence through /health
        rather than refusing to start.
        """
        from apexpulse.storage.match_state import MatchStateManager
        from apexpulse.storage.state import create_state_store

        self.store = create_state_store(self.settings)
        await self.store.start()
        self.manager = MatchStateManager(self.store, self.settings)

        try:
            from apexpulse.inference.engine import InferenceEngine

            self.engine = InferenceEngine.from_checkpoint(settings=self.settings)
        except FileNotFoundError:
            logger.warning("model_not_loaded", reason="no checkpoint found")
        except Exception as exc:  # a broken checkpoint must not stop the API
            logger.error("model_load_failed", error=str(exc))

        logger.info(
            "api_started",
            state_backend=self.settings.state_backend,
            model_loaded=self.model_loaded,
        )

    async def shutdown(self) -> None:
        """Release resources held for the process lifetime."""
        if self.store is not None:
            await self.store.stop()
            self.store = None
        self.manager = None
        self.engine = None
        logger.info("api_stopped")

    def require_manager(self) -> MatchStateManager:
        """Return the state manager, or fail loudly if startup did not run.

        Raises:
            RuntimeError: If the application state was never started.
        """
        if self.manager is None:
            raise RuntimeError("application state is not started")
        return self.manager


def get_state(request: Any) -> AppState:
    """Return the shared application state for a request."""
    state: AppState = request.app.state.apex
    return state
