"""FastAPI application, REST routes, and WebSocket broadcast."""

from apexpulse.api.app import create_app
from apexpulse.api.dependencies import AppState, get_state

__all__ = ["AppState", "create_app", "get_state"]
