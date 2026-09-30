"""WebSocket routes.

Clients connect here and receive updates as they happen rather than polling.
Two endpoints: one scoped to a single match, one carrying every match, because a
dashboard showing a match grid wants the latter and a match page wants the
former.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from apexpulse import __version__
from apexpulse.api.connections import ALL_MATCHES
from apexpulse.api.routes.matches import to_snapshot
from apexpulse.api.ws_schemas import WelcomeMessage
from apexpulse.logging import get_logger

if TYPE_CHECKING:
    from apexpulse.api.dependencies import AppState

logger = get_logger(__name__)

router = APIRouter(tags=["stream"])


async def _welcome(websocket: WebSocket, state: AppState, match_id: str) -> None:
    """Send the opening frame, including current state when there is any."""
    snapshot = None
    if match_id != ALL_MATCHES and state.manager is not None:
        stored = await state.manager.get_snapshot(match_id)
        if stored is not None:
            snapshot = to_snapshot(stored, state)

    message = WelcomeMessage(
        match_id=match_id,
        server_version=__version__,
        model_loaded=state.model_loaded,
        snapshot=snapshot,
    )
    await websocket.send_json(message.model_dump(mode="json"))


async def _serve(websocket: WebSocket, match_id: str) -> None:
    """Hold a connection open until the client disconnects.

    The server pushes; it does not expect the client to say anything. Reads
    still happen so a disconnect is noticed promptly rather than on the next
    failed send, which could be seconds away in a quiet match.
    """
    state: AppState = websocket.app.state.apex
    await state.connections.connect(websocket, match_id)

    try:
        await _welcome(websocket, state, match_id)
        while True:
            # Any inbound text is ignored; the call exists to detect the close.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # a broken socket must not take down the server
        logger.warning("websocket_error", match_id=match_id, error=str(exc))
    finally:
        state.connections.disconnect(websocket, match_id)


@router.websocket("/ws")
async def stream_all(websocket: WebSocket) -> None:
    """Stream updates for every live match."""
    await _serve(websocket, ALL_MATCHES)


@router.websocket("/ws/{match_id}")
async def stream_match(websocket: WebSocket, match_id: str) -> None:
    """Stream updates for one match."""
    await _serve(websocket, match_id)
