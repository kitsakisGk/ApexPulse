"""WebSocket connection management.

Tracks which clients are connected and which match each one is watching, so a
tick for one match is sent only to the clients that asked for it rather than
fanned out to everyone.

Two failure modes drive the design:

* **A dead socket.** A browser tab closes without a clean handshake and the
  server only learns when a send fails. Sends must therefore treat failure as
  routine and prune rather than raise.
* **A slow client.** One client on a poor connection must not hold up the
  others, so sends run concurrently and a failure on one is isolated.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from apexpulse.logging import get_logger

if TYPE_CHECKING:
    from fastapi import WebSocket

logger = get_logger(__name__)

ALL_MATCHES = "*"
"""Subscription key for a client watching every match."""


@dataclass
class ConnectionStats:
    """Counters describing broadcast activity."""

    connections_opened: int = 0
    connections_closed: int = 0
    messages_sent: int = 0
    send_failures: int = 0

    @property
    def delivery_rate(self) -> float:
        """Share of attempted sends that succeeded; 1.0 before any were tried."""
        attempted = self.messages_sent + self.send_failures
        return 1.0 if attempted == 0 else self.messages_sent / attempted


@dataclass
class ConnectionManager:
    """Registry of live WebSocket clients, grouped by the match they watch."""

    _subscribers: dict[str, set[WebSocket]] = field(
        default_factory=lambda: defaultdict(set), init=False, repr=False
    )
    _stats: ConnectionStats = field(default_factory=ConnectionStats, init=False, repr=False)

    @property
    def stats(self) -> ConnectionStats:
        """Counters for this manager."""
        return self._stats

    @property
    def connection_count(self) -> int:
        """Distinct sockets currently connected."""
        return len({socket for group in self._subscribers.values() for socket in group})

    def subscriber_count(self, match_id: str) -> int:
        """Clients that would receive a message for ``match_id``.

        Includes clients watching every match, since they receive it too.
        """
        return len(
            self._subscribers.get(match_id, set()) | self._subscribers.get(ALL_MATCHES, set())
        )

    async def connect(self, websocket: WebSocket, match_id: str = ALL_MATCHES) -> None:
        """Accept ``websocket`` and subscribe it to ``match_id``."""
        await websocket.accept()
        self._subscribers[match_id].add(websocket)
        self._stats.connections_opened += 1
        logger.info(
            "websocket_connected",
            match_id=match_id,
            connections=self.connection_count,
        )

    def disconnect(self, websocket: WebSocket, match_id: str = ALL_MATCHES) -> None:
        """Remove ``websocket`` from ``match_id``'s subscribers."""
        group = self._subscribers.get(match_id)
        if group is not None:
            group.discard(websocket)
            if not group:
                del self._subscribers[match_id]
        self._stats.connections_closed += 1
        logger.info(
            "websocket_disconnected",
            match_id=match_id,
            connections=self.connection_count,
        )

    def _targets(self, match_id: str) -> set[WebSocket]:
        """Return every socket that should receive a message for ``match_id``."""
        return self._subscribers.get(match_id, set()) | self._subscribers.get(ALL_MATCHES, set())

    async def broadcast(self, match_id: str, payload: dict[str, Any]) -> int:
        """Send ``payload`` to every client watching ``match_id``.

        Returns:
            The number of clients that received it.

        Sends run concurrently so one slow client cannot delay the rest, and a
        socket that fails is pruned rather than raising: a closed tab is the
        normal case, not an error worth propagating.
        """
        targets = self._targets(match_id)
        if not targets:
            return 0

        results = await asyncio.gather(
            *(self._send(socket, payload) for socket in targets),
            return_exceptions=True,
        )

        delivered = 0
        for socket, result in zip(targets, results, strict=True):
            if result is True:
                delivered += 1
            else:
                self._prune(socket)

        self._stats.messages_sent += delivered
        self._stats.send_failures += len(targets) - delivered
        return delivered

    async def _send(self, websocket: WebSocket, payload: dict[str, Any]) -> bool:
        """Send one payload, reporting failure rather than raising."""
        try:
            await websocket.send_json(payload)
        except Exception:  # a closed socket is routine, not exceptional
            return False
        return True

    def _prune(self, websocket: WebSocket) -> None:
        """Drop a socket from every subscription it holds."""
        for match_id in list(self._subscribers):
            group = self._subscribers[match_id]
            group.discard(websocket)
            if not group:
                del self._subscribers[match_id]

    async def close_all(self) -> None:
        """Close every connection, for shutdown."""
        sockets = {socket for group in self._subscribers.values() for socket in group}
        for socket in sockets:
            # A socket that is already gone is a successful close.
            with contextlib.suppress(Exception):
                await socket.close()
        self._subscribers.clear()
        logger.info("websockets_closed", count=len(sockets))
