"""Tests for the WebSocket channel and the broadcaster.

A push channel fails differently from a request/response API: the interesting
cases are a client that vanishes mid-stream, a slow client holding up the rest,
and frames going to the wrong subscriber. These focus there.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from apexpulse.api import ALL_MATCHES, ConnectionManager, StreamBroadcaster, create_app
from apexpulse.api.dependencies import AppState
from apexpulse.api.ws_schemas import MessageType
from apexpulse.config import Settings
from apexpulse.producer import MatchSimulator
from apexpulse.schemas.enums import RoundEndReason, Team
from apexpulse.schemas.events import MatchEndEvent, RoundEndEvent, TickEvent

BASE_TIME = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


class FakeSocket:
    """A WebSocket stand-in that records what it was sent.

    Avoids a real connection so the manager's fan-out and pruning can be tested
    directly, including the failure path a live socket only reaches by breaking.
    """

    def __init__(self, *, fail: bool = False) -> None:
        self.sent: list[dict[str, Any]] = []
        self.fail = fail
        self.accepted = False
        self.closed = False

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, payload: dict[str, Any]) -> None:
        if self.fail:
            raise ConnectionResetError("client is gone")
        self.sent.append(payload)

    async def close(self) -> None:
        self.closed = True


# -- Connection manager -------------------------------------------------------


async def test_connecting_accepts_and_registers() -> None:
    manager = ConnectionManager()
    socket = FakeSocket()

    await manager.connect(socket, "m-1")  # type: ignore[arg-type]

    assert socket.accepted is True
    assert manager.connection_count == 1
    assert manager.subscriber_count("m-1") == 1


async def test_a_broadcast_reaches_only_that_match() -> None:
    """A tick for one match must not be sent to a client watching another."""
    manager = ConnectionManager()
    watching_one = FakeSocket()
    watching_other = FakeSocket()

    await manager.connect(watching_one, "m-1")  # type: ignore[arg-type]
    await manager.connect(watching_other, "m-2")  # type: ignore[arg-type]

    delivered = await manager.broadcast("m-1", {"type": "tick"})

    assert delivered == 1
    assert len(watching_one.sent) == 1
    assert watching_other.sent == []


async def test_a_wildcard_client_receives_every_match() -> None:
    manager = ConnectionManager()
    wildcard = FakeSocket()
    specific = FakeSocket()

    await manager.connect(wildcard, ALL_MATCHES)  # type: ignore[arg-type]
    await manager.connect(specific, "m-1")  # type: ignore[arg-type]

    await manager.broadcast("m-1", {"n": 1})
    await manager.broadcast("m-2", {"n": 2})

    assert len(wildcard.sent) == 2
    assert len(specific.sent) == 1


async def test_a_dead_socket_is_pruned_rather_than_raising() -> None:
    """A closed tab is the normal case, not an error worth propagating."""
    manager = ConnectionManager()
    healthy = FakeSocket()
    dead = FakeSocket(fail=True)

    await manager.connect(healthy, "m-1")  # type: ignore[arg-type]
    await manager.connect(dead, "m-1")  # type: ignore[arg-type]

    delivered = await manager.broadcast("m-1", {"type": "tick"})

    assert delivered == 1, "the healthy client still received it"
    assert manager.subscriber_count("m-1") == 1, "the dead one was pruned"
    assert manager.stats.send_failures == 1


async def test_one_failing_client_does_not_block_the_others() -> None:
    manager = ConnectionManager()
    sockets = [FakeSocket(fail=index == 2) for index in range(5)]
    for socket in sockets:
        await manager.connect(socket, "m-1")  # type: ignore[arg-type]

    delivered = await manager.broadcast("m-1", {"type": "tick"})

    assert delivered == 4
    assert all(len(socket.sent) == 1 for socket in sockets if not socket.fail)


async def test_disconnecting_removes_the_subscription() -> None:
    manager = ConnectionManager()
    socket = FakeSocket()
    await manager.connect(socket, "m-1")  # type: ignore[arg-type]

    manager.disconnect(socket, "m-1")  # type: ignore[arg-type]

    assert manager.connection_count == 0
    assert await manager.broadcast("m-1", {"n": 1}) == 0


async def test_broadcasting_to_nobody_is_free() -> None:
    manager = ConnectionManager()

    assert await manager.broadcast("nobody-watching", {"n": 1}) == 0
    assert manager.stats.messages_sent == 0


async def test_closing_all_disconnects_every_client() -> None:
    manager = ConnectionManager()
    sockets = [FakeSocket() for _ in range(3)]
    for index, socket in enumerate(sockets):
        await manager.connect(socket, f"m-{index}")  # type: ignore[arg-type]

    await manager.close_all()

    assert all(socket.closed for socket in sockets)
    assert manager.connection_count == 0


async def test_delivery_rate_starts_optimistic() -> None:
    """No attempts means no failures, not a zero rate."""
    assert ConnectionManager().stats.delivery_rate == 1.0


# -- The endpoint -------------------------------------------------------------


def test_connecting_receives_a_welcome_frame() -> None:
    from fastapi.testclient import TestClient

    app = create_app(Settings(state_backend="memory"))

    with TestClient(app) as client, client.websocket_connect("/ws/m-1") as socket:
        welcome = socket.receive_json()

    assert welcome["type"] == MessageType.WELCOME
    assert welcome["match_id"] == "m-1"
    assert welcome["server_version"]


def test_the_wildcard_endpoint_reports_its_scope() -> None:
    from fastapi.testclient import TestClient

    app = create_app(Settings(state_backend="memory"))

    with TestClient(app) as client, client.websocket_connect("/ws") as socket:
        welcome = socket.receive_json()

    assert welcome["match_id"] == ALL_MATCHES


def test_two_clients_on_one_match_are_both_tracked() -> None:
    from fastapi.testclient import TestClient

    app = create_app(Settings(state_backend="memory"))

    # Nested deliberately: the second connection must open while the first is
    # still live, which is the state the assertion checks.
    with TestClient(app) as client:  # noqa: SIM117
        with client.websocket_connect("/ws/m-1") as first:
            first.receive_json()
            with client.websocket_connect("/ws/m-1") as second:
                second.receive_json()
                manager = app.state.apex.connections
                assert manager.subscriber_count("m-1") == 2


def test_disconnecting_is_recorded() -> None:
    from fastapi.testclient import TestClient

    app = create_app(Settings(state_backend="memory"))

    with TestClient(app) as client:
        with client.websocket_connect("/ws/m-1") as socket:
            socket.receive_json()

        stats = app.state.apex.connections.stats
        assert stats.connections_opened == 1
        assert stats.connections_closed == 1
        assert app.state.apex.connections.connection_count == 0


# -- Broadcaster --------------------------------------------------------------


@pytest.fixture
async def broadcaster() -> tuple[StreamBroadcaster, ConnectionManager]:
    """A broadcaster with no model, so frames are exercised without inference."""
    state = AppState(settings=Settings(state_backend="memory"))
    return StreamBroadcaster(state=state), state.connections


async def test_an_unwatched_match_costs_nothing(broadcaster) -> None:
    """Projection work is skipped when nobody is subscribed."""
    caster, _ = broadcaster

    for event in MatchSimulator(seed=3, tick_rate_hz=2.0, start_time=BASE_TIME).run():
        await caster.handle(event)
        if caster.stats.ticks_seen >= 40:
            break

    assert caster.stats.ticks_seen == 40
    assert caster.stats.state_frames == 0
    assert caster.stats.prediction_frames == 0


async def test_state_frames_are_throttled(broadcaster) -> None:
    """Ticks arrive at 8 Hz; a browser cannot repaint a player grid that often."""
    caster, manager = broadcaster
    socket = FakeSocket()
    await manager.connect(socket, "apex-001")  # type: ignore[arg-type]

    for event in MatchSimulator(seed=3, tick_rate_hz=8.0, start_time=BASE_TIME).run():
        await caster.handle(event)
        if caster.stats.ticks_seen >= 120:
            break

    assert caster.stats.throttled > 0, "no ticks were suppressed"
    assert caster.stats.state_frames < caster.stats.ticks_seen


async def test_a_round_end_is_never_throttled(broadcaster) -> None:
    """A round ends once; suppressing it would lose the result."""
    caster, manager = broadcaster
    socket = FakeSocket()
    await manager.connect(socket, "m-1")  # type: ignore[arg-type]

    for number in (1, 2, 3):
        await caster.handle(
            RoundEndEvent(
                match_id="m-1",
                timestamp=BASE_TIME,
                sequence=number,
                round_number=number,
                winner=Team.CT,
                reason=RoundEndReason.T_ELIMINATED,
                score_ct=number,
                score_t=0,
            )
        )

    assert caster.stats.round_frames == 3
    assert len(socket.sent) == 3
    assert all(frame["type"] == MessageType.ROUND_END for frame in socket.sent)


async def test_a_match_end_frame_carries_the_result(broadcaster) -> None:
    caster, manager = broadcaster
    socket = FakeSocket()
    await manager.connect(socket, "m-1")  # type: ignore[arg-type]

    await caster.handle(
        MatchEndEvent(
            match_id="m-1",
            timestamp=BASE_TIME,
            sequence=999,
            winner=Team.CT,
            score_ct=13,
            score_t=7,
        )
    )

    assert len(socket.sent) == 1
    frame = socket.sent[0]
    assert frame["type"] == MessageType.MATCH_END
    assert frame["winner"] == "CT"
    assert frame["score_ct"] == 13


async def test_a_finished_match_releases_its_window(broadcaster) -> None:
    caster, manager = broadcaster
    socket = FakeSocket()
    await manager.connect(socket, "m-1")  # type: ignore[arg-type]

    caster.window_for("m-1")
    await caster.handle(
        MatchEndEvent(
            match_id="m-1",
            timestamp=BASE_TIME,
            sequence=999,
            winner=Team.T,
            score_ct=7,
            score_t=13,
        )
    )

    assert caster.window_for("m-1").tick_count == 0


async def test_frames_carry_the_match_they_belong_to(broadcaster) -> None:
    """A client watching two matches must be able to tell frames apart."""
    caster, manager = broadcaster
    wildcard = FakeSocket()
    await manager.connect(wildcard, ALL_MATCHES)  # type: ignore[arg-type]

    for match_id in ("m-1", "m-2"):
        await caster.handle(
            RoundEndEvent(
                match_id=match_id,
                timestamp=BASE_TIME,
                sequence=1,
                round_number=1,
                winner=Team.CT,
                reason=RoundEndReason.T_ELIMINATED,
                score_ct=1,
                score_t=0,
            )
        )

    assert {frame["match_id"] for frame in wildcard.sent} == {"m-1", "m-2"}


async def test_throttle_rate_is_zero_before_any_ticks(broadcaster) -> None:
    caster, _ = broadcaster

    assert caster.stats.throttle_rate == 0.0


async def test_a_tick_reaches_a_subscriber(broadcaster) -> None:
    """The join that matters: a consumed event becomes a pushed frame."""
    caster, manager = broadcaster
    socket = FakeSocket()
    await manager.connect(socket, "apex-001")  # type: ignore[arg-type]

    for event in MatchSimulator(seed=3, tick_rate_hz=2.0, start_time=BASE_TIME).run():
        await caster.handle(event)
        if isinstance(event, TickEvent) and socket.sent:
            break

    assert socket.sent, "no frame reached the subscriber"
    assert socket.sent[0]["match_id"] == "apex-001"
