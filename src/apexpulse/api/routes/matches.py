"""Match state and history routes.

These are the read paths the dashboard polls before a WebSocket connection is
established, and the ones any other consumer uses. Each reads from the live
state store rather than replaying the event stream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Request, status

from apexpulse.api.dependencies import get_state
from apexpulse.api.schemas import (
    ErrorResponse,
    MatchHistoryResponse,
    MatchListItem,
    MatchListResponse,
    MatchSnapshot,
    MomentumSummary,
    PlayerSummary,
    PredictionSummary,
    RoundOutcome,
    RoundSummary,
    ScoreLine,
    TeamSummary,
)
from apexpulse.schemas.enums import Team

if TYPE_CHECKING:
    from apexpulse.api.dependencies import AppState
    from apexpulse.storage.match_state import LiveSnapshot, MatchHistory

router = APIRouter(prefix="/matches", tags=["matches"])

NOT_FOUND: dict[int | str, dict[str, Any]] = {status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}}


def _team_summary(snapshot: LiveSnapshot, team: Team) -> TeamSummary:
    """Project one side's state into the wire model."""
    state = snapshot.state
    players = state.players_on(team)
    economy = state.economy_ct if team is Team.CT else state.economy_t

    return TeamSummary(
        alive=sum(1 for player in players if player.is_alive),
        health=sum(player.health for player in players),
        money=economy.money,
        equipment_value=economy.equipment_value,
        consecutive_losses=economy.consecutive_losses,
    )


def _to_snapshot(snapshot: LiveSnapshot, state: AppState) -> MatchSnapshot:
    """Project stored state into the API response, scoring it if a model is loaded."""
    match_state = snapshot.state
    round_state = match_state.round_state

    prediction: PredictionSummary | None = None
    if state.engine is not None:
        from apexpulse.stream.window import WindowedMetrics

        metrics = WindowedMetrics(
            kills_ct=snapshot.momentum.kills_ct,
            kills_t=snapshot.momentum.kills_t,
            span_seconds=snapshot.momentum.window_seconds,
        )
        scored = state.engine.predict(match_state, metrics)
        if scored.scored:
            state.predictions_served += 1
            prediction = PredictionSummary(
                ct_win_probability=scored.ct_win_probability,
                t_win_probability=scored.t_win_probability,
                favoured_side=scored.favoured_side,
                confidence=scored.confidence,
                latency_ms=scored.latency_ms,
            )

    return MatchSnapshot(
        match_id=snapshot.match_id,
        map_name=match_state.map_name.value,
        sequence=snapshot.sequence,
        timestamp=snapshot.timestamp,
        score=ScoreLine(ct=match_state.score_ct, t=match_state.score_t),
        round=RoundSummary(
            number=round_state.round_number,
            phase=round_state.phase.value,
            seconds_remaining=round_state.seconds_remaining,
            bomb_planted=round_state.bomb_planted,
            bomb_seconds_remaining=round_state.bomb_seconds_remaining,
        ),
        ct=_team_summary(snapshot, Team.CT),
        t=_team_summary(snapshot, Team.T),
        players=tuple(
            PlayerSummary(
                player_id=player.player_id,
                name=player.name,
                team=player.team.value,
                health=player.health,
                armour=player.armour,
                money=player.money,
                weapon=player.primary_weapon.value if player.primary_weapon else None,
                alive=player.is_alive,
                kills=player.kills,
                deaths=player.deaths,
            )
            for player in match_state.players
        ),
        momentum=MomentumSummary(
            kills_ct=snapshot.momentum.kills_ct,
            kills_t=snapshot.momentum.kills_t,
            kill_delta=snapshot.momentum.kill_delta,
            window_seconds=snapshot.momentum.window_seconds,
        ),
        prediction=prediction,
    )


def _to_history(history: MatchHistory) -> MatchHistoryResponse:
    """Project stored round history into the API response."""
    streak = history.current_streak()

    return MatchHistoryResponse(
        match_id=history.match_id,
        rounds_played=history.rounds_played,
        rounds=tuple(
            RoundOutcome(
                round_number=result.round_number,
                winner=result.winner.value,
                reason=result.reason.value,
                score=ScoreLine(ct=result.score_ct, t=result.score_t),
                timestamp=result.timestamp,
            )
            for result in history.rounds
        ),
        ct_wins=history.wins_for(Team.CT),
        t_wins=history.wins_for(Team.T),
        streak_side=streak[0].value if streak else None,
        streak_length=streak[1] if streak else 0,
    )


@router.get("", response_model=MatchListResponse, summary="List live matches")
async def list_matches(request: Request) -> MatchListResponse:
    """Return every match currently holding live state."""
    state = get_state(request)
    manager = state.require_manager()

    items: list[MatchListItem] = []
    for snapshot in await manager.live_snapshots():
        match_state = snapshot.state

        probability: float | None = None
        if state.engine is not None:
            scored = state.engine.predict(match_state)
            probability = scored.ct_win_probability if scored.scored else None

        items.append(
            MatchListItem(
                match_id=snapshot.match_id,
                map_name=match_state.map_name.value,
                score=ScoreLine(ct=match_state.score_ct, t=match_state.score_t),
                round_number=match_state.round_state.round_number,
                phase=match_state.round_state.phase.value,
                ct_win_probability=probability,
            )
        )

    return MatchListResponse(count=len(items), matches=tuple(items))


@router.get(
    "/{match_id}",
    response_model=MatchSnapshot,
    responses=NOT_FOUND,
    summary="Current state of one match",
)
async def get_match(match_id: str, request: Request) -> MatchSnapshot:
    """Return the latest snapshot for ``match_id``.

    Raises:
        HTTPException: 404 when the match has no live state, which also covers a
            match whose state has expired.
    """
    state = get_state(request)
    snapshot = await state.require_manager().get_snapshot(match_id)

    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no live state for match {match_id!r}",
        )

    return _to_snapshot(snapshot, state)


@router.get(
    "/{match_id}/history",
    response_model=MatchHistoryResponse,
    summary="Completed rounds for one match",
)
async def get_match_history(match_id: str, request: Request) -> MatchHistoryResponse:
    """Return every completed round for ``match_id``.

    An unknown match returns an empty history rather than a 404: "no rounds yet"
    and "never heard of it" are the same answer to a dashboard drawing a chart.
    """
    state = get_state(request)
    history = await state.require_manager().get_history(match_id)
    return _to_history(history)
