"""Tests for the CS2 domain models.

These schemas are the contract every later stage depends on, so the assertions
focus on the invariants a malformed producer could otherwise violate silently.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from apexpulse.schemas import (
    MatchState,
    PlayerState,
    Position,
    RoundEndReason,
    RoundPhase,
    RoundState,
    Team,
    TeamEconomy,
    Weapon,
    constants,
    weapons_for,
)
from apexpulse.schemas.enums import MapName
from apexpulse.schemas.events import (
    BombPlantedEvent,
    KillEvent,
    MatchEndEvent,
    MatchStartEvent,
    RoundEndEvent,
    TickEvent,
    parse_event,
    serialise_event,
)

# -- Fixtures -----------------------------------------------------------------


def make_player(
    player_id: str = "p1",
    team: Team = Team.CT,
    *,
    health: int = 100,
    money: int = 4_000,
    **overrides: object,
) -> PlayerState:
    """Build a valid player, overriding only what a test cares about."""
    fields: dict[str, object] = {
        "player_id": player_id,
        "name": f"player_{player_id}",
        "team": team,
        "health": health,
        "money": money,
    }
    fields.update(overrides)
    return PlayerState(**fields)  # type: ignore[arg-type]


def make_state(**overrides: object) -> MatchState:
    """Build a valid five-a-side match snapshot."""
    players = tuple(
        make_player(f"ct{i}", Team.CT) for i in range(constants.PLAYERS_PER_TEAM)
    ) + tuple(make_player(f"t{i}", Team.T) for i in range(constants.PLAYERS_PER_TEAM))

    fields: dict[str, object] = {
        "match_id": "m-1",
        "map_name": MapName.MIRAGE,
        "round_state": RoundState(round_number=1, phase=RoundPhase.LIVE, seconds_remaining=90.0),
        "players": players,
        "economy_ct": TeamEconomy(team=Team.CT, money=20_000, equipment_value=15_000),
        "economy_t": TeamEconomy(team=Team.T, money=18_000, equipment_value=14_000),
    }
    fields.update(overrides)
    return MatchState(**fields)  # type: ignore[arg-type]


# -- Enums --------------------------------------------------------------------


def test_team_opponent_is_symmetric() -> None:
    assert Team.CT.opponent is Team.T
    assert Team.T.opponent is Team.CT
    assert Team.CT.opponent.opponent is Team.CT


def test_side_specific_loadouts_are_disjoint_where_it_matters() -> None:
    """The AK is T-only and the M4 CT-only; shared utility appears in both."""
    assert Weapon.AK47 in weapons_for(Team.T)
    assert Weapon.AK47 not in weapons_for(Team.CT)
    assert Weapon.M4A4 in weapons_for(Team.CT)
    assert Weapon.M4A4 not in weapons_for(Team.T)
    assert Weapon.AWP in weapons_for(Team.CT) & weapons_for(Team.T)


def test_every_weapon_has_a_declared_cost() -> None:
    from apexpulse.schemas.enums import WEAPON_COST

    assert set(WEAPON_COST) == set(Weapon)


# -- Economy rules ------------------------------------------------------------


@pytest.mark.parametrize(
    ("losses", "expected"),
    [(0, 0), (1, 1_400), (2, 1_900), (3, 2_400), (4, 2_900), (5, 3_400), (9, 3_400)],
)
def test_loss_bonus_climbs_then_caps(losses: int, expected: int) -> None:
    assert constants.loss_bonus(losses) == expected


def test_full_buy_reflects_combined_team_money() -> None:
    rich = TeamEconomy(team=Team.CT, money=25_000, equipment_value=0)
    poor = TeamEconomy(team=Team.CT, money=3_000, equipment_value=0)

    assert rich.is_full_buy is True
    assert poor.is_full_buy is False


# -- PlayerState --------------------------------------------------------------


def test_player_liveness_derives_from_health() -> None:
    assert make_player(health=1).is_alive is True
    assert make_player(health=0).is_alive is False


def test_dead_players_cannot_retain_equipment() -> None:
    with pytest.raises(ValidationError, match="dead player"):
        make_player(health=0, armour=100)


def test_helmet_requires_armour() -> None:
    with pytest.raises(ValidationError, match="has_helmet requires"):
        make_player(armour=0, has_helmet=True)


def test_only_cts_carry_defuse_kits() -> None:
    with pytest.raises(ValidationError, match="defuse kit"):
        make_player(team=Team.T, has_defuse_kit=True)

    assert make_player(team=Team.CT, has_defuse_kit=True).has_defuse_kit is True


def test_money_is_capped_at_the_game_maximum() -> None:
    with pytest.raises(ValidationError):
        make_player(money=constants.MAX_MONEY + 1)


def test_unknown_fields_are_rejected() -> None:
    """extra='forbid' catches producer typos at the boundary."""
    with pytest.raises(ValidationError):
        make_player(helth=100)


def test_models_are_immutable() -> None:
    player = make_player()

    with pytest.raises(ValidationError):
        player.health = 50  # type: ignore[misc]


# -- Position -----------------------------------------------------------------


def test_distance_is_euclidean() -> None:
    origin = Position(x=0.0, y=0.0, z=0.0)
    point = Position(x=3.0, y=4.0, z=0.0)

    assert origin.distance_to(point) == pytest.approx(5.0)


def test_positions_outside_the_world_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Position(x=99_999.0, y=0.0, z=0.0)


# -- RoundState ---------------------------------------------------------------


def test_planted_bomb_requires_a_countdown() -> None:
    with pytest.raises(ValidationError, match="bomb_seconds_remaining is required"):
        RoundState(
            round_number=1,
            phase=RoundPhase.BOMB_PLANTED,
            seconds_remaining=0.0,
            bomb_planted=True,
        )


def test_countdown_without_a_plant_is_rejected() -> None:
    with pytest.raises(ValidationError, match="only valid while"):
        RoundState(
            round_number=1,
            phase=RoundPhase.LIVE,
            seconds_remaining=30.0,
            bomb_seconds_remaining=20.0,
        )


def test_planted_phase_requires_the_flag() -> None:
    with pytest.raises(ValidationError, match="requires bomb_planted"):
        RoundState(round_number=1, phase=RoundPhase.BOMB_PLANTED, seconds_remaining=10.0)


def test_a_valid_plant_round_trips() -> None:
    state = RoundState(
        round_number=5,
        phase=RoundPhase.BOMB_PLANTED,
        seconds_remaining=0.0,
        bomb_planted=True,
        bomb_seconds_remaining=32.5,
    )

    assert state.bomb_seconds_remaining == 32.5


# -- MatchState ---------------------------------------------------------------


def test_alive_counts_and_man_advantage_track_health() -> None:
    players = tuple(make_player(f"ct{i}", Team.CT) for i in range(5)) + tuple(
        make_player(f"t{i}", Team.T, health=0 if i < 2 else 100) for i in range(5)
    )
    state = make_state(players=players)

    assert state.alive_ct == 5
    assert state.alive_t == 3
    assert state.man_advantage == 2


def test_duplicate_player_ids_are_rejected() -> None:
    players = (make_player("dup", Team.CT), make_player("dup", Team.T))

    with pytest.raises(ValidationError, match="unique"):
        make_state(players=players)


def test_economies_must_describe_their_own_side() -> None:
    with pytest.raises(ValidationError, match="CT and T respectively"):
        make_state(economy_ct=TeamEconomy(team=Team.T, money=0, equipment_value=0))


def test_players_on_filters_by_side() -> None:
    state = make_state()

    assert len(state.players_on(Team.CT)) == 5
    assert all(p.team is Team.T for p in state.players_on(Team.T))


# -- Events -------------------------------------------------------------------


def test_tick_state_must_belong_to_the_event_match() -> None:
    with pytest.raises(ValidationError, match=r"state\.match_id"):
        TickEvent(match_id="other", sequence=1, state=make_state())


def test_assister_cannot_be_the_killer_or_the_victim() -> None:
    with pytest.raises(ValidationError, match="assister_id must differ"):
        KillEvent(
            match_id="m-1",
            sequence=1,
            round_number=1,
            killer_id="a",
            victim_id="b",
            assister_id="a",
            weapon=Weapon.AK47,
            victim_team=Team.CT,
        )


def test_kills_may_have_no_killer() -> None:
    """Suicides and world damage produce a victim with no killer."""
    event = KillEvent(
        match_id="m-1",
        sequence=1,
        round_number=1,
        victim_id="b",
        weapon=Weapon.KNIFE,
        victim_team=Team.T,
    )

    assert event.killer_id is None


@pytest.mark.parametrize(
    ("winner", "reason"),
    [
        (Team.CT, RoundEndReason.BOMB_EXPLODED),
        (Team.CT, RoundEndReason.CT_ELIMINATED),
        (Team.T, RoundEndReason.BOMB_DEFUSED),
        (Team.T, RoundEndReason.T_ELIMINATED),
        (Team.T, RoundEndReason.TIME_EXPIRED),
    ],
)
def test_impossible_round_outcomes_are_rejected(winner: Team, reason: RoundEndReason) -> None:
    with pytest.raises(ValidationError):
        RoundEndEvent(
            match_id="m-1",
            sequence=1,
            round_number=1,
            winner=winner,
            reason=reason,
            score_ct=1,
            score_t=0,
        )


@pytest.mark.parametrize(
    ("winner", "reason"),
    [
        (Team.CT, RoundEndReason.T_ELIMINATED),
        (Team.CT, RoundEndReason.BOMB_DEFUSED),
        (Team.CT, RoundEndReason.TIME_EXPIRED),
        (Team.T, RoundEndReason.CT_ELIMINATED),
        (Team.T, RoundEndReason.BOMB_EXPLODED),
    ],
)
def test_legitimate_round_outcomes_are_accepted(winner: Team, reason: RoundEndReason) -> None:
    event = RoundEndEvent(
        match_id="m-1",
        sequence=1,
        round_number=1,
        winner=winner,
        reason=reason,
        score_ct=1,
        score_t=0,
    )

    assert event.winner is winner


def test_match_end_winner_must_match_the_scoreline() -> None:
    with pytest.raises(ValidationError, match="does not match the final score"):
        MatchEndEvent(match_id="m-1", sequence=99, winner=Team.CT, score_ct=10, score_t=13)


def test_bomb_site_is_restricted_to_a_or_b() -> None:
    with pytest.raises(ValidationError):
        BombPlantedEvent(
            match_id="m-1",
            sequence=1,
            round_number=1,
            planter_id="t1",
            site="C",  # type: ignore[arg-type]
        )


# -- Serialisation ------------------------------------------------------------


def test_events_round_trip_through_the_discriminated_union() -> None:
    original = TickEvent(match_id="m-1", sequence=42, state=make_state())

    restored = parse_event(serialise_event(original))

    assert isinstance(restored, TickEvent)
    assert restored.sequence == 42
    assert restored.state.match_id == "m-1"


def test_the_discriminator_selects_the_concrete_type() -> None:
    events = [
        MatchStartEvent(
            match_id="m-1",
            sequence=0,
            map_name=MapName.NUKE,
            team_ct_name="Vitality",
            team_t_name="NAVI",
        ),
        KillEvent(
            match_id="m-1",
            sequence=1,
            round_number=1,
            killer_id="ct0",
            victim_id="t0",
            weapon=Weapon.AWP,
            victim_team=Team.T,
        ),
        MatchEndEvent(match_id="m-1", sequence=2, winner=Team.CT, score_ct=13, score_t=7),
    ]

    for event in events:
        assert type(parse_event(serialise_event(event))) is type(event)


def test_timestamps_survive_serialisation_as_utc() -> None:
    stamped = datetime(2026, 9, 8, 12, 30, 45, tzinfo=UTC)
    event = MatchEndEvent(
        match_id="m-1", sequence=1, timestamp=stamped, winner=Team.T, score_ct=5, score_t=13
    )

    restored = parse_event(serialise_event(event))

    assert restored.timestamp == stamped


def test_malformed_payloads_raise_validation_errors() -> None:
    with pytest.raises(ValidationError):
        parse_event(b'{"event_type": "not_a_real_event", "match_id": "m-1", "sequence": 0}')


def test_serialised_payload_is_compact_enough_for_a_high_tick_rate() -> None:
    """A full 10-player snapshot must stay small; ticks are the hot path."""
    payload = serialise_event(TickEvent(match_id="m-1", sequence=1, state=make_state()))

    assert len(payload) < 8_192


def test_computed_fields_are_emitted_but_never_accepted_as_input() -> None:
    """Regression: strict models must accept their own serialised output.

    ``is_alive`` and friends are serialised for the dashboard's benefit, so a
    payload replayed through the parser carries them. With ``extra='forbid'`` and
    no filtering, parsing our own output would fail.
    """
    state = make_state()
    dumped = state.model_dump()

    assert "alive_ct" in dumped, "computed fields must reach the dashboard"
    assert MatchState.model_validate(dumped) == state


def test_derived_values_cannot_be_forged_by_a_producer() -> None:
    """A payload claiming a false computed value is ignored, not trusted."""
    player = PlayerState.model_validate(
        {
            "player_id": "p1",
            "name": "p1",
            "team": Team.CT,
            "health": 0,
            "money": 0,
            "is_alive": True,
        }
    )

    assert player.is_alive is False
