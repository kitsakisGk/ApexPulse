"""Synthetic CS2 match simulator.

Drives a match through its real structure — freezetime, buy decisions, duels,
plants, defuses, and the MR12 scoreline — emitting the telemetry events the rest of
the pipeline consumes.

The simulation is deterministic given a seed. That matters more than realism: it
makes the producer tests reproducible, lets the Day 7 training set be regenerated
exactly, and means a bug found in a match can be replayed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from apexpulse.schemas import constants
from apexpulse.schemas.enums import (
    WEAPON_COST,
    MapName,
    RoundEndReason,
    RoundPhase,
    Team,
    Weapon,
    weapons_for,
)
from apexpulse.schemas.events import (
    BombDefusedEvent,
    BombPlantedEvent,
    KillEvent,
    MatchEndEvent,
    MatchStartEvent,
    RoundEndEvent,
    RoundStartEvent,
    TelemetryEvent,
    TickEvent,
)
from apexpulse.schemas.models import MatchState, PlayerState, Position, RoundState, TeamEconomy

if TYPE_CHECKING:
    from collections.abc import Iterator

# -- Tuning -------------------------------------------------------------------
# Calibrated against 12 seeded matches (229 rounds) to land inside real
# competitive rates: 52.4% CT round wins (real ~51-53%) and a 55.0% plant rate
# (real ~45-55%), with all five round outcomes represented.
#
# The two rates are coupled: an unplanted round that runs out of time is a CT win
# by definition, so lowering the plant rate raises CT's share. Retune them
# together and re-measure across seeds, never from a single match.

DUEL_INTERVAL_SECONDS = 14.0
"""Mean gap between engagements while a round is live.

Tuned so a round rarely wipes a side before the plant window opens; at 8s the
simulator eliminated a team early in almost every round and plants never landed.
"""

CT_DUEL_BASE_WIN_RATE = 0.485
EQUIPMENT_ADVANTAGE_WEIGHT = 0.18
"""How strongly an equipment edge tilts a duel; keeps buys meaningful."""

PLANT_ATTEMPT_RATE = 0.56
"""Share of rounds in which the Ts commit to a plant, given they survive to try.

Expressed per round rather than per tick: a per-tick probability compounds over a
115-second round to near-certainty, which produced a plant in every round.
"""

PLANT_WINDOW_START = 0.20
PLANT_WINDOW_END = 0.65
"""Fraction of the round clock within which a committed plant lands."""

POST_PLANT_DEFUSE_RATE = 0.45
HEADSHOT_RATE = 0.42

ARMOUR_BUY_THRESHOLD = 1_000
KIT_BUY_THRESHOLD = 3_000

_MAP_EXTENT = 2_000.0
"""Half-width of the playable area used when sampling positions."""


@dataclass
class _Player:
    """Mutable per-player simulation state, projected into `PlayerState` per tick."""

    player_id: str
    name: str
    team: Team
    health: int = constants.MAX_HEALTH
    armour: int = 0
    has_helmet: bool = False
    has_defuse_kit: bool = False
    money: int = constants.STARTING_MONEY
    weapon: Weapon | None = None
    position: Position | None = None
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    damage_dealt: int = 0

    @property
    def alive(self) -> bool:
        return self.health > 0

    @property
    def equipment_value(self) -> int:
        """Buy value of everything currently carried."""
        value = WEAPON_COST.get(self.weapon, 0) if self.weapon else 0
        if self.armour:
            value += constants.KEVLAR_HELMET_COST if self.has_helmet else constants.KEVLAR_COST
        if self.has_defuse_kit:
            value += constants.DEFUSE_KIT_COST
        return value

    def snapshot(self) -> PlayerState:
        """Project into the immutable wire model."""
        return PlayerState(
            player_id=self.player_id,
            name=self.name,
            team=self.team,
            health=self.health,
            armour=self.armour,
            has_helmet=self.has_helmet,
            has_defuse_kit=self.has_defuse_kit,
            money=self.money,
            primary_weapon=self.weapon,
            position=self.position,
            kills=self.kills,
            deaths=self.deaths,
            assists=self.assists,
            damage_dealt=self.damage_dealt,
        )

    def reset_for_round(self) -> None:
        """Restore health and clear per-round equipment before the buy phase."""
        self.health = constants.MAX_HEALTH
        self.armour = 0
        self.has_helmet = False
        self.has_defuse_kit = False
        self.weapon = None


@dataclass
class _TeamLedger:
    """Round-outcome bookkeeping needed to compute the loss bonus."""

    score: int = 0
    consecutive_losses: int = 0

    def record_win(self) -> None:
        self.score += 1
        self.consecutive_losses = 0

    def record_loss(self) -> None:
        self.consecutive_losses += 1


@dataclass
class MatchSimulator:
    """Generate a complete, deterministic CS2 match as telemetry events.

    Args:
        match_id: Identifier stamped on every emitted event.
        map_name: Map the match is played on.
        seed: Seeds the RNG; the same seed always yields the same match.
        tick_rate_hz: Snapshots emitted per simulated second.
        start_time: Timestamp of the first event.
    """

    match_id: str = "apex-001"
    map_name: MapName = MapName.MIRAGE
    seed: int = 42
    tick_rate_hz: float = 8.0
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))

    _rng: random.Random = field(init=False, repr=False)
    _players: list[_Player] = field(init=False, repr=False, default_factory=list)
    _ct: _TeamLedger = field(init=False, repr=False, default_factory=_TeamLedger)
    _t: _TeamLedger = field(init=False, repr=False, default_factory=_TeamLedger)
    _sequence: int = field(init=False, repr=False, default=0)
    _elapsed: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self) -> None:
        if self.tick_rate_hz <= 0:
            raise ValueError("tick_rate_hz must be positive")
        self._rng = random.Random(self.seed)
        self._players = self._create_roster()

    # -- Construction ---------------------------------------------------------

    def _create_roster(self) -> list[_Player]:
        roster: list[_Player] = []
        for index in range(constants.PLAYERS_PER_TEAM):
            roster.append(_Player(f"ct{index}", f"CT_Player_{index}", Team.CT))
        for index in range(constants.PLAYERS_PER_TEAM):
            roster.append(_Player(f"t{index}", f"T_Player_{index}", Team.T))
        return roster

    def _team(self, team: Team) -> list[_Player]:
        return [player for player in self._players if player.team is team]

    def _ledger(self, team: Team) -> _TeamLedger:
        return self._ct if team is Team.CT else self._t

    def _alive(self, team: Team) -> list[_Player]:
        return [player for player in self._team(team) if player.alive]

    # -- Timing ---------------------------------------------------------------

    @property
    def _tick_seconds(self) -> float:
        return 1.0 / self.tick_rate_hz

    def _now(self) -> datetime:
        return self.start_time + timedelta(seconds=self._elapsed)

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _random_position(self) -> Position:
        return Position(
            x=self._rng.uniform(-_MAP_EXTENT, _MAP_EXTENT),
            y=self._rng.uniform(-_MAP_EXTENT, _MAP_EXTENT),
            z=self._rng.uniform(-128.0, 128.0),
        )

    # -- Economy --------------------------------------------------------------

    def _buy_phase(self, team: Team) -> None:
        """Spend each player's money on the best loadout they can afford."""
        purchasable = sorted(
            (weapon for weapon in weapons_for(team) if WEAPON_COST[weapon] > 0),
            key=lambda weapon: WEAPON_COST[weapon],
            reverse=True,
        )
        # Every player spawns with their side's free pistol, so nobody ever enters
        # a round holding only a knife — a pistol round would otherwise leave the
        # weapon slot empty.
        sidearm = Weapon.USP if team is Team.CT else Weapon.GLOCK

        for player in self._team(team):
            budget = player.money
            player.weapon = sidearm

            for weapon in purchasable:
                cost = WEAPON_COST[weapon]
                # Keep enough back for armour; a rifle with no kevlar is a bad buy.
                if cost <= budget - ARMOUR_BUY_THRESHOLD:
                    player.weapon = weapon
                    budget -= cost
                    break

            if budget >= constants.KEVLAR_HELMET_COST:
                player.armour = constants.MAX_ARMOUR
                player.has_helmet = True
                budget -= constants.KEVLAR_HELMET_COST
            elif budget >= constants.KEVLAR_COST:
                player.armour = constants.MAX_ARMOUR
                budget -= constants.KEVLAR_COST

            if (
                team is Team.CT
                and budget >= constants.DEFUSE_KIT_COST
                and player.money >= KIT_BUY_THRESHOLD
            ):
                player.has_defuse_kit = True
                budget -= constants.DEFUSE_KIT_COST

            player.money = budget

    def _award(self, player: _Player, amount: int) -> None:
        player.money = min(player.money + amount, constants.MAX_MONEY)

    def _pay_round_end(self, winner: Team, reason: RoundEndReason) -> None:
        """Apply win rewards and loss bonuses once a round is decided."""
        loser = winner.opponent
        win_reward = (
            constants.ROUND_WIN_REWARD_BOMB
            if reason is RoundEndReason.BOMB_EXPLODED
            else constants.ROUND_WIN_REWARD
        )
        for player in self._team(winner):
            self._award(player, win_reward)

        bonus = constants.loss_bonus(self._ledger(loser).consecutive_losses)
        for player in self._team(loser):
            self._award(player, bonus)

    def _team_economy(self, team: Team) -> TeamEconomy:
        players = self._team(team)
        return TeamEconomy(
            team=team,
            money=sum(player.money for player in players),
            equipment_value=sum(player.equipment_value for player in players),
            consecutive_losses=self._ledger(team).consecutive_losses,
        )

    # -- Combat ---------------------------------------------------------------

    def _duel_win_probability(self) -> float:
        """CT probability of winning the next duel, adjusted for equipment.

        A side holding better gear wins more often, which is what makes the
        economy features predictive rather than decorative.
        """
        ct_value = sum(player.equipment_value for player in self._alive(Team.CT))
        t_value = sum(player.equipment_value for player in self._alive(Team.T))
        total = ct_value + t_value

        if total == 0:
            return CT_DUEL_BASE_WIN_RATE

        edge = (ct_value - t_value) / total
        return min(0.85, max(0.15, CT_DUEL_BASE_WIN_RATE + edge * EQUIPMENT_ADVANTAGE_WEIGHT))

    def _resolve_duel(self, round_number: int) -> KillEvent | None:
        """Kill one player, chosen by the equipment-weighted duel model."""
        ct_alive = self._alive(Team.CT)
        t_alive = self._alive(Team.T)
        if not ct_alive or not t_alive:
            return None

        ct_wins = self._rng.random() < self._duel_win_probability()
        killer = self._rng.choice(ct_alive if ct_wins else t_alive)
        victim = self._rng.choice(t_alive if ct_wins else ct_alive)

        victim.health = 0
        victim.deaths += 1
        # A dead player carries nothing; the schema enforces this too.
        victim.armour = 0
        victim.has_helmet = False
        victim.has_defuse_kit = False
        victim.weapon = None
        killer.kills += 1
        killer.damage_dealt += constants.MAX_HEALTH
        self._award(killer, constants.KILL_REWARD_DEFAULT)

        position = self._random_position()
        victim.position = position

        return KillEvent(
            match_id=self.match_id,
            timestamp=self._now(),
            sequence=self._next_sequence(),
            round_number=round_number,
            killer_id=killer.player_id,
            victim_id=victim.player_id,
            weapon=killer.weapon or Weapon.KNIFE,
            headshot=self._rng.random() < HEADSHOT_RATE,
            victim_team=victim.team,
            position=position,
        )

    # -- Snapshots ------------------------------------------------------------

    def _match_state(self, round_state: RoundState) -> MatchState:
        return MatchState(
            match_id=self.match_id,
            map_name=self.map_name,
            timestamp=self._now(),
            score_ct=self._ct.score,
            score_t=self._t.score,
            round_state=round_state,
            players=tuple(player.snapshot() for player in self._players),
            economy_ct=self._team_economy(Team.CT),
            economy_t=self._team_economy(Team.T),
        )

    def _tick(self, round_state: RoundState) -> TickEvent:
        return TickEvent(
            match_id=self.match_id,
            timestamp=self._now(),
            sequence=self._next_sequence(),
            state=self._match_state(round_state),
        )

    # -- Round ----------------------------------------------------------------

    def _simulate_round(self, round_number: int) -> Iterator[TelemetryEvent]:
        """Play one round to completion, yielding every event it produces."""
        for player in self._players:
            player.reset_for_round()
        self._buy_phase(Team.CT)
        self._buy_phase(Team.T)

        yield RoundStartEvent(
            match_id=self.match_id,
            timestamp=self._now(),
            sequence=self._next_sequence(),
            round_number=round_number,
            score_ct=self._ct.score,
            score_t=self._t.score,
        )

        # Freezetime: no action, but the dashboard still needs state.
        freeze_ticks = int(constants.FREEZETIME_SECONDS * self.tick_rate_hz)
        for _ in range(freeze_ticks):
            yield self._tick(
                RoundState(
                    round_number=round_number,
                    phase=RoundPhase.FREEZETIME,
                    seconds_remaining=constants.ROUND_SECONDS,
                )
            )
            self._elapsed += self._tick_seconds

        yield from self._simulate_live_round(round_number)

    def _simulate_live_round(self, round_number: int) -> Iterator[TelemetryEvent]:
        """Run the live phase until the round resolves."""
        remaining = constants.ROUND_SECONDS
        bomb_planted = False
        bomb_remaining: float | None = None
        bomb_position: Position | None = None
        next_duel_in = self._rng.expovariate(1.0 / DUEL_INTERVAL_SECONDS)

        # Decide the plant once per round rather than sampling every tick, then
        # pick when in the round it happens. Ts must still survive to execute it.
        will_plant = self._rng.random() < PLANT_ATTEMPT_RATE
        plant_at = constants.ROUND_SECONDS * (
            1.0 - self._rng.uniform(PLANT_WINDOW_START, PLANT_WINDOW_END)
        )

        winner: Team | None = None
        reason: RoundEndReason | None = None

        while winner is None:
            phase = RoundPhase.BOMB_PLANTED if bomb_planted else RoundPhase.LIVE
            yield self._tick(
                RoundState(
                    round_number=round_number,
                    phase=phase,
                    seconds_remaining=max(0.0, remaining),
                    bomb_planted=bomb_planted,
                    bomb_seconds_remaining=bomb_remaining,
                    bomb_position=bomb_position,
                )
            )

            step = self._tick_seconds
            self._elapsed += step
            remaining -= step
            next_duel_in -= step
            if bomb_remaining is not None:
                bomb_remaining = max(0.0, bomb_remaining - step)

            # 1. Engagements.
            if next_duel_in <= 0:
                kill = self._resolve_duel(round_number)
                if kill is not None:
                    yield kill
                next_duel_in = self._rng.expovariate(1.0 / DUEL_INTERVAL_SECONDS)

            # 2. Elimination ends the round immediately.
            if not self._alive(Team.T):
                winner, reason = Team.CT, RoundEndReason.T_ELIMINATED
                continue
            if not self._alive(Team.CT):
                # Ts still lose if the bomb is down and nobody can defuse it.
                if bomb_planted:
                    winner, reason = Team.T, RoundEndReason.BOMB_EXPLODED
                else:
                    winner, reason = Team.T, RoundEndReason.CT_ELIMINATED
                continue

            # 3. Plant, once the round clock reaches the chosen moment.
            if not bomb_planted and will_plant and remaining <= plant_at:
                planter = self._rng.choice(self._alive(Team.T))
                bomb_planted = True
                bomb_remaining = constants.BOMB_TIMER_SECONDS
                bomb_position = self._random_position()
                self._award(planter, constants.PLANT_REWARD)
                for t_player in self._team(Team.T):
                    self._award(t_player, constants.TEAM_PLANT_REWARD)

                yield BombPlantedEvent(
                    match_id=self.match_id,
                    timestamp=self._now(),
                    sequence=self._next_sequence(),
                    round_number=round_number,
                    planter_id=planter.player_id,
                    site="A" if self._rng.random() < 0.5 else "B",
                    position=bomb_position,
                )
                continue

            # 4. Post-plant resolution.
            if bomb_planted and bomb_remaining is not None and bomb_remaining <= 0.0:
                if self._rng.random() < POST_PLANT_DEFUSE_RATE:
                    defuser = self._rng.choice(self._alive(Team.CT))
                    self._award(defuser, constants.DEFUSE_REWARD)
                    yield BombDefusedEvent(
                        match_id=self.match_id,
                        timestamp=self._now(),
                        sequence=self._next_sequence(),
                        round_number=round_number,
                        defuser_id=defuser.player_id,
                        seconds_remaining=0.0,
                    )
                    winner, reason = Team.CT, RoundEndReason.BOMB_DEFUSED
                else:
                    winner, reason = Team.T, RoundEndReason.BOMB_EXPLODED
                continue

            # 5. Time expiry — a CT win, provided the bomb is not down.
            if remaining <= 0.0 and not bomb_planted:
                winner, reason = Team.CT, RoundEndReason.TIME_EXPIRED

        assert reason is not None
        self._ledger(winner).record_win()
        self._ledger(winner.opponent).record_loss()
        self._pay_round_end(winner, reason)

        yield RoundEndEvent(
            match_id=self.match_id,
            timestamp=self._now(),
            sequence=self._next_sequence(),
            round_number=round_number,
            winner=winner,
            reason=reason,
            score_ct=self._ct.score,
            score_t=self._t.score,
        )

    # -- Match ----------------------------------------------------------------

    @property
    def _is_decided(self) -> bool:
        return (
            self._ct.score >= constants.ROUNDS_TO_WIN
            or self._t.score >= constants.ROUNDS_TO_WIN
            or self._ct.score + self._t.score >= constants.MAX_REGULATION_ROUNDS
        )

    def run(self) -> Iterator[TelemetryEvent]:
        """Yield every event of a complete match, in order.

        The generator is lazy, so a caller can stop early — the API only needs a
        live window, not the whole match in memory.
        """
        yield MatchStartEvent(
            match_id=self.match_id,
            timestamp=self._now(),
            sequence=self._next_sequence(),
            map_name=self.map_name,
            team_ct_name="Team Vitality",
            team_t_name="Natus Vincere",
        )

        round_number = 1
        while not self._is_decided:
            yield from self._simulate_round(round_number)
            round_number += 1

        # A 12-12 regulation draw has no winner under this simplified ruleset;
        # award it to the side ahead, breaking an exact tie in CT's favour.
        winner = Team.CT if self._ct.score >= self._t.score else Team.T
        if self._ct.score == self._t.score:
            self._ct.score += 1

        yield MatchEndEvent(
            match_id=self.match_id,
            timestamp=self._now(),
            sequence=self._next_sequence(),
            winner=winner,
            score_ct=self._ct.score,
            score_t=self._t.score,
        )
