"""Real-time feature extraction.

Turns a match snapshot into the fixed-length numeric vector the win-probability
model scores. The same code path runs offline over the DuckDB training set and
online per tick, which is what keeps training and serving consistent — a feature
computed one way in training and another way in production is the most common
cause of a model that benchmarks well and fails live.

Design rules applied throughout:

* **Symmetric, not absolute.** Features describe the *difference* between sides
  (``alive_delta``, ``economy_ratio``) rather than raw per-side values, so the
  model learns the balance of the round rather than memorising team identities.
* **Bounded.** Ratios are normalised into ``[-1, 1]`` or ``[0, 1]`` so no feature
  dominates by scale and inference needs no separate normaliser at serving time.
* **Causal.** Nothing derived from the round's outcome may appear here, or the
  model would be trained on the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from apexpulse.schemas import constants
from apexpulse.schemas.enums import RoundPhase, Team

if TYPE_CHECKING:
    from apexpulse.schemas.models import MatchState
    from apexpulse.stream.window import WindowedMetrics

FEATURE_NAMES: Final[tuple[str, ...]] = (
    # -- Manpower ------------------------------------------------------------
    "alive_ct",
    "alive_t",
    "alive_delta",
    "alive_ratio",
    "health_ratio",
    # -- Round clock ---------------------------------------------------------
    "time_fraction",
    "bomb_planted",
    "bomb_time_fraction",
    # -- Economy -------------------------------------------------------------
    "money_ratio",
    "equipment_ratio",
    "loss_streak_delta",
    # -- Match context -------------------------------------------------------
    "score_delta",
    "round_fraction",
    # -- Momentum ------------------------------------------------------------
    "kill_delta_window",
    "engagement_pace",
)
"""Ordered feature names; the vector's index order is part of the model contract."""

FEATURE_COUNT: Final = len(FEATURE_NAMES)

_MAX_ALIVE: Final = float(constants.PLAYERS_PER_TEAM)
_MAX_TEAM_HEALTH: Final = float(constants.PLAYERS_PER_TEAM * constants.MAX_HEALTH)
_MAX_KILL_DELTA: Final = float(constants.PLAYERS_PER_TEAM)
_MAX_PACE: Final = 1.0
"""Kills per second treated as maximal pace; beyond this the feature saturates."""


def _ratio(ct_value: float, t_value: float) -> float:
    """Return a symmetric advantage in ``[-1, 1]``.

    Positive favours CT. Returns 0.0 when both sides are at zero, which is a tie
    rather than an undefined value.
    """
    total = ct_value + t_value
    if total <= 0:
        return 0.0
    return (ct_value - t_value) / total


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    """Constrain ``value`` to ``[low, high]``."""
    return max(low, min(high, value))


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """A named, ordered feature vector for one tick."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != FEATURE_COUNT:
            raise ValueError(f"expected {FEATURE_COUNT} features, got {len(self.values)}")

    def as_dict(self) -> dict[str, float]:
        """Return the vector keyed by feature name."""
        return dict(zip(FEATURE_NAMES, self.values, strict=True))

    def __getitem__(self, name: str) -> float:
        """Return one feature by name."""
        try:
            return self.values[FEATURE_NAMES.index(name)]
        except ValueError as exc:
            raise KeyError(name) from exc

    def __len__(self) -> int:
        return len(self.values)


@dataclass
class FeatureExtractor:
    """Compute model features from match state.

    Args:
        include_momentum: Whether windowed momentum features are populated. When
            a caller has no window — scoring a historical row, for example — the
            momentum features are emitted as zeros so the vector keeps its shape.
    """

    include_momentum: bool = True
    _feature_index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._feature_index = {name: index for index, name in enumerate(FEATURE_NAMES)}

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Ordered feature names produced by this extractor."""
        return FEATURE_NAMES

    def extract(
        self,
        state: MatchState,
        metrics: WindowedMetrics | None = None,
    ) -> FeatureVector:
        """Return the feature vector describing ``state``.

        Args:
            state: The snapshot to describe.
            metrics: Windowed momentum, or ``None`` to emit zeros for those
                features.
        """
        return FeatureVector(
            (
                *self._manpower(state),
                *self._clock(state),
                *self._economy(state),
                *self._match_context(state),
                *self._momentum(metrics),
            )
        )

    # -- Feature groups -------------------------------------------------------

    def _manpower(self, state: MatchState) -> tuple[float, ...]:
        """Who is left standing — the single strongest predictor in CS2."""
        alive_ct = float(state.alive_ct)
        alive_t = float(state.alive_t)

        health_ct = sum(player.health for player in state.players_on(Team.CT))
        health_t = sum(player.health for player in state.players_on(Team.T))

        return (
            alive_ct / _MAX_ALIVE,
            alive_t / _MAX_ALIVE,
            _clamp((alive_ct - alive_t) / _MAX_ALIVE),
            _ratio(alive_ct, alive_t),
            # Health captures damage a kill count misses: five players at 20 HP
            # is a far weaker position than five at full.
            _ratio(health_ct, health_t),
        )

    def _clock(self, state: MatchState) -> tuple[float, ...]:
        """Time pressure, and whose favour it runs in.

        Before the plant the clock favours CT — running it out is a CT win. After
        the plant it inverts, so the bomb timer is tracked separately rather than
        folded into one feature.
        """
        round_state = state.round_state
        time_fraction = round_state.seconds_remaining / constants.ROUND_SECONDS

        bomb_fraction = 0.0
        if round_state.bomb_planted and round_state.bomb_seconds_remaining is not None:
            bomb_fraction = round_state.bomb_seconds_remaining / constants.BOMB_TIMER_SECONDS

        return (
            _clamp(time_fraction, 0.0, 1.0),
            1.0 if round_state.bomb_planted else 0.0,
            _clamp(bomb_fraction, 0.0, 1.0),
        )

    def _economy(self, state: MatchState) -> tuple[float, ...]:
        """Buying power and equipment on the field."""
        economy_ct = state.economy_ct
        economy_t = state.economy_t

        return (
            _ratio(economy_ct.money, economy_t.money),
            _ratio(economy_ct.equipment_value, economy_t.equipment_value),
            # A longer loss streak means a larger bonus next round, which shapes
            # how willing a side is to spend now.
            _clamp(
                (economy_ct.consecutive_losses - economy_t.consecutive_losses)
                / len(constants.LOSS_BONUS_LADDER)
            ),
        )

    def _match_context(self, state: MatchState) -> tuple[float, ...]:
        """Where the round sits in the match."""
        rounds_played = state.score_ct + state.score_t

        return (
            _clamp((state.score_ct - state.score_t) / constants.ROUNDS_TO_WIN),
            _clamp(rounds_played / constants.MAX_REGULATION_ROUNDS, 0.0, 1.0),
        )

    def _momentum(self, metrics: WindowedMetrics | None) -> tuple[float, ...]:
        """Recent form, which instantaneous state cannot express."""
        if metrics is None or not self.include_momentum:
            return (0.0, 0.0)

        return (
            _clamp(metrics.kill_delta / _MAX_KILL_DELTA),
            _clamp(metrics.kills_per_second / _MAX_PACE, 0.0, 1.0),
        )

    # -- Bulk extraction ------------------------------------------------------

    def extract_batch(
        self,
        states: list[MatchState],
        metrics: list[WindowedMetrics | None] | None = None,
    ) -> list[FeatureVector]:
        """Extract features for many states, preserving order."""
        if metrics is None:
            return [self.extract(state) for state in states]
        if len(metrics) != len(states):
            raise ValueError("states and metrics must be the same length")
        return [self.extract(state, metric) for state, metric in zip(states, metrics, strict=True)]


def is_scoreable(state: MatchState) -> bool:
    """Whether ``state`` is worth scoring.

    Freezetime and finished rounds carry no live signal: nothing has happened yet,
    or everything already has. Scoring them wastes inference and pollutes the
    dashboard with a probability that cannot move.
    """
    return state.round_state.phase in {RoundPhase.LIVE, RoundPhase.BOMB_PLANTED}
