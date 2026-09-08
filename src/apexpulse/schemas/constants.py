"""CS2 competitive ruleset constants.

Both the replay simulator and the feature extractor derive behaviour from these
values, so they are defined once here rather than duplicated at each call site.
Values follow the standard MR12 competitive configuration.
"""

from __future__ import annotations

from typing import Final

# -- Match structure ----------------------------------------------------------

ROUNDS_PER_HALF: Final = 12
"""Rounds each side plays before teams swap."""

ROUNDS_TO_WIN: Final = 13
"""Rounds needed to win a match outright under MR12."""

MAX_REGULATION_ROUNDS: Final = 24
"""Rounds played before a 12-12 scoreline goes to overtime."""

PLAYERS_PER_TEAM: Final = 5

# -- Round timing (seconds) ---------------------------------------------------

FREEZETIME_SECONDS: Final = 20.0
ROUND_SECONDS: Final = 115.0
"""Round clock, 1:55 under the competitive ruleset."""

BOMB_TIMER_SECONDS: Final = 40.0
"""Countdown from a successful plant to detonation."""

DEFUSE_SECONDS_WITH_KIT: Final = 5.0
DEFUSE_SECONDS_WITHOUT_KIT: Final = 10.0

# -- Economy ------------------------------------------------------------------

STARTING_MONEY: Final = 800
MAX_MONEY: Final = 16_000

KILL_REWARD_DEFAULT: Final = 300
"""Standard reward; SMGs and the AWP deviate but are not modelled per-weapon."""

PLANT_REWARD: Final = 300
"""Paid to the planter."""

TEAM_PLANT_REWARD: Final = 800
"""Paid to every T on a plant, even in a lost round."""

DEFUSE_REWARD: Final = 300

ROUND_WIN_REWARD: Final = 3_250
ROUND_WIN_REWARD_BOMB: Final = 3_500
"""Awarded to Ts when the bomb detonates."""

LOSS_BONUS_LADDER: Final[tuple[int, ...]] = (1_400, 1_900, 2_400, 2_900, 3_400)
"""Loss bonus by consecutive-loss count, capped at the final entry."""

FULL_BUY_THRESHOLD: Final = 4_000
"""Per-player money at which a team is considered fully equipped."""

# -- Player state -------------------------------------------------------------

MAX_HEALTH: Final = 100
MAX_ARMOUR: Final = 100
KEVLAR_COST: Final = 650
KEVLAR_HELMET_COST: Final = 1_000
DEFUSE_KIT_COST: Final = 400


def loss_bonus(consecutive_losses: int) -> int:
    """Return the loss bonus for a team on ``consecutive_losses`` straight defeats.

    Args:
        consecutive_losses: Rounds lost in a row; 1 after the first loss.

    Returns:
        The per-player payout, held at the ladder's final value once it maxes out.
    """
    if consecutive_losses <= 0:
        return 0
    index = min(consecutive_losses, len(LOSS_BONUS_LADDER)) - 1
    return LOSS_BONUS_LADDER[index]
