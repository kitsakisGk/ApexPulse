"""Closed vocabularies for the CS2 telemetry domain.

Every enum is a string enum so events serialise to readable JSON and survive a
round trip through Kafka without a schema registry.
"""

from __future__ import annotations

from enum import StrEnum


class Team(StrEnum):
    """The two sides of a CS2 match."""

    CT = "CT"
    T = "T"

    @property
    def opponent(self) -> Team:
        """Return the opposing side."""
        return Team.T if self is Team.CT else Team.CT


class RoundPhase(StrEnum):
    """Lifecycle of a single round.

    Phases advance strictly forward; the bomb-planted phase replaces ``LIVE`` once
    the bomb is down, because the win condition changes at that moment.
    """

    FREEZETIME = "freezetime"
    LIVE = "live"
    BOMB_PLANTED = "bomb_planted"
    OVER = "over"


class RoundEndReason(StrEnum):
    """How a round was decided.

    The distinction matters for modelling: a time expiry is a CT win under very
    different circumstances than an elimination.
    """

    CT_ELIMINATED = "ct_eliminated"
    T_ELIMINATED = "t_eliminated"
    BOMB_DEFUSED = "bomb_defused"
    BOMB_EXPLODED = "bomb_exploded"
    TIME_EXPIRED = "time_expired"


class EventType(StrEnum):
    """Discriminator for the telemetry event union."""

    MATCH_START = "match_start"
    ROUND_START = "round_start"
    TICK = "tick"
    KILL = "kill"
    BOMB_PLANTED = "bomb_planted"
    BOMB_DEFUSED = "bomb_defused"
    ROUND_END = "round_end"
    MATCH_END = "match_end"


class Weapon(StrEnum):
    """Weapons tracked by the simulator.

    Restricted to the loadouts that actually move win probability; cosmetic
    variants are collapsed into their base weapon.
    """

    KNIFE = "knife"
    GLOCK = "glock"
    USP = "usp"
    P250 = "p250"
    DEAGLE = "deagle"
    MAC10 = "mac10"
    MP9 = "mp9"
    UMP45 = "ump45"
    GALIL = "galil"
    FAMAS = "famas"
    AK47 = "ak47"
    M4A4 = "m4a4"
    M4A1S = "m4a1s"
    AWP = "awp"
    SG553 = "sg553"
    AUG = "aug"


class MapName(StrEnum):
    """Active-duty map pool."""

    DUST2 = "de_dust2"
    MIRAGE = "de_mirage"
    INFERNO = "de_inferno"
    NUKE = "de_nuke"
    OVERPASS = "de_overpass"
    ANCIENT = "de_ancient"
    ANUBIS = "de_anubis"


# -- Economy ------------------------------------------------------------------
# Buy costs drive the economy features the model consumes, so they live beside the
# weapon vocabulary rather than in the simulator.

WEAPON_COST: dict[Weapon, int] = {
    Weapon.KNIFE: 0,
    Weapon.GLOCK: 0,
    Weapon.USP: 0,
    Weapon.P250: 300,
    Weapon.DEAGLE: 700,
    Weapon.MAC10: 1050,
    Weapon.MP9: 1250,
    Weapon.UMP45: 1200,
    Weapon.GALIL: 1800,
    Weapon.FAMAS: 2050,
    Weapon.AK47: 2700,
    Weapon.M4A4: 3100,
    Weapon.M4A1S: 2900,
    Weapon.AWP: 4750,
    Weapon.SG553: 3000,
    Weapon.AUG: 3300,
}

T_WEAPONS: frozenset[Weapon] = frozenset(
    {
        Weapon.KNIFE,
        Weapon.GLOCK,
        Weapon.P250,
        Weapon.DEAGLE,
        Weapon.MAC10,
        Weapon.GALIL,
        Weapon.AK47,
        Weapon.SG553,
        Weapon.AWP,
    }
)
"""Weapons a T-side player can buy."""

CT_WEAPONS: frozenset[Weapon] = frozenset(
    {
        Weapon.KNIFE,
        Weapon.USP,
        Weapon.P250,
        Weapon.DEAGLE,
        Weapon.MP9,
        Weapon.UMP45,
        Weapon.FAMAS,
        Weapon.M4A4,
        Weapon.M4A1S,
        Weapon.AUG,
        Weapon.AWP,
    }
)
"""Weapons a CT-side player can buy."""


def weapons_for(team: Team) -> frozenset[Weapon]:
    """Return the weapons purchasable by ``team``."""
    return T_WEAPONS if team is Team.T else CT_WEAPONS
