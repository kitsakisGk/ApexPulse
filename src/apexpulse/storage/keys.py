"""Key layout for the live state store.

Every key the pipeline writes is built here, so the namespace is visible in one
place and a rename cannot leave orphaned readers behind. Keys follow
``apexpulse:<entity>:<id>[:<facet>]``, matching the convention Redis tooling
expects for grouping.
"""

from __future__ import annotations

from typing import Final

NAMESPACE: Final = "apexpulse"

MATCH_PREFIX: Final = f"{NAMESPACE}:match:"
"""Prefix for live match snapshots."""

ROUNDS_SUFFIX: Final = ":rounds"
"""Facet holding a match's completed-round history."""

INDEX_KEY: Final = f"{NAMESPACE}:matches:live"
"""Key holding the set of match ids with live state."""


def match_key(match_id: str) -> str:
    """Return the key holding ``match_id``'s current snapshot."""
    return f"{MATCH_PREFIX}{match_id}"


def rounds_key(match_id: str) -> str:
    """Return the key holding ``match_id``'s completed-round history."""
    return f"{MATCH_PREFIX}{match_id}{ROUNDS_SUFFIX}"


def match_pattern() -> str:
    """Return a glob matching every live match snapshot.

    The trailing guard excludes facet keys such as ``…:rounds``, which share the
    match prefix but are not snapshots.
    """
    return f"{MATCH_PREFIX}*"


def match_id_from_key(key: str) -> str | None:
    """Extract the match id from a snapshot key, or ``None`` if it is a facet.

    Args:
        key: A key previously produced by :func:`match_key` or :func:`rounds_key`.
    """
    if not key.startswith(MATCH_PREFIX):
        return None
    remainder = key.removeprefix(MATCH_PREFIX)
    if ":" in remainder:
        return None
    return remainder
