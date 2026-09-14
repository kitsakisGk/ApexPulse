"""Offline feature extraction over the historical training set.

The online extractor works from a :class:`MatchState`; the DuckDB training view
returns flat rows. This module applies the *same* arithmetic to those rows, so a
feature means the same thing whether it was computed during training or during a
live tick.

The formulas are deliberately duplicated in vectorised form rather than looping
row-wise through :class:`FeatureExtractor`: 130k rows per training run makes the
per-row model construction cost dominate. The parity test in the suite asserts
both paths agree, which is what makes the duplication safe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from apexpulse.features.extractor import FEATURE_NAMES
from apexpulse.schemas import constants

if TYPE_CHECKING:
    import pandas as pd

LABEL_COLUMN = "ct_won"
"""Supervised target: 1 when CT won the round this tick belonged to."""


def _ratio(ct: pd.Series, t: pd.Series) -> pd.Series:
    """Vectorised symmetric advantage in ``[-1, 1]``; 0.0 where both are zero."""
    total = ct + t
    return ((ct - t) / total.where(total > 0)).fillna(0.0)


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a frame of model features derived from ``frame``.

    Args:
        frame: Rows from the ``training_data`` view.

    Returns:
        A frame whose columns are exactly :data:`FEATURE_NAMES`, in order.
    """
    import numpy as np
    import pandas as pd

    max_alive = float(constants.PLAYERS_PER_TEAM)
    alive_ct = frame["alive_ct"].astype(float)
    alive_t = frame["alive_t"].astype(float)

    bomb_planted = frame["bomb_planted"].astype(float)
    bomb_seconds = frame["bomb_seconds"].fillna(0.0).astype(float)

    features = pd.DataFrame(index=frame.index)

    features["alive_ct"] = alive_ct / max_alive
    features["alive_t"] = alive_t / max_alive
    features["alive_delta"] = ((alive_ct - alive_t) / max_alive).clip(-1.0, 1.0)
    features["alive_ratio"] = _ratio(alive_ct, alive_t)
    features["health_ratio"] = _ratio(
        frame["health_ct"].astype(float), frame["health_t"].astype(float)
    )

    features["time_fraction"] = (
        frame["seconds_remaining"].astype(float) / constants.ROUND_SECONDS
    ).clip(0.0, 1.0)
    features["bomb_planted"] = bomb_planted
    features["bomb_time_fraction"] = (bomb_seconds / constants.BOMB_TIMER_SECONDS).clip(0.0, 1.0)

    features["money_ratio"] = _ratio(
        frame["money_ct"].astype(float), frame["money_t"].astype(float)
    )
    features["equipment_ratio"] = _ratio(
        frame["equipment_ct"].astype(float), frame["equipment_t"].astype(float)
    )
    features["loss_streak_delta"] = (
        (frame["losses_ct"].astype(float) - frame["losses_t"].astype(float))
        / len(constants.LOSS_BONUS_LADDER)
    ).clip(-1.0, 1.0)

    features["score_delta"] = (
        (frame["score_ct"].astype(float) - frame["score_t"].astype(float)) / constants.ROUNDS_TO_WIN
    ).clip(-1.0, 1.0)
    features["round_fraction"] = (
        (frame["score_ct"].astype(float) + frame["score_t"].astype(float))
        / constants.MAX_REGULATION_ROUNDS
    ).clip(0.0, 1.0)

    # Momentum is not persisted per tick, so offline rows carry zeros. The online
    # extractor fills these from the sliding window.
    features["kill_delta_window"] = np.zeros(len(frame), dtype=float)
    features["engagement_pace"] = np.zeros(len(frame), dtype=float)

    return features[list(FEATURE_NAMES)]


def build_training_set(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split ``frame`` into a feature matrix and its label column.

    Raises:
        KeyError: If the label column is absent, which means the caller passed
            raw ticks rather than the labelled training view.
    """
    if LABEL_COLUMN not in frame.columns:
        raise KeyError(f"{LABEL_COLUMN!r} is missing; pass rows from the 'training_data' view")
    return build_feature_frame(frame), frame[LABEL_COLUMN].astype(int)


def split_by_match(
    frame: pd.DataFrame,
    *,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``frame`` into train and test sets along match boundaries.

    Splitting by row would leak: consecutive ticks from one round are nearly
    identical, so the same round would appear on both sides of the split and the
    reported accuracy would be meaningless. Whole matches go to one side or the
    other.

    Args:
        frame: Rows from the ``training_data`` view.
        test_fraction: Approximate share of matches held out.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")

    match_ids = sorted(frame["match_id"].unique())
    if len(match_ids) < 2:
        raise ValueError("splitting by match requires at least two matches")

    holdout_size = max(1, round(len(match_ids) * test_fraction))
    holdout = set(match_ids[-holdout_size:])

    is_test = frame["match_id"].isin(holdout)
    return frame[~is_test], frame[is_test]


def describe_features(features: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Summarise each feature's range, for sanity-checking a training set."""
    return {
        name: {
            "min": float(features[name].min()),
            "max": float(features[name].max()),
            "mean": float(features[name].mean()),
            "std": float(features[name].std()),
        }
        for name in features.columns
    }
