"""Prediction confidence scoring.

A probability alone does not say whether it should be trusted. Two states can
both read 65% while one sits in territory the model saw thousands of times and
the other is an extrapolation from a handful of rounds. A broadcast gauge that
treats those identically will eventually be confidently wrong on air.

Confidence here combines three independent signals:

* **Decisiveness** — distance from an even call. A 95% read carries more
  information than a 51% one.
* **Support** — whether the feature vector resembles states the model was
  trained on. An unfamiliar position is an extrapolation.
* **Stability** — whether the recent prediction sequence is steady or thrashing.
  A number bouncing 40/70/45 across three ticks is not settled.

The three are reported separately as well as combined, so a low score can be
explained rather than merely displayed.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from apexpulse.features.extractor import FeatureVector

STABILITY_WINDOW = 8
"""Recent predictions retained per match when judging stability."""

VOLATILITY_SCALE = 0.15
"""Standard deviation at which stability is considered fully degraded.

A well-behaved win probability drifts; it does not jump. Swings of 15 points
across a few ticks mean the model is reacting to noise.
"""

MIN_STABILITY_SAMPLES = 3
"""Predictions needed before stability is meaningful; below this it is unknown."""


class ConfidenceBand(StrEnum):
    """Coarse confidence label for display."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @classmethod
    def from_score(cls, score: float) -> ConfidenceBand:
        """Bucket a confidence score into a band."""
        if score >= 0.66:
            return cls.HIGH
        if score >= 0.33:
            return cls.MEDIUM
        return cls.LOW


@dataclass(frozen=True, slots=True)
class ConfidenceScore:
    """A prediction's confidence, and the components that produced it."""

    score: float
    decisiveness: float
    support: float
    stability: float
    band: ConfidenceBand

    @property
    def is_trustworthy(self) -> bool:
        """Whether the prediction is solid enough to drive a headline number."""
        return self.band is not ConfidenceBand.LOW

    def explain(self) -> str:
        """Return the weakest component, which is what a reader wants to know."""
        components = {
            "decisiveness": self.decisiveness,
            "training support": self.support,
            "stability": self.stability,
        }
        name, value = min(components.items(), key=lambda item: item[1])
        return f"{name} is the limiting factor at {value:.0%}"

    def as_dict(self) -> dict[str, float | str]:
        """Return a JSON-serialisable record."""
        return {
            "score": round(self.score, 4),
            "decisiveness": round(self.decisiveness, 4),
            "support": round(self.support, 4),
            "stability": round(self.stability, 4),
            "band": self.band.value,
        }


def decisiveness(probability: float) -> float:
    """Return how far ``probability`` sits from an even call, in ``[0, 1]``."""
    return min(1.0, abs(probability - 0.5) * 2.0)


def stability(recent: Sequence[float]) -> float:
    """Return how settled a recent prediction sequence is, in ``[0, 1]``.

    Args:
        recent: Predictions in arrival order, oldest first.

    Returns:
        1.0 for a perfectly steady sequence, falling towards 0.0 as it thrashes.
        Returns 1.0 when there are too few samples to judge: absence of evidence
        should not read as evidence of instability.
    """
    if len(recent) < MIN_STABILITY_SAMPLES:
        return 1.0

    mean = sum(recent) / len(recent)
    variance = sum((value - mean) ** 2 for value in recent) / len(recent)
    deviation = math.sqrt(variance)

    return max(0.0, 1.0 - deviation / VOLATILITY_SCALE)


@dataclass
class FeatureSupport:
    """Judges whether a feature vector resembles the training distribution.

    Holds the per-feature mean and standard deviation observed during training.
    A live vector is scored by how many standard deviations it sits from that
    centre, averaged across features — a cheap proxy for density that costs one
    pass over fifteen floats rather than a nearest-neighbour search.
    """

    means: tuple[float, ...] = ()
    deviations: tuple[float, ...] = ()
    threshold_sigma: float = 3.0
    """Distance at which a feature is considered fully unfamiliar."""

    def __post_init__(self) -> None:
        if len(self.means) != len(self.deviations):
            raise ValueError("means and deviations must be the same length")
        if self.threshold_sigma <= 0:
            raise ValueError("threshold_sigma must be positive")

    @property
    def is_fitted(self) -> bool:
        """Whether a training distribution has been recorded."""
        return bool(self.means)

    @classmethod
    def from_frame(cls, features: object, *, threshold_sigma: float = 3.0) -> FeatureSupport:
        """Build support statistics from a training feature frame."""
        means = tuple(float(value) for value in features.mean())  # type: ignore[attr-defined]
        deviations = tuple(float(value) for value in features.std())  # type: ignore[attr-defined]
        return cls(means=means, deviations=deviations, threshold_sigma=threshold_sigma)

    def score(self, vector: FeatureVector) -> float:
        """Return how well ``vector`` is supported by training data, in ``[0, 1]``.

        Returns 1.0 when no distribution has been recorded: an unfitted support
        model should not penalise every prediction.
        """
        if not self.is_fitted:
            return 1.0

        total = 0.0
        counted = 0
        for index, value in enumerate(vector.values):
            if index >= len(self.means):
                break
            deviation = self.deviations[index]
            if deviation <= 1e-9:
                # A feature that never varied in training tells us nothing about
                # familiarity, so it neither helps nor hurts.
                continue
            sigmas = abs(value - self.means[index]) / deviation
            total += min(1.0, sigmas / self.threshold_sigma)
            counted += 1

        if counted == 0:
            return 1.0
        return max(0.0, 1.0 - total / counted)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable record."""
        return {
            "means": list(self.means),
            "deviations": list(self.deviations),
            "threshold_sigma": self.threshold_sigma,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> FeatureSupport:
        """Rebuild support statistics from :meth:`to_dict` output."""
        return cls(
            means=tuple(payload.get("means", ())),  # type: ignore[arg-type]
            deviations=tuple(payload.get("deviations", ())),  # type: ignore[arg-type]
            threshold_sigma=float(payload.get("threshold_sigma", 3.0)),  # type: ignore[arg-type]
        )


@dataclass
class ConfidenceScorer:
    """Scores prediction confidence, tracking stability per match.

    Args:
        support: Training-distribution statistics; unfitted by default.
        window: Recent predictions retained per match.
    """

    support: FeatureSupport = field(default_factory=FeatureSupport)
    window: int = STABILITY_WINDOW

    _history: dict[str, deque[float]] = field(default_factory=dict, init=False, repr=False)

    def observe(self, match_id: str, probability: float) -> None:
        """Record a prediction so later scores can judge stability."""
        if match_id not in self._history:
            self._history[match_id] = deque(maxlen=self.window)
        self._history[match_id].append(probability)

    def reset(self, match_id: str) -> None:
        """Drop a match's history, at a round boundary or match end."""
        self._history.pop(match_id, None)

    def recent(self, match_id: str) -> tuple[float, ...]:
        """Return the retained predictions for ``match_id``, oldest first."""
        return tuple(self._history.get(match_id, ()))

    def score(
        self,
        match_id: str,
        probability: float,
        vector: FeatureVector | None = None,
    ) -> ConfidenceScore:
        """Score the confidence of one prediction.

        The three components are combined by their geometric mean rather than an
        average: a prediction that is decisive but unsupported should not be
        rescued by its other two components. A weak link drags the whole score.
        """
        decisive = decisiveness(probability)
        supported = self.support.score(vector) if vector is not None else 1.0
        steady = stability(self.recent(match_id))

        product = max(decisive, 1e-6) * max(supported, 1e-6) * max(steady, 1e-6)
        combined = product ** (1 / 3)

        return ConfidenceScore(
            score=combined,
            decisiveness=decisive,
            support=supported,
            stability=steady,
            band=ConfidenceBand.from_score(combined),
        )
