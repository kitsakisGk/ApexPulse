"""Low-latency win-probability scoring.

Turns a live match snapshot into a probability the dashboard can render. The
whole design target is per-tick latency: at 8 Hz across several concurrent
matches, anything that allocates or re-validates per call shows up immediately in
the p99.

Three decisions follow from that:

* **Predict on a raw numpy array**, not a pandas DataFrame. Building a one-row
  frame per tick costs more than the tree traversal it feeds.
* **Reuse one buffer.** The feature vector is written into a preallocated array
  rather than allocated per call.
* **Skip what cannot move.** Freezetime and finished rounds carry no live signal,
  so they short-circuit before touching the model at all.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

from apexpulse.features import FEATURE_COUNT, FEATURE_NAMES, FeatureExtractor, is_scoreable
from apexpulse.logging import get_logger
from apexpulse.ml.calibrator import ProbabilityCalibrator

if TYPE_CHECKING:
    from pathlib import Path

    from apexpulse.config import Settings
    from apexpulse.schemas.models import MatchState
    from apexpulse.stream.window import WindowedMetrics

logger = get_logger(__name__)

NEUTRAL_PROBABILITY = 0.5
"""Returned when a state cannot be scored; an even round is the honest default."""

LATENCY_WINDOW = 1_000
"""Recent samples retained for percentile reporting."""

LATENCY_BUDGET_MS = 20.0
"""Per-tick budget. Exceeding it means the dashboard falls behind the match."""

# Measured on the 57-tree checkpoint, 2,000 live snapshots, single-tick path:
#
#     stage                       cost      share
#     booster prediction        1.220 ms    72.7%
#     feature extraction        0.058 ms     3.5%
#     scoreability check        0.001 ms     0.1%
#
# The model dominates, so that is where the optimisation went: `inplace_predict`
# takes the numpy row directly and avoids building a DMatrix per call, which
# alone costs 0.875 ms — 52% of the current total.
#
# Trimming trees is the remaining lever, and `max_trees` exposes it:
#
#     trees   log loss   ROC AUC   latency
#        30     0.6218    0.7025   1.56 ms
#        58     0.6165    0.7087   2.83 ms
#
# It is not applied by default: p99 already sits at 2.96 ms against a 20 ms
# budget, so spending accuracy on latency nobody needs would be the wrong trade.


@dataclass(frozen=True, slots=True)
class Prediction:
    """A scored snapshot."""

    match_id: str
    round_number: int
    ct_win_probability: float
    latency_ms: float
    scored: bool
    features: dict[str, float] = field(default_factory=dict)

    @property
    def t_win_probability(self) -> float:
        """Complement of the CT probability."""
        return 1.0 - self.ct_win_probability

    @property
    def favoured_side(self) -> str:
        """The side the model currently favours, or ``even`` within a point."""
        if abs(self.ct_win_probability - 0.5) < 0.01:
            return "even"
        return "CT" if self.ct_win_probability > 0.5 else "T"

    @property
    def confidence(self) -> float:
        """Distance from an even call, scaled to ``[0, 1]``."""
        return abs(self.ct_win_probability - 0.5) * 2.0


@dataclass
class LatencyStats:
    """Rolling latency percentiles over recent predictions."""

    samples: deque[float] = field(default_factory=lambda: deque(maxlen=LATENCY_WINDOW))
    total_predictions: int = 0
    skipped: int = 0

    def record(self, latency_ms: float) -> None:
        """Record one scored prediction."""
        self.samples.append(latency_ms)
        self.total_predictions += 1

    @property
    def count(self) -> int:
        """Samples currently retained."""
        return len(self.samples)

    def percentile(self, fraction: float) -> float:
        """Return the latency at ``fraction`` of the retained distribution."""
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        index = min(len(ordered) - 1, int(fraction * len(ordered)))
        return ordered[index]

    @property
    def mean_ms(self) -> float:
        """Mean latency over retained samples."""
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def p50_ms(self) -> float:
        """Median latency."""
        return self.percentile(0.50)

    @property
    def p95_ms(self) -> float:
        """95th percentile latency."""
        return self.percentile(0.95)

    @property
    def p99_ms(self) -> float:
        """99th percentile latency — the number a live service is judged on."""
        return self.percentile(0.99)

    def as_dict(self) -> dict[str, float]:
        """Return the percentiles as a plain mapping."""
        return {
            "count": float(self.count),
            "total_predictions": float(self.total_predictions),
            "skipped": float(self.skipped),
            "mean_ms": round(self.mean_ms, 4),
            "p50_ms": round(self.p50_ms, 4),
            "p95_ms": round(self.p95_ms, 4),
            "p99_ms": round(self.p99_ms, 4),
        }


class InferenceEngine:
    """Score match snapshots against a trained checkpoint.

    Args:
        booster: A loaded XGBoost booster.
        calibrator: Probability calibrator; the identity when none was fitted.
        metadata: Checkpoint metadata, used to verify the feature contract.
        best_iteration: Trees to use when predicting; ``None`` reads the value
            recorded with the checkpoint.
        max_trees: Cap the trees used, trading accuracy for latency. Left unset
            by default; see the measurements above before changing it.

    Raises:
        ValueError: If the checkpoint was trained on a different feature set.
    """

    def __init__(
        self,
        booster: Any,
        calibrator: ProbabilityCalibrator | None = None,
        metadata: dict[str, Any] | None = None,
        *,
        best_iteration: int | None = None,
        max_trees: int | None = None,
    ) -> None:
        import numpy as np

        self._booster = booster
        self._calibrator = calibrator or ProbabilityCalibrator.identity()
        self._metadata = metadata or {}
        self._extractor = FeatureExtractor()
        self._stats = LatencyStats()

        trained_features = self._metadata.get("features")
        if trained_features and list(trained_features) != list(FEATURE_NAMES):
            raise ValueError(
                "checkpoint feature order does not match this build; retrain the model"
            )

        resolved = best_iteration
        if resolved is None:
            resolved = self._metadata.get("best_iteration")
        if resolved is None:
            resolved = getattr(booster, "best_iteration", None)

        trees = int(resolved) + 1 if resolved is not None else None
        if max_trees is not None:
            if max_trees < 1:
                raise ValueError("max_trees must be at least 1")
            trees = max_trees if trees is None else min(trees, max_trees)
        self._iteration_range = (0, trees) if trees is not None else None

        # One reusable row. Allocating a (1, 15) array per tick is measurable at
        # 8 Hz across concurrent matches.
        self._buffer = np.zeros((1, FEATURE_COUNT), dtype=np.float32)

    @property
    def tree_count(self) -> int | None:
        """Trees used per prediction, or ``None`` when the whole model is used."""
        return self._iteration_range[1] if self._iteration_range else None

    @classmethod
    def from_checkpoint(
        cls,
        *,
        directory: Path | None = None,
        settings: Settings | None = None,
        max_trees: int | None = None,
    ) -> Self:
        """Load an engine from a saved checkpoint.

        Raises:
            FileNotFoundError: If no checkpoint exists at the given location.
        """
        from apexpulse.ml.training import load_model

        booster, calibrator, metadata = load_model(directory=directory, settings=settings)
        engine = cls(booster, calibrator, metadata, max_trees=max_trees)
        logger.info(
            "inference_engine_loaded",
            trained_at=metadata.get("trained_at"),
            best_iteration=metadata.get("best_iteration"),
            trees_used=engine.tree_count,
            calibrated=not engine._calibrator.is_identity,
        )
        return engine

    @property
    def stats(self) -> LatencyStats:
        """Rolling latency statistics."""
        return self._stats

    @property
    def metadata(self) -> dict[str, Any]:
        """Metadata recorded alongside the checkpoint."""
        return dict(self._metadata)

    def predict(
        self,
        state: MatchState,
        metrics: WindowedMetrics | None = None,
        *,
        include_features: bool = False,
    ) -> Prediction:
        """Score ``state`` and return the CT win probability.

        Args:
            state: The snapshot to score.
            metrics: Windowed momentum for this tick, when available.
            include_features: Attach the feature vector to the result. Off by
                default because building the dict costs more than the prediction.
        """
        started = time.perf_counter()

        if not is_scoreable(state):
            # Freezetime and finished rounds cannot move: skip the model entirely.
            self._stats.skipped += 1
            return Prediction(
                match_id=state.match_id,
                round_number=state.round_state.round_number,
                ct_win_probability=NEUTRAL_PROBABILITY,
                latency_ms=(time.perf_counter() - started) * 1_000,
                scored=False,
            )

        vector = self._extractor.extract(state, metrics)
        self._buffer[0, :] = vector.values

        raw = float(self._predict_raw(self._buffer)[0])
        probability = float(self._calibrator.apply([raw])[0])

        latency_ms = (time.perf_counter() - started) * 1_000
        self._stats.record(latency_ms)

        return Prediction(
            match_id=state.match_id,
            round_number=state.round_state.round_number,
            ct_win_probability=probability,
            latency_ms=latency_ms,
            scored=True,
            features=vector.as_dict() if include_features else {},
        )

    def predict_batch(self, states: list[MatchState]) -> list[Prediction]:
        """Score many states in one pass.

        Batching amortises the per-call overhead, which matters when replaying a
        match or backfilling predictions over history.
        """
        import numpy as np

        if not states:
            return []

        scoreable = [(index, state) for index, state in enumerate(states) if is_scoreable(state)]
        results: list[Prediction | None] = [None] * len(states)

        for index, state in enumerate(states):
            if is_scoreable(state):
                continue
            self._stats.skipped += 1
            results[index] = Prediction(
                match_id=state.match_id,
                round_number=state.round_state.round_number,
                ct_win_probability=NEUTRAL_PROBABILITY,
                latency_ms=0.0,
                scored=False,
            )

        if scoreable:
            started = time.perf_counter()
            matrix = np.empty((len(scoreable), FEATURE_COUNT), dtype=np.float32)
            for row, (_, state) in enumerate(scoreable):
                matrix[row, :] = self._extractor.extract(state).values

            raw = self._predict_raw(matrix)
            calibrated = self._calibrator.apply(raw)
            per_state_ms = ((time.perf_counter() - started) * 1_000) / len(scoreable)

            for row, (index, state) in enumerate(scoreable):
                self._stats.record(per_state_ms)
                results[index] = Prediction(
                    match_id=state.match_id,
                    round_number=state.round_state.round_number,
                    ct_win_probability=float(calibrated[row]),
                    latency_ms=per_state_ms,
                    scored=True,
                )

        return [result for result in results if result is not None]

    def _predict_raw(self, matrix: Any) -> Any:
        """Run the booster over a feature matrix.

        Uses ``inplace_predict`` where available: it takes the numpy array
        directly, skipping the DMatrix construction that dominates single-row
        latency.
        """
        inplace = getattr(self._booster, "inplace_predict", None)
        if inplace is not None:
            if self._iteration_range is not None:
                return inplace(matrix, iteration_range=self._iteration_range)
            return inplace(matrix)

        import xgboost as xgb  # pragma: no cover - fallback for older builds

        dmatrix = xgb.DMatrix(matrix, feature_names=list(FEATURE_NAMES))
        if self._iteration_range is not None:
            return self._booster.predict(dmatrix, iteration_range=self._iteration_range)
        return self._booster.predict(dmatrix)
