"""Probability calibration for the win-probability model.

A boosted tree optimises log loss, which makes it a good ranker but not
necessarily an honest one. The first trained checkpoint ranked well (ROC AUC
0.698) while overstating its confidence: predictions in the 60-70% band won only
54% of the time. A viewer reading "64%" off a gauge deserves better than that.

An isotonic regression fitted on held-out predictions corrects the mapping while
preserving the ranking, so AUC is untouched and only the numbers move. The fit is
stored as plain breakpoints rather than a pickled estimator, so the artifact
survives a scikit-learn upgrade.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from apexpulse.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

logger = get_logger(__name__)

CALIBRATOR_FILENAME = "calibrator.json"
"""Breakpoints of the fitted isotonic mapping, beside the booster."""

MIN_SAMPLES = 100
"""Below this, a fit would model noise; the identity mapping is returned instead."""


@dataclass(frozen=True, slots=True)
class ProbabilityCalibrator:
    """A monotonic map from raw model output to a calibrated probability.

    Stored as matched ``(x, y)`` breakpoints and applied by linear interpolation.
    Because the map is monotonic, it cannot reorder predictions: ranking metrics
    such as ROC AUC are identical before and after.
    """

    breakpoints_x: tuple[float, ...]
    breakpoints_y: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.breakpoints_x) != len(self.breakpoints_y):
            raise ValueError("breakpoint arrays must be the same length")

    @property
    def is_identity(self) -> bool:
        """Whether this calibrator leaves probabilities unchanged."""
        return len(self.breakpoints_x) < 2

    def apply(self, probabilities: Any) -> Any:
        """Return calibrated probabilities for ``probabilities``."""
        import numpy as np

        values = np.asarray(probabilities, dtype=float)
        if self.is_identity:
            return values

        # `interp` clamps to the end points, so predictions outside the fitted
        # range map to the nearest calibrated value rather than extrapolating.
        return np.interp(values, self.breakpoints_x, self.breakpoints_y)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "kind": "isotonic",
            "breakpoints_x": list(self.breakpoints_x),
            "breakpoints_y": list(self.breakpoints_y),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProbabilityCalibrator:
        """Rebuild a calibrator from :meth:`to_dict` output."""
        return cls(
            breakpoints_x=tuple(payload.get("breakpoints_x", ())),
            breakpoints_y=tuple(payload.get("breakpoints_y", ())),
        )

    @classmethod
    def identity(cls) -> ProbabilityCalibrator:
        """Return a calibrator that changes nothing."""
        return cls(breakpoints_x=(), breakpoints_y=())

    def save(self, path: Path) -> Path:
        """Write the calibrator to ``path``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> ProbabilityCalibrator:
        """Load a calibrator, falling back to the identity when absent."""
        if not path.is_file():
            return cls.identity()
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def fit_calibrator(
    labels: pd.Series,
    probabilities: Any,
) -> ProbabilityCalibrator:
    """Fit an isotonic mapping from raw predictions to observed outcome rates.

    Args:
        labels: Binary outcomes, 1 where CT won.
        probabilities: Raw model output for the same rows.

    Returns:
        A fitted calibrator, or the identity when there is too little data or
        only one outcome present to learn from.
    """
    import numpy as np
    from sklearn.isotonic import IsotonicRegression

    truth = np.asarray(labels, dtype=float)
    raw = np.asarray(probabilities, dtype=float)

    if len(raw) < MIN_SAMPLES or len(np.unique(truth)) < 2:
        logger.warning("calibrator_skipped", samples=len(raw))
        return ProbabilityCalibrator.identity()

    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(raw, truth)

    # Sample the fitted step function on a fixed grid. Storing the grid rather
    # than the estimator keeps the artifact independent of scikit-learn's
    # internal representation.
    grid = np.linspace(raw.min(), raw.max(), 256)
    mapped = model.predict(grid)

    logger.info(
        "calibrator_fitted",
        samples=len(raw),
        raw_range=[round(float(raw.min()), 4), round(float(raw.max()), 4)],
    )
    return ProbabilityCalibrator(
        breakpoints_x=tuple(float(value) for value in grid),
        breakpoints_y=tuple(float(value) for value in mapped),
    )
