"""Model drift detection.

A model is trained once and served indefinitely. The game does not hold still:
a patch changes economy values, a meta shift alters how rounds are played, or an
upstream bug quietly corrupts a feature. The model keeps answering confidently
while its answers stop meaning anything.

Drift is detected by comparing the live feature distribution against the one
recorded at training time. Two complementary measures are used:

* **Population Stability Index** — the standard measure in production ML. It
  compares binned distributions and has well-established thresholds, so a number
  can be acted on rather than merely observed.
* **Mean shift in standard deviations** — cheap, and catches a feature that has
  moved wholesale without changing shape, which PSI on coarse bins can miss.

Detection is deliberately separate from the inference path: drift is a property
of a distribution, not of one prediction, so it is computed over a batch rather
than per tick.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from apexpulse.features import FEATURE_NAMES
from apexpulse.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)

PSI_BINS = 10
"""Quantile bins used when comparing distributions."""

PSI_STABLE = 0.10
"""Below this, the distribution is considered unchanged. Industry convention."""

PSI_SHIFTED = 0.25
"""Above this, the distribution has moved enough to warrant retraining."""

MIN_SAMPLES = 200
"""Live samples required before a verdict is meaningful."""

_EPSILON = 1e-6
"""Floor applied to bin proportions so an empty bin cannot produce infinity."""


class DriftSeverity(StrEnum):
    """How far a distribution has moved."""

    STABLE = "stable"
    MODERATE = "moderate"
    SEVERE = "severe"
    UNKNOWN = "unknown"
    """Too few samples to judge; not the same as stable."""

    @classmethod
    def from_psi(cls, psi: float) -> DriftSeverity:
        """Bucket a PSI value using the conventional thresholds."""
        if psi < PSI_STABLE:
            return cls.STABLE
        if psi < PSI_SHIFTED:
            return cls.MODERATE
        return cls.SEVERE


@dataclass(frozen=True, slots=True)
class FeatureDrift:
    """Drift measured for a single feature."""

    name: str
    psi: float
    mean_shift_sigma: float
    reference_mean: float
    live_mean: float
    severity: DriftSeverity

    @property
    def has_drifted(self) -> bool:
        """Whether this feature has moved beyond the stable threshold."""
        return self.severity in {DriftSeverity.MODERATE, DriftSeverity.SEVERE}

    def describe(self) -> str:
        """Return a one-line summary for an operator."""
        direction = "up" if self.live_mean > self.reference_mean else "down"
        return (
            f"{self.name}: PSI {self.psi:.3f} ({self.severity.value}), "
            f"mean {direction} {abs(self.mean_shift_sigma):.2f}σ "  # noqa: RUF001
            f"({self.reference_mean:+.3f} -> {self.live_mean:+.3f})"
        )


@dataclass
class DriftReport:
    """Drift across every feature, with an overall verdict."""

    features: tuple[FeatureDrift, ...]
    sample_count: int
    severity: DriftSeverity

    @property
    def drifted_features(self) -> tuple[FeatureDrift, ...]:
        """Features that moved, worst first."""
        return tuple(
            sorted(
                (feature for feature in self.features if feature.has_drifted),
                key=lambda feature: feature.psi,
                reverse=True,
            )
        )

    @property
    def max_psi(self) -> float:
        """Largest PSI observed across features."""
        return max((feature.psi for feature in self.features), default=0.0)

    @property
    def should_retrain(self) -> bool:
        """Whether the drift is severe enough to warrant a new checkpoint."""
        return self.severity is DriftSeverity.SEVERE

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        return {
            "severity": self.severity.value,
            "sample_count": self.sample_count,
            "max_psi": round(self.max_psi, 5),
            "should_retrain": self.should_retrain,
            "drifted": [
                {
                    "name": feature.name,
                    "psi": round(feature.psi, 5),
                    "mean_shift_sigma": round(feature.mean_shift_sigma, 4),
                    "severity": feature.severity.value,
                }
                for feature in self.drifted_features
            ],
        }

    def summary(self) -> str:
        """Return a human-readable verdict."""
        if self.severity is DriftSeverity.UNKNOWN:
            return f"insufficient data: {self.sample_count} samples, need {MIN_SAMPLES}"
        drifted = self.drifted_features
        if not drifted:
            return f"stable across {len(self.features)} features (max PSI {self.max_psi:.3f})"
        worst = drifted[0]
        return (
            f"{self.severity.value}: {len(drifted)} of {len(self.features)} features moved, "
            f"worst is {worst.name} at PSI {worst.psi:.3f}"
        )


def population_stability_index(
    reference: Sequence[float],
    live: Sequence[float],
    *,
    bins: int = PSI_BINS,
) -> float:
    """Return the PSI between a reference and a live sample.

    Bin edges come from the reference quantiles, so each reference bin holds a
    comparable share and the comparison is not dominated by a feature's scale.

    Args:
        reference: Values observed during training.
        live: Values observed in production.
        bins: Number of quantile bins.

    Returns:
        0.0 for identical distributions, rising without bound as they diverge.
    """
    import numpy as np

    reference_array = np.asarray(reference, dtype=float)
    live_array = np.asarray(live, dtype=float)

    if reference_array.size == 0 or live_array.size == 0:
        return 0.0

    edges = np.unique(np.quantile(reference_array, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        # A feature that was constant in training cannot be binned; report drift
        # only if it has started varying.
        return 0.0 if float(live_array.std()) < _EPSILON else float(PSI_SHIFTED)

    # Open the outer edges so live values beyond the training range still land.
    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts, _ = np.histogram(reference_array, bins=edges)
    live_counts, _ = np.histogram(live_array, bins=edges)

    reference_share = np.maximum(reference_counts / reference_array.size, _EPSILON)
    live_share = np.maximum(live_counts / live_array.size, _EPSILON)

    return float(np.sum((live_share - reference_share) * np.log(live_share / reference_share)))


@dataclass
class DriftDetector:
    """Compares live feature distributions against the training reference.

    Args:
        reference: Per-feature training samples, keyed by feature name.
        min_samples: Live samples required before a verdict is issued.
    """

    reference: dict[str, tuple[float, ...]] = field(default_factory=dict)
    min_samples: int = MIN_SAMPLES

    @property
    def is_fitted(self) -> bool:
        """Whether a reference distribution has been recorded."""
        return bool(self.reference)

    @classmethod
    def from_frame(cls, features: Any, *, sample_limit: int = 20_000) -> DriftDetector:
        """Build a detector from a training feature frame.

        Args:
            features: Frame whose columns are the model's features.
            sample_limit: Reference rows retained per feature. The full training
                set would make the artifact large for no gain: PSI over 20,000
                samples is already stable to three decimals.
        """
        sampled = (
            features
            if len(features) <= sample_limit
            else features.sample(n=sample_limit, random_state=0)
        )
        return cls(
            reference={
                name: tuple(float(value) for value in sampled[name])
                for name in features.columns
                if name in set(FEATURE_NAMES)
            }
        )

    def detect(self, live: dict[str, Sequence[float]]) -> DriftReport:
        """Compare ``live`` feature samples against the reference.

        Args:
            live: Per-feature samples observed in production.
        """
        import numpy as np

        sample_count = min((len(values) for values in live.values()), default=0)

        if not self.is_fitted or sample_count < self.min_samples:
            return DriftReport(
                features=(),
                sample_count=sample_count,
                severity=DriftSeverity.UNKNOWN,
            )

        drifts: list[FeatureDrift] = []
        for name, reference_values in self.reference.items():
            live_values = live.get(name)
            if not live_values:
                continue

            reference_array = np.asarray(reference_values, dtype=float)
            live_array = np.asarray(live_values, dtype=float)

            psi = population_stability_index(reference_values, live_values)
            reference_mean = float(reference_array.mean())
            live_mean = float(live_array.mean())
            reference_std = float(reference_array.std())
            shift = (
                0.0 if reference_std < _EPSILON else (live_mean - reference_mean) / reference_std
            )

            drifts.append(
                FeatureDrift(
                    name=name,
                    psi=psi,
                    mean_shift_sigma=shift,
                    reference_mean=reference_mean,
                    live_mean=live_mean,
                    severity=DriftSeverity.from_psi(psi),
                )
            )

        overall = _worst_severity(drifts)
        report = DriftReport(features=tuple(drifts), sample_count=sample_count, severity=overall)

        if report.should_retrain:
            logger.warning(
                "model_drift_detected",
                severity=overall.value,
                max_psi=round(report.max_psi, 4),
                drifted=[feature.name for feature in report.drifted_features],
            )
        return report

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        return {
            "min_samples": self.min_samples,
            "reference": {name: list(values) for name, values in self.reference.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DriftDetector:
        """Rebuild a detector from :meth:`to_dict` output."""
        return cls(
            reference={
                name: tuple(values) for name, values in payload.get("reference", {}).items()
            },
            min_samples=int(payload.get("min_samples", MIN_SAMPLES)),
        )


def _worst_severity(drifts: Sequence[FeatureDrift]) -> DriftSeverity:
    """Return the most severe verdict across features.

    One badly drifted feature is enough to invalidate a prediction, so the
    overall verdict takes the worst rather than an average.
    """
    if not drifts:
        return DriftSeverity.STABLE
    if any(feature.severity is DriftSeverity.SEVERE for feature in drifts):
        return DriftSeverity.SEVERE
    if any(feature.severity is DriftSeverity.MODERATE for feature in drifts):
        return DriftSeverity.MODERATE
    return DriftSeverity.STABLE


def format_drift_table(report: DriftReport) -> str:
    """Render a drift report as a fixed-width table."""
    if report.severity is DriftSeverity.UNKNOWN:
        return f"  {report.summary()}"

    lines = [
        "  feature                   PSI     shift    severity",
        "  " + "-" * 52,
    ]
    ordered = sorted(report.features, key=lambda feature: feature.psi, reverse=True)
    for feature in ordered:
        marker = " " if not feature.has_drifted else "*"
        lines.append(
            f"  {feature.name:<22} {feature.psi:>7.4f}  {feature.mean_shift_sigma:>+6.2f}σ  "  # noqa: RUF001
            f"{feature.severity.value:<9}{marker}"
        )
    lines.append("  " + "-" * 52)
    lines.append(f"  {report.summary()}")
    return "\n".join(lines)


__all__ = [
    "MIN_SAMPLES",
    "PSI_BINS",
    "PSI_SHIFTED",
    "PSI_STABLE",
    "DriftDetector",
    "DriftReport",
    "DriftSeverity",
    "FeatureDrift",
    "format_drift_table",
    "population_stability_index",
]
