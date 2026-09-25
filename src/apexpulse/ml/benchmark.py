"""Accuracy benchmarking by match situation.

A single held-out log loss says the model works on average. It does not say
whether it works in the situations that matter. A win-probability gauge that is
excellent in 5v5 openings and useless in post-plant clutches is worse than its
headline number suggests, because clutches are exactly when people are watching.

This module slices held-out predictions by situation and reports accuracy within
each, so a weakness is visible rather than averaged away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from apexpulse.ml.training import evaluate

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pandas as pd

MIN_SLICE_SIZE = 50
"""Rows required before a slice's metrics are reported rather than noise."""


@dataclass(frozen=True, slots=True)
class SliceResult:
    """Accuracy within one match situation."""

    name: str
    count: int
    log_loss: float
    roc_auc: float
    accuracy: float
    base_rate: float
    skill_score: float

    @property
    def beats_baseline(self) -> bool:
        """Whether the model adds information over guessing this slice's base rate."""
        return self.skill_score > 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        return {
            "name": self.name,
            "count": self.count,
            "log_loss": round(self.log_loss, 5),
            "roc_auc": round(self.roc_auc, 5),
            "accuracy": round(self.accuracy, 5),
            "base_rate": round(self.base_rate, 5),
            "skill_score": round(self.skill_score, 5),
        }


@dataclass
class BenchmarkReport:
    """Accuracy across every situation, plus the overall figure."""

    overall: SliceResult
    slices: tuple[SliceResult, ...]

    @property
    def weakest(self) -> SliceResult | None:
        """The slice with the lowest skill score, which is where to look first."""
        return min(self.slices, key=lambda item: item.skill_score, default=None)

    @property
    def failing_slices(self) -> tuple[SliceResult, ...]:
        """Situations where the model does not beat that slice's base rate."""
        return tuple(item for item in self.slices if not item.beats_baseline)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        return {
            "overall": self.overall.as_dict(),
            "slices": [item.as_dict() for item in self.slices],
            "failing": [item.name for item in self.failing_slices],
        }


# Situations worth measuring separately. Each takes the held-out frame and
# returns a boolean mask. They deliberately overlap: a post-plant clutch belongs
# in both `post-plant` and `clutch`, and both readings are interesting.
SITUATIONS: dict[str, Callable[[pd.DataFrame], Any]] = {
    "early round": lambda frame: frame["seconds_remaining"] > 80,
    "mid round": lambda frame: (
        (frame["seconds_remaining"] <= 80) & (frame["seconds_remaining"] > 30)
    ),
    "late round": lambda frame: (frame["seconds_remaining"] <= 30) & (~frame["bomb_planted"]),
    "post-plant": lambda frame: frame["bomb_planted"],
    "even manpower": lambda frame: frame["alive_ct"] == frame["alive_t"],
    "ct ahead": lambda frame: frame["alive_ct"] > frame["alive_t"],
    "t ahead": lambda frame: frame["alive_ct"] < frame["alive_t"],
    "lopsided": lambda frame: (frame["alive_ct"] - frame["alive_t"]).abs() >= 3,
    "clutch": lambda frame: (frame["alive_ct"] <= 1) | (frame["alive_t"] <= 1),
    "full buy": lambda frame: (frame["equipment_ct"] > 20_000) & (frame["equipment_t"] > 20_000),
    "eco round": lambda frame: (frame["equipment_ct"] < 10_000) | (frame["equipment_t"] < 10_000),
}


def benchmark_by_situation(
    frame: pd.DataFrame,
    labels: pd.Series,
    probabilities: Sequence[float],
    *,
    min_slice_size: int = MIN_SLICE_SIZE,
) -> BenchmarkReport:
    """Measure accuracy within each match situation.

    Args:
        frame: Held-out rows from the ``training_data`` view, carrying the raw
            columns the situation filters read.
        labels: Binary outcomes aligned to ``frame``.
        probabilities: Predicted CT win probability aligned to ``frame``.
        min_slice_size: Rows required before a slice is reported.

    Returns:
        A report whose slices cover only situations with enough data to judge.
    """
    import numpy as np

    predicted = np.asarray(probabilities, dtype=float)
    overall_metrics = evaluate(labels, predicted)
    overall = SliceResult(
        name="overall",
        count=len(predicted),
        log_loss=overall_metrics.log_loss,
        roc_auc=overall_metrics.roc_auc,
        accuracy=overall_metrics.accuracy,
        base_rate=overall_metrics.base_rate,
        skill_score=overall_metrics.skill_score,
    )

    results: list[SliceResult] = []
    for name, predicate in SITUATIONS.items():
        mask = predicate(frame).to_numpy()
        count = int(mask.sum())
        if count < min_slice_size:
            continue

        slice_labels = labels[mask]
        if slice_labels.nunique() < 2:
            # Every round in this slice went the same way, so log loss against a
            # perfect base rate is degenerate and AUC is undefined.
            continue

        metrics = evaluate(slice_labels, predicted[mask])
        results.append(
            SliceResult(
                name=name,
                count=count,
                log_loss=metrics.log_loss,
                roc_auc=metrics.roc_auc,
                accuracy=metrics.accuracy,
                base_rate=metrics.base_rate,
                skill_score=metrics.skill_score,
            )
        )

    return BenchmarkReport(overall=overall, slices=tuple(results))


def format_benchmark_table(report: BenchmarkReport) -> str:
    """Render a benchmark report as a fixed-width table."""
    lines = [
        "  situation            n        base    acc     AUC     skill",
        "  " + "-" * 58,
    ]

    def row(result: SliceResult, marker: str = "") -> str:
        return (
            f"  {result.name:<18} {result.count:>7,}  {result.base_rate:>5.1%}  "
            f"{result.accuracy:>5.1%}  {result.roc_auc:>6.4f}  {result.skill_score:>+6.1%}{marker}"
        )

    lines.append(row(report.overall))
    lines.append("  " + "-" * 58)
    for result in sorted(report.slices, key=lambda item: item.skill_score, reverse=True):
        lines.append(row(result, "" if result.beats_baseline else "  <- no skill"))

    lines.append("  " + "-" * 58)
    weakest = report.weakest
    if weakest is not None:
        lines.append(f"  weakest situation: {weakest.name} at {weakest.skill_score:+.1%} skill")
    return "\n".join(lines)


__all__ = [
    "MIN_SLICE_SIZE",
    "SITUATIONS",
    "BenchmarkReport",
    "SliceResult",
    "benchmark_by_situation",
    "format_benchmark_table",
]
