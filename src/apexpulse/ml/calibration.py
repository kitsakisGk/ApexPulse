"""Calibration analysis for the win-probability model.

Accuracy says whether the model picks the right side. Calibration says whether
its *numbers* are honest: of every tick where it said 70%, CT should have won
about 70% of them. A broadcast gauge is worthless without that, so this is the
measure that matters most for the product.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """One band of predicted probability and what actually happened in it."""

    lower: float
    upper: float
    count: int
    mean_predicted: float
    observed_rate: float

    @property
    def error(self) -> float:
        """Signed gap between prediction and reality; positive means overconfident."""
        return self.mean_predicted - self.observed_rate

    @property
    def label(self) -> str:
        """Human-readable band, e.g. ``60-70%``."""
        return f"{self.lower:.0%}-{self.upper:.0%}"


@dataclass
class CalibrationReport:
    """Reliability of a model's probabilities across the full range."""

    bins: tuple[CalibrationBin, ...]
    expected_calibration_error: float
    max_calibration_error: float

    @property
    def is_well_calibrated(self) -> bool:
        """Whether the average error is small enough to display as a percentage.

        5% is the threshold at which a viewer would not notice the gap between
        the gauge and reality.
        """
        return self.expected_calibration_error < 0.05

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        return {
            "expected_calibration_error": round(self.expected_calibration_error, 5),
            "max_calibration_error": round(self.max_calibration_error, 5),
            "is_well_calibrated": self.is_well_calibrated,
            "bins": [
                {
                    "range": bin_.label,
                    "count": bin_.count,
                    "predicted": round(bin_.mean_predicted, 4),
                    "observed": round(bin_.observed_rate, 4),
                    "error": round(bin_.error, 4),
                }
                for bin_ in self.bins
            ],
        }


def assess_calibration(
    labels: pd.Series,
    probabilities: Any,
    *,
    bin_count: int = 10,
) -> CalibrationReport:
    """Bin predictions and compare each band against its observed outcome rate.

    Args:
        labels: Binary outcomes, 1 where CT won.
        probabilities: Predicted probability of a CT win.
        bin_count: Number of equal-width bands across ``[0, 1]``.

    Returns:
        A report whose expected error is weighted by bin population, so a badly
        calibrated band holding three samples cannot dominate the headline number.
    """
    import numpy as np

    truth = labels.to_numpy(dtype=float)
    predicted = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bin_count + 1)

    bins: list[CalibrationBin] = []
    weighted_error = 0.0
    max_error = 0.0
    total = len(predicted)

    for index in range(bin_count):
        lower, upper = edges[index], edges[index + 1]
        # The final bin is closed on the right so a prediction of exactly 1.0 lands.
        in_bin = (
            (predicted >= lower) & (predicted <= upper)
            if index == bin_count - 1
            else (predicted >= lower) & (predicted < upper)
        )
        count = int(in_bin.sum())
        if count == 0:
            continue

        mean_predicted = float(predicted[in_bin].mean())
        observed = float(truth[in_bin].mean())
        error = abs(mean_predicted - observed)

        weighted_error += (count / total) * error
        max_error = max(max_error, error)
        bins.append(
            CalibrationBin(
                lower=float(lower),
                upper=float(upper),
                count=count,
                mean_predicted=mean_predicted,
                observed_rate=observed,
            )
        )

    return CalibrationReport(
        bins=tuple(bins),
        expected_calibration_error=weighted_error,
        max_calibration_error=max_error,
    )


def format_reliability_table(report: CalibrationReport) -> str:
    """Render the report as a fixed-width table for the terminal."""
    lines = [
        "  band      n        predicted   observed   error",
        "  " + "-" * 50,
    ]
    for bin_ in report.bins:
        marker = " " if abs(bin_.error) < 0.05 else "*"
        lines.append(
            f"  {bin_.label:<9} {bin_.count:>7,}  "
            f"{bin_.mean_predicted:>9.1%}  {bin_.observed_rate:>9.1%}  "
            f"{bin_.error:>+7.1%} {marker}"
        )
    lines.append("  " + "-" * 50)
    lines.append(
        f"  expected calibration error  {report.expected_calibration_error:.2%}"
        f"   (max {report.max_calibration_error:.2%})"
    )
    return "\n".join(lines)
