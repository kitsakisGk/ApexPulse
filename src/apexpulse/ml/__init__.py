"""Offline training, calibration, and model artifact management."""

from apexpulse.ml.calibration import (
    CalibrationBin,
    CalibrationReport,
    assess_calibration,
    format_reliability_table,
)
from apexpulse.ml.calibrator import (
    CALIBRATOR_FILENAME,
    ProbabilityCalibrator,
    fit_calibrator,
)
from apexpulse.ml.confidence import (
    ConfidenceBand,
    ConfidenceScore,
    ConfidenceScorer,
    FeatureSupport,
    decisiveness,
    stability,
)
from apexpulse.ml.drift import (
    DriftDetector,
    DriftReport,
    DriftSeverity,
    FeatureDrift,
    format_drift_table,
    population_stability_index,
)
from apexpulse.ml.training import (
    DEFAULT_PARAMS,
    METADATA_FILENAME,
    MODEL_FILENAME,
    TrainingMetrics,
    TrainingResult,
    evaluate,
    load_model,
    save_model,
    train_model,
)

__all__ = [
    "CALIBRATOR_FILENAME",
    "DEFAULT_PARAMS",
    "METADATA_FILENAME",
    "MODEL_FILENAME",
    "CalibrationBin",
    "CalibrationReport",
    "ConfidenceBand",
    "ConfidenceScore",
    "ConfidenceScorer",
    "DriftDetector",
    "DriftReport",
    "DriftSeverity",
    "FeatureDrift",
    "FeatureSupport",
    "ProbabilityCalibrator",
    "TrainingMetrics",
    "TrainingResult",
    "assess_calibration",
    "decisiveness",
    "evaluate",
    "fit_calibrator",
    "format_drift_table",
    "format_reliability_table",
    "load_model",
    "population_stability_index",
    "save_model",
    "stability",
    "train_model",
]
