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
    "ProbabilityCalibrator",
    "TrainingMetrics",
    "TrainingResult",
    "assess_calibration",
    "evaluate",
    "fit_calibrator",
    "format_reliability_table",
    "load_model",
    "save_model",
    "train_model",
]
