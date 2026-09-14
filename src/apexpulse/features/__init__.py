"""Real-time feature extraction from raw match state."""

from apexpulse.features.dataset import (
    LABEL_COLUMN,
    build_feature_frame,
    build_training_set,
    describe_features,
    split_by_match,
)
from apexpulse.features.extractor import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    FeatureExtractor,
    FeatureVector,
    is_scoreable,
)

__all__ = [
    "FEATURE_COUNT",
    "FEATURE_NAMES",
    "LABEL_COLUMN",
    "FeatureExtractor",
    "FeatureVector",
    "build_feature_frame",
    "build_training_set",
    "describe_features",
    "is_scoreable",
    "split_by_match",
]
