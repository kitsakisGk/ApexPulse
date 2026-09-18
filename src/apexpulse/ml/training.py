"""Offline training for the win-probability model.

Trains a gradient-boosted classifier that answers one question per tick: what is
the probability that CT wins this round?

Two choices drive everything else here:

* **Probabilities, not labels.** A dashboard showing "CT 73%" needs a calibrated
  probability, so the model is judged on log loss and Brier score rather than
  accuracy. A model that is 95% accurate but always says 0.99 is useless.
* **Split by match, never by row.** Consecutive ticks in a round are nearly
  identical. A random row split would put near-duplicates on both sides and
  report an accuracy that cannot survive contact with an unseen match.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from apexpulse.config import get_settings
from apexpulse.features import FEATURE_NAMES, build_training_set, split_by_match
from apexpulse.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from apexpulse.config import Settings

logger = get_logger(__name__)

MODEL_FILENAME = "win_probability.json"
"""Booster artifact. XGBoost's JSON format is version-portable; pickle is not."""

METADATA_FILENAME = "model_metadata.json"
"""Feature order, metrics, and provenance for the checkpoint beside it."""

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "auc"],
    "max_depth": 5,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "reg_lambda": 2.0,
    "tree_method": "hist",
}
"""Deliberately conservative.

Depth 5 with a high ``min_child_weight`` keeps the trees shallow: the dataset has
tens of thousands of rows but only a few hundred independent rounds, so a deeper
model memorises rounds rather than learning the game.
"""


@dataclass
class TrainingMetrics:
    """Evaluation of a trained model on held-out matches."""

    log_loss: float
    brier_score: float
    roc_auc: float
    accuracy: float
    base_rate: float
    baseline_log_loss: float

    @property
    def skill_score(self) -> float:
        """Improvement over always predicting the base rate.

        1.0 is perfect, 0.0 means the model adds nothing over guessing the
        dataset's average, and a negative value means it is actively harmful.
        """
        if self.baseline_log_loss <= 0:
            return 0.0
        return 1.0 - (self.log_loss / self.baseline_log_loss)

    def as_dict(self) -> dict[str, float]:
        """Return metrics as a plain mapping, including the derived skill score."""
        return {**asdict(self), "skill_score": self.skill_score}


@dataclass
class TrainingResult:
    """Everything a training run produced."""

    metrics: TrainingMetrics
    feature_importance: dict[str, float]
    train_rows: int
    test_rows: int
    train_matches: int
    test_matches: int
    best_iteration: int
    params: dict[str, Any] = field(default_factory=dict)
    trained_at: str = ""

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the run."""
        return {
            "trained_at": self.trained_at,
            "features": list(FEATURE_NAMES),
            "metrics": self.metrics.as_dict(),
            "feature_importance": self.feature_importance,
            "dataset": {
                "train_rows": self.train_rows,
                "test_rows": self.test_rows,
                "train_matches": self.train_matches,
                "test_matches": self.test_matches,
            },
            "best_iteration": self.best_iteration,
            "params": self.params,
        }


def evaluate(labels: pd.Series, probabilities: Any) -> TrainingMetrics:
    """Score predicted probabilities against the truth.

    Args:
        labels: Binary outcomes, 1 where CT won.
        probabilities: Predicted probability of a CT win, same length and order.
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

    truth = labels.to_numpy()
    predicted = np.asarray(probabilities, dtype=float)
    base_rate = float(truth.mean())

    # A model must beat the trivial strategy of always predicting the base rate.
    baseline = np.full_like(predicted, base_rate)

    return TrainingMetrics(
        log_loss=float(log_loss(truth, predicted, labels=[0, 1])),
        brier_score=float(brier_score_loss(truth, predicted)),
        roc_auc=float(roc_auc_score(truth, predicted)) if len(set(truth)) > 1 else 0.5,
        accuracy=float(accuracy_score(truth, predicted >= 0.5)),
        base_rate=base_rate,
        baseline_log_loss=float(log_loss(truth, baseline, labels=[0, 1])),
    )


def train_model(
    frame: pd.DataFrame,
    *,
    test_fraction: float = 0.2,
    num_rounds: int = 400,
    early_stopping_rounds: int = 30,
    params: dict[str, Any] | None = None,
    seed: int = 42,
) -> tuple[Any, TrainingResult]:
    """Train a win-probability model on ``frame``.

    Args:
        frame: Rows from the ``training_data`` view.
        test_fraction: Approximate share of *matches* held out for evaluation.
        num_rounds: Maximum boosting rounds.
        early_stopping_rounds: Stop once held-out log loss stops improving.
        params: Booster overrides merged over :data:`DEFAULT_PARAMS`.
        seed: Seeds the booster so a run is reproducible.

    Returns:
        The trained booster and a record of the run.
    """
    import xgboost as xgb

    if frame.empty:
        raise ValueError("cannot train on an empty dataset")

    train_frame, test_frame = split_by_match(frame, test_fraction=test_fraction)
    train_features, train_labels = build_training_set(train_frame)
    test_features, test_labels = build_training_set(test_frame)

    if train_labels.nunique() < 2:
        raise ValueError("the training split contains only one outcome")

    settings = {**DEFAULT_PARAMS, **(params or {}), "seed": seed}

    train_matrix = xgb.DMatrix(
        train_features, label=train_labels, feature_names=list(FEATURE_NAMES)
    )
    test_matrix = xgb.DMatrix(test_features, label=test_labels, feature_names=list(FEATURE_NAMES))

    logger.info(
        "training_started",
        train_rows=len(train_features),
        test_rows=len(test_features),
        train_matches=train_frame["match_id"].nunique(),
        test_matches=test_frame["match_id"].nunique(),
    )

    booster = xgb.train(
        settings,
        train_matrix,
        num_boost_round=num_rounds,
        evals=[(train_matrix, "train"), (test_matrix, "test")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )

    predictions = booster.predict(test_matrix, iteration_range=(0, booster.best_iteration + 1))
    metrics = evaluate(test_labels, predictions)

    # `gain` answers "how much did splitting on this feature improve the model",
    # which is the question a reader of the README actually has. `weight` would
    # just count splits and favour high-cardinality features.
    # get_score is typed as possibly returning per-output lists; this model is
    # single-output, so collapse to a scalar rather than assuming the shape.
    raw_importance = booster.get_score(importance_type="gain")
    gains = {
        name: float(value[0] if isinstance(value, list) else value)
        for name, value in raw_importance.items()
    }
    total_gain = sum(gains.values()) or 1.0
    importance = {name: round(gains.get(name, 0.0) / total_gain, 6) for name in FEATURE_NAMES}

    result = TrainingResult(
        metrics=metrics,
        feature_importance=dict(sorted(importance.items(), key=lambda item: item[1], reverse=True)),
        train_rows=len(train_features),
        test_rows=len(test_features),
        train_matches=int(train_frame["match_id"].nunique()),
        test_matches=int(test_frame["match_id"].nunique()),
        best_iteration=int(booster.best_iteration),
        params=settings,
        trained_at=datetime.now(UTC).isoformat(),
    )

    logger.info(
        "training_complete",
        log_loss=round(metrics.log_loss, 4),
        roc_auc=round(metrics.roc_auc, 4),
        accuracy=round(metrics.accuracy, 4),
        skill_score=round(metrics.skill_score, 4),
        best_iteration=result.best_iteration,
    )
    return booster, result


def save_model(
    booster: Any,
    result: TrainingResult,
    *,
    directory: Path | None = None,
    settings: Settings | None = None,
) -> tuple[Path, Path]:
    """Write the booster and its metadata; return both paths."""
    settings = settings or get_settings()
    target = directory or settings.model_dir
    target.mkdir(parents=True, exist_ok=True)

    model_path = target / MODEL_FILENAME
    metadata_path = target / METADATA_FILENAME

    booster.save_model(str(model_path))
    metadata_path.write_text(json.dumps(result.summary(), indent=2), encoding="utf-8")

    logger.info("model_saved", model=str(model_path), metadata=str(metadata_path))
    return model_path, metadata_path


def load_model(
    *,
    directory: Path | None = None,
    settings: Settings | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load a saved booster and its metadata.

    Raises:
        FileNotFoundError: If no checkpoint exists at the given location.
    """
    import xgboost as xgb

    settings = settings or get_settings()
    target = directory or settings.model_dir
    model_path = target / MODEL_FILENAME
    metadata_path = target / METADATA_FILENAME

    if not model_path.is_file():
        raise FileNotFoundError(f"no model at {model_path}; run 'apexpulse train' to create one")

    booster = xgb.Booster()
    booster.load_model(str(model_path))

    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return booster, metadata
