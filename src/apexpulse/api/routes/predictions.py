"""Prediction and model-introspection routes.

The dashboard needs the live number, but a portfolio API should also answer
"why". These routes expose the feature vector behind a prediction and the
model's own metrics, so the gauge can be interrogated rather than merely trusted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import Field

from apexpulse.api.dependencies import get_state
from apexpulse.api.schemas import ApiModel, ErrorResponse

if TYPE_CHECKING:
    from apexpulse.api.dependencies import AppState

router = APIRouter(tags=["predictions"])

UNAVAILABLE: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
    status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
}


class FeatureContribution(ApiModel):
    """One feature's live value and its weight in the trained model."""

    name: str
    value: float = Field(description="Value for this tick, bounded to [-1, 1].")
    importance: float = Field(
        ge=0.0, le=1.0, description="Share of total gain this feature carried in training."
    )


class ExplainedPrediction(ApiModel):
    """A prediction with the features that produced it."""

    match_id: str
    round_number: int
    ct_win_probability: float = Field(ge=0.0, le=1.0)
    favoured_side: str
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    features: tuple[FeatureContribution, ...] = Field(
        description="Ordered by training importance, most influential first."
    )


class ModelInfo(ApiModel):
    """Metadata about the checkpoint currently serving."""

    loaded: bool
    trained_at: str | None = None
    features: tuple[str, ...] = ()
    trees: int | None = Field(default=None, description="Trees used per prediction.")
    roc_auc: float | None = None
    log_loss: float | None = None
    skill_score: float | None = Field(
        default=None, description="Improvement over always guessing the base rate."
    )
    accuracy: float | None = None
    train_rows: int | None = None
    test_matches: int | None = None
    top_features: tuple[FeatureContribution, ...] = ()


def _require_engine(state: AppState) -> Any:
    """Return the loaded engine, or fail with a 503.

    Raises:
        HTTPException: 503 when no checkpoint is loaded, which is a service
            condition rather than a bad request.
    """
    if state.engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="no model is loaded; run 'apexpulse train' to create a checkpoint",
        )
    return state.engine


@router.get(
    "/matches/{match_id}/prediction",
    response_model=ExplainedPrediction,
    responses=UNAVAILABLE,
    summary="Win probability with the features behind it",
)
async def explain_prediction(match_id: str, request: Request) -> ExplainedPrediction:
    """Score ``match_id``'s current state and return the features that drove it.

    Raises:
        HTTPException: 503 when no model is loaded, 404 when the match has no
            live state or its current tick cannot be scored.
    """
    from apexpulse.stream.window import WindowedMetrics

    state = get_state(request)
    engine = _require_engine(state)

    snapshot = await state.require_manager().get_snapshot(match_id)
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no live state for match {match_id!r}",
        )

    metrics = WindowedMetrics(
        kills_ct=snapshot.momentum.kills_ct,
        kills_t=snapshot.momentum.kills_t,
        span_seconds=snapshot.momentum.window_seconds,
    )
    prediction = engine.predict(snapshot.state, metrics, include_features=True)

    if not prediction.scored:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"match {match_id!r} is in "
                f"{snapshot.state.round_state.phase.value}, which carries no live signal"
            ),
        )

    state.predictions_served += 1
    importance: dict[str, float] = engine.metadata.get("feature_importance", {})

    contributions = sorted(
        (
            FeatureContribution(
                name=name,
                value=value,
                importance=float(importance.get(name, 0.0)),
            )
            for name, value in prediction.features.items()
        ),
        key=lambda item: item.importance,
        reverse=True,
    )

    return ExplainedPrediction(
        match_id=prediction.match_id,
        round_number=prediction.round_number,
        ct_win_probability=prediction.ct_win_probability,
        favoured_side=prediction.favoured_side,
        confidence=prediction.confidence,
        latency_ms=prediction.latency_ms,
        features=tuple(contributions),
    )


@router.get("/model", response_model=ModelInfo, summary="Checkpoint metadata and metrics")
async def model_info(request: Request) -> ModelInfo:
    """Describe the checkpoint currently serving.

    Returns ``loaded: false`` rather than an error when no model is present, so
    a dashboard can render the absence instead of handling an exception.
    """
    state = get_state(request)

    if state.engine is None:
        return ModelInfo(loaded=False)

    metadata = state.engine.metadata
    metrics = metadata.get("metrics", {})
    importance: dict[str, float] = metadata.get("feature_importance", {})

    top = tuple(
        FeatureContribution(name=name, value=0.0, importance=float(gain))
        for name, gain in sorted(importance.items(), key=lambda item: item[1], reverse=True)[:5]
    )

    dataset = metadata.get("dataset", {})
    return ModelInfo(
        loaded=True,
        trained_at=metadata.get("trained_at"),
        features=tuple(metadata.get("features", ())),
        trees=state.engine.tree_count,
        roc_auc=metrics.get("roc_auc"),
        log_loss=metrics.get("log_loss"),
        skill_score=metrics.get("skill_score"),
        accuracy=metrics.get("accuracy"),
        train_rows=dataset.get("train_rows"),
        test_matches=dataset.get("test_matches"),
        top_features=top,
    )
