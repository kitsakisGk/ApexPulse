"""Tests for confidence scoring, drift detection, and situation benchmarking.

These three answer the questions a single accuracy number cannot: should this
prediction be trusted, has the world moved away from the training data, and
where specifically does the model fail?
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from apexpulse.config import Settings
from apexpulse.features import FEATURE_NAMES, FeatureExtractor, build_training_set, split_by_match
from apexpulse.ml import (
    ConfidenceBand,
    ConfidenceScorer,
    DriftDetector,
    DriftSeverity,
    FeatureSupport,
    benchmark_by_situation,
    decisiveness,
    format_benchmark_table,
    format_drift_table,
    population_stability_index,
    stability,
    train_model,
)
from apexpulse.ml.drift import MIN_SAMPLES, PSI_SHIFTED, PSI_STABLE
from apexpulse.producer import MatchSimulator
from apexpulse.schemas.enums import MapName, RoundPhase, Team
from apexpulse.schemas.models import MatchState, PlayerState, RoundState, TeamEconomy
from apexpulse.storage import DuckDBSink

BASE_TIME = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def build_state(*, alive_ct: int = 5, alive_t: int = 5) -> MatchState:
    """Build a minimal live snapshot."""
    players = tuple(
        PlayerState(
            player_id=f"ct{i}",
            name=f"CT{i}",
            team=Team.CT,
            health=100 if i < alive_ct else 0,
            money=1_000,
        )
        for i in range(5)
    ) + tuple(
        PlayerState(
            player_id=f"t{i}",
            name=f"T{i}",
            team=Team.T,
            health=100 if i < alive_t else 0,
            money=1_000,
        )
        for i in range(5)
    )
    return MatchState(
        match_id="m-1",
        map_name=MapName.MIRAGE,
        timestamp=BASE_TIME,
        round_state=RoundState(round_number=3, phase=RoundPhase.LIVE, seconds_remaining=60.0),
        players=players,
        economy_ct=TeamEconomy(team=Team.CT, money=5_000, equipment_value=12_000),
        economy_t=TeamEconomy(team=Team.T, money=5_000, equipment_value=12_000),
    )


@pytest.fixture(scope="module")
async def dataset():
    """A labelled dataset shared across the module."""
    async with DuckDBSink(path=":memory:", settings=Settings(), batch_size=10_000) as sink:
        for seed in range(12):
            simulator = MatchSimulator(
                match_id=f"m-{seed:02d}", seed=seed, tick_rate_hz=2.0, start_time=BASE_TIME
            )
            for event in simulator.run():
                await sink.handle(event)
        await sink.flush()
        return sink.training_frame()


# -- Confidence components ----------------------------------------------------


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.5, 0.0), (0.75, 0.5), (1.0, 1.0), (0.0, 1.0), (0.25, 0.5)],
)
def test_decisiveness_measures_distance_from_an_even_call(
    probability: float, expected: float
) -> None:
    assert decisiveness(probability) == pytest.approx(expected)


def test_a_steady_sequence_is_fully_stable() -> None:
    assert stability([0.7, 0.7, 0.7, 0.7]) == pytest.approx(1.0)


def test_a_thrashing_sequence_scores_low_stability() -> None:
    """A number bouncing 40/75/45/80 is not a settled read."""
    assert stability([0.4, 0.75, 0.45, 0.8]) < 0.3


def test_too_few_samples_read_as_stable_not_unstable() -> None:
    """Absence of evidence must not read as evidence of instability."""
    assert stability([0.6]) == 1.0
    assert stability([0.6, 0.9]) == 1.0


def test_a_gentle_drift_stays_mostly_stable() -> None:
    """A win probability that moves smoothly is behaving correctly."""
    assert stability([0.60, 0.62, 0.64, 0.66]) > 0.8


# -- Feature support ----------------------------------------------------------


def test_an_unfitted_support_model_penalises_nothing() -> None:
    vector = FeatureExtractor().extract(build_state())

    assert FeatureSupport().score(vector) == 1.0


async def test_a_typical_state_is_better_supported_than_an_extreme_one(dataset) -> None:
    features, _ = build_training_set(dataset)
    support = FeatureSupport.from_frame(features)
    extractor = FeatureExtractor()

    typical = support.score(extractor.extract(build_state(alive_ct=5, alive_t=5)))
    extreme = support.score(extractor.extract(build_state(alive_ct=0, alive_t=5)))

    assert typical > extreme


async def test_support_stays_within_unit_bounds(dataset) -> None:
    features, _ = build_training_set(dataset)
    support = FeatureSupport.from_frame(features)
    extractor = FeatureExtractor()

    for ct in range(6):
        for t in range(6):
            score = support.score(extractor.extract(build_state(alive_ct=ct, alive_t=t)))
            assert 0.0 <= score <= 1.0, f"{ct}v{t} scored {score}"


def test_mismatched_support_arrays_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        FeatureSupport(means=(0.1, 0.2), deviations=(0.1,))


def test_a_non_positive_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="threshold_sigma must be positive"):
        FeatureSupport(means=(0.1,), deviations=(0.1,), threshold_sigma=0.0)


def test_support_round_trips_through_json() -> None:
    original = FeatureSupport(means=(0.1, 0.2), deviations=(0.3, 0.4), threshold_sigma=2.5)

    restored = FeatureSupport.from_dict(original.to_dict())

    assert restored.means == original.means
    assert restored.threshold_sigma == original.threshold_sigma


# -- Confidence scoring -------------------------------------------------------


def test_a_decisive_prediction_scores_higher_than_a_coin_flip() -> None:
    scorer = ConfidenceScorer()

    decisive = scorer.score("m-1", 0.95)
    uncertain = scorer.score("m-1", 0.51)

    assert decisive.score > uncertain.score
    assert decisive.band is ConfidenceBand.HIGH
    assert uncertain.band is ConfidenceBand.LOW


def test_a_weak_component_drags_the_whole_score() -> None:
    """Geometric mean, not average: a decisive but unstable read is not trusted."""
    scorer = ConfidenceScorer()
    for value in (0.2, 0.9, 0.3, 0.95):
        scorer.observe("m-1", value)

    unstable = scorer.score("m-1", 0.95)

    assert unstable.stability < 0.3
    assert unstable.score < 0.7, "instability must pull the score down"


def test_the_explanation_names_the_limiting_component() -> None:
    scorer = ConfidenceScorer()
    for value in (0.1, 0.9, 0.2, 0.95):
        scorer.observe("m-1", value)

    explanation = scorer.score("m-1", 0.99).explain()

    assert "stability" in explanation


def test_history_is_tracked_per_match() -> None:
    scorer = ConfidenceScorer()
    scorer.observe("match-a", 0.9)
    scorer.observe("match-b", 0.1)

    assert scorer.recent("match-a") == (0.9,)
    assert scorer.recent("match-b") == (0.1,)


def test_resetting_clears_one_match_only() -> None:
    scorer = ConfidenceScorer()
    scorer.observe("match-a", 0.9)
    scorer.observe("match-b", 0.1)

    scorer.reset("match-a")

    assert scorer.recent("match-a") == ()
    assert scorer.recent("match-b") == (0.1,)


def test_history_is_bounded_by_the_window() -> None:
    scorer = ConfidenceScorer(window=4)
    for value in range(10):
        scorer.observe("m-1", value / 10)

    assert len(scorer.recent("m-1")) == 4


@pytest.mark.parametrize(
    ("score", "band"),
    [
        (0.9, ConfidenceBand.HIGH),
        (0.5, ConfidenceBand.MEDIUM),
        (0.1, ConfidenceBand.LOW),
    ],
)
def test_bands_bucket_scores(score: float, band: ConfidenceBand) -> None:
    assert ConfidenceBand.from_score(score) is band


def test_only_a_low_band_is_untrustworthy() -> None:
    scorer = ConfidenceScorer()

    assert scorer.score("m-1", 0.95).is_trustworthy is True
    assert scorer.score("m-1", 0.50).is_trustworthy is False


# -- Population stability index -----------------------------------------------


def test_identical_distributions_have_zero_psi() -> None:
    import numpy as np

    values = np.random.default_rng(0).normal(0, 1, 5_000).tolist()

    assert population_stability_index(values, values) < 0.01


def test_a_shifted_distribution_is_flagged() -> None:
    import numpy as np

    rng = np.random.default_rng(1)
    reference = rng.normal(0, 1, 5_000).tolist()
    shifted = rng.normal(2, 1, 5_000).tolist()

    assert population_stability_index(reference, shifted) > PSI_SHIFTED


def test_a_slightly_moved_distribution_stays_stable() -> None:
    """Sampling noise must not trip the alarm."""
    import numpy as np

    rng = np.random.default_rng(2)
    reference = rng.normal(0, 1, 10_000).tolist()
    resampled = rng.normal(0, 1, 10_000).tolist()

    assert population_stability_index(reference, resampled) < PSI_STABLE


def test_an_empty_sample_yields_zero() -> None:
    assert population_stability_index([], [1.0, 2.0]) == 0.0
    assert population_stability_index([1.0, 2.0], []) == 0.0


def test_a_constant_reference_only_flags_new_variation() -> None:
    """A feature that never varied in training cannot be binned."""
    constant = [0.5] * 1_000

    assert population_stability_index(constant, constant) == 0.0
    assert population_stability_index(constant, [0.1, 0.9] * 500) >= PSI_SHIFTED


# -- Drift detection ----------------------------------------------------------


async def test_unseen_matches_from_the_same_generator_do_not_trigger_retraining(
    dataset,
) -> None:
    """Ordinary variation between matches must not raise the retrain flag.

    The assertion is on severity, not on every feature reading stable. Several
    features are strongly match-correlated — economy and scoreline especially —
    so across the four held-out matches this fixture holds, a moderate PSI is
    expected sampling noise. Measured on the same generator, `money_ratio` reads
    0.147 over twelve matches and 0.017 over forty; nothing drifted in either.

    Severe is the verdict that triggers action, so severe is what must stay
    clear.
    """
    train_frame, test_frame = split_by_match(dataset, test_fraction=0.3)
    train_features, _ = build_training_set(train_frame)
    test_features, _ = build_training_set(test_frame)

    detector = DriftDetector.from_frame(train_features)
    report = detector.detect({name: test_features[name].tolist() for name in FEATURE_NAMES})

    assert report.should_retrain is False, report.summary()
    assert report.severity is not DriftSeverity.UNKNOWN


async def test_a_per_match_feature_needs_matches_not_rows(dataset) -> None:
    """Document the limitation rather than hide it.

    `score_delta` is constant within a match, so 14,000 rows drawn from four
    matches carry four independent values. PSI reads that as drift when nothing
    has drifted, which is why the module docstring records the measured curve:
    1.10 at twelve matches, 0.19 at twenty, 0.11 at forty.
    """
    _, test_frame = split_by_match(dataset, test_fraction=0.3)

    held_out_matches = test_frame["match_id"].nunique()
    distinct_values = test_frame["score_ct"].sub(test_frame["score_t"]).nunique()

    assert held_out_matches < 10, "this fixture is deliberately small"
    # Far fewer distinct values than rows: the rows are not independent samples.
    assert distinct_values < len(test_frame) / 100


async def test_a_shifted_feature_is_detected(dataset) -> None:
    """Simulates a patch that changes economy values."""
    train_features, _ = build_training_set(dataset)
    detector = DriftDetector.from_frame(train_features)

    live = {name: train_features[name].tolist() for name in FEATURE_NAMES}
    live["money_ratio"] = [value * 0.2 + 0.7 for value in live["money_ratio"]]

    report = detector.detect(live)

    assert report.severity is DriftSeverity.SEVERE
    assert report.should_retrain is True
    assert report.drifted_features[0].name == "money_ratio"


async def test_a_feature_stuck_at_a_constant_is_detected(dataset) -> None:
    """The shape of an upstream bug: a feature silently stops varying."""
    train_features, _ = build_training_set(dataset)
    detector = DriftDetector.from_frame(train_features)

    live = {name: train_features[name].tolist() for name in FEATURE_NAMES}
    live["alive_delta"] = [0.0] * len(live["alive_delta"])

    report = detector.detect(live)

    assert report.should_retrain is True
    assert any(feature.name == "alive_delta" for feature in report.drifted_features)


async def test_too_few_samples_report_unknown_not_stable(dataset) -> None:
    """Silence is not a clean bill of health."""
    train_features, _ = build_training_set(dataset)
    detector = DriftDetector.from_frame(train_features)

    live = {name: train_features[name].tolist()[:10] for name in FEATURE_NAMES}
    report = detector.detect(live)

    assert report.severity is DriftSeverity.UNKNOWN
    assert report.should_retrain is False
    assert str(MIN_SAMPLES) in report.summary()


def test_an_unfitted_detector_reports_unknown() -> None:
    report = DriftDetector().detect({"alive_delta": [0.1] * 500})

    assert report.severity is DriftSeverity.UNKNOWN


async def test_the_worst_feature_sets_the_overall_verdict(dataset) -> None:
    """One badly drifted input invalidates a prediction, so no averaging."""
    train_features, _ = build_training_set(dataset)
    detector = DriftDetector.from_frame(train_features)

    live = {name: train_features[name].tolist() for name in FEATURE_NAMES}
    live["time_fraction"] = [0.99] * len(live["time_fraction"])

    report = detector.detect(live)

    assert report.severity is DriftSeverity.SEVERE
    assert len(report.drifted_features) >= 1


async def test_a_detector_round_trips_through_json(dataset) -> None:
    train_features, _ = build_training_set(dataset)
    original = DriftDetector.from_frame(train_features, sample_limit=500)

    restored = DriftDetector.from_dict(original.to_dict())

    assert set(restored.reference) == set(original.reference)
    assert restored.min_samples == original.min_samples


async def test_the_drift_table_renders(dataset) -> None:
    train_features, _ = build_training_set(dataset)
    detector = DriftDetector.from_frame(train_features)
    report = detector.detect({name: train_features[name].tolist() for name in FEATURE_NAMES})

    table = format_drift_table(report)

    assert "PSI" in table
    assert "alive_delta" in table


# -- Situation benchmarking ---------------------------------------------------


@pytest.fixture(scope="module")
async def benchmark(dataset):
    """Train once, then benchmark the held-out predictions by situation."""
    import xgboost as xgb

    booster, calibrator, _ = train_model(dataset, num_rounds=150, seed=7)
    _, test_frame = split_by_match(dataset, test_fraction=0.2)
    features, labels = build_training_set(test_frame)

    matrix = xgb.DMatrix(features, feature_names=list(FEATURE_NAMES))
    raw = booster.predict(matrix, iteration_range=(0, booster.best_iteration + 1))
    probabilities = calibrator.apply(raw)

    return benchmark_by_situation(test_frame, labels, probabilities)


async def test_the_benchmark_covers_the_expected_situations(benchmark) -> None:
    names = {item.name for item in benchmark.slices}

    assert "post-plant" in names
    assert "even manpower" in names
    assert benchmark.overall.name == "overall"


async def test_lopsided_positions_are_easier_than_even_ones(benchmark) -> None:
    """The model should be sharpest when manpower already decided the round."""
    by_name = {item.name: item for item in benchmark.slices}

    lopsided = by_name.get("lopsided")
    even = by_name.get("even manpower")
    if lopsided is None or even is None:
        pytest.skip("slices too small in this dataset")

    assert lopsided.skill_score > even.skill_score
    assert lopsided.roc_auc > even.roc_auc


async def test_the_overall_figure_beats_the_base_rate(benchmark) -> None:
    assert benchmark.overall.skill_score > 0.0


async def test_the_weakest_slice_is_identified(benchmark) -> None:
    weakest = benchmark.weakest

    assert weakest is not None
    assert all(weakest.skill_score <= item.skill_score for item in benchmark.slices)


async def test_small_slices_are_omitted(dataset) -> None:
    """A twenty-row slice would report noise as a result."""
    import xgboost as xgb

    booster, _, _ = train_model(dataset, num_rounds=60, seed=7)
    _, test_frame = split_by_match(dataset, test_fraction=0.2)
    features, labels = build_training_set(test_frame)
    matrix = xgb.DMatrix(features, feature_names=list(FEATURE_NAMES))
    probabilities = booster.predict(matrix, iteration_range=(0, booster.best_iteration + 1))

    report = benchmark_by_situation(test_frame, labels, probabilities, min_slice_size=10_000)

    assert all(item.count >= 10_000 for item in report.slices)


async def test_the_benchmark_table_renders(benchmark) -> None:
    table = format_benchmark_table(benchmark)

    assert "situation" in table
    assert "overall" in table
    assert "weakest situation" in table


async def test_the_report_serialises(benchmark) -> None:
    payload = benchmark.as_dict()

    assert payload["overall"]["name"] == "overall"
    assert isinstance(payload["slices"], list)
    assert isinstance(payload["failing"], list)
