"""No-IO smoke coverage for the current confidence-scoring package."""

from src.core.confidence_score import ConfidenceScore, get_confidence_scoring_info


def test_confidence_scoring_cerqual_range_and_combination_smoke() -> None:
    """Confidence scoring should support CERQual creation, ranges, and combination without IO."""
    score = ConfidenceScore.create_with_cerqual(
        methodological_limitations=0.8,
        relevance=0.7,
        coherence=0.75,
        adequacy_of_data=0.65,
        evidence_weight=2,
        source="test_confidence_scoring_smoke",
    )

    range_score = score.set_confidence_range(0.55, 0.85)
    combined = range_score.combine_with(ConfidenceScore.create_medium_confidence())

    assert get_confidence_scoring_info()["module"] == "confidence_score"
    assert 0.0 <= score.value <= 1.0
    assert range_score.confidence_range == (0.55, 0.85)
    assert 0.0 <= combined.value <= 1.0
    assert combined.propagation_method.value == "bayesian_evidence_power"
    assert combined.metadata["combination_method"] == "bayesian_evidence_power"
