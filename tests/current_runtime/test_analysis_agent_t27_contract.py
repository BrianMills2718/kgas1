"""Runtime contract checks for AnalysisAgent relationship extraction inputs."""

import pytest

from src.orchestration.agents.analysis_agent import _normalize_entities_for_t27


def test_t23a_entities_convert_to_t27_contract() -> None:
    """T23A output should be accepted by the analysis-agent relationship bridge."""
    entities = [
        {
            "surface_form": "Alice",
            "entity_type": "PERSON",
            "start_pos": 0,
            "end_pos": 5,
            "confidence": 0.91,
            "chunk_ref": "chunk-1",
        },
        {
            "surface_form": "Acme Corp",
            "entity_type": "ORG",
            "start_pos": 15,
            "end_pos": 24,
            "chunk_ref": "chunk-1",
        },
    ]

    normalized = _normalize_entities_for_t27(entities)

    assert normalized[0]["text"] == "Alice"
    assert normalized[0]["entity_type"] == "PERSON"
    assert normalized[0]["start"] == 0
    assert normalized[0]["end"] == 5
    assert normalized[0]["confidence"] == 0.91
    assert normalized[0]["surface_form"] == "Alice"
    assert normalized[1]["confidence"] == 0.8


def test_existing_t27_entities_pass_through_with_default_confidence() -> None:
    """Already-normalized T27 entities should remain usable."""
    entities = [
        {"text": "Alice", "entity_type": "PERSON", "start": 0, "end": 5},
        {"text": "Acme Corp", "entity_type": "ORG", "start": 15, "end": 24, "confidence": 0.95},
    ]

    normalized = _normalize_entities_for_t27(entities)

    assert normalized[0] == {
        "text": "Alice",
        "entity_type": "PERSON",
        "start": 0,
        "end": 5,
        "confidence": 0.8,
    }
    assert normalized[1]["confidence"] == 0.95


def test_invalid_entities_fail_loudly() -> None:
    """Unknown entity shapes should not be silently passed to T27."""
    with pytest.raises(ValueError, match="not T27-compatible"):
        _normalize_entities_for_t27([{"name": "Alice", "type": "PERSON"}])
