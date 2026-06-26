"""Runtime contract checks for T49 query entity extraction."""

from src.tools.phase1.multihop_query.query_entity_extractor import QueryEntityExtractor


class _NoDatabaseConnection:
    """Placeholder connection manager for pattern-only extraction tests."""

    driver = None


def test_potential_entities_do_not_swallow_lowercase_question_text() -> None:
    """Natural-language questions should still expose capitalized entity names."""
    extractor = QueryEntityExtractor(_NoDatabaseConnection())

    assert extractor._extract_potential_entities("Who is connected to Alice?") == ["Alice"]


def test_potential_entities_keep_multiword_entity_names() -> None:
    """Multi-token proper names should remain available for database lookup."""
    extractor = QueryEntityExtractor(_NoDatabaseConnection())

    entities = extractor._extract_potential_entities("How is Alice connected to Acme Corporation?")

    assert "Alice" in entities
    assert "Acme Corporation" in entities
    assert "How" not in entities
    assert "How is Alice connected to Acme" not in entities
