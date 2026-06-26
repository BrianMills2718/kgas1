"""Runtime contract checks for T49 result ranking."""

from src.tools.phase1.multihop_query.result_ranker import ResultRanker


def test_ranker_deduplicates_related_entities_by_names_not_local_ids() -> None:
    """Repeated local smoke-test nodes should not crowd out distinct semantic answers."""
    ranker = ResultRanker()
    results = [
        {
            "result_type": "related_entity",
            "query_entity": "Alice",
            "query_entity_id": "entity_a1",
            "related_entity": "Seattle",
            "related_entity_id": "entity_s1",
            "entity_type": "GPE",
            "confidence": 0.8,
            "pagerank_score": 0.02,
            "connection_count": 3,
            "explanation": "Seattle is connected to Alice.",
        },
        {
            "result_type": "related_entity",
            "query_entity": "Alice",
            "query_entity_id": "entity_a2",
            "related_entity": "Seattle",
            "related_entity_id": "entity_s2",
            "entity_type": "GPE",
            "confidence": 0.9,
            "pagerank_score": 0.03,
            "connection_count": 5,
            "explanation": "Seattle is connected to Alice.",
        },
        {
            "result_type": "related_entity",
            "query_entity": "Alice",
            "query_entity_id": "entity_a1",
            "related_entity": "Acme Corporation",
            "related_entity_id": "entity_o1",
            "entity_type": "ORG",
            "confidence": 0.75,
            "pagerank_score": 0.01,
            "connection_count": 2,
            "explanation": "Acme Corporation is connected to Alice.",
        },
    ]

    ranked = ranker.rank_query_results(results, "Who is connected to Alice?", min_confidence=0.1)

    assert len(ranked) == 2
    semantic_pairs = {(item["query_entity"], item["related_entity"]) for item in ranked}
    assert semantic_pairs == {("Alice", "Seattle"), ("Alice", "Acme Corporation")}
    seattle = next(item for item in ranked if item["related_entity"] == "Seattle")
    assert seattle["related_entity_id"] == "entity_s2"


def test_ranker_deduplicates_paths_by_path_names_and_relationships() -> None:
    """Path duplicates should collapse even when local IDs differ."""
    ranker = ResultRanker()
    results = [
        {
            "result_type": "path",
            "source_entity_id": "entity_a1",
            "target_entity_id": "entity_b1",
            "path": ["Alice", "Acme Corporation"],
            "relationship_types": ["WORKS_FOR"],
            "path_length": 1,
            "path_weight": 0.8,
            "confidence": 0.8,
        },
        {
            "result_type": "path",
            "source_entity_id": "entity_a2",
            "target_entity_id": "entity_b2",
            "path": ["Alice", "Acme Corporation"],
            "relationship_types": ["WORKS_FOR"],
            "path_length": 1,
            "path_weight": 0.9,
            "confidence": 0.85,
        },
    ]

    ranked = ranker.rank_query_results(results, "Alice Acme Corporation", min_confidence=0.1)

    assert len(ranked) == 1
    assert ranked[0]["source_entity_id"] == "entity_a2"
