"""Runtime contract checks for source-scoped T49 query filtering."""

from src.tools.phase1.multihop_query.connection_manager import Neo4jConnectionManager
from src.tools.phase1.multihop_query.path_finder import PathFinder
from src.tools.phase1.multihop_query.query_entity_extractor import QueryEntityExtractor


class _CapturingEntityConnection:
    """Capture entity lookup scope without touching Neo4j."""

    driver = object()

    def __init__(self) -> None:
        self.calls = []

    def find_entities_by_name(self, entity_name, limit=10, source_refs=None):
        self.calls.append((entity_name, limit, source_refs))
        return [
            {
                "entity_id": "entity-1",
                "canonical_name": entity_name,
                "entity_type": "PERSON",
                "confidence": 0.9,
                "pagerank_score": 0.01,
            }
        ]


class _CapturingPathConnection:
    """Capture path-finder source scope without touching Neo4j."""

    driver = object()

    def __init__(self) -> None:
        self.pair_scope = []
        self.related_scope = []

    def find_paths_between_entities(self, source_id, target_id, max_hops, limit_per_hop=5, source_refs=None):
        self.pair_scope.append(source_refs)
        return []

    def find_related_entities(self, entity_id, max_hops, limit=20, source_refs=None):
        self.related_scope.append(source_refs)
        return []


class _CapturingQueryConnection:
    """Capture generated Cypher and params without opening a database connection."""

    driver = object()

    def __init__(self) -> None:
        self.calls = []

    def execute_query(self, cypher, parameters=None):
        self.calls.append((cypher, parameters or {}))
        return []


def test_query_entity_extractor_passes_source_refs_to_database_lookup() -> None:
    """Entity lookup should filter duplicate names by source scope when provided."""
    connection = _CapturingEntityConnection()
    extractor = QueryEntityExtractor(connection)

    entities = extractor.extract_query_entities("Who is connected to Alice?", source_refs=["doc://scope"])

    assert entities[0]["canonical_name"] == "Alice"
    assert connection.calls == [("Alice", 3, ["doc://scope"])]


def test_path_finder_passes_source_refs_to_pair_and_related_queries() -> None:
    """Path expansion should keep the same source scope for all graph lookups."""
    connection = _CapturingPathConnection()
    finder = PathFinder(connection)
    query_entities = [
        {"entity_id": "alice", "canonical_name": "Alice"},
        {"entity_id": "acme", "canonical_name": "Acme"},
    ]

    finder.find_multihop_paths(query_entities, max_hops=2, result_limit=10, source_refs=["doc://scope"])

    assert connection.pair_scope == [["doc://scope"]]
    assert connection.related_scope == [["doc://scope"], ["doc://scope"]]


def test_connection_manager_entity_lookup_uses_optional_source_refs_filter(monkeypatch) -> None:
    """Entity-name lookup Cypher should include a source_refs filter only parameterized by scope."""
    connection = object.__new__(Neo4jConnectionManager)
    connection.driver = None
    capture = _CapturingQueryConnection()
    monkeypatch.setattr(connection, "execute_query", capture.execute_query)

    connection.find_entities_by_name("Alice", limit=3, source_refs=["doc://scope"])

    cypher, params = capture.calls[0]
    assert "e.source_refs" in cypher
    assert params["source_refs"] == ["doc://scope"]


def test_connection_manager_path_queries_use_source_refs_for_nodes_and_edges(monkeypatch) -> None:
    """Scoped path queries should require matching source refs on traversed nodes and edges."""
    connection = object.__new__(Neo4jConnectionManager)
    connection.driver = None
    capture = _CapturingQueryConnection()
    monkeypatch.setattr(connection, "execute_query", capture.execute_query)

    connection.find_paths_between_entities("alice", "acme", max_hops=1, source_refs=["doc://scope"])
    connection.find_related_entities("alice", max_hops=1, source_refs=["doc://scope"])

    path_cypher, path_params = capture.calls[0]
    related_cypher, related_params = capture.calls[1]
    assert "nodes(path)" in path_cypher
    assert "relationships(path)" in path_cypher
    assert "nodes(path)" in related_cypher
    assert "relationships(path)" in related_cypher
    assert path_params["source_refs"] == ["doc://scope"]
    assert related_params["source_refs"] == ["doc://scope"]
