"""Runtime contract checks for graph-builder adapter behavior."""

import pytest

from src.analytics.graph_builder import GraphBuilder
from src.tools.phase1.t31_entity_builder_unified import T31EntityBuilderUnified
from src.tools.phase1.t34_edge_builder_unified import T34EdgeBuilderUnified


class _FailingEdgeBuilder:
    """Raise if the graph builder calls T34 for an empty relationship list."""

    def execute(self, request):
        raise AssertionError("edge builder should not be called")


class _CapturingNeo4jManager:
    """Capture graph validation queries without touching a real database."""

    def __init__(self) -> None:
        self.queries = []

    async def execute_read_query(self, query, params=None):
        """Record the query and return bounded graph summary data."""
        self.queries.append((query, params))
        return [{"node_count": 4, "relationship_count": 5}]


@pytest.mark.asyncio
async def test_graph_builder_allows_node_only_graphs_without_t34_call() -> None:
    """A document with entities but no extracted relationships should still build a node-only graph."""
    graph_builder = object.__new__(GraphBuilder)
    graph_builder.edge_builder = _FailingEdgeBuilder()

    result = await graph_builder._build_edges_real([], ["storage://document/test"])

    assert result == {
        "status": "success",
        "edges": [],
        "edge_count": 0,
        "confidence": 1.0,
        "neo4j_operations": 0,
        "weight_distribution": {},
        "relationship_types": {},
    }


@pytest.mark.asyncio
async def test_graph_connectivity_validation_uses_bounded_summary_query() -> None:
    """Request-time graph validation should not run unbounded component traversal."""
    graph_builder = object.__new__(GraphBuilder)
    graph_builder.neo4j_manager = _CapturingNeo4jManager()

    result = await graph_builder._analyze_graph_connectivity()

    query = graph_builder.neo4j_manager.queries[0][0]
    assert "CALL {" not in query
    assert "[*]" not in query
    assert result["connectivity_check"] == "not_computed_unbounded_traversal_skipped"
    assert result["node_count"] == 4
    assert result["relationship_count"] == 5


def test_t31_persists_source_refs_on_new_entity_nodes(monkeypatch) -> None:
    """New Neo4j entity writes should carry source refs for future scoped verification."""
    captured = {}
    builder = object.__new__(T31EntityBuilderUnified)

    def fake_create(entity_info, mentions):
        captured["entity_info"] = entity_info
        return {
            "status": "success",
            "neo4j_id": "node-1",
            "properties": {"source_refs": entity_info["source_refs"]},
        }

    monkeypatch.setattr(builder, "_create_neo4j_entity_node", fake_create)

    entity = builder._build_single_entity(
        "PERSON_0001",
        [{"text": "Alice", "entity_type": "PERSON", "confidence": 0.9}],
        ["storage://document/source-scope"],
    )

    assert entity is not None
    assert captured["entity_info"]["source_refs"] == ["storage://document/source-scope"]
    assert entity["properties"]["source_refs"] == ["storage://document/source-scope"]


def test_t34_persists_source_refs_on_new_relationship_edges(monkeypatch) -> None:
    """New Neo4j relationship writes should carry source refs for future scoped verification."""
    captured = {}
    builder = object.__new__(T34EdgeBuilderUnified)
    builder.driver = object()
    builder.min_weight = 0.1
    builder.max_weight = 1.0
    builder.confidence_weight_factor = 0.8

    def fake_create(relationship, weight, source_refs):
        captured["source_refs"] = source_refs
        return {
            "status": "success",
            "neo4j_rel_id": "rel-1",
            "properties": {"source_refs": source_refs},
        }

    monkeypatch.setattr(builder, "_create_neo4j_relationship_edge", fake_create)

    edge = builder._build_single_edge(
        {
            "subject": {"text": "Alice"},
            "object": {"text": "Acme"},
            "relationship_type": "WORKS_FOR",
            "confidence": 0.9,
            "evidence_text": "Alice works for Acme.",
        },
        ["storage://document/source-scope"],
    )

    assert edge is not None
    assert captured["source_refs"] == ["storage://document/source-scope"]
    assert edge["properties"]["source_refs"] == ["storage://document/source-scope"]
