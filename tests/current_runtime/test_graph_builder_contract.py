"""Runtime contract checks for graph-builder adapter behavior."""

import pytest

from src.analytics.graph_builder import GraphBuilder


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
