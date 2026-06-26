"""Runtime contract checks for graph-builder adapter behavior."""

import pytest

from src.analytics.graph_builder import GraphBuilder


class _FailingEdgeBuilder:
    """Raise if the graph builder calls T34 for an empty relationship list."""

    def execute(self, request):
        raise AssertionError("edge builder should not be called")


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
