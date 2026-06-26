"""Compatibility checks for Neo4j manager APIs used by analytics callers."""

import pytest

from src.core.neo4j_manager import Neo4jDockerManager


class _FakeQueryExecutor:
    """Capture queries delegated through the manager compatibility method."""

    def __init__(self) -> None:
        self.calls = []

    def execute_query(self, query, params=None):
        """Return a deterministic result while recording the delegated call."""
        self.calls.append((query, params))
        return [{"ok": True}]


@pytest.mark.asyncio
async def test_execute_read_query_delegates_to_existing_query_executor() -> None:
    """Graph analytics callers expect an async read-query method on the manager."""
    manager = object.__new__(Neo4jDockerManager)
    manager.query_executor = _FakeQueryExecutor()

    result = await manager.execute_read_query("RETURN $value AS value", {"value": 1})

    assert result == [{"ok": True}]
    assert manager.query_executor.calls == [("RETURN $value AS value", {"value": 1})]
