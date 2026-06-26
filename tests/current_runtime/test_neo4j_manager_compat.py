"""Compatibility checks for Neo4j manager APIs used by analytics callers."""

import pytest

from src.core.neo4j_manager import Neo4jDockerManager


class _FakeResult:
    """Iterate like a Neo4j result over dict-convertible records."""

    def __iter__(self):
        """Return deterministic records from the fake session."""
        return iter([{"ok": True}])


class _FakeSession:
    """Capture internal read queries executed by the manager."""

    def __init__(self, calls) -> None:
        self.calls = []
        self._shared_calls = calls

    def __enter__(self):
        """Enter the fake session context."""
        return self

    def __exit__(self, exc_type, exc, tb):
        """Exit the fake session context."""
        return False

    def run(self, query, params=None):
        """Record the query and return a deterministic result."""
        self.calls.append((query, params))
        self._shared_calls.append((query, params))
        return _FakeResult()


class _FakeDriver:
    """Provide fake Neo4j sessions."""

    def __init__(self) -> None:
        self.calls = []

    def session(self):
        """Return a context-manager session."""
        return _FakeSession(self.calls)


class _FakeConnectionManager:
    """Provide the fake driver expected by Neo4jDockerManager."""

    def __init__(self) -> None:
        self.driver = _FakeDriver()

    def get_driver(self):
        """Return the fake Neo4j driver."""
        return self.driver


@pytest.mark.asyncio
async def test_execute_read_query_uses_internal_driver_session() -> None:
    """Graph analytics callers expect an async read-query method on the manager."""
    manager = object.__new__(Neo4jDockerManager)
    manager.connection_manager = _FakeConnectionManager()

    result = await manager.execute_read_query("RETURN $value AS value", {"value": 1})

    assert result == [{"ok": True}]
    assert manager.connection_manager.driver.calls == [("RETURN $value AS value", {"value": 1})]
