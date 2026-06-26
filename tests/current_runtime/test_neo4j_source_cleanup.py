"""Runtime contract checks for source-scoped Neo4j cleanup safety."""

import pytest

from scripts.neo4j_source_cleanup import cleanup_scope, count_scope, validate_source_ref


class _Record:
    """Minimal Neo4j record double."""

    def __init__(self, count: int) -> None:
        self._count = count

    def __getitem__(self, key: str) -> int:
        assert key == "count"
        return self._count


class _Result:
    """Minimal Neo4j result double."""

    def __init__(self, count: int) -> None:
        self._count = count

    def single(self) -> _Record:
        return _Record(self._count)


class _Session:
    """Capture Cypher and params without touching Neo4j."""

    def __init__(self) -> None:
        self.calls = []

    def run(self, cypher, **params):
        self.calls.append((cypher, params))
        return _Result(len(self.calls))


@pytest.mark.parametrize("source_ref", ["", "*", "all", "ALL", "abc", "neo4j://graph/main"])
def test_cleanup_rejects_empty_or_broad_scope(source_ref: str) -> None:
    """Cleanup must not accept empty, wildcard, or broad graph scopes."""
    with pytest.raises(ValueError):
        validate_source_ref(source_ref)


def test_cleanup_accepts_explicit_source_ref() -> None:
    """A concrete document/run source ref is a valid cleanup scope."""
    assert validate_source_ref("storage://document/abc123") == "storage://document/abc123"


def test_cleanup_dry_run_counts_only_exact_source_ref() -> None:
    """Dry-run should only count relationships and isolated nodes for the exact source ref."""
    session = _Session()

    counts = count_scope(session, "storage://document/abc123")

    assert counts.mode == "dry-run"
    assert counts.scoped_relationships == 1
    assert counts.isolated_scoped_nodes == 2
    assert all("coalesce(" in cypher for cypher, _ in session.calls)
    assert all(params == {"source_ref": "storage://document/abc123"} for _, params in session.calls)


def test_cleanup_execute_deletes_only_scoped_relationships_and_isolated_nodes() -> None:
    """Execute mode should not use broad DETACH DELETE or unscoped MATCH deletion."""
    session = _Session()

    counts = cleanup_scope(session, "storage://document/abc123")

    assert counts.mode == "execute"
    assert counts.scoped_relationships == 1
    assert counts.isolated_scoped_nodes == 2
    for cypher, params in session.calls:
        assert "DETACH DELETE" not in cypher
        assert "$source_ref IN coalesce(" in cypher
        assert params == {"source_ref": "storage://document/abc123"}
