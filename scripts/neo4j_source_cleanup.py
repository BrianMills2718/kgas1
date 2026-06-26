#!/usr/bin/env python3
"""Dry-run-first cleanup for one explicit Neo4j source_ref scope.

This helper exists to avoid broad graph cleanup. It deletes only relationships
whose `source_refs` list contains the exact requested source ref, then deletes
only source-scoped nodes that have no remaining relationships.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase


FORBIDDEN_SOURCE_REFS = {"", "*", "all", "ALL", "neo4j://graph/main"}


@dataclass(frozen=True)
class CleanupCounts:
    """Counts reported by a scoped Neo4j cleanup dry-run or execution."""

    source_ref: str
    scoped_relationships: int
    isolated_scoped_nodes: int
    mode: str


def validate_source_ref(source_ref: str) -> str:
    """Validate that cleanup scope is explicit and not broad."""
    normalized = source_ref.strip()
    if normalized in FORBIDDEN_SOURCE_REFS:
        raise ValueError("source_ref must be an explicit non-broad value")
    if len(normalized) < 8:
        raise ValueError("source_ref is too short to be a safe cleanup scope")
    if re.search(r"[*?]", normalized):
        raise ValueError("source_ref wildcards are not allowed")
    return normalized


def load_env_file(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE entries from .env without overriding the shell."""
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def get_neo4j_driver() -> Any:
    """Create a Neo4j driver from env or ignored local .env credentials."""
    load_env_file()
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise RuntimeError("NEO4J_PASSWORD is required; set env or ignored .env")
    return GraphDatabase.driver(uri, auth=(user, password))


def count_scope(session: Any, source_ref: str) -> CleanupCounts:
    """Count scoped relationships and isolated scoped nodes without mutating data."""
    rel_count = session.run(
        """
        MATCH ()-[r]->()
        WHERE $source_ref IN coalesce(r.source_refs, [])
        RETURN count(r) AS count
        """,
        source_ref=source_ref,
    ).single()["count"]
    node_count = session.run(
        """
        MATCH (n)
        WHERE $source_ref IN coalesce(n.source_refs, [])
          AND NOT (n)--()
        RETURN count(n) AS count
        """,
        source_ref=source_ref,
    ).single()["count"]
    return CleanupCounts(
        source_ref=source_ref,
        scoped_relationships=rel_count,
        isolated_scoped_nodes=node_count,
        mode="dry-run",
    )


def cleanup_scope(session: Any, source_ref: str) -> CleanupCounts:
    """Delete only relationships and isolated nodes for one explicit source ref."""
    rel_deleted = session.run(
        """
        MATCH ()-[r]->()
        WHERE $source_ref IN coalesce(r.source_refs, [])
        WITH r, count(r) AS count
        DELETE r
        RETURN count
        """,
        source_ref=source_ref,
    ).single()["count"]
    node_deleted = session.run(
        """
        MATCH (n)
        WHERE $source_ref IN coalesce(n.source_refs, [])
          AND NOT (n)--()
        WITH n, count(n) AS count
        DELETE n
        RETURN count
        """,
        source_ref=source_ref,
    ).single()["count"]
    return CleanupCounts(
        source_ref=source_ref,
        scoped_relationships=rel_deleted,
        isolated_scoped_nodes=node_deleted,
        mode="execute",
    )


def run_cleanup(source_ref: str, execute: bool = False) -> CleanupCounts:
    """Run a dry-run or scoped cleanup for a validated source ref."""
    validated_source_ref = validate_source_ref(source_ref)
    driver = get_neo4j_driver()
    try:
        with driver.session() as session:
            if execute:
                return cleanup_scope(session, validated_source_ref)
            return count_scope(session, validated_source_ref)
    finally:
        driver.close()


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ref", required=True, help="Exact source_refs value to inspect/delete")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Delete scoped relationships and isolated scoped nodes. Omit for dry-run.",
    )
    args = parser.parse_args()

    counts = run_cleanup(args.source_ref, execute=args.execute)
    print(
        f"mode={counts.mode} source_ref={counts.source_ref!r} "
        f"relationships={counts.scoped_relationships} isolated_nodes={counts.isolated_scoped_nodes}"
    )
    if counts.mode == "dry-run":
        print("No data deleted. Re-run with --execute to delete only this exact source_ref scope.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
