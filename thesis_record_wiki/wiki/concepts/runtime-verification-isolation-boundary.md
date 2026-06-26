---
type: Concept
title: Runtime Verification Isolation Boundary
description: Safe boundary for future Neo4j-backed runtime verification without deleting accumulated local graph evidence.
tags: [concept, runtime-verification, neo4j, isolation, safety]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../sources/current-runtime-import-check-2026-06-25.md
  - ../../investigations/2026-06-25-analyze-endpoint-document-pipeline.md
confidence: high
---

# Summary

The current runtime repair work proved that a tiny `.txt` document can execute the real complete GraphRAG pipeline through Neo4j node and edge creation. Repeated verification also accumulated duplicate local smoke-test entities such as Alice, Acme, Bob, and Seattle.

That accumulated graph is shared local state. It may contain useful evidence from prior probes, and deleting it is a destructive action. Future runtime verification should isolate new test data rather than cleaning the existing graph in place.

# Boundary

Safe next implementation work should use one of these isolation approaches:

1. Add per-run provenance fields such as `verification_run_id`, source document reference, or test label to smoke-test nodes and relationships.
2. Filter verification queries by that run/source scope so old smoke-test nodes cannot dominate result ranking.
3. Use a dedicated test database, container, or graph namespace for destructive cleanup tests.
4. Create explicit cleanup code only for data that the same test run created and can identify unambiguously.

Unsafe work without Brian's explicit approval:

- deleting all local Neo4j nodes or relationships;
- running broad `MATCH (n) DETACH DELETE n` cleanup against the shared local graph;
- rewriting proof expectations by ignoring accumulated duplicates instead of scoping the query;
- treating the current graph as disposable simply because it was created by tests.

# Acceptance Criteria For A Future Safe Slice

A future code slice is safe if it can prove:

- every smoke-test node and relationship has a run/source marker;
- query tests assert only against the current run/source scope;
- cleanup, if present, targets only the current run/source marker;
- tests pass when older Alice/Acme/Bob/Seattle nodes already exist;
- no raw archive content or historical thesis evidence is modified.

# Current Implementation State

The first safe implementation slice is additive: new T31 entity-node writes and T34 relationship-edge writes now carry `source_refs` from the graph-builder input. Focused current-runtime tests prove source refs are propagated without requiring a live Neo4j database.

The second safe implementation slice threads optional `source_refs` filtering through T49 query entity lookup, path expansion, and complete-pipeline smoke queries. When no source refs are supplied, query behavior remains unchanged. When source refs are supplied, entity lookup and graph traversal are constrained to nodes and relationships carrying that source ref.

This does not delete or rewrite existing graph data. On 2026-06-26, the local Neo4j database was dumped to `~/archive/phd_thesis_work/neo4j/20260626-075357/neo4j.dump` before live verification, and the source-scoped complete-pipeline smoke tests passed against the running container. The remaining cleanup question is now whether to add a cleanup command limited to one explicit `source_refs` value.

The cleanup helper now exists as `scripts/neo4j_source_cleanup.py`. It is dry-run-first, rejects empty/wildcard/broad source refs, deletes scoped relationships first, and deletes only scoped nodes that have no remaining relationships. It should remain an operator-triggered command because even scoped deletion is destructive for those scoped records.

# Why This Matters

The PhD record is partly about how claims evolved under real verification pressure. Local runtime artifacts are not as authoritative as the preserved archive, but they can still explain why later repairs were made. Isolation preserves that history while keeping future tests deterministic.

# Links

- [Current Runtime Import Check 2026-06-25](/wiki/sources/current-runtime-import-check-2026-06-25.md)
- [Current Status Verification Discipline](/wiki/concepts/current-status-verification-discipline.md)
- [Relationship Extraction Bottleneck](/wiki/concepts/relationship-extraction-bottleneck.md)

# Citations

[1] `../sources/current-runtime-import-check-2026-06-25.md`  
[2] `../../investigations/2026-06-25-analyze-endpoint-document-pipeline.md`
