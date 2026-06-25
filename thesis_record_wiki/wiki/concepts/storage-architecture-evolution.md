---
type: Concept
title: Storage Architecture Evolution
description: KGAS storage decisions evolved from tri-store avoidance to Neo4j plus SQLite, then to Neo4j plus PostgreSQL at larger analytical scale.
tags: [concept, storage, architecture, neo4j, sqlite, postgresql]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-003-Vector-Store-Consolidation.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-009-Bi-Store-Database-Strategy.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-030-PostgreSQL-Migration-Strategy.md
confidence: medium
---

# Summary

KGAS storage architecture evolved by reducing avoidable operational complexity first, then reintroducing relational database strength only when scale justified it.

The through-line is bi-store separation:

- graph and embeddings belong with graph analysis in Neo4j
- operational metadata, provenance, workflow state, and PII belong outside the graph
- the non-graph store can be SQLite for local academic simplicity or PostgreSQL for larger analytical workloads [1][2][3]

# Decision Pattern

## Simplicity over tri-store consistency

The earliest storage decision in this slice rejects Neo4j + SQLite + Qdrant because it needs outbox/reconciliation machinery to keep graph and vector state aligned. Neo4j 5.13+ native vector indexes made graph + vector writes atomic enough for the project's stated scope. [1]

## Separation over single-store purity

ADR-009 keeps two stores because KGAS needs both graph-native research operations and relational operational metadata. This is not a generic polyglot-persistence preference; it is tied to academic graph analysis, provenance, checkpoints, and PII handling. [2]

## Scale threshold over premature PostgreSQL

ADR-030 accepts PostgreSQL when workloads exceed 50,000 entities and need correlation, regression, window functions, concurrent analytics, BI connectivity, and larger corpus support. It does not retroactively invalidate the SQLite choice for smaller single-researcher workflows. [3]

# Implications

Storage claims in KGAS documents should be interpreted with three qualifiers:

- **Date**: older target docs may expect SQLite where later ADRs expect PostgreSQL.
- **Scale**: SQLite is design-appropriate for small/local academic use; PostgreSQL is design-appropriate for large analytical corpora.
- **Status level**: ADRs describe accepted target architecture; implementation status still needs roadmap, evidence, or code verification.

# Links

- [Digimon Lineage Data Storage ADRs](/wiki/sources/digimon-lineage-data-storage-adrs.md)
- [Documentation Status Truthfulness](/wiki/concepts/documentation-status-truthfulness.md)
- [Reality Verification Arc](/wiki/concepts/reality-verification-arc.md)
- [Academic Proof Of Concept Scope](/wiki/concepts/academic-proof-of-concept-scope.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-003-Vector-Store-Consolidation.md`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-009-Bi-Store-Database-Strategy.md`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-030-PostgreSQL-Migration-Strategy.md`
