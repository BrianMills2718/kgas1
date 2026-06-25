---
type: Source
title: Digimon Lineage Data Storage ADRs
description: ADR slice covering KGAS vector-store consolidation, Neo4j plus SQLite bi-store rationale, and PostgreSQL migration threshold.
tags: [source, architecture, adr, storage, neo4j, sqlite, postgresql]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-003-Vector-Store-Consolidation.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-009-Bi-Store-Database-Strategy.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-030-PostgreSQL-Migration-Strategy.md
confidence: medium
---

# Summary

This ADR slice records KGAS storage architecture as a time-ordered design evolution:

1. ADR-003 collapses a planned Neo4j + SQLite + Qdrant tri-store into Neo4j + SQLite by using Neo4j 5.13+ native vector indexing. [1]
2. ADR-009 explains the resulting Neo4j + SQLite bi-store as appropriate for single-node academic research: Neo4j handles entities, relationships, graph algorithms, and embeddings; SQLite handles provenance, workflow state, configuration, and PII vault records. [2]
3. ADR-030 later accepts a Neo4j + PostgreSQL target for 50,000+ entity workloads, large corpus analytics, statistical functions, and concurrent analytical access. [3]

# Storage Evolution

## Tri-store rejected

ADR-003 says the previous target architecture used Neo4j for graph data, Qdrant for embeddings, and SQLite for PII/workflow state. The rejection was not because Qdrant was weak, but because cross-store consistency created an extra transactional outbox, reconciliation service, eventual-consistency windows, more deployment work, and orphan-vector risk. [1]

The chosen replacement was Neo4j + SQLite:

- Neo4j stores graph data and entity embeddings in a native HNSW vector index.
- SQLite remains for encrypted PII and workflow state.
- A `VectorStore` strategy interface is retained so Qdrant or another vector backend can be restored if future scale requires it. [1]

## Bi-store justified

ADR-009 makes the academic workflow rationale explicit. Graph research questions need traversals, community detection, path analysis, centrality, and Cypher. Operational metadata needs provenance, checkpoints, PII storage, configuration, and ACID-style audit records. [2]

The ADR rejects Neo4j-only because operational metadata would pollute the graph, SQLite-only because recursive graph queries and vector search would not fit the research workload, and PostgreSQL-only because server setup and graph extensions were considered too heavy for the then-current local academic environment. [2]

## PostgreSQL accepted at scale threshold

ADR-030 later changes the non-graph store target from SQLite to PostgreSQL when the entity scale rises above 50,000. The named drivers are SQLite single-writer limits, poor performance on complex joins, missing window/statistical functions, and reported 45+ second correlation analysis on 50K entities. [3]

The target remains bi-store rather than single-store:

- Neo4j remains graph storage.
- PostgreSQL replaces SQLite for operational and analytical relational storage.
- `pgvector` appears as part of the PostgreSQL stack, but the ADR does not remove Neo4j from the graph role.
- Apache AGE is deferred rather than accepted as a Neo4j replacement. [3]

# Interpretation

The storage ADRs are not contradictory if read as scale-indexed:

- **small/local academic POC**: Neo4j + SQLite is preferred for simplicity and reproducibility.
- **larger analytical corpus**: Neo4j + PostgreSQL is accepted once concurrency, statistics, and 50K+ entity workloads become requirements.
- **dedicated vector database**: Qdrant remains a future option behind an interface, not the default architecture. [1][2][3]

This matters when interpreting later status claims. A document saying "bi-store" may mean either Neo4j + SQLite or Neo4j + PostgreSQL depending on date, scale assumptions, and whether it is target architecture or current implementation status.

# Links

- [Storage Architecture Evolution](/wiki/concepts/storage-architecture-evolution.md)
- [Academic Proof Of Concept Scope](/wiki/concepts/academic-proof-of-concept-scope.md)
- [Digimon Lineage Architecture ADRs Map](/wiki/sources/digimon-lineage-architecture-adrs-map.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-003-Vector-Store-Consolidation.md`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-009-Bi-Store-Database-Strategy.md`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-030-PostgreSQL-Migration-Strategy.md`
