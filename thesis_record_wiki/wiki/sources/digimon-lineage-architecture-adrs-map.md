---
type: Source
title: Digimon Lineage Architecture ADRs Map
description: First bounded map of architecture ADRs in the large Digimons lineage bundle.
tags: [source, architecture, adrs, kgas, decisions]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-011-Academic-Research-Focus.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-012-Single-Node-Design.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-014-Error-Handling-Strategy.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-022-Theory-Selection-Architecture.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-027-Analytical-Purpose-Clarification.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-028-Tool-Interface-Layer-Architecture.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/architectural-exploration-references.md
confidence: medium
---

# Summary

`docs/architecture/adrs/` contains 32 files and about 452K of architecture decision material. This first ADR slice maps the core decision backbone rather than ingesting every ADR in detail.

The sampled ADRs show KGAS being intentionally framed as an academic research proof-of-concept, local single-node system, fail-fast transparent research tool, two-layer theory extraction/analysis architecture, human question-driven analytical-purpose system, and three-layer tool interface architecture. [1][2][3][4][5][6]

# Core Decision Backbone

1. **Academic proof-of-concept over enterprise product**: ADR-011 explicitly deprioritizes enterprise scalability, 24/7 monitoring, distributed processing, and enterprise security in favor of correctness, provenance, flexibility, and local academic workflows. [1]
2. **Single-node local deployment**: ADR-012 chooses local researcher machines, embedded/local Neo4j + SQLite, single-user usage, offline capability, and reproducibility over distributed/cloud architecture. [2]
3. **Fail-fast error handling**: ADR-014 rejects graceful degradation and silent failures because academic research integrity requires visible failures with complete context. [3]
4. **Two-layer theory architecture**: ADR-022 separates theory structure extraction from question-driven analysis after Carter speech failures exposed theory-context mismatch and false precision. [4]
5. **Human question-driven analytical tier selection**: ADR-027 distinguishes text analysis, world analysis, effect analysis, and prescriptive design guidance. [5]
6. **Three-layer tool interface architecture**: ADR-028 reconciles raw/adapted tool implementation, internal KGASTool contracts, and external MCP exposure. [6]

# Relationship To Lit Review Evidence

The ADRs explain why the lit-review experiment kept returning to explicit analytical purpose and balanced five-purpose validation. ADR-022 and ADR-027 are especially close to [Automated Theory Extraction](/wiki/concepts/automated-theory-extraction.md), [Schema Extraction Pipeline Evolution](/wiki/concepts/schema-extraction-pipeline-evolution.md), and [Balance Driven Validation](/wiki/concepts/balance-driven-validation.md).

# Exploration Boundary

`architectural-exploration-references.md` says the architecture directory also preserved exploratory alternatives from `Thinking_out_loud/Architectural_Exploration/`. Those documents should be evaluated against stable ADRs rather than treated as accepted architecture. [7]

# Next ADR Slices

Recommended follow-up slices:

- data/storage ADRs: vector-store consolidation, bi-store database, PostgreSQL migration
- uncertainty/quality ADRs: confidence ontology, CERQual/IC uncertainty, quality system, entity resolution
- tool/orchestration ADRs: phase interfaces, pipeline orchestrator, MCP, structured output, tool interface layers
- analysis expansion ADRs: cross-modal, ABM, statistical analysis, schema ecosystem

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-011-Academic-Research-Focus.md`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-012-Single-Node-Design.md`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-014-Error-Handling-Strategy.md`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-022-Theory-Selection-Architecture.md`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-027-Analytical-Purpose-Clarification.md`  
[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-028-Tool-Interface-Layer-Architecture.md`  
[7] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/architectural-exploration-references.md`
