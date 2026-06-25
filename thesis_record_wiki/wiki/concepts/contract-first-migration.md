---
type: Concept
title: Contract-First Migration
description: Migration from split KGAS tool interfaces toward a single KGASTool contract.
tags: [contracts, migration, kgas-tool, interfaces]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_core_sparse/CLAUDE.md
  - ../archive_full_record/lineage_variants/digimon_core_sparse/contracts/README.md
  - ../archive_full_record/lineage_variants/digimon_core_sparse/Contract_First_Uncertainty_Analysis.md
  - ../archive_full_record/lineage_variants/digimon_core_sparse/Evidence_Contract_First_Implementation.md
confidence: high
---

# Summary

Contract-first migration is the sparse variant's answer to KGAS interface fragmentation. The migration target was a single `KGASTool` interface with `ToolRequest` / `ToolResult` contracts, theory integration, confidence scoring, and provenance tracking. [1][3]

# Root Cause

The uncertainty analysis identifies three competing tool interface definitions:

- `src/core/tool_protocol.py`, used by the orchestrator.
- `src/core/tool_contract.py`, the ADR-001 contract-first design.
- Legacy `base_tool.py` definitions with inconsistent `ToolRequest` / `ToolResult` shapes. [3]

This split meant the orchestrator passed raw dictionaries while tools or target contracts expected structured request objects.

# Relationship To Type-Based Composition

[Type-Based Tool Composition](/wiki/concepts/type-based-tool-composition.md) focuses on semantic data types and exact schemas for compatibility. Contract-first migration is a related but more architectural move: it standardizes tool execution interfaces, service contracts, validation, provenance, and orchestration boundaries.

# Evidence And Caveat

The evidence document says a layer-2 adapter and orchestrator compatibility adapter were implemented, and that field adapters were identified as a code smell. It also says execution issues remained because of orchestrator/tool request mismatches and ProvenanceService API mismatch. [4]

# Citations

[1] `../archive_full_record/lineage_variants/digimon_core_sparse/CLAUDE.md`  
[2] `../archive_full_record/lineage_variants/digimon_core_sparse/contracts/README.md`  
[3] `../archive_full_record/lineage_variants/digimon_core_sparse/Contract_First_Uncertainty_Analysis.md`  
[4] `../archive_full_record/lineage_variants/digimon_core_sparse/Evidence_Contract_First_Implementation.md`
