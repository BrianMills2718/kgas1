---
type: Concept
title: Vertical Slice vs Main System
description: Coexisting simple vertical-slice architecture and complex main-system architecture in the large Digimons lineage bundle.
tags: [concept, architecture, vertical-slice, main-system]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/CLAUDE.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/operations/vertical-slice-system-relationship-investigation.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/operations/vertical-slice-functionality-verification.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/operations/claude-md-claims-verification.md
confidence: medium
---

# Summary

`digimon_lineage_Digimons` preserves a key architectural split: a simple working vertical slice and a larger, more complex main system. The split is a major reason status claims became hard to interpret. [1][2]

# Vertical Slice

The vertical slice lives under `tool_compatability/poc/vertical_slice/` and uses direct embedded services plus simple tool adapters. The verification docs report:

- `VectorTool` and `TableTool` registered successfully.
- The framework discovered a `TEXT -> VECTOR -> TABLE` chain.
- Chain execution stored 1536-dimensional embeddings in SQLite.
- The database included tables for embeddings, data, provenance, and relationships.
- Uncertainty existed structurally but was hardcoded to `0.0`. [3][4]

This makes the vertical slice a real proof-of-concept, not just two disconnected tools.

# Main System

The main system under `/src/` used a more complex service manager, contracts, tool requests, and dependency injection. The vertical-slice/system investigation says this architecture failed in at least one observed path because Neo4j configuration/authentication errors cascaded through service creation and tool initialization. [2]

The later roadmap consolidation investigation also says the main system contained substantial functionality that the narrow vertical-slice investigation initially missed, including WorkflowAgent, monitoring, dashboard infrastructure, and a larger tool registry. [1]

# Interpretation

The correct historical reading is not "vertical slice good, main system worthless." It is:

- The vertical slice provided a simpler, verified path for thesis-critical behavior.
- The main system contained substantial implementation work but was harder to verify because of dependency/configuration complexity.
- Later roadmap consolidation tried to reconcile those two views.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/CLAUDE.md`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/operations/vertical-slice-system-relationship-investigation.md`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/operations/vertical-slice-functionality-verification.md`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/operations/claude-md-claims-verification.md`
