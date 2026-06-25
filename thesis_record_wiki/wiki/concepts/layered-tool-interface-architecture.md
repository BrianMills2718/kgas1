---
type: Concept
title: Layered Tool Interface Architecture
description: KGAS reconciled contract-first tools, pipeline adapters, and MCP exposure as three layers rather than competing tool-interface designs.
tags: [concept, tools, orchestration, mcp, contracts, structured-output]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-001-Phase-Interface-Design.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-002-Pipeline-Orchestrator-Architecture.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-013-MCP-Protocol-Integration.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-017-Structured-Output-Migration.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-028-Tool-Interface-Layer-Architecture.md
confidence: medium
---

# Summary

Layered tool interface architecture is the KGAS resolution to a recurring integration failure: tool logic, internal orchestration, and external agent access need different interfaces, but they must not drift independently.

The accepted layer model is:

- **Layer 1**: raw implementation and legacy adapters
- **Layer 2**: internal KGASTool contract with theory, confidence, metadata, and provenance
- **Layer 3**: MCP JSON-compatible external access for AI and other clients [5]

# Why It Matters

This concept explains several later KGAS/Digimons threads:

- [Contract-First Migration](/wiki/concepts/contract-first-migration.md): legacy tools needed a common internal contract, not just more adapters.
- [Type-Based Tool Composition](/wiki/concepts/type-based-tool-composition.md): composability depends on precise input/output surfaces.
- [MCP Autoloop Interface](/wiki/concepts/mcp-autoloop-interface.md): MCP is best read as an external agent-facing surface, not the whole tool architecture.
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md): a tool working in Layer 1 does not prove it is exposed correctly through Layer 2 or Layer 3.

# Structured Output Role

Structured output is part of this architecture because tool orchestration often depends on LLM-produced decisions and data. ADR-017 records the migration from manual JSON parsing toward Pydantic schemas, centralized LLM service, monitoring, and fail-fast validation. [4]

The historical implementation detail matters: ADR-017 used `json_object`, while newer ecosystem policy prefers `json_schema`. The durable idea is schema-first validated output, not the exact 2025 provider mode.

# Status Caveat

The layer architecture is a target and migration framework. ADR-001 and ADR-002 include implementation-status claims, but those claims are time-indexed and should not be reused without checking current roadmap/code evidence.

# Links

- [Digimon Lineage Tool Orchestration ADRs](/wiki/sources/digimon-lineage-tool-orchestration-adrs.md)
- [Digimon Lineage Tool Compatibility](/wiki/sources/digimon-lineage-tool-compatibility.md)
- [Graph Build Manifest](/wiki/concepts/graph-build-manifest.md)
- [Adaptive Operator Routing](/wiki/concepts/adaptive-operator-routing.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-001-Phase-Interface-Design.md`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-002-Pipeline-Orchestrator-Architecture.md`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-013-MCP-Protocol-Integration.md`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-017-Structured-Output-Migration.md`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-028-Tool-Interface-Layer-Architecture.md`
