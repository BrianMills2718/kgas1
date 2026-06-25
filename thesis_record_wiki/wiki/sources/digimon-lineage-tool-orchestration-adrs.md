---
type: Source
title: Digimon Lineage Tool Orchestration ADRs
description: ADR slice covering KGAS contract-first tool interfaces, pipeline orchestration, MCP exposure, structured output, and three-layer tool interface architecture.
tags: [source, architecture, adr, tools, orchestration, mcp, structured-output]
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

This ADR slice explains how KGAS tried to make tools agent-drivable and composable:

1. ADR-001 defines a contract-first internal tool interface with `ToolRequest`, `ToolResult`, theory schema integration, confidence, metadata, and provenance. [1]
2. ADR-002 introduces a `PipelineOrchestrator` and adapter pattern to reduce duplicated phase workflows and bridge legacy tools. [2]
3. ADR-013 accepts MCP as the external API layer for AI and external research-tool access. [3]
4. ADR-017 accepts schema-first structured LLM output using Pydantic validation and monitoring. [4]
5. ADR-028 reconciles the apparent conflict between contract-first tools, adapters, and MCP by defining them as three complementary layers. [5]

# Three-Layer Interpretation

ADR-028 is the key interpretive document. It says the earlier tool-interface ADRs are not competing designs:

- **Layer 1: Tool implementation**: raw tool logic and legacy adapters, represented by ADR-002.
- **Layer 2: Internal contract**: KGAS internal services and agent orchestration, represented by ADR-001.
- **Layer 3: External API access**: MCP exposure for AI and external clients, represented by ADR-013. [5]

The intended flow is Layer 3 wraps Layer 2, and Layer 2 wraps or implements Layer 1. No direct Layer 1 to Layer 3 communication is supposed to bypass the internal contract layer. [5]

# Orchestration Problem

ADR-002 says the inherited GraphRAG/KGAS workflows had severe duplication, inconsistent error handling, import path hacks, print-statement logging, and no unified workflow/tool interface. The chosen response was a unified `PipelineOrchestrator`, configurable pipeline factory, and adapters for existing tools. [2]

This makes the tool-interface thread part of the same lineage as [Contract-First Migration](/wiki/concepts/contract-first-migration.md): the central problem was not simply tool count, but incompatible surfaces discovered too late at integration time.

# MCP Boundary

ADR-013 treats MCP as external access, not the internal execution substrate. MCP tools call Layer 2 KGASTool contracts internally and expose JSON-compatible functions for Claude or other MCP clients. This helps explain the later distinction between direct internal interfaces and agent-facing tool surfaces. [3]

# Structured Output

ADR-017 records a major LLM reliability migration away from manual JSON parsing toward centralized structured output with Pydantic schemas, monitoring, and schema-first validation. It reports historical improvements from roughly 80% manual parsing reliability to above 95% structured-operation reliability. [4]

Important caveat: ADR-017's historical implementation uses LiteLLM `response_format={"type": "json_object"}`. In the broader current ecosystem policy, `json_schema` is the preferred standard. Treat ADR-017 as evidence of a 2025 reliability migration, not as current best-practice guidance for new work.

# Status Caveat

ADR-001 explicitly says the contract-first design was partially implemented at the time of that document: 10 tools still used legacy interfaces and 9 tools had the unified interface. The ADR describes target architecture and migration direction; current implementation status must be checked in roadmap/evidence/code before making completion claims. [1]

# Links

- [Layered Tool Interface Architecture](/wiki/concepts/layered-tool-interface-architecture.md)
- [Contract-First Migration](/wiki/concepts/contract-first-migration.md)
- [Type-Based Tool Composition](/wiki/concepts/type-based-tool-composition.md)
- [MCP Autoloop Interface](/wiki/concepts/mcp-autoloop-interface.md)
- [Digimon Lineage Architecture ADRs Map](/wiki/sources/digimon-lineage-architecture-adrs-map.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-001-Phase-Interface-Design.md`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-002-Pipeline-Orchestrator-Architecture.md`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-013-MCP-Protocol-Integration.md`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-017-Structured-Output-Migration.md`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-028-Tool-Interface-Layer-Architecture.md`
