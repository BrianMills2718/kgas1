---
type: Concept
title: MCP Autoloop Interface
description: Historical and later MCP interface layers in the digimon_autoloop lineage variant.
tags: [concept, mcp, autoloop, interface]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_autoloop/MCP_IMPLEMENTATION_TRACKER.md
  - ../archive_full_record/lineage_variants/digimon_autoloop/MCP_INTEGRATION_DETAILED_PLAN.md
  - ../archive_full_record/lineage_variants/digimon_autoloop/docs/UKRF_MCP_INTERFACE_SPEC.md
  - ../archive_full_record/lineage_variants/digimon_autoloop/CLAUDE.md
  - ../archive_full_record/lineage_variants/digimon_autoloop/docs/ACTIVE_DOCS.md
confidence: medium
---

# Summary

`digimon_autoloop` contains two MCP layers that should not be conflated:

- Early 2025-06 MCP implementation planning: foundation server, client manager, shared context, tool migration, multi-agent coordination, production hardening. [1][2]
- Later DIGIMON tool surface: MCP/direct access to a mature operators-first benchmark and retrieval system. [4]

# Early MCP Tracker

The 2025-06-06 implementation tracker records only checkpoint 1.1 as passed and shows overall MCP progress at 8%. Tool migration, multi-agent coordination, and production phases are all marked not started in that tracker. [1]

The detailed plan is still useful as design lineage: it specifies checkpoints, tests, evidence templates, performance targets, and expected files/classes for an MCP foundation. [2]

# UKRF Integration Spec

The UKRF MCP spec defines historical interfaces for StructGPT and Autocoder integration: SQL generation, table QA, entity extraction, capability generation, and tool validation. It also defines a shared context with session IDs, entity registry, schema mappings, and execution history. [3]

However, the later active-doc index explicitly says UKRF/master-orchestrator roadmaps are historical unless actively rewritten to match the current repo. [5]

# Later Status Boundary

By the later operators-first DIGIMON docs, MCP is one supported interface among others, alongside direct Python benchmark mode. The wiki should preserve the early MCP/UKRF plans as important lineage while treating the later README/CLAUDE/FUNCTIONALITY set as the more current status source for this variant. [4][5]

# Citations

[1] `../archive_full_record/lineage_variants/digimon_autoloop/MCP_IMPLEMENTATION_TRACKER.md`  
[2] `../archive_full_record/lineage_variants/digimon_autoloop/MCP_INTEGRATION_DETAILED_PLAN.md`  
[3] `../archive_full_record/lineage_variants/digimon_autoloop/docs/UKRF_MCP_INTERFACE_SPEC.md`  
[4] `../archive_full_record/lineage_variants/digimon_autoloop/CLAUDE.md`  
[5] `../archive_full_record/lineage_variants/digimon_autoloop/docs/ACTIVE_DOCS.md`
