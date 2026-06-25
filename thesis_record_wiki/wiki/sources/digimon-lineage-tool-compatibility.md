---
type: Source
title: Digimon Lineage Tool Compatibility
description: Source summary for the large Digimons lineage bundle's tool_compatability subtree and its active vertical-slice context.
tags: [source, tool-compatibility, vertical-slice, type-based-composition]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tool_compatability/README.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tool_compatability/CLAUDE.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tool_compatability/DECISION_DOCUMENT.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tool_compatability/METHODICAL_IMPLEMENTATION_PLAN.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tool_compatability/PROOF_OF_CONCEPT_PLAN.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tool_compatability/poc/README.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tool_compatability/poc/vertical_slice/CLAUDE.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tool_compatability/tool_disposition_plan.md
confidence: high
---

# Summary

The `tool_compatability` subtree in `digimon_lineage_Digimons` is the large-bundle copy of the type-based tool composition decision lineage, with additional active-context markers. It has 267 files and is small enough to ingest in two or three passes. [1][2]

The root `CLAUDE.md` says this subtree is the adapter/framework direction for modular tool chaining and should not be treated as broad repo governance. The nested vertical-slice `CLAUDE.md` says `poc/vertical_slice` is the active framework path and should be preferred over the broader legacy codebase for working proof-of-concept chain tasks. [2][7]

# Decision Thread

The decision document is the same content previously found in `Digimons_clean_for_real/tool_compatability`: after five failed approaches, pursue type-based composition with direct data passing, exact Pydantic schemas per semantic type, and compatibility by input/output type matching. [3]

This large-bundle slice adds stronger local context:

- The README lists the type-based approach as the current focus. [1]
- The methodical plan expands the POC into multi-input support, schema versioning, memory management, semantic compatibility, and transaction/rollback work. [4]
- The POC plan names critical unknowns: memory boundaries, schema evolution, error recovery, performance overhead, and branching. [5]
- The tool disposition plan maps 38 tools toward roughly 15 operators. [8]

# Tool Consolidation

The disposition plan is historically important because it shows a change from counting tools to designing operator boundaries:

- 14 loaders -> `T01_UniversalLoader`
- entity extraction, relationship extraction, and graph persistence -> `T23_GraphExtractor`
- graph analytics kept mostly separate where each operator preserves a distinct analytical capability
- social/external data generalized into `T85_SocialExplorer` [8]

# Interpretation

This slice links [Type-Based Tool Composition](/wiki/concepts/type-based-tool-composition.md) with [Vertical Slice vs Main System](/wiki/concepts/vertical-slice-vs-main-system.md). The type-based decision was not only a theoretical compatibility idea; in the large lineage bundle it became the named active path for a working vertical-slice framework.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tool_compatability/README.md`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tool_compatability/CLAUDE.md`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tool_compatability/DECISION_DOCUMENT.md`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tool_compatability/METHODICAL_IMPLEMENTATION_PLAN.md`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tool_compatability/PROOF_OF_CONCEPT_PLAN.md`  
[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tool_compatability/poc/README.md`  
[7] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tool_compatability/poc/vertical_slice/CLAUDE.md`  
[8] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tool_compatability/tool_disposition_plan.md`
