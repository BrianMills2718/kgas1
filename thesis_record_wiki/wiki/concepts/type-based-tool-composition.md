---
type: Concept
title: Type-Based Tool Composition
description: Compatibility strategy using semantic data types and exact schemas to compose KGAS tools.
tags: [tools, composition, pydantic, contracts]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/Digimons_clean_for_real/tool_compatability/README.md
  - ../archive_full_record/lineage_variants/Digimons_clean_for_real/tool_compatability/DECISION_DOCUMENT.md
  - ../archive_full_record/lineage_variants/Digimons_clean_for_real/tool_compatability/poc/README.md
confidence: high
---

# Summary

Type-based tool composition was a proposed solution to KGAS tool incompatibility. The core idea was to define semantic data types such as `TEXT`, `ENTITIES`, and `GRAPH`, give each type exactly one Pydantic schema, and compose tools by matching declared output and input types. [1]

# Problem It Addressed

The decision document says KGAS had 38 incompatible tools, 75 implementation files with duplicates/versions, wrongly factored boundaries, and five failed compatibility approaches. [1]

The same decision lineage also appears inside the large `digimon_lineage_Digimons` bundle. That copy adds active-context markers: subtree instructions identify `tool_compatability` as the adapter/framework direction, and `poc/vertical_slice` as the active working proof-of-concept path. See [Digimon Lineage Tool Compatibility](/wiki/sources/digimon-lineage-tool-compatibility.md).

# Design Principles

- Use semantic data types rather than generic Python types.
- Use exact schemas per type.
- Pass data directly for small/medium payloads.
- Keep graph data in Neo4j when references are more appropriate.
- Discover chains automatically by matching output type to input type.
- Validate the approach through a POC with memory, failure recovery, schema evolution, and performance tests. [2]

# Historical Status

This should be treated as a decision-and-POC direction, not automatically as a completed migration. The README shows POC development partially checked off, with registry implementation, test tools, edge cases, benchmarking, and go/no-go still listed as pending. [3]

The large-bundle methodical plan expands the POC into a more ambitious PhD tool-composition system: multi-input architecture, schema versioning, memory management for large documents, semantic compatibility, state management/rollback, composition-agent selection, learning, and benchmark validation. [4]

# Tool Consolidation Direction

The large-bundle tool disposition plan shows the intended operator-boundary cleanup: merge 14 file loaders into a universal loader, merge entity/relationship/node/edge steps into a graph extractor, and keep distinct graph analytics as separate operators when they preserve different analytical capabilities. [5]

# Related Pages

- [Tool Compatibility Decision](/wiki/sources/tool-compatibility-decision.md)
- [Digimon Lineage Tool Compatibility](/wiki/sources/digimon-lineage-tool-compatibility.md)
- [Digimons Clean For Real](/wiki/variants/digimons-clean-for-real.md)
- [Vertical Slice vs Main System](/wiki/concepts/vertical-slice-vs-main-system.md)
- [KGAS](/wiki/entities/kgas.md)

# Citations

[1] `../archive_full_record/lineage_variants/Digimons_clean_for_real/tool_compatability/DECISION_DOCUMENT.md`  
[2] `../archive_full_record/lineage_variants/Digimons_clean_for_real/tool_compatability/poc/README.md`  
[3] `../archive_full_record/lineage_variants/Digimons_clean_for_real/tool_compatability/README.md`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tool_compatability/METHODICAL_IMPLEMENTATION_PLAN.md`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tool_compatability/tool_disposition_plan.md`
