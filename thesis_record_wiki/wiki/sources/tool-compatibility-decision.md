---
type: SourceSummary
title: Tool Compatibility Decision
description: Source summary for type-based tool composition decision and POC documents.
tags: [source, tool-compatibility, type-based-composition, poc]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/Digimons_clean_for_real/tool_compatability/README.md
  - ../archive_full_record/lineage_variants/Digimons_clean_for_real/tool_compatability/DECISION_DOCUMENT.md
  - ../archive_full_record/lineage_variants/Digimons_clean_for_real/tool_compatability/poc/README.md
confidence: high
---

# Summary

The tool-compatibility documents in `Digimons_clean_for_real` frame a later core engineering problem: KGAS had many tools, but their interfaces were incompatible and wrongly factored. The decision document says there were 38 tools with incompatible interfaces, 75 implementation files with duplicates/versions, and multiple failed compatibility approaches. [1]

The recommended solution was type-based composition with direct data passing: roughly ten semantic data types, exact Pydantic schemas per type, and compatibility determined by matching declared input/output semantic types. [1]

The POC README explains the design in concrete terms: semantic data types, exact schemas, direct data passing except graph references in Neo4j, automatic chain discovery, memory/failure/schema/performance tests, and a migration path from 38 tools toward a smaller set of properly bounded operators. [2]

# Failed Approaches Recorded

- Unified god-object data contract.
- Generic Python type matching.
- Pipeline accumulation.
- Field-name matching.
- More complex ORM/semantic role matching.

# Key Takeaways

- This is a major thesis-evolution thread: the problem shifted from "many tools exist" to "tools must compose under truthful contracts."
- The chosen strategy matches later ecosystem preferences for typed boundaries and Pydantic contracts.
- The decision was still contingent on POC evidence; do not record it as fully implemented unless later evidence proves it.

# Pages Informed

- [Type-Based Tool Composition](/wiki/concepts/type-based-tool-composition.md)
- [Digimons Clean For Real](/wiki/variants/digimons-clean-for-real.md)
- [Research Lineage](/wiki/concepts/research-lineage.md)

# Citations

[1] `../archive_full_record/lineage_variants/Digimons_clean_for_real/tool_compatability/DECISION_DOCUMENT.md`  
[2] `../archive_full_record/lineage_variants/Digimons_clean_for_real/tool_compatability/poc/README.md`  
[3] `../archive_full_record/lineage_variants/Digimons_clean_for_real/tool_compatability/README.md`
