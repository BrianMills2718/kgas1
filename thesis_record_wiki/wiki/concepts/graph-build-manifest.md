---
type: Concept
title: Graph Build Manifest
description: DIGIMON design requirement that graph builds declare topology, attributes, artifacts, provenance, and enrichments before tools are exposed.
tags: [concept, graph, manifest, tool-gating]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_autoloop/docs/GRAPH_ATTRIBUTE_MODEL.md
  - ../archive_full_record/lineage_variants/digimon_autoloop/docs/TOOL_CAPABILITY_MATRIX.md
confidence: medium
---

# Summary

The graph build manifest is a later DIGIMON design response to a recurring truthfulness problem: tools should only be exposed when the current graph build actually has the required topology, attributes, and derived artifacts. [1][2]

# Why It Matters

The attribute model maps JayLZhou GraphRAG graph families into DIGIMON:

- chunk tree -> tree graph / balanced tree graph
- passage graph -> passage graph
- KG/TKG/RKG -> entity-graph attribute profiles rather than totally separate systems [1]

The same document says KG, TKG, and RKG should be projections from one maximal entity-graph build, while tree and passage graphs need separate topology builds. [1]

# Tool Gating

The tool capability matrix makes the rule strict: if a build lacks the needed topology, attributes, or artifacts, the corresponding tool should not be exposed. Examples:

- `entity.ppr` needs an entity graph and sparse matrices.
- `relationship.vdb` needs useful relationship text plus a relationship VDB.
- `community.from_entity` needs community reports.
- `chunk.text_search` is corpus-backed and not graph-specific. [2]

This connects directly to benchmark truthfulness: an adaptive agent cannot fairly choose tools if the tool list overstates what the current build can do.

# Relationship To Earlier Variants

The manifest design is a later, more specific version of the documentation-status and contract-first concerns visible in earlier variants. It turns "do not overstate capability" into a concrete runtime/tool-exposure contract.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_autoloop/docs/GRAPH_ATTRIBUTE_MODEL.md`  
[2] `../archive_full_record/lineage_variants/digimon_autoloop/docs/TOOL_CAPABILITY_MATRIX.md`
