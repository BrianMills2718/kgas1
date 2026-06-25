---
type: Concept
title: GraphRAG Upstream Lineage
description: Relationship between Brian's Digimon/KGAS work and JayLZhou GraphRAG / DIGIMON.
tags: [graphrag, upstream, digimon, lineage]
created: 2026-06-25
updated: 2026-06-25
sources:
  - https://github.com/JayLZhou/GraphRAG
confidence: medium
---

# Summary

Brian flagged that much of the local Digimon work is an extension or fork of JayLZhou/GraphRAG. This wiki should therefore treat upstream GraphRAG/DIGIMON as part of the historical record. The upstream README describes DIGIMON as a deep analysis of GraphRAG systems focused on modularizing and decoupling graph-based RAG methods. [1]

For the source summary, see [JayLZhou GraphRAG Upstream](/wiki/sources/jaylzhou-graphrag-upstream.md).

# Upstream Concepts To Preserve

- Multiple GraphRAG methods runnable through method-specific YAML configs.
- Graph type taxonomy: chunk tree, passage graph, KG, textual KG, rich KG.
- Retrieval decomposition into entity, relationship, chunk, subgraph, and community operators.
- The idea that methods can be understood as combinations of retrieval operators.

# Local Extension Threads

Local KGAS/Digimon work appears to extend this upstream frame in several directions:

- [Documentation Status Truthfulness](/wiki/concepts/documentation-status-truthfulness.md)
- [Type-Based Tool Composition](/wiki/concepts/type-based-tool-composition.md)
- [Contract-First Migration](/wiki/concepts/contract-first-migration.md)
- Theory operationalization and computational social science framing in [KGAS](/wiki/entities/kgas.md)

# Open Questions

- Which local directories are direct forks of upstream GraphRAG versus inspired rewrites?
- Which upstream commit, if any, was the fork point for each preserved local variant?
- Which local methods correspond directly to upstream methods such as RAPTOR, LightRAG, HippoRAG, ToG, or GraphRAG?

# Citations

[1] `https://github.com/JayLZhou/GraphRAG` retrieved 2026-06-25.
