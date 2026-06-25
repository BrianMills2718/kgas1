---
type: Concept
title: Relationship Extraction Bottleneck
description: Stress-test evidence that KGAS could extract entities while failing to extract relationships.
tags: [bottleneck, relationship-extraction, graph-construction, evidence]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_core_sparse/ARCHITECTURAL_BOTTLENECKS_ANALYSIS.md
confidence: high
---

# Summary

The relationship extraction bottleneck is one of the clearest technical warnings in the preserved record. The architectural bottleneck analysis says a multi-document stress test processed 25 documents and extracted 398 entities, but found zero relationships. [1]

# Why It Matters

For a GraphRAG system, entity extraction alone is insufficient. If relationship extraction is missing, failing silently, or not invoked, graph construction becomes entity-only and the system cannot deliver the intended graph-based reasoning layer.

# Recorded Root Cause Hypothesis

The bottleneck analysis says relationship extraction tool T27 was either not called in the multi-document pipeline or failed silently. It recommends auditing T27 invocation, adding relationship extraction to multi-document tests, and validating with a simple two-document case. [1]

# Related Pages

- [Digimon Core Sparse Contract Layer](/wiki/sources/digimon-core-sparse-contract-layer.md)
- [Contract-First Migration](/wiki/concepts/contract-first-migration.md)
- [KGAS](/wiki/entities/kgas.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_core_sparse/ARCHITECTURAL_BOTTLENECKS_ANALYSIS.md`
