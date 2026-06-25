---
type: Source
title: Lit Review Debug Improved Results
description: Inventory and interpretation of improved three-phase debug outputs plus the empty analysis_results directory.
tags: [source, lit-review, debug, phase-output, semantic-hypergraph, carter, empty-directory]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/debug_improved/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/analysis_results/
confidence: high
---

# Summary

The preserved `debug_improved/` directory contains 9 JSON files with aggregate hash `e56d52dde9c4f9c0ea7dd84af358924176cb23427c66a77a69e03291003caab9`. The adjacent `analysis_results/` directory exists but contains no files. [1][2]

The debug files preserve three-phase intermediate outputs for:

- Carter minimal test
- Menezes/Roth Semantic Hypergraphs
- Semantic Hypergraphs duplicate/alternate run [1]

# Phase Output Inventory

| Case | Phase 1 vocabulary | Phase 2 counts | Phase 3 model |
|---|---:|---|---|
| Carter minimal test | 30 terms | 12 entities, 5 relationships, 5 actions, 3 measures, 0 operators | `property_graph`, 9 node types, 11 edge types, 12 property definitions |
| Menezes/Roth Semantic Hypergraphs | 61 terms | 4 entities, 5 relationships, 7 actions, 3 measures, 1 operator | `hypergraph`, 11 node types, 7 edge types, 22 property definitions |
| Semantic Hypergraphs alternate run | 57 terms | 35 entities, 2 relationships, 12 actions, 2 measures, 4 operators | `hypergraph`, 55 node types, 14 edge types, 18 property definitions |

# Interpretation

This directory is useful because it preserves intermediate products, not only final schemas. It shows how the three-phase process changed representation:

- Phase 1: vocabulary and source-theory characterization
- Phase 2: typed classification into entities, relationships, properties, actions, measures, modifiers, truth values, and operators
- Phase 3: model-type selection and schema structure

The two Semantic Hypergraph runs both route to `hypergraph`, but their Phase 2 and Phase 3 counts differ substantially. That supports [Schema Variant Drift](/wiki/concepts/schema-variant-drift.md): reruns or related inputs can preserve different slices of the same theory even when the final model type agrees.

# Empty Directory Note

`analysis_results/` exists but has no files. This is negative evidence for the root README's broader claim that generated analysis outputs live there. Some analysis results are preserved elsewhere, especially root-level `grusch_analysis_results.json`, `results/`, `carter_analysis_output/`, and `evidence/`. [2]

# Links

- [Lit Review Data And Examples](/wiki/sources/lit-review-data-examples.md)
- [Lit Review Schema Creation Production Path](/wiki/sources/lit-review-schema-creation-production-path.md)
- [Lit Review Semantic Hypergraph Schema Family](/wiki/sources/lit-review-semantic-hypergraph-schema-family.md)
- [Schema Extraction Pipeline Evolution](/wiki/concepts/schema-extraction-pipeline-evolution.md)
- [Schema Variant Drift](/wiki/concepts/schema-variant-drift.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/debug_improved/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/analysis_results/`
