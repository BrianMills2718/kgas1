---
type: Source
title: Lit Review Semantic Hypergraph Schema Family
description: Inventory of the preserved semantic-hypergraph schema variants and debug outputs in the mature schemas corpus.
tags: [source, lit-review, semantic-hypergraph, schema-variants, notation, inventory]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/semantic_hypergraph/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/semantic_hypergraph/semantic_hypergraph_enhanced.yml
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/semantic_hypergraph/sh_multi_pass_complete.yml
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/semantic_hypergraph/sh_option1_expanded.yml
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/semantic_hypergraph/sh_option2_implementation.yml
confidence: high
---

# Summary

The preserved `schemas/semantic_hypergraph/` folder contains 19 files: 12 top-level YAML schema variants, 1 top-level JSON phase output, 3 `debug/` JSON outputs, and 3 `debug_improved/` JSON outputs. Its aggregate hash over sorted relative filenames and per-file hashes is `2684a60c11000d9f7ac9b982febdbf1bbb7557010af28dbdabdf88e294df8bb2`. [1]

This page complements [Lit Review Semantic Hypergraph Application Results](/wiki/sources/lit-review-semantic-hypergraph-application-results.md), which focuses on critique and application artifacts. This page records the schema-variant family itself.

# Variant Family

The folder preserves multiple schema forms:

- `semantic_hypergraph_complete.yml`, `semantic_hypergraph_fixed.yml`, and `semantic_hypergraph_schema.yml`: hypergraph schema-blueprint variants. [1]
- `semantic_hypergraph_enhanced.yml`: hypergraph schema with notation system, pattern examples, 8 type-inference rules, and implementation algorithms. [2]
- `sh_enhanced.yml`: application-stage schema with output mapping for atoms, hyperedges, discourse structures, and analysis. [1]
- `sh_multi_pass_complete.yml`: multi-pass extraction artifact with notation, formal rules, algorithms, implementation, evaluation, and examples. [3]
- `sh_option1_expanded.yml`: expanded integrated schema with type codes, role codes, special symbols, composite notations, pattern library, algorithms, and domain applications. [4]
- `sh_option2_implementation.yml`: implementation-specific extraction organized under `implementation_specification`. [5]
- `sh_simplified.yml`, `sh_simplified_v2.yml`, `sh_theory_only.yml`, and `sh_properly_mapped.yml`: simplified/theory-only/mapped variants. [1]

# Interpretation

This is a concentrated example of schema-family evolution. The same formal theory was represented as:

- integrated schema
- implementation-only extraction
- enhanced notation-aware schema
- multi-pass extraction record
- application-stage schema
- simplified and theory-only variants

That matches the critique chain already captured in the application-results page: formal notation was repeatedly lost unless the extraction path explicitly represented type codes, role codes, pattern syntax, and algorithms as theory content.

# Links

- [Lit Review Semantic Hypergraph Application Results](/wiki/sources/lit-review-semantic-hypergraph-application-results.md)
- [Formal Notation As Theory Content](/wiki/concepts/formal-notation-as-theory-content.md)
- [Schema Variant Drift](/wiki/concepts/schema-variant-drift.md)
- [Schema Extraction Pipeline Evolution](/wiki/concepts/schema-extraction-pipeline-evolution.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/semantic_hypergraph/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/semantic_hypergraph/semantic_hypergraph_enhanced.yml`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/semantic_hypergraph/sh_multi_pass_complete.yml`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/semantic_hypergraph/sh_option1_expanded.yml`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/semantic_hypergraph/sh_option2_implementation.yml`
