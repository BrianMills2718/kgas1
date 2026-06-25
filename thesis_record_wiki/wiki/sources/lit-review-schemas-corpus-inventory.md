---
type: Source
title: Lit Review Schemas Corpus Inventory
description: Inventory of the preserved mature schemas directory, including Young, semantic hypergraph, Turner/social identity, ELM, and extracted-theory artifacts.
tags: [source, lit-review, schemas, inventory, semantic-hypergraph, young-1996, social-identity]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/elaboration_likelihood_model/elm_v10_summary.txt
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/young1996/execution_prompt_example.txt
confidence: high
---

# Summary

The preserved `schemas/` directory is a later, more mature schema corpus than the broad `archive/old_schemas/` directory. It contains 54 files with this file-type mix: 19 `.yml`, 8 `.yaml`, 20 `.json`, and 7 `.txt`. The directory-level aggregate hash over sorted relative filenames and per-file hashes is `339014c4c3391432dbdbd13b0d9189ebeb7885201c383679c754468e261e32d2`. [1]

This is only an inventory slice. The directory is large enough that semantic subfolder deep dives should be separate commits.

# Subfolder Inventory

| Subfolder | Files | Size | Notes |
|---|---:|---:|---|
| `semantic_hypergraph` | 19 | 312K | schema variants plus debug/debug_improved JSON outputs |
| `extracted` | 18 | 420K | Turner/social-identity fulltext and multi-theory extraction outputs |
| `young1996` | 6 | 72K | Young 1996 schema variants and execution prompt example |
| `extracted_single` | 5 | 120K | Turner single-theory extraction outputs |
| `social_identity_theory` | 3 | 100K | Turner extracted text and SIT schema JSON/YAML |
| `elaboration_likelihood_model` | 2 | 16K | ELM schema and summary |
| root file | 1 | 4K | `test_simple_schema.yml` |

# Root Test Fixture

The root `test_simple_schema.yml` is a synthetic/simple hypergraph fixture: citation `Test Author (2024), "Simple Hypergraph Theory"`, model type `property_graph`, and a generic schema for hypergraphs, vertices, hyperedges, incidence, adjacency, arity, uniformity, order, size, degree, and related measures/actions. Treat it as a pipeline/test artifact rather than a substantive thesis literature source. [4]

# Notable Evidence

The ELM summary presents a compact theory schema with classification, entities, relationships, and four analysis steps. It classifies ELM as Micro level, Effect component, and Interdependent metatheory, with entities for receiver, message, route, and attitude-change outcome. [2]

The Young 1996 execution prompt is not just a schema. It is an operational prompt for running computational cognitive-mapping analysis: use relationship categories, truth values, modifiers, concept extraction, relationship extraction, salience calculation, structural measures, and output JSON over a Carter speech. [3]

# Interpretation

`schemas/` appears to be the bridge between theory schemas and executable application. Compared with `archive/old_schemas/`, it includes:

- variant families for the same theory
- JSON debug/intermediate outputs
- execution prompts
- extracted fulltext sidecars
- single-theory and multi-theory extraction paths

This supports [Schema Extraction Pipeline Evolution](/wiki/concepts/schema-extraction-pipeline-evolution.md): the work moved from broad schema generation toward staged extraction, comparison, execution prompts, and application-ready artifacts.

# Links

- [Schema Extraction Pipeline Evolution](/wiki/concepts/schema-extraction-pipeline-evolution.md)
- [Lit Review Old Schema Archive Inventory](/wiki/sources/lit-review-old-schema-archive-inventory.md)
- [Lit Review Semantic Hypergraph Application Results](/wiki/sources/lit-review-semantic-hypergraph-application-results.md)
- [Lit Review Universal Theory Applicator](/wiki/sources/lit-review-universal-theory-applicator.md)
- [Schema Variant Drift](/wiki/concepts/schema-variant-drift.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/elaboration_likelihood_model/elm_v10_summary.txt`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/young1996/execution_prompt_example.txt`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/test_simple_schema.yml`
