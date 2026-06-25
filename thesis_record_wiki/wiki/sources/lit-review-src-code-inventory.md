---
type: Source
title: Lit Review Src Code Inventory
description: Directory-level inventory of preserved lit-review source code for schema creation, application, testing, UI, and visualization.
tags: [source, lit-review, code, inventory, schema-creation, schema-application]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/
confidence: high
---

# Summary

The preserved `src/` directory contains 90 files across five top-level code areas. Its aggregate hash over sorted relative filenames and per-file hashes is `7b2f77f34a2caff0b4a69e212736893516ceb81a2596a77801339bca020ba2c0`. [1]

This is a directory inventory, not a code review or runtime verification pass.

# Subdirectory Inventory

| Subdirectory | Files | Bytes | Role |
|---|---:|---:|---|
| `schema_creation` | 40 | 354,569 | extraction processors, prompt loading, multiphase/multipass variants, implementation extraction, and schema-generation tests |
| `schema_application` | 24 | 219,986 | applying generated schemas/theories to Carter, Young/WorldView, Semantic Hypergraph, and universal applicator experiments |
| `testing` | 14 | 44,015 | debug scripts, processor comparison, O3 checks, semantic-hypergraph tests, and execution traces |
| `visualization` | 9 | 82,444 | cognitive-map and semantic-hypergraph visualization scripts |
| `ui` | 3 | 44,256 | Streamlit/YAML analysis UI scripts |

# Interpretation

The code inventory shows that the lit-review experiment had moved beyond schema generation into application and inspection surfaces:

- schema creation code produced and improved theory schemas
- schema application code attempted to execute theories over texts
- testing code compared processors and checked model/provider behavior
- visualization code made cognitive maps and hypergraph instances inspectable
- UI code offered a manual review surface

This supports the broader pattern that the thesis work evolved from "extract schema from paper" into a full loop: extract, apply, test, visualize, inspect, and revise.

# Links

- [Schema Extraction Pipeline Evolution](/wiki/concepts/schema-extraction-pipeline-evolution.md)
- [Lit Review Schema Creation Production Path](/wiki/sources/lit-review-schema-creation-production-path.md)
- [Lit Review Universal Theory Applicator](/wiki/sources/lit-review-universal-theory-applicator.md)
- [Lit Review Semantic Hypergraph Application Results](/wiki/sources/lit-review-semantic-hypergraph-application-results.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/`
