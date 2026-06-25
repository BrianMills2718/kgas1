---
type: Source
title: Lit Review Old Schema Archive Inventory
description: Inventory of 56 archived raw/generated theory schemas across influence, persuasion, behavior change, information disorder, argumentation, worldviews, and related domains.
tags: [source, lit-review, schemas, archive, inventory]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/archive/old_schemas/
confidence: high
---

# Summary

The `archive/old_schemas/` directory preserves 56 YAML schemas from an earlier lit-review schema-generation phase.

This slice inventories the directory rather than synthesizing every schema. It is useful because it shows the breadth of theory domains attempted before later pipeline refinements: persuasion, information disorder, influence operations, behavioral forecasting, intervention mapping, reasoned action, radicalization, social marketing, worldviews, operational code/cognitive mapping, argumentation, cultural evolution, vaccine hesitancy, and more.

# Directory Shape

Observed facts:

- 56 files
- all files have `.yml` extension
- 55 parse as YAML dictionaries
- 1 parse error: `foundations_individuals_raw.yml`
- common top-level fields across parseable files: `citation`, `annotation`, `model_type`, `rationale`, `schema_blueprint`
- 11 parseable files also include `json_schema`

Aggregate manifest hash over sorted filenames plus per-file SHA-256 hashes:

```text
793202f295f3169c0d3255f452d0807ebda199dc422a8215431eedaf5597143d
```

# Representative Schemas

Representative examples:

- `young1996_raw.yml`: WorldView cognitive mapping, model type `property_graph`, with concepts and differentiated relationships such as positive-cause, negative-cause, attribute, condition, if-then, possess, and is-a. [1]
- `information_disorder_part2-3_raw.yml`: information-disorder solutions and policy response schema, model type `property_graph`, covering source identification, fact-checking, media literacy, regulation, and platform self-regulation. [2]
- `disarm_framework_raw.yml`: DISARM framework schema, model type `property_graph`, covering phases, tactics, techniques, countermeasures, detections, incidents, and playbooks. [3]

# Interpretation

The old schema archive shows that the thesis work had already explored a broad social-science and influence-operations schema corpus before the later, more disciplined schema-extraction and validation phases.

The archive also shows why later discipline was needed:

- schema breadth was high
- parseability was not perfect
- model-type labels were assigned, but later evidence shows model-form routing needed calibration
- raw schemas are valuable historical material, not automatically validated theory contracts

# Links

- [Old Schema Corpus Breadth](/wiki/concepts/old-schema-corpus-breadth.md)
- [Schema Extraction Pipeline Evolution](/wiki/concepts/schema-extraction-pipeline-evolution.md)
- [Model Form Routing](/wiki/concepts/model-form-routing.md)
- [Formal Notation As Theory Content](/wiki/concepts/formal-notation-as-theory-content.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/archive/old_schemas/young1996_raw.yml`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/archive/old_schemas/information_disorder_part2-3_raw.yml`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/archive/old_schemas/disarm_framework_raw.yml`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/archive/old_schemas/`
