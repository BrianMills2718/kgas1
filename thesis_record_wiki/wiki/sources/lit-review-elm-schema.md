---
type: Source
title: Lit Review ELM Schema
description: Compact Elaboration Likelihood Model schema and summary from the mature schemas corpus.
tags: [source, lit-review, elm, persuasion, schema, sequential]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/elaboration_likelihood_model/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/elaboration_likelihood_model/elm_v10.yml
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/elaboration_likelihood_model/elm_v10_summary.txt
confidence: high
---

# Summary

The preserved `schemas/elaboration_likelihood_model/` folder contains two files: `elm_v10.yml` and `elm_v10_summary.txt`. Its aggregate hash over sorted filenames and per-file hashes is `bea9dfc3c2dc1ea789c5a40d51abdf35565fe261aabb9eb548c71e8a893494ea`. [1]

The schema models Petty and Cacioppo's Elaboration Likelihood Model as a simplified, micro-level effect theory with sequential execution. [2][3]

# Schema Shape

`elm_v10.yml` contains:

- `theory_id`: `elaboration_likelihood_model`
- `theory_name`: Elaboration Likelihood Model of Persuasion (ELM)
- `classification`: Micro level, Effect component, Interdependent metatheory, simplified complexity tier
- `ontology`: 4 entities and 6 relationships
- `execution`: 4 sequential analysis steps
- `telos`: prediction of attitude valence and strength
- `metadata`: implementation status `extracted`, validation status `pending` [2]

# Model Content

The entities are Receiver, Message, Route, and Attitude_Change_Outcome. The relationships encode motivation and ability enabling elaboration, elaboration selecting central/peripheral route, message arguments or cues influencing attitude through the selected route, and route determining attitude strength. [2][3]

The execution steps are:

1. assess receiver motivation and ability
2. compute elaboration likelihood
3. select persuasion route
4. map arguments/cues to predicted attitude and attitude strength [2][3]

# Interpretation

ELM is a useful contrast to [Lit Review Young1996 Schema Family](/wiki/sources/lit-review-young1996-schema-family.md). Young 1996 required many schema variants, computational algorithms, salience measures, and execution prompts. ELM is represented here as a compact sequential process with a small ontology and explicit prediction telos.

This supports [Model Form Routing](/wiki/concepts/model-form-routing.md): not every theory needs a property graph or complex multi-pass representation.

# Links

- [Model Form Routing](/wiki/concepts/model-form-routing.md)
- [Lit Review Young1996 Schema Family](/wiki/sources/lit-review-young1996-schema-family.md)
- [Lit Review Schemas Corpus Inventory](/wiki/sources/lit-review-schemas-corpus-inventory.md)
- [Schema Extraction Pipeline Evolution](/wiki/concepts/schema-extraction-pipeline-evolution.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/elaboration_likelihood_model/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/elaboration_likelihood_model/elm_v10.yml`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/elaboration_likelihood_model/elm_v10_summary.txt`
