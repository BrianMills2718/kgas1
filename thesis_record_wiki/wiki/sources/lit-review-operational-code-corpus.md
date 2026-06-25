---
type: Source
title: Lit Review Operational Code Corpus
description: Operational-code and Young WorldView literature slice, including five source texts and four Young 1996 schema variants.
tags: [source, lit-review, operational-code, worldview, young-1996, schema-variants]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/operational_code_analysis/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/operational_code_analysis/Cognitive Mapping Meets Semantic Networks.txt
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/operational_code_analysis/Systematic Procedures for Operational Code Analysis.txt
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/operational_code_analysis/young1996_complete.yml
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/operational_code_analysis/young1996_faithful.yml
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/operational_code_analysis/young1996_multiphase.yml
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/operational_code_analysis/young1996_o3_test.yml
confidence: high
---

# Summary

The preserved `operational_code_analysis/` folder contains nine files: five source-text extracts and four Young 1996 schema variants. The folder-level aggregate hash over sorted filenames and per-file hashes is `9f4bd14cdfc78781178933fdc83b10736643ed69184a4823eb6509cf2d003af4`. [1]

The source texts cover Young's WorldView paper, Walker/Schafer/Young's systematic operational-code scoring procedure, Walker's historical evolution of operational-code analysis, a public-material validity thesis, and a VICS/populist-right validity thesis. [2][3][4][5][6]

# Why This Slice Matters

Operational code and WorldView recur across the thesis record:

- Young 1996 is a validation case in [Lit Review Validation Results](/wiki/sources/lit-review-validation-results.md).
- Young 1996 is later treated as hybrid in [Lit Review Prompt Variation Model Routing](/wiki/sources/lit-review-prompt-variation-model-routing.md).
- Young 1996 schemas support the universal applicator example in [Lit Review Universal Theory Applicator](/wiki/sources/lit-review-universal-theory-applicator.md).
- Carter output applies cognitive mapping/WorldView ideas to a concrete speech in [Carter Theory Analysis Output](/wiki/sources/carter-theory-analysis-output.md).

# Source Family

The folder is not just one paper. It preserves a small operational-code research lineage:

- Young 1996: enhanced cognitive mapping using semantic-network formalism and the WorldView system. [2]
- Walker/Schafer/Young 1998: VICS-based operational-code indices over public speeches, tested on Jimmy Carter. [3]
- Walker 1990: evolution from Leites/George/Holsti toward a synthesis of cognitive and affective influences on foreign-policy decisions. [4]
- Hulsbus 2019: concern about whether public speech material validly represents leaders' beliefs. [5]
- Luengas 2020: concern that VICS may misread populist communication style because surface wording can obscure contextual meaning. [6]

# Young 1996 Schema Variants

The four preserved Young YAML files show schema-variant drift:

| File | Model Type | Shape | Notable Content |
|---|---|---|---|
| `young1996_complete.yml` | WorldView enhanced cognitive mapping using semantic-network formalism | nested dictionary definitions | 12 relationship categories, 5 truth values, 6 modifiers, 4 structural measures, 3 comparative measures, 3 reasoning methods, 5 extraction requirements. [7] |
| `young1996_faithful.yml` | WorldView enhanced cognitive mapping using semantic-network formalism | nested dictionary definitions | Preserves 45 action categories, 12 relationship categories, truth values, modifiers, measures, and support features. [8] |
| `young1996_multiphase.yml` | `property_graph` | list of 25 definitions | Recasts the theory as entity/action/measure/property/relationship definitions. [9] |
| `young1996_o3_test.yml` | `property_graph` | list of 25 definitions | Similar property-graph recasting from a different test pass. [10] |

# Caveats

The `young1996_faithful.yml` file contains unquoted YAML values `true` and `false` in `truth_values`; Python YAML parsing coerces them to booleans. This is a representation caveat, not necessarily a conceptual error in the source artifact. [8]

The `young1996_complete.yml` and `young1996_faithful.yml` variants preserve theory-specific structures more directly than the property-graph variants. The property-graph variants are useful as generalized schema outputs, but they compress WorldView-specific details into generic entities, relationships, properties, actions, and measures. [7][8][9][10]

# Interpretation

This slice is strong evidence for [Formal Notation As Theory Content](/wiki/concepts/formal-notation-as-theory-content.md) and [Schema Variant Drift](/wiki/concepts/schema-variant-drift.md). Young's theory is not just "a graph"; its relationship codes, action categories, truth values, modifiers, salience counts, structural measures, comparative measures, and reasoning procedures are part of the theory content. Generic property-graph extraction preserves some topology but risks losing representational commitments.

# Links

- [Schema Variant Drift](/wiki/concepts/schema-variant-drift.md)
- [Formal Notation As Theory Content](/wiki/concepts/formal-notation-as-theory-content.md)
- [Model Form Routing](/wiki/concepts/model-form-routing.md)
- [Prompt Calibration Loop](/wiki/concepts/prompt-calibration-loop.md)
- [Lit Review Literature Corpus Inventory](/wiki/sources/lit-review-literature-corpus-inventory.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/operational_code_analysis/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/operational_code_analysis/Cognitive Mapping Meets Semantic Networks.txt`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/operational_code_analysis/Systematic Procedures for Operational Code Analysis.txt`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/operational_code_analysis/The Evolution of Operational Code Analysis.txt`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/operational_code_analysis/Public material in operational code analysis: An assessment.txt`  
[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/operational_code_analysis/The VICS Test: Does Operational Code Analysis Falter for The Populist Right?.txt`  
[7] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/operational_code_analysis/young1996_complete.yml`  
[8] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/operational_code_analysis/young1996_faithful.yml`  
[9] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/operational_code_analysis/young1996_multiphase.yml`  
[10] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/operational_code_analysis/young1996_o3_test.yml`
