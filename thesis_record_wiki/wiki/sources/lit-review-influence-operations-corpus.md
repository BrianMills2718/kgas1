---
type: Source
title: Lit Review Influence Operations Corpus
description: Inventory and generated-artifact summary for the preserved influence-operations literature folder.
tags: [source, lit-review, influence-operations, rand, property-graph, generated-schema]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/influence_operations/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/influence_operations/foundations_framework_improved.yml
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/influence_operations/foundations_framework_multiphase.yml
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/influence_operations/debug_improved/Foundations of Effective Influence Operations Chapter6-7 A Framework for influence operations_phase1.json
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/influence_operations/debug_improved/Foundations of Effective Influence Operations Chapter6-7 A Framework for influence operations_phase2.json
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/influence_operations/debug_improved/Foundations of Effective Influence Operations Chapter6-7 A Framework for influence operations_phase3.json
confidence: high
---

# Summary

The preserved `influence_operations/` folder is the largest lit-review literature subfolder: 15 files, including 10 source-text extracts, 2 YAML schema outputs, and 3 phase-debug JSON outputs. Its folder-level aggregate hash over sorted relative filenames and per-file hashes is `b158c83a953e92002f875c921260991e66334e62f2823d1d2b887ea2fe2f9360`. [1]

This slice captures folder structure and generated artifacts. It does not fully synthesize the underlying RAND/START influence-operations literature.

# Source Breadth

The source texts include RAND's `Foundations of Effective Influence Operations` chapters and appendices, an IVEO knowledge matrix, and START/LAS Influence-to-Action Model material. The RAND chapter extracts cover individual influence, groups and networks, adversary leadership coalitions, mass publics, a planning framework, case studies, and planning methodologies. [1]

This breadth makes the folder a strong example of multi-level influence modeling: individual cognition, group dynamics, leadership coalitions, mass publics, campaign planning, assessment metrics, and narrative impact.

# Generated Artifacts

The Chapter 6-7 framework has two preserved YAML schema variants and three phase-debug outputs:

| Artifact | Shape | Key Counts |
|---|---|---:|
| `foundations_framework_improved.yml` | property-graph schema with detailed definition list | 98 definitions across 30 categories. [2] |
| `foundations_framework_multiphase.yml` | compact property-graph schema | 26 definitions across entity/action/property/measure/relationship categories. [3] |
| `phase1.json` | vocabulary extraction | 100 vocabulary items. [4] |
| `phase2.json` | classified concepts | 50 entities, 3 relationships, 21 properties, 17 actions, 3 measures, 4 modifiers. [5] |
| `phase3.json` | final schema object | 11 node types, 20 edge types, property definitions, 4 modifiers, 2 truth values. [6] |

# Framework Content

The improved schema frames the RAND Chapter 6-7 material as a property graph because the framework links heterogeneous actors, objectives, stakeholders, messages, attitude structures, metrics, planning questions, and dose-response curves through many-to-many relationships. It emphasizes nine planning questions, target audiences, stakeholder salience, message campaigns, influence strategies, positive/negative inducements, conditional reciprocity, message acceptance, and adaptive assessment. [2][4][5][6]

# Interpretation

This folder is a high-density example of [Multi-Level Influence Modeling](/wiki/concepts/multi-level-influence-modeling.md). It also shows why a generic property graph can be justified for some sources: the theory object is explicitly multi-level, relational, metric-bearing, and campaign-oriented. That is different from using property graph as a default for every theory.

# Links

- [Multi-Level Influence Modeling](/wiki/concepts/multi-level-influence-modeling.md)
- [Literature Corpus As Theory Testbed](/wiki/concepts/literature-corpus-as-theory-testbed.md)
- [Model Form Routing](/wiki/concepts/model-form-routing.md)
- [Schema Variant Drift](/wiki/concepts/schema-variant-drift.md)
- [Lit Review Literature Corpus Inventory](/wiki/sources/lit-review-literature-corpus-inventory.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/influence_operations/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/influence_operations/foundations_framework_improved.yml`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/influence_operations/foundations_framework_multiphase.yml`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/influence_operations/debug_improved/Foundations of Effective Influence Operations Chapter6-7 A Framework for influence operations_phase1.json`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/influence_operations/debug_improved/Foundations of Effective Influence Operations Chapter6-7 A Framework for influence operations_phase2.json`  
[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/influence_operations/debug_improved/Foundations of Effective Influence Operations Chapter6-7 A Framework for influence operations_phase3.json`
