---
type: Source
title: Lit Review Literature Corpus Inventory
description: Inventory of the preserved lit-review literature directory, including annotated bibliography scope, topical folders, file types, and schema-location caveats.
tags: [source, lit-review, literature, corpus, inventory]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/Index.md
confidence: high
---

# Summary

The preserved `literature/` directory is the raw theory-source corpus behind much of the lit-review schema work. It contains 73 files across 26 topical folders plus `Index.md`. The file mix is 58 `.txt` extracted source files, 8 `.yml` schema files, 5 `.json` debug/intermediate files, 1 `.md` index, and 1 `.py` post-processing script. [1]

The directory-level aggregate hash over sorted relative filenames and per-file hashes is `6b9b791db84ab252158b0c3d0b20d6034d7d9c7d6aa2a83e8088d8fa1d123f7c`. [1]

# Annotated Bibliography Scope

`Index.md` is an annotated bibliography with topic-code taxonomy and 47 paper entries. Its codes cover methodology, domain, and application dimensions: cognitive mapping, semantic networks, schema theory, content analysis, operational code analysis, belief systems, worldview theory, political science, psychology, social psychology, decision making, crisis decision making, negotiation, leadership analysis, cultural analysis, measurement, and theory development. [2]

The indexed papers span operational code analysis, worldview theory, argumentation, behavior change, persuasion, framing, information disorder, information operations, social marketing, social media rituals, radicalization, intervention tools, risk/framing effects, and social innovation. [2]

# Folder Breadth

Topical folders and preserved file counts:

| Folder | Files |
|---|---:|
| `influence_operations` | 15 |
| `operational_code_analysis` | 9 |
| `behavior_change` | 8 |
| `social_marketing` | 7 |
| `decision_analysis` | 3 |
| `argumentation` | 2 |
| `attitude_change` | 2 |
| `conversion_theory` | 2 |
| `framing_theory` | 2 |
| `health_communication` | 2 |
| `information_disorder` | 2 |
| `information_operations` | 2 |
| `intervention_tools` | 2 |
| `persuasion` | 2 |
| single-file folders | 14 |

# Schema-Location Caveat

The index usually lists schema locations ending in `_raw.yml`, but the preserved directory mostly contains `.txt` files. Only eight `.yml` files are present in this corpus copy:

- `decision_analysis/poliheuristic_theory_analysis.yml`
- `framing_theory/chong_druckman_2007_framing_theory.yml`
- `influence_operations/foundations_framework_improved.yml`
- `influence_operations/foundations_framework_multiphase.yml`
- `operational_code_analysis/young1996_complete.yml`
- `operational_code_analysis/young1996_faithful.yml`
- `operational_code_analysis/young1996_multiphase.yml`
- `operational_code_analysis/young1996_o3_test.yml`

This suggests the index preserves an intended or earlier schema organization that is not fully materialized in this archive slice. The missing `_raw.yml` files may exist elsewhere in the lineage, may have been renamed into the old schema archive, or may never have been committed here. Treat `Index.md` schema paths as provenance claims, not verified file existence. [1][2]

# Interpretation

The literature corpus shows the thesis work was not centered on one theory family. It was a broad testbed for converting heterogeneous social-science theories into computational schemas. This explains why later work discovered model-form routing problems: the corpus deliberately mixes graphs, sequences, taxonomies, statistical/experimental findings, intervention frameworks, argument structures, and operational-code scoring systems.

# Links

- [Literature Corpus As Theory Testbed](/wiki/concepts/literature-corpus-as-theory-testbed.md)
- [Old Schema Corpus Breadth](/wiki/concepts/old-schema-corpus-breadth.md)
- [Lit Review Old Schema Archive Inventory](/wiki/sources/lit-review-old-schema-archive-inventory.md)
- [Model Form Routing](/wiki/concepts/model-form-routing.md)
- [Analyst Assembler Pattern](/wiki/concepts/analyst-assembler-pattern.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/Index.md`
