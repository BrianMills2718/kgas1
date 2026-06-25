---
type: Source
title: Lit Review Social Marketing Corpus
description: Social-marketing literature folder with CORE/sharedProps/post_process artifacts that confirm the common-ontology injection pattern.
tags: [source, lit-review, social-marketing, common-ontology, post-processing]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/social_marketing/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/social_marketing/CORE.json
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/social_marketing/sharedProps.json
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/social_marketing/post_process.py
confidence: high
---

# Summary

The preserved `social_marketing/` folder contains seven files: four source-text extracts, `CORE.json`, `sharedProps.json`, and `post_process.py`. The folder-level aggregate hash over sorted filenames and per-file hashes is `b7bc1836385c85a53a9efe32a71df6fa041dec5962e12bed44b2423147049b45`. [1]

This slice is important because its support files independently confirm the [Analyst Assembler Pattern](/wiki/concepts/analyst-assembler-pattern.md) and common-ontology injection strategy seen in the legacy system docs.

# Source Breadth

The source texts cover social marketing history/domains, principles, determinants/context/consequences for individual and social change, and a planned-social-change article. The chapter extracts emphasize wicked problems, marketing orientation, segmentation, marketing mix, market asymmetries, micro/macro change, social networks, organizations, communities, physical environments, markets, and policy contexts. [1]

# CORE And Shared Props

`CORE.json` defines reusable schema components:

- `Entity`
- `Role`
- `NaryTuple`
- `Argument`

`sharedProps.json` defines a reusable `sharedProps` object. [2][3]

These are the same categories described in the legacy post-processing guide: common definitions are injected into theory-specific schema `$defs` so individual theory outputs share an interoperable foundation.

# Post-Processing Script

`post_process.py` loads an AI-generated YAML file, loads `CORE.json`, loads `sharedProps.json`, injects both dictionaries into the target schema's `$defs`, and writes a final processed YAML artifact. [4]

The configured filenames are:

- input AI YAML: `social_marketing_kotler_zaltman_raw.yml`
- CORE definitions: `CORE.json`
- shared props: `sharedProps.json`
- output YAML: `social_marketing_kotler_zaltman_schema.yml`

# Provenance Caveat

The configured input and output YAML files are not present in this preserved folder copy. The folder preserves the assembler machinery and source texts, but not the corresponding social-marketing raw/generated YAML artifacts in this location. Treat this as a partial provenance slice rather than a complete runnable example. [1][4]

# Interpretation

This is a concrete implementation of the thesis's early separation of semantic extraction from deterministic schema assembly. The LLM-facing artifact is expected to be a theory-specific YAML schema, while `post_process.py` provides deterministic injection of common ontology definitions.

The social-marketing folder therefore links the broad literature corpus to the common-ontology engineering path: it is not merely another source folder.

# Links

- [Analyst Assembler Pattern](/wiki/concepts/analyst-assembler-pattern.md)
- [Lit Review Legacy System Framing](/wiki/sources/lit-review-legacy-system-framing.md)
- [Literature Corpus As Theory Testbed](/wiki/concepts/literature-corpus-as-theory-testbed.md)
- [Schema Extraction Pipeline Evolution](/wiki/concepts/schema-extraction-pipeline-evolution.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/social_marketing/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/social_marketing/CORE.json`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/social_marketing/sharedProps.json`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/literature/social_marketing/post_process.py`
