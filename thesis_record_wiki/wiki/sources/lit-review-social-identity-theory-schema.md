---
type: Source
title: Lit Review Social Identity Theory Schema
description: Smaller Turner 1986 social identity theory schema slice adjacent to the extracted and extracted_single Turner outputs.
tags: [source, lit-review, social-identity, self-categorization, turner-1986, schema]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/social_identity_theory/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/social_identity_theory/turner_1986_sit_schema_20250810_105656.json
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/social_identity_theory/turner_1986_sit_schema_20250810_105656.yaml
confidence: high
---

# Summary

The preserved `schemas/social_identity_theory/` folder contains three files: one Turner/Oakes extracted text sidecar and one Self-Categorization Theory schema saved as both JSON and YAML. The folder aggregate hash over sorted filenames and per-file hashes is `0ddf46a44f3b22d17faa9e870b776b0477ec22df4ed5ebd6c700893829218812`. [1]

The extracted text hash matches the Turner/Oakes fulltext sidecars in `schemas/extracted/` and `schemas/extracted_single/`, so this is another extraction/schema pass over the same source paper. [1]

# Schema Shape

The JSON and YAML schema files both identify:

- `theory_id`: `self_categorization_theory`
- `theory_name`: Self-Categorization Theory (Social Identity Approach)
- authors: Turner and Oakes
- publication year: 1986

The preserved keys include `classification`, `ontology`, `execution`, `telos`, `metadata`, and `paper_specific`. [2][3]

# Relationship To Turner Extraction Comparison

This slice sits between the single-theory and multi-theory outputs:

- like `extracted_single`, it focuses on Self-Categorization Theory
- unlike the grouped `extracted/` path, it does not preserve six separate theory/metatheory records
- unlike the later single-theory output, its top-level schema shape uses `classification`, `ontology`, `execution`, and `telos` rather than `theoretical_structure`, `computational_representation`, and `algorithms`

That makes it another example of [Schema Variant Drift](/wiki/concepts/schema-variant-drift.md), but within the Turner/social-identity family rather than Young 1996.

# Links

- [Lit Review Turner Social Identity Extraction](/wiki/sources/lit-review-turner-social-identity-extraction.md)
- [Multi Theory Extraction Split](/wiki/concepts/multi-theory-extraction-split.md)
- [Schema Variant Drift](/wiki/concepts/schema-variant-drift.md)
- [Lit Review Schemas Corpus Inventory](/wiki/sources/lit-review-schemas-corpus-inventory.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/social_identity_theory/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/social_identity_theory/turner_1986_sit_schema_20250810_105656.json`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/social_identity_theory/turner_1986_sit_schema_20250810_105656.yaml`
