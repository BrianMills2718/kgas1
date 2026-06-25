---
type: Source
title: Lit Review Turner Social Identity Extraction
description: Comparison of preserved multi-theory and single-theory extraction outputs for Turner and Oakes 1986 social identity paper.
tags: [source, lit-review, social-identity, turner-1986, multi-theory, extraction]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/extracted/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/extracted_single/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/extracted/British J Social Psychol - September 1986 - Turner - The significance of the social identity concept for social psychology_all_theories_20250810_112451.json
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/extracted_single/British J Social Psychol - September 1986 - Turner - The significance of the social identity concept for social psychology_summary_20250810_123047.json
confidence: high
---

# Summary

The preserved Turner/Oakes extraction artifacts compare two extraction modes over the same 1986 British Journal of Social Psychology paper: a multi-theory path in `schemas/extracted/` and a single-theory path in `schemas/extracted_single/`.

`schemas/extracted/` contains 18 files with aggregate hash `f2fc07c559e1dd8c995ec701b876763e8241235621fae0343c3a66ddd3db3deb`. `schemas/extracted_single/` contains 5 files with aggregate hash `6ed6089a4204ea4f4f479b9cebb591e22c84610fd5829d520742cdc90dc81852`. [1][2]

# Source Paper

The extracted fulltext is Turner and Oakes, "The significance of the social identity concept for social psychology with reference to individualism, interactionism and social influence." The opening text frames the paper as contrasting individualism and interactionism, introducing social identity theory, social influence, and group polarization, and arguing that social identity demonstrates non-individualistic social psychology. [1][2]

The same fulltext hash appears repeatedly across extracted and extracted_single copies, indicating duplicated source text sidecars rather than distinct papers.

# Multi-Theory Extraction

The multi-theory identification output reports `theories_count: 6` and `extraction_strategy: nested`. The extracted theories include:

- `individualism_metatheory`
- `interactionism_metatheory`
- `social_identity_theory_intergroup_distinctiveness`
- `self_categorization_theory`
- `referent_informational_social_influence_theory`
- `group_polarization_prototype_convergence`

The combined `all_theories` artifact stores six theory records; the per-theory JSON/YAML artifacts share a schema with keys for `theory_id`, `theory_name`, `classification`, `ontology`, `execution`, `telos`, `operationalization_clarity`, and `uncertainty_dimensions`. [3]

# Single-Theory Extraction

The single-theory path identifies the primary theory as Self-Categorization Theory. It records core mechanisms, applications, an `extends_theory` relationship to Social Identity Theory, and a full theory object with `theoretical_structure`, `computational_representation`, `algorithms`, `telos`, and validity evidence. [4]

# Interpretation

This slice is concrete evidence for [Multi Theory Extraction Split](/wiki/concepts/multi-theory-extraction-split.md). The same source can be represented either as a nested set of metatheories and derivative theories or as one primary theory with extensions. Those outputs are not interchangeable: they answer different modeling questions.

This also explains why later prompt-routing work cared about theory-count detection. A single-theory extraction can be useful when the goal is one executable schema, but it can hide metatheoretical structure and derivative theory relationships.

# Links

- [Multi Theory Extraction Split](/wiki/concepts/multi-theory-extraction-split.md)
- [Schema Extraction Pipeline Evolution](/wiki/concepts/schema-extraction-pipeline-evolution.md)
- [Literature Corpus As Theory Testbed](/wiki/concepts/literature-corpus-as-theory-testbed.md)
- [Lit Review Schemas Corpus Inventory](/wiki/sources/lit-review-schemas-corpus-inventory.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/extracted/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/extracted_single/`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/extracted/British J Social Psychol - September 1986 - Turner - The significance of the social identity concept for social psychology_all_theories_20250810_112451.json`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/extracted_single/British J Social Psychol - September 1986 - Turner - The significance of the social identity concept for social psychology_summary_20250810_123047.json`
