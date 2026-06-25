---
type: Concept
title: Schema Extraction Pipeline Evolution
description: Evolution from simple theory extraction toward multiphase, full-vocabulary, model-adaptive schema generation.
tags: [concept, schema-extraction, prompts, lit-review, theory-extraction]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_creation/fix_information_loss.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_creation/multiphase_processor_improved.py
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_creation/theory_extractor_v13_single.py
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_creation/theory_extractor_v13_multi.py
confidence: medium
---

# Summary

Schema extraction evolved from quick/simple extraction toward a more explicit production path: exhaustive vocabulary extraction, ontological classification, full-vocabulary schema generation, and then v13 full-paper single- or multi-theory extraction. [1][2][3][4]

# Core Correction

The central pipeline correction is information preservation between phases. The archive note says Phase 3 was losing terms because it only saw a Phase 2 summary. The improved processor passes complete Phase 1 vocabulary into Phase 3 and requires the final schema to include all Phase 1 terms. [1][2]

# Design Direction

The direction of travel is:

- no artificial vocabulary cap
- no text truncation for full-paper extraction
- no default to property graph when table/matrix, sequence, tree, timeline, hypergraph, or custom structure is better
- external prompts rather than buried prompt strings
- Pydantic/structured parsing in newer multiphase code
- single-theory and multi-theory variants depending on paper structure [2][3][4]

# Why It Matters

This thread explains why the lit-review experiment was not merely "make an ontology from a paper." It became a pipeline-design problem: preserve theoretical vocabulary, choose the right representation, handle multiple theories, and make generated schemas operational enough to apply to texts.

[Lit Review Validation Results](/wiki/sources/lit-review-validation-results.md) gives the first ingested empirical reason for this design pressure: model-type selection can be right for Young and Lofland-Stark while wrong for framing effects.

# Caveat

The archive also shows an unfinished transition to stronger engineering discipline: hardcoded paths, multiple extractor generations, and `json_object` usage coexist with better prompt externalization and Pydantic parsing. Treat the pipeline as historically valuable and conceptually mature in places, but not cleanly productionized without further verification.

# Links

- [Formal Notation As Theory Content](/wiki/concepts/formal-notation-as-theory-content.md)
- [Automated Theory Extraction](/wiki/concepts/automated-theory-extraction.md)
- [Lit Review Schema Creation Production Path](/wiki/sources/lit-review-schema-creation-production-path.md)
- [Multi-Theory Application Artifact](/wiki/concepts/multi-theory-application-artifact.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_creation/fix_information_loss.md`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_creation/multiphase_processor_improved.py`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_creation/theory_extractor_v13_single.py`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_creation/theory_extractor_v13_multi.py`
