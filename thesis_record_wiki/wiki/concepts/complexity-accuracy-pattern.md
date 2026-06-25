---
type: Concept
title: Complexity Accuracy Pattern
description: Lit-review validation pattern that baseline extraction accuracy improves as theory structure becomes simpler.
tags: [concept, validation, complexity, model-type-selection]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/validation_results/validation_summary.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/validation_results/complexity_testing/simple_theory_comparison_report.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/validation_results/cross_domain/framing_comparison_report.md
confidence: medium
---

# Summary

The complexity-accuracy pattern is the lit-review validation claim that baseline extraction accuracy improves as theoretical complexity decreases. Simple sequential theories are easiest; medium theories get useful vocabulary but may select the wrong representation; complex/hybrid theories require expert guidance. [1][2][3]

# Evidence Ingested So Far

The strongest directly ingested evidence is:

- Lofland-Stark simple theory: baseline selected `sequence`, extracted complete vocabulary, and preserved sequential logic. [2]
- Framing effects medium cross-domain theory: baseline extracted useful vocabulary but selected `property_graph` when the report says `table_matrix` was correct. [3]
- Young 1996 medium political science methodology: baseline selected `property_graph` correctly and extracted a substantial vocabulary. [1]

# Strategic Implication

This pattern matters because it suggests a practical thesis implementation strategy:

- use automated extraction for simple and structured theories
- add model-type decision support for medium theories
- reserve complex hybrid synthesis for expert-guided workflows

That is more credible and operationally useful than claiming the whole theory-extraction problem was solved uniformly.

[Balance Driven Validation](/wiki/concepts/balance-driven-validation.md) is a related but separate validation pattern: it checks purpose parity, while complexity/accuracy checks which theory structures are suitable for automation.

# Caution

The validation summary later claims 100% validation completion and 100% model-type accuracy across all complexity levels, but the same file still contains `TBD` rows for medium/complex tests and discourse diversification. Until those follow-up artifacts are ingested, this page preserves the narrower, better-supported pattern.

# Links

- [Lit Review Validation Results](/wiki/sources/lit-review-validation-results.md)
- [Schema Extraction Pipeline Evolution](/wiki/concepts/schema-extraction-pipeline-evolution.md)
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/validation_results/validation_summary.md`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/validation_results/complexity_testing/simple_theory_comparison_report.md`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/validation_results/cross_domain/framing_comparison_report.md`
