---
type: Source
title: Lit Review Validation Results
description: Source summary for validation_results reports and selected baseline/debug outputs in the lit-review experiment.
tags: [source, lit-review, validation, model-type-selection, complexity]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/validation_results/validation_summary.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/validation_results/academic_papers/young1996_comparison_report.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/validation_results/cross_domain/framing_comparison_report.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/validation_results/complexity_testing/simple_theory_comparison_report.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/validation_results/academic_papers/young1996_baseline_output.yml
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/validation_results/cross_domain/framing_simple_output.yml
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/validation_results/complexity_testing/lofland_stark_baseline_output.yml
confidence: medium
---

# Summary

`validation_results` is a compact validation archive for the lit-review experiment. It contains reports and outputs for:

- Young 1996 cognitive mapping as an academic paper / political science methodology test. [2][5]
- Framing effects as cross-domain economics/psychology validation. [3][6]
- Lofland-Stark conversion theory as a simple sequential-theory complexity test. [4][7]

# Main Pattern

The most durable finding is not the final "100% complete" claim; it is the pattern across test cases:

- Young 1996: good vocabulary extraction and correct `property_graph` selection for a medium political science/methodology paper. [2][5]
- Framing effects: good vocabulary but wrong baseline model type, because the report says the theory should be `table_matrix` rather than `property_graph`. [3][6]
- Lofland-Stark: strong baseline performance and correct `sequence` model type for a simple seven-step theory. [4][7]

This supports the validation summary's strategic recommendation: automate simpler theories for high-volume processing and reserve complex/hybrid theories for expert-guided analysis. [1]

# Evidence Strength

The validation set is stronger than roadmap prose because it includes comparison reports plus baseline YAML/debug outputs. For example, Young's debug outputs record 53 Phase 1 vocabulary terms, Phase 2 classification counts, and Phase 3 selecting `property_graph` with 14 node types and 20 edge types. [5]

However, the reports are still internal validation artifacts. They compare baseline and "advanced" analysis, but the ingested reports do not yet show an external reviewer, locked gold set, or independent benchmark.

# Internal Contradiction

The validation summary contains a temporal/status compression problem. Earlier tables mark medium/complex theory and discourse-type diversification as `TBD`, while a later section says validation is 100% complete, all critical problems are solved, and 100% model-type accuracy was achieved. [1]

This should be treated as a [Documentation Status Truthfulness](/wiki/concepts/documentation-status-truthfulness.md) issue rather than resolved fact. The conservative reading is:

- phases 1, 2, and simple complexity testing are directly represented by reports
- broader "all complexity levels" and "100% model type accuracy" claims require follow-up evidence

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/validation_results/validation_summary.md`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/validation_results/academic_papers/young1996_comparison_report.md`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/validation_results/cross_domain/framing_comparison_report.md`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/validation_results/complexity_testing/simple_theory_comparison_report.md`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/validation_results/academic_papers/young1996_baseline_output.yml`  
[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/validation_results/cross_domain/framing_simple_output.yml`  
[7] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/validation_results/complexity_testing/lofland_stark_baseline_output.yml`
