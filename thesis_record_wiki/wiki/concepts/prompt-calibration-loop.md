---
type: Concept
title: Prompt Calibration Loop
description: Iterative pattern for turning prompts into reliable detectors by testing known cases, identifying bias, refining criteria, and adding decision logic.
tags: [concept, prompt-design, calibration, model-routing, validation]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/prompt_variations/specialized_model_detection/comprehensive_results_analysis.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/prompt_variations/specialized_model_detection/confidence_integration_results.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/testing/experimental_processor_optimized.py
confidence: high
---

# Summary

The prompt calibration loop is the experimental pattern visible in the prompt-variation artifacts:

1. Write an initial detector prompt.
2. Run it on known cases.
3. Compare detector confidence and rationale to expected model forms.
4. Identify systematic bias.
5. Refine prompt criteria.
6. Add confidence-integration or arbitration rules.
7. Retest and only then route downstream generation.

# Example

The property-graph detector initially over-classified theories as networks because nearly all theories have relationships among concepts. The refined detector narrowed the criterion to whether network structure is the primary theoretical contribution. That reduced false confidence on Heilman/framing and Lofland-Stark while preserving high confidence for Young 1996. [1][2]

The testing code later wraps this pattern into `experimental_processor_optimized.py`: theory-count detection, specialized detector prompts, hybrid detection, hybrid schema generation, and decision-rule integration. Its confidence extraction is simplified, so it is evidence of experimental workflow rather than a production router. [3]

# Rule

Prompt outputs are not evidence by themselves. A detector becomes useful only after calibration against known cases and after its failure modes are written down.

# Links

- [Lit Review Prompt Variation Model Routing](/wiki/sources/lit-review-prompt-variation-model-routing.md)
- [Lit Review Testing Code](/wiki/sources/lit-review-testing-code.md)
- [Model Form Routing](/wiki/concepts/model-form-routing.md)
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/prompt_variations/specialized_model_detection/comprehensive_results_analysis.md`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/prompt_variations/specialized_model_detection/confidence_integration_results.md`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/testing/experimental_processor_optimized.py`
