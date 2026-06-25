---
type: Source
title: Lit Review Prompt Variation Model Routing
description: Prompt-variation artifacts for theory-count detection, specialized model-form detection, confidence integration, and hybrid schema generation.
tags: [source, lit-review, prompt-variation, model-routing, hybrid, calibration]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/prompt_variations/theory_count_detection/theory_count_prompt_v1.txt
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/prompt_variations/theory_count_detection/young_result_v1.txt
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/prompt_variations/theory_count_detection/heilman_result_v1.txt
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/prompt_variations/specialized_model_detection/comprehensive_results_analysis.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/prompt_variations/specialized_model_detection/confidence_integration_results.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/prompt_variations/hybrid_integration/hybrid_detector.txt
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/prompt_variations/hybrid_integration/hybrid_schema_generator.txt
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/prompt_variations/hybrid_integration/young1996_hybrid_result.txt
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/prompt_variations/hybrid_integration/owl_stress_test_hybrid_result.txt
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/prompt_variations/hybrid_integration/young1996_hybrid_schema.yml
confidence: high
---

# Summary

This slice covers the prompt-variation work behind model routing: detecting whether a paper contains one theory, multiple theories, an integration, or a review; detecting representational form; calibrating detector confidence; and generating hybrid schemas when multiple representational paradigms are needed.

The main value is the calibration loop. The artifacts show the project identifying a detector bias, refining prompts, adding decision rules, and then introducing hybrid detection and schema generation.

# Theory Count Detection

`theory_count_prompt_v1.txt` distinguishes single theory, multiple theories, theory integration, and review/survey papers. It asks for individual theories, base theories, integration type, relationships among theories, confidence, and reasoning. [1]

Results show the prompt making useful distinctions:

- Young 1996 is classified as an integration of cognitive mapping and semantic networks into WorldView, with high confidence. [2]
- Heilman/framing is classified as multiple theories: core framing effects, taxonomy of framing effects, and Prospect Theory explanation, with high confidence. [3]

# Specialized Model Detection

The specialized-detector analysis compares table/matrix, property-graph, and sequence detectors across Heilman, Lofland-Stark, and Young 1996. [4]

Key findings:

- Heilman/framing should be table/matrix; table detector was correct but low-confidence at 0.3.
- Lofland-Stark should be sequence; sequence detector worked perfectly at 1.0.
- Young 1996 revealed hybrid structure because table/matrix, graph, and sequence detectors all had high confidence.
- The first property-graph detector had over-confidence bias: it treated "has relationships" as "is a network theory."
- A refined property-graph detector fixed over-confidence in Heilman and Lofland while preserving high confidence for Young. [4][5]

# Confidence Integration

The confidence-integration report proposes decision rules rather than trusting raw max confidence. Early rules privilege high sequence confidence, adjust table thresholds, and require property-graph confidence only when other forms are low. Later rules detect hybrid cases when multiple forms exceed threshold. [4][5]

This is important because model routing is not just prompt design; it needs arbitration logic across detector outputs.

# Hybrid Detection And Schema Generation

The hybrid detector prompt asks whether a theory fundamentally requires multiple structural paradigms. It requires evidence for distinct components, integration mechanisms, integration type, and whether a single model could suffice. [6]

The hybrid schema generator prompt then asks for a unified schema that preserves component integrity, maps relationships across components, and supports cross-component reasoning. [7]

Representative results:

- Young 1996 is detected as hybrid: cognitive mapping as table/matrix plus semantic networks as property graph, integrated into a WorldView framework. [8]
- The OWL stress-test schema is detected as a complex hybrid with property graph, table/matrix, sequence, statistical, logical, and inference-engine components. [9]
- A generated Young 1996 hybrid schema preserves both adjacency-matrix cognitive mapping and property-graph semantic networks, with integration mappings between them. [10]

# Interpretation

The prompt-variation work is a concrete example of [Prompt Calibration Loop](/wiki/concepts/prompt-calibration-loop.md): start with a detector prompt, test it on known cases, identify systematic bias, refine prompt criteria, add confidence-integration logic, and only then route downstream schema generation.

It also deepens [Model Form Routing](/wiki/concepts/model-form-routing.md): representational form should be detected and validated before forcing theory output into a single schema type.

# Links

- [Prompt Calibration Loop](/wiki/concepts/prompt-calibration-loop.md)
- [Model Form Routing](/wiki/concepts/model-form-routing.md)
- [Complexity Conservation In Theory Application](/wiki/concepts/complexity-conservation-in-theory-application.md)
- [Lit Review Model Form Detection Results](/wiki/sources/lit-review-model-form-detection-results.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/prompt_variations/theory_count_detection/theory_count_prompt_v1.txt`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/prompt_variations/theory_count_detection/young_result_v1.txt`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/prompt_variations/theory_count_detection/heilman_result_v1.txt`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/prompt_variations/specialized_model_detection/comprehensive_results_analysis.md`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/prompt_variations/specialized_model_detection/confidence_integration_results.md`  
[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/prompt_variations/hybrid_integration/hybrid_detector.txt`  
[7] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/prompt_variations/hybrid_integration/hybrid_schema_generator.txt`  
[8] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/prompt_variations/hybrid_integration/young1996_hybrid_result.txt`  
[9] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/prompt_variations/hybrid_integration/owl_stress_test_hybrid_result.txt`  
[10] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/prompt_variations/hybrid_integration/young1996_hybrid_schema.yml`
