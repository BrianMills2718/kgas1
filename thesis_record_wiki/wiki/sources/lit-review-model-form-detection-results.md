---
type: Source
title: Lit Review Model Form Detection Results
description: Optimized experimental model-form detection outputs for Lofland-Stark conversion theory and Heilman/Miclea framing theory.
tags: [source, lit-review, model-form, architecture-comparison, validation]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/architecture_comparison/lofland_optimized_result.yml
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/architecture_comparison/heilman_optimized_result.yml
confidence: high
---

# Summary

This slice covers the preserved `experimental_testing/architecture_comparison` outputs. Despite the directory name, the preserved evidence is not a full sequential-vs-parallel-vs-hybrid architecture benchmark. The sibling directories `parallel_results/`, `hybrid_results/`, and `sequential_results/` exist but contain no files in this archive.

The two available YAML files are better understood as optimized model-form detection outputs: they classify theory texts by whether they are table/matrix, property graph, sequence, or hybrid representations.

# Lofland-Stark Result

The Lofland-Stark conversion theory result analyzes `validation_results/complexity_testing/lofland_stark_simple_theory.txt`. It identifies a single theory and classifies it as a sequential funnel model. [1]

Key extracted sequence steps:

- Tension
- Religious Problem-Solving Perspective
- Seekership
- Turning Point
- Affective Bonds
- Neutralization of Extra-Cult Attachments
- Intensive Interaction [1]

The specialized detectors assign low confidence to table/matrix and property-graph representations, while the sequence detector has confidence 1.0. Hybrid detection is false with confidence 0.95. [1]

# Heilman / Framing Result

The Heilman/Miclea framing result analyzes `validation_results/cross_domain/framing_theory_core.txt`. It detects multiple theory components: Framing Effects Theory, a taxonomy of framing effects, and Prospect Theory. [2]

The specialized detectors assign low confidence to table/matrix, property graph, and sequence forms. Hybrid detection is false with confidence 0.95 and describes a "Decision Evaluation Process" with `model_type: statistical`, grounded in Prospect Theory. [2]

# Internal Caveat

The Heilman result ends with:

```yaml
model_type: sequence
confidence: 0.8
```

That conflicts with its own detector rationale, which says sequence confidence is 0.2 and the primary component is statistical/probabilistic. This should be treated as an internal consistency issue in the optimized result, not as a settled classification.

# Interpretation

These files are useful because they show the project moving toward model-form routing: not every theory should be represented as a graph, table, sequence, or hybrid just because the system can support those forms.

But the evidence is limited:

- only two result files are present
- no runner script or comparison report was found in the preserved `architecture_comparison` directory
- the named parallel/hybrid/sequential result directories are empty
- one result has an internal label/rationale mismatch

# Links

- [Model Form Routing](/wiki/concepts/model-form-routing.md)
- [Automated Theory Extraction](/wiki/concepts/automated-theory-extraction.md)
- [Complexity Accuracy Pattern](/wiki/concepts/complexity-accuracy-pattern.md)
- [Complexity Conservation In Theory Application](/wiki/concepts/complexity-conservation-in-theory-application.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/architecture_comparison/lofland_optimized_result.yml`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/architecture_comparison/heilman_optimized_result.yml`
