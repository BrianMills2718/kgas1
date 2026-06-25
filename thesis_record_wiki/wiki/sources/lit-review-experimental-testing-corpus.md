---
type: Source
title: Lit Review Experimental Testing Corpus
description: Inventory and interpretation of the experimental_testing directory as a whole, including validation summaries, architecture-comparison outputs, prompt-variation experiments, and empty validation_retesting.
tags: [source, lit-review, experimental-testing, validation, prompt-routing, model-form, hybrid]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/final_validation_summary.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/optimization_findings.md
confidence: high
---

# Summary

The preserved `experimental_testing/` directory contains 41 files, 119,107 bytes, with aggregate hash `d8b9c752008b40b218e7b90734c82f59215c8b9d19f5ca781aacab8a3715d9d7`. [1]

It contains:

| Subtree | Files | Bytes | Aggregate hash |
|---|---:|---:|---|
| `architecture_comparison/` | 2 | 14,548 | `e31c0897fde67f8714331ec9f7693cfb822e09f46f20d39cd1e60284ece21d6a` |
| `prompt_variations/` | 37 | 86,498 | `add018aea53e96f84bdab7ce6ffa67d124d48d7a80e5f7cb6d549f3d4ac02e75` |
| `validation_retesting/` | 0 | 0 | `empty` |

The architecture comparison, prompt variations, and empty retesting subtree already have detailed source pages. This page records the directory-level synthesis and top-level summary files.

# Top-Level Summary Claims

`final_validation_summary.md` claims that comprehensive experimental testing solved the major architectural problems: property-graph bias, multi-theory papers, model-type selection, and hybrid processing. It reports 100% validation completion and 100% success across Heilman framing, Lofland-Stark conversion, Young 1996, and an OWL stress test. [2]

`optimization_findings.md` gives the proposed architecture: theory-count detection, parallel specialized detection or theory segmentation, hybrid detection/integration, and schema generation. It reports high confidence in theory-count detection, sequence detection, property-graph v2, hybrid detection, and hybrid schema generation, with table-matrix calibration as a minor remaining issue. [3]

# Existing Detail Pages

- [Lit Review Model Form Detection Results](/wiki/sources/lit-review-model-form-detection-results.md) covers the two optimized architecture-comparison files.
- [Lit Review Prompt Variation Model Routing](/wiki/sources/lit-review-prompt-variation-model-routing.md) covers theory-count, specialized detector, confidence integration, and hybrid prompt artifacts.
- [Lit Review Validation Retesting Empty Directories](/wiki/sources/lit-review-validation-retesting-empty.md) covers the empty retesting subtree.

# Interpretation

This directory is the strongest expression of the "optimized routing solves it" storyline. It should be preserved, but read with the existing caveats:

- top-level summaries are internal generated validation claims
- detailed pages preserve smaller inconsistencies and empty retesting evidence
- "100% accuracy" is over a small hand-picked validation set, not a broad external benchmark
- model-form routing and prompt calibration remain methodologically important even if final summaries are overconfident

# Links

- [Lit Review Model Form Detection Results](/wiki/sources/lit-review-model-form-detection-results.md)
- [Lit Review Prompt Variation Model Routing](/wiki/sources/lit-review-prompt-variation-model-routing.md)
- [Lit Review Validation Retesting Empty Directories](/wiki/sources/lit-review-validation-retesting-empty.md)
- [Prompt Calibration Loop](/wiki/concepts/prompt-calibration-loop.md)
- [Model Form Routing](/wiki/concepts/model-form-routing.md)
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/final_validation_summary.md`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/optimization_findings.md`
