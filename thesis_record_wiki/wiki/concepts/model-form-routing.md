---
type: Concept
title: Model Form Routing
description: Routing theory schemas to an appropriate representational form such as sequence, table, graph, statistical model, or hybrid rather than forcing every theory into one structure.
tags: [concept, model-form, routing, schema, theory-extraction]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/architecture_comparison/lofland_optimized_result.yml
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/architecture_comparison/heilman_optimized_result.yml
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/docs/N-ARY_RELATIONS_GUIDE.md
confidence: medium
---

# Summary

Model form routing is the idea that a theory should be represented in the form its own structure warrants: sequence, table/matrix, property graph, statistical/probabilistic model, or hybrid. The experimental model-form detection results preserve an early version of this routing idea.

# Examples

The Lofland-Stark result treats conversion theory as a sequential funnel. A graph or table could depict it, but the theory's main claim is an ordered accumulation of conditions. [1]

The framing result detects multiple theoretical components and Prospect Theory, but contains an internal inconsistency between its rationale and final `model_type`. That makes it evidence for the need for routing plus validation, not evidence that the router was solved. [2]

[Lit Review ELM Schema](/wiki/sources/lit-review-elm-schema.md) gives a positive simple case: the Elaboration Likelihood Model is represented as a compact sequential process with four entities, six relationships, and four analysis steps. This contrasts with the complex Young 1996 schema family and reinforces that representation should follow theory form.

The n-ary relations guide turns the same idea into a modeling decision procedure: use hypergraphs when n-ary relations are central and recursive, property graphs when binary relations dominate and occasional n-ary relations can be reified, and tables when fixed-arity row/column structure is natural. [3]

# Rule

Do not force every theory into the same representation just because the platform has a preferred graph format. First detect the theory's model form, then choose the schema/application path.

# Links

- [Lit Review Model Form Detection Results](/wiki/sources/lit-review-model-form-detection-results.md)
- [Lit Review ELM Schema](/wiki/sources/lit-review-elm-schema.md)
- [Lit Review Docs Bundle](/wiki/sources/lit-review-docs-bundle.md)
- [Formal Notation As Theory Content](/wiki/concepts/formal-notation-as-theory-content.md)
- [Schema Extraction Pipeline Evolution](/wiki/concepts/schema-extraction-pipeline-evolution.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/architecture_comparison/lofland_optimized_result.yml`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/experimental_testing/architecture_comparison/heilman_optimized_result.yml`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/docs/N-ARY_RELATIONS_GUIDE.md`
