---
type: Source
title: Lit Review Schema Application Code
description: Inventory and interpretation of schema_application code that applies theory schemas to Carter, Young/WorldView, Semantic Hypergraph, information disorder, and universal-applicator workflows.
tags: [source, lit-review, code, schema-application, universal-applicator, openai, hardcoded-paths]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_application/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_application/universal_theory_applicator.py
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_application/test_and_critique_universal.py
confidence: high
---

# Summary

The preserved `src/schema_application/` folder contains 24 Python scripts with aggregate hash `acd2f93622227c46c5cb88883fc5318b345364111484e40fe0a126f3b3ee6307`. [1]

This folder is the code bridge from generated theory schemas to actual text analysis. It applies Young/WorldView, Semantic Hypergraph, Carter speech analysis, information-disorder extraction, and the universal theory applicator.

# Structural Facts

- 24 scripts have a `__main__` guard.
- 17 scripts use direct OpenAI/OpenAI-like client calls.
- 16 scripts contain hardcoded `/home/brian/...` paths.
- Several scripts write to historical `results/` and `carter_analysis_output/` paths. [1]

These facts make the folder strong historical evidence but weak current infrastructure without migration.

# Universal Applicator

`universal_theory_applicator.py` defines:

- `StageResult`
- `TheoryApplication`
- `UniversalTheoryApplicator`
- `save_application_result`

It loads a theory schema, obtains `application_stages` or defaults to extraction/filtering/structuring/analysis, builds per-stage prompts, calls OpenAI using `response_format={"type": "json_object"}`, post-processes stage outputs, and saves a final theory-application result. [2]

This is the clearest implementation of the schema-driven application idea described in [Lit Review Universal Theory Applicator](/wiki/sources/lit-review-universal-theory-applicator.md).

# Test And Critique Harness

`test_and_critique_universal.py` applies:

- Young enhanced schema to Carter diplomatic speech
- Semantic Hypergraph enhanced schema to an Iran debate excerpt

It saves outputs to `results/young_universal_test.yml` and `results/sh_universal_test.yml`, then critiques extracted concepts, relationships, connectedness, type distribution, and overall strengths/weaknesses. [3]

# Application Families

The folder contains distinct application families:

- Young/WorldView Carter scripts: `apply_young*`, `apply_worldview*`, `process_carter_speech.py`, and Carter demo/analysis scripts.
- Semantic Hypergraph scripts: `apply_sh*`, `apply_semantic_hypergraph_to_iran.py`, Ground News and Iran debate variants.
- Universal applicator scripts: `universal_theory_applicator.py`, `test_universal_applicator.py`, `test_and_critique_universal.py`.
- Generic schema extraction: `extract_with_schema.py`.
- Result parsing and critique: `parse_enhanced_results.py`, `test_improved_instantiation.py`.

# Caveats

The code is historically valuable but not directly production-ready:

- direct OpenAI usage rather than shared `llm_client`
- `json_object` rather than enforced JSON Schema structured output
- hardcoded `/home/brian/lit_review` paths in many scripts
- scripts assume historical local result/schema/text locations

This should be treated as executable design evidence and migration source material, not a clean maintained application layer.

# Links

- [Lit Review Universal Theory Applicator](/wiki/sources/lit-review-universal-theory-applicator.md)
- [Complexity Conservation In Theory Application](/wiki/concepts/complexity-conservation-in-theory-application.md)
- [Lit Review Young1996 Schema Family](/wiki/sources/lit-review-young1996-schema-family.md)
- [Lit Review Semantic Hypergraph Application Results](/wiki/sources/lit-review-semantic-hypergraph-application-results.md)
- [Lit Review Src Code Inventory](/wiki/sources/lit-review-src-code-inventory.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_application/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_application/universal_theory_applicator.py`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_application/test_and_critique_universal.py`
