---
type: Source
title: Lit Review Universal Theory Applicator
description: Universal theory applicator framework, critique, schema enhancement pattern, and Young 1996 application result from the lit-review experiment.
tags: [source, lit-review, theory-application, universal-applicator, schema]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_application/universal_theory_applicator.py
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_application/test_and_critique_universal.py
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_application/test_universal_applicator.py
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/docs/UNIVERSAL_APPLICATOR_CRITIQUE.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/docs/SCHEMA_ENHANCEMENT_TEMPLATE.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/young1996/young1996_enhanced.yml
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/results/young_universal_test.yml
confidence: high
---

# Summary

The universal theory applicator is the lit-review experiment's attempt to generalize theory application across schemas. Instead of writing a custom program for every theory, it uses schema-defined stages, prompt templates, output mappings, and summary rules.

It is an important bridge from "extract a theory schema" to "apply any theory schema to empirical text." The preserved critique is candid: the framework generalizes the control flow, but complexity moves into schema design, prompt engineering, validation, and theory-specific quality metrics.

# Framework Shape

`universal_theory_applicator.py` defines:

- `StageResult`
- `TheoryApplication`
- `UniversalTheoryApplicator`
- schema-driven `application_stages`
- prompt template variables such as `{text}`, `{domain}`, `{node_types}`, and `{previous.stage_name}`
- post-processing rules for filtering and simple transforms
- output mapping from stage outputs to final theory-specific structures
- summary rules such as counts and connectedness [1]

The default stage pattern is extraction -> filtering -> structuring -> analysis, but the critique notes that other stage names and orders can work because stage behavior is schema-configured. [1][4]

# Critique

The dedicated critique says the applicator is truly theory-agnostic: no theory-specific logic is embedded in the applicator code, and all theory knowledge sits in schemas. Its strengths are flexible stages, rich template variables, post-processing, and output mapping. [4]

The weaknesses are equally important:

- no output validation
- only linear processing
- no branching, loops, or iterative refinement
- error propagation across stages
- quality depends heavily on schema prompt design
- no built-in quality metrics
- inefficient context use for long documents [4]

The critique's central conclusion is that complexity is conserved, not eliminated. The work shifts from theory-specific code to theory-specific schema design, prompt engineering, output validation, and quality metrics. [4]

# Schema Enhancement Pattern

`SCHEMA_ENHANCEMENT_TEMPLATE.md` defines how a theory schema should be enhanced for the applicator:

- add `theory_name`
- define `application_stages`
- map outputs back into schema structures
- define `summary_rules`
- use stage variables and previous-stage outputs
- specify post-processing rules and metadata fields [5]

It gives two major stage patterns:

- extract -> filter -> structure -> analyze for entity/relationship theories
- type -> build -> compose for formal systems [5]

# Young 1996 Example

The enhanced Young 1996 schema adds application stages for extraction, filtering, structuring, and analysis. It includes nodes, connections, properties, modifiers, filtering criteria, prompt templates, and output mapping. [6]

The preserved `young_universal_test.yml` result shows a concrete application to Carter-style diplomatic text. It extracted concepts such as United States, Soviet Union, SALT talks, arms control, nuclear arsenals, US-Soviet cooperation, and global stability, then produced relationships such as cooperate, oppose, support, possess, and negotiate with truth values, modifiers, salience, and evidence snippets. [7]

No preserved `sh_universal_test.yml` was found in this source tree during this slice. The test script contains an SH test path, but the corresponding result file was absent. [2]

# Current Caveats

By current ecosystem standards, this is historical prototype code:

- uses direct `OpenAI()` client rather than shared `llm_client`
- uses `response_format={"type": "json_object"}` rather than `json_schema`
- trusts JSON output without Pydantic validation per stage
- uses `/tmp` in test scaffolding and hardcoded local paths in related application scripts
- has no durable observability, trace IDs, or budget controls [1][3]

These caveats do not erase the architectural value. The preserved work captures a real transition from one-off theory application scripts toward schema-driven, agent-drivable theory application.

# Links

- [Complexity Conservation In Theory Application](/wiki/concepts/complexity-conservation-in-theory-application.md)
- [Automated Theory Extraction](/wiki/concepts/automated-theory-extraction.md)
- [Formal Notation As Theory Content](/wiki/concepts/formal-notation-as-theory-content.md)
- [Lit Review Semantic Hypergraph Application Results](/wiki/sources/lit-review-semantic-hypergraph-application-results.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_application/universal_theory_applicator.py`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_application/test_and_critique_universal.py`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/schema_application/test_universal_applicator.py`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/docs/UNIVERSAL_APPLICATOR_CRITIQUE.md`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/docs/SCHEMA_ENHANCEMENT_TEMPLATE.md`  
[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/schemas/young1996/young1996_enhanced.yml`  
[7] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/results/young_universal_test.yml`
