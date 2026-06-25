---
type: Source
title: Lit Review Legacy System Framing
description: Archived legacy overview, post-processing guide, project overview, and CLAUDE notes explaining the early Analyst/Assembler methodology and later prompt-testing agenda.
tags: [source, lit-review, legacy-system, methodology, project-overview]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/archive/legacy_system/overview.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/archive/legacy_system/post_processing.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/archive/legacy_system/prompt.txt
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/archive/project_overviews/project-Overview_2025.06270119.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/archive/claude_md_archive/CLAUDE_2025.0846.md
confidence: high
---

# Summary

This slice captures the archived conceptual framing for the lit-review experiment before the later validation, prompt-calibration, and multi-agent phases.

The legacy overview defines the strategic goal as converting 100+ social and behavioral science papers into a unified, computationally analyzable knowledge base. It frames the problem as translating nuanced qualitative theories into formal, interoperable structures while preserving high fidelity. [1]

# Analyst And Assembler

The early methodology is a two-stage "Analyst & Assembler" workflow:

- LLM as analyst: read the academic paper, interpret the theory, and produce a simple YAML schema blueprint.
- deterministic script as assembler: read the blueprint, select the correct top-level JSON Schema shape by `model_type`, create categorized definitions, and inject universal CORE/shared definitions. [1]

This anticipates later separation-of-concerns lessons: LLMs handle semantic interpretation, while deterministic code handles repetitive assembly and validity.

# Post-Processing

The post-processing guide narrows the assembler task to injecting universal boilerplate definitions into `$defs`, including `Entity`, `Role`, `NaryTuple`, `Argument`, and `sharedProps`. It includes a Python script that loads the AI YAML output, loads CORE/shared JSON definitions, merges them into `$defs`, and writes a processed YAML artifact. [2]

This is an early version of the common-ontology/interoperability strategy that later appears in the project overview and schema-evolution work.

# Project Overview

The archived project overview expands the goal into computational social science theory modeling:

- descriptive categorization
- explanatory analysis
- predictive forecasting
- causal inference
- intervention design
- multi-paradigm representations: property graph, table/matrix, sequence, tree, timeline, statistical, logical, causal, and hybrid [4]

It defines a three-phase processing pipeline: vocabulary extraction, ontological classification, and theory-adaptive schema generation. [4]

# Archived CLAUDE Notes

The archived CLAUDE notes show the system after validation had already found critical problems:

- property-graph bias
- multi-theory papers incorrectly unified into single schemas
- inverse complexity/accuracy relationship
- missing specialized detection prompts
- need to test theory-count detection, specialized model detection, hybrid processing, and sequential-vs-parallel architectures [5]

This connects directly to [Lit Review Prompt Variation Model Routing](/wiki/sources/lit-review-prompt-variation-model-routing.md), which preserves the later prompt experiments and calibration loop.

# Interpretation

The legacy system framing shows continuity rather than random exploration:

1. start with high-fidelity theory-to-schema conversion
2. split LLM interpretation from deterministic assembly
3. discover property-graph bias and multi-theory failure
4. add specialized model routing, hybrid detection, and prompt calibration
5. move toward universal schema-driven application

# Links

- [Analyst Assembler Pattern](/wiki/concepts/analyst-assembler-pattern.md)
- [Schema Extraction Pipeline Evolution](/wiki/concepts/schema-extraction-pipeline-evolution.md)
- [Prompt Calibration Loop](/wiki/concepts/prompt-calibration-loop.md)
- [Model Form Routing](/wiki/concepts/model-form-routing.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/archive/legacy_system/overview.md`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/archive/legacy_system/post_processing.md`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/archive/legacy_system/prompt.txt`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/archive/project_overviews/project-Overview_2025.06270119.md`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/archive/claude_md_archive/CLAUDE_2025.0846.md`
