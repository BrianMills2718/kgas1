---
type: Source
title: Lit Review Testing Code
description: Inventory and interpretation of preserved testing/debug scripts for schema processors, provider behavior, Semantic Hypergraph extraction, and multi-agent phase execution.
tags: [source, lit-review, code, testing, debug, prompt-calibration, multi-agent, openai]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/testing/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/testing/compare_processors.py
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/testing/experimental_processor_optimized.py
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/testing/test_computational_schema.py
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/testing/run_multi_agent_implementation.py
confidence: high
---

# Summary

The preserved `src/testing/` folder contains 14 Python scripts with aggregate hash `d384ee477f4bea21060134f919b61b4c9856ed6b62b4241c20b02f3da1e8d267`. [1]

This folder is not a conventional unit-test suite. It is a collection of historical debug scripts, provider checks, schema-comparison scripts, extraction tests, optimized-prompt processors, computational-schema execution tests, and multi-agent phase runners.

# Script Inventory

| Script | Hash prefix | Role | Flags |
|---|---|---|---|
| `compare_processors.py` | `4f58347102c8` | compares original and improved schema outputs by definition counts, categories, domain/range specificity, and feature presence | YAML |
| `debug_imports.py` | `ebf6d344dfbf` | import/debug check around provider stack | OpenAI |
| `debug_openai_connection.py` | `fb33efb116e` | OpenAI connection/debug check | OpenAI |
| `direct_process_test.py` | `404efae3e34a` | direct processor test | hardcoded local path |
| `enhance_young1996_schema.py` | `85a5e6115c73` | Young 1996 schema enhancement script | OpenAI, YAML, hardcoded local path |
| `experimental_processor_optimized.py` | `74e40fd48402` | optimized experimental processor combining theory-count, specialized, hybrid, and schema-generation prompts | OpenAI, YAML |
| `run_multi_agent_implementation.py` | `06f13e2b5f53` | runner for six-phase multi-agent implementation harness | hardcoded local path |
| `run_sh_extraction.py` | `a92a04c67d96` | Semantic Hypergraph extraction runner | hardcoded local path |
| `test_computational_schema.py` | `954a717e5995` | executes and validates a computational Young schema through OpenAI | OpenAI, YAML, hardcoded local path |
| `test_o3.py` | `d20cb7fa0db3` | simple O3/provider test | OpenAI |
| `test_o3_parse.py` | `c1669eae3fc3` | O3 parse/structured-output experiment | OpenAI, `json_object` |
| `test_semantic_hypergraph_extraction.py` | `563922b1644e` | three-phase Semantic Hypergraph extraction test | YAML, hardcoded local path |
| `test_sh_minimal.py` | `f09ffe99a0af` | minimal Semantic Hypergraph structured-output test | OpenAI, hardcoded local path |
| `trace_execution.py` | `29a92ccf7fc7` | execution tracing/timeout helper | hardcoded local path |

# Testing Threads

## Schema Comparison

`compare_processors.py` compares original and improved schema YAML files by total definitions, category shifts, non-generic domain/range pairs, subcategories, model type, modifiers, truth values, and operators. [2]

This is evidence for the [Schema Extraction Pipeline Evolution](/wiki/concepts/schema-extraction-pipeline-evolution.md) thread: improvements were evaluated partly by whether outputs preserved richer typed structure, not only by whether a YAML file existed.

## Prompt And Model Routing

`experimental_processor_optimized.py` implements an experimental pipeline with theory-count detection, specialized table/property-graph/sequence detectors, hybrid detection, hybrid schema generation, and decision-rule integration. [3]

This script operationalizes the [Prompt Calibration Loop](/wiki/concepts/prompt-calibration-loop.md) and [Model Form Routing](/wiki/concepts/model-form-routing.md) artifacts as runnable experimental code, though its confidence extraction is explicitly simplified.

## Computational Schema Execution

`test_computational_schema.py` loads a Young 1996 execution prompt, calls OpenAI to execute the computational schema, then calls OpenAI again to validate the result against Young standards such as relationship categories, salience scoring, structural measures, and truth values/modifiers. [4]

This preserves an important research aspiration: theory schemas were meant to become executable analytical procedures, not only documentation.

## Multi-Agent Harness Runner

`run_multi_agent_implementation.py` wraps `AutoPhaseManager` and runs six phases: purpose classification, vocabulary extraction, schema generation, integration pipeline, reasoning engine, and production validation. It prints a hard 100/100 phase-gate requirement and equal sophistication across descriptive, explanatory, predictive, causal, and intervention purposes. [5]

# Caveats

- Most scripts are ad hoc historical runners, not maintained tests with assertions.
- Several scripts depend on historical working directories and local files.
- Direct OpenAI usage is common; this predates the later shared-client discipline.
- Some scripts use emoji-rich console output and broad exception reporting, so they should not be treated as current fail-loud infrastructure.
- Provider behavior experiments are useful provenance, but current model/API behavior should be rechecked before drawing present-day conclusions.

# Links

- [Lit Review Src Code Inventory](/wiki/sources/lit-review-src-code-inventory.md)
- [Lit Review Prompt Variation Model Routing](/wiki/sources/lit-review-prompt-variation-model-routing.md)
- [Prompt Calibration Loop](/wiki/concepts/prompt-calibration-loop.md)
- [Model Form Routing](/wiki/concepts/model-form-routing.md)
- [Multi Agent Evidence Harness](/wiki/concepts/multi-agent-evidence-harness.md)
- [Lit Review Semantic Hypergraph Schema Family](/wiki/sources/lit-review-semantic-hypergraph-schema-family.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/testing/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/testing/compare_processors.py`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/testing/experimental_processor_optimized.py`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/testing/test_computational_schema.py`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/testing/run_multi_agent_implementation.py`
