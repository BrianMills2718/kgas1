---
type: Source
title: Lit Review Root Files
description: Inventory and interpretation of top-level files in the preserved lit_review experiment root, including overview docs, prompts, logs, standalone papers, and environment files.
tags: [source, lit-review, root-files, prompts, logs, security, provenance]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/PROJECT_OVERVIEW.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/CLAUDE.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/meta_schema_8_prompt.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/stress_test_visualization.md
confidence: high
---

# Summary

The top level of `experiments/lit_review/` contains 22 files with aggregate hash `20d7bad3fe203a23f27dee1f46625d232038b67cec78bf49834c9dd0e0a266d8`. [1]

These files mix project framing, active-agent instructions, prompt drafts, extraction logs, source texts, standalone runners, generated results, environment/config files, and accidental package-install output artifacts.

# Root File Inventory

| File | Hash prefix | Role |
|---|---|---|
| `.env` | `67284c084d06` | Sensitive environment file; value not reproduced in this wiki. |
| `.env.example` | `21b53de3aa31` | Example env file documenting `OPENAI_API_KEY` and fixed `OPENAI_MODEL=o3`. |
| `.gitignore` | `74f13dd8d757` | Ignores envs, caches, logs, temp files, UI env, analysis result JSON, and other generated material. |
| `=0.3.0` | `02007ccf8c01` | Captured package-install output, likely accidental root artifact. |
| `=1.35.0` | `e3b0c44298fc` | Empty accidental root artifact. |
| `CLAUDE.md` | `cc6cec7a7ab0` | Active local agent instructions for the lit-review experiment. |
| `PROJECT_OVERVIEW.md` | `45f7ca676efd` | Strategic project overview and architecture/methodology summary. |
| `complete_application.log` | `2431142dae9a` | WorldView complete-schema application log over Carter speech. |
| `event graphs: modeling reality's temporal and causal fabric` | `ac2a5b5a5356` | Standalone event-graphs source text. |
| `example_ses_explanation.md` | `500efdb45f9c` | Social-Ecological Systems hybrid-schema explanation. |
| `extract_turner_1986.py` | `1fb7f2fe4e9` | Root-level Turner/Oakes extraction script. |
| `extraction_log.txt` | `7551415a8152` | Log for processing Young 1996 cognitive-mapping source. |
| `grusch_analysis_results.json` | `e15c9d4a5df9` | Grusch/UAP information-disorder analysis output. |
| `iterative_extraction.log` | `6b623233aacf` | Iterative extraction log for Young 1996. |
| `meta_schema_8_prompt.md` | `9a9ef3377d2f` | Meta-schema v8 extraction prompt. |
| `multipass_extraction.log` | `53223ada2279` | Multi-pass extraction log for Young 1996. |
| `prompt_2025.06280355.txt` | `6bcd4bd2cbbe` | Root prompt draft for academic theory extraction. |
| `prompt_2025.06280355txt` | `e0a18147c288` | Extensionless/variant prompt draft. |
| `prompt_enhanced_2024.txt` | `d078c12c38d0` | Enhanced root prompt draft. |
| `run_semantic_hypergraph_extraction.py` | `26fb69aef43e` | Standalone Semantic Hypergraph extraction runner. |
| `semantic_hypergraphs.txt` | `c9d2b6454a0c` | Semantic Hypergraphs paper text. |
| `socio-semantic networks.txt` | `5319bf4316ab` | Socio-Semantic Networks paper text. |
| `stress_test_visualization.md` | `0b4b5ddf4a1c` | Synthetic adaptive-governance/hybrid-schema visualization. |

# Project Framing

`PROJECT_OVERVIEW.md` states the strategic goal as an automated system that reads academic papers, extracts theoretical constructs, and instantiates those constructs with data to produce results comparable to the original paper's methodology. It describes the core workflow as academic paper to schema extraction to data application to comparable results. [2]

The same overview records recent findings that became recurring wiki themes: information loss between phases, notation systems as critical for Semantic Hypergraphs, implementation details requiring multi-pass extraction, and explicit hypergraph support. [2]

`CLAUDE.md` is the local agent operating contract. It requires new code under `src/`, schemas under `schemas/{theory_name}/`, results under `results/{theory_name}/`, no loose Python files in the root, full-document application in one shot, no truncating application input, and O3 as the configured model. [3]

# Prompt And Stress-Test Artifacts

`meta_schema_8_prompt.md` asks for high-fidelity extraction into a meta-schema with model-type classification, reasoning-engine selection, compatible operators, entities, connections, properties, modifiers, optional axioms/analytics/process, and telos. [4]

`stress_test_visualization.md` is a synthetic adaptive-governance example combining belief networks, institutional payoff matrices, governance adaptation sequences, collective-intelligence statistical models, and normative logic. It is useful as a stress case for hybrid representation design. [5]

# Logs And Generated Root Outputs

The root logs preserve intermediate execution traces:

- `complete_application.log` reports a complete WorldView application over Carter material, including belief extraction, semantic network analysis, structural measures, and success-criteria assessment.
- `extraction_log.txt`, `iterative_extraction.log`, and `multipass_extraction.log` preserve attempts to process the Young 1996 cognitive-mapping source.

These are historical execution traces, not current reproducibility proof.

# Security And Cleanup Concerns

The preserved `.env` appears to contain a real API key or secret-like credential. The value is intentionally not reproduced here. Before any public sharing, this credential should be assumed compromised and rotated, and any shareable archive should exclude or redact the raw `.env`. The raw archive was not modified during this wiki ingest. [1]

The root also contains accidental-looking files named `=0.3.0` and `=1.35.0`. The first contains package-install output; the second is empty. They should be treated as clutter/provenance rather than intellectually meaningful thesis material.

# Links

- [Lit Review Theory Extraction Experiment](/wiki/sources/lit-review-theory-extraction-experiment.md)
- [Automated Theory Extraction](/wiki/concepts/automated-theory-extraction.md)
- [Schema Extraction Pipeline Evolution](/wiki/concepts/schema-extraction-pipeline-evolution.md)
- [Formal Notation As Theory Content](/wiki/concepts/formal-notation-as-theory-content.md)
- [Model Form Routing](/wiki/concepts/model-form-routing.md)
- [Information Disorder Application Artifact](/wiki/concepts/information-disorder-application-artifact.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/PROJECT_OVERVIEW.md`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/CLAUDE.md`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/meta_schema_8_prompt.md`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/stress_test_visualization.md`
