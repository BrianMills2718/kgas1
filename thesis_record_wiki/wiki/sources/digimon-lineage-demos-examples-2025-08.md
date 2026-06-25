---
type: SourceSummary
title: Digimon Lineage Demos Examples 2025 08
description: Small August 2025 demo/examples archive preserving ontology UI, bi-store database demo, reliability demonstration, direct Gemini validation scripts, natural-language-to-DAG proof, sample documents, and a multimodal output JSON.
tags: [source, digimon-lineage, archive, demos, examples, streamlit, ontology, neo4j, sqlite, gemini, dag, multimodal]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/demos_examples_2025_08/
confidence: high
---

# Summary

`archive/demos_examples_2025_08/` is a 13-file, 63,774-byte archive of small KGAS demos and examples. Its aggregate content-manifest hash is `c10d2afebaa6569dbae2bba72778cd06c825cb3b61d1c7c5580eaa0be6bdc550`. [1]

This slice overlaps with the larger scripts archive, but it is worth preserving separately because it contains a compact Streamlit ontology UI, direct validation scripts, a two-database transaction demo, a final natural-language-to-DAG proof script, two sample documents, and a preserved multimodal output JSON. [1]

# Inventory

| File | Role |
| --- | --- |
| `streamlit_app.py` | Streamlit UI for academic ontology generation through conversation. [2] |
| `demo_both_databases.py` / `demo_real_neo4j_data.py` | Neo4j/SQLite database demos. [3] |
| `demonstrate_reliability_fixes.py` | Reliability demonstration for audit immutability, performance tracking, and SLA monitoring. [4] |
| `direct_final_validation.py`, `direct_gemini_validation.py`, `direct_t57_validation.py` | Direct Gemini validation scripts. [5] |
| `final_working_test.py` | Natural-language-to-DAG-to-real-tool CLI proof script. [6] |
| `simple_yaml_test.py` | Workflow-agent YAML generation/truncation probe. [7] |
| `fix_provenance_calls.py` | Small provenance API repair script. [8] |
| `genuine_dag_demo_document.txt`, `multimodal_demo_document_real.txt` | Sample input documents for demos. [9] |
| `multimodal_demo_outputs.json` | Preserved small multimodal output summary. [10] |

# Ontology Streamlit UI

`streamlit_app.py` implements an academic-focused ontology generation UI. It imports a Gemini ontology generator when available, falls back when imports/configuration fail, defines ontology dataclasses, stores chat/session state, and renders ontology previews with Streamlit, pandas, Plotly, and NetworkX. [2]

This artifact connects to the broader ontology-generation and schema-extraction thread: it is a user-facing review/generation surface, not just backend code.

# Database And Reliability Demos

`demo_both_databases.py` demonstrates a distributed transaction manager coordinating Neo4j and SQLite operations. It creates entities in both databases, verifies both stores, simulates a failed SQLite operation, rolls back, and checks that the failed entity does not exist. It depends on a local Neo4j service and demo credentials. [3]

`demonstrate_reliability_fixes.py` initializes provenance, performance tracking, and SLA monitoring components; creates/modifies a citation; verifies audit trail integrity; times demo operations; and shows an SLA violation path. [4]

# Validation And NL-To-DAG

The direct validation scripts use Gemini to validate implementation claims. `direct_t57_validation.py` reads a `repomix-t57.xml` bundle and asks Gemini to assess path-analysis claims such as Dijkstra/BFS/Bellman-Ford, all-pairs paths, flow analysis, reachability, test coverage, unified BaseTool interface, and academic-ready output. [5]

`final_working_test.py` is a proof-style script for the natural-language-to-DAG-to-execution path. It creates a Carter Center text file and runs `scripts/ask_fixed.py` through a subprocess, declaring success if the command exits cleanly and the output contains a success marker. [6]

`simple_yaml_test.py` isolates YAML generation and truncation by asking the workflow agent to generate a simple PDF-loading workflow, then parsing the returned YAML. [7]

# Preserved Output

`multimodal_demo_outputs.json` records a small output summary: 14 graph entities, 9 graph relationships, 9 table rows, 384 vector dimensions, top PageRank entries headed by Apple Inc., and an execution time of about 0.043 seconds. [10]

This is a tiny generated artifact, not broad benchmark evidence, but it is useful as a concrete example of the graph/table/vector demo shape.

# Caveats

Several files depend on local services, live provider keys, generated repomix inputs, or historical script paths. The validation scripts are prompts to external validators rather than captured validator reports unless paired with saved outputs elsewhere. [5]

The targeted credential scan found no literal OpenAI or Google API keys in this archive. [1]

# Interpretation

This directory is a compact user-facing demo and validation leftover bundle. It is lower priority than the major architecture/evidence archives, but it preserves how KGAS was being shown: ontology generation UI, bi-store transactions, reliability instrumentation, direct claim validation, natural-language-to-DAG execution, and graph/table/vector multimodal summaries.

# Relationship To Wiki

- [Digimon Lineage Scripts Archive 2025 08](digimon-lineage-scripts-archive-2025-08.md): larger overlapping scripts archive.
- [Digimon Lineage Recovered UI Demo Surface](../concepts/recovered-ui-demo-surface.md): broader preserved UI/demo surface concept.
- [Digimon Lineage Generated Outputs 2025 08](digimon-lineage-generated-outputs-2025-08.md): larger generated-output archive.
- [Digimon Lineage Analysis Validation 2025 08](digimon-lineage-analysis-validation-2025-08.md): validation configs and direct validator scripts.
- [Digimon Lineage Archived Experimental Tests](digimon-lineage-archived-experimental-tests.md): related test/proof archive.
- [Model Form Routing](../concepts/model-form-routing.md): related graph/table/vector multimodal pattern.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/demos_examples_2025_08/`

[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/demos_examples_2025_08/streamlit_app.py`

[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/demos_examples_2025_08/demo_both_databases.py`

[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/demos_examples_2025_08/demonstrate_reliability_fixes.py`

[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/demos_examples_2025_08/direct_t57_validation.py`

[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/demos_examples_2025_08/final_working_test.py`

[7] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/demos_examples_2025_08/simple_yaml_test.py`

[8] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/demos_examples_2025_08/fix_provenance_calls.py`

[9] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/demos_examples_2025_08/genuine_dag_demo_document.txt`

[10] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/demos_examples_2025_08/multimodal_demo_outputs.json`
