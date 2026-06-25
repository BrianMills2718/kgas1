---
type: Source
title: Digimon Lineage Old Backups Results
description: Nine-file old-backups results slice with provenance traces, Phase D success report, validation failures, performance outputs, and tool-interface audit.
tags: [source, old-backups, results, validation, provenance, tool-interface]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/results/
confidence: high
---

# Summary

`archive/old_backups_2025_08/results/` is a compact output-artifact slice inside the large Digimons lineage: 9 files, 36,491 bytes, aggregate SHA-256 `3f88b07fc82e7f88cbc83cf33b1e28b061e7620fbaccef6acc1790245fa60d36`.

It preserves mixed historical evidence from August 2025: successful provenance traces and a 7/7 Phase D integration report alongside a 0/6 validation report, a failed end-to-end workflow, an unhealthy production-readiness result, and a tool-interface audit showing 8 compliant tools out of 36. [1][2][3][4][5]

# Inventory

| File | Role |
| --- | --- |
| `agent_dag_provenance.json` | Seven-operation agent-driven DAG provenance trace. |
| `multi_doc_provenance.json` | Twenty-one-operation multi-document provenance trace. |
| `Evidence_Phase_D_Integration.json` | Phase D integration report claiming 7/7 tests passed. |
| `validation_results.json` | Six-check validation report with 0/6 passing. |
| `test_results_end_to_end.json` | End-to-end workflow result with document processed but analysis failed. |
| `test_results_production_readiness.json` | Production-readiness environment/system-health result. |
| `Evidence_Performance_Benchmarks.json` | Performance benchmark output with chunking, entity extraction, PageRank, and memory metrics. |
| `performance_results.json` | Smaller performance result with entity extraction, Neo4j, and tool-execution status. |
| `tool_interface_audit_results.txt` | Detailed tool-interface compliance audit over 36 tools. |

# Positive Evidence

The provenance traces are concrete workflow records. `agent_dag_provenance.json` records seven completed operations using `T01_PDF_LOADER`, `T15A_TEXT_CHUNKER`, `T27_RELATIONSHIP_EXTRACTOR`, `TEMPORAL`, `CLUSTERING`, `T31_ENTITY_BUILDER`, and `T68_PAGERANK`. `multi_doc_provenance.json` records 21 completed operations across three test documents, including entity extraction, relationship extraction, temporal processing, clustering, cross-modal linking, graph/edge building, PageRank, and multihop query. [1][2]

`Evidence_Phase_D_Integration.json` reports 7/7 tests passed and 100.0% success for integration points including entity resolution plus batch processing, dashboard visualization, cross-document visualization, streaming checkpoint recovery, enhanced engine pipeline, end-to-end workflow, and GraphRAGUI plus dashboard. [3]

The performance outputs preserve concrete benchmark numbers, including LLM extraction runs taking roughly 6.9-9.9 seconds for short texts in one benchmark, and a later small benchmark reporting entity-extraction average time around 0.006 seconds plus Neo4j average operation time around 0.0065 seconds. [6][7]

# Negative Evidence And Caveats

The validation report is strongly negative: 0 of 6 checks passed. Failures include missing or changed tool-registry import, absent `ServiceManager.health_check`, entity extraction returning an object without `.get`, Neo4j config being non-subscriptable, missing API-client model-listing method, and missing `ToolRegistry.tools`. [4]

The end-to-end result also failed. It says the document was processed, but zero entities and zero relationships were extracted, graph analysis failed, zero insights were generated, and the error was "Connectivity is undefined for the null graph." [5]

The production-readiness result is mixed: Neo4j and SQLite connection checks passed, service manager and monitoring were available, and required credential variables were present, but the overall health was false because of a 0.0% success rate. The stored result mentions credential presence only; do not reproduce or export secrets from the preserved archive. [8]

The tool-interface audit shows the interface-migration problem sharply. Eight legacy `src.tools.phase1` tools were compliant, but the total audit found only 8 compliant tools out of 36, a 22.2% compliance rate. Many unified tool class imports failed, and some cross-modal/phase3 tools did not match the expected `BaseTool`/`ToolRequest` contract. [9]

# Interpretation

This slice is good evidence that KGAS/Digimons had real provenance and reporting machinery, but it should not be read as system success. The preserved outputs are internally contradictory at the claim level:

- one integration report says 7/7 passed
- one validation report says 0/6 passed
- one end-to-end workflow fails with a null graph
- one production-readiness report says environment pieces are present but overall health is false
- one interface audit says only 22.2% of audited tools comply

This belongs with [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md), [Current Status Verification Discipline](/wiki/concepts/current-status-verification-discipline.md), and [Layered Tool Interface Architecture](/wiki/concepts/layered-tool-interface-architecture.md).

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/results/agent_dag_provenance.json`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/results/multi_doc_provenance.json`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/results/Evidence_Phase_D_Integration.json`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/results/validation_results.json`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/results/test_results_end_to_end.json`  
[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/results/Evidence_Performance_Benchmarks.json`  
[7] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/results/performance_results.json`  
[8] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/results/test_results_production_readiness.json`  
[9] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/results/tool_interface_audit_results.txt`
