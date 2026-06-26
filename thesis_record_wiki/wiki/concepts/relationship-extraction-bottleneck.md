---
type: Concept
title: Relationship Extraction Bottleneck
description: Stress-test evidence that KGAS could extract entities while failing to extract relationships.
tags: [bottleneck, relationship-extraction, graph-construction, evidence]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_core_sparse/ARCHITECTURAL_BOTTLENECKS_ANALYSIS.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/evidence_reports_2025_08/ARCHITECTURAL_BOTTLENECKS_ANALYSIS.md
  - ../src/orchestration/agents/analysis_agent.py
  - ../src/tools/phase1/t23a_spacy_ner_unified.py
  - ../src/tools/phase1/t27_relationship_extractor_unified.py
  - ../src/tools/compatibility/t27_adapter.py
  - ../src/analytics/complete_pipeline.py
  - ../src/tools/phase1/phase1_mcp_tools.py
  - ../src/orchestration/real_dag_orchestrator.py
  - ../tests/current_runtime/test_analysis_agent_t27_contract.py
  - ../tests/current_runtime/test_real_dag_t27_dataflow.py
  - ../requirements.txt
  - ../tests/current_runtime/test_spacy_model_dependency.py
confidence: high
---

# Summary

The relationship extraction bottleneck is one of the clearest technical warnings in the preserved record. The architectural bottleneck analysis says a multi-document stress test processed 25 documents and extracted 398 entities, but found zero relationships. [1]

The same warning appears in the large Digimons lineage August 2025 evidence-report bundle, not only in the sparse variant. That makes it a repeated lineage-level caveat rather than an isolated copy of old sparse-variant documentation. [2]

# Why It Matters

For a GraphRAG system, entity extraction alone is insufficient. If relationship extraction is missing, failing silently, or not invoked, graph construction becomes entity-only and the system cannot deliver the intended graph-based reasoning layer.

# Recorded Root Cause Hypothesis

The bottleneck analysis says relationship extraction tool T27 was either not called in the multi-document pipeline or failed silently. It recommends auditing T27 invocation, adding relationship extraction to multi-document tests, and validating with a simple two-document case. [1]

# Current Runtime Follow-Up

A 2026-06-25 current-code investigation reproduced a more specific version of this failure mode. Current T27 requires each entity to have `text`, `entity_type`, `start`, and `end`, while current T23A emits `surface_form`, `entity_type`, `start_pos`, and `end_pos`. The analysis-agent path was forwarding T23A entities directly into MCP relationship extraction, so it could fail before extraction rather than produce relationships. [3][4][5]

The analysis-agent, complete-pipeline, and Phase 1 MCP boundaries now normalize T23A output to the current T27 contract before calling T27. Focused current-runtime tests cover conversion, pass-through for already-normalized T27 entities, fail-loud behavior for unknown entity shapes, analysis-agent propagation, and complete-pipeline propagation. A direct T27 fixture probe returned two relationships for both native T27 entities and converted T23A entities. [3][6][7][8][9]

This does not close every relationship-extraction risk. `en-core-web-sm==3.8.0` is now installed, declared in `requirements.txt`, and covered by a runtime test confirming the parser component is available, but the successful T27 fixture evidence still comes from pattern extraction. The real DAG now passes upstream entities into T27, so the remaining current-code caveat is narrower: parser-derived relationship output needs a richer fixture if it becomes an explicit target. [10][11][12][13]

# Related Pages

- [Digimon Core Sparse Contract Layer](/wiki/sources/digimon-core-sparse-contract-layer.md)
- [Digimon Lineage Evidence Reports 2025 08](/wiki/sources/digimon-lineage-evidence-reports-2025-08.md)
- [Current Runtime Import Check 2026-06-25](/wiki/sources/current-runtime-import-check-2026-06-25.md)
- [Contract-First Migration](/wiki/concepts/contract-first-migration.md)
- [KGAS](/wiki/entities/kgas.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_core_sparse/ARCHITECTURAL_BOTTLENECKS_ANALYSIS.md`
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/evidence_reports_2025_08/ARCHITECTURAL_BOTTLENECKS_ANALYSIS.md`
[3] `../src/orchestration/agents/analysis_agent.py`
[4] `../src/tools/phase1/t23a_spacy_ner_unified.py`
[5] `../src/tools/phase1/t27_relationship_extractor_unified.py`
[6] `../src/tools/compatibility/t27_adapter.py`
[7] `../src/analytics/complete_pipeline.py`
[8] `../src/tools/phase1/phase1_mcp_tools.py`
[9] `../tests/current_runtime/test_analysis_agent_t27_contract.py`
[10] `../src/orchestration/real_dag_orchestrator.py`
[11] `../tests/current_runtime/test_real_dag_t27_dataflow.py`
[12] `../requirements.txt`
[13] `../tests/current_runtime/test_spacy_model_dependency.py`
