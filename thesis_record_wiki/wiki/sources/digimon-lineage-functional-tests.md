---
type: Source
title: Digimon Lineage Functional Tests
description: Focused inventory of the preserved tests/functional directory, including user-workflow test intent, MCP/UI/phase coverage, no-mocks policy tests, and mock/runtime caveats.
tags: [source, digimon-lineage, tests, functional, user-workflows, mcp, ui, no-mocks, runtime-status]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/functional/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/functional/CLAUDE.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/functional/test_no_mocks_policy.py
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/functional/test_tools_functional_real.py
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/functional/test_ui_complete.py
confidence: medium
---

# Summary

`tests/functional/` is a 46-file, 573,594-byte functional-test corpus with local aggregate hash `70ed1803fde0155d785ca524720842ffe00dbb13095fdb209f22cd0ad4d882da`. [1]

The directory-level instructions define functional tests as user-perspective tests for end-to-end workflows, UI behavior, MCP tool behavior, realistic research scenarios, quality, reproducibility, and performance. [2]

The actual preserved directory contains 44 root `test_*.py` files plus mirrored `CLAUDE.md` / `AGENTS.md` instruction files. `AGENTS.md` is a symlink to `CLAUDE.md`, so it is an agent-surface mirror rather than separate policy content. [1]

# Inventory

| Area | Files | Bytes | Aggregate hash | Role |
|---|---:|---:|---|---|
| root test files | 44 | 511,326 | `a532a30a99095e27528b154dea7c0fc4841f51736c93e57f95a69e84ef781cda` | functional tests for UI, MCP, phases, GraphRAG workflows, environment/config, response generation, dashboards, and no-mocks behavior |
| root metadata/files | 2 | 62,268 | `f72f57a718ef6f002b71206906e3db37a37eda60bce6309bd6a6afd56e28ab51` | `CLAUDE.md` plus symlinked `AGENTS.md` instruction mirror |

# Functional Scope

The instruction file defines four main categories:

- end-to-end research workflows such as document-to-insight pipelines, multi-document analysis, cross-modal workflows, and theory-aware processing
- UI tests for Streamlit workflows, orchestration, visualization, and export
- MCP tool tests for availability, tool behavior, chained workflows, and error handling
- research-scenario tests for academic use cases, quality validation, performance, and reproducibility [2]

This is the functional counterpart to [Digimon Lineage Integration Tests](/wiki/sources/digimon-lineage-integration-tests.md): integration tests focus on components working together, while functional tests frame the system from user workflows and research scenarios.

# Preserved Test Threads

Representative preserved tests include:

- `test_ui_complete.py`: complete UI user-journey testing with upload, process, visualization, query, and export framing. [5]
- `test_mcp_tools_complete.py`, `test_mcp_tools_live.py`, `test_mcp_tool_chains.py`, and service-level MCP tests: MCP availability, live tools, and chained tool behavior. [1]
- `test_pdf_workflow_complete.py`, `test_graphrag_system_direct.py`, `test_integration_complete.py`, and phase-specific tests: PDF-to-answer, direct GraphRAG phases, phase transitions, and cross-phase data flow. [1]
- `test_no_mocks_policy.py` and `test_tools_functional_real.py`: explicit no-mocks / real-tool functional pressure. [3][4]
- `test_enhanced_batch_processing.py`, `test_enhanced_response_generation.py`, `test_adaptive_execution.py`, and `test_dynamic_dag_building.py`: later functional coverage around adaptive execution, DAG construction, response generation, and batch processing. [1]

# No-Mocks Evidence And Caveat

`test_no_mocks_policy.py` directly tests whether Phase 1 tools fail instead of returning mock data when Neo4j is unavailable. [3]

`test_tools_functional_real.py` frames itself as real tool execution with no lazy mocking, stubs, or fallbacks. [4]

But the same functional directory also contains tests that use mocks or mock data for UI, scheduler, API-key, phase, or dependency isolation. That is not automatically wrong, but it means "functional" and "no mocks" are not uniform properties of every file in the directory. Treat the no-mocks claims as specific to the relevant tests, not to the whole corpus.

# Runtime Status Caveat

Several functional tests require external state: running UI servers, API keys, Neo4j, MCP services, installed optional dependencies, Streamlit, or old repository-root import paths. [1][2]

The source corpus preserves test definitions, not execution reports. Future work should pair this page with test output artifacts or a fresh reconstructed run before making pass/fail claims.

# Interpretation

The functional tests show the project trying to validate whether KGAS could deliver usable research workflows:

- real or realistic document processing
- MCP-accessible tools and chains
- UI workflows beyond import checks
- phase-level GraphRAG execution
- environment/config failure behavior
- user-facing quality and performance expectations

This strengthens [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md) by showing a dedicated functional layer, while also reinforcing [Current Status Verification Discipline](/wiki/concepts/current-status-verification-discipline.md): source-code tests alone are not current runtime proof.

# Links

- [Digimon Lineage Tests Corpus](/wiki/sources/digimon-lineage-tests-corpus.md)
- [Digimon Lineage Integration Tests](/wiki/sources/digimon-lineage-integration-tests.md)
- [Digimon Lineage Archived Root Tests](/wiki/sources/digimon-lineage-archived-root-tests.md)
- [Current Runtime Import Check 2026-06-25](/wiki/sources/current-runtime-import-check-2026-06-25.md)
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md)
- [Current Status Verification Discipline](/wiki/concepts/current-status-verification-discipline.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/functional/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/functional/CLAUDE.md`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/functional/test_no_mocks_policy.py`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/functional/test_tools_functional_real.py`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/functional/test_ui_complete.py`
