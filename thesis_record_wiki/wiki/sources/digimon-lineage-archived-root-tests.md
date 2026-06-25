---
type: Source
title: Digimon Lineage Archived Root Tests
description: Historical inventory of the 100-file archived_root_tests corpus, preserving early root-level test strategy, real-execution claims, and runtime-proof caveats.
tags: [source, digimon-lineage, tests, archived-root-tests, historical, verification, no-mocks, schemas, dag, mcp, llm]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/archived_root_tests/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/archived_root_tests/test_real_execution_no_mocks.py
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/archived_root_tests/test_real_kgas_workflow.py
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/archived_root_tests/test_dag_15_tools_real.py
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/archived_root_tests/test_parser_baseline.py
confidence: medium
---

# Summary

`tests/archived_root_tests/` is a 100-file, 1,364,253-byte historical test corpus with local aggregate hash `425025f1b8b06d9bdd580e67c25c3af49c7260a002c6f4351e6c43561beb274d`. [1]

This directory appears to preserve root-level tests that were later separated from the active test tree. Its value is historical: it captures what the project was trying to validate across real execution, DAG workflows, MCP, LLM integration, schema systems, formula parsing, Neo4j, vertical slices, service registration, and cross-modal analysis. [1]

The central caveat is that many file names and docstrings use success language such as `real`, `final`, `fixed`, `working`, and `successful`. Those labels are evidence of test intent and developer framing, not proof that the tests currently pass.

# Inventory

| Measure | Value |
|---|---:|
| Files | 100 |
| Bytes | 1,364,253 |
| Local aggregate hash | `425025f1b8b06d9bdd580e67c25c3af49c7260a002c6f4351e6c43561beb274d` |
| Largest prefix family | `test_real_*`, 15 files, 282,404 bytes |
| Other repeated families | `gemini`, `comprehensive`, `enhanced`, `final`, `neo4j`, `vertical`, `mcp`, `service`, `sophisticated` |

The top-level [Digimon Lineage Tests Corpus](/wiki/sources/digimon-lineage-tests-corpus.md) reports a different subtree hash for this folder because it used the parent `tests/` aggregation context. This page's hash is local to `archived_root_tests/`.

# Main Themes

The archived root tests preserve several test-strategy threads:

- **Real execution and no-mocks pressure**: `test_real_execution_no_mocks.py`, `test_real_kgas_workflow.py`, `test_real_15_tool_chain_with_llm.py`, `test_real_dag_execution.py`, `test_real_dag_no_agents.py`, and `test_real_working_tools.py`.
- **Long tool-chain and DAG execution**: `test_15_tool_chain.py`, `test_dag_15_tools_real.py`, `test_real_dag_with_tools.py`, `test_dynamic_tool_orchestration.py`, and `test_dynamic_parallel_execution.py`.
- **Schema/formalism exploration**: `test_nary_graph_schemas.py`, `test_rdf_owl_schemas.py`, `test_typedb_style_schemas.py`, `test_uml_class_schemas.py`, `test_orm_schemas.py`, `test_sophisticated_schemas.py`, and `test_schema_system.py`.
- **Formula/parser work**: `test_parser_baseline.py`, `test_enhanced_formula_parser.py`, `test_hybrid_formula_parser.py`, `test_formula_implementation.py`, `test_real_formula_implementation.py`, and `test_integrated_formula_tools.py`.
- **LLM/API probes**: Gemini, o4-mini, universal LLM, enhanced API client, structured output, T23C migration, and sophisticated extraction tests.
- **MCP/service/runtime integration**: direct MCP tools, MCP real workflow, service registration, service integration, security integration, provenance persistence, performance monitoring, and Neo4j connection tests. [1]

# Real-Execution Claims

Several files explicitly try to separate real execution from simulated success:

- `test_real_execution_no_mocks.py` frames itself as using real tools, databases, API calls, and realistic timing, with a fail-loud orientation. [2]
- `test_real_kgas_workflow.py` frames itself as a real KGAS workflow through MCP/server/tool paths with no simulations. [3]
- `test_dag_15_tools_real.py` frames DAG execution as 15+ real tool operations. [4]
- `test_parser_baseline.py` says it establishes actual parser performance before making claims. [5]

This is important evidence for [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md): the project was trying to move beyond component-name or mocked-success claims.

# Historical Caveats

The same corpus also has several reasons not to treat it as current proof:

- Some tests require external services, databases, API keys, or installed optional dependencies.
- Several files manipulate `sys.path` or assume old repository-root layout.
- Some tests are script-style probes rather than conventional pytest test modules.
- The corpus includes both no-mocks claims and files that use mocks or simulations for specific purposes.
- "final", "fixed", "working", and "successful" in filenames may reflect the state at the time of writing, not the current state of the preserved project.

Future work should connect this corpus to preserved outputs or fresh runtime checks before using it to make pass/fail claims.

# Interpretation

The archived root tests are best read as a history of pressure points:

- Could KGAS run actual multi-step workflows rather than only isolated tools?
- Could theory schemas become executable code or concrete extraction behavior?
- Could the project support multiple formal schema styles?
- Could LLM/API integration be made reliable across providers?
- Could Neo4j, MCP, services, provenance, security, and performance monitoring work together?

That makes this directory more valuable as design and verification lineage than as a green-test badge.

# Links

- [Digimon Lineage Tests Corpus](/wiki/sources/digimon-lineage-tests-corpus.md)
- [Digimon Lineage Integration Tests](/wiki/sources/digimon-lineage-integration-tests.md)
- [Digimon Lineage Scripts Corpus](/wiki/sources/digimon-lineage-scripts-corpus.md)
- [Current Runtime Import Check 2026-06-25](/wiki/sources/current-runtime-import-check-2026-06-25.md)
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md)
- [Current Status Verification Discipline](/wiki/concepts/current-status-verification-discipline.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/archived_root_tests/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/archived_root_tests/test_real_execution_no_mocks.py`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/archived_root_tests/test_real_kgas_workflow.py`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/archived_root_tests/test_dag_15_tools_real.py`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/archived_root_tests/test_parser_baseline.py`
