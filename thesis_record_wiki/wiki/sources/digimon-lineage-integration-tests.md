---
type: Source
title: Digimon Lineage Integration Tests
description: Focused inventory of the preserved tests/integration directory, including active-main-system claims, test-index mismatch, and unverified runtime-status caveats.
tags: [source, digimon-lineage, tests, integration, main-system, verification, no-mocks, runtime-status]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/integration/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/integration/README.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/integration/TEST_INDEX_ANALYSIS.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/integration/CLAUDE.md
confidence: medium
---

# Summary

`tests/integration/` is a 74-file, 1,132,554-byte preserved integration-test corpus with local aggregate hash `68e9c46d0f17c47bd55bcb1648a9203a413e013d6a168aecb4d5999482b71d3f`. [1]

The README frames this directory as active integration tests for the main `/src/` system, moved from the repository root on 2025-08-29, and explicitly distinct from the `tool_compatability/poc/vertical_slice` work. [2]

The strongest evidence value is not that the tests passed. The strongest evidence value is that the project preserved a large set of integration targets around end-to-end pipelines, orchestration, services, theory support, Neo4j, MCP, tool adapters, and cross-modal workflows while also preserving an unfinished status index. [1][3]

# Directory Inventory

| Area | Files | Bytes | Aggregate hash | Role |
|---|---:|---:|---|---|
| root test files | 63 | 927,112 | `3d00069ae3746532b74d499154d86724b4d338d35f2a1fa928e6ecc8aae06571` | main integration test files |
| root metadata/files | 5 | 51,504 | `e7178a493bf165bb86c8dfa13f09dda9564793ec62b91e4a62ca1dd7aec9a5c1` | README, CLAUDE instructions, test index, package markers |
| `config/` | 1 | 2,463 | `25fa481f5c08b2a27bd96253c0376642e710b4a3acc452004c43e372807e754b` | default integration configuration |
| `data/` | 3 | 102,400 | `0d3faa27f8c0190dece1ad4c93beb954ad6fab810204f729524197e16e2ebe69` | preserved provenance/memory test data |
| `logs/` | 2 | 49,075 | `230efe1e5b0262eaafe6dfbb7b6ec2e3beade473fc70c8bc94b27732a05e6128` | preserved Super Digimon logs |

# Scope Signals

The root test files include large pipeline and orchestration tests:

- `test_complete_integration_real.py`: 1,313 lines, described by the preserved index as comprehensive real-functionality testing with no mocks, stress testing, load testing, and performance benchmarks. [1][3]
- `test_agent_workflow_comprehensive.py`: 988 lines, broad workflow coverage by filename and size. [1]
- `test_pipeline_orchestrator_integration.py`: 926 lines, described as pre-decomposition behavior capture with comprehensive mocking. [3]
- `test_unified_tools_integration.py`: 747 lines, unified-tools integration by filename and size. [1]
- `test_t15a_chunker_integration.py` and `test_t23c_ontology_extractor_integration.py`: 728 and 685 lines, tool-specific integration targets. [1]
- `test_llm_ontology_integration.py`, `test_multi_document_processing.py`, `test_cross_document_relationships.py`, and `test_dynamic_execution_e2e.py`: each over 500 lines and oriented toward ontology, multi-document, relationship, and dynamic execution coverage. [1]

Smaller tests preserve targeted checks such as Neo4j auth, registry discovery, Pandas tools, universal LLM import/integration, package integration, phase switching, and cross-modal registration. [1][2]

# Index Mismatch

There are three preserved count signals:

- the README lists 6 representative test files
- `TEST_INDEX_ANALYSIS.md` says it is indexing 59 integration tests
- the actual preserved directory contains 63 root `test_*.py` files and 74 files total [1][2][3]

This mismatch should be preserved as provenance rather than normalized away. It likely reflects directory evolution after the README/index were written or partial documentation of a larger test tree.

# Runtime Status Caveat

`TEST_INDEX_ANALYSIS.md` is explicitly unfinished: it reports 12 of 59 tests analyzed, 0 verified working, 0 verified broken, and many individual entries with `Working Status: UNKNOWN`. [3]

That makes this corpus strong evidence of verification intent and integration-test breadth, but weak evidence of actual current runtime health. It should be paired with preserved CI output, test reports, or fresh reconstructed-environment runs before being used as a pass/fail claim.

# Integration Testing Contract

The directory-level `CLAUDE.md` defines integration testing as realistic interaction across components, services, databases, and external dependencies. It describes expected categories: pipeline integration, database integration, service integration, and cross-modal integration. [4]

Important test-design norms in that file:

- integration tests should use actual service implementations instead of mocks
- data and environments should be isolated
- realistic data and performance assertions should be used
- error scenarios and resource cleanup should be included
- the phase-adapter pattern should import specific adapter classes instead of a nonexistent factory [4]

This connects the integration corpus to [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md): the project had rules for stronger evidence, but a rule document is not a substitute for executed results.

# Links

- [Digimon Lineage Tests Corpus](/wiki/sources/digimon-lineage-tests-corpus.md)
- [Digimon Lineage Scripts Corpus](/wiki/sources/digimon-lineage-scripts-corpus.md)
- [Current Runtime Import Check 2026-06-25](/wiki/sources/current-runtime-import-check-2026-06-25.md)
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md)
- [Current Status Verification Discipline](/wiki/concepts/current-status-verification-discipline.md)
- [Vertical Slice vs Main System](/wiki/concepts/vertical-slice-vs-main-system.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/integration/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/integration/README.md`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/integration/TEST_INDEX_ANALYSIS.md`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/integration/CLAUDE.md`
