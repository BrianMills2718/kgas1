---
type: SourceSummary
title: Digimon Lineage Archived Experimental Tests
description: Archived experimental KGAS test corpus covering redundant functional MCP/PDF workflows, adversarial and TORC stress suites, API/contract/configuration/database checks, PageRank fixes, enhanced pipeline, and vertical-slice probes.
tags: [source, digimon-lineage, archive, tests, experimental, stress, torc, mcp, contracts, configuration, pagerank, vertical-slice]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_experimental/
confidence: high
---

# Summary

`archive/archived_experimental/` is a 33-file, 576,378-byte archive of experimental KGAS tests. Its aggregate content-manifest hash is `352e42e1ae4b7348100d36923495460830d08571d94df6fa7f775945d82df649`. [1]

The archive is mainly a historical test-design and reliability-pressure corpus. It includes redundant functional MCP/PDF workflow tests, broad adversarial/TORC stress frameworks, API/contract/configuration/database tests, PageRank fix verifications, enhanced pipeline tests, and vertical-slice probes. [1]

# Inventory

| Area | Files | Role |
| --- | ---: | --- |
| `functional_redundant/` | 15 | Redundant functional tests for MCP tools, direct/running-server MCP access, PDF-to-answer workflows, full pipeline integration, and cross-component integration. [2] |
| `stress/` | 5 | Adversarial, extreme stress, compatibility validation, all-phase stress, and TORC reliability frameworks. [3] |
| Root tests | 13 | API contract, API standardization, configuration, contract validation, database, enhanced pipeline, PageRank, identity, optimized workflow, and vertical-slice tests. [4] |
| Empty directories | 3 | `e2e/`, `fixtures/`, and `utilities/` exist but contain no files in this archive. [1] |

# Functional Redundant Tests

The `functional_redundant/` directory is an archived set of overlapping workflow tests. `test_all_mcp_tools.py` attempts to test 29 MCP tools across identity, provenance, quality, workflow-state, vertical-slice, and system categories. It writes a JSON result file and reports pass/fail counts from direct function calls. [5]

`test_cross_component_integration.py` is broader: it defines a synthetic research document and attempts real data-flow checks across PDF processing, entity extraction, graph construction, visualization, query, phase-to-phase integration, MCP/core integration, error propagation, and performance. [6]

The repeated PDF-to-answer workflow variants are useful historically because they show the same vertical-slice objective being approached through several different test shapes before the archive labeled them redundant. [2]

# Stress And TORC

The `stress/` directory preserves a reliability-testing design layer. `test_torc_framework.py` defines TORC as Time, Operational Resilience, and Compatibility, with metrics for response-time percentiles, throughput, recovery time, error rates under load, graceful degradation, interface compliance, backwards compatibility, cross-component compatibility, and version tolerance. [7]

`test_adversarial_comprehensive.py` runs component isolation, cross-phase compatibility, stress/load, edge-case robustness, failure recovery, performance under load, resource management, concurrent access, data corruption resilience, and API contract validation. It then computes TORC-style metrics and recommendations. [8]

These stress files are important as reliability methodology, but their presence alone is not evidence that KGAS passed the stress suites. The wiki should treat them as archived test harnesses unless paired with captured result artifacts.

# Root-Level Tests

The root-level tests cover API standardization, configuration, contracts, database connections, PageRank fixes, identity-service consolidation, enhanced pipeline, optimized workflow, and vertical slice. [4]

`test_api_contracts.py` targets parameter naming and interface consistency, including workflow state service method signatures and legacy-to-standard parameter migration. [9]

`test_contract_validation.py` is more conventional pytest-style contract validation, but it uses mock PDF loader and text chunker classes to test contract mechanics rather than full tool behavior. [10]

`test_configuration_system.py` exercises default config loading, YAML overrides, environment overrides, validation, PageRank config integration, and identity-service config integration. This is a useful precursor to the later configuration-centralization policy. [4]

# PageRank And Vertical Slice

Several root tests focus on PageRank initialization and optimization. `test_final_pagerank_verification.py` records a specific fix: PageRank should be initialized with service objects rather than only Neo4j strings, and it prints a confirmation message when the direct call works. [11]

`test_vertical_slice.py` is more problematic as evidence. It describes a PDF-to-PageRank-to-answer workflow, but the script creates a text file, says PDF loading is mocked/modified for testing, and then prints component implementation status rather than executing a full workflow. This is a preserved status/probe artifact, not strong end-to-end proof. [12]

# Caveats

Some tests are proper assertion-based pytest-style checks; others are standalone scripts that print PASS/FAIL or implementation status. Treat them differently when using the archive as evidence.

Some tests depend on local services such as Neo4j, historical module paths, or generated local files. The empty `e2e/`, `fixtures/`, and `utilities/` directories show that the experimental archive was partially organized but not fully populated. [1]

The targeted credential scan found no literal OpenAI or Google API keys in this archive. [1]

# Interpretation

This archive shows KGAS moving from demo/proof scripts toward broader reliability instrumentation. The core themes are MCP coverage, cross-component integration, API contract standardization, centralized configuration, PageRank initialization repair, vertical-slice ambition, and TORC/adversarial stress methodology.

It also preserves a recurring evidence-quality issue: broad test harnesses and printed completion claims can look stronger than the actual executed proof. The most reliable use of this archive is as historical test-intent and methodology provenance, not as a definitive current-status record.

# Relationship To Wiki

- [Digimon Lineage Scripts Archive 2025 08](digimon-lineage-scripts-archive-2025-08.md): neighboring script/demo/fix/test archive with similar caveats.
- [Digimon Lineage Archived Root Tests](digimon-lineage-archived-root-tests.md): broader historical root-test corpus.
- [Digimon Lineage Functional Tests](digimon-lineage-functional-tests.md): active functional-test corpus comparison point.
- [Digimon Lineage Reliability Tests](digimon-lineage-reliability-tests.md): reliability corpus with status-boundary caveats.
- [Digimon Lineage Performance Tests](digimon-lineage-performance-tests.md): benchmark and performance-test definitions.
- [Test Evidence Layer](../concepts/test-evidence-layer.md): synthesis of broad test intent versus runtime proof.
- [Evidence Claim Discipline](../concepts/evidence-claim-discipline.md): claim-level distinctions needed for this archive.
- [Current Status Verification Discipline](../concepts/current-status-verification-discipline.md): avoid treating historical tests as current status.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_experimental/`

[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_experimental/functional_redundant/`

[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_experimental/stress/`

[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_experimental/`

[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_experimental/functional_redundant/test_all_mcp_tools.py`

[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_experimental/functional_redundant/test_cross_component_integration.py`

[7] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_experimental/stress/test_torc_framework.py`

[8] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_experimental/stress/test_adversarial_comprehensive.py`

[9] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_experimental/test_api_contracts.py`

[10] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_experimental/test_contract_validation.py`

[11] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_experimental/test_final_pagerank_verification.py`

[12] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_experimental/test_vertical_slice.py`
