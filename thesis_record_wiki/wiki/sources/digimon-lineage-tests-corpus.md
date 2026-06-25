---
type: Source
title: Digimon Lineage Tests Corpus
description: Top-level inventory of the 519-file tests directory, including testing principles, category structure, and verification caveats.
tags: [source, digimon-lineage, tests, verification, integration, functional, reliability, no-mocks]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/CLAUDE.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/integration/README.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/integration/TEST_INDEX_ANALYSIS.md
confidence: medium
---

# Summary

`tests/` is a 519-file, 7,813,141-byte test corpus with aggregate hash `33018ee61072be02089d195d510f658a46a6a2f0ec635c8fdedca65045f6b329`. [1]

The test suite instructions frame it as a comprehensive GraphRAG test suite spanning unit, functional, integration, performance, security, UI, error-scenario, and reliability tests. Its strongest methodological claim is "real execution over mocks" for functional tests and complete workflows. [2]

This page is a top-level inventory only. The integration tests and archived root tests are large enough to deserve separate later slices.

# Directory Inventory

| Test area | Files | Bytes | Aggregate hash | Role |
|---|---:|---:|---|---|
| `analytics/` | 14 | 221,646 | `1fb79c351bf6fd754cd73f8981d4c1c80760bf0dbcfc0932377e1ae12917c13d` | cross-modal, centrality, clustering, graph export, temporal analysis |
| `archived_experimental/` | 33 | 576,378 | `3f377d28c05aaa31bf5e1bb74edbe3197cc2db7d9768b6ab8e11eeb0dd7b98f3` | archived experimental tests around contracts, configuration, database, PageRank, vertical slice |
| `archived_root_tests/` | 100 | 1,364,253 | `f2f79591d3248bde79e86064ab377b8b46091d4dd4d18058f8c4351d12c3faa3` | migrated historical root tests across schemas, DAGs, LLMs, MCP, Neo4j, vertical slice, workflows |
| `error_scenarios/` | 8 | 146,407 | `e4613ff210d1b1769f5935e44c9622cf2c3d807fbe5df1cee03cc478479b0c56` | edge cases, persistence recovery, failure modes, resilience |
| `functional/` | 46 | 573,594 | `068fffc89731339e9614387c8a9121cb563c707c51f7b5b33a2585334019c3ca` | real tool/UI/MCP/PDF/workflow functional tests |
| `integration/` | 74 | 1,132,554 | `14c688d3edecbab29ace16d074cd07f1aceab5b29dbad32b3ef65bf96da949d8` | end-to-end, orchestration, service, theory, MCP, Neo4j, package integration |
| `performance/` | 16 | 320,008 | `c9c936e342802bdf91a0a2ca46d148bcf0e44c332c40a11357fac9f3a5374f5f` | async, analytics, agent, production, scalability, resource tests |
| `reliability/` | 33 | 456,724 | `e3bc0caedbca68419844ac1cc8a2810d20057bd408325ccb55e22baa9c4b4345` | reliability certification, load, failure, transactions, audit, citation, health, SLA |
| `unit/` | 75 | 1,376,810 | `4103d35e85c2f0061bf5d590a276cc8cb552a948b361c8db7326c290514a621f` | component-level tests across core, monitoring, orchestration |
| other test areas | 118 | 1,544,767 | included in root aggregate | fixtures, mocks, monitoring, phase3, security, tools, UI, utilities, validation, vector index health |

# Testing Principles

The tests instructions define several important project norms:

- functional tests must execute real tool operations, not mocked core functionality
- functional tests should use real data and measurable assertions
- integration tests should validate tool chains, service communication, and data flow
- performance tests should track timing/resource use on realistic inputs
- security tests should actually attempt attack/failure cases
- error-scenario tests should cover real failure conditions and recovery behavior [2]

These principles are important provenance for [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md). The presence of these rules shows an explicit attempt to prevent "mock-passing" claims, but it does not prove the tests passed.

# Integration Test Metadata

The integration README says these are active integration tests for the main `/src/` system and were moved from the root directory on 2025-08-29. It explicitly says they test different components than the `tool_compatability/poc/vertical_slice` work. [3]

`TEST_INDEX_ANALYSIS.md` is a systematic index for 59 integration tests. It records purpose/dependency/status fields, but many entries are still marked `UNKNOWN`. That file is itself useful negative evidence: the test corpus was being mapped, but working status was not fully resolved in the preserved index. [4]

# Interpretation

The tests corpus is evidence of a serious verification push:

- there are dedicated categories for functional, integration, reliability, performance, security, UI, and error scenarios
- the no-mocks principle is explicit
- integration tests distinguish main `/src/` from the vertical-slice POC
- archived root tests preserve a broad history of experimental tests rather than deleting them

The main caveat is that test presence, test names, and testing principles are not runtime proof. Future wiki work should ingest preserved test outputs, CI logs, or run selected tests in a reconstructed environment before converting this into pass/fail claims.

# Links

- [Digimon Lineage Scripts Corpus](/wiki/sources/digimon-lineage-scripts-corpus.md)
- [Digimon Lineage Evidence Archives](/wiki/sources/digimon-lineage-evidence-archives.md)
- [Current Runtime Import Check 2026-06-25](/wiki/sources/current-runtime-import-check-2026-06-25.md)
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md)
- [Current Status Verification Discipline](/wiki/concepts/current-status-verification-discipline.md)
- [Vertical Slice vs Main System](/wiki/concepts/vertical-slice-vs-main-system.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/CLAUDE.md`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/integration/README.md`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/integration/TEST_INDEX_ANALYSIS.md`
