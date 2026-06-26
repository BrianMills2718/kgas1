---
title: Legacy Vertical Slice Test Revival Plan
date: 2026-06-26
status: plan
scope: tool_compatability/poc/vertical_slice
---

# Purpose

Define a safe plan for reviving the legacy vertical-slice POC tests without executing import-time graph cleanup, stale credentials, hardcoded historical paths, or missing-data scripts.

# Sources Consulted

- `tool_compatability/poc/vertical_slice/THESIS_REQUIREMENTS.md`
- `tool_compatability/poc/vertical_slice/DOCUMENTATION_AUDIT.md`
- `tool_compatability/poc/vertical_slice/IMPORT_PATH_ARCHITECTURE_AUDIT.md`
- `tool_compatability/poc/vertical_slice/INTEGRATION_PATTERNS_EXTRACTED.md`
- `tool_compatability/poc/vertical_slice/test_5_tools.py`
- `tool_compatability/poc/vertical_slice/test_5_tools_fixed.py`
- `tool_compatability/poc/vertical_slice/test_6_tools.py`
- `tool_compatability/poc/vertical_slice/test_complex_pipeline.py`
- `tool_compatability/poc/vertical_slice/test_connections.py`
- `tool_compatability/poc/vertical_slice/test_document_isolation.py`
- `tool_compatability/poc/vertical_slice/test_integration.py`
- `tool_compatability/poc/vertical_slice/test_proper_kg_extraction.py`
- `tool_compatability/poc/vertical_slice/test_services.py`
- `tool_compatability/poc/vertical_slice/test_tools.py`
- `tool_compatability/poc/vertical_slice/tests/test_vertical_slice.py`
- `tool_compatability/poc/vertical_slice/thesis_evidence/quick_paired_test.py`
- `investigations/2026-06-26-pytest-collection-performance.md`

# Current State

The legacy vertical-slice folder is a preserved POC area, not a safe default pytest suite. It contains valuable architectural material about:

- modular tool chaining;
- service-tool wrapper patterns;
- Neo4j plus SQLite integration;
- uncertainty and provenance requirements;
- document-isolation evidence goals;
- thesis-facing success criteria.

It also contains unsafe test-discovery behavior:

- top-level executable code in files named `test*.py`;
- absolute historical path loading such as `/home/brian/projects/Digimons`;
- stale Neo4j credentials such as `devpassword`;
- import-time graph cleanup using `MATCH (n:VSEntity) DETACH DELETE n`;
- generated working files in the repository directory;
- missing relative data dependencies such as `ground_truth_paired/metadata/doc_001_clean.json`;
- direct provider/API dependency paths.

# Risk Classes

| Class | Files | Risk | Safe default |
| --- | --- | --- | --- |
| A: docs/audits | `*.md`, `evidence/*.md` | Read-only value, no runtime risk. | Keep as historical sources. |
| B: pure-ish unit tests | `test_services.py` portions, simple service checks | May create local SQLite files; imports historical modules. | Migrate selectively after temp-dir isolation. |
| C: Neo4j live tests | `test_connections.py`, `test_tools.py`, `tests/test_vertical_slice.py` | Use stale credentials or shared graph. | Require env-gated isolated Neo4j scope. |
| D: import-time pipeline scripts | `test_5_tools.py`, `test_5_tools_fixed.py`, `test_6_tools.py`, `test_proper_kg_extraction.py`, `test_document_isolation.py` | Execute work during collection and delete `VSEntity` nodes. | Do not collect until refactored. |
| E: thesis evidence scripts | `thesis_evidence/*.py`, especially `quick_paired_test.py` | Missing working-directory assumptions, provider/Neo4j dependencies, output writes. | Treat as scripts until fixture paths are explicit. |

# Migration Target

Create a separate, explicit test namespace:

```text
tests/legacy_vertical_slice/
```

Do not point default pytest at this namespace until migrated tests are safe to collect and run. Default pytest should remain scoped to `tests/current_runtime`.

# Revival Sequence

## Phase 1: Inventory-Only Gate

Acceptance criteria:

- no legacy `test*.py` file executes work at import time;
- every file with top-level execution is classified;
- no Neo4j writes occur during collection;
- no absolute `/home/brian/projects/Digimons` path is required for collection.

Recommended work:

1. Rename script-style files from `test_*.py` to `*_script.py`, or move them under a non-collected `legacy_scripts/` directory.
2. Preserve original names in a manifest so provenance is not lost.
3. Add a short `README.md` in `tool_compatability/poc/vertical_slice/` explaining that default pytest intentionally excludes the folder.

## Phase 2: Unit-Test Extraction

Acceptance criteria:

- tests run without Neo4j, provider APIs, or project-root writes;
- SQLite tests use `tmp_path`;
- imports are package-relative or rooted through the repo path, not historical absolute paths.

Candidate tests:

- `VectorService` shape checks;
- `TableService` save/retrieve checks with temp DB;
- tool capability registration without live Neo4j;
- simple chain discovery using fake/in-memory tools.

## Phase 3: Live Neo4j Harness

Acceptance criteria:

- tests skip unless `NEO4J_PASSWORD` is present;
- all writes carry a unique run/source scope;
- cleanup deletes only exact run/source scope after the test;
- no broad `MATCH (n:VSEntity) DETACH DELETE n`;
- a backup/restore note exists before any shared graph test expansion.

Candidate tests:

- connection smoke;
- document-isolation proof using exact run marker;
- graph persister writes under scoped labels/properties;
- no shared global cleanup.

## Phase 4: Thesis Evidence Harness

Acceptance criteria:

- fixture paths are relative to the script file or passed as parameters;
- generated output goes to `tmp_path` or a clearly ignored output directory;
- provider calls are mocked only with explicit `# mock-ok` rationale or gated as live tests;
- evidence/result summaries are separated from runtime proof claims.

# Files To Migrate First

Recommended first migration slice:

1. Extract `test_services.py` table-service checks into `tests/legacy_vertical_slice/test_table_service.py` using `tmp_path`.
2. Add a no-IO framework capability registration test from `test_integration.py`.
3. Leave Neo4j and provider-dependent tests untouched.

Why this first: it proves the legacy area can be safely represented in pytest without touching graph state or external providers.

# Files To Defer

Defer these until a live-harness plan exists:

- `test_5_tools.py`
- `test_5_tools_fixed.py`
- `test_6_tools.py`
- `test_proper_kg_extraction.py`
- `test_document_isolation.py`
- `tests/test_vertical_slice.py`
- `thesis_evidence/run_thesis_collection.py`
- `thesis_evidence/evidence_collector.py`

# Stop Lines

Do not:

- run broad pytest over `tool_compatability/poc/vertical_slice`;
- execute scripts with stale `devpassword`;
- execute broad Neo4j cleanup;
- change or delete historical POC files without a provenance-preserving move plan;
- publish outputs from this folder without public/export review.

# Recommendation

Keep `pyproject.toml` default `testpaths = ["tests/current_runtime"]`. Start revival only with a small no-IO extraction slice from `test_services.py` and `test_integration.py`, then decide whether the legacy vertical-slice runtime is worth modernizing.
