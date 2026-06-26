---
type: Source
title: Current Runtime Live Neo4j Rerun 2026 06 26
description: Evidence record for rerunning the full current-runtime test suite with local Neo4j credentials loaded.
tags: [source, runtime, tests, neo4j, verification]
created: 2026-06-26
updated: 2026-06-26
source_path: tests/current_runtime
confidence: high
---

> Sources consulted: command output from `PYTHONPATH=. pytest -q tests/current_runtime -rs` with the repo-local ignored Neo4j environment loaded. Status: observed local verification record; no graph cleanup or source mutation was executed.

# Summary

The full `tests/current_runtime` suite was rerun with local Neo4j credentials loaded from the ignored repository `.env`, against the already-running healthy local Neo4j container.

Latest observed result:

```text
76 passed, 6 warnings in 89.66s
```

This upgrades the current-runtime evidence from "70 passed, 6 credential-gated skips" to a full live local pass for this environment.

# What Changed From The Skipped Run

Without `NEO4J_PASSWORD` in the environment, six tests skip:

- one complete-pipeline Neo4j runtime test;
- five cross-modal API contract tests requiring live Neo4j.

With local credentials loaded, those tests ran and passed. No code changes were required for this rerun.

# Warning Caveat

The latest run emitted six Neo4j driver warnings:

```text
UserWarning: Expected a result with a single record, but found multiple.
```

Affected tests:

- `test_complete_pipeline_processes_tiny_txt_with_neo4j`
- `test_analyze_document_txt_upload_runs_live_complete_pipeline`
- `test_analyze_document_pdf_upload_runs_live_complete_pipeline`
- `test_analyze_document_markdown_upload_runs_live_complete_pipeline`
- `test_analyze_document_docx_upload_runs_live_complete_pipeline`
- `test_batch_analyze_runs_live_complete_pipeline_for_txt_upload`

Interpretation: these warnings are non-failing but worth preserving. They likely reflect accumulated local graph state or queries that expect one record while repeated smoke data can produce multiple matching records. This is not a reason to delete graph data; use source scoping or isolated test databases for future cleanup-sensitive work.

# Safety Notes

- No Neo4j cleanup was executed.
- No raw archive files were touched.
- No credentials are reproduced here.
- The local `.env` remains ignored and should not be published.
- The result is local runtime evidence, not dissertation validation or production readiness.

# Links

- [Runtime Verification Isolation Boundary](/wiki/concepts/runtime-verification-isolation-boundary.md)
- [Thesis Recovery Current State 2026 06 26](/wiki/concepts/thesis-recovery-current-state-2026-06-26.md)
- [Current Status Verification Discipline](/wiki/concepts/current-status-verification-discipline.md)
- [KGAS Dissertation Claim Map](/wiki/concepts/kgas-dissertation-claim-map.md)

# Citations

[1] `tests/current_runtime` live runs on 2026-06-26 with local ignored Neo4j credentials loaded.  
[2] `/wiki/concepts/runtime-verification-isolation-boundary.md`
