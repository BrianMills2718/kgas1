# Plan #1 Closeout Review

**Date:** 2026-06-26
**Scope:** Safe autonomous completion work for Plan #1, excluding human-gated public/export, destructive cleanup execution, and live LLM-cost decisions.
**Disposition:** Runtime recovery is substantially complete for narrow current endpoints; full program completion remains blocked only where Plan #1 requires Brian review or external budget/credential approval.

## Findings

### 1. Current single-document runtime is proven for four formats

**Severity:** Resolved runtime gap

`/api/analyze` now has live Neo4j-backed proof for `.txt`, `.pdf`, `.md`, and `.docx`. Each path uses a real phase-1 loader, downstream T15/T23/T27/T31/T34/T49 stages, source-scoped graph writes/queries, and temporary upload cleanup.

Evidence:

- `tests/current_runtime`: `61 passed, 5 skipped`.
- Live document smoke: `.txt`, `.pdf`, `.md`, `.docx` all passed together.
- FastAPI deprecation warnings were removed by the lifespan migration.
- Wiki lint remains `196` pages, health `100/100`.
- `pip check` is clean.

### 2. Legacy `.doc` should stay out of scope unless explicitly requested

**Severity:** Low

The current proven Word path is `.docx` via `python-docx`. The legacy `.doc` binary format has no proven loader path in the current completion program. Keeping it at explicit 501 is safer than adding a converter dependency or shelling out to office tooling during an overnight safe run.

Disposition: keep `.doc` explicit 501 unless Brian requests legacy binary Word recovery.

### 3. Batch analysis is now the next safe runtime slice

**Severity:** Medium

`/api/batch/analyze` still returns 501 even though the single-document analyze path is now proven across the current supported formats. This is no longer blocked by missing format proof. It can be implemented safely by creating a job, running each file through the same `_analyze_document_upload(...)` path, collecting per-file results/errors, and leaving unsupported `.doc` as a per-file error.

Disposition: implement batch wiring as the next safe slice. It should remain in-process and local; no new persistence, external services, archive mutation, or graph cleanup.

### 4. Public/export and LLM recommendation remain real blockers

**Severity:** Human-gated

Public/export can leak raw preserved credentials/logs/sensitive material if done carelessly, and live `/api/recommend` can spend money or depend on external credentials. These are not engineering blockers to work around autonomously; they are explicit Plan #1 stop points.

Disposition: defer until Brian chooses privacy posture and LLM budget/provider.

## Gate Status

| Gate | Status |
|---|---|
| Preservation | Complete for safe work; raw archive untouched. |
| Runtime Environment | Complete locally; Neo4j backed up, `.env` ignored, dependencies clean. |
| Narrow E2E Pipeline | Complete for `.txt`, `.pdf`, `.md`, `.docx`. |
| Query Isolation | Complete for tested source-scoped writes and T49 filters. |
| Cleanup Safety | Helper complete; execute mode remains operator-triggered. |
| API Honesty | Complete for known unsupported paths; batch remains truthful 501 until next slice. |
| Review Gate | Complete for current safe runtime scope. |
| Public/Export | Blocked by Brian review. |
| LLM/Recommendation | Blocked by credential/cost approval. |

## Next Safe Slice

Wire `/api/batch/analyze` to the proven single-document analysis path:

- accept multiple uploads;
- create a job record;
- process files sequentially through `_analyze_document_upload(...)`;
- record per-file results and per-file errors;
- keep unsupported formats as errors rather than aborting the whole batch;
- add current-runtime tests for success, mixed unsupported file handling, and job-status reporting.

This is safe because it only composes already-proven local behavior and does not delete graph data, mutate preserved archives, publish anything, or call external LLMs.
