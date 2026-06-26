# Plan #1: Full Program Completion

**Status:** In Progress
**Type:** design
**Priority:** High
**Blocked By:** human review for destructive cleanup, public export, and LLM-cost decisions
**Blocks:** runtime hardening, review gate, public/export packaging

---

## Gap

**Current:** The thesis/KGAS record is now preserved in a Karpathy-style wiki, current runtime repairs have proven the narrow `.txt` complete-pipeline path, Neo4j is backed up, and live source-scoped smoke tests pass. The project still lacks a single completion contract that says when the recovered program is "done, running, and reviewed."

**Target:** A documented, executable completion program with gates for preservation, runtime, review, cleanup safety, and public/export readiness. Agents should be able to continue safe work without asking Brian, while deferring destructive or credential/cost-bearing decisions.

**Why:** Brian wants a full record of the PhD work and a working recovered program, but the repo contains years of historical claims, partial implementations, archived evidence, and live graph state. Completion must be evidence-based, not a broad cleanup or success narrative.

---

## References Reviewed

- `CLAUDE.md` - project-wide execution, evidence, commit, and safety policy.
- `docs/plans/CLAUDE.md` - implementation-plan convention.
- `docs/plans/TEMPLATE.md` - plan schema.
- `thesis_record_wiki/PROGRESS.md` - durable mission, completed runtime/wiki slices, deferred risk decisions.
- `thesis_record_wiki/wiki/concepts/runtime-verification-isolation-boundary.md` - Neo4j source-scoping and cleanup safety boundary.
- `thesis_record_wiki/wiki/concepts/public-export-security-boundary.md` - public/export security boundary.
- `thesis_record_wiki/wiki/sources/current-runtime-import-check-2026-06-25.md` - current runtime evidence and repair chronology.
- `investigations/2026-06-25-analyze-endpoint-document-pipeline.md` - complete-pipeline/API backing-path investigation.
- Memory context: `agent-memory recall 'phd_thesis_work KGAS completion runtime verification active decisions' --project phd_thesis_work` - no relevant past sessions found.

---

## Research Basis For This Slice

No additional research beyond repo-local references was needed. This plan is a recovery/completion contract over the current repo and preservation wiki, not a new architecture proposal.

---

## Capabilities

This plan does not create a new shared callable capability. It coordinates existing project capabilities and verification gates.

---

## Files Affected

- `docs/plans/01_full_program_completion.md` (create)
- `docs/plans/CLAUDE.md` (update active-plan index)
- `thesis_record_wiki/PROGRESS.md` (update next phase and completion criteria pointer)
- `thesis_record_wiki/wiki/log.md` (record planning update)

Future implementation slices may touch source/tests only after this plan names the gate they satisfy.

---

## Modality Split

- **Deductive / plan-first:** preservation invariants, repo hygiene, local environment bootstrap, source-scoped Neo4j cleanup command, API status honesty, test/review gates.
- **Exploratory / ladder:** quality of recovered thesis narrative, usefulness of public/export bundles, whether historical archived subsystems should be revived or only documented.
- **Human-review required:** any destructive Neo4j cleanup, raw archive mutation, public export, publishing to GitHub if it could expose credentials/private data, and live LLM-cost-bearing work.

---

## Full Completion Success Criteria

The full program is complete only when all gates below are satisfied or explicitly closed as out of scope by Brian.

| Gate | Success Criteria | Evidence Artifact | Current Status |
|---|---|---|---|
| Preservation | Raw `archive_full_record/` is not rewritten; wiki remains navigable and source-cited; public-export boundary exists. | wiki lint 100/100; `thesis_record_wiki/PROGRESS.md`; public-export concept | Mostly satisfied; ongoing maintenance |
| Runtime Environment | Local setup has reproducible Python deps, local Neo4j credentials, and backup procedure. | `.env` ignored locally; Neo4j dump path/hash; `pip check` | Satisfied locally |
| Narrow E2E Pipeline | `.txt`, tiny `.pdf`, tiny `.md`, and tiny `.docx` upload/API paths run real document-loading/T15/T23/T27/T31/T34/T49 stages with Neo4j source scoping. | live source-scoped smoke tests pass | Satisfied for `.txt`, tiny `.pdf`, tiny `.md`, and tiny `.docx` |
| Query Isolation | New graph writes carry `source_refs`; T49 can filter by `source_refs`; old smoke data cannot satisfy scoped proof. | current-runtime tests; live smoke result | Satisfied for tested path |
| Cleanup Safety | Any cleanup command is source-scoped, dry-run capable, backed by tests, and never broad-deletes graph data. | future cleanup test + command docs | Planned, safe if scoped |
| API Honesty | Unsupported/unwired endpoints return explicit 501/503 rather than mock success. | current-runtime API tests | Mostly satisfied for known endpoints |
| Review Gate | Code review pass covers security, destructive actions, LLM boundaries, and remaining stale/mock claims. | review report in `investigations/` or wiki source page | Complete for safe runtime scope |
| Public/Export | Any shareable bundle is derived, scanned, documented, and excludes/separates raw credential-bearing archives. | export manifest + scan output | Human-review required |
| LLM/Recommendation | Real `/api/recommend` is tested only after selector credentials/cost are approved. | credential-aware test report | Deferred |

---

## Slice Roadmap

### Slice 1 - Completion Contract And Baseline

**Status:** In Progress

**Done when:**
- this plan is committed and linked from progress;
- wiki lint remains 100/100;
- current-runtime tests pass or failures are recorded as blockers;
- unsafe tasks are in the concern register.

### Slice 2 - Source-Scoped Cleanup Command

**Status:** Complete

**Safe scope:** implement a dry-run-first cleanup helper that accepts an explicit `source_ref` and deletes only relationships/nodes carrying that exact source ref. It must refuse empty or broad scope values.

**Done when:**
- [x] tests prove empty/broad scope is rejected;
- [x] dry-run reports candidate counts;
- [x] execution deletes only the scoped data in a synthetic fixture;
- [x] docs say broad cleanup is forbidden.

Evidence: `scripts/neo4j_source_cleanup.py`, `tests/current_runtime/test_neo4j_source_cleanup.py`, `tests/current_runtime` pass count 58 passed / 2 skipped, and dry-run `storage://document/nonexistent-dry-run-scope` reports zero candidates without deletion.

### Slice 3 - Runtime Review Bundle

**Status:** Complete

**Safe scope:** generate a review report over current runtime status, endpoint honesty, and remaining unproven formats. No code changes unless review findings are unambiguous.

**Done when:**
- [x] report lists tests run, pass/fail counts, remaining 501/503 endpoints, and evidence grades;
- [x] current-runtime tests pass;
- [x] wiki records the review result.

Evidence: `investigations/2026-06-26-runtime-completion-review.md`, latest `tests/current_runtime` pass count 58 passed / 2 skipped, live Neo4j smoke result 2 passed, and wiki lint health 100/100.

### Slice 4 - Public/Export Plan

**Status:** Blocked by Brian review

**Done when:**
- Brian chooses target audience and privacy posture;
- export manifest excludes raw secrets/sensitive datasets or documents their handling;
- export candidate is scanned before any publication.

### Slice 5 - LLM-Backed Recommendation

**Status:** Blocked by credential/cost approval

**Done when:**
- approved LLM selector is configured;
- recommendation endpoint has one live cost-bounded test;
- cost/result is recorded.

### Slice 6 - Narrow PDF Analyze Proof

**Status:** Complete

**Safe scope:** preserve the existing complete-pipeline path and prove `.pdf` uploads by passing the real temp-file suffix into T01. Do not modify raw archive documents or broaden support to `.docx`, `.doc`, or `.md`.

**Done when:**
- [x] focused API tests prove `.pdf` dispatch preserves suffix and source traceability;
- [x] live Neo4j-backed smoke proves a tiny text-bearing PDF reaches graph/query stages;
- [x] unproven `.docx`, `.doc`, and `.md` formats remain explicit 501.

Evidence: `src/api/cross_modal_api.py`, `tests/current_runtime/test_cross_modal_api_contract.py`, focused API tests 22 passed / 2 skipped, live TXT+PDF smoke 2 passed, and full `tests/current_runtime` count 59 passed / 3 skipped.

### Slice 7 - Narrow Markdown Analyze Proof

**Status:** Complete

**Safe scope:** route `.md` uploads through the existing T03 text-compatible loader and the same downstream complete-pipeline stages. Do not claim rich Markdown structure parsing or Word/PowerPoint support.

**Done when:**
- [x] focused API tests prove `.md` dispatch preserves suffix and source traceability;
- [x] live Neo4j-backed smoke proves a tiny Markdown file reaches graph/query stages;
- [x] unproven `.docx` and `.doc` formats remain explicit 501;
- [x] phase-1 loader provenance drift is repaired from `used={}` to `inputs=[]` for the current provenance service.

Evidence: `src/analytics/complete_pipeline.py`, `src/tools/phase1/t03_text_loader_unified.py`, `tests/current_runtime/test_cross_modal_api_contract.py`, focused API tests 23 passed / 3 skipped, live TXT+PDF+Markdown smoke 3 passed, full `tests/current_runtime` count 60 passed / 4 skipped, and `pip check` clean after adding `chardet>=5.0.0`.

### Slice 8 - Narrow DOCX Analyze Proof

**Status:** Complete

**Safe scope:** route `.docx` uploads through the existing T02 Word loader and the same downstream complete-pipeline stages. Do not claim legacy `.doc` support.

**Done when:**
- [x] focused API tests prove `.docx` dispatch preserves suffix and source traceability;
- [x] live Neo4j-backed smoke proves a tiny generated DOCX reaches graph/query stages;
- [x] legacy `.doc` remains explicit 501.

Evidence: `src/analytics/complete_pipeline.py`, `src/tools/phase1/t02_word_loader_unified.py`, `tests/current_runtime/test_cross_modal_api_contract.py`, focused API tests 24 passed / 4 skipped, live TXT+PDF+Markdown+DOCX smoke 4 passed, full `tests/current_runtime` count 61 passed / 5 skipped, and `pip check` clean.

### Slice 9 - Runtime Review Status Refresh

**Status:** Complete

**Safe scope:** update the Plan #1 runtime review artifact to reflect the later `.pdf`, `.md`, and `.docx` proof slices without changing runtime behavior.

**Done when:**
- [x] review addendum lists current proven formats and still-deferred endpoints;
- [x] next safe slice is updated after legacy `.doc` is classified.

Evidence: `investigations/2026-06-26-runtime-completion-review.md` addendum.

### Slice 10 - FastAPI Lifespan Modernization

**Status:** Complete

**Safe scope:** replace deprecated `@app.on_event("startup")` usage with FastAPI lifespan initialization while preserving the same service initialization behavior.

**Done when:**
- [x] current-runtime tests pass without FastAPI `on_event` deprecation warnings;
- [x] live TXT/PDF/Markdown/DOCX smoke still passes.

Evidence: `src/api/cross_modal_api.py`, API contract tests 24 passed / 4 skipped, full `tests/current_runtime` count 61 passed / 5 skipped, and live four-format smoke 4 passed.

### Slice 11 - Plan #1 Closeout Review

**Status:** Complete

**Safe scope:** separate completed runtime gates from human-gated blockers and select the next safe runtime slice.

**Done when:**
- [x] closeout review lists gate status;
- [x] unsafe/human-gated work remains deferred;
- [x] next safe slice is identified.

Evidence: `investigations/2026-06-26-plan1-closeout-review.md`.

---

## Required Tests

### New Tests (TDD)

| Test File | Test Function | What It Verifies |
|---|---|---|
| `tests/current_runtime/test_neo4j_source_cleanup.py` | `test_cleanup_rejects_empty_scope` | Cleanup cannot run without explicit source ref. |
| `tests/current_runtime/test_neo4j_source_cleanup.py` | `test_cleanup_dry_run_counts_only_source_ref` | Dry-run is scoped to exact source ref. |
| `tests/current_runtime/test_neo4j_source_cleanup.py` | `test_cleanup_deletes_only_source_ref_fixture` | Execution deletes only scoped synthetic fixture data. |

### Existing Tests (Must Pass)

| Test Pattern | Why |
|---|---|
| `tests/current_runtime` | Current repaired runtime boundary. |
| `python /home/brian/projects/.agents/skills/karpathy-wiki/scripts/lint.py thesis_record_wiki` | Wiki health and link integrity. |
| `.venv/bin/python -m pip check` | Dependency consistency. |

---

## Acceptance Criteria

- [ ] Completion plan exists and is the active plan.
- [ ] Every future implementation slice maps to one success gate above.
- [ ] Safe autonomous tasks are executed in thin verified slices.
- [ ] Unsafe tasks are documented and deferred instead of guessed.
- [ ] Raw PhD archive/history remains intact.
- [ ] Full current-runtime tests pass before claiming runtime progress.
- [ ] Wiki lint is 100/100 before committing wiki/progress updates.
- [ ] Code review/report exists before declaring "done and reviewed."

---

## Concern Register

| ID | Concern | Risk | Disposition |
|---|---|---|---|
| C1 | Broad Neo4j cleanup could delete historical/local evidence. | Data loss | Defer; only source-scoped dry-run cleanup is safe. |
| C2 | Public export could leak preserved credentials, `.env`, logs, or sensitive datasets. | Privacy/security/publication risk | Human review required; use public-export boundary. |
| C3 | LLM-backed recommendation can spend money and may require live API keys. | Cost/credential risk | Human approval required. |
| C4 | Historical docs overclaim capabilities relative to current runtime. | False completion claim | Review gate must separate current runtime proof from archive evidence. |
| C5 | Legacy Word `.doc` remains unproven through current `/api/analyze`. | Runtime gap | Keep explicit 501 unless a real legacy `.doc` loader is proven. |
| C6 | Source-scoped cleanup exists, but executing it against real smoke-test source refs is still destructive for those scoped records. | Scoped data deletion | Keep as operator-triggered; do not execute automatically. |
| C7 | `/api/batch/analyze` is intentionally 501 until it wraps a proven single-document path. | Runtime gap | Next safe slice now that single-document formats are proven. |
| C8 | FastAPI startup/shutdown deprecation warnings remained in runtime tests. | Maintenance debt | Resolved by Slice 10 lifespan migration. |
| C9 | Legacy `.doc` support would require a separate loader/proof path for old binary Word files. | Runtime gap / scope creep | Keep explicit 501 unless Brian asks for legacy binary Word support. |
| C10 | Public/export and live LLM recommendation are not engineering guesses; they require Brian's privacy/budget decisions. | Human-gated blocker | Do not bypass; continue only safe local runtime work. |

---

## Notes

This is a hybrid recovery program. The safe path is not to make the old thesis system look complete; it is to preserve the record, prove narrow current capabilities, and mark remaining gaps truthfully.
