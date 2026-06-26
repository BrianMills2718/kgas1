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
| Narrow E2E Pipeline | `.txt` upload/API path runs real T01/T15/T23/T27/T31/T34/T49 stages with Neo4j source scoping. | live source-scoped smoke tests pass | Satisfied for `.txt` |
| Query Isolation | New graph writes carry `source_refs`; T49 can filter by `source_refs`; old smoke data cannot satisfy scoped proof. | current-runtime tests; live smoke result | Satisfied for tested path |
| Cleanup Safety | Any cleanup command is source-scoped, dry-run capable, backed by tests, and never broad-deletes graph data. | future cleanup test + command docs | Planned, safe if scoped |
| API Honesty | Unsupported/unwired endpoints return explicit 501/503 rather than mock success. | current-runtime API tests | Mostly satisfied for known endpoints |
| Review Gate | Code review pass covers security, destructive actions, LLM boundaries, and remaining stale/mock claims. | review report in `investigations/` or wiki source page | Planned |
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

**Status:** Planned

**Safe scope:** generate a review report over current runtime status, endpoint honesty, and remaining unproven formats. No code changes unless review findings are unambiguous.

**Done when:**
- report lists tests run, pass/fail counts, remaining 501/503 endpoints, and evidence grades;
- current-runtime tests pass;
- wiki records the review result.

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
| C5 | Some formats (`.pdf`, `.docx`, `.md`) remain unproven through current `/api/analyze`. | Runtime gap | Keep explicit 501 until separately proven. |
| C6 | Source-scoped cleanup exists, but executing it against real smoke-test source refs is still destructive for those scoped records. | Scoped data deletion | Keep as operator-triggered; do not execute automatically. |

---

## Notes

This is a hybrid recovery program. The safe path is not to make the old thesis system look complete; it is to preserve the record, prove narrow current capabilities, and mark remaining gaps truthfully.
