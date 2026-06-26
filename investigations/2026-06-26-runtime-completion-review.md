# Runtime Completion Review

**Date:** 2026-06-26
**Scope:** Plan #1 runtime completion gates for the recovered KGAS / thesis program.
**Disposition:** Reviewed with blockers documented; current narrow `.txt` runtime is locally proven, but full program completion is not yet claimable.

## Findings

### 1. Non-text document analysis remains explicitly unwired

**Severity:** Medium

`/api/analyze` advertises `.pdf`, `.docx`, `.doc`, `.txt`, and `.md` as allowed extensions, but the current real complete-pipeline path is wired only for `.txt` uploads. Non-text uploads return `501` with an explicit message. This is the right behavior for honesty, but it blocks any claim that document analysis is complete across the advertised file formats.

Evidence:

- `src/api/cross_modal_api.py:155` defines allowed extensions.
- `src/api/cross_modal_api.py:168` returns `501` unless `file_ext == ".txt"`.
- `tests/current_runtime/test_cross_modal_api_contract.py:148` asserts the non-text 501 boundary.

Recommended next safe slice: prove one additional format at a time, starting with a tiny local `.pdf` fixture only if the existing loader path can be verified without modifying preserved archives.

### 2. Batch analysis is still a truthful 501 endpoint

**Severity:** Medium

`/api/batch/analyze` no longer returns mock/demo success. It now returns an explicit `501` because the batch path is not wired to the current document analysis pipeline. This is honest and safer than placeholder output, but it means batch processing is not part of the completed runtime.

Evidence:

- `src/api/cross_modal_api.py:252` defines `/api/batch/analyze`.
- `src/api/cross_modal_api.py:264` returns `501`.
- `tests/current_runtime/test_cross_modal_api_contract.py:428` covers this contract.

Recommended next safe slice: leave blocked until single-document formats are proven; batch should wrap a proven single-document contract rather than create a second unproven path.

### 3. LLM-backed recommendation remains credential and cost gated

**Severity:** Medium

`/api/recommend` is wired to the current mode-selector contract and returns `503` when the selector is unavailable, but a live LLM-backed selector proof is deferred. This is appropriate because it requires external credentials and may spend money.

Evidence:

- `src/api/cross_modal_api.py:220` defines `/api/recommend`.
- `src/api/cross_modal_api.py:230` returns `503` if the mode selector is unavailable.
- `src/api/cross_modal_api.py:245` maps selector runtime failures to `503`.
- Secret/config scan found environment-variable usage and placeholder examples, not a committed local `.env` secret in the scanned scope.

Recommended next safe slice: do not run live LLM recommendation automatically. Add a cost-bounded test plan first, then run only after Brian approves the provider and budget.

### 4. Source-scoped Neo4j cleanup helper is safe by design, but live execution is still destructive for the selected scope

**Severity:** Medium

The cleanup helper rejects broad scopes, supports dry-run mode, and uses exact `source_refs` matching. Execute mode does not use `DETACH DELETE`, but it still deletes relationships and isolated nodes for the selected source ref. It should remain operator-triggered and should not run automatically during completion.

Evidence:

- `scripts/neo4j_source_cleanup.py:34` validates explicit source refs.
- `scripts/neo4j_source_cleanup.py:69` counts scoped relationships and isolated scoped nodes.
- `scripts/neo4j_source_cleanup.py:96` deletes only scoped relationships and isolated scoped nodes.
- `scripts/neo4j_source_cleanup.py:140` exposes dry-run by default and `--execute` as an explicit flag.
- `tests/current_runtime/test_neo4j_source_cleanup.py:40` rejects empty, wildcard, short, and broad graph scopes.
- `tests/current_runtime/test_neo4j_source_cleanup.py:65` asserts execute mode is scoped and does not use `DETACH DELETE`.

Recommended next safe slice: keep using source-scoped query isolation for tests; reserve cleanup execution for a named source ref after a dry-run count is reviewed.

### 5. Public/export readiness is not complete

**Severity:** Medium

The raw archive is intentionally preserved and may contain credentials, logs, local paths, and sensitive historical material. Public export must be derived, scanned, and reviewed. No raw archive mutation should be used as a publication strategy.

Evidence:

- `thesis_record_wiki/wiki/concepts/public-export-security-boundary.md` documents the boundary.
- Secret/config scan found many environment-variable references and placeholder examples across current code/docs, plus prior wiki pages that intentionally record raw-archive credential caveats.

Recommended next safe slice: create a derived export manifest and scan procedure, but do not publish or sanitize the raw archive in place.

### 6. FastAPI lifecycle warnings are non-blocking modernization debt

**Severity:** Low

The current runtime test suite passes, but pytest emits FastAPI lifecycle deprecation warnings. These do not block the current proof, but they should be cleaned up before treating the API as maintained production code.

Recommended next safe slice: migrate startup/shutdown handlers to lifespan only after current completion gates are stable.

## Verified Evidence

The latest verified runtime state recorded during this completion pass:

- `tests/current_runtime`: `58 passed, 2 skipped`.
- Live Neo4j smoke tests:
  - `tests/current_runtime/test_complete_pipeline_neo4j_runtime.py::test_complete_pipeline_processes_tiny_txt_with_neo4j`
  - `tests/current_runtime/test_cross_modal_api_contract.py::test_analyze_document_txt_upload_runs_live_complete_pipeline`
  - Result: `2 passed`.
- Wiki lint: `196` pages, health `100/100`.
- Plan validation: clean for `docs/plans/01_full_program_completion.md`.
- `pip check`: clean.
- Neo4j backup:
  - `~/archive/phd_thesis_work/neo4j/20260626-075357/neo4j.dump`
  - SHA-256 `553d57c74eb1ac3619755e3af41be81ebc1dd00fd52e2005b21f7d5cbbb630dc`
- Source-scoped cleanup dry-run against `storage://document/nonexistent-dry-run-scope`: zero candidate relationships and zero candidate isolated nodes; no data deleted.

## Completion Gate Status

| Gate | Review Status | Evidence Grade |
|---|---|---|
| Preservation | Satisfied for current work; raw archive untouched and wiki lint clean. | A |
| Runtime environment | Satisfied locally with ignored `.env`, healthy Neo4j container, and backup. | A |
| Narrow `.txt` E2E pipeline | Satisfied for live local Neo4j smoke tests. | A |
| Query isolation | Satisfied for new writes and tested T49 query filtering. | A |
| Cleanup safety | Helper and tests complete; live execute remains operator-triggered. | B |
| API honesty | Satisfied for known unwired endpoints checked here. | A |
| Review gate | This report satisfies the initial Plan #1 runtime review bundle. | B |
| Public/export | Deferred for human review. | D |
| LLM recommendation | Deferred for credential/cost approval. | D |

## Deferred Unsafe Or Ambiguous Tasks

- Do not run broad Neo4j cleanup.
- Do not run `scripts/neo4j_source_cleanup.py --execute` automatically against real source refs.
- Do not publish or export raw `archive_full_record/`.
- Do not sanitize raw archive material in place.
- Do not run live LLM-backed recommendation tests without an approved provider and budget.
- Do not claim full multi-format document support until each file type has a live or fixture-backed proof.

## Recommendation

The next safe runtime slice is a narrow `.pdf` capability probe: inspect the existing PDF loader path, create or locate a tiny non-sensitive fixture, and prove whether `/api/analyze` can support exactly that format. If the existing loader path is ambiguous or mutates preserved material, stop at an investigation note and leave `.pdf` at 501.

## Addendum: Later Same-Day Runtime Proofs

After this initial review, Plan #1 advanced through three additional safe slices:

- `5ba418f` proved a tiny `.pdf` upload through the existing T01 complete-pipeline path.
- `1734ce1` proved a tiny `.md` upload through T03 text-compatible loading, added the missing `chardet` dependency, and repaired phase-1 loader provenance calls from `used={}` to `inputs=[]`.
- `52c5fb6` proved a tiny `.docx` upload through T02 Word loading.

Updated verified evidence:

- `tests/current_runtime`: `61 passed, 5 skipped`.
- Live Neo4j document smoke tests:
  - `.txt`
  - `.pdf`
  - `.md`
  - `.docx`
  - Result: `4 passed`.
- Wiki lint remains `196` pages, health `100/100`.
- Plan validation and plan-index sync are clean.
- `pip check` remains clean.

Updated runtime status:

| Format / Endpoint | Status |
|---|---|
| `.txt` `/api/analyze` | Proven through live complete pipeline. |
| `.pdf` `/api/analyze` | Proven for a tiny generated text-bearing PDF through live complete pipeline. |
| `.md` `/api/analyze` | Proven for tiny Markdown as text-compatible input through live complete pipeline. |
| `.docx` `/api/analyze` | Proven for a tiny generated DOCX through live complete pipeline. |
| `.doc` `/api/analyze` | Still explicit 501; no legacy binary Word loader is proven. |
| `/api/batch/analyze` | Still explicit 501. |
| `/api/recommend` live LLM selector | Still deferred for credential/cost approval. |
| public/export bundle | Still deferred for human review. |

The current recommendation changes accordingly: do not spend more effort on legacy `.doc` unless Brian explicitly wants old binary Word support. The next safe engineering slice is FastAPI lifespan modernization, because runtime tests consistently pass with `@app.on_event` deprecation warnings.
