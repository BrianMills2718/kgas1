# Plan #5: Approved Preservation Gates

**Status:** Complete
**Priority:** High
**Created:** 2026-06-26
**Owner:** Agent
**Blocks:** private export, live recommendation proof, Neo4j safety decision, legacy document preservation

---

## Gap

Brian approved the remaining safe directions after the PhD/KGAS organization closeout:

- private GitHub is acceptable for the shareable documentation export;
- provenance history must be preserved, and updated docs are not automatically better than old docs;
- live LLM recommendation work may use `gpt-5.4-mini` through `llm_client`;
- Neo4j cleanup is acceptable only when it is safe and does not lose data;
- legacy `.doc` preservation matters if old binary Word content exists.

These approvals need to be converted into bounded, verified implementation slices without mutating raw archives, deleting graph state, or publishing raw material.

## Research

Reviewed before planning:

- `docs/public_export/EXPORT_REVIEW_2026-06-26.md` for docs-only export scan status.
- `docs/plans/01_full_program_completion.md` through `04_final_safe_docs_consolidation.md` for prior gates and stop lines.
- `src/api/cross_modal_api.py`, `src/analytics/mode_selection_service.py`, and `src/analytics/cross_modal_service_registry.py` for `/api/recommend` wiring.
- `~/projects/llm_client/README.md` and local model registry output for `gpt-5.4-mini` availability.
- `scripts/neo4j_source_cleanup.py` and prior Neo4j dump record for cleanup safety.
- Permission-safe `archive_full_record` search for legacy `.doc` / `.docx` files.

## Safe Scope

- Build and push a documentation-only private GitHub export.
- Migrate mode recommendation to the governed `llm_client` adapter and prove a tiny live recommendation call if credentials work.
- Record a provenance/difference-review program for comparing historical/current docs without declaring newer docs better by default.
- Run only non-destructive Neo4j cleanup discovery/dry-run work unless exact deletion scope and backup evidence are recorded.
- Preserve legacy `.doc` concern by documenting current inventory and keeping unsupported `.doc` explicit, not silent.

## Stop Lines

Do not:

- include `archive_full_record/`, `.env`, logs, databases, bundles, tarballs, zip files, or Neo4j dumps in an export;
- publish any export publicly;
- execute Neo4j cleanup deletion without exact source refs and backup confirmation;
- mutate raw preserved archive files;
- silently drop legacy `.doc` content.

## Success Criteria

1. Private GitHub docs-only export exists and has scan/inventory evidence recorded.
2. `/api/recommend` mode-selection LLM path uses `llm_client` with model `gpt-5.4-mini` by default/configuration and has focused tests.
3. A live `/api/recommend` or direct mode-selector smoke is attempted through `llm_client`; result or blocker is recorded.
4. Provenance/difference-review criteria are recorded for comparing historical variants against current docs.
5. Neo4j cleanup remains non-destructive unless exact source refs and backup are documented.
6. Legacy `.doc` preservation status is recorded, including current inventory result.
7. Wiki/progress/plan records are updated, checks pass, changes are committed and pushed.

## Slices

### Slice 1 - Private Docs Export

**Status:** Complete

**Evidence:**
- private repository: `BrianMills2718/kgas-thesis-record`;
- visibility verified as `PRIVATE`;
- export commit: `048568d`;
- local candidate: `exports/private_github_candidate_20260626_105635/`;
- inventory: 258 files, 2.4M;
- forbidden-file scan: zero forbidden file classes;
- record: `docs/public_export/PRIVATE_GITHUB_EXPORT_2026-06-26.md`.

**Done when:**
- [x] fresh docs-only export candidate is built;
- [x] forbidden-file scan is clean;
- [x] private GitHub repository is created or updated;
- [x] export URL and commit are recorded.

### Slice 2 - `llm_client` Recommendation Adapter

**Status:** Complete

**Evidence:**
- new adapter: `src/analytics/governed_llm_client_adapter.py`;
- registry provider: `llm_client`;
- default model: `gpt-5.4-mini`;
- startup environment overrides: `KGAS_LLM_PROVIDER`, `KGAS_RECOMMEND_MODEL`, `KGAS_RECOMMEND_MAX_BUDGET`;
- focused tests: `PYTHONPATH=. pytest tests/current_runtime/test_cross_modal_api_contract.py -q` passed with 29 passed, 5 skipped.

**Done when:**
- [x] mode selector can use a governed `llm_client` adapter;
- [x] default model is `gpt-5.4-mini` unless config/env overrides it;
- [x] focused tests cover adapter wiring without live spend.

### Slice 3 - Live Recommendation Smoke

**Status:** Complete

**Evidence:**
- path: direct `ModeSelectionService.select_optimal_mode(...)` with `GovernedLLMClientAdapter`;
- model requested: `gpt-5.4-mini`;
- model resolved: `openrouter/openai/gpt-5.4-mini`;
- trace id: `kgas.mode_selection:5620148fdd7ef259`;
- result: `graph_analysis`;
- secondary modes: `hybrid_graph_table`, `table_analysis`;
- confidence: `0.97`;
- cost: `$0.000699`;
- usage: 699 total tokens.

**Done when:**
- [x] one tiny live recommendation smoke is attempted;
- [x] model, trace id, result, cost, or blocker is recorded.

### Slice 4 - Provenance Difference Review Program

**Status:** Complete

**Evidence:**
- review guidance: `docs/references/HISTORICAL_VARIANT_REVIEW_PROGRAM_2026-06-26.md`.

**Done when:**
- [x] review criteria are explicit;
- [x] current-vs-historical docs are not treated as a linear improvement chain;
- [x] recommended review order is recorded.

### Slice 5 - Neo4j And Legacy Document Preservation

**Status:** Complete

**Evidence:**
- Neo4j deletion was not executed.
- Existing backup remains the safety checkpoint: `~/archive/phd_thesis_work/neo4j/20260626-075357/neo4j.dump`, SHA-256 `553d57c74eb1ac3619755e3af41be81ebc1dd00fd52e2005b21f7d5cbbb630dc`.
- Cleanup execution remains blocked on exact source refs; broad cleanup is not approved.
- Permission-safe inventory found no `.doc` or `.docx` files under `archive_full_record/`.
- Legacy `.doc` remains explicit 501 in `/api/analyze`; if legacy files are found later, inventory and preserve originals before conversion work.

**Done when:**
- [x] Neo4j cleanup status is non-destructive or exact safe execution evidence is recorded;
- [x] legacy `.doc` inventory status is recorded;
- [x] no data-loss action is performed.
