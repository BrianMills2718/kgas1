# Documentation Link Quality Pass

Date: 2026-06-26
Status: Safe checker fix complete; historical docs backlog recorded.

## Scope

Brian asked to continue after safe organization closeout. This pass checked the organized thesis documentation surfaces without deleting, publishing, mutating raw archive content, running graph cleanup, or calling external LLM services.

## Checks Run

- `.venv/bin/python scripts/check_markdown_links.py --repo-root . thesis_record_wiki/wiki`
- `.venv/bin/python scripts/check_markdown_links.py --repo-root . docs investigations`
- `.venv/bin/python scripts/check_doc_coupling.py --validate-config`
- `.venv/bin/python scripts/meta/check_agents_sync.py --repo-root . --check`
- `.venv/bin/python scripts/meta/validate_plan.py --plan-file docs/plans/01_full_program_completion.md`
- `.venv/bin/python scripts/meta/validate_plan.py --plan-file docs/plans/02_safe_organization_completion.md`
- `.venv/bin/python scripts/sync_plan_status.py --check`

## Finding 1 - Wiki Link Checker False Positives

Initial link-check output treated `/wiki/...` and `/PROGRESS.md` as filesystem absolute paths. That produced many false positives against the thesis record wiki even though the Karpathy wiki lint reported health `100/100`.

Fix:

- `/wiki/...` now resolves to `thesis_record_wiki/wiki/...`.
- `/PROGRESS.md` now resolves to `thesis_record_wiki/PROGRESS.md`.
- Added focused regression tests in `tests/current_runtime/test_markdown_link_checker.py`.

Result:

- `tests/current_runtime/test_markdown_link_checker.py`: 2 passed.
- `thesis_record_wiki/wiki` markdown link check: clean.

## Finding 2 - Historical Docs Link Backlog

After the wiki-root fix, `docs` and `investigations` still have 72 broken local markdown links. They are concentrated in old architecture/UI documentation:

| File | Broken links |
| --- | ---: |
| `docs/architecture/ARCHITECTURE_OVERVIEW.md` | 56 |
| `docs/architecture/adrs/ADR-028-Tool-Interface-Layer-Architecture.md` | 6 |
| `docs/architecture/adrs/ADR-001-Phase-Interface-Design.md` | 5 |
| `docs/getting-started/ui-readme.md` | 4 |
| `docs/architecture/specifications/PROVENANCE.md` | 1 |

These appear to reference historical roadmap, ADR, systems, data, and UI documentation that is not present in the cleaned current repo. Some corresponding sources are preserved under `archive_full_record/` and represented in the thesis wiki, but rewriting these links would require a content decision: should the current docs point to archived raw files, synthesized wiki pages, or be marked historical/stale?

Safe disposition:

- Do not guess replacements in this pass.
- Keep this as a documented backlog for a future architecture-doc consolidation pass.
- If the docs are meant for publication, decide whether to archive these historical docs, rewrite them against current wiki pages, or restore selected historical targets as derived docs.

## Other Checks

- Doc-coupling config validation passed.
- `AGENTS.md` sync check passed.
- Plan #1 and Plan #2 validation passed.
- Plan status sync passed.

## Recommendation

The next safe documentation step, if Brian wants to continue, is an explicit architecture-doc consolidation pass:

1. Decide whether `docs/architecture/ARCHITECTURE_OVERVIEW.md` is current authoritative documentation or historical provenance.
2. If current, rewrite links to current docs/wiki pages and remove dead roadmap/ADR references.
3. If historical, add a stale/historical banner and exclude it from strict publication checks.
4. Do the same classification for the two ADR files and `docs/getting-started/ui-readme.md`.
