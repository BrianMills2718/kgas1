# Plan #3: Documentation Quality Pass

**Status:** Complete
**Priority:** Medium
**Created:** 2026-06-26
**Owner:** Agent
**Blocks:** safer thesis wiki/export navigation checks

---

## Gap

After Plan #2 closed safe organization, Brian asked to continue. The next safe layer was documentation-quality verification. The existing markdown checker produced a large failure set because it treated thesis wiki-root links like `/wiki/...` and `/PROGRESS.md` as filesystem absolute paths instead of resolving them into `thesis_record_wiki/`.

## Research

Reviewed before changing behavior:

- `scripts/check_markdown_links.py` for local-link resolution behavior.
- `thesis_record_wiki/wiki/overview.md` for `/PROGRESS.md` wiki-root usage.
- `thesis_record_wiki/wiki/*` link-check output before and after the checker fix.
- `docs` / `investigations` link-check output after the checker fix.
- Existing tests under `tests/current_runtime/` to choose a minimal regression-test location.

## Safe Scope

Fix only deterministic checker false positives for wiki-root links. Do not rewrite historical architecture docs to guess missing targets.

## Files Affected

- `scripts/check_markdown_links.py`
- `tests/current_runtime/test_markdown_link_checker.py`
- `investigations/2026-06-26-documentation-link-quality-pass.md`
- `docs/plans/03_documentation_quality_pass.md`
- `docs/plans/CLAUDE.md`
- `thesis_record_wiki/PROGRESS.md`
- `thesis_record_wiki/wiki/log.md`

## Success Criteria

1. `/wiki/...` links resolve to `thesis_record_wiki/wiki/...`.
2. `/PROGRESS.md` resolves to `thesis_record_wiki/PROGRESS.md`.
3. Regression tests cover both link styles.
4. `thesis_record_wiki/wiki` markdown link check passes.
5. Remaining docs/investigations link failures are recorded instead of guessed.
6. Wiki lint, plan validation, plan sync, and diff checks pass.

## Result

Complete. The checker now handles thesis wiki-root links, the focused tests pass, the wiki link check is clean, and the remaining 72 docs/investigations link failures are recorded as a historical documentation backlog.
