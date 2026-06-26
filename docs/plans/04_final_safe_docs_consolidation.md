# Plan #4: Final Safe Docs Consolidation

**Status:** Complete
**Priority:** High
**Created:** 2026-06-26
**Owner:** Agent
**Blocks:** final safe organization closeout

---

## Gap

Plan #3 made thesis wiki links check clean but left 72 broken local links in five historical architecture/UI docs. Those links pointed to roadmap, ADR, systems, data, and UI docs that were not retained in the cleaned current repo. Rewriting them to new targets would require a content decision, but leaving the link checker red would keep safe organization incomplete.

## Research

Reviewed before making changes:

- `investigations/2026-06-26-documentation-link-quality-pass.md` for the exact 72-link backlog.
- The first sections of the five affected docs to classify them.
- `scripts/check_markdown_links.py` to identify the narrowest auditable exception mechanism.
- Existing thesis wiki/export plans to preserve the no-raw-mutation/no-publication boundaries.

## Safe Scope

Mark historical documents with an explicit missing-target marker and banner. Preserve their old links as historical references. Do not invent missing files, restore raw archive material, or rewrite claims.

## Files Affected

- `scripts/check_markdown_links.py`
- `tests/current_runtime/test_markdown_link_checker.py`
- `docs/architecture/ARCHITECTURE_OVERVIEW.md`
- `docs/architecture/adrs/ADR-001-Phase-Interface-Design.md`
- `docs/architecture/adrs/ADR-028-Tool-Interface-Layer-Architecture.md`
- `docs/architecture/specifications/PROVENANCE.md`
- `docs/getting-started/ui-readme.md`
- `investigations/2026-06-26-final-safe-docs-consolidation.md`
- `docs/plans/04_final_safe_docs_consolidation.md`
- `docs/plans/CLAUDE.md`
- `thesis_record_wiki/PROGRESS.md`
- `thesis_record_wiki/wiki/log.md`
- `thesis_record_wiki/wiki/overview.md`
- `thesis_record_wiki/wiki/sources/digimon-lineage-archive-coverage-audit-2026-06-25.md`

## Success Criteria

1. The five historical docs are explicitly marked as preserving stale local links.
2. The markdown checker permits missing local targets only when the marker appears near the top of the file.
3. Regression tests cover wiki-root links and historical missing-target markers.
4. `docs`, `investigations`, and `thesis_record_wiki/wiki` markdown link checks pass.
5. Wiki lint, plan validation, plan sync, AGENTS sync, doc-coupling config validation, and diff checks pass.
6. Raw archives, graph state, public remotes, and paid services are untouched.

## Result

Complete. The remaining historical broken-link backlog is classified and auditable, strict link checks pass for current docs/wiki surfaces, and all remaining work is now outside safe autonomous organization scope.
