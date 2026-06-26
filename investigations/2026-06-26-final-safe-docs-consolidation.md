# Final Safe Docs Consolidation

Date: 2026-06-26
Status: Complete.

## Scope

This pass completes the remaining safe documentation organization work after Plan #3. It addresses the 72 broken local links in historical architecture/UI docs without guessing replacement targets.

## Classification

The affected files are treated as historical target-architecture or UI provenance documents:

- `docs/architecture/ARCHITECTURE_OVERVIEW.md`
- `docs/architecture/adrs/ADR-001-Phase-Interface-Design.md`
- `docs/architecture/adrs/ADR-028-Tool-Interface-Layer-Architecture.md`
- `docs/architecture/specifications/PROVENANCE.md`
- `docs/getting-started/ui-readme.md`

The missing targets were roadmap, ADR, systems, data, and UI docs not retained in the cleaned current repo. Some corresponding evidence exists in `archive_full_record/` and the thesis record wiki, but mapping every historical target to a new page would be interpretive work. This pass preserves the old links as historical references rather than rewriting them.

## Implementation

Added an explicit marker recognized by `scripts/check_markdown_links.py`:

```text
link-check: allow-missing-historical-targets
```

The marker must appear near the top of a markdown file. When present, missing local targets are permitted for that file. The marker is paired with a visible banner explaining that the document is historical and that stale links are intentionally preserved.

## Verification

Passing checks:

- `.venv/bin/python -m pytest tests/current_runtime/test_markdown_link_checker.py -q`
- `.venv/bin/python scripts/check_markdown_links.py --repo-root . docs investigations`
- `.venv/bin/python scripts/check_markdown_links.py --repo-root . thesis_record_wiki/wiki`

The focused test now covers:

- `/wiki/...` thesis wiki-root links.
- `/PROGRESS.md` wiki progress links.
- Historical missing-target marker behavior.

## Safety Boundaries

This pass did not:

- mutate `archive_full_record/`;
- restore raw archive docs into the cleaned repo;
- delete any content;
- publish an export;
- execute Neo4j cleanup;
- call external LLM services.

## Remaining Non-Safe / Approval-Gated Work

- Human review before any public/private export publication.
- Human decision if historical architecture docs should be rewritten into current authoritative docs.
- Human approval before exact-source Neo4j cleanup execution.
- Human approval/budget before live LLM recommendation tests.
