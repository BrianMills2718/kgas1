# Implementation Plans

Track all implementation work here.

## Active Plans

| # | Name | Priority | Status | Blocks |
|---|------|----------|--------|--------|
| 1 | [Full Program Completion](01_full_program_completion.md) | High | ⏸️ Blocked | runtime hardening, review gate, public/export packaging |
| 2 | [Safe Organization Completion](02_safe_organization_completion.md) | High | ✅ Complete | final safe organization closeout |
| 3 | [Documentation Quality Pass](03_documentation_quality_pass.md) | Medium | ✅ Complete | safer thesis wiki/export navigation checks |
| 4 | [Final Safe Docs Consolidation](04_final_safe_docs_consolidation.md) | High | ✅ Complete | final safe organization closeout |

## Status Key

| Status | Meaning |
|--------|---------|
| Planned | Ready to implement |
| In Progress | Being worked on |
| Blocked | Waiting on dependency |
| Complete | Implemented and verified |

## Creating a New Plan

1. Copy `TEMPLATE.md` to `NN_name.md`
2. Fill in gap, steps, required tests
3. Add to this index
4. Commit with `[Plan #N]` prefix

## Trivial Changes

Not everything needs a plan. Use `[Trivial]` for:
- Less than 20 lines changed
- No changes to `src/` (production code)
- No new files created

```bash
git commit -m "[Trivial] Fix typo in README"
```

## Completing Plans

```bash
python scripts/meta/complete_plan.py --plan N
```

This verifies tests pass and records completion evidence.
