# Plan #2: Safe Organization Completion

**Status:** Complete
**Priority:** High
**Created:** 2026-06-26
**Owner:** Agent
**Blocks:** final safe organization closeout for PhD/KGAS work

---

## Gap

The PhD/KGAS work now has a preserved raw archive, a Karpathy-style thesis record wiki, current runtime evidence, and public-export policy. The remaining safe organization work needs a precise completion definition so agents do not keep inventing new cleanup tasks or cross into risky actions such as deletion, raw archive mutation, graph cleanup execution, public publishing, or paid LLM calls.

## Research

Reviewed before planning:

- `docs/plans/01_full_program_completion.md` for existing runtime/public-export gates and stop lines.
- `docs/public_export/README.md` and `docs/public_export/EXPORT_MANIFEST_DRAFT.md` for the export boundary and include/exclude model.
- `investigations/2026-06-26-blocked-gates-decision-brief.md` for remaining Brian approval gates.
- `thesis_record_wiki/wiki/concepts/public-export-security-boundary.md` for the no-in-place-sanitization rule.
- Current local export scan outputs under ignored `exports/public_candidate_20260626_094828*` for file count, size, and scan-hit evidence.

## Mission

Finish all safe, local, reversible organization work related to the PhD thesis record. Preserve the raw work history, make the derived wiki/docs/export state navigable, verify the organization surfaces, and stop only when the remaining work is reduced to explicit Brian approval gates.

## Non-Goals / Stop Lines

Do not do any of the following under this plan:

- delete, rewrite, or sanitize `archive_full_record/` in place;
- execute Neo4j cleanup deletion;
- publish to GitHub or any public/private remote beyond normal commits to this repo;
- call paid/external LLM services;
- include `.env`, logs, databases, bundles, compressed archives, or raw full-record snapshots in an export candidate;
- change runtime behavior unless required to verify documentation.

## Files Affected

- `docs/plans/02_safe_organization_completion.md` (create)
- `docs/plans/CLAUDE.md` (active plan index)
- `docs/public_export/EXPORT_REVIEW_2026-06-26.md` (create)
- `thesis_record_wiki/PROGRESS.md` (update closeout state)
- `thesis_record_wiki/wiki/log.md` (record organization closeout)

## Success Criteria

Safe organization is totally done when all of these are true:

1. **Preservation boundary verified:** `archive_full_record/` remains untouched by this plan, and risky paths are documented as excluded from export.
2. **Wiki health verified:** deterministic wiki lint reports health `100/100`.
3. **Plan state verified:** plan validation and plan-status sync pass.
4. **Export candidate reviewed locally:** a temporary documentation-only export candidate is built outside the raw archive, inventoried, size-counted, and secret-pattern scanned.
5. **Export findings recorded:** scan/inventory results are summarized in a committed review artifact, including any false positives or review-needed terms.
6. **Repo state durable:** changes are committed and pushed; the worktree is clean and aligned with `origin/master`.
7. **Remaining work bounded:** any unfinished work is listed only as explicit Brian approval gates, not vague agent follow-up.

## Slices

### Slice 1 - Completion Criteria

**Status:** Complete

**Done when:**
- [x] this plan exists;
- [x] the active plan index points to it;
- [x] success criteria and stop lines are explicit.

### Slice 2 - Local Export Candidate Scan

**Status:** Complete

**Safe scope:** build a temporary docs-only export candidate under ignored `exports/`, excluding raw archives, logs, databases, local env, bundles, and compressed archives.

**Done when:**
- [x] candidate contains only approved documentation/wiki paths;
- [x] inventory and size reports are generated;
- [x] secret-pattern scan is generated.

Evidence: local ignored candidate `exports/public_candidate_20260626_094828/`, 251 files, 2.4M, 74 secret-pattern scan hits, and no forbidden file types found in the candidate.

### Slice 3 - Export Review Artifact

**Status:** Complete

**Done when:**
- [x] export scan findings are summarized in `docs/public_export/EXPORT_REVIEW_2026-06-26.md`;
- [x] publication remains unapproved;
- [x] next approval gates are explicit.

Evidence: `docs/public_export/EXPORT_REVIEW_2026-06-26.md`.

### Slice 4 - Final Verification

**Status:** Complete

**Done when:**
- [x] wiki lint passes;
- [x] Plan #1 and Plan #2 validate;
- [x] plan status sync passes;
- [x] `git diff --check` passes;
- [x] docs-only nature is documented.

Evidence: wiki lint health `100/100`; Plan #1 and Plan #2 validation clean; plan status sync clean; `git diff --check` clean; candidate forbidden-file check returned no files.

### Slice 5 - Closeout

**Status:** Complete

**Done when:**
- [x] progress/wiki log record the closeout;
- [x] commits are pushed;
- [x] worktree is clean;
- [x] no further safe unambiguous organization tasks remain.

Evidence: commit `936e892` records the safe organization criteria and export review; final closeout commit records this slice and the clean pushed state.

## Remaining Approval Gates After Completion

- Brian must approve any actual publication or repository/export creation.
- Brian must approve live LLM recommendation tests and budget.
- Brian must approve exact source refs before Neo4j cleanup execution.
- Brian must request legacy `.doc` support before old binary Word work begins.
