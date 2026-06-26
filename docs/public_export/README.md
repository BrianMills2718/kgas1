# Public Export Readiness

Date: 2026-06-26
Status: Draft; no export has been approved or published.

## Purpose

This directory defines how to prepare a shareable derivative of the PhD/KGAS record without mutating or exposing the preserved raw archive.

## Safe Default

Do not publish this repository or any archive-derived bundle until Brian reviews and approves the exact export candidate.

The raw record exists for internal preservation:

- `archive_full_record/`
- `logs/`
- `data/`
- local `.env`
- Neo4j dumps and other recovery artifacts

These are not public-export inputs unless a later review explicitly approves a redacted derivative.

## Candidate Shape

The first export candidate should be documentation-first:

- `README.md`
- `RECOVERY_ARCHIVE_MANIFEST_2026-04-04.md`
- `docs/`
- `investigations/`
- `thesis_record_wiki/wiki/`
- `thesis_record_wiki/raw/source_manifest.md`

This candidate should communicate the work history and current runtime evidence without bundling raw logs, databases, environment files, git bundles, tarballs, or preserved full-record snapshots.

## Required Review Before Publishing

1. Build the export candidate into a temporary directory outside the raw archive.
2. Run file inventory and size summary.
3. Run secret-pattern scan.
4. Review for local paths, credentials, API request URLs, full logs, sensitive datasets, and human-identifying research data.
5. Record exclusions and rationale in `EXPORT_MANIFEST_DRAFT.md`.
6. Prefer a private GitHub repository unless Brian explicitly approves public release.

## Stop Conditions

Stop and ask Brian before:

- copying anything from `archive_full_record/` into an export bundle;
- including `.env`, logs, databases, tarballs, zip files, git bundles, or Neo4j dumps;
- publishing to a public remote;
- rewriting raw preserved history to sanitize it.

## Related Files

- `EXPORT_MANIFEST_DRAFT.md`
- `../plans/01_full_program_completion.md`
- `../../investigations/2026-06-26-blocked-gates-decision-brief.md`
- `../../thesis_record_wiki/wiki/concepts/public-export-security-boundary.md`
