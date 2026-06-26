# Historical Variant Review Program

Date: 2026-06-26
Status: Active review guidance

## Purpose

Brian wants the PhD / KGAS record to preserve provenance and evolution. Later files are not automatically better than earlier files. Historical variants should be compared as evidence of design movement, not collapsed into a single "latest wins" narrative.

## Review Principles

1. Preserve raw sources.
   - Do not edit, delete, normalize, or repair `archive_full_record/` as part of review.
   - Treat generated wiki pages as derived navigation, not source truth.

2. Compare variants by claim, not by timestamp alone.
   - A later document can be a regression, a scope shift, a cleanup artifact, or an overclaim.
   - An earlier document can preserve rationale that later implementation docs omit.

3. Separate intent, implementation, validation, and status.
   - Intent: what the system was supposed to become.
   - Implementation: what code/config existed.
   - Validation: what was tested or observed.
   - Status: what was actually true at a point in time.

4. Preserve contradiction explicitly.
   - Do not silently overwrite older claims.
   - Mark supersession, uncertainty, or conflict in wiki pages with source paths and dates.

5. Prefer source-backed deltas.
   - Every material difference should cite the compared files or source-manifest entries.
   - Where runtime evidence exists, it should be labeled separately from documentation claims.

## Difference Review Checklist

For each historical cluster:

- identify all variants and their source paths;
- record timestamps, git commit context, or archive placement when available;
- summarize the strongest claim in each variant;
- identify what changed: scope, architecture, evidence standard, implementation detail, terminology, or status;
- decide whether the change is improvement, regression, reframing, partial supersession, or unresolved tension;
- update the relevant wiki concept/source pages with the conclusion and citations;
- append a `wiki/log.md` entry.

## Recommended Review Order

1. Proposal and dissertation framing variants.
2. Architecture and ADR variants.
3. Theory schema and extraction-pipeline variants.
4. Validation/evidence/status reports.
5. Runtime code path versus documentation claims.
6. UI/workbench claims versus recovered implementations.

## Current Stop Lines

- Do not declare current docs better solely because they are current.
- Do not delete old variants after synthesis.
- Do not execute Neo4j cleanup to make runtime evidence cleaner unless exact source refs and a backup are documented.
- Do not treat legacy binary documents as unsupported forever; if `.doc` files are found later, inventory and preserve them before any conversion attempt.

