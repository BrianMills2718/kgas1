# Thesis Record Wiki - Schema

This directory is a Karpathy-style LLM wiki for Brian's KGAS / Digimons / PhD thesis work record.

The purpose is preservation and orientation: keep a durable, source-grounded map of what existed, how the project evolved, what was attempted, what was recovered, and what remains historically important after Brian left the thesis program. A good answer from this wiki should distinguish raw evidence from derived synthesis, cite the underlying source paths, and preserve uncertainty instead of smoothing over messy history.

## Layers

- `raw/` is the source manifest layer. It does not duplicate the 17G archive; it points to immutable source roots already in this repo.
- `wiki/` is the derived navigation and synthesis layer maintained by agents.
- `CLAUDE.md` is this operating contract. Update it when Brian corrects the preservation model.

## Source Of Truth

The source of truth is the existing repo plus the ignored preservation layer:

- `../RECOVERY_ARCHIVE_MANIFEST_2026-04-04.md`
- `../archive_full_record/metadata/recovery_inventory.tsv`
- `../archive_full_record/metadata/recovery_inventory_errors.tsv`
- `../archive_full_record/metadata/git_bundle_status.tsv`
- `../archive_full_record/filesystem_snapshots/`
- `../archive_full_record/lineage_variants/`
- `../archive_full_record/git_bundles/`
- Current tracked repo files and git history.

Do not move, delete, rewrite, normalize, or deduplicate raw archive material from this wiki. The wiki is allowed to summarize and index it.

## Directory Map

```text
raw/
  README.md
  source_manifest.md
wiki/
  index.md
  log.md
  overview.md
  sources/
  entities/
  concepts/
  timeline/
  variants/
  _candidates/
```

## Page Types

Every wiki page except `index.md` and `log.md` uses YAML frontmatter with `type`.

Allowed initial types:

- `SourceSummary`
- `Entity`
- `Concept`
- `Timeline`
- `LineageVariant`
- `OpenQuestion`

Recommended frontmatter:

```yaml
---
type:
title:
description:
tags: []
created:
updated:
sources: []
confidence: high | medium | low | speculative
---
```

## House Rules

- Preserve the full historical record before cleanup. No cleanup action should happen inside `archive_full_record/` until the relevant material has a wiki page and a recovery path.
- Treat the wiki as derived. If a wiki page conflicts with a raw source, the raw source wins.
- Use citations to repo-relative paths for every non-obvious claim.
- Mark contradictions and uncertainty explicitly. Do not silently reconcile conflicting project status claims.
- Separate phases of evolution: early Digimons experiments, KGAS implementation, cleaned external-evaluation repo, recovery archive, and post-program preservation.
- The current branch `backup/2026-05-23/phd_thesis_work-master` is itself part of the historical record, not just the active branch.
- Do not copy the 17G archive into `raw/`; use manifests and path citations to avoid duplication.

## Ingest Workflow

1. Add or identify the source path in `raw/source_manifest.md`.
2. Record source hash when the source is small enough to hash cheaply.
3. Write or update a page in `wiki/sources/`.
4. Integrate across `wiki/variants/`, `wiki/timeline/`, `wiki/concepts/`, and `wiki/entities/`.
5. Update `wiki/index.md`.
6. Append to `wiki/log.md`.

## Query Workflow

1. Read `wiki/index.md`.
2. Read relevant wiki pages.
3. Drill into raw paths only when precision matters.
4. Answer with confidence and citations.
5. File durable answers back into the wiki when they clarify the project history.

## Initial Preservation Priorities

1. Map `archive_full_record/lineage_variants/` into one page per variant.
2. Map `archive_full_record/filesystem_snapshots/` without modifying permission-protected paths.
3. Build an evolution timeline from git history plus recovery metadata.
4. Identify high-value thesis ideas and implementation evidence without claiming completion beyond the evidence.
5. Only after the wiki is useful, decide whether any archive payload should move to central `~/archive`.
