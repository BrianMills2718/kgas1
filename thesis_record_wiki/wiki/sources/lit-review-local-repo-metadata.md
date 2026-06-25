---
type: Source
title: Lit Review Local Repo Metadata
description: Provenance summary for the preserved lit_review .git directory and local .claude settings.
tags: [source, lit-review, git, provenance, local-metadata, trip-backup]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/.git/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/.claude/settings.json
confidence: medium
---

# Summary

The preserved `experiments/lit_review/` folder includes its own `.git/` metadata and a tiny `.claude/settings.json` file. This is source-provenance material, not ordinary thesis content.

The `.claude/settings.json` file only sets long shell timeouts:

- `BASH_DEFAULT_TIMEOUT_MS`: `600000`
- `BASH_MAX_TIMEOUT_MS`: `600000` [2]

# Git Provenance

Using the preserved `.git/` directory with the preserved `lit_review/` worktree reports:

- active branch: `trip-backup-20260224_145348`
- remote: `https://github.com/BrianMills2718/sb_ontologies.git`
- local branches: `main` at `ec1aea3`, `trip-backup-20260224_145348` at `f5d3d38`
- remote branches preserved in metadata: `origin/main` at `ec1aea3`, `origin/trip-backup-20260224_145348` at `f5d3d38`
- total commits reachable from all refs: 15 [1]

`git status --short --branch` reports only the branch line, with no modified/untracked worktree entries visible from the preserved checkout at inspection time. [1]

# Recent History

The last visible commits are:

| Commit | Preserved subject | Interpretation |
|---|---|---|
| `f5d3d38` | `WIP trip backup 20260224_145348: untracked/new files auto-save` | Trip-backup commit adding untracked/new files, including Turner/Social Identity outputs, ELM schema files, schema-creation v10/v13 extractors, and two accidental package-output files. |
| `1c6d13a` | `WIP trip backup 20260224_145348: tracked changes auto-save` | Prior tracked-changes autosave on the trip-backup branch. |
| `ec1aea3` | `feat: Major enhancements to theory extraction and application pipeline` | `main`/`origin/main` head; broad changes to meta-schema docs/prompts, Carter/WorldView/Semantic Hypergraph results, schema application scripts, visualization scripts, and schema variants. |
| `602b71a` | `feat: improve theory extraction to distinguish theory from application` | Earlier extraction-method improvement. |
| `509a53e` | `feat: increase token limits to 180k for O3 model` | Earlier model/config scaling change. |
| `a2cf189` | `feat: implement simplified 4-component meta-schema` | Earlier simplified meta-schema change. |
| `1fbed02` | `feat: implement multi-pass extraction system with 5 specialized passes` | Earlier multipass extraction change. |
| `819ebf0` | `feat: Major improvements to schema extraction pipeline` | Earlier schema-extraction pipeline improvement. |

# Interpretation

This local repository metadata explains why some files in the preserved `lit_review/` tree look like late autosaves rather than polished project artifacts. The `trip-backup` branch contains a WIP autosave over `main`, and the top commit adds the Turner/social-identity extraction outputs and schema-creation extractor variants that are now summarized in the schema and Turner/SIT wiki pages.

The remote URL also shows that this experiment had a separate repository identity, `sb_ontologies`, even though it is currently preserved inside the larger KGAS/Digimons thesis archive. That is important for future reconstruction: exact commit history claims for lit-review work should use this nested `.git/` metadata, while thesis-wide lineage claims should use the outer recovery inventory and current `kgas1` repo history.

# Links

- [Lit Review Theory Extraction Experiment](/wiki/sources/lit-review-theory-extraction-experiment.md)
- [Lit Review Schemas Corpus Inventory](/wiki/sources/lit-review-schemas-corpus-inventory.md)
- [Lit Review Turner Social Identity Extraction](/wiki/sources/lit-review-turner-social-identity-extraction.md)
- [Lit Review Social Identity Theory Schema](/wiki/sources/lit-review-social-identity-theory-schema.md)
- [Lit Review ELM Schema](/wiki/sources/lit-review-elm-schema.md)
- [Schema Extraction Pipeline Evolution](/wiki/concepts/schema-extraction-pipeline-evolution.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/.git/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/.claude/settings.json`
