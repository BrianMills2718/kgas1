---
type: LineageVariant
title: Filesystem Snapshot 2026-04-04
description: Moved snapshot preserving archive-only local material from before the clean repo restore.
tags: [variant, filesystem-snapshot, recovery]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/metadata/recovery_inventory.tsv
  - ../archive_full_record/metadata/recovery_inventory_errors.tsv
confidence: high
---

# Summary

This snapshot lives at `../archive_full_record/filesystem_snapshots/phd_thesis_work_filesystem_snapshot_20260404_192641`. The recovery inventory records it as a moved snapshot from `/home/brian/projects/phd_thesis_work_archives_backup_20260404_192641`, with 790,738,057 readable bytes, 15,858 files, 2,099 directories, 35 symlinks, and 2 skipped paths. [1]

Its role is different from the lineage variants: it preserves archive-only filesystem material that existed around the clean repo restore. It may contain dated backups and data directories that are not represented as clean git histories.

# Known Contents

At shallow depth, the snapshot contains dated archive folders including `archived08060215`, `archived_2025.0719`, `archived_2025.0726`, `archived_2025_08_01`, `archived_2025_08_02`, and `docs_2025.0731`.

# Verification Gaps

Two permission-denied paths were recorded under `archived_2025.0719/data_backups/experimental_implementations/data/`: a Redis append-only directory and a Neo4j import directory. [2]

See [Verification Gaps](/wiki/concepts/verification-gaps.md) before treating those directories as inspected.

# Citations

[1] `../archive_full_record/metadata/recovery_inventory.tsv`  
[2] `../archive_full_record/metadata/recovery_inventory_errors.tsv`
