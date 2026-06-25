---
type: LineageVariant
title: Digimons Clean For Real
description: Local KGAS snapshot with cleanup-focused branch history.
tags: [variant, digimons, cleanup]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/metadata/recovery_inventory.tsv
confidence: high
---

# Summary

`Digimons_clean_for_real` is preserved at `../archive_full_record/lineage_variants/Digimons_clean_for_real`. The recovery manifest describes it as a local KGAS snapshot with cleanup-focused branch history. The inventory records 71,797,823 readable bytes, 2,992 files, 463 directories, and git head `9b9e999689a7ef4b078ff22503447acfc451825c`. [1]

# Interpretation

This variant likely helps reconstruct cleanup decisions that preceded the current clean repo. It should be compared against [Current Clean Repo](/wiki/variants/current-clean-repo.md) before any conclusion about what was intentionally removed.

The phrase "clean for real" is itself a historical clue: this may represent an intermediate cleanup attempt, not necessarily the final authoritative state. Future ingest should inspect its branch log and docs before using it as a baseline.

# Citations

[1] `../archive_full_record/metadata/recovery_inventory.tsv`
