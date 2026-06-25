---
type: LineageVariant
title: Digimons Minimal
description: Local repo matching kgas1 origin/master commit 2c59a1f.
tags: [variant, digimons, minimal]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/metadata/recovery_inventory.tsv
confidence: high
---

# Summary

`Digimons_minimal` is preserved at `../archive_full_record/lineage_variants/Digimons_minimal`. The recovery manifest describes it as a local repo matching `kgas1` `origin/master` commit `2c59a1f`. The inventory records 45,378,404 readable bytes, 1,609 files, 357 directories, and git head `2c59a1f718ec04d6595ba281e18d27f0c12609fe`. [1]

# Interpretation

This appears to be a compact clean-lineage reference point. Use it when comparing the cleaned tracked repo against the preserved historical variants.

Because it is small and has a known git head, it is a good candidate for the next detailed ingest pass. Compare its README, CLAUDE, docs, and source layout against [Current Clean Repo](/wiki/variants/current-clean-repo.md) and [Digimons Clean For Real](/wiki/variants/digimons-clean-for-real.md).

Initial ingest confirms it is a clean reference line with the same public KGAS framing as the current repo README, but with internal `CLAUDE.md` context that still foregrounds the tool-compatibility vertical slice and unresolved implementation gaps. [2] This makes it useful for comparing public-facing cleanup against internal engineering reality.

# Related Pages

- [Digimons Minimal Clean Reference](/wiki/sources/digimons-minimal-clean-reference.md)
- [Documentation Status Truthfulness](/wiki/concepts/documentation-status-truthfulness.md)

# Citations

[1] `../archive_full_record/metadata/recovery_inventory.tsv`  
[2] `../archive_full_record/lineage_variants/Digimons_minimal/CLAUDE.md`
