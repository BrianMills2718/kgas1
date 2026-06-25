---
type: Concept
title: Full Record Preservation
description: Preserve the complete messy thesis work record before cleanup or archival decisions.
tags: [preservation, archive, policy]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../RECOVERY_ARCHIVE_MANIFEST_2026-04-04.md
  - ../archive_full_record/metadata/recovery_inventory.tsv
confidence: high
---

# Summary

Full record preservation means the clean tracked repo is not enough. The preservation target is the complete surviving local record of how the thesis work evolved, including abandoned branches, old snapshots, large local archives, evidence files, prototypes, and verification metadata.

# Rules For This Repository

- Do not delete archive payloads from `../archive_full_record/` as part of cleanup.
- Do not assume a large directory is redundant because a clean repo exists.
- Before moving any archive payload to central `~/archive`, create or update wiki pages that say what it is and how to recover it.
- Keep recovery metadata first-class because it records what was copied, skipped, and bundled.

# Why This Matters

The recovery manifest explicitly says the archive layer was created because the clean checkout did not include the full local filesystem record. [1] The inventory shows multiple distinct lineage variants with different sizes, file counts, and git heads. [2]

# Citations

[1] `../RECOVERY_ARCHIVE_MANIFEST_2026-04-04.md`  
[2] `../archive_full_record/metadata/recovery_inventory.tsv`
