---
type: SourceSummary
title: Recovery Archive Manifest 2026-04-04
description: Canonical source explaining the full-record archive layer and preserved KGAS / Digimons lineage material.
tags: [source, recovery, archive-full-record]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../RECOVERY_ARCHIVE_MANIFEST_2026-04-04.md
confidence: high
---

# Summary

The recovery archive manifest records full-record preservation work done on 2026-04-04 for the KGAS / Digimons lineage. It states that the tracked `kgas1` checkout restored a clean repository state but did not include the full local filesystem record. The archive layer under `../archive_full_record/` was created to avoid losing material that may have been intentionally or unintentionally removed from the tracked repo. [1]

# Key Takeaways

- `archive_full_record/` is ignored by git intentionally and is a preservation area, not normal working-tree content. [1]
- The archive contains a moved filesystem snapshot, multiple lineage variant copies, recovery metadata, and portable git bundles. [1]
- Several experimental files were restored into the tracked repo because they appeared to have been missed from the working path. [1]
- The manifest's next step is deliberate curation from `archive_full_record/`, one recovery slice at a time. [1]

# Pages Informed

- [Full Record Preservation](/wiki/concepts/full-record-preservation.md)
- [Evolution Timeline](/wiki/timeline/evolution-timeline.md)
- [Current Clean Repo](/wiki/variants/current-clean-repo.md)
- [Overview](/wiki/overview.md)

# Citations

[1] `../RECOVERY_ARCHIVE_MANIFEST_2026-04-04.md`
