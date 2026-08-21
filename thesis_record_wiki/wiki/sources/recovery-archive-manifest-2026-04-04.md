---
type: SourceSummary
title: Recovery Archive Manifest 2026-04-04
description: Canonical source explaining the full-record archive layer and preserved KGAS / Digimons lineage material.
tags: [source, recovery, archive-full-record]
created: 2026-06-25
updated: 2026-08-20
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
- On 2026-08-20 the intact desktop archive was mirrored into `/home/brian/code/kgas1/archive_full_record/`; counts match at 234,993 files, 26,910 directories, and 242 symlinks, and checksum comparison reported no content differences. [1]
- A 2,562,037,008-byte gzip snapshot with 262,145 members and an adjacent SHA-256 sidecar now exists in the OneDrive-managed backup folder. Its local integrity is verified; completed cloud upload is not claimed. [1]
- All seven portable Git bundles verify in the laptop mirror. [1]
- The manifest's next step is deliberate curation from `archive_full_record/`, one recovery slice at a time. [1]

# Pages Informed

- [Full Record Preservation](/wiki/concepts/full-record-preservation.md)
- [Evolution Timeline](/wiki/timeline/evolution-timeline.md)
- [Current Clean Repo](/wiki/variants/current-clean-repo.md)
- [Overview](/wiki/overview.md)
- [Thesis Context Preservation State 2026 08 20](/wiki/concepts/thesis-context-preservation-state-2026-08-20.md)

# Citations

[1] `../RECOVERY_ARCHIVE_MANIFEST_2026-04-04.md`
