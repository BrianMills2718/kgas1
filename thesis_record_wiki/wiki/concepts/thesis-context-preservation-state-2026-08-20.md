---
type: Concept
title: Thesis Context Preservation State 2026 08 20
description: Current map of the KGAS thesis record, verified raw-archive copies, related external thesis notes, and authority boundaries.
tags: [concept, checkpoint, preservation, recovery, thesis-record, kgas]
created: 2026-08-20
updated: 2026-08-20
status: current
sources:
  - ../../../RECOVERY_ARCHIVE_MANIFEST_2026-04-04.md
  - ../../raw/source_manifest.md
  - ../sources/current-repo-context.md
confidence: high
---

# Summary

The KGAS thesis context is not one folder and should not be reorganized by
moving files out of their source projects. It has four deliberately separate
layers:

1. the current tracked KGAS checkout and its Git history;
2. the ignored raw archive containing filesystem snapshots, lineage variants,
   and portable Git bundles;
3. this derived wiki, which organizes and interprets the record; and
4. related thesis notes in the Second Brain, personal vault, and later Digimon
   application project.

The safe organizing principle is therefore **index across sources, preserve
within sources**. Code and documentation can remain intertwined where that is
part of the historical evidence; the wiki supplies the clean reading paths.

# Current Location Map

| Layer | Current location | Authority / lifecycle |
| --- | --- | --- |
| Tracked KGAS | `/home/brian/code/kgas1` | Current code, tracked documentation, and Git evolution evidence; active line is `master` |
| Desktop raw source | `desktop:/home/brian/projects/phd_thesis_work/archive_full_record/` | Retained recovery source; do not clean or normalize in place |
| Laptop raw mirror | `/home/brian/code/kgas1/archive_full_record/` | Verified byte-preservation mirror, intentionally ignored by Git |
| Compressed backup | `/mnt/c/Users/thela/OneDrive/brian-disk-archive/kgas-thesis/archive_full_record-20260820.tar.gz` | Tested recovery snapshot with adjacent SHA-256 sidecar; present in a OneDrive-managed folder, but cloud-upload completion is not independently verified |
| Thesis wiki | `/home/brian/code/kgas1/thesis_record_wiki/` | Derived navigation and synthesis; raw sources win on conflict |
| Private docs export | `BrianMills2718/kgas-thesis-record` | Private derived documentation copy; not a raw backup |
| Other thesis context | See `raw/source_manifest.md` | Remains under each owning project and is indexed without silent merging |

# Verification Receipt

The desktop source and laptop mirror each contain 234,993 regular files,
26,910 directories, and 242 symlinks. A checksum-mode dry comparison reported
no differences for the normal-account-readable archive. Three root-owned Redis
files were copied through a read-only container mount and verified separately
by SHA-256. All seven portable Git bundles pass `git bundle verify`.

The compressed backup is 2,562,037,008 bytes, contains 262,145 members, passes
`gzip -t`, and has SHA-256:

`dd10bc4783979f4887a2e49fd26d82ee9e85dc7b8492e81c4e31d80eae0d8a9b`

# Current Versus Historical

The active tracked line is `master`. Commit
`2dfab76fe4181a1734001b666b634449d56c69fb`, formerly described through branch
`backup/2026-05-23/phd_thesis_work-master`, is still reachable from `master`
and remains useful historical evidence. The old branch label should not be
used as a present-tense status claim.

Dated pages in this wiki remain valid as dated checkpoints unless newer source
evidence contradicts them. They should not be rewritten merely to sound
current; present-tense entry pages should link to this checkpoint instead.

# Reading Order

For a quick return to the work:

1. this preservation checkpoint;
2. [Thesis Record Reading Guide 2026 06 26](thesis-record-reading-guide-2026-06-26.md);
3. [KGAS Dissertation Claim Map](kgas-dissertation-claim-map.md); and
4. [Current Repo Context](../sources/current-repo-context.md).

Use [Full Record Preservation](full-record-preservation.md) before any cleanup,
deduplication, publication, or archive rewrite.

# Remaining Caveats

- The backup file is present in the OneDrive-managed filesystem, but this
  checkpoint does not claim that the OneDrive service has completed remote
  upload.
- Historical `recovery_inventory_errors.tsv` remains part of the April record.
  The three Redis files implicated by old permissions are now present in the
  laptop mirror, but the historical error record should not be rewritten.
- Private GitHub repositories preserve tracked material only; they do not
  replace the raw archive.
- Public sharing remains a separate reviewed/exported-artifact decision.

# Citations

[1] `../../../RECOVERY_ARCHIVE_MANIFEST_2026-04-04.md`

[2] `../../raw/source_manifest.md`

[3] `../sources/current-repo-context.md`
