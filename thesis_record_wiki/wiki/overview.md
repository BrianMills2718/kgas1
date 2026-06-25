---
type: Concept
title: Thesis Record Overview
description: Top-level map of the KGAS / Digimons thesis work record and how to preserve it.
tags: [thesis-record, kgas, preservation]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../RECOVERY_ARCHIVE_MANIFEST_2026-04-04.md
  - ../archive_full_record/metadata/recovery_inventory.tsv
  - ../README.md
  - ../CLAUDE.md
confidence: medium
---

# Summary

This wiki is a derived navigation layer for Brian's KGAS / Digimons / PhD thesis work. The central preservation fact is that the tracked repo is only the cleaned working checkout; the full local history was intentionally preserved under `../archive_full_record/` on 2026-04-04. The recovery manifest says that the archive layer exists to avoid losing material that may have been intentionally or unintentionally removed from the tracked repo during cleanup. [1]

The current tracked repo presents KGAS as an academic research GraphRAG system connected to the dissertation topic "Theoretical Foundations for LLM-Generated Ontologies and Analysis of Fringe Discourse." [2] The current `CLAUDE.md` also preserves older operational context about tool compatibility, vertical slices, uncertainty propagation, provenance, reasoning traces, and documentation cleanup. [3]

# Preservation Model

The archive is not clutter to delete. It is the full record of project evolution. The wiki should make that record navigable while keeping raw sources unchanged.

Initial major source buckets:

- [Current Clean Repo](/wiki/variants/current-clean-repo.md)
- [Filesystem Snapshot 2026-04-04](/wiki/variants/filesystem-snapshot-2026-04-04.md)
- [Digimon Lineage Digimons](/wiki/variants/digimon-lineage-digimons.md)
- [Digimons Old](/wiki/variants/digimons-old.md)
- [Digimon v2](/wiki/variants/digimon-v2.md)
- Smaller lineage variants under `../archive_full_record/lineage_variants/`

Initial source summaries:

- [Recovery Archive Manifest 2026-04-04](/wiki/sources/recovery-archive-manifest-2026-04-04.md)
- [Recovery Inventory](/wiki/sources/recovery-inventory.md)
- [Current Repo Context](/wiki/sources/current-repo-context.md)

# What This Wiki Is For

- Reconstruct how the thesis system evolved over time.
- Separate raw evidence from later cleanup and interpretation.
- Preserve abandoned or superseded ideas without polluting active repo search.
- Make future archive decisions reversible and explainable.
- Help Brian recover the intellectual arc of the work after leaving the thesis program.

The first pass is intentionally conservative: it identifies buckets and risks rather than attempting to flatten the record into one narrative. The next pass should ingest variant README/CLAUDE/docs files and then revise the lineage pages with more confident descriptions.

# Current Cautions

- Two permission-denied paths are recorded in the recovery errors metadata; treat those as verification gaps, not as proof of absence. [4]
- Some lineage variants have `destination_git_head` recorded as `ERROR`; those need later review rather than deletion. [5]
- The active branch is `backup/2026-05-23/phd_thesis_work-master`, not `master`, and includes post-backup commits. [6]
- See [Verification Gaps](/wiki/concepts/verification-gaps.md) before interpreting missing data or failed git-head reads.

# Citations

[1] `../RECOVERY_ARCHIVE_MANIFEST_2026-04-04.md`  
[2] `../README.md`  
[3] `../CLAUDE.md`  
[4] `../archive_full_record/metadata/recovery_inventory_errors.tsv`  
[5] `../archive_full_record/metadata/recovery_inventory.tsv`  
[6] Git history on `backup/2026-05-23/phd_thesis_work-master`, HEAD `2dfab76fe4181a1734001b666b634449d56c69fb`
