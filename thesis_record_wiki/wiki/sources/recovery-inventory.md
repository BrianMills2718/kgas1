---
type: SourceSummary
title: Recovery Inventory
description: Metadata source listing preserved archive variants, readable byte counts, file counts, git heads, and verification errors.
tags: [source, inventory, recovery, verification]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/metadata/recovery_inventory.tsv
  - ../archive_full_record/metadata/recovery_inventory_errors.tsv
  - ../archive_full_record/metadata/git_bundle_status.tsv
confidence: high
---

# Summary

The recovery inventory is the structured metadata for the preservation layer. It records source and destination paths, readable bytes, file counts, directory counts, symlink counts, skipped paths, and git heads where available. [1]

# Preserved Units

| Unit | Kind | Readable Bytes | Files | Dirs | Git Head |
| --- | --- | ---: | ---: | ---: | --- |
| `phd_thesis_work_filesystem_snapshot_20260404_192641` | filesystem snapshot | 790,738,057 | 15,858 | 2,099 | n/a |
| `Digimons_minimal` | lineage variant | 45,378,404 | 1,609 | 357 | `2c59a1f718ec04d6595ba281e18d27f0c12609fe` |
| `Digimons_clean_for_real` | lineage variant | 71,797,823 | 2,992 | 463 | `9b9e999689a7ef4b078ff22503447acfc451825c` |
| `digimons_old` | lineage variant | 5,863,244,246 | 169,698 | 18,817 | `e6c796a5e252b1df78df2393da693f7da5f69c9b` |
| `digimon_core_sparse` | lineage variant | 24,025,718 | 1,116 | 118 | `ERROR` |
| `Digimons_docs` | lineage variant | 10,939,697 | 928 | 310 | `269088673707f7b0f5b9138e7d5bcd8e1117b0e9` |
| `digimon_lineage_Digimons` | lineage variant | 9,334,155,151 | 78,514 | 9,619 | `e14164e818588b474b68eb0e525ea78d9a10ce1c` |
| `digimon-v2` | lineage variant | 1,209,315,652 | 36,166 | 4,799 | `1b980d9edf721be615998ebfa45dfd589d9655b2` |
| `digimon_autoloop` | lineage variant | 94,670,392 | 1,295 | 121 | `ERROR` |

# Verification Errors

The inventory records two permission-denied paths under the filesystem snapshot, repeated during verification. They involve Redis append-only data and Neo4j import data under `archived_2025.0719/data_backups/experimental_implementations/data/`. These should be treated as preserved-but-not-fully-readable verification gaps. [2]

# Git Bundles

Git bundles were successfully created for the current repo and all listed major git-backed variants in `../archive_full_record/git_bundles/`. [3]

# Pages Informed

- [Research Lineage](/wiki/concepts/research-lineage.md)
- [Verification Gaps](/wiki/concepts/verification-gaps.md)
- [Evolution Timeline](/wiki/timeline/evolution-timeline.md)

# Citations

[1] `../archive_full_record/metadata/recovery_inventory.tsv`  
[2] `../archive_full_record/metadata/recovery_inventory_errors.tsv`  
[3] `../archive_full_record/metadata/git_bundle_status.tsv`
