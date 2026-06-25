# Source Manifest

Initial source set for the Thesis Record Wiki.

| Source | Role | Hash / Identifier | Notes |
| --- | --- | --- | --- |
| `../CLAUDE.md` | Current repo operating context | `sha256:98452e6a49a4f7f71d7e0897441993f9c71507ace0fb0cb85aa9f97632a17273` | Contains KGAS implementation guide and historical sprint/status notes. |
| `../README.md` | Current repo public overview | `sha256:77429c375715acc1402c44e9ee50b5d9b6b670c6b29161187b128f02760f9dca` | Describes KGAS as academic research GraphRAG system. |
| `../RECOVERY_ARCHIVE_MANIFEST_2026-04-04.md` | Recovery manifest | `sha256:f240caafa14019e520e3c2c4314a10c86f7542ed66b6c22519f843d84293bcdc` | Canonical explanation of why `archive_full_record/` exists. |
| `../archive_full_record/metadata/recovery_inventory.tsv` | Recovery inventory | `sha256:91f664414195a82927b3f69655fc3d2f47c6474dfcf869c519d8d3e73f030e95` | Source/destination sizes, counts, skipped paths, git heads. |
| `../archive_full_record/metadata/recovery_inventory_errors.tsv` | Recovery verification errors | `sha256:68656353f6d6e4b945702ffcc5f0e730b50ca25b2ba8b0e29fb58006f0310509` | Permission-denied paths encountered during verification. |
| `../archive_full_record/metadata/git_bundle_status.tsv` | Git bundle status | `sha256:b00e35aacf943da3872a4755f610566f50b1f0c37c4e0fd13e3471222baae730` | Lists bundle creation status for preserved repos. |
| `../archive_full_record/filesystem_snapshots/` | Preserved filesystem snapshots | Directory source | Large raw preservation area; do not modify from wiki tasks. |
| `../archive_full_record/lineage_variants/` | Preserved lineage variants | Directory source | Largest source set; one wiki page per major variant. |
| `../archive_full_record/git_bundles/` | Portable git bundles | Directory source | Recovery artifacts for current repo and major preserved variants. |
| Git history on `backup/2026-05-23/phd_thesis_work-master` | Evolution evidence | `HEAD 2dfab76fe4181a1734001b666b634449d56c69fb` | Includes cleanup, recovery, and post-backup commits. |
