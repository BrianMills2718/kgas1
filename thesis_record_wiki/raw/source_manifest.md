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
| `../archive_full_record/lineage_variants/Digimons_docs/README.md` | Documentation variant overview | `sha256:61a19ab2a30a9705d6c182915ad919fbc5b5b36dde9ea6cad16d6ff1d0ccffec` | Defines the docs repo as complete KGAS documentation with roadmap as status source. |
| `../archive_full_record/lineage_variants/Digimons_docs/docs/architecture/README.md` | Architecture documentation scope | `sha256:a2fe0c370b49d617e8b404a11e7fc85d882c0649981a89be834488b98e4a6203` | Separates target architecture from implementation status. |
| `../archive_full_record/lineage_variants/Digimons_docs/docs/roadmap/ROADMAP_OVERVIEW_CONSERVATIVE.md` | Conservative verified status | `sha256:7f8194cd655b53a788d0f1e39602b6f5926355f8303e7e37d06fb045442dd54f` | Gives cautious 2025-07-31 implementation status and gaps. |
| `../archive_full_record/lineage_variants/Digimons_minimal/README.md` | Minimal clean repo overview | `sha256:77429c375715acc1402c44e9ee50b5d9b6b670c6b29161187b128f02760f9dca` | Same public KGAS framing as current clean repo README. |
| `../archive_full_record/lineage_variants/Digimons_minimal/CLAUDE.md` | Minimal clean repo operating context | `sha256:81cc83432ecd45eb679d901dcb03af2b3915a39c72502bc4d61191cd1ab39a5a` | Captures 2025-09-03 implementation and documentation optimization context. |
| `../archive_full_record/lineage_variants/Digimons_clean_for_real/tool_compatability/README.md` | Tool compatibility overview | `sha256:2c00e214477bfec74a7162152f5122bf945eb046ccc6bf6484b7150b1804e4c8` | Frames type-based composition as the active solution direction. |
| `../archive_full_record/lineage_variants/Digimons_clean_for_real/tool_compatability/DECISION_DOCUMENT.md` | Tool compatibility decision | `sha256:f1ffd9beba57c95fb84b63d9de523f99fbb77b27b0f3a1e0e2c4b840b89bb68e` | Recommends type-based composition after five failed approaches. |
| `../archive_full_record/lineage_variants/Digimons_clean_for_real/tool_compatability/poc/README.md` | Type-based composition POC | `sha256:e997182ac58983769176e3588e88e43639350e37b66d0ed1dcabdc443f08b7a3` | Specifies POC design, test criteria, and migration path. |
