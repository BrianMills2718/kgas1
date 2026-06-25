---
type: SourceSummary
title: Digimon Lineage Temp Analysis 2025 08
description: Mechanical archive-analysis bundle from August 2025 with large file listings, uncommitted-file lists, history notes, and helper/debug scripts for KGAS archive cleanup.
tags: [source, digimon-lineage, archive, temp-analysis, provenance, cleanup]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/temp_analysis_2025_08/
confidence: high
---

# Summary

`temp_analysis_2025_08/` is a mechanical provenance and cleanup-analysis bundle inside the large Digimons archive. It contains 16 text/script files totaling 19,659,025 bytes, with aggregate content hash `ce196b41883a6858c924a36d5ec7702334f8b81dc94444bf468719135655d62b`. [1]

Its highest value is not as architecture evidence. It is a map of archive state around August 2025: giant file listings, modification/creation-order listings, a focused August 11-12 file list, uncommitted-file inventory, a few history examples, and utility scripts for architecture-document generation plus Neo4j/SQLite/debug inspection. [1]

# File Inventory

| File | Bytes | Observed Role |
| --- | ---: | --- |
| `all_files_list.txt` | 11,449,494 | Full archive-style file listing with 92,533 lines; includes `.git` internals and generated artifacts, so it is evidence of filesystem state rather than curated source count. [2] |
| `files_by_creation_date.txt` | 8,119,085 | Creation/order-style listing with 92,540 lines; first entries include the temp-analysis files themselves. [3] |
| `august_11_12_files.txt` | 32,677 | 188-line focused list dominated by August 12 proposal/uncertainty rewrite material under `ARCHIVE_BEFORE_CLEANUP_20250805/`. [4] |
| `uncommitted_files.txt` | 8,320 | 262-line list of uncommitted paths including agent commands, `.env.template`, workflows, evidence docs, architecture docs, experiments, outputs, and scripts. [5] |
| `file_history_analysis.txt` | 1,084 | Short examples linking selected files to commit history and moves. [6] |
| `file_summary.txt` | 232 | Small summary claiming last-1000-file statistics; appears incomplete or odd and should not be over-interpreted. [7] |
| `check_file_history.sh` | 997 | Shell helper for checking file history. [1] |
| `check_neo4j_data.py` | 1,771 | Neo4j inspection helper using demo local auth. [8] |
| `check_neo4j_relationships.py` | 1,674 | Neo4j relationship inspection helper using demo local auth. [8] |
| `check_sqlite_data.py` | 4,498 | SQLite inspection helper that can also connect to Neo4j using demo local auth. [8] |
| `concatenate_architecture_docs.py` | 4,161 | Concatenates architecture docs into `KGAS_COMPREHENSIVE_ARCHITECTURE.md`. [9] |
| `split_architecture_docs.py` | 9,473 | Splits `KGAS_COMPREHENSIVE_ARCHITECTURE.md` into four logical architecture documents and creates an index. [9] |
| `extract_all_adrs.py` | 8,642 | Extracts `docs/architecture/adrs/ADR-*.md` into compiled ADR documents and an ADR index. [9] |
| `debug_chunk_refs.py` | 6,505 | Debug helper for chunk-reference association. [8] |
| `debug_relationships.py` | 8,478 | Debug helper for multi-document relationships. [8] |
| `debug_yaml_generation.py` | 1,934 | Debug helper for YAML generation. [8] |

# High-Value Provenance Notes

The focused `august_11_12_files.txt` list points strongly at `archive/ARCHIVE_BEFORE_CLEANUP_20250805/proposal_rewrite_20250812_about_to_drop_all_complex_uncerainty/`, including HSPC materials, IC uncertainty integration docs, proposal old/full-example files, theory schema files, and related generated objects. That makes `ARCHIVE_BEFORE_CLEANUP_20250805/` the right next archive slice after this one. [4]

The `file_history_analysis.txt` examples preserve three useful anchors: an uncertainty stress-test validator first added in the July 24 MCP/API pivot commit, a `VERIFICATION.md` document moved from `docs/current/` to `docs/development/testing/`, and a proposal-rewrite critical analysis file modified on August 12. These are examples, not a complete history reconstruction. [6]

`uncommitted_files.txt` is useful because it records a messy working state rather than a cleaned release state. It includes paths across `.claude`, GitHub workflow config, evidence, architecture, experiments, scripts, and outputs, which supports the broader thesis-record goal of preserving how the work evolved rather than only the final cleaned state. [5]

# Script Caveats

The database/debug scripts are source-code evidence only. They reference local Neo4j/SQLite assumptions and historical project paths, and should not be rerun casually as validation evidence. A targeted credential-pattern scan found only the demo Neo4j password string `testpassword` in this temp-analysis slice, not literal OpenAI or Google API keys. [8]

The architecture document scripts are historically useful because they show an attempted transformation between many architecture docs, a comprehensive architecture document, split architecture sections, and compiled ADR indexes. They should be treated as documentation-generation machinery, not proof that generated docs were current or correct. [9]

# Interpretation

This bundle is best classified as archive provenance. It helps answer "what existed around cleanup time?", "which files were considered uncommitted or recently modified?", and "which generated architecture documents were being assembled?" It should not be used by itself to decide whether any KGAS capability worked at runtime.

# Relationship To Wiki

- [Digimon Lineage Archive Coverage Audit 2026-06-25](digimon-lineage-archive-coverage-audit-2026-06-25.md): this page resolves the first high-priority not-yet-represented archive queue.
- [Digimon Lineage Architecture Docs](digimon-lineage-architecture-docs.md): related target-design documentation bundle that the temp-analysis architecture scripts appear to manipulate.
- [Digimon Lineage Architecture ADRs Map](digimon-lineage-architecture-adrs-map.md): related ADR decision-history slice targeted by the ADR extraction helper.
- [Current Status Verification Discipline](../concepts/current-status-verification-discipline.md): relevant caution for separating file inventory, generated docs, source code, and runtime verification.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/temp_analysis_2025_08/`

[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/temp_analysis_2025_08/all_files_list.txt`

[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/temp_analysis_2025_08/files_by_creation_date.txt`

[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/temp_analysis_2025_08/august_11_12_files.txt`

[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/temp_analysis_2025_08/uncommitted_files.txt`

[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/temp_analysis_2025_08/file_history_analysis.txt`

[7] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/temp_analysis_2025_08/file_summary.txt`

[8] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/temp_analysis_2025_08/{check_neo4j_data.py,check_neo4j_relationships.py,check_sqlite_data.py,debug_chunk_refs.py,debug_relationships.py,debug_yaml_generation.py}`

[9] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/temp_analysis_2025_08/{concatenate_architecture_docs.py,split_architecture_docs.py,extract_all_adrs.py}`
