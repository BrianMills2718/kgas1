---
type: SourceSummary
title: Digimon Lineage Archive Coverage Audit 2026-06-25
description: Coverage audit for top-level directories under the large Digimons lineage archive, identifying represented and not-yet-represented archive areas after uncertainty-tree ingest.
tags: [source, digimon-lineage, archive, coverage-audit, progress]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/
confidence: high
---

# Summary

This audit covers top-level directories under `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/` after the old-backups, UI, uncertainty, archived-uncertainty, evidence, generated-report, and archived-root-test ingests. The archive is about 226 MB total. [1]

The audit is not a substitute for source summaries. It is a queue-control page: it records which archive areas are already represented in the wiki and which substantial areas remain. [1]

# Represented Areas

The following top-level archive areas are already represented by dedicated wiki pages or covered as duplicate/parent material:

| Directory | Files | Bytes | Coverage Note |
| --- | ---: | ---: | --- |
| `archived_root_tests/` | 100 | 1,364,253 | Covered by archived-root-tests page. |
| `archived_uncertainty_tests/` | 178 | 59,048,419 | Covered by archived-uncertainty overview, docs/validation, datasets, and code-delta pages. |
| `evidence/` | 51 | 344,272 | Covered by evidence archive/current pages. |
| `evidence_reports_2025_08/` | 13 | 108,279 | Covered by evidence reports page. |
| `generated_reports/` | 82 | 2,762,642 | Covered by generated reports page. |
| `old_backups_2025_08/` | 301 | 32,979,605 | Covered by old-backups slice family. |
| `ui/` | 95 | 38,046,097 | Covered by recovered UI corpus page. |
| `ui_archive_2025_08/` | 91 | 37,992,367 | Covered as near-duplicate UI archive. |
| `uncertainty_stress_test/` | 95 | 9,218,730 | Covered by uncertainty stress-test slice family. |
| `validation-reports/` | 2 | 30,030 | Covered by old-backups validation reports page. |
| `temp_analysis_2025_08/` | 16 | 19,659,025 | Covered by temp-analysis provenance page. |
| `ARCHIVE_BEFORE_CLEANUP_20250805/` | 261 | 8,286,966 | Covered by overview page; proposal-rewrite, phases, and initiatives still need dedicated sub-slices. |

# Not-Yet-Represented Areas

The following top-level archive areas were not found in the wiki index by this audit and should be triaged in descending value/risk order:

| Directory | Files | Bytes | Initial Priority |
| --- | ---: | ---: | --- |
| `theoretical_exploration/` | 107 | 5,854,782 | High: proposal materials and theoretical/exploratory thesis framing likely important to intellectual lineage. |
| `gemini-review-tool/` | 166 | 3,083,515 | Medium-high: external review tool outputs/bundles may contain critique/evaluation provenance. |
| `docs_architecture_cleanup_2025_08_29/` | 62 | 2,182,341 | Medium-high: architecture cleanup docs and generated architecture documents; likely overlaps with architecture pages but needs explicit audit. |
| `generated_outputs_2025_08/` | 10 | 1,885,050 | Medium: generated output artifacts, including `reasoning_traces.db` and clean-architecture bundle. |
| `root_cleanup_2025_08_29/` | 63 | 1,413,590 | Medium: root cleanup artifact and specialized app remnants. |
| `analysis_validation_2025_08/` | 40 | 1,122,878 | Medium: validation artifacts requiring evidence-level review. |
| `agent_stress_testing/` | 52 | 977,712 | Medium: agent workflow/memory/research stress tests. |
| `scripts_archive_2025_08/` | 67 | 626,156 | Medium-low: script archive likely best handled as corpus inventory. |
| `archived_experimental/` | 33 | 576,378 | Medium-low: experimental tests/utilities; likely corpus inventory. |
| `proposal_rewrite_condensed/` | 25 | 333,418 | Medium-low but conceptually important; likely overlaps with theoretical exploration. |
| `old_docs_2025_08/` | 12 | 289,025 | Medium-low: old docs bundle; check for supersession relationships. |
| `temp_debug_files/` | 32 | 169,254 | Low unless it contains unique failure evidence. |
| `old_claude_md_versions/` | 7 | 123,514 | Medium: historical agent instructions may explain policy evolution. |
| `demos_examples_2025_08/` | 13 | 63,774 | Low-medium: demo/example corpus. |
| `archived_implementations/` | 4 | 53,730 | Low-medium: small implementation remnants. |
| `doc_generation_scripts/` | 4 | 26,430 | Low: scripts for docs generation unless tied to generated claims. |

# Recommended Next Slice

Next ingest `ARCHIVE_BEFORE_CLEANUP_20250805/proposal_rewrite_20250812_about_to_drop_all_complex_uncerainty/`, then `theoretical_exploration/` as a separate bounded slice because it is more likely to preserve thesis evolution and proposal reasoning. [1]

# Relationship To Wiki

- [Digimon Lineage Archived Uncertainty Tests Overview](/wiki/sources/digimon-lineage-archived-uncertainty-tests-overview.md): most recent large archive subtree brought to represented status.
- [Digimon Lineage Temp Analysis 2025 08](/wiki/sources/digimon-lineage-temp-analysis-2025-08.md): first high-priority not-yet-represented archive queue brought to represented status.
- [Digimon Lineage Archive Before Cleanup 2025 08 05 Overview](/wiki/sources/digimon-lineage-archive-before-cleanup-2025-08-05-overview.md): overview of the pre-cleanup roadmap archive and its sub-slice queue.
- [Digimon Lineage Old Backups Results](/wiki/sources/digimon-lineage-old-backups-results.md): start of the old-backups represented slice family.
- [Digimon Lineage UI Recovered Components](/wiki/sources/digimon-lineage-ui-recovered-components.md): represented UI/archive UI slice.
- [Progress](../../PROGRESS.md): working queue and commit-level progress tracker.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/`
