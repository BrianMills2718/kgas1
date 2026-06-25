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
| `ARCHIVE_BEFORE_CLEANUP_20250805/` | 261 | 8,286,966 | Covered by overview plus proposal-rewrite, phases, initiatives, archived-investigations, analysis, residual-planning, and root-files sub-slices. |
| `theoretical_exploration/` | 107 | 5,854,782 | Covered by overview, proposal-materials, full-example architecture, thinking-out-loud, proposal-evolution, and schema-v14 post-MVP pages; HSPC remains the main optional follow-up. |

# Not-Yet-Represented Areas

The following top-level archive areas were not found in the wiki index by this audit and should be triaged in descending value/risk order:

| Directory | Files | Bytes | Initial Priority |
| --- | ---: | ---: | --- |
| `gemini-review-tool/` | 166 | 3,083,515 | Covered by Gemini review-tool archive overview; deeper architecture-validation/roadmap-critique bundle pages remain optional. |
| `docs_architecture_cleanup_2025_08_29/` | 62 | 2,182,341 | Covered by architecture-cleanup overview; deeper IC uncertainty or generated-doc comparison remains optional. |
| `generated_outputs_2025_08/` | 10 | 1,885,050 | Covered by generated-outputs page; deeper DB trace inspection remains optional. |
| `root_cleanup_2025_08_29/` | 63 | 1,413,590 | Covered by root-cleanup page; deeper specialized Twitter explorer review remains optional. |
| `analysis_validation_2025_08/` | 40 | 1,122,878 | Covered by analysis-validation page; deeper bundle/script-level validation remains optional. |
| `agent_stress_testing/` | 52 | 977,712 | Covered by agent-stress-testing page; deeper replay/test-level validation remains optional. |
| `scripts_archive_2025_08/` | 67 | 626,156 | Covered by scripts-archive-2025-08 page; deeper per-script replay remains optional. |
| `archived_experimental/` | 33 | 576,378 | Covered by archived-experimental-tests page; deeper replay remains optional. |
| `proposal_rewrite_condensed/` | 25 | 333,418 | Covered by proposal-rewrite-condensed page; overlaps with theoretical exploration full-example material. |
| `old_docs_2025_08/` | 12 | 289,025 | Covered by old-docs-2025-08 page; deeper IC uncertainty extraction remains optional. |
| `temp_debug_files/` | 32 | 169,254 | Low unless it contains unique failure evidence. |
| `old_claude_md_versions/` | 7 | 123,514 | Covered by old-CLAUDE-md-versions page. |
| `demos_examples_2025_08/` | 13 | 63,774 | Covered by demos-examples-2025-08 page; overlaps with scripts archive. |
| `archived_implementations/` | 4 | 53,730 | Covered by archived-implementations-ui page; all four files are UI remnants. |
| `doc_generation_scripts/` | 4 | 26,430 | Covered by doc-generation-scripts page; explains generated architecture/ADR bundles. |

# Recommended Next Slice

Next inspect `temp_debug_files/`, the last not-yet-represented top-level archive queue from this audit. It may only need a negative/low-value page if it is scratch debugging. [1]

# Relationship To Wiki

- [Digimon Lineage Archived Uncertainty Tests Overview](/wiki/sources/digimon-lineage-archived-uncertainty-tests-overview.md): most recent large archive subtree brought to represented status.
- [Digimon Lineage Temp Analysis 2025 08](/wiki/sources/digimon-lineage-temp-analysis-2025-08.md): first high-priority not-yet-represented archive queue brought to represented status.
- [Digimon Lineage Archive Before Cleanup 2025 08 05 Overview](/wiki/sources/digimon-lineage-archive-before-cleanup-2025-08-05-overview.md): overview of the pre-cleanup roadmap archive and its sub-slice queue.
- [Digimon Lineage Theoretical Exploration Overview](/wiki/sources/digimon-lineage-theoretical-exploration-overview.md): overview of the theoretical/proposal archive and sub-slice queue.
- [Digimon Lineage Theoretical Exploration Proposal Materials](/wiki/sources/digimon-lineage-theoretical-exploration-proposal-materials.md): proposal-writing and research-planning sub-slice from theoretical exploration.
- [Digimon Lineage Theoretical Exploration Full Example Architecture](/wiki/sources/digimon-lineage-theoretical-exploration-full-example-architecture.md): full-example architecture sub-slice from theoretical exploration.
- [Digimon Lineage Theoretical Exploration Thinking Out Loud](/wiki/sources/digimon-lineage-theoretical-exploration-thinking-out-loud.md): exploratory analysis-philosophy, architecture, and implementation-claim sub-slice from theoretical exploration.
- [Digimon Lineage Theoretical Exploration Proposal Evolution](/wiki/sources/digimon-lineage-theoretical-exploration-proposal-evolution.md): proposal fragments, assessments, historical versions, and critique-response sub-slice from theoretical exploration.
- [Digimon Lineage Theoretical Exploration Schema v14 Post MVP](/wiki/sources/digimon-lineage-theoretical-exploration-schema-v14-post-mvp.md): post-MVP schema-evolution sub-slice from theoretical exploration.
- [Digimon Lineage Proposal Rewrite Condensed](/wiki/sources/digimon-lineage-proposal-rewrite-condensed.md): condensed SCT full-example, StructGPT, uncertainty, and dynamic-tool archive.
- [Digimon Lineage Gemini Review Tool Archive](/wiki/sources/digimon-lineage-gemini-review-tool-archive.md): Gemini review/validation tool archive and generated critique overview.
- [Digimon Lineage Docs Architecture Cleanup 2025 08 29](/wiki/sources/digimon-lineage-docs-architecture-cleanup-2025-08-29.md): architecture cleanup archive and IC uncertainty supersession overview.
- [Digimon Lineage Generated Outputs 2025 08](/wiki/sources/digimon-lineage-generated-outputs-2025-08.md): generated output artifacts, SQLite schemas, and repomix bundle caveats.
- [Digimon Lineage Analysis Validation 2025 08](/wiki/sources/digimon-lineage-analysis-validation-2025-08.md): validation archive with claim checks, configs, bundles, and scripts.
- [Digimon Lineage Agent Stress Testing](/wiki/sources/digimon-lineage-agent-stress-testing.md): agent workflow/adaptive-planning stress-test archive overview.
- [Digimon Lineage Scripts Archive 2025 08](/wiki/sources/digimon-lineage-scripts-archive-2025-08.md): archived debug/demo/fix/old-analysis/test scripts corpus.
- [Digimon Lineage Archived Experimental Tests](/wiki/sources/digimon-lineage-archived-experimental-tests.md): archived redundant functional, stress, and root test corpus.
- [Digimon Lineage Demos Examples 2025 08](/wiki/sources/digimon-lineage-demos-examples-2025-08.md): small demo/example corpus with ontology UI and validation leftovers.
- [Digimon Lineage Archived Implementations UI](/wiki/sources/digimon-lineage-archived-implementations-ui.md): small archived Streamlit UI implementation-remnants bundle.
- [Digimon Lineage Doc Generation Scripts](/wiki/sources/digimon-lineage-doc-generation-scripts.md): generated architecture/ADR bundle script provenance.
- [Digimon Lineage Root Cleanup 2025 08 29](/wiki/sources/digimon-lineage-root-cleanup-2025-08-29.md): root cleanup and duplicate entry-point archive overview.
- [Digimon Lineage Old Claude Md Versions](/wiki/sources/digimon-lineage-old-claude-md-versions.md): historical agent-instruction archive overview.
- [Digimon Lineage Old Docs 2025 08](/wiki/sources/digimon-lineage-old-docs-2025-08.md): superseded old-docs archive covering contract-first, structured-output, operations, and uncertainty notes.
- [Digimon Lineage Old Backups Results](/wiki/sources/digimon-lineage-old-backups-results.md): start of the old-backups represented slice family.
- [Digimon Lineage UI Recovered Components](/wiki/sources/digimon-lineage-ui-recovered-components.md): represented UI/archive UI slice.
- [Progress](../../PROGRESS.md): working queue and commit-level progress tracker.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/`
