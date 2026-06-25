---
type: Source
title: Digimon Lineage Old Backups Current Coverage
description: Preserved 21 MB HTML coverage report for old-backups src/core, showing 206 files and 2.99% total coverage.
tags: [source, old-backups, coverage, test-evidence, negative-evidence]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/outputs/coverage/current/
confidence: high
---

# Summary

`archive/old_backups_2025_08/outputs/coverage/current/` is a preserved HTML coverage report: 215 files, 21 MB, aggregate SHA-256 `7b1208489419a178e01eab56ea3d7fa37fca337f39642b7365d34be7b5a91106`. The machine-readable report index is `status.json`; the human top-level page is `index.html`. [1][2]

The report is strong negative evidence for broad `src/core` test coverage at that historical point. It covers 206 `src/core` files, 27,115 statements, 26,303 missing statements, 585 excluded statements, and total coverage of 2.99%. [1][2]

# Scope

The report is not a focused unified-tool coverage report. It is a broad `src/core` coverage report. Every indexed source path starts under `src/core`.

This distinction matters because other reports in the preserved lineage say specific unified tools had much higher coverage, such as 84-91% for T01/T02/T23A in the August evidence reports. Those narrower results do not contradict this broad 2.99% `src/core` report; they describe different coverage surfaces. See [Digimon Lineage Evidence Reports 2025 08](/wiki/sources/digimon-lineage-evidence-reports-2025-08.md).

# Coverage Totals

| Metric | Value |
| --- | ---: |
| Files indexed | 206 |
| Statements | 27,115 |
| Missing statements | 26,303 |
| Excluded statements | 585 |
| Total coverage | 2.99% |
| Coverage.py version | 7.8.2 |
| Status format | 5 |

# Representative Files

The report has many 0% files. The largest 0% files include:

- `src/core/neo4j_manager_backup.py`: 420 statements, 420 missing
- `src/core/phase_adapters_backup.py`: 381 statements, 381 missing
- `src/core/execution_monitor.py`: 369 statements, 369 missing
- `src/core/evidence_logger.py`: 364 statements, 364 missing
- `src/core/error_taxonomy.py`: 308 statements, 308 missing

The highest-coverage nontrivial files still show a limited tested surface:

- `src/core/anyio_api_client.py`: 244 statements, 33 missing, 86.48%
- `src/core/dependency_injection.py`: 139 statements, 19 missing, 86.33%
- `src/core/logging_config.py`: 76 statements, 18 missing, 76.32%
- `src/core/config_manager.py`: 284 statements, 102 missing, 64.08%
- `src/core/unified_service_interface.py`: 72 statements, 30 missing, 58.33%

# Interpretation

This page should be used as broad coverage-status evidence, not as a full test-quality judgment. It says that much of `src/core` was unexercised by the coverage run that produced this HTML report. It does not say that no subsystem had strong tests.

The key claim-discipline rule is:

- Broad `src/core` coverage in this report: 2.99%.
- Narrow unified-tool coverage in other reports: higher, but scoped to selected tools.
- Current runtime coverage: not established by this historical report.

This belongs with [Test Evidence Layer](/wiki/concepts/test-evidence-layer.md), [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md), and [Current Status Verification Discipline](/wiki/concepts/current-status-verification-discipline.md).

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/outputs/coverage/current/status.json`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/outputs/coverage/current/index.html`
