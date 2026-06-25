---
type: Source
title: Digimon Lineage Old Backups Error Reports
description: Twenty-two generated error/escalation JSON reports from old_backups_2025_08, dominated by validation load-test errors and zero successful recoveries.
tags: [source, old-backups, errors, escalation, recovery, negative-evidence]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/error_reports/
confidence: high
---

# Summary

`archive/old_backups_2025_08/error_reports/` contains 22 generated JSON error/escalation reports, 328 KB total, aggregate SHA-256 `56ec35e529cb98ea662f6d1e1f27d298b8a9595a2def97b40293509c9db45f75`.

This slice is evidence that KGAS/Digimons had an error-reporting and escalation-report format, with recovery strategies registered. It is also strong negative evidence for recovery effectiveness in these preserved runs: across the reports, recovery attempts total 55 and successful recoveries total 0. [1][2]

# Report Groups

| Report group | Count |
| --- | ---: |
| `ESCALATION_VALIDATION_FAILED` | 5 |
| `ESCALATION_DATABASE_OPERATION_FAILED` | 2 |
| `ESCALATION_NEO4J_CONNECTION_LOST` | 2 |
| `ESCALATION_PDF_FILE_NOT_FOUND` | 2 |
| `ESCALATION_PDF_PROCESSING_FAILED` | 2 |
| `ESCALATION_SECURITY_VIOLATION` | 2 |
| `ESCALATION_SPACY_MODEL_NOT_LOADED` | 2 |
| `ESCALATION_SYSTEM_FAILURE` | 2 |
| `ESCALATION_THRESHOLD_TEST` | 2 |
| `CRITICAL_HANDLER_FAILURE` | 1 |

# Aggregate Error Pattern

Aggregating `statistics.most_common_errors` across the reports gives this pattern:

| Error code | Count |
| --- | ---: |
| `VALIDATION_FAILED` | 3,730 |
| `DATABASE_OPERATION_FAILED` | 70 |
| `PDF_PROCESSING_FAILED` | 55 |
| `THRESHOLD_TEST` | 12 |
| `PDF_FILE_NOT_FOUND` | 6 |
| `SYSTEM_FAILURE` | 6 |
| `SPACY_MODEL_NOT_LOADED` | 4 |
| `SECURITY_VIOLATION` | 4 |
| `NEO4J_CONNECTION_LOST` | 2 |

By category, validation dominates with 3,730 events, followed by processing with 83, storage with 72, and security with 4. By severity, the reports aggregate 3,730 warnings, 149 errors, and 10 critical events. [1][2]

# Recovery Evidence

The reports preserve six registered recovery strategies in the critical handler report: retry for database connection and network timeout, fallback for PDF processing, reset for resource exhaustion, and terminate for configuration error and security violation. [1]

However, the escalation reports show zero successful recoveries. The aggregated recovery statistics across all 22 reports are:

- total recovery attempts: 55
- total recovery successes: 0
- observed recovery success rate in these reports: 0.0%

# Health Scores

System health scores range from 50.0 to 100.0, with an average of about 74.18 across the 22 reports. High-health reports often correspond to small or no-error cases; validation load-test reports and database/PDF processing escalation reports frequently show health score 50.0. [2]

# Interpretation

This is useful operational evidence, but it has to be read carefully:

- It proves error/escalation reporting artifacts existed.
- It shows the system could aggregate error statistics and recovery strategies.
- It does not prove recovery worked in these runs.
- The validation load-test reports are synthetic-looking stress evidence, not direct evidence of user-facing failures.

This page connects to [Test Evidence Layer](/wiki/concepts/test-evidence-layer.md), [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md), and [Current Status Verification Discipline](/wiki/concepts/current-status-verification-discipline.md).

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/error_reports/CRITICAL_HANDLER_FAILURE_20250804_030641.json`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/old_backups_2025_08/error_reports/`
