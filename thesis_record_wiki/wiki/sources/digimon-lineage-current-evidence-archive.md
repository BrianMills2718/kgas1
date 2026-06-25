---
type: Source
title: Digimon Lineage Current Evidence Archive
description: Inventory and interpretation of archive/evidence/current, including timestamped verification records, corrected performance claims, mixed integration results, and evidence-grade caveats.
tags: [source, digimon-lineage, evidence, runtime-evidence, performance, integration, verification, negative-evidence]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/evidence/current/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/evidence/current/Evidence.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/evidence/current/Evidence_Actual_Performance_Summary.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/evidence/current/Evidence_Real_Speedup_Measurement.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/evidence/current/Evidence_Full_Integration_Tests.md
confidence: medium
---

# Summary

`archive/evidence/current/` is a 46-file, 292,963-byte evidence corpus with aggregate hash `5dd24786622945f288144ca960ab485f837f319d689604bd09755d87a844548c`. [1]

This is a higher-grade evidence layer than test definitions because it contains timestamped verification records, performance measurements, integration-test results, and targeted evidence reports. It is still historical evidence, not current-environment proof.

# Inventory Shape

The directory includes:

- a large `Evidence.md` file with timestamped verification records, hashes, system information, successes, and failures
- performance reports and speedup measurements
- phase completion reports
- fallback removal and real-service implementation reports
- full integration and tool contract integration reports
- structured output, provenance, MCP adapter, and visualization evidence files [1]

# Mixed Runtime Evidence

`Evidence.md` is especially valuable because it preserves negative evidence alongside successes:

- successful entity batch integration records with document counts and entity counts
- repeated batch dashboard and cross-document visualization failures caused by a NetworkX-related error
- streaming checkpoint timeout after 30 seconds
- enhanced-engine and end-to-end errors caused by a missing `service_manager` argument
- dashboard error records with empty error payloads [2]

That makes this file stronger than a clean summary report: it preserves what failed.

# Corrected Performance Evidence

`Evidence_Actual_Performance_Summary.md` explicitly corrects overstated performance:

- LLM entity extraction exceeded target at 83.39% F1
- multi-document speedup was corrected from a claimed 10.8x to 1.24x
- maximum parallelism was corrected from 12 to 6 nodes
- real tool processing times were used to argue that tools were doing real work [3]

`Evidence_Real_Speedup_Measurement.md` gives a narrower sequential-vs-parallel measurement:

- sequential total: 0.288 seconds for 20 nodes
- parallel total: 0.247 seconds across 6 batches
- actual speedup: 1.17x
- time saved: 0.041 seconds [4]

These reports are important because they downgrade exaggerated claims while preserving a smaller real speedup.

# Full Integration Caveat

`Evidence_Full_Integration_Tests.md` reports a failed full integration result:

- status: failed
- success rate: 40%
- document loading failed
- PageRank passed
- multihop query was partial
- LLM fail-fast passed
- no-fallbacks failed [5]

The same report also contains optimistic "key achievements" language, so it must be cited carefully. The pass/fail table is the stronger claim boundary.

# Interpretation

This current evidence archive is a major upgrade over source-code inventory:

- it contains direct measured values
- it preserves failures and partials
- it corrects earlier inflated performance claims
- it shows that "real processing" and "full integration success" were different claims

Future synthesis should treat this directory as historical runtime evidence and compare it against [Current Runtime Import Check 2026-06-25](/wiki/sources/current-runtime-import-check-2026-06-25.md) before saying anything about current functionality.

# Links

- [Digimon Lineage Generated Reports](/wiki/sources/digimon-lineage-generated-reports.md)
- [Test Evidence Layer](/wiki/concepts/test-evidence-layer.md)
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md)
- [Current Status Verification Discipline](/wiki/concepts/current-status-verification-discipline.md)
- [Current Runtime Import Check 2026-06-25](/wiki/sources/current-runtime-import-check-2026-06-25.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/evidence/current/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/evidence/current/Evidence.md`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/evidence/current/Evidence_Actual_Performance_Summary.md`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/evidence/current/Evidence_Real_Speedup_Measurement.md`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/evidence/current/Evidence_Full_Integration_Tests.md`
