---
type: SourceSummary
title: ADR 029 Location Verification 2026 06 26
description: Targeted verification showing that ADR-029 was missing from one primary ADR tree but recovered as a byte-identical five-file bundle in digimon_core_sparse and the architecture-cleanup archive.
tags: [source, verification, uncertainty, adr, provenance]
created: 2026-06-26
updated: 2026-06-26
sources:
  - ../archive_full_record/lineage_variants/digimon_core_sparse/docs/architecture/adrs/ADR-029-IC-Informed-Uncertainty-Framework/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/docs_architecture_cleanup_2025_08_29/ADR-029-IC-Informed-Uncertainty-Framework/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/docs_architecture_cleanup_2025_08_29/ARCHIVE_MANIFEST.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/docs_architecture_cleanup_2025_08_29/ARCHIVE_LOG.md
confidence: high
---

# Summary

ADR-029 is not unrecovered. A targeted search on 2026-06-26 found the `ADR-029-IC-Informed-Uncertainty-Framework/` bundle in two preserved locations:

- `../archive_full_record/lineage_variants/digimon_core_sparse/docs/architecture/adrs/ADR-029-IC-Informed-Uncertainty-Framework/` [1]
- `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/docs_architecture_cleanup_2025_08_29/ADR-029-IC-Informed-Uncertainty-Framework/` [2]

The older wiki caveat remains useful but needs scope: ADR-029 was missing from the inspected `digimon_lineage_Digimons/docs/architecture/adrs` tree, not from the entire recovered thesis archive. [3]

# File Inventory

Both recovered bundles contain the same five files:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `ADR-029-IC-Informed-Uncertainty-Framework.md` | 6,390 | `13f1ae106c669b70b8dcb3ffc4d64d2b01b32e09180edebf76f302f14b9528f1` |
| `IC_UNCERTAINITY_NOTES_2025.0728.md` | 100,412 | `f1487f3eaf5867670d7d2657515b2ef9e6f0ceeee833f27fcbcbaa33cc4e9d22` |
| `entity_resolution_under_uncertainty.md` | 27,255 | `1dffa33923449a1470a3251f3435acbf87d69865b8b31dca9109e8ea68759f3c` |
| `kgas_uncertainty_framework_comprehensive7.md` | 23,278 | `232b06e2211820fa8186d3f9bbe9373b9e912d4424171bba4ec58ac50718c4b8` |
| `uncertainty_considerations_to_resolve.md` | 21,305 | `6d0af831a9bc6f73bc9c678fa9cbe520848b9752a72485539413677bbe223d11` |

The matching hashes show the `digimon_core_sparse` and architecture-cleanup copies are byte-identical for this bundle. [1][2]

# What ADR-029 Says

ADR-029 is dated 2025-07-29 and marked `Accepted`. It establishes an IC-informed uncertainty framework based on the Comprehensive7 specification. Its stated elements are ICD-203 probability bands, ICD-206 source quality assessment, Heuer-inspired information-paradox awareness, mathematical propagation with root-sum-squares, and a single integrated LLM uncertainty analysis. [4]

The Comprehensive7 specification expands that into an architecture for source quality, probability/confidence language, structured analytic techniques, entity resolution, mathematical propagation, and cross-modal uncertainty reduction. [5]

# Supersession And Abandonment Boundary

The architecture-cleanup manifest is the decisive status boundary. It says the ADR-029 directory and IC uncertainty materials were archived because Brian confirmed the IC uncertainty framework should be archived. The cleanup rationale says architecture docs should describe target state and that abandoned architectural approaches belong in history. [6][7]

Therefore the corrected status is:

- **Recovered:** yes, as a five-file bundle.
- **Accepted at the time:** yes, ADR-029 itself is marked accepted.
- **Canonical current architecture:** no, the cleanup archive says the IC uncertainty approach was abandoned/archived.
- **Current runtime proof:** not established by this verification.

# Current Repo Reference Caveat

Current architecture files still contain references to ADR-029, including provenance and architecture-overview pages. Those references are historical or stale unless reconciled with the cleanup archive. [8][9]

Future summaries should say: ADR-029/Comprehensive7 was an accepted IC-informed uncertainty architecture that was later archived after the approach was abandoned. Do not repeat the older broad phrase "ADR-029 missing" without the narrower scope.

# Relationship To Wiki

- [Digimon Lineage Docs Architecture Cleanup 2025 08 29](digimon-lineage-docs-architecture-cleanup-2025-08-29.md): prior source page that already noted the archived ADR-029 bundle and abandonment.
- [Digimon Lineage Uncertainty Quality ADRs](digimon-lineage-uncertainty-quality-adrs.md): earlier ADR slice whose missing-ADR-029 caveat is now scope-corrected by this page.
- [Uncertainty Framework Evolution](/wiki/concepts/uncertainty-framework-evolution.md): uncertainty design-history concept that should now read ADR-029 as recovered but later archived.
- [Uncertainty Framework Consolidation 2026 06 26](/wiki/concepts/uncertainty-framework-consolidation-2026-06-26.md): dated map updated by this verification.
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md): this page is an example of narrowing an overbroad caveat when better provenance is found.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_core_sparse/docs/architecture/adrs/ADR-029-IC-Informed-Uncertainty-Framework/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/docs_architecture_cleanup_2025_08_29/ADR-029-IC-Informed-Uncertainty-Framework/`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/docs_architecture_cleanup_2025_08_29/ADR-029-IC-Informed-Uncertainty-Framework/ADR-029-IC-Informed-Uncertainty-Framework.md`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/docs_architecture_cleanup_2025_08_29/ADR-029-IC-Informed-Uncertainty-Framework/kgas_uncertainty_framework_comprehensive7.md`  
[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/docs_architecture_cleanup_2025_08_29/ARCHIVE_MANIFEST.md`  
[7] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/docs_architecture_cleanup_2025_08_29/ARCHIVE_LOG.md`  
[8] `../docs/architecture/specifications/PROVENANCE.md`  
[9] `../docs/architecture/ARCHITECTURE_OVERVIEW.md`
