---
type: Concept
title: Thesis Record Overview
description: Top-level map of the KGAS / Digimons thesis work record and how to preserve it.
tags: [thesis-record, kgas, preservation]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../RECOVERY_ARCHIVE_MANIFEST_2026-04-04.md
  - ../archive_full_record/metadata/recovery_inventory.tsv
  - ../README.md
  - ../CLAUDE.md
confidence: medium
---

# Summary

This wiki is a derived navigation layer for Brian's KGAS / Digimons / PhD thesis work. The central preservation fact is that the tracked repo is only the cleaned working checkout; the full local history was intentionally preserved under `../archive_full_record/` on 2026-04-04. The recovery manifest says that the archive layer exists to avoid losing material that may have been intentionally or unintentionally removed from the tracked repo during cleanup. [1]

The current tracked repo presents KGAS as an academic research GraphRAG system connected to the dissertation topic "Theoretical Foundations for LLM-Generated Ontologies and Analysis of Fringe Discourse." [2] The current `CLAUDE.md` also preserves older operational context about tool compatibility, vertical slices, uncertainty propagation, provenance, reasoning traces, and documentation cleanup. [3]

For ongoing ingest state and next slices, see [Progress](/PROGRESS.md).

# Preservation Model

The archive is not clutter to delete. It is the full record of project evolution. The wiki should make that record navigable while keeping raw sources unchanged.

Initial major source buckets:

- [Current Clean Repo](/wiki/variants/current-clean-repo.md)
- [Filesystem Snapshot 2026-04-04](/wiki/variants/filesystem-snapshot-2026-04-04.md)
- [Digimon Lineage Digimons](/wiki/variants/digimon-lineage-digimons.md)
- [Digimons Old](/wiki/variants/digimons-old.md)
- [Digimon v2](/wiki/variants/digimon-v2.md)
- Smaller lineage variants under `../archive_full_record/lineage_variants/`

Initial source summaries:

- [Recovery Archive Manifest 2026-04-04](/wiki/sources/recovery-archive-manifest-2026-04-04.md)
- [Recovery Inventory](/wiki/sources/recovery-inventory.md)
- [Current Repo Context](/wiki/sources/current-repo-context.md)

# What This Wiki Is For

- Reconstruct how the thesis system evolved over time.
- Separate raw evidence from later cleanup and interpretation.
- Preserve abandoned or superseded ideas without polluting active repo search.
- Make future archive decisions reversible and explainable.
- Help Brian recover the intellectual arc of the work after leaving the thesis program.

The first pass is intentionally conservative: it identifies buckets and risks rather than attempting to flatten the record into one narrative. The next pass should ingest variant README/CLAUDE/docs files and then revise the lineage pages with more confident descriptions.

After the first small-variant ingest, two organizing themes are visible:

- [Documentation Status Truthfulness](/wiki/concepts/documentation-status-truthfulness.md) - cleanup work repeatedly tried to separate target architecture from verified implementation status.
- [Type-Based Tool Composition](/wiki/concepts/type-based-tool-composition.md) - later engineering work focused on making KGAS tools composable through exact typed contracts.
- [GraphRAG Upstream Lineage](/wiki/concepts/graphrag-upstream-lineage.md) - local Digimon/KGAS should be read as connected to JayLZhou GraphRAG / DIGIMON, not as an isolated origin.
- [Contract-First Migration](/wiki/concepts/contract-first-migration.md) - `digimon_core_sparse` records a formal attempt to collapse split tool interfaces into KGASTool contracts.
- [Relationship Extraction Bottleneck](/wiki/concepts/relationship-extraction-bottleneck.md) - sparse-variant stress tests show a critical failure mode where entities were extracted but relationships were not.
- [Adaptive Operator Routing](/wiki/concepts/adaptive-operator-routing.md) - `digimon_autoloop` sharpens the thesis into a falsifiable benchmark question and records negative development evidence.
- [MCP Autoloop Interface](/wiki/concepts/mcp-autoloop-interface.md) - autoloop preserves early MCP/UKRF integration plans and later MCP/direct tool-surface status.
- [Graph Build Manifest](/wiki/concepts/graph-build-manifest.md) - later DIGIMON docs require tool exposure to be driven by persisted graph build facts rather than assumed capability.
- [Reality Verification Arc](/wiki/concepts/reality-verification-arc.md) - `digimon_lineage_Digimons` records a September 2025 correction sequence around inflated implementation claims and roadmap consolidation.
- [Vertical Slice vs Main System](/wiki/concepts/vertical-slice-vs-main-system.md) - the large lineage bundle preserves a simple verified vertical slice alongside a more complex main-system architecture.
- [Digimon Lineage Tool Compatibility](/wiki/sources/digimon-lineage-tool-compatibility.md) - the large bundle connects type-based composition to the active vertical-slice framework path.
- [Digimon Lineage Architecture Docs](/wiki/sources/digimon-lineage-architecture-docs.md) - architecture target-design docs, ADR cascades, limitations, and critical uncertainty review.
- [Uncertainty Traceability Architecture](/wiki/concepts/uncertainty-traceability-architecture.md) - the internal critique of uncertainty modeling, provenance gaps, and research decision support.
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md) - evidence archive correction showing that component success was incorrectly overstated as system integration success.
- [Lit Review Theory Extraction Experiment](/wiki/sources/lit-review-theory-extraction-experiment.md) - separate experiment subsystem for extracting theory schemas from academic papers and applying them to data.
- [Automated Theory Extraction](/wiki/concepts/automated-theory-extraction.md) - a thesis-core method thread connecting schema extraction, model-type selection, and evidence discipline.
- [Carter Theory Analysis Output](/wiki/sources/carter-theory-analysis-output.md) - generated output slice applying two academic theories to one Carter speech.
- [Multi-Theory Application Artifact](/wiki/concepts/multi-theory-application-artifact.md) - evidence form for checking whether theory extraction/application produced concrete inspectable outputs.
- [Lit Review Schema Creation Production Path](/wiki/sources/lit-review-schema-creation-production-path.md) - code and prompt slice documenting how schema extraction was produced or evolved.
- [Schema Extraction Pipeline Evolution](/wiki/concepts/schema-extraction-pipeline-evolution.md) - information-loss correction, no-truncation extraction, and adaptive model-type selection thread.
- [Lit Review Validation Results](/wiki/sources/lit-review-validation-results.md) - reports and outputs for Young 1996, framing effects, and Lofland-Stark validation cases.
- [Complexity Accuracy Pattern](/wiki/concepts/complexity-accuracy-pattern.md) - conservative validation lesson that simple theories are the best automation target.
- [Lit Review Phase 2-3 Evidence](/wiki/sources/lit-review-phase2-3-evidence.md) - balanced vocabulary/schema evidence with a Phase 2 summary-versus-test contradiction.
- [Balance Driven Validation](/wiki/concepts/balance-driven-validation.md) - validation metric pattern useful for bias detection but insufficient alone for theory fidelity.
- [Lit Review Phase 4 Integration Pipeline](/wiki/sources/lit-review-phase4-integration-pipeline.md) - integration evidence whose test, certification, and remediation claims need time-indexed reading.

# Current Cautions

- Two permission-denied paths are recorded in the recovery errors metadata; treat those as verification gaps, not as proof of absence. [4]
- Some lineage variants have `destination_git_head` recorded as `ERROR`; those need later review rather than deletion. [5]
- The active branch is `backup/2026-05-23/phd_thesis_work-master`, not `master`, and includes post-backup commits. [6]
- See [Verification Gaps](/wiki/concepts/verification-gaps.md) before interpreting missing data or failed git-head reads.

# Citations

[1] `../RECOVERY_ARCHIVE_MANIFEST_2026-04-04.md`  
[2] `../README.md`  
[3] `../CLAUDE.md`  
[4] `../archive_full_record/metadata/recovery_inventory_errors.tsv`  
[5] `../archive_full_record/metadata/recovery_inventory.tsv`  
[6] Git history on `backup/2026-05-23/phd_thesis_work-master`, HEAD `2dfab76fe4181a1734001b666b634449d56c69fb`
