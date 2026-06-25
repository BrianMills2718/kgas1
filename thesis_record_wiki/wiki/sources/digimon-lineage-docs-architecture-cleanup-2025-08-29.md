---
type: SourceSummary
title: Digimon Lineage Docs Architecture Cleanup 2025 08 29
description: Architecture documentation cleanup archive covering generated duplicate docs, over-engineered service implementation guides, vertical-slice plans, and archived IC uncertainty framework materials.
tags: [source, digimon-lineage, archive, architecture, cleanup, uncertainty, ic-methods, generated-docs]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/docs_architecture_cleanup_2025_08_29/
confidence: high
---

# Summary

`archive/docs_architecture_cleanup_2025_08_29/` is a 62-file architecture cleanup archive totaling about 2.3 MB. Its aggregate content-manifest hash is `b40107136086ac24bace0779074f608d83f09fa417b3783866f2831cae749d9f`. [1]

The archive was created to remove confusing architecture-documentation clutter after extracting useful patterns. Its manifest says architecture documentation should describe the target state, not implementation roadmaps, over-engineered rejected approaches, or abandoned architectural directions. [2]

# Inventory

| Area | Files | Aggregate Hash | Role |
| --- | ---: | --- | --- |
| Root files | 13 | `0bb08ec8de18b6b510c497c48757601a21a3e252f59496cf5e8c8dccff0451d1` | Archive manifest/log, ADR-017, service implementation guides, vertical-slice integration plans, and correction/final variants. [1] |
| `ADR-029-IC-Informed-Uncertainty-Framework/` | 5 | `2a73fc1e1d32b009e383b4d28feaf34d112c7c8d69035e30f6719b3669847226` | Accepted ADR-029 and extensive IC uncertainty notes later archived after the approach was abandoned. [1] |
| `generated_documents/` | 8 | `93c25a22f5cbc92a3f9d7ab2a1bb639cc1ac637f11bef83bdb0d9abd1a92cba7` | Auto-generated architecture compilations and ADR indexes archived as duplicate/conflicting content. [1] |
| `ic_uncertainty_proposal_condensed/` | 4 | `9d033202a458b356e56acf4af9aa372b2f659277bc8fff4d1015cf0a0f8c6cc5` | Condensed IC uncertainty analysis focused on schema, cross-modal lossiness, and tool-level CERQual. [1] |
| `ic_uncertainty_proposal_rewrite/` | 32 | `2dc2b1837225d8821ff70219ff11828f72634455b0ed7c5982309e1b52e1ddaf` | Large IC uncertainty planning/rewrite corpus, including roadmap, stress test, performance/user-alignment assessments, and uncertainty-evolution notes. [1] |

# Archive Rationale

The manifest says vertical-slice integration plans were archived because they were implementation roadmaps, not target architecture. It says their patterns were extracted to canonical vertical-slice and simple service-implementation docs. [2]

The service-tool implementation guides were archived because the "bulletproof" variants were rejected as over-engineered in favor of a KISS service implementation guide. [2]

Generated architecture documents were archived because they were programmatically generated duplicates of manually maintained documents and could mislead readers about current architecture. [3]

# IC Uncertainty Supersession

ADR-017 accepted integration of Intelligence Community analytical techniques into KGAS, including information value assessment, stopping rules, ACH, calibration, and future mental-model auditing. It framed LLMs as flexible analysts that could adapt IC techniques to academic contexts. [4]

ADR-029 later accepted an IC-informed uncertainty framework using ICD-203 probability bands, ICD-206 source quality, Heuer principles, root-sum-squares propagation, and a single integrated LLM uncertainty analysis. It superseded earlier degradation-only and Bayesian aggregation approaches. [5]

The cleanup archive’s manifest and log are the crucial supersession layer: they say the IC uncertainty framework was archived after user confirmation that the approach should be abandoned. [2] [3]

# Critical Reversal Evidence

The IC uncertainty rewrite corpus contains a critical insights summary that says the current confidence system was fundamentally broken due to a category error. It identifies a missing synthesis bridge from tool outputs to research findings, misapplication of CERQual to individual computational tools, ICD-206 scale mismatch for aggregate datasets, lack of propagation thresholds, and missing evidence-synthesis protocols. [6]

That summary gives strategic options: full research synthesis, computational confidence only, human-AI hybrid synthesis, or integration with existing tools. It warns that the current system could invalidate academic outputs if claims were not corrected. [6]

This is one of the clearest examples in the archive of a design idea moving from accepted ADR to abandoned or scope-reduced approach after deeper methodological critique.

# Generated Architecture Docs

The generated-documents subtree includes comprehensive architecture compilations, a complete ADR compilation, and generated architecture indexes. The archive log says these were removed because they duplicated/conflicted with manually maintained architecture documentation. [3]

These files may still be useful as historical snapshots, but they should not be treated as canonical target architecture unless corroborated by active architecture docs or current verification pages.

# Credential Scan

A targeted scan of this architecture-cleanup archive found no literal OpenAI or Google API keys. [1]

# Interpretation

This archive is high-value because it preserves the cleanup decision itself: target architecture should be concise and current, while implementation roadmaps, over-engineered variants, generated compilations, and abandoned uncertainty frameworks belong in history.

For thesis reconstruction, the most important point is the IC uncertainty arc. It shows a serious attempt to import IC analytic rigor, followed by recognition that uncertainty assessment at the tool, source, dataset, evidence, and research-claim levels cannot be collapsed into one framework without category errors.

# Relationship To Wiki

- [Digimon Lineage Archive Coverage Audit 2026-06-25](digimon-lineage-archive-coverage-audit-2026-06-25.md): queue-control page that identified this top-level archive area.
- [Digimon Lineage Uncertainty Quality ADRs](digimon-lineage-uncertainty-quality-adrs.md): earlier ADR slice that noted uncertainty ADR evolution and missing/archived ADR-029 caveat.
- [Digimon Lineage Theoretical Exploration Schema v14 Post MVP](digimon-lineage-theoretical-exploration-schema-v14-post-mvp.md): related later schema/uncertainty separation.
- [Uncertainty Framework Evolution](../concepts/uncertainty-framework-evolution.md): related uncertainty-design history.
- [Evidence Claim Discipline](../concepts/evidence-claim-discipline.md): relevant guardrail for avoiding research-confidence overclaims.
- [Current Status Verification Discipline](../concepts/current-status-verification-discipline.md): related distinction between target architecture, generated docs, and verified status.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/docs_architecture_cleanup_2025_08_29/`

[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/docs_architecture_cleanup_2025_08_29/ARCHIVE_MANIFEST.md`

[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/docs_architecture_cleanup_2025_08_29/ARCHIVE_LOG.md`

[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/docs_architecture_cleanup_2025_08_29/ADR-017-IC-Analytical-Techniques-Integration.md`

[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/docs_architecture_cleanup_2025_08_29/ADR-029-IC-Informed-Uncertainty-Framework/ADR-029-IC-Informed-Uncertainty-Framework.md`

[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/docs_architecture_cleanup_2025_08_29/ic_uncertainty_proposal_rewrite/CRITICAL_INSIGHTS_SUMMARY.md`
