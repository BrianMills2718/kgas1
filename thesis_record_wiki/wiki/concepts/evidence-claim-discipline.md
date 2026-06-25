---
type: Concept
title: Evidence Claim Discipline
description: KGAS/Digimons practice of matching claims to the level of validation actually performed.
tags: [concept, evidence, verification, claims]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/evidence/archived/false_claims_2025_08_03/README.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/evidence/current/Evidence.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/evidence_reports_2025_08/COMPLETE_EVIDENCE_TRACE.md
confidence: high
---

# Summary

Evidence claim discipline is the rule that a claim must match the level of validation actually performed. The large Digimons lineage contains an unusually explicit correction: component-level success was not enough to claim system integration success. [1]

The lit-review Phase 2 evidence adds a second example: a remediation summary can claim 100% success while a stored test result still records a failing integration-quality test. See [Lit Review Phase 2-3 Evidence](/wiki/sources/lit-review-phase2-3-evidence.md).

Phase 4 adds the same lesson at integration scale: stored tests can pass while certification reports remain weak, and later remediation may supersede earlier benchmark claims. See [Lit Review Phase 4 Integration Pipeline](/wiki/sources/lit-review-phase4-integration-pipeline.md).

Phase 6 extends the issue into deployment language: internal production-validation simulations can support "production validation package exists" without proving an externally deployed production service. See [Lit Review Phase 6 Production Validation](/wiki/sources/lit-review-phase6-production-validation.md).

The uncertainty ADRs add another claim-discipline rule: do not cite "the KGAS uncertainty framework" as one stable object without naming the document/date, because several frameworks were explicitly superseded and a referenced ADR-029 path is missing from the preserved ADR tree. See [Digimon Lineage Uncertainty Quality ADRs](/wiki/sources/digimon-lineage-uncertainty-quality-adrs.md).

The analysis-expansion ADRs add a scale warning: broad accepted architecture and validation claims for cross-modal, schema, ABM, and statistical capabilities should not be treated as current working-system status without separate implementation evidence. See [Digimon Lineage Analysis Expansion ADRs](/wiki/sources/digimon-lineage-analysis-expansion-adrs.md).

The lit-review multi-agent harness is a stronger evidence pattern than self-reporting because it separates implementation from evaluation, but its claims remain historical artifacts until rerun or otherwise reproduced. See [Lit Review Multi Agent System](/wiki/sources/lit-review-multi-agent-system.md).

The current-code verification slice adds the final distinction: even current docs can be stale, so status summaries must separate architecture intent, evidence reports, files present on disk, and runtime execution. See [Current Code Verification 2026-06-25](/wiki/sources/current-code-verification-2026-06-25.md).

The old-backups stress-test slice gives a compact example inside one directory: an earlier machine-readable run reports 0.4 / `DEVELOPMENT_READY`, a later run reports 0.8 / `PRODUCTION_READY`, the final report still flags cross-modal semantic preservation at 40%, and a separate patch report claims 100% preservation on a smaller patched demonstration. See [Digimon Lineage Old Backups Stress Test Root](/wiki/sources/digimon-lineage-old-backups-stress-test-root.md).

The uncertainty core-services slice adds a code-level version of the same rule: six service files exist and contain real API-call implementations, but fallback/default paths and an optimized-engine helper/import mismatch mean the right claim is "implementation artifact exists," not "runtime verified here." See [Digimon Lineage Uncertainty Stress Test Core Services](/wiki/sources/digimon-lineage-uncertainty-stress-test-core-services.md).

The uncertainty validation slice adds an output-presence rule: LLM-native 7/7 and SocialMaze excellent-calibration claims have preserved result files, but SocialMaze is mock mode and the detailed ground-truth/bias-analysis output files are absent from that directory. See [Digimon Lineage Uncertainty Stress Test Validation](/wiki/sources/digimon-lineage-uncertainty-stress-test-validation.md).

The uncertainty testing slice adds another validation-level boundary: a root 5/5 success summary exists for the IC-inspired harness, but the tests are constructed demonstrations and the root summary preserves only truncated stdout excerpts. See [Digimon Lineage Uncertainty Stress Test Testing](/wiki/sources/digimon-lineage-uncertainty-stress-test-testing.md).

The uncertainty Bayesian slice is a naming caution: files named "real" and "production" still use simulated LLM analyses in the preserved code, so status claims must be based on behavior and outputs rather than filenames. See [Digimon Lineage Uncertainty Stress Test Bayesian](/wiki/sources/digimon-lineage-uncertainty-stress-test-bayesian.md).

The uncertainty docs slice is a documentation-evidence caution: a methodology document can report validation-study numbers while a specification document marks every implementation checklist item unchecked. Use those as design/status claims to corroborate, not as standalone proof. See [Digimon Lineage Uncertainty Stress Test Docs](/wiki/sources/digimon-lineage-uncertainty-stress-test-docs.md).

The uncertainty setup slice is a demo-automation caution: Docker/Neo4j setup scripts and complete-demo code exist, but no demo-output files were found in that setup directory, and one import path appears fragile. See [Digimon Lineage Uncertainty Stress Test Setup](/wiki/sources/digimon-lineage-uncertainty-stress-test-setup.md).

# Core Lesson

The archived false-claims README states the key rule plainly: component success does not equal system integration success. It says individual tools and interfaces had been tested, but full auto-registration, agent-tool integration, and real workflow execution had not been validated. [1]

# Why It Matters

This thread explains why the thesis record contains both impressive evidence files and later negative/corrective investigations. Those are not simply contradictions; they often operate at different validation levels:

- component evidence
- interface compliance evidence
- workflow demonstration evidence
- system integration evidence
- locked benchmark evidence

Future summaries must name the validation level.

# Relationship To Other Pages

- [Reality Verification Arc](/wiki/concepts/reality-verification-arc.md): later status estimates corrected earlier under- and over-claims.
- [Documentation Status Truthfulness](/wiki/concepts/documentation-status-truthfulness.md): roadmap and architecture claims need separate truth surfaces.
- [Uncertainty Traceability Architecture](/wiki/concepts/uncertainty-traceability-architecture.md): uncertainty/provenance claims require audit trails, not only design intent.
- [Adaptive Operator Routing](/wiki/concepts/adaptive-operator-routing.md): development benchmark evidence should not be treated as locked proof.
- [Balance Driven Validation](/wiki/concepts/balance-driven-validation.md): balance metrics are useful evidence but do not by themselves prove theoretical fidelity.
- [Uncertainty Framework Evolution](/wiki/concepts/uncertainty-framework-evolution.md): uncertainty claims must be tied to the active or superseded framework being referenced.
- [Analysis Expansion Architecture](/wiki/concepts/analysis-expansion-architecture.md): broad capability surfaces require per-capability verification.
- [Multi Agent Evidence Harness](/wiki/concepts/multi-agent-evidence-harness.md): implementation/evaluation separation as a procedural guardrail.
- [Current Status Verification Discipline](/wiki/concepts/current-status-verification-discipline.md): current functionality requires direct checkout and runtime checks.
- [Digimon Lineage Old Backups Stress Test Framework](/wiki/sources/digimon-lineage-old-backups-stress-test-framework.md): mock-mode fallbacks and registry/schema disagreement as claim-level caveats.
- [Digimon Lineage Uncertainty Stress Test Core Services](/wiki/sources/digimon-lineage-uncertainty-stress-test-core-services.md): source-code implementation evidence constrained by runtime caveats.
- [Digimon Lineage Uncertainty Stress Test Validation](/wiki/sources/digimon-lineage-uncertainty-stress-test-validation.md): preserved validation outputs versus validation scripts and missing-output boundaries.
- [Digimon Lineage Uncertainty Stress Test Testing](/wiki/sources/digimon-lineage-uncertainty-stress-test-testing.md): constructed harness success versus generalization proof.
- [Digimon Lineage Uncertainty Stress Test Bayesian](/wiki/sources/digimon-lineage-uncertainty-stress-test-bayesian.md): filename/status-label caveats for simulated LLM Bayesian prototypes.
- [Digimon Lineage Uncertainty Stress Test Docs](/wiki/sources/digimon-lineage-uncertainty-stress-test-docs.md): design documents, unchecked checklist items, and uncorroborated validation-study claims.
- [Digimon Lineage Uncertainty Stress Test Setup](/wiki/sources/digimon-lineage-uncertainty-stress-test-setup.md): setup/demo automation without preserved successful demo outputs.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/evidence/archived/false_claims_2025_08_03/README.md`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/evidence/current/Evidence.md`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/evidence_reports_2025_08/COMPLETE_EVIDENCE_TRACE.md`
