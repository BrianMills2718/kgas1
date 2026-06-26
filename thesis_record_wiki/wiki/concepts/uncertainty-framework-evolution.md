---
type: Concept
title: Uncertainty Framework Evolution
description: KGAS uncertainty design evolved from normalized confidence scores through CERQual and quality degradation toward local LLM expert assessments with auditable reasoning.
tags: [concept, uncertainty, quality, entity-resolution, confidence]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-004-Normative-Confidence-Score-Ontology.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-007-uncertainty-metrics.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-010-Quality-System-Design.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-025-Entity-Resolution-Architecture.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/UNCERTAINTY_20250825.md
confidence: medium
---

# Summary

KGAS uncertainty architecture changed because early confidence approaches were either too shallow for academic research or too complex to sustain. The later direction is not "perfect uncertainty measurement"; it is auditable expert reasoning about construct mappings.

The archived uncertainty stress test adds an intermediate implementation arc: CERQual/Bayesian/LLM services, IC-inspired analytical tests, bias analysis, and Davis multi-resolution/multi-perspective validation. It shows both the ambition and the reason later summaries must preserve bias/readiness caveats. See [Digimon Lineage Uncertainty Stress Test Root](/wiki/sources/digimon-lineage-uncertainty-stress-test-root.md).

The stress-test analysis directory explains where the Davis turn came from: six-paper rapid analysis, targeted agent extraction notes, and a synthesis explicitly mapping Davis's multi-method uncertainty ideas onto KGAS. See [Digimon Lineage Uncertainty Stress Test Analysis](/wiki/sources/digimon-lineage-uncertainty-stress-test-analysis.md).

The stress-test core-services directory shows the corresponding implementation experiment: Bayesian aggregation, CERQual assessment, formal Bayesian updating with LLM-estimated parameters, LLM-native contextual confidence, and a cached/parallel optimization attempt. It is implementation evidence, not current runtime proof. See [Digimon Lineage Uncertainty Stress Test Core Services](/wiki/sources/digimon-lineage-uncertainty-stress-test-core-services.md).

The validation directory then shows which claims had preserved output artifacts: basic connectivity, formal Bayesian medical and extraordinary-claim examples, a seven-case LLM-native versus rule-based comparison, and SocialMaze mock-mode analysis. Missing ground-truth and bias-analysis outputs remain an important caveat. See [Digimon Lineage Uncertainty Stress Test Validation](/wiki/sources/digimon-lineage-uncertainty-stress-test-validation.md).

The testing directory shows the IC-inspired harness behind the root 5/5 stress-test summary: information value, stopping rules, ACH, calibration, and mental-model auditing. Its results are meaningful as constructed-harness evidence, not broad external validation. See [Digimon Lineage Uncertainty Stress Test Testing](/wiki/sources/digimon-lineage-uncertainty-stress-test-testing.md).

The Bayesian directory preserves an earlier psychological-trait inference prototype: simulated LLM evidence extraction and likelihood ratios feeding deterministic Bayesian updates. It is useful for understanding the method idea, but not proof of live LLM integration. See [Digimon Lineage Uncertainty Stress Test Bayesian](/wiki/sources/digimon-lineage-uncertainty-stress-test-bayesian.md).

The docs directory preserves the mathematical/design rationale behind the stress test. It is useful for formulas and intended architecture, but the unchecked implementation checklist and uncorroborated validation-study claims mean it should not be used alone as completion evidence. See [Digimon Lineage Uncertainty Stress Test Docs](/wiki/sources/digimon-lineage-uncertainty-stress-test-docs.md).

# Evolution

1. **Contract normalization**: ADR-004 tried to make all tools report confidence through one Pydantic model. This addressed interface compatibility but not deeper research uncertainty. [1]
2. **CERQual research framework**: ADR-007 replaced simple confidence with a qualitative social-science uncertainty framework, but was later superseded. [2]
3. **Quality degradation**: ADR-010 records a confidence-degradation service and then marks it superseded because simple multiplication was mathematically weak and overly pessimistic. [3]
4. **Entity-resolution application**: ADR-025 applies uncertainty design to pronouns, group references, ambiguity, and research-suitability thresholds. [4]
5. **Local construct-mapping assessment**: the later 2025-08 note says each tool should assess how well its output represents its target construct, with explicit reasoning and no hardcoded uncertainty rules. [5]

# Key Lesson

The durable thesis insight is the movement from numeric confidence as a field to uncertainty as an auditable reasoning object:

- every uncertainty value needs reasoning
- deterministic tools can still have construct-validity uncertainty
- aggregation can reduce uncertainty when evidence converges
- high uncertainty should continue with visibility unless the tool actually fails
- thresholds should serve research decisions, not pretend to be objective truth

# Why It Matters

This concept connects the KGAS implementation arc to the thesis question. LLM-generated ontologies and discourse analyses are useful only if uncertainty and entity ambiguity remain inspectable. Entity resolution is especially central because discourse analysis often depends on who ambiguous actors refer to and how confident the system is about that mapping.

# Verification Caveat

Several files cite an ADR-029 IC-informed uncertainty framework, but that directory was not found in the preserved `docs/architecture` tree during this ingest. Until found elsewhere, the wiki should treat the local-assessment note and active uncertainty-propagation architecture as the available later-state evidence, not as a complete ADR-029 substitute.

# Links

- [Digimon Lineage Uncertainty Quality ADRs](/wiki/sources/digimon-lineage-uncertainty-quality-adrs.md)
- [Digimon Lineage Uncertainty Stress Test Root](/wiki/sources/digimon-lineage-uncertainty-stress-test-root.md)
- [Digimon Lineage Uncertainty Stress Test Analysis](/wiki/sources/digimon-lineage-uncertainty-stress-test-analysis.md)
- [Digimon Lineage Uncertainty Stress Test Core Services](/wiki/sources/digimon-lineage-uncertainty-stress-test-core-services.md)
- [Digimon Lineage Uncertainty Stress Test Validation](/wiki/sources/digimon-lineage-uncertainty-stress-test-validation.md)
- [Digimon Lineage Uncertainty Stress Test Testing](/wiki/sources/digimon-lineage-uncertainty-stress-test-testing.md)
- [Digimon Lineage Uncertainty Stress Test Bayesian](/wiki/sources/digimon-lineage-uncertainty-stress-test-bayesian.md)
- [Digimon Lineage Uncertainty Stress Test Docs](/wiki/sources/digimon-lineage-uncertainty-stress-test-docs.md)
- [Uncertainty Framework Consolidation 2026 06 26](/wiki/concepts/uncertainty-framework-consolidation-2026-06-26.md)
- [Uncertainty Traceability Architecture](/wiki/concepts/uncertainty-traceability-architecture.md)
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md)
- [Complexity Accuracy Pattern](/wiki/concepts/complexity-accuracy-pattern.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-004-Normative-Confidence-Score-Ontology.md`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-007-uncertainty-metrics.md`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-010-Quality-System-Design.md`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-025-Entity-Resolution-Architecture.md`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/UNCERTAINTY_20250825.md`
