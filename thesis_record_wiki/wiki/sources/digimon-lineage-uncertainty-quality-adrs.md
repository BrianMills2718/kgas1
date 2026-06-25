---
type: Source
title: Digimon Lineage Uncertainty Quality ADRs
description: ADR and active-doc slice covering KGAS confidence ontology, CERQual uncertainty, quality system design, entity resolution, and later local-assessment uncertainty model.
tags: [source, architecture, adr, uncertainty, quality, entity-resolution]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-004-Normative-Confidence-Score-Ontology.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-007-uncertainty-metrics.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-010-Quality-System-Design.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-025-Entity-Resolution-Architecture.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/UNCERTAINTY_20250825.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/systems/uncertainty-propagation-architecture.md
confidence: medium
---

# Summary

This slice shows KGAS uncertainty design evolving through several superseded frameworks before landing on a pragmatic local-assessment model:

1. ADR-004 proposes a canonical `ConfidenceScore` contract to eliminate incompatible confidence semantics across tools, but is marked superseded by ADR-007. [1]
2. ADR-007 proposes a CERQual-based uncertainty framework, but is itself marked superseded by an IC-informed ADR-029 framework. [2]
3. ADR-010 documents a confidence-degradation quality system and says it was superseded by a Comprehensive7 / IC-informed approach. [3]
4. ADR-025 accepts a balanced entity-resolution architecture using LLM-informed disambiguation, mathematical coherence, and research-methodology guidance. [4]
5. `UNCERTAINTY_20250825.md` later reframes the system as local subjective expert assessment plus simple mathematical combination after each tool records its own construct-mapping uncertainty. [5]
6. `systems/uncertainty-propagation-architecture.md` preserves a more elaborate five-stage uncertainty pipeline and decision-gate design, while noting that cross-modal integration uncertainty was removed as complexity without research value. [6]

# Supersession Chain

The uncertainty thread is not one stable design. It is a sequence of attempted simplifications and replacements:

- **normalized confidence contract**: one `ConfidenceScore` model for every tool [1]
- **CERQual framework**: social-science uncertainty with contextual entity resolution, temporal KG, Bayesian pipeline, and distribution preservation [2]
- **quality degradation model**: confidence decreases through processing stages, later judged mathematically weak or too pessimistic [3]
- **IC-informed / Comprehensive7 references**: cited as the current framework by ADR-007 and ADR-010, but the referenced ADR-029 directory was not found in the preserved ADR tree during this ingest
- **local subjective expert assessment**: each tool assesses its own construct mapping, stores reasoning, and pipeline-level uncertainty is combined after local assessments [5]

# Entity Resolution

ADR-025 is accepted rather than marked superseded. It applies the uncertainty framework to entity resolution:

- use LLM context intelligence for pronouns, group references, and strategic ambiguity
- keep frequency counts separate from confidence
- avoid probability addition across instances
- preserve uncertainty distributions for genuinely unresolved cases
- produce research suitability guidance: quantitative, mixed-method, or qualitative/exploratory [4]

This connects uncertainty architecture to a concrete KGAS thesis risk: discourse analysis often depends on ambiguous references such as "we", "they", groups, factions, and strategic vagueness.

# Later Local-Assessment Model

The 2025-08 uncertainty note is the cleanest later position in this slice. It says uncertainty scores are subjective expert assessments by design, not calibrated measurements. Every tool operation is a construct mapping, such as PDF to text, text to entities, or graph to clusters. The tool reports one uncertainty number and reasoning for whether its output validly represents the target construct. [5]

The note rejects hardcoded uncertainty reductions, global uncertainty, and rule-based propagation. Aggregation can reduce uncertainty when the aggregating tool sees convergent evidence and makes a contextual assessment. Sequential pipeline uncertainty can then be combined mathematically after local assessments. [5]

# Verification Gap

ADR-007, ADR-010, ADR-025, and the uncertainty propagation architecture all reference an ADR-029 IC-informed framework path. A focused `find` under `docs/architecture` did not locate that ADR-029 directory in the preserved `digimon_lineage_Digimons` archive during this ingest. Treat ADR-029 as a cited but currently missing source until another archive slice locates it.

# Links

- [Uncertainty Framework Evolution](/wiki/concepts/uncertainty-framework-evolution.md)
- [Uncertainty Traceability Architecture](/wiki/concepts/uncertainty-traceability-architecture.md)
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md)
- [Digimon Lineage Architecture ADRs Map](/wiki/sources/digimon-lineage-architecture-adrs-map.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-004-Normative-Confidence-Score-Ontology.md`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-007-uncertainty-metrics.md`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-010-Quality-System-Design.md`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/adrs/ADR-025-Entity-Resolution-Architecture.md`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/UNCERTAINTY_20250825.md`  
[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/architecture/systems/uncertainty-propagation-architecture.md`
