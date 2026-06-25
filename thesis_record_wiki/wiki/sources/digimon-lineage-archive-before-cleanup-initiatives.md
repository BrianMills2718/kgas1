---
type: SourceSummary
title: Digimon Lineage Archive Before Cleanup Initiatives
description: Cross-cutting initiative slice from the August 2025 pre-cleanup roadmap archive, covering theory extraction integration, uncertainty implementation, storage consistency, PostgreSQL migration, orchestration, imports, identity, risk, performance, and resource plans.
tags: [source, digimon-lineage, archive, initiatives, roadmap, storage, orchestration, uncertainty]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ARCHIVE_BEFORE_CLEANUP_20250805/initiatives/
confidence: high
---

# Summary

`ARCHIVE_BEFORE_CLEANUP_20250805/initiatives/` is a 19-file cross-cutting plan bundle from the August 2025 roadmap archive. It totals about 228 KB by directory audit and has aggregate content-manifest hash `2903676d1015d80128fab347e4ee12f72f7782a45813a34310a05e13e23737a5`. [1]

The initiatives are mostly planning and design artifacts rather than completion evidence. Their value is that they show the architecture pressures KGAS was trying to resolve: integrating the standalone theory-extraction system, making uncertainty auditable, keeping Neo4j and SQLite consistent, deciding when PostgreSQL becomes necessary, moving from hardcoded vertical-slice execution to dynamic orchestration, resolving import failures, consolidating identity services, and setting risk/performance/success metrics. [1]

# File Inventory

| File Or Area | Role |
| --- | --- |
| `theory-extraction-integration-plan.md` | Critical MVP plan to wrap `/experiments/lit_review` and integrate theory extraction with ServiceManager, stores, tools, MCP, and cross-modal analysis. [2] |
| `uncertainty-implementation-plan.md` | Layered uncertainty roadmap from basic confidence to contextual entity resolution and Bayesian propagation. [3] |
| `two_stage_additional_notes.md` | Research-analysis memo for two-stage architecture: theory schema structures data, analytic prompt evaluates. [4] |
| Risk-management framework file | Quantitative risk framework for Phase 7 service architecture and Phase 8 external integrations. [5] |
| `INITIALIZATION_SEQUENCE_SPECIFICATION.md` | Initialization order and validation plan for config, imports, databases, tools, agents, orchestration, and rollback. [1] |
| `MODULE_IMPORT_RESOLUTION.md` | Diagnosis of MCP import failures caused by relative-import assumptions across execution contexts. [6] |
| `bi-store-consistency-plan.md` | Tentative high-priority plan for Neo4j + SQLite transaction coordination and entity-ID synchronization. [7] |
| `postgresql-migration-plan.md` | Scale-triggered migration plan from SQLite to PostgreSQL + pgvector for 50,000+ entity research corpora. [8] |
| `concurrency-strategy.md` | AnyIO structured-concurrency target architecture and warning about blocking calls in async contexts. [9] |
| `dynamic-orchestration-initiative.md` | Planned shift from hardcoded vertical-slice pipeline to MCP/DAG/agent/adaptive/provenance orchestration. [10] |
| `identity/` | Identity-service clarification and migration plan for basic, enhanced, and FAISS variants. [11] |
| `performance/` and `performance-monitoring-baselines.md` | Performance optimization and baseline-monitoring plans. [1] |
| `standardized-success-metrics.md` | Cross-phase success-metric and threshold framework. [1] |
| Other files | Error handling, resource management, programmatic dependency analysis, module import, and initialization plans. [1] |

# Theory Extraction Integration

The theory-extraction integration plan is explicit about the central status split: the experimental lit-review system is described as fully functional with strong validation claims, while the main KGAS integration is marked not connected to ServiceManager, data stores, or the tool pipeline. The target is a wrapper-pattern integration through a `TheoryExtractionService`, Neo4j/SQLite persistence, T-THEORY tools, MCP exposure, and cross-modal integration. [2]

This is consistent with the proposal and lit-review wiki record: the theory-extraction work had substantial standalone artifacts, but the integration claim requires separate evidence. [2]

# Uncertainty And Two-Stage Analysis

The uncertainty implementation plan starts from Layer 1 basic confidence scores and immediately identifies uncertainty provenance integration as required for completion. Later layers include contextual entity resolution and Bayesian uncertainty propagation. [3]

The two-stage notes generalize the architecture: theory schemas structure the data first, then analytic prompts evaluate and interpret. They also bring in digital humanities, qualitative-data management, LLM-as-judge, self-consistency, guardrails, human supervision, and resource-management considerations. [4]

# Storage, Identity, And Concurrency

The bi-store plan says KGAS's Neo4j + SQLite architecture lacked atomic cross-store transactions, entity-ID synchronization, rollback procedures, and consistency validation. It proposes a transaction coordinator, transaction log, centralized entity-ID manager, and consistency validator. [7]

The PostgreSQL plan is explicitly scale-triggered rather than default: PostgreSQL + pgvector becomes relevant for 1,000+ papers, 50,000+ entities, complex correlations, concurrent analytics, and advanced statistical/window functions. [8]

The concurrency strategy labels AnyIO structured concurrency as the target architecture while warning that 20+ blocking `time.sleep()` calls in async contexts created resource-leak and event-loop-blocking risks. This aligns with the reliability phase's catastrophic async-resource-leak concern. [9]

The identity-service notes preserve a versioning problem: the basic identity service was active via ServiceManager, while enhanced and FAISS versions existed but were not integrated. A later migration note says a consolidated implementation had been created but migration was pending. [11]

# Orchestration And Imports

The dynamic-orchestration initiative says the vertical slice was still hardcoded and sequential, without MCP integration, adaptive execution, agent execution, parallel execution, or persistent provenance. The planned target was MCP-based tool access, DAG execution, WorkflowAgent execution, adaptive conditional branches, and provenance persistence. [10]

The module-import resolution note diagnoses a concrete failure mode: MCP tool loading failed when relative imports were run in the wrong execution context, leading to limited mode and reduced functionality. Several solutions are considered, including absolute imports, fallback import strategies, package-structure enforcement, and an import-service pattern. [6]

# Risk, Metrics, And Claim Discipline

The risk-management framework preserves a more formal planning style, using risk matrices, probability distributions, Monte Carlo examples, and phase-specific risks for service architecture and external integrations. [5]

The standardized-success-metrics file tries to make phase completion measurable, but because this initiative bundle is mostly planning material, those metrics should be treated as intended gates unless corroborated by separate evidence outputs. [1]

A targeted scan of this initiative slice found no literal OpenAI or Google API keys in the raw files. [1]

# Interpretation

This initiative slice should be used to understand what problems KGAS maintainers knew about and how they intended to address them. It is especially valuable for architectural lineage because it connects separate wiki threads: theory extraction, uncertainty, bi-store storage, import fragility, identity-service variants, dynamic orchestration, and reliability/fail-fast work.

It should not be used as proof that those initiatives were completed.

# Relationship To Wiki

- [Digimon Lineage Archive Before Cleanup 2025 08 05 Overview](digimon-lineage-archive-before-cleanup-2025-08-05-overview.md): parent archive overview.
- [Digimon Lineage Archive Before Cleanup Phases](digimon-lineage-archive-before-cleanup-phases.md): related phase/evidence chronology.
- [Digimon Lineage Proposal Rewrite 2025 08 12](digimon-lineage-proposal-rewrite-2025-08-12.md): related dissertation-facing theory/uncertainty framing.
- [Storage Architecture Evolution](../concepts/storage-architecture-evolution.md): related Neo4j/SQLite/PostgreSQL arc.
- [Layered Tool Interface Architecture](../concepts/layered-tool-interface-architecture.md): related MCP/tool interface and orchestration arc.
- [Uncertainty Framework Evolution](../concepts/uncertainty-framework-evolution.md): related uncertainty design arc.
- [Evidence Claim Discipline](../concepts/evidence-claim-discipline.md): relevant because these are plans and gates, not completion proof.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ARCHIVE_BEFORE_CLEANUP_20250805/initiatives/`

[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ARCHIVE_BEFORE_CLEANUP_20250805/initiatives/theory-extraction-integration-plan.md`

[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ARCHIVE_BEFORE_CLEANUP_20250805/initiatives/uncertainty-implementation-plan.md`

[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ARCHIVE_BEFORE_CLEANUP_20250805/initiatives/two_stage_additional_notes.md`

[5] Risk-management framework file under `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ARCHIVE_BEFORE_CLEANUP_20250805/initiatives/`

[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ARCHIVE_BEFORE_CLEANUP_20250805/initiatives/MODULE_IMPORT_RESOLUTION.md`

[7] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ARCHIVE_BEFORE_CLEANUP_20250805/initiatives/bi-store-consistency-plan.md`

[8] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ARCHIVE_BEFORE_CLEANUP_20250805/initiatives/postgresql-migration-plan.md`

[9] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ARCHIVE_BEFORE_CLEANUP_20250805/initiatives/concurrency-strategy.md`

[10] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ARCHIVE_BEFORE_CLEANUP_20250805/initiatives/dynamic-orchestration-initiative.md`

[11] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ARCHIVE_BEFORE_CLEANUP_20250805/initiatives/identity/`
