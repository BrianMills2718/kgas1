---
type: Entity
title: KGAS
description: Knowledge Graph Analysis System, the thesis implementation line for LLM-generated ontologies and fringe discourse analysis.
tags: [kgas, thesis, graphrag]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../README.md
  - ../CLAUDE.md
confidence: medium
---

# Summary

KGAS is the Knowledge Graph Analysis System described in the current README as the implementation for the dissertation topic "Theoretical Foundations for LLM-Generated Ontologies and Analysis of Fringe Discourse." [1]

The system's recurring implementation themes are entity extraction, relationship mapping, Neo4j graph storage, GraphRAG-style querying, provenance, uncertainty, reasoning traces, and tool orchestration. [1][2]

The first lineage ingest adds two historically important KGAS concerns: truthful separation of roadmap status from target architecture, and tool composability through semantic typed contracts. See [Documentation Status Truthfulness](/wiki/concepts/documentation-status-truthfulness.md) and [Type-Based Tool Composition](/wiki/concepts/type-based-tool-composition.md).

Brian clarified that much of the Digimon material extends or forks JayLZhou GraphRAG / DIGIMON. That upstream relationship should be explicit in any KGAS history. See [GraphRAG Upstream Lineage](/wiki/concepts/graphrag-upstream-lineage.md).

The `digimon_core_sparse` slice adds a contract-first migration thread and a concrete graph-construction warning: entity extraction can work while relationship extraction fails or is not invoked. See [Contract-First Migration](/wiki/concepts/contract-first-migration.md) and [Relationship Extraction Bottleneck](/wiki/concepts/relationship-extraction-bottleneck.md).

The `digimon_autoloop` slice captures a later operators-first DIGIMON state: 28 typed operators, MCP/direct tool access, benchmark modes, two-model graph-build/query design, and an explicit go/no-go plan for adaptive routing. See [Digimon Autoloop Operator Routing](/wiki/sources/digimon-autoloop-operator-routing.md), [Adaptive Operator Routing](/wiki/concepts/adaptive-operator-routing.md), and [Graph Build Manifest](/wiki/concepts/graph-build-manifest.md).

The first `digimon_lineage_Digimons` slice captures the September 2025 active state: a large repository with root KGAS README/CLAUDE guidance, an evidence log, a consolidated conservative roadmap, and operations investigations that corrected implementation-status claims. See [Digimon Lineage Active State](/wiki/sources/digimon-lineage-active-state.md), [Reality Verification Arc](/wiki/concepts/reality-verification-arc.md), and [Vertical Slice vs Main System](/wiki/concepts/vertical-slice-vs-main-system.md).

The architecture-docs slice captures the target design side of KGAS: theory automation, cross-modal analysis, bi-store storage, agent workflows, ADR cascades, and an internal uncertainty/traceability critique. See [Digimon Lineage Architecture Docs](/wiki/sources/digimon-lineage-architecture-docs.md) and [Uncertainty Traceability Architecture](/wiki/concepts/uncertainty-traceability-architecture.md).

The evidence-archive slice captures the verification-quality side: evidence files include both raw successes/errors and explicit archival of false system-integration claims. See [Digimon Lineage Evidence Archives](/wiki/sources/digimon-lineage-evidence-archives.md) and [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md).

The `experiments/lit_review` slice captures a separate thesis experiment line focused directly on automated theory extraction and application: papers become schemas, schemas are applied to data, and results are compared to the source paper. See [Lit Review Theory Extraction Experiment](/wiki/sources/lit-review-theory-extraction-experiment.md) and [Automated Theory Extraction](/wiki/concepts/automated-theory-extraction.md).

The Carter output sub-slice shows one concrete generated artifact from that line: cognitive mapping and framing theory applied to a Carter speech, then integrated. See [Carter Theory Analysis Output](/wiki/sources/carter-theory-analysis-output.md) and [Multi-Theory Application Artifact](/wiki/concepts/multi-theory-application-artifact.md).

The schema-creation source slice documents the production-path side of that line: external prompts, full-vocabulary handoff between phases, no-truncation extractor variants, and adaptive model-type selection. See [Lit Review Schema Creation Production Path](/wiki/sources/lit-review-schema-creation-production-path.md) and [Schema Extraction Pipeline Evolution](/wiki/concepts/schema-extraction-pipeline-evolution.md).

The validation-results slice gives the first conservative validation pattern: simple theories appear better suited to automation, while medium/cross-domain theories can still miss model type. See [Lit Review Validation Results](/wiki/sources/lit-review-validation-results.md) and [Complexity Accuracy Pattern](/wiki/concepts/complexity-accuracy-pattern.md).

The Phase 2-3 evidence slice shows how KGAS/lit-review tried to operationalize balanced descriptive, explanatory, predictive, causal, and intervention purposes. See [Lit Review Phase 2-3 Evidence](/wiki/sources/lit-review-phase2-3-evidence.md) and [Balance Driven Validation](/wiki/concepts/balance-driven-validation.md).

The Phase 4 slice shows an integrated balanced-purpose pipeline, but its archived evidence mixes passing tests, weak certification metrics, and later remediation claims. See [Lit Review Phase 4 Integration Pipeline](/wiki/sources/lit-review-phase4-integration-pipeline.md).

The Phase 5 slice extends that pipeline into cross-purpose reasoning and synthesis, with internally consistent demo evidence but still internal production-readiness claims. See [Lit Review Phase 5 Reasoning Engine](/wiki/sources/lit-review-phase5-reasoning-engine.md).

The Phase 6 slice records internal production validation and deployment simulation. It is useful as a final evidence package for the lit-review subsystem, but not proof of external production deployment. See [Lit Review Phase 6 Production Validation](/wiki/sources/lit-review-phase6-production-validation.md).

The ADR map clarifies that KGAS architecture intentionally centered academic proof-of-concept use: local single-node deployment, fail-fast errors, two-layer theory architecture, human question-driven analysis, and layered tools. See [Digimon Lineage Architecture ADRs Map](/wiki/sources/digimon-lineage-architecture-adrs-map.md) and [Academic Proof Of Concept Scope](/wiki/concepts/academic-proof-of-concept-scope.md).

The data-storage ADR slice adds a scale-indexed storage history: Qdrant was removed to avoid tri-store consistency complexity, Neo4j + SQLite was justified for local academic workflows, and Neo4j + PostgreSQL was later accepted for 50,000+ entity analytical workloads. See [Digimon Lineage Data Storage ADRs](/wiki/sources/digimon-lineage-data-storage-adrs.md) and [Storage Architecture Evolution](/wiki/concepts/storage-architecture-evolution.md).

The uncertainty/quality ADR slice shows why KGAS uncertainty claims must be read historically: confidence-score normalization, CERQual, and quality degradation were superseded, while later notes emphasize local LLM expert assessments with reasoning and mathematically coherent combination. See [Digimon Lineage Uncertainty Quality ADRs](/wiki/sources/digimon-lineage-uncertainty-quality-adrs.md) and [Uncertainty Framework Evolution](/wiki/concepts/uncertainty-framework-evolution.md).

The tool/orchestration ADR slice explains the intended agent-drivable surface: raw tools/adapters, internal KGASTool contracts, external MCP access, and schema-validated LLM outputs. See [Digimon Lineage Tool Orchestration ADRs](/wiki/sources/digimon-lineage-tool-orchestration-adrs.md) and [Layered Tool Interface Architecture](/wiki/concepts/layered-tool-interface-architecture.md).

The analysis-expansion ADR slice shows the larger KGAS thesis ambition: graph/table/vector movement, multiple schema paradigms, theory-to-ABM simulation, SEM/statistical analysis, and local automation interfaces. See [Digimon Lineage Analysis Expansion ADRs](/wiki/sources/digimon-lineage-analysis-expansion-adrs.md) and [Analysis Expansion Architecture](/wiki/concepts/analysis-expansion-architecture.md).

The lit-review multi-agent slice adds a procedural verification layer: isolated implementation agents, external evaluation agents, remediation, and 100/100 gates were used around the six lit-review phases. See [Lit Review Multi Agent System](/wiki/sources/lit-review-multi-agent-system.md) and [Multi Agent Evidence Harness](/wiki/concepts/multi-agent-evidence-harness.md).

The recovered UI slice preserves human-facing demo and inspection surfaces: static/mock UI, FastAPI backend UI, Streamlit dashboard, and a React analysis dashboard component. See [Digimon Lineage UI Recovered Components](/wiki/sources/digimon-lineage-ui-recovered-components.md) and [Recovered UI Demo Surface](/wiki/concepts/recovered-ui-demo-surface.md).

The current-code verification slice checks the cleaned checkout directly and finds important status gaps: documented root entry points and `src/ui` are absent, `src/mcp_server.py` and `src/api/cross_modal_api.py` exist, and Qdrant compatibility/mock code remains. See [Current Code Verification 2026-06-25](/wiki/sources/current-code-verification-2026-06-25.md).

The runtime import slice adds first execution evidence for the cleaned checkout: `src.core.tool_contract` imports, `src.api.cross_modal_api` fails on `AnalysisRequest` export wiring, and `src.mcp_server` fails because `neo4j` is not installed in the active environment. See [Current Runtime Import Check 2026-06-25](/wiki/sources/current-runtime-import-check-2026-06-25.md).

# Historical Position

KGAS appears to be the cleaned and renamed/organized continuation of earlier Digimons work. The preservation layer keeps multiple Digimons variants so the conceptual and implementation lineage can be reconstructed rather than inferred from the clean repo alone. See [Research Lineage](/wiki/concepts/research-lineage.md).

# Status Caveat

The current README and `CLAUDE.md` do not present one perfectly consistent status snapshot. The README emphasizes academic research capability and known limitations. The `CLAUDE.md` includes 2025 sprint and investigation notes, including claims about resolved uncertainties and integration plans. Treat status claims as time-indexed.

# Citations

[1] `../README.md`  
[2] `../CLAUDE.md`
