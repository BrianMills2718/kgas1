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

# Historical Position

KGAS appears to be the cleaned and renamed/organized continuation of earlier Digimons work. The preservation layer keeps multiple Digimons variants so the conceptual and implementation lineage can be reconstructed rather than inferred from the clean repo alone. See [Research Lineage](/wiki/concepts/research-lineage.md).

# Status Caveat

The current README and `CLAUDE.md` do not present one perfectly consistent status snapshot. The README emphasizes academic research capability and known limitations. The `CLAUDE.md` includes 2025 sprint and investigation notes, including claims about resolved uncertainties and integration plans. Treat status claims as time-indexed.

# Citations

[1] `../README.md`  
[2] `../CLAUDE.md`
