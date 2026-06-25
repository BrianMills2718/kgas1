# Overview

* [Overview](overview.md) - top-level map of the thesis work record and preservation strategy.
* [Progress](../PROGRESS.md) - durable mission, acceptance criteria, completed commits, and next slices for the thesis record wiki.

# Sources

* [Recovery Archive Manifest](sources/recovery-archive-manifest-2026-04-04.md) - why `archive_full_record/` exists and what it preserves.
* [Recovery Inventory](sources/recovery-inventory.md) - preservation metadata, sizes, file counts, git heads, and verification errors.
* [Current Repo Context](sources/current-repo-context.md) - current README, CLAUDE notes, branch, and git history baseline.
* [Digimons Documentation Repository](sources/digimons-docs-documentation-repository.md) - documentation-only lineage variant and its source-of-truth split.
* [Digimons Minimal Clean Reference](sources/digimons-minimal-clean-reference.md) - compact 2025-09-04 clean KGAS reference line.
* [Tool Compatibility Decision](sources/tool-compatibility-decision.md) - type-based composition decision and POC context.
* [JayLZhou GraphRAG Upstream](sources/jaylzhou-graphrag-upstream.md) - external DIGIMON / GraphRAG reference repository.
* [Digimon Core Sparse Contract Layer](sources/digimon-core-sparse-contract-layer.md) - sparse variant contract-first migration, evidence, and bottleneck findings.
* [Digimon Autoloop Operator Routing](sources/digimon-autoloop-operator-routing.md) - later operators-first DIGIMON state, benchmark evidence, and supported-surface boundary.
* [Digimon Lineage Active State](sources/digimon-lineage-active-state.md) - large Digimons lineage root state and September 2025 reality-verification arc.
* [Digimon Lineage Tool Compatibility](sources/digimon-lineage-tool-compatibility.md) - large lineage tool_compatability subtree and active vertical-slice context.
* [Digimon Lineage Architecture Docs](sources/digimon-lineage-architecture-docs.md) - large lineage architecture target-design docs and uncertainty/traceability critical review.
* [Digimon Lineage Evidence Archives](sources/digimon-lineage-evidence-archives.md) - evidence archive slice focused on false-claim correction and system-level validation discipline.
* [Lit Review Theory Extraction Experiment](sources/lit-review-theory-extraction-experiment.md) - lit_review experiment subsystem for automated theory extraction, schema generation, and validation claims.
* [Carter Theory Analysis Output](sources/carter-theory-analysis-output.md) - generated Carter speech outputs applying cognitive mapping and framing theory.
* [Lit Review Schema Creation Production Path](sources/lit-review-schema-creation-production-path.md) - schema_creation code and prompts behind the lit-review extraction pipeline.
* [Lit Review Validation Results](sources/lit-review-validation-results.md) - validation reports for Young 1996, framing effects, and Lofland-Stark complexity testing.
* [Lit Review Phase 2-3 Evidence](sources/lit-review-phase2-3-evidence.md) - evidence artifacts for balanced vocabulary extraction and schema generation.
* [Lit Review Phase 4 Integration Pipeline](sources/lit-review-phase4-integration-pipeline.md) - integration pipeline evidence with certification/status caveats.
* [Lit Review Phase 5 Reasoning Engine](sources/lit-review-phase5-reasoning-engine.md) - cross-purpose reasoning engine evidence and demo-scale caveats.
* [Lit Review Phase 6 Production Validation](sources/lit-review-phase6-production-validation.md) - production validation evidence package and deployment-claim caveats.
* [Digimon Lineage Architecture ADRs Map](sources/digimon-lineage-architecture-adrs-map.md) - first ADR decision-history slice for large lineage architecture docs.
* [Digimon Lineage Data Storage ADRs](sources/digimon-lineage-data-storage-adrs.md) - storage ADR slice covering Qdrant removal, Neo4j/SQLite rationale, and PostgreSQL migration threshold.
* [Digimon Lineage Uncertainty Quality ADRs](sources/digimon-lineage-uncertainty-quality-adrs.md) - uncertainty, quality, and entity-resolution ADR slice with supersession caveats.
* [Digimon Lineage Tool Orchestration ADRs](sources/digimon-lineage-tool-orchestration-adrs.md) - contract, adapter, MCP, and structured-output ADR slice.
* [Digimon Lineage Analysis Expansion ADRs](sources/digimon-lineage-analysis-expansion-adrs.md) - cross-modal, schema, ABM, statistics, and local API ADR slice.
* [Lit Review Multi Agent System](sources/lit-review-multi-agent-system.md) - isolated multi-agent implementation/evaluation harness and six-phase evaluation record.
* [Digimon Lineage UI Recovered Components](sources/digimon-lineage-ui-recovered-components.md) - archived and recovered KGAS UI/demo/dashboard material.
* [Current Code Verification 2026-06-25](sources/current-code-verification-2026-06-25.md) - focused current-checkout verification of storage, tool, API/MCP, UI, entry-point, and test claims.
* [Current Runtime Import Check 2026-06-25](sources/current-runtime-import-check-2026-06-25.md) - import-only runtime check for current contract, cross-modal API, and MCP modules.
* [Current Runtime Repair Plan 2026-06-25](sources/current-runtime-repair-plan-2026-06-25.md) - non-invasive source-level plan for repairing first import/runtime blockers.
* [Lit Review Semantic Hypergraph Application Results](sources/lit-review-semantic-hypergraph-application-results.md) - semantic-hypergraph extraction/application results, critiques, visualizations, and scripts.
* [Lit Review Universal Theory Applicator](sources/lit-review-universal-theory-applicator.md) - schema-driven universal theory application framework, critique, and Young 1996 result.
* [Lit Review Model Form Detection Results](sources/lit-review-model-form-detection-results.md) - optimized experimental outputs for sequence/table/graph/hybrid model-form detection.
* [Lit Review Prompt Variation Model Routing](sources/lit-review-prompt-variation-model-routing.md) - prompt-calibration artifacts for theory-count, model-form, confidence, and hybrid routing.
* [Lit Review Validation Retesting Empty Directories](sources/lit-review-validation-retesting-empty.md) - negative evidence that preserved validation_retesting subdirectories contain no files.

# Entities

* [KGAS](entities/kgas.md) - Knowledge Graph Analysis System, the thesis implementation line.

# Concepts

* [Full Record Preservation](concepts/full-record-preservation.md) - policy for preserving the complete messy historical record before cleanup.
* [Research Lineage](concepts/research-lineage.md) - how Digimons variants, KGAS, archives, and cleaned repo relate.
* [Verification Gaps](concepts/verification-gaps.md) - known places where the record is incomplete, permission-limited, or needs later review.
* [Documentation Status Truthfulness](concepts/documentation-status-truthfulness.md) - recurring attempt to separate target architecture from verified implementation status.
* [Type-Based Tool Composition](concepts/type-based-tool-composition.md) - later compatibility strategy for turning incompatible tools into composable typed operators.
* [GraphRAG Upstream Lineage](concepts/graphrag-upstream-lineage.md) - how local Digimon/KGAS work relates to JayLZhou GraphRAG and DIGIMON method decomposition.
* [Contract-First Migration](concepts/contract-first-migration.md) - migration from split tool interfaces toward KGASTool contracts.
* [Relationship Extraction Bottleneck](concepts/relationship-extraction-bottleneck.md) - evidence that graph construction failed when relationship extraction was missing or silent.
* [Adaptive Operator Routing](concepts/adaptive-operator-routing.md) - falsifiable thesis that per-question operator routing should beat fixed graph and non-graph baselines.
* [MCP Autoloop Interface](concepts/mcp-autoloop-interface.md) - historical MCP/UKRF plans and later MCP tool-surface status.
* [Graph Build Manifest](concepts/graph-build-manifest.md) - build-manifest and tool-gating contract for truthful graph capability exposure.
* [Reality Verification Arc](concepts/reality-verification-arc.md) - September 2025 correction chain around implementation claims and roadmap consolidation.
* [Vertical Slice vs Main System](concepts/vertical-slice-vs-main-system.md) - coexisting simple verified vertical slice and complex main-system architecture.
* [Uncertainty Traceability Architecture](concepts/uncertainty-traceability-architecture.md) - KGAS uncertainty, provenance, traceability, and over-engineering tension.
* [Evidence Claim Discipline](concepts/evidence-claim-discipline.md) - rule that claims must match component, integration, workflow, or benchmark validation level.
* [Automated Theory Extraction](concepts/automated-theory-extraction.md) - lit-review experiment line for extracting theory schemas from papers and applying them to data.
* [Multi-Theory Application Artifact](concepts/multi-theory-application-artifact.md) - concrete generated-output pattern for applying multiple theories to one empirical text.
* [Schema Extraction Pipeline Evolution](concepts/schema-extraction-pipeline-evolution.md) - evolution toward full-vocabulary, model-adaptive, no-truncation theory extraction.
* [Complexity Accuracy Pattern](concepts/complexity-accuracy-pattern.md) - validation pattern that simpler theories are more reliable automation targets.
* [Balance Driven Validation](concepts/balance-driven-validation.md) - five-purpose balance metric pattern and its evidence/overfitting risk.
* [Academic Proof Of Concept Scope](concepts/academic-proof-of-concept-scope.md) - scope decision prioritizing academic correctness, provenance, and local reproducibility.
* [Storage Architecture Evolution](concepts/storage-architecture-evolution.md) - KGAS storage evolution from tri-store avoidance to Neo4j/SQLite and Neo4j/PostgreSQL.
* [Uncertainty Framework Evolution](concepts/uncertainty-framework-evolution.md) - movement from confidence fields to auditable local uncertainty reasoning.
* [Layered Tool Interface Architecture](concepts/layered-tool-interface-architecture.md) - three-layer reconciliation of implementation tools, internal contracts, and MCP access.
* [Analysis Expansion Architecture](concepts/analysis-expansion-architecture.md) - KGAS expansion from graph extraction to cross-modal, schema, simulation, statistical, and local automation modes.
* [Multi Agent Evidence Harness](concepts/multi-agent-evidence-harness.md) - isolated implementation/evaluation pattern used to enforce 100/100 phase gates.
* [Recovered UI Demo Surface](concepts/recovered-ui-demo-surface.md) - preserved static, FastAPI, Streamlit, and React UI/demo surfaces.
* [Current Status Verification Discipline](concepts/current-status-verification-discipline.md) - rule for separating architecture, evidence, current code, and runtime status.
* [Formal Notation As Theory Content](concepts/formal-notation-as-theory-content.md) - lesson that formal notation and algorithms are first-class schema content for formal theories.
* [Complexity Conservation In Theory Application](concepts/complexity-conservation-in-theory-application.md) - lesson that universal frameworks move rather than eliminate theory-specific complexity.
* [Model Form Routing](concepts/model-form-routing.md) - routing theories to sequence, table, graph, statistical, or hybrid forms instead of forcing one representation.
* [Prompt Calibration Loop](concepts/prompt-calibration-loop.md) - detector-prompt pattern: test known cases, identify bias, refine criteria, and add arbitration logic.

# Timeline

* [Evolution Timeline](timeline/evolution-timeline.md) - initial timeline from git history and recovery metadata.

# Lineage Variants

* [Current Clean Repo](variants/current-clean-repo.md) - tracked repo state after cleanup and recovery.
* [Filesystem Snapshot 2026-04-04](variants/filesystem-snapshot-2026-04-04.md) - moved snapshot preserving pre-cleanup archive-only material.
* [Digimons Minimal](variants/digimons-minimal.md) - local repo matching `kgas1` `origin/master` commit `2c59a1f`.
* [Digimons Clean For Real](variants/digimons-clean-for-real.md) - cleanup-focused branch history snapshot.
* [Digimons Old](variants/digimons-old.md) - large pre-KGAS / earlier Digimons archive.
* [Digimon Core Sparse](variants/digimon-core-sparse.md) - sparse / contract-oriented variant with many evidence documents.
* [Digimons Docs](variants/digimons-docs.md) - documentation extraction variant.
* [Digimon Lineage Digimons](variants/digimon-lineage-digimons.md) - largest preserved lineage bundle.
* [Digimon v2](variants/digimon-v2.md) - later Digimon lineage repo.
* [Digimon Autoloop](variants/digimon-autoloop.md) - autoloop-related lineage variant.
