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
- [Lit Review Phase 5 Reasoning Engine](/wiki/sources/lit-review-phase5-reasoning-engine.md) - cross-purpose reasoning evidence with internally consistent but demo-scale production claims.
- [Lit Review Phase 6 Production Validation](/wiki/sources/lit-review-phase6-production-validation.md) - internal production-validation/deployment-simulation package with remediated final claims and stress-test limits.
- [Digimon Lineage Architecture ADRs Map](/wiki/sources/digimon-lineage-architecture-adrs-map.md) - ADR backbone for scope, deployment, errors, theory architecture, analytical purpose, and tool layers.
- [Academic Proof Of Concept Scope](/wiki/concepts/academic-proof-of-concept-scope.md) - reason to read "production" claims as internal academic readiness unless external deployment evidence exists.
- [Digimon Lineage Data Storage ADRs](/wiki/sources/digimon-lineage-data-storage-adrs.md) - storage ADR slice explaining Qdrant removal, Neo4j/SQLite bi-store rationale, and PostgreSQL migration threshold.
- [Storage Architecture Evolution](/wiki/concepts/storage-architecture-evolution.md) - scale-indexed interpretation of KGAS storage claims across target docs and implementation-status evidence.
- [Digimon Lineage Uncertainty Quality ADRs](/wiki/sources/digimon-lineage-uncertainty-quality-adrs.md) - uncertainty, quality, and entity-resolution ADR slice showing multiple superseded frameworks and the later local-assessment model.
- [Uncertainty Framework Evolution](/wiki/concepts/uncertainty-framework-evolution.md) - conceptual arc from numeric confidence normalization to auditable construct-mapping uncertainty.
- [Digimon Lineage Tool Orchestration ADRs](/wiki/sources/digimon-lineage-tool-orchestration-adrs.md) - tool-interface ADR slice connecting contracts, adapters, MCP, and structured output.
- [Layered Tool Interface Architecture](/wiki/concepts/layered-tool-interface-architecture.md) - three-layer interpretation of KGAS tool surfaces and status caveats.
- [Digimon Lineage Analysis Expansion ADRs](/wiki/sources/digimon-lineage-analysis-expansion-adrs.md) - cross-modal, schema, simulation, statistical, and local API architecture ambitions.
- [Analysis Expansion Architecture](/wiki/concepts/analysis-expansion-architecture.md) - thesis-level expansion from graph building toward multi-representation social theory analysis.
- [Digimon Lineage Ops Scaffolding](/wiki/sources/digimon-lineage-ops-scaffolding.md) - CI, docs governance, interface validation, Docker deployment, and dependency-locking surfaces with status caveats.
- [Digimon Lineage Config And Contracts](/wiki/sources/digimon-lineage-config-contracts.md) - central config loader/settings, orchestration/monitoring configs, phase interfaces, theory validator, and partial tool-contract graph.
- [Lit Review Multi Agent System](/wiki/sources/lit-review-multi-agent-system.md) - multi-agent implementation/evaluation harness and six-phase external evaluation record.
- [Lit Review Multi Agent System Corpus](/wiki/sources/lit-review-multi-agent-system-corpus.md) - full 648-file inventory separating the outer lit-review harness from the nested V5.2 generator/evidence/remediation corpus.
- [Multi Agent Evidence Harness](/wiki/concepts/multi-agent-evidence-harness.md) - procedural evidence pattern for isolated implementation, external evaluation, remediation, and 100/100 gates.
- [Digimon Lineage UI Recovered Components](/wiki/sources/digimon-lineage-ui-recovered-components.md) - recovered and archived KGAS UI/demo/dashboard files.
- [Recovered UI Demo Surface](/wiki/concepts/recovered-ui-demo-surface.md) - human-facing inspection/demo layer for upload, analysis, graph status, query, and export workflows.
- [Current Code Verification 2026-06-25](/wiki/sources/current-code-verification-2026-06-25.md) - current cleaned-checkout verification of storage, tool, API/MCP, UI, entry-point, and test claims.
- [Current Runtime Import Check 2026-06-25](/wiki/sources/current-runtime-import-check-2026-06-25.md) - import-only runtime evidence: tool contracts import, cross-modal API import fails on `AnalysisRequest`, and MCP import fails on missing `neo4j`.
- [Current Runtime Repair Plan 2026-06-25](/wiki/sources/current-runtime-repair-plan-2026-06-25.md) - conservative repair order for cross-modal API import/calling-contract mismatch and MCP dependency readiness.
- [Current Status Verification Discipline](/wiki/concepts/current-status-verification-discipline.md) - status rule separating architecture intent, evidence reports, current files, and runtime proof.
- [Lit Review Semantic Hypergraph Application Results](/wiki/sources/lit-review-semantic-hypergraph-application-results.md) - semantic-hypergraph extraction/application results showing notation, table, algorithm, and application-fidelity gaps.
- [Formal Notation As Theory Content](/wiki/concepts/formal-notation-as-theory-content.md) - schema-fidelity rule for formal theories with notation systems and pattern languages.
- [Lit Review Universal Theory Applicator](/wiki/sources/lit-review-universal-theory-applicator.md) - schema-driven universal theory application framework, critique, enhancement template, and Young 1996 output.
- [Complexity Conservation In Theory Application](/wiki/concepts/complexity-conservation-in-theory-application.md) - lesson that generalization shifts theory-specific work into schemas, prompts, validation, and metrics.
- [Lit Review Model Form Detection Results](/wiki/sources/lit-review-model-form-detection-results.md) - optimized experimental outputs for classifying theory representational form, with an internal consistency caveat.
- [Model Form Routing](/wiki/concepts/model-form-routing.md) - rule that schema/application form should follow the theory structure rather than platform preference.
- [Lit Review Prompt Variation Model Routing](/wiki/sources/lit-review-prompt-variation-model-routing.md) - prompt-variation work for theory count, specialized detectors, confidence integration, and hybrid schemas.
- [Prompt Calibration Loop](/wiki/concepts/prompt-calibration-loop.md) - method for turning detector prompts into calibrated routing components.
- [Lit Review Validation Retesting Empty Directories](/wiki/sources/lit-review-validation-retesting-empty.md) - negative evidence that the preserved retesting subtree has directories but no files.
- [Lit Review Experimental Testing Corpus](/wiki/sources/lit-review-experimental-testing-corpus.md) - 41-file experimental-testing inventory with top-level 100%-validation claims and links to detailed routing/retesting pages.
- [Lit Review Local Repo Metadata](/wiki/sources/lit-review-local-repo-metadata.md) - nested `sb_ontologies` git provenance, trip-backup autosave branch, and local agent timeout settings.
- [Grusch UAP Information Disorder Output](/wiki/sources/grusch-uap-information-disorder-output.md) - structured generated application output over a UAP hearing/public-information case.
- [Information Disorder Application Artifact](/wiki/concepts/information-disorder-application-artifact.md) - generated-output pattern for actors, messages, interpreters, credibility, transparency, and narrative contestation.
- [Lit Review Old Schema Archive Inventory](/wiki/sources/lit-review-old-schema-archive-inventory.md) - inventory of 56 archived early theory schemas, common schema fields, and one parseability gap.
- [Old Schema Corpus Breadth](/wiki/concepts/old-schema-corpus-breadth.md) - finding that the thesis attempted a broad interdisciplinary schema corpus before later validation discipline.
- [Lit Review Legacy System Framing](/wiki/sources/lit-review-legacy-system-framing.md) - archived Analyst/Assembler methodology, computational social science framing, and critical validation problems.
- [Analyst Assembler Pattern](/wiki/concepts/analyst-assembler-pattern.md) - workflow split between LLM interpretation and deterministic schema assembly.
- [Lit Review Literature Corpus Inventory](/wiki/sources/lit-review-literature-corpus-inventory.md) - 73-file preserved corpus with 47 indexed papers across theory, persuasion, intervention, information, and behavior-change domains.
- [Literature Corpus As Theory Testbed](/wiki/concepts/literature-corpus-as-theory-testbed.md) - corpus-level explanation for why model routing and hybrid representations became necessary.
- [Lit Review Operational Code Corpus](/wiki/sources/lit-review-operational-code-corpus.md) - operational-code/WorldView source family and Young 1996 schema-variant comparison.
- [Schema Variant Drift](/wiki/concepts/schema-variant-drift.md) - evidence that extraction pass, prompt, and representation frame can preserve different theory content for the same source.
- [Lit Review Influence Operations Corpus](/wiki/sources/lit-review-influence-operations-corpus.md) - largest literature subfolder, with RAND/START source texts and generated Chapter 6-7 framework artifacts.
- [Multi-Level Influence Modeling](/wiki/concepts/multi-level-influence-modeling.md) - pattern where property-graph representation is justified by multi-level actors, messages, strategies, metrics, and feedback loops.
- [Lit Review Social Marketing Corpus](/wiki/sources/lit-review-social-marketing-corpus.md) - social-marketing folder preserving CORE/sharedProps/post-processing machinery and a missing generated-YAML caveat.
- [Lit Review Remaining Literature Folders](/wiki/sources/lit-review-remaining-literature-folders.md) - grouped remainder inventory, with poliheuristic and framing-theory generated schema notes.
- [Lit Review Schemas Corpus Inventory](/wiki/sources/lit-review-schemas-corpus-inventory.md) - mature schema corpus with variant families, debug JSON, fulltext sidecars, and execution prompts.
- [Lit Review Turner Social Identity Extraction](/wiki/sources/lit-review-turner-social-identity-extraction.md) - paired extraction outputs showing six-theory nested extraction versus single Self-Categorization Theory extraction.
- [Multi Theory Extraction Split](/wiki/concepts/multi-theory-extraction-split.md) - extraction-mode design choice between theoretical landscape preservation and executable single-theory output.
- [Lit Review Social Identity Theory Schema](/wiki/sources/lit-review-social-identity-theory-schema.md) - smaller adjacent Self-Categorization Theory schema pass over the same Turner/Oakes source.
- [Lit Review Young1996 Schema Family](/wiki/sources/lit-review-young1996-schema-family.md) - mature Young/WorldView schema family spanning theory-only, multi-pass, enhanced, computational, and execution-prompt variants.
- [Lit Review ELM Schema](/wiki/sources/lit-review-elm-schema.md) - compact sequential persuasion schema contrasting with Young's complex schema family.
- [Lit Review Semantic Hypergraph Schema Family](/wiki/sources/lit-review-semantic-hypergraph-schema-family.md) - 19-file schema-variant and debug-output inventory for the formal Semantic Hypergraph theory.
- [Lit Review Src Code Inventory](/wiki/sources/lit-review-src-code-inventory.md) - 90-file code inventory showing schema creation, application, testing, visualization, and UI surfaces.
- [Lit Review Schema Application Code](/wiki/sources/lit-review-schema-application-code.md) - 24-script application layer connecting theory schemas to Carter, Young/WorldView, Semantic Hypergraph, and universal-applicator runs.
- [Lit Review Visualization Code](/wiki/sources/lit-review-visualization-code.md) - 9-script inspection layer for Carter cognitive maps and Semantic Hypergraph instances.
- [Lit Review UI Code](/wiki/sources/lit-review-ui-code.md) - 3-file Streamlit review surface for selecting schemas/texts, running analysis, browsing results, charting outputs, and chatting over prior JSON.
- [Lit Review Testing Code](/wiki/sources/lit-review-testing-code.md) - 14-script testing/debug corpus for schema comparisons, prompt routing, provider checks, SH extraction, and six-phase multi-agent execution.
- [Lit Review Root Files](/wiki/sources/lit-review-root-files.md) - top-level project framing, prompts, logs, source texts, environment files, and accidental package-output artifacts.
- [Lit Review Docs Bundle](/wiki/sources/lit-review-docs-bundle.md) - 13-file methodology/reference bundle covering meta-schema, multiphase extraction, n-ary relations, schema enhancement, universal-applicator critique, and archived strategies.
- [Lit Review Data And Examples](/wiki/sources/lit-review-data-examples.md) - 28-file grounding slice with source papers, test texts, Carter examples, hybrid schemas, OWL/causal stress tests, and Grusch prompt material.
- [Lit Review Evidence Corpus](/wiki/sources/lit-review-evidence-corpus.md) - 114-file six-phase evidence corpus, including Phase 1 purpose classification and links to detailed Phase 2-6 pages.
- [Lit Review Debug Improved Results](/wiki/sources/lit-review-debug-improved-results.md) - 9 JSON intermediate phase outputs for Carter and Semantic Hypergraph cases, with empty `analysis_results/` negative evidence.
- [Lit Review Results Corpus](/wiki/sources/lit-review-results-corpus.md) - top-level result files showing WorldView/Carter application attempts, cognitive-mapping critique, simplified meta-schema findings, and generated Iran/influence/SH artifacts.

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
