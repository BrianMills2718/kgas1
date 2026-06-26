---
type: Progress
title: Thesis Record Wiki Progress
description: Durable mission, acceptance criteria, completed commits, and next slices for the thesis record wiki.
tags: [progress, mission, thesis-record]
created: 2026-06-25
updated: 2026-06-25
confidence: high
---

# Mission: Thesis Record Wiki

## Objective

Build a Karpathy-style wiki over Brian's KGAS / Digimons / PhD thesis record so the full messy historical evolution is preserved, navigable, and auditable without modifying the raw archive.

## Acceptance Criteria

- [ ] Raw preservation material under `archive_full_record/` is not moved, deleted, repaired, or rewritten during wiki work.
- [ ] Every ingest creates or updates source/concept/entity/variant pages with citations to immutable source paths.
- [ ] Large source bundles are split into bounded slices before synthesis.
- [ ] `wiki/index.md`, `wiki/log.md`, and `raw/source_manifest.md` stay current.
- [ ] The wiki linter reports `100/100` before each commit.
- [ ] Each verified ingest slice is committed and pushed.

## Constraints

- Treat the wiki as derived navigation and synthesis, not ground truth.
- Preserve both positive evidence and negative evidence; do not make the thesis look cleaner or more successful than the sources warrant.
- Broken `.git` worktree pointers are provenance facts, not repair tasks, unless Brian explicitly requests recovery.
- Recommend the next step after every completed slice.

## Current Phase

Continue bounded ingest of `archive_full_record/lineage_variants/digimon_lineage_Digimons`, currently working through archived uncertainty stress-test slices.

## Completed

- `5c32569` initialized the thesis record wiki.
- `97a3fb7` ingested `Digimons_docs`, `Digimons_minimal`, and `Digimons_clean_for_real/tool_compatability`.
- `78e0f90` ingested JayLZhou GraphRAG upstream lineage and `digimon_core_sparse`.
- `901461d` ingested `digimon_autoloop`, including adaptive operator routing, MCP/autoloop interface, graph build manifest, negative MuSiQue dev evidence, and broken worktree pointer provenance.
- `dba77e8` ingested first `digimon_lineage_Digimons` root-state slice, including source summary, reality-verification arc, vertical-slice/main-system split, and progress tracker.
- `41fcead` ingested `digimon_lineage_Digimons/tool_compatability`, including large-bundle source summary and updates to type-based composition.
- `871f7e9` ingested `digimon_lineage_Digimons/docs/architecture` top-level slice, including architecture source summary and uncertainty traceability concept.
- `bf80bf9` ingested the evidence archive slice, including false-claim correction and evidence validation-level concept.
- `54d1422` ingested first `experiments/lit_review` slice, including automated theory extraction source summary, schema/application method concept, and validation-claim caveats.
- `c780da3` ingested `experiments/lit_review/carter_analysis_output`, including concrete generated output artifacts and multi-theory application artifact concept.
- `603f173` ingested `experiments/lit_review/src/schema_creation`, including prompt structure, information-loss fix, v13 extractors, and schema extraction pipeline evolution concept.
- `e006732` ingested `experiments/lit_review/validation_results`, including Young/framing/Lofland reports, selected baseline outputs, and complexity/accuracy pattern concept.
- `02e5258` ingested lit-review Phase 2-3 evidence slice, including vocabulary/schema balance claims, stored test outputs, and Phase 2 summary/test contradiction.
- `eb2165c` ingested lit-review Phase 4 integration pipeline slice, including integration claims, test artifacts, certification caveats, and remediation chronology.
- `6f1ba35` ingested lit-review Phase 5 reasoning engine slice, including cross-purpose reasoners, balance/integration metrics, and demo-scale caveats.
- `c4463a0` ingested lit-review Phase 6 production validation slice, including production-validation claims, final remediation results, and stress-test/deployment caveats.
- `48bd98a` ingested the first architecture ADR map slice, including scope/deployment/errors/theory/purpose/tool-layer backbone and academic proof-of-concept scope.
- `9bbc0e9` ingested the data/storage ADR slice, including vector-store consolidation, Neo4j + SQLite bi-store rationale, and PostgreSQL migration threshold.
- `28bed51` ingested the uncertainty/quality ADR slice, including confidence ontology, CERQual supersession, quality-system supersession, entity resolution, later local-assessment uncertainty model, and missing ADR-029 caveat.
- `c2b9aad` ingested the tool/orchestration ADR slice, including contract-first Layer 2, pipeline adapters, MCP Layer 3, structured output migration, and three-layer interface reconciliation.
- `1fcdd0d` ingested the analysis-expansion ADR slice, including cross-modal analysis, ABM simulation, statistical/SEM expansion, schema ecosystem, and local-only REST API.
- `ba02a82` ingested the lit-review multi-agent system slice, including isolated implementation/evaluation harness, current status claims, six phase evaluation results, and Phase 2 11/12-vs-100/100 caveat.
- `d43ae5b` ingested the UI recovered-components slice, including duplicate recovered/archive UI README, static/Streamlit/FastAPI UI surfaces, dashboard evidence, and recovered React component.
- `981e769` added the current-code verification slice, including storage, tool contract, API/MCP, UI, entry-point, and tests/current-layout findings.
- `236d6f5` added the runtime import check slice, including successful `src.core.tool_contract` import and failed `src.api.cross_modal_api` / `src.mcp_server` imports.
- `42e29c0` documented active-environment follow-up on the runtime import slice, including missing `neo4j`, direct `AnalysisRequest` import success, and `pip check` conflict.
- `2d7a878` added the non-invasive runtime repair plan for cross-modal API import/calling-contract mismatch and MCP dependency readiness.
- `7142a69` ingested the semantic-hypergraph application-results slice, including extraction critiques, notation inventory, application scripts, visualized instances, and formal-notation-as-theory-content concept.
- `dab78d3` ingested the universal theory applicator slice, including schema-driven stage architecture, critique, enhancement template, Young 1996 result, and complexity-conservation concept.
- `bfaa496` ingested the model-form detection slice, including Lofland-Stark sequential-funnel result, Heilman/framing internal inconsistency caveat, empty sibling result directories, and model-form-routing concept.
- `d700688` ingested the prompt-variation model-routing slice, including theory-count detection, specialized detector calibration, confidence integration, hybrid detection, and generated Young 1996 hybrid schema.
- `3a77822` documented negative evidence for empty `experimental_testing/validation_retesting` subdirectories.
- `9e17ffd` ingested the Grusch/UAP information-disorder output slice, including structured generated analysis and empty `analysis_results/` caveat.
- `5fa54dc` ingested the old schema archive inventory slice, including 56 YAML schemas, 55 parseable dictionaries, one malformed file, aggregate directory hash, and old-schema-corpus-breadth concept.
- `9b2fb37` ingested the legacy-system framing slice, including Analyst/Assembler methodology, post-processing guide, project overview, archived CLAUDE validation problems, and analyst-assembler-pattern concept.
- `bb67108` ingested the literature corpus inventory slice, including 73 preserved files, 47 indexed papers, topical folder counts, schema-location caveat, and literature-corpus-as-theory-testbed concept.
- `d2bb70a` ingested the operational-code corpus slice, including five source texts, four Young schema variants, schema-variant-drift concept, and YAML truth-value parsing caveat.
- `2184edf` ingested the influence-operations corpus slice, including folder inventory, RAND Chapter 6-7 generated artifacts, schema/debug counts, and multi-level-influence-modeling concept.
- `704edff` ingested the social-marketing corpus slice, including CORE/sharedProps/post_process artifacts, missing configured YAML caveat, and analyst-assembler-pattern update.
- `09c25a4` ingested the remaining literature folders grouped slice, including 23-folder inventory, poliheuristic schema note, framing schema note, and literature-corpus-as-theory-testbed update.
- `da010a0` ingested the mature schemas corpus inventory, including 54-file count, subfolder inventory, aggregate hash, ELM summary note, Young execution prompt note, and schema-extraction-pipeline-evolution update.
- `65f5de8` ingested the Turner social-identity extraction comparison, including extracted/extracted_single aggregate hashes, six-theory nested output, single Self-Categorization Theory output, and multi-theory-extraction-split concept.
- `6e7052b` ingested the social-identity-theory schema slice, including three-file aggregate hash, SIT JSON/YAML schema shape, and relationship to Turner extracted/extracted_single outputs.
- `63f37ea` ingested the Young1996 schema-family slice, including six-file aggregate hash, computational/enhanced/multi-pass variants, execution prompt, and schema-variant-drift update.
- `fb6bc1b` ingested the ELM schema slice, including two-file aggregate hash, compact sequential schema shape, and model-form-routing update.
- `893f3d8` ingested the semantic-hypergraph schema-family slice, including 19-file aggregate hash, enhanced/multi-pass/option variants, debug-output inventory, and formal-notation concept link.
- `24fef8c` documented the simple schema fixture caveat for `schemas/test_simple_schema.yml` as a synthetic hypergraph test artifact rather than substantive source.
- `f674a43` ingested the lit-review src code inventory, including 90-file aggregate hash and subdirectory roles for schema_creation, schema_application, testing, ui, and visualization.
- `31ff970` ingested the schema_application code slice, including 24-script aggregate hash, direct OpenAI/hardcoded path counts, universal applicator implementation, and caveats.
- `5efdb91` ingested the visualization code slice, including 9-script aggregate hash, Carter cognitive-map visualization scripts, Semantic Hypergraph visualization scripts, and hardcoded-path caveats.
- `dc6dce5` ingested the UI code slice, including 3-file aggregate hash, Streamlit schema-analysis review surface, YAML corpus statistics script, and launcher caveats.
- `1ce0e6f` ingested the testing code slice, including 14-script aggregate hash, processor comparison, prompt-routing, computational-schema execution, SH extraction, provider-debug, and multi-agent runner caveats.
- `1e5bb6e` ingested the lit-review root-files slice, including 22-file aggregate hash, project framing docs, prompt drafts, extraction logs, source texts, sensitive `.env` caveat, and root clutter.
- `0b500fa` ingested the lit-review docs bundle slice, including 13-file aggregate hash, meta-schema, extraction methodology, n-ary relation guide, universal-applicator critique, archived strategies, TypeDB PDFs, and Carter visualization artifacts.
- `8c4adc7` ingested the data/examples grounding slice, including 17-file data hash, 11-file examples hash, source papers, test texts, Carter examples, hybrid schemas, OWL/causal stress tests, UI requirements, and model-form negative evidence.
- `521a2e2` ingested the evidence corpus inventory slice, including 114-file aggregate hash, six phase aggregates, Phase 1 purpose-classification summary, and links to existing Phase 2-6 pages.
- `d55fb34` ingested the debug_improved/analysis_results slice, including 9-file debug hash, Carter and Semantic Hypergraph phase counts, and empty analysis_results caveat.
- `843f17d` ingested the results corpus slice, including top-level results hash, semantic_hypergraph and young1996 subtree hashes, WorldView/Carter application assessments, cognitive-mapping critiques, and simplified meta-schema findings.
- `75a160a` ingested the experimental_testing corpus slice, including 41-file aggregate hash, top-level validation/optimization summaries, architecture comparison/prompt variation hashes, and empty retesting link.
- `32a5da2` ingested the multi_agent_system corpus slice, including 648-file aggregate hash, outer lit-review harness inventory, nested V5.2 evidence package inventory, template/test inventory, and evaluation/remediation caveats.
- `8a3367e` ingested the lit-review local repo metadata slice, including nested `sb_ontologies` remote, trip-backup branch, commit-summary provenance, and local `.claude` timeout settings.
- `f576061` ingested the large-lineage ops scaffolding slice, including `.github`, `docker`, and `requirements` inventories plus deployment/status caveats.
- `02bc97f` ingested the large-lineage config/contracts slice, including core config, orchestration/monitoring, phase interfaces, theory validator, and nine preserved tool contracts.
- `39a97d6` ingested the large-lineage config/schemas slice, including meta-schema versions, V11 theoretical-honesty shift, concrete theory examples, and tool-contract schema.
- `bb295c6` ingested the large-lineage legacy-tools duplicate slice, including byte-identical `tools/` and `config/legacy_tools/` inventories and cleanup caveat.
- `34be44a` ingested the large-lineage scripts corpus slice, including 127-file validation/verification/repair/demo/analysis/monitoring/testing inventory.
- `aa76124` ingested the large-lineage tests corpus slice, including 519-file category inventory, no-mocks testing principles, and integration-status caveats.
- `36aa579` ingested the large-lineage integration-tests slice, including 74-file main-system integration inventory, README/index/count mismatch, and unknown runtime-status caveat.
- `f23734b` ingested the large-lineage archived-root-tests slice, including 100-file historical root-level test inventory and runtime-proof caveats.
- `fc3a72b` ingested the large-lineage functional-tests slice, including 46-file user-workflow inventory, UI/MCP/phase/no-mocks themes, and runtime caveats.
- `0b76794` ingested the large-lineage reliability-tests slice, including 33-file reliability inventory, README status boundaries, real-database test scope, and certification caveats.
- `ab33997` ingested the large-lineage performance-tests slice, including 16-file benchmark-definition inventory and benchmark-output caveat.
- `f691716` ingested the large-lineage error-scenarios-tests slice, including 8-file failure-mode inventory and runtime-proof caveats.
- `1e7e37f` ingested the large-lineage security-tests slice, including 5-file security inventory, hardcoded-credential negative evidence, and public-export caveat.
- `6120db2` synthesized the test-evidence-layer concept, including claim-level distinctions across test definitions, status metadata, historical generated reports, and runtime proof.
- `c10ca8b` ingested the generated-reports slice, including 82-file report inventory, reliability supersession chronology, coverage caveat, and integration evidence report summary.
- `4e557d1` ingested the current-evidence-archive slice, including 46-file evidence/current inventory, timestamped verification failures/successes, corrected speedup claims, and failed full-integration caveat.
- `96d7814` ingested the evidence-reports-2025-08 slice, including 13-file August evidence-report inventory, DAG/traceability demonstrations, coverage caveats, Carter analysis traceability, and repeated relationship-extraction bottleneck evidence.
- `d8eed87` ingested the old-backups-results slice, including 9-file output-artifact inventory, provenance traces, Phase D success report, 0/6 validation report, failed end-to-end workflow, production-readiness caveat, and 22.2% interface audit.
- `e8b4778` ingested the old-backups-current-coverage slice, including 215-file/21 MB coverage HTML report, 206 indexed `src/core` files, 27,115 statements, and 2.99% total coverage.
- `3a025f6` ingested the old-backups-validation-reports slice, including two skeptical Phase 2/Phase 3 validation reports about simulated processing, placeholders, shallow tests, and overclaiming.
- `778cd27` documented the empty old-backups benchmark-results slice, showing that `benchmark_results/` exists but contains zero files.
- `5c790b5` ingested the old-backups-error-reports slice, including 22 generated JSON escalation reports, validation load-test dominance, six recovery strategies, and 0/55 successful recoveries.
- `b99dd9d` documented the empty old-backups-monitoring-output slice, showing that `monitoring_output/` exists but contains zero files.
- `e767a10` ingested the old-backups-output-reports slice, including Phase A 6/6 pass report, Phase B 20/20 pass report, and tool-registry status showing 12/123 tools implemented.
- `1a40b6d` ingested the old-backups-logs slice, including 44,622-line `super_digimon.log`, Neo4j/API failure patterns, successful pipeline traces, and raw credential-leak caveat.
- `b14f762` ingested the old-backups-theory-output slice, including Prospect Theory JSON/report/CSV generated-output artifact and input-to-parameter drift caveat.
- `71b3c42` ingested the old-backups-architecture-overview slice, including eight architecture/ADR overview files and architecture-intent caveat.
- `b0a676a` ingested the old-backups-backup-archives slice, including three tarballs, backup history metadata, duplicate ID/path caveat, and unencrypted `.env` caveat.
- `65665ee` ingested the old-backups stress-test root/framework slices, including 30-file subtree inventory, integration-score progression, cross-modal preservation failure/patch reports, support schemas/theory/tool-capability files, and mock-mode caveats.
- `5a28850` ingested the old-backups Carter analysis output slice, including generated themes/rhetoric/insights and empty evidence-sidecar caveat.
- `fb682d2` expanded the archived UI corpus record, including `archive/ui`, near-duplicate `ui_archive_2025_08`, archived implementations, logs/uploads/exports, and readiness-vs-backend caveats.
- `7e85506` ingested the uncertainty-stress-test root slice, including IC-inspired tests, CERQual/Bayesian/LLM services, Davis validation, SocialM MaZE artifacts, 75% readiness warning, and bias caveats.
- `82f9d71` ingested the uncertainty-stress-test analysis slice, including Davis rapid analysis, 157 prepared chunks, six agent notes, synthesis findings, and 3.8% coverage caveat.
- `4ea7913` ingested the uncertainty-stress-test core-services slice, including six Python implementation files, Bayesian/CERQual/LLM-native service roles, direct GPT-4 API patterns, fallback/default behavior, and optimized-engine runtime caveats.
- `1942fde` ingested the uncertainty-stress-test validation slice, including preserved connectivity, formal Bayesian, LLM-native comparison, SocialMaze mock-mode outputs, validator scripts, missing-output caveats, and hardcoded-path caveats.
- `3435403` ingested the uncertainty-stress-test testing slice, including nine-file IC-inspired harness, root 5/5 stress-test summary linkage, methodology walkthrough, synthetic-benchmark boundary, truncated-output caveat, and hardcoded-path rerun caveats.
- `e2d58af` ingested the uncertainty-stress-test Bayesian slice, including three psychological-trait Bayesian prototype scripts, simulated LLM likelihood-ratio methods, research-prior update pattern, and "real"/"production" label caveats.
- `ab9160f` ingested the uncertainty-stress-test docs slice, including methodology/specification files, formula/design content, unchecked implementation checklist, performance-target boundary, and uncorroborated validation-claim caveats.
- `2808973` ingested the uncertainty-stress-test setup slice, including Neo4j Docker manager, one-click setup helpers, complete demo script, demo password, optional Docker volume deletion path, fragile sibling-validator import, and missing demo-output caveats.
- `d615339` ingested the uncertainty-stress-test optimization slice, including mock parallelism, preserved estimated speedup JSON, cache assumptions, hardcoded historical paths, and no-literal-key finding.
- `0ae4fde` ingested the uncertainty-stress-test organization slice, including useful/non-useful symlink classification, 48/53 broken symlink caveat, and follow-on pointer to archived personality-prediction material.
- `fc4b9d2` ingested the archived-uncertainty-tests overview, including duplicate reorganized copy, 2025-07 experiments datasets/config/results, incomplete personality-prediction checkout, cleanup/deletion recommendations, and privacy/security caveats.
- `b0f9fc4` ingested the archived uncertainty experiments docs/validation slice, including 13-doc/17-validation inventory, production-ready versus fix-bias-first status tension, Kunst claims, LLM-native 7/7 result, mock SocialMaze caveat, and rerun caveats.
- `a5a2029` ingested the archived uncertainty datasets slice, including per-file hashes, dataset structure, tweet/object counts, identifier/privacy risk, count inconsistencies, empty-key config, and July 27 result summary.
- `cab7b79` ingested the archived uncertainty experiments code-delta slice, including subset/difference analysis and the two unique experiment tests: direct spaCy NER diagnostic and mock-capable SocialMaze harness.
- `aff3471` added the large-lineage archive coverage audit, including represented/not-yet-represented top-level archive directories and recommended next slices.
- `6751ca4` ingested the temp-analysis-2025-08 archive provenance slice, including file-list counts, uncommitted-file inventory, history examples, helper/debug script roles, and next-slice pointer to `ARCHIVE_BEFORE_CLEANUP_20250805/`.
- `0897fd4` ingested the archive-before-cleanup-2025-08-05 overview slice, including roadmap status claims, cleanup caveats, directory counts, theory-integration status, and sub-slice queue.
- `f571924` ingested the proposal-rewrite-2025-08-12 thesis lineage slice, including proposal positioning, critique incorporation, IC/uncertainty planning tension, validation matrices, HSPC/reference inventory, and concept links.
- `13c32a1` ingested the archive-before-cleanup phases slice, including 75-file phase inventory, Phase C completion caveats, reliability freeze, TDD migration, fail-fast, theory-to-code, universal LLM, and placeholder-key caveat.
- `2541db4` ingested the archive-before-cleanup initiatives slice, including theory extraction integration, uncertainty, two-stage analysis, bi-store/PostgreSQL, concurrency, dynamic orchestration, import failures, identity variants, risk, metrics, and no-literal-key finding.
- `e4acc23` ingested the archive-before-cleanup archived-investigations slice, including service-investigation cleanup inventory, duplicate/consolidation categories, template-only markers, and status-tension caveats.
- `727fe47` ingested the archive-before-cleanup analysis slice, including codebase/ops/performance review, KGAS theory/data architecture, reproducibility design, Claude Code orchestration, claim caveats, and no-literal-key finding.
- `1cff5b8` ingested the archive-before-cleanup residual-planning slice, including Phase C/entity-resolution caveats, observation-focused performance tracking, archived benchmark requirements, post-MVP agent-intelligence plans, and no-literal-key finding.
- `4fcc772` ingested the archive-before-cleanup root-files slice, including archive rationale, roadmap status claims, two-layer theory implementation/integration status, full-system DAG example, cleanup-improvement audit tension, and no-literal-key finding.
- `e45e400` updated the archive coverage queue, marking `ARCHIVE_BEFORE_CLEANUP_20250805/` sufficiently represented at top level and recommending `theoretical_exploration/` next.
- `1f4ab20` ingested the theoretical-exploration overview slice, including 107-file inventory, top-subtree hashes, archive/KISS boundary, sub-slice queue, and no-literal-key finding.
- `efbd845` ingested the theoretical-exploration proposal-materials slice, including dissertation framing, academic guidance, validation critiques/matrix, HSPC pointer, worked examples, safety framework, deprecated concepts, and no-literal-key finding.
- `87ba2b5` ingested the theoretical-exploration full-example architecture slice, including dynamic tool generation, uncertainty decisions, DAG execution/assessment, schema/source artifacts, and no-literal-key finding.
- `89a3604` ingested the theoretical-exploration thinking-out-loud slice, including analysis-philosophy, two-stage-analysis, cross-modal orchestration, six-level automation, implementation-claim caveats, and no-literal-key finding.
- `27f2bba` ingested the theoretical-exploration proposal-evolution slice, including fragments, assessment documentation, historical versions, critique-response mapping, proof-of-concept tone edits, and no-literal-key finding.
- `fc73c4f` ingested the theoretical-exploration schema-v14 post-MVP slice, including operationalization clarity, parameter uncertainty, method selection, multidimensional uncertainty, IC-at-execution, DAG-aware propagation, and no-literal-key finding.
- `2101506` ingested the gemini-review-tool archive overview, including 166-file inventory, review-tool purpose, workflow fixes, validation findings, generated docs-review caveats, roadmap critique config, and no-literal-key finding.
- `1e4eb0f` ingested the docs-architecture-cleanup-2025-08-29 overview, including generated-document cleanup rationale, over-engineered service-guide archival, IC uncertainty ADR-to-abandonment arc, category-error critique, and no-literal-key finding.
- `3687a70` ingested the generated-outputs-2025-08 slice, including performance/SLA JSON, real-vector proof, provenance/reasoning-trace SQLite schema and row counts, repomix-bundle caveats, and no-literal-key finding.
- `61b5aaf` ingested the analysis-validation-2025-08 slice, including validation archive inventory, development-standards validation, three Gemini claim validations, reliability/MCP/final validation configs, chronology/supersession caveat, and no-literal-key finding.
- `cc5ae4e` ingested the agent-stress-testing slice, including dual-agent framework goals, working-system claims, T15A/T23A proof demo, adaptive-agent demo caveats, trace artifact inventory, and no-literal-key finding.
- `6005315` ingested the root-cleanup-2025-08-29 slice, including duplicate app entry-point cleanup rationale, tool-composition checkpoint, archived apps/Twitter explorer/k8s material, cross-modal test output, placeholder-secret caveat, and no-literal-key finding.
- `4f4717d` ingested the old-CLAUDE-md-versions slice, including contract-first migration guidance, MVP Day 1 guide, Phase 2.1 completion tasks, evidence-first repair policy, current-policy caveat, and no-literal-key finding.
- `ddd5d9f` ingested the old-docs-2025-08 slice, including contract-first interface split, service API drift, structured-output migration, documentation/status separation, operations-label caveat, IC uncertainty notes, and placeholder-key finding.
- `c2c768f` ingested the proposal-rewrite-condensed slice, including SCT full-example DAGs, StructGPT-inspired schema discovery/data interfaces, graph fusion, dynamic tool generation, uncertainty-framework notes, pure-LLM uncertainty critique, and no-literal-key finding.
- `5e912a9` ingested the scripts-archive-2025-08 slice, including debug/demo/fix/old-analysis/test inventory, repair-script policy lineage, DAG/cross-modal/LLM/Neo4j/MCP harnesses, relationship-debugging notes, hardcoded-path caveats, and no-literal-key finding.
- `f83a5a8` ingested the archived-experimental-tests slice, including redundant-functional/stress/root-test inventory, MCP/PDF workflow tests, TORC/adversarial stress methodology, API/contract/configuration checks, PageRank/vertical-slice caveats, empty-directory notes, and no-literal-key finding.
- `895754c` ingested the demos-examples-2025-08 slice, including ontology Streamlit UI, Neo4j/SQLite transaction demo, reliability demonstration, direct Gemini validators, natural-language-to-DAG proof, sample documents, multimodal output JSON, and no-literal-key finding.
- `0f4fa2f` ingested the archived-implementations-ui slice, including four Streamlit/PDF-upload UI remnants, vertical-slice UI, standardized phase-interface UI, simple debug UI, smoke test, and no-literal-key finding.
- `5e70807` ingested the doc-generation-scripts slice, including architecture concatenation, architecture split, ADR extraction/indexing, Gemini architecture review script, generated-doc provenance caveats, and no-literal-key finding.
- `9ad6de6` ingested the temp-debug-files slice, including database/debug/structured-output/claim-validation/reliability inventory, facade POC transcript fragment, overlap caveats, and no-literal-key finding.
- `4430129` added the KGAS evolution checkpoint synthesis, including top-level archive coverage completion, GraphRAG-to-theory-system arc, evidence/status caveats, relationship-extraction and uncertainty risks, and recommended deep dives.
- `1401d56` repaired the current cross-modal API import/calling-contract boundary, added focused current-runtime tests, verified passing `src.api.cross_modal_api` import, and recorded remaining `neo4j`/`torchvision` dependency blockers.
- `f2704ad` completed the isolated KGAS environment slice, including project-local `.venv` requirements install, added `python-multipart`/`fastmcp`/`psutil`, generated empty-default `config/default.yaml`, passing isolated target imports, and non-blocking SymPy warning.
- `bec53f2` completed SymPy dependency support, including `sympy>=1.14.0`, isolated MCP import with `mcp_is_none=False`, HybridFormulaParser SymPy support, and successful MCP tool registration.

## Next

1. Next recommended step: trace the relationship-extraction bottleneck across T27/current code/archive evidence.
3. Security follow-up: treat API keys visible in preserved logs as compromised before any public sharing or archive export.
4. Security follow-up: treat the preserved `.env` credentials in lit-review and backup tarballs as compromised before any public sharing or archive export.
