---
type: LineageVariant
title: Digimon Lineage Digimons
description: Largest preserved lineage bundle and likely central source for later Digimons evolution.
tags: [variant, digimons, lineage, large-archive]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/metadata/recovery_inventory.tsv
confidence: medium
---

# Summary

`digimon_lineage_Digimons` is preserved at `../archive_full_record/lineage_variants/digimon_lineage_Digimons`. The recovery manifest identifies it as the largest preserved lineage bundle. The inventory records 9,334,155,151 readable bytes, 78,514 files, 9,619 directories, 157 symlinks, and git head `e14164e818588b474b68eb0e525ea78d9a10ce1c`. [1]

Focused root ingest shows this bundle preserves a September 2025 active KGAS state with documentation truthfulness, roadmap consolidation, and vertical-slice/main-system reconciliation work. The first slice should be read through [Digimon Lineage Active State](/wiki/sources/digimon-lineage-active-state.md), [Reality Verification Arc](/wiki/concepts/reality-verification-arc.md), and [Vertical Slice vs Main System](/wiki/concepts/vertical-slice-vs-main-system.md).

The `tool_compatability` slice shows the same type-based composition decision seen in `Digimons_clean_for_real`, but with active local context: the subtree is explicitly marked as the adapter/framework direction and the nested `poc/vertical_slice` path is marked as the active proof-of-concept framework. See [Digimon Lineage Tool Compatibility](/wiki/sources/digimon-lineage-tool-compatibility.md).

The architecture-docs slice confirms that `docs/architecture/` is target design, not status tracking, and adds a critical internal review of uncertainty/provenance over-engineering. See [Digimon Lineage Architecture Docs](/wiki/sources/digimon-lineage-architecture-docs.md) and [Uncertainty Traceability Architecture](/wiki/concepts/uncertainty-traceability-architecture.md).

The evidence-archive slice shows that some success claims were later archived as false because they relied on component-level tests rather than system integration validation. See [Digimon Lineage Evidence Archives](/wiki/sources/digimon-lineage-evidence-archives.md) and [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md).

The `experiments/lit_review` slice is a separate 51M experiment subsystem with 2,768 files and its own `.git` directory. It preserves automated theory extraction/application work, schema-based ontology docs, validation summaries, and phase completion evidence. See [Lit Review Theory Extraction Experiment](/wiki/sources/lit-review-theory-extraction-experiment.md) and [Automated Theory Extraction](/wiki/concepts/automated-theory-extraction.md).

The `experiments/lit_review/carter_analysis_output` slice is a concrete generated-output sub-slice: six files applying Young 1996 cognitive mapping and Chong & Druckman 2007 framing theory to a Carter speech, then integrating the results. See [Carter Theory Analysis Output](/wiki/sources/carter-theory-analysis-output.md).

The `experiments/lit_review/src/schema_creation` slice documents production-path evolution behind those outputs: prompt externalization, improved multiphase extraction, information-loss correction, v13 single/multi-theory extractors, and caveats around hardcoded historical paths. See [Lit Review Schema Creation Production Path](/wiki/sources/lit-review-schema-creation-production-path.md).

The `experiments/lit_review/validation_results` slice records validation reports and outputs. The conservative finding is a complexity/accuracy pattern rather than blanket success: Lofland-Stark sequence worked especially well, framing effects exposed a model-type error, and Young 1996 worked as a property graph case. See [Lit Review Validation Results](/wiki/sources/lit-review-validation-results.md).

The `experiments/lit_review/evidence/phase2_vocabulary_extraction` and `phase3_schema_generation` slice records balance-driven evidence. Phase 3 has consistent 8/8 passing test artifacts; Phase 2 preserves a contradiction between 100% remediation claims and a failing stored test result. See [Lit Review Phase 2-3 Evidence](/wiki/sources/lit-review-phase2-3-evidence.md).

The `experiments/lit_review/evidence/phase4_integration_pipeline` slice records integrated-pipeline evidence. It should be read time-indexed because benchmark/balance reports preserve certification weaknesses while remediation/completion summaries claim later 100/100 resolution. See [Lit Review Phase 4 Integration Pipeline](/wiki/sources/lit-review-phase4-integration-pipeline.md).

The `experiments/lit_review/evidence/phase5_reasoning_engine` slice records cross-purpose reasoning over the balanced pipeline. It reports perfect balance and 0.819 integration quality, but the evidence remains demo-scale. See [Lit Review Phase 5 Reasoning Engine](/wiki/sources/lit-review-phase5-reasoning-engine.md).

# Shallow Contents

Shallow inspection shows archive, archived, config, contracts, data, dev, docker, docs, evidence, examples, experiments, investigation, logs, requirements, research, scripts, src, tests, tool compatibility material, tools, and recovered UI components.

# Interpretation

This is likely the highest-value lineage source after the current repo. Continue ingesting in slices: root README/CLAUDE/roadmap, tool compatibility, architecture docs, evidence archives, experiments, ADRs, Carter analysis outputs, and recovered UI.

# Current Git Caveat

At focused ingest time, the preserved git repo reported `master...kgas/master [ahead 1]` with a modified nested `experiments/tool_compatability/GraphRAG` path. The recovery inventory's git head is therefore only one preserved reference point; later local working-tree state should be investigated before making exact commit-history claims. [2]

# Citations

[1] `../archive_full_record/metadata/recovery_inventory.tsv`
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/.git` and `git status` output observed during 2026-06-25 ingest.
