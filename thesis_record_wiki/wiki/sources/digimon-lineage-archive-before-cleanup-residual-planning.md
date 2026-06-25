---
type: SourceSummary
title: Digimon Lineage Archive Before Cleanup Residual Planning
description: Residual issues, performance, and post-MVP planning slice from the August 2025 pre-cleanup archive, including entity-resolution limitations, observation-focused performance tracking, archived rigid benchmarks, and future agent-intelligence enhancements.
tags: [source, digimon-lineage, archive, planning, performance, post-mvp, entity-resolution]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ARCHIVE_BEFORE_CLEANUP_20250805/issues/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ARCHIVE_BEFORE_CLEANUP_20250805/performance/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ARCHIVE_BEFORE_CLEANUP_20250805/post-mvp/
confidence: high
---

# Summary

This page groups the small residual planning directories from `ARCHIVE_BEFORE_CLEANUP_20250805/`: `issues/`, `performance/`, and `post-mvp/`. Together they preserve 11 markdown files totaling 61,031 bytes by file inventory. [1][2][3]

The bundle records three linked decisions: accept Phase C as functionally complete despite entity-resolution and benchmark gaps, track current performance observationally instead of with rigid requirements, and move advanced agent-intelligence/performance frameworks to post-MVP. [4][5][6]

# File Inventory

| Directory | Files | Aggregate Hash | Role |
| --- | ---: | --- | --- |
| `issues/` | 2 | `50497ccb1f958ffd2e3a79a8091c6b59cf951e759b1f760f2e28d30295158b26` | Phase C performance and entity-resolution issue notes. [1] |
| `performance/` | 2 | `f50cafc90393efe2d7df18c9610cc2d3d293ddf0cbfde4ce8f52e2bec90c0b43` | Observation-focused performance tracking. [2] |
| `post-mvp/` | 7 | `9a5114ae7bda1ec2f8ab853539b8465d988bd34aea80ae062ada1648095283d8` | Deferred agent-intelligence and rigid performance-requirement material. [3] |

# Entity Resolution And Phase C

The entity-resolution issue says coreference resolution was achieving only 24% F1 against a target above 60%. It attributes the result to regex/string-similarity methods, non-integrated SpaCy NER, test ground truth that expected pronouns and non-entities, and a mismatch between regex tests and true entity-resolution evaluation. [4]

The paired Phase C note says Phase C was functionally complete with known performance limitations: 76 of 81 tests passing, memory usage optimized, and entity resolution/disambiguation/performance benchmarks deferred to Phase D or later. [5]

The key thesis-record point is not that entity resolution was solved. It is that the project explicitly accepted a vertical-slice boundary while preserving the need for LLM-based or stronger NLP-based entity resolution and redesigned ground truth for later work. [4][5]

# Performance Philosophy Shift

The active performance directory is intentionally lightweight. Its README says performance tracking should observe system behavior, identify patterns, and avoid rigid targets, demanding benchmarks, pass/fail criteria, or mandatory optimization schedules. [6]

The performance journal records qualitative observations: document processing, entity operations, graph building, and queries were considered adequate for research workflows as of 2025-08-05, with the caveat that performance varies by document type and complexity. [7]

This is a policy/design signal: the archive separates performance understanding from performance pressure.

# Post-MVP Deferrals

The post-MVP README says these initiatives are future-focused, enhancement-oriented, research-directed, and performance-focused, after core functionality is stable. It lists adaptive workflow replanning, relationship identity resolution, and archived performance requirements. [8]

The agent-intelligence directory proposes two enhancements:

- Adaptive workflow replanning: WorkflowAgent should adapt plans after intermediate execution results instead of executing one static workflow end-to-end. [9]
- Relationship identity resolution: T34 Edge Builder should deduplicate semantically equivalent relationship edges, analogous to T31 entity identity resolution. [10]

These documents are design intent, not implementation evidence.

# Archived Rigid Benchmarks

The post-MVP performance-requirements README explains why benchmark material was moved out of the active performance area: it contained hard targets, pass/fail indicators, demanding schedules, and prescriptive mandates that conflicted with the preferred observation-based approach. [11]

The preserved benchmark-results file still contains useful measurement claims and techniques, including a table saying targets were met or exceeded and a conclusion claiming 2.5-4.8x speedups and 100K-entity single-node capacity. Because the same directory README says this requirements style was archived, those benchmark claims should be treated as historical target/measurement material, not current obligations or verified current status. [11][12]

# Credential Scan

A targeted scan of these residual planning directories found no literal OpenAI or Google API keys. [1][2][3]

# Interpretation

This slice is valuable because it preserves project-management judgment: keep the MVP focused, document known limitations, do not let performance benchmarks create premature pressure, and defer advanced agent intelligence until core workflows are stable.

For future cleanup or research reconstruction, this page should be read with the phase and initiative pages: Phase C completion claims depend on the entity-resolution caveat, and post-MVP agent plans depend on stable tool contracts, error handling, and performance feedback systems. [5][9]

# Relationship To Wiki

- [Digimon Lineage Archive Before Cleanup 2025 08 05 Overview](digimon-lineage-archive-before-cleanup-2025-08-05-overview.md): parent archive overview.
- [Digimon Lineage Archive Before Cleanup Phases](digimon-lineage-archive-before-cleanup-phases.md): related Phase C and Phase D chronology.
- [Digimon Lineage Archive Before Cleanup Initiatives](digimon-lineage-archive-before-cleanup-initiatives.md): related performance, orchestration, and identity plans.
- [Relationship Extraction Bottleneck](../concepts/relationship-extraction-bottleneck.md): related entity/relationship quality bottleneck.
- [Adaptive Operator Routing](../concepts/adaptive-operator-routing.md): related adaptive planning/routing lineage.
- [Evidence Claim Discipline](../concepts/evidence-claim-discipline.md): relevant for benchmark and completion claims.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ARCHIVE_BEFORE_CLEANUP_20250805/issues/`

[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ARCHIVE_BEFORE_CLEANUP_20250805/performance/`

[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ARCHIVE_BEFORE_CLEANUP_20250805/post-mvp/`

[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ARCHIVE_BEFORE_CLEANUP_20250805/issues/entity-resolution-performance.md`

[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ARCHIVE_BEFORE_CLEANUP_20250805/issues/phase-c-performance-optimizations.md`

[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ARCHIVE_BEFORE_CLEANUP_20250805/performance/README.md`

[7] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ARCHIVE_BEFORE_CLEANUP_20250805/performance/performance-journal.md`

[8] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ARCHIVE_BEFORE_CLEANUP_20250805/post-mvp/README.md`

[9] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ARCHIVE_BEFORE_CLEANUP_20250805/post-mvp/agent-intelligence/adaptive_workflow_replanning_enhancement.md`

[10] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ARCHIVE_BEFORE_CLEANUP_20250805/post-mvp/agent-intelligence/relationship_identity_resolution_enhancement.md`

[11] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ARCHIVE_BEFORE_CLEANUP_20250805/post-mvp/performance-requirements/README.md`

[12] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ARCHIVE_BEFORE_CLEANUP_20250805/post-mvp/performance-requirements/benchmark-results.md`
