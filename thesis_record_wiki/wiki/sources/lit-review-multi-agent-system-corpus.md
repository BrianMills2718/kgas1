---
type: Source
title: Lit Review Multi Agent System Corpus
description: Corpus-level inventory of the preserved multi_agent_system directory, including outer lit-review phase harness and nested V5.2 generator/evidence package.
tags: [source, lit-review, multi-agent, evidence, phases, corpus, v5-2]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/multi_agent_system/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/multi_agent_system/USAGE_GUIDE.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/multi_agent_system/docs/current_phase_status.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/multi_agent_system/multi_agent_system/MULTI_AGENT_SYSTEM_GUIDE.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/multi_agent_system/multi_agent_system/phases/evaluation_clean/
confidence: medium
---

# Summary

`experiments/lit_review/multi_agent_system/` is a 648-file, 7,850,625-byte preserved corpus with aggregate hash `b7c6f639e8829141ca0d126d2124f4fac8905963b511d0ddf18e8fad7b3694aa`. [1]

The existing [Lit Review Multi Agent System](/wiki/sources/lit-review-multi-agent-system.md) page summarizes the method and the six lit-review evaluation files. This page records the whole directory topology because the preserved folder contains two related but distinct layers:

- the outer lit-review multi-agent harness, including `USAGE_GUIDE.md`, `auto_phase_manager.py`, `multi_agent_toolkit.py`, `docs/`, `examples/`, and a six-phase `phases/` directory [1][2]
- a nested `multi_agent_system/` package that preserves a broader V5.2 system-generation workflow with evidence packages, isolated phase instructions, evaluation/reevaluation files, templates, and tests [1][3][4][5]

# Directory Inventory

| Path | Files | Bytes | Aggregate hash | Interpretation |
|---|---:|---:|---|---|
| `docs/` | 2 | 19,827 | `29b8f138aeb596bd6a3c916fb9d5b81edf833223e7864d8cee9943e7b1675951` | change tracking and current V5.2 phase status |
| `examples/` | 1 | 4,708 | `b0b522ce80a87dff8f2d4f3b19c11b2305c8b157c9de002ed8bee64701e2d0e9` | simple usage example |
| `multi_agent_system/` | 622 | 7,706,941 | `807c6a257ec14dd2b75fe6a01debcde61bd9449838ae4abdf790ecd484576bb4` | nested package/evidence corpus for V5.2 generator work |
| `phases/` | 18 | 90,345 | `3634b8a8baac480f6305b725e7774b974a714798899142ad1670740046008ae4` | outer six-phase lit-review isolated instructions and clean evaluation results |
| root files | 5 | 28,804 | individual hashes in manifest | guide, toolkit, phase manager, setup, package marker |

# Outer Harness

The outer harness is the part most directly tied to the lit-review theory-extraction experiment. It preserves:

- `USAGE_GUIDE.md`, a reusable guide for isolated implementation agents, external evaluation agents, and 100/100 phase gates [2]
- `auto_phase_manager.py` and `multi_agent_toolkit.py`, local orchestration utilities for running phases [1]
- `phases/evaluation_clean/phase1_evaluation_result.md` through `phase6_evaluation_result.md`, the six clean lit-review evaluations already summarized in [Lit Review Multi Agent System](/wiki/sources/lit-review-multi-agent-system.md) [1]

The outer `phases/` directory should not be confused with the nested package's `phases/` directory. The outer one maps to purpose classification, vocabulary extraction, schema generation, integration pipeline, reasoning engine, and production validation for the lit-review experiment.

# Nested V5.2 Package

The nested `multi_agent_system/multi_agent_system/` package is the dominant preserved corpus:

| Nested path | Files | Bytes | Aggregate hash | Role |
|---|---:|---:|---|---|
| `evidence/` | 486 | 6,428,079 | `debf061c972cf026002237fa61745024417b0534bd0ec8550570fcf30874e05f` | implementation evidence, tests, summaries, generated systems, remediation artifacts |
| `phases/` | 84 | 1,033,838 | `2a613a729549a46beb26189261aab2c3fb9bcac4d8bc4c104cc6d3534ceb4a74` | isolated phase instructions plus evaluation and reevaluation results |
| `template/` | 45 | 82,043 | `89253ff0720ce8ecb533adbda503ed008e9178c2f9648915b17baf4476127811` | reusable implementation/evaluation project template and two example projects |
| `tests/` | 6 | 145,294 | `6f49805511bbebbbed716ed994c8d0fa61282d7d6d7863791b5e97b12994fa64` | component, integration, schema-generator, and security tests |

The nested guide describes V5.2 as a multi-agent implementation system: implementation agents receive only phase instructions; evaluation agents receive only evidence and criteria; raw execution must prove functionality; failures trigger remediation. [4]

# Nested Evidence Topology

The nested evidence corpus contains phase-specific implementation packages:

| Evidence folder | Files | Bytes | Aggregate hash | Notes |
|---|---:|---:|---|---|
| `documentation_maintenance` | 70 | 2,005,075 | `e3d0bc4d40617429a2355fff6377cf1e4ea3edf1363489bfe9731b9375afafdf` | archive organization, consistency, documentation validation, phase progress, repo snapshots |
| `phase1_failure_hiding_removal` | 44 | 171,821 | `79b0d3aa1af6f4a4ef639a208b22d9da8e18d1a8f74597d1c5af2575eaa7a952` | failure-hiding removal evidence |
| `phase2_component_library` | 23 | 301,853 | `e5a241e3fec068bf1c5f3ba0e90065f060c3ada8a52df42a7b915f40bd13c16f` | component registry, schema framework, lifecycle, security validation |
| `phase2_validation_driven_orchestrator` | 6 | 32,594 | `3b16791c16873715e963a53fc92287a7dd8843adffe38df196a7f9517f72e4a4` | validation-driven orchestrator evidence |
| `phase3_blueprint_schema` | 11 | 231,648 | `1e31596fe5828278fd50586ac9176e9f8ed0d99994dc331bf89838b30bbf95d1` | blueprint schema parser and validation code |
| `phase3_blueprint_schema_v5` | 6 | 22,573 | `d5a22a4e0f6bb33f2f8debe12c802a4a37082a9189cad8958073442611d6c968` | V5 blueprint schema evidence package |
| `phase3_enhanced_generation` | 24 | 92,676 | `c779195db939224025fb3db4dc05af211d680c8177037924fd46aba1425d8ab8` | enhanced generation plan, tests, and external evaluation report |
| `phase4_enhanced_component_generation` | 6 | 35,215 | `f94752215f29db38884251806d06930d2c975e8431b72a440c3f4ffcb21a6cc0` | enhanced component-generation evidence |
| `phase4_validation_orchestrator` | 25 | 561,555 | `d3da8bc4ad3eb5688ff41095447c716c9b331f36d8db94e65a8570a95bb94c55` | validation orchestrator and multi-level integration code |
| `phase5_database_integration` | 42 | 810,520 | `51b4de143eee160fd91d75cb7a6532cb3efce161de1556424bca7064fd91316f` | database connection/store/schema-management evidence |
| `phase5_database_integration_mainline` | 55 | 894,447 | `aaf9281e1c7891100c779fecbb9ff51a01eaae09ff972d007808370d57b66d42` | mainline database integration and critical integration tests |
| `phase6_end_to_end_tests` | 9 | 67,243 | `808cf21bfe21125885cd1c16efe2d69af82048225b4e21636bc02f461cad2d88` | end-to-end functional evidence package |
| `phase6_harness` | 39 | 471,379 | `81fe430264073eb01e7290432a64b0a31e60aa2c76207356cf79a5c99effb28f` | SystemExecutionHarness implementation and performance tests |
| `phase7_generation` | 44 | 398,446 | `445ec67ba14e0f09e6267014c48555c26d172f33f3bb6a78ec895b177d5f44b3` | two-phase generation pipeline evidence |
| `phase8_failhard_compliance` | 10 | 66,604 | `dc9be94658ef20ac6484e05252e2d9bed15599355921dfdbb11a382f6d4efffb` | fail-hard compliance demo and generated systems |
| `phase8_production_ready_generation` | 72 | 264,430 | `306b883c7f329c12b0bb04644062ea50c614a12dd85b8f9c866b7f9c5c4df7f5` | production-ready generation templates, deployment fixes, validation report |

# Evaluation And Remediation Record

The nested `phases/evaluation_clean/` directory preserves both failures and later passes. This matters because it is evidence of iteration, not a single clean success story. [5]

Examples:

- Phase 2 has initial FAIL and reevaluation FAIL files before a final PASS. [5]
- Phase 4 has an initial FAIL and a later PASS after remediation, with the reevaluation still noting partial functionality/refinement needs. [5]
- Phase 8 production-ready generation records a conditional pass at 76/100, with unverified deployment/load-testing claims. [5]
- Phase 9 records an 88/100 PASS and later a 100/100 reevaluation; the first pass mentions fallback behavior, which should be read against the claimed fail-hard discipline. [5]
- Phase 10 has evaluation and reevaluation files, so the blueprint-builder phase should be treated as an iterative evidence thread rather than a single outcome. [5]
- Documentation maintenance also has evaluation and reevaluation files; the reevaluation claims 100% completion after remediation. [5]

# Interpretation

This corpus is important for the thesis record for two reasons.

First, it preserves a procedural quality-control pattern that Brian repeatedly wanted agents to follow: isolated instructions, external evidence evaluation, no partial credit, remediation, and phase-specific commits. That connects directly to [Multi Agent Evidence Harness](/wiki/concepts/multi-agent-evidence-harness.md) and [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md).

Second, it also records the danger of self-contained agent evaluation narratives. The status document claims V5.2 complete and functionally production ready while also saying performance optimization remains required. The evaluation-clean folder preserves failures, conditional passes, partial-functionality notes, and remediation cycles. Those details should be kept visible before using any "100%" or "production ready" line as thesis evidence. [3][5]

# Links

- [Lit Review Multi Agent System](/wiki/sources/lit-review-multi-agent-system.md)
- [Lit Review Testing Code](/wiki/sources/lit-review-testing-code.md)
- [Lit Review Evidence Corpus](/wiki/sources/lit-review-evidence-corpus.md)
- [Multi Agent Evidence Harness](/wiki/concepts/multi-agent-evidence-harness.md)
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md)
- [Academic Proof Of Concept Scope](/wiki/concepts/academic-proof-of-concept-scope.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/multi_agent_system/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/multi_agent_system/USAGE_GUIDE.md`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/multi_agent_system/docs/current_phase_status.md`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/multi_agent_system/multi_agent_system/MULTI_AGENT_SYSTEM_GUIDE.md`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/multi_agent_system/multi_agent_system/phases/evaluation_clean/`
