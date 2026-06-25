---
type: SourceSummary
title: Digimon Lineage Uncertainty Stress Test Core Services
description: Core Python service implementations for the archived uncertainty stress test, including Bayesian aggregation, CERQual assessment, LLM-native uncertainty scoring, formal Bayesian updating, and an optimized cached/parallel variant with runtime caveats.
tags: [source, digimon-lineage, uncertainty, bayesian, cerqual, llm, implementation]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/core_services/
confidence: high
---

# Summary

This slice covers `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/core_services/`. The directory contains six Python files, 3,550 total lines, 135,296 bytes, and aggregate hash `aggregate-sha256:a0551ce28172fd5c1d877ff573ac05a7d1e5b2b47fe2e56a4ca5910f3dbe5508`. [1]

Historically, this is the clearest implementation-side evidence for the archived uncertainty stress test. It preserves multiple attempts to turn uncertainty from an architecture document into executable services: a basic uncertainty orchestrator, Bayesian evidence aggregation, formal Bayesian updating with LLM-estimated parameters, CERQual qualitative-evidence assessment, an LLM-native contextual-confidence engine, and an optimized cached/parallel variant. [1]

# Inventory

| File | Lines | Role |
| --- | ---: | --- |
| `bayesian_aggregation_service.py` | 462 | `Evidence` data class and `BayesianAggregationService` for LLM-assisted evidence quality, likelihood, weight, batch aggregation, and reports. [2] |
| `cerqual_assessor.py` | 813 | `StudyMetadata`, `CERQualEvidence`, `CERQualAssessment`, and `CERQualAssessor` for methodological limitations, relevance, coherence, adequacy, overall reasoning, and reports. [3] |
| `formal_bayesian_llm_engine.py` | 627 | `BayesianParameters`, `BayesianUpdate`, and `FormalBayesianLLMEngine` for LLM parameter estimation plus explicit Bayes factor/posterior math. [4] |
| `llm_native_uncertainty_engine.py` | 714 | `ContextualConfidenceScore` and `LLMNativeUncertaintyEngine` for LLM-determined epistemic priors, contextual evidence assessment, confidence synthesis, updates, metrics, and reports. [5] |
| `optimized_llm_native_engine.py` | 280 | `CacheEntry` and `OptimizedLLMNativeEngine` for cached LLM calls and parallel prior/evidence assessment. [6] |
| `uncertainty_engine.py` | 654 | `ConfidenceScore` and `UncertaintyEngine`, an orchestrator for claim/evidence extraction, initial confidence, confidence updates, and cross-modal uncertainty translation. [7] |

# Implementation Pattern

All five direct service engines use `OPENAI_API_KEY` or a passed `api_key`, construct a bearer `Authorization` header, and call `https://api.openai.com/v1/chat/completions` through `aiohttp`. The hardcoded model is `gpt-4`; temperatures are usually `0.1`, with the LLM-native engine using `0.0` plus `seed: 42` for deterministic intent. [2][3][4][5][7]

The services generally ask the LLM for JSON, then extract a JSON object by locating the first `{` and last `}` before `json.loads`. This is historically important because it shows the project moving toward structured uncertainty outputs before the later ecosystem rule requiring schema-constrained structured output. [2][3][4][5][7]

The formal Bayesian engine is the most explicit math implementation: it separates LLM-determined prior/likelihood/evidence parameters from deterministic posterior calculations, Bayes factors, odds, belief-change classification, and confidence bounds. That matches the thesis direction of using LLMs for judgment while keeping aggregation math inspectable. [4]

The CERQual service preserves the qualitative-social-science thread: methodological limitations, relevance, coherence, and adequacy are assessed separately and combined with adjustable dimension weights. This connects the superseded ADR-007 CERQual design to a concrete service implementation. [3]

# Caveats

This page records source-code evidence, not a rerun. The core services may have been runnable in their original environment, but the wiki has not executed their test functions or verified live API behavior. [1]

The code contains multiple fallback/default paths after API, parsing, or assessment failures. For example, Bayesian aggregation returns default quality and likelihood dictionaries on parsing or exception failures; the LLM-native engine returns default prior/evidence/synthesis assessments; CERQual returns default dimension assessments; and the underlying API wrappers return an empty string on non-200 responses or exceptions. [2][3][5][7]

The optimized engine should be treated as an experimental optimization artifact, not proven working code. It imports `ConfidenceScore` from `llm_native_uncertainty_engine.py`, but that file defines `ContextualConfidenceScore`, not `ConfidenceScore`; it also calls helper methods such as `_parse_prior_response`, `_parse_evidence_response`, `_parse_synthesis_response`, and `_create_confidence_score`, and those definitions were not found in the six-file core-services directory. [5][6]

No literal OpenAI or Google API keys were found in this `core_services/` folder during a targeted credential-pattern scan. The files do construct bearer headers from environment-provided keys, so raw logs and outputs elsewhere still need the separate credential-leak caution already recorded for other archive slices. [1]

# Relationship To Wiki

- [Digimon Lineage Uncertainty Stress Test Root](/wiki/sources/digimon-lineage-uncertainty-stress-test-root.md): root documentation claims these services were implemented and integrated; this page identifies the actual code slice behind that claim.
- [Digimon Lineage Uncertainty Stress Test Analysis](/wiki/sources/digimon-lineage-uncertainty-stress-test-analysis.md): Davis analysis provides methodological grounding; these services are the implementation counterpart.
- [Uncertainty Framework Evolution](/wiki/concepts/uncertainty-framework-evolution.md): the code shows the intermediate move from CERQual/Bayesian architecture toward LLM-native contextual uncertainty.
- [Uncertainty Traceability Architecture](/wiki/concepts/uncertainty-traceability-architecture.md): the services implement reasoning/report structures but still expose traceability risks through JSON substring parsing and silent default-return paths.
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md): source presence and implementation structure support "code exists" claims, not "currently verified runtime" claims.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/core_services/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/core_services/bayesian_aggregation_service.py`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/core_services/cerqual_assessor.py`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/core_services/formal_bayesian_llm_engine.py`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/core_services/llm_native_uncertainty_engine.py`  
[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/core_services/optimized_llm_native_engine.py`  
[7] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/uncertainty_stress_test/core_services/uncertainty_engine.py`
