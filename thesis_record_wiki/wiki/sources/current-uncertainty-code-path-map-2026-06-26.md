---
type: SourceSummary
title: Current Uncertainty Code Path Map 2026 06 26
description: Source-level map of uncertainty and confidence code paths in the current cleaned checkout, distinguishing ADR-004/CERQual confidence support from archived ADR-029 IC framework evidence.
tags: [source, current-code, uncertainty, confidence, runtime-boundary]
created: 2026-06-26
updated: 2026-06-26
sources:
  - ../src/core/confidence_score.py
  - ../src/core/confidence_scoring/
  - ../src/core/tool_contract.py
  - ../src/core/tool_adapter.py
  - ../src/core/tool_adapter_layer2.py
  - ../src/analytics/complete_pipeline.py
  - ../src/analytics/graph_builder.py
  - ../src/analytics/graph_query_engine.py
  - ../src/tools/phase1/t68_pagerank_optimized.py
  - ../src/tools/phase2/t23c_ontology_aware_extractor_unified.py
confidence: high
---

# Summary

The current cleaned checkout contains a real confidence-scoring subsystem, but it is best described as ADR-004-style `ConfidenceScore` plus CERQual/range/temporal extensions. It is not a direct current implementation of the archived ADR-029 / Comprehensive7 IC uncertainty framework. [1][2][11]

The uncertainty-related current code paths are concentrated in:

- `src/core/confidence_score.py` as the public interface; [1]
- `src/core/confidence_scoring/` as decomposed data-model, combination, CERQual, range, temporal, calculator, and factory components; [2]
- tool contract/adapters that require or infer `ConfidenceScore`; [7][8][9]
- selected tools such as optimized PageRank and ontology-aware extraction that initialize or calculate confidence scores; [12][13]
- the current proven upload pipeline, which carries source references through graph build/query paths but does not visibly apply the full confidence-scoring subsystem in the source slices reviewed here. [10][14][15]

# Core Confidence Package

`src/core/confidence_score.py` says it implements the normative confidence scoring system according to ADR-004 and has been updated for uncertainty ranges and CERQual assessment. It wraps decomposed components and exposes convenience methods such as `combine_with`, `apply_temporal_decay`, `add_cerqual_evidence`, `set_confidence_range`, `combine_with_range_preservation`, and `create_with_cerqual`. [1]

`src/core/confidence_scoring/data_models.py` defines `PropagationMethod` with three methods: `bayesian_evidence_power`, `dempster_shafer`, and `min_max`. The `ConfidenceScore` model stores a point value, optional confidence range, evidence weight, CERQual dimensions, temporal metadata, missing-data/coverage flags, aggregate distribution fields, source, propagation method, and metadata. [3]

`src/core/confidence_scoring/cerqual_assessment.py` implements CERQual dimensions and weighted combination. It defaults missing CERQual dimensions to moderate/high values when partial data is supplied, then blends base confidence and CERQual score. [4]

`src/core/confidence_scoring/combination_methods.py` implements Bayesian evidence power, Dempster-Shafer, and min/max combiners. It also has fallback averaging behavior if combination fails. [5]

`src/core/confidence_scoring/temporal_range_methods.py` implements temporal decay, validity windows, range setting, range-preserving combination, and range-only confidence creation. [6]

`src/core/confidence_scoring/factory_methods.py` provides high/medium/low presets, temporal confidence, validity-window confidence, aggregate confidence, and uncertain-range confidence. [2]

`src/core/confidence_scoring/confidence_calculator.py` creates confidence scores from SpaCy confidence, LLM responses, vector similarity, graph centrality, statistical tests, and domain-specific text/argument/sentiment/belief/community/psychological-state calculations. These calculators are heuristic source-to-confidence adapters, not validated dissertation measurements by themselves. [2]

# Integration Points

`src/core/tool_contract.py` makes `ToolResult.confidence` a required `ConfidenceScore` and uses low confidence for standardized error results. [7]

`src/core/tool_adapter.py` converts legacy tool outputs into `ToolResult`. If a legacy result has a confidence dict, it parses it as a `ConfidenceScore`; if it has a numeric confidence, it creates medium confidence with that value; otherwise it defaults to medium confidence. [8]

`src/core/tool_adapter_layer2.py` similarly calculates confidence from wrapped-tool results and defaults successful executions to medium confidence when no confidence is supplied. [9]

`src/tools/phase1/t68_pagerank_optimized.py` initializes a high-confidence base score and has a PageRank-specific confidence calculation using PageRank score, node degree, graph size, convergence confidence, and original entity confidence. [12]

`src/tools/phase2/t23c_ontology_aware_extractor_unified.py` initializes a high-confidence base score for ontology-aware extraction, with evidence weight justified by ontology, LLM reasoning, theory validation, semantic alignment, contextual analysis, and multi-modal evidence. [13]

# Proven Runtime Boundary

The recently proven `.txt`/`.pdf`/`.md`/`.docx` upload pipeline does not become an uncertainty proof merely because the confidence package exists. The current source slices reviewed here show `src/analytics/complete_pipeline.py` passing `source_refs` into graph building and querying; `src/analytics/graph_builder.py` writes `source_refs` to entity and edge records; and `src/analytics/graph_query_engine.py` accepts `source_refs` filters. [10][14][15]

That is valuable provenance work, but it is different from a current runtime proof of ADR-029, Comprehensive7, CERQual aggregation, or confidence propagation through every stage.

# Test Evidence Boundary

A targeted test search did not find dedicated current tests named for confidence or uncertainty under `tests/`. Source search found confidence-scoring imports/usages in current code and tools, but no focused test file was identified in this pass. This is a coverage caveat, not proof of absence across all possible historical archives. [16]

# Claim Boundary

Correct current-status claim:

> The current cleaned checkout contains an ADR-004-derived confidence model with CERQual, range, temporal, and combination extensions, plus adapters/tools that can carry or infer `ConfidenceScore` values. The checked current runtime evidence proves bounded document ingestion/graph/query paths and source-ref provenance, not a fully validated ADR-029 IC uncertainty framework.

Do not claim:

- current KGAS implements archived ADR-029/Comprehensive7 end to end;
- confidence defaults in adapters are validation evidence;
- existence of CERQual fields proves CERQual is applied throughout the upload runtime;
- source-ref propagation is the same thing as uncertainty propagation;
- fallback averaging/default confidence paths satisfy the fail-loud policy.

# Relationship To Wiki

- [ADR 029 Location Verification 2026 06 26](adr-029-location-verification-2026-06-26.md): separates recovered archived ADR-029 from current code status.
- [Uncertainty Framework Consolidation 2026 06 26](/wiki/concepts/uncertainty-framework-consolidation-2026-06-26.md): high-level uncertainty lineage map updated by this current-code boundary.
- [Current Status Verification Discipline](/wiki/concepts/current-status-verification-discipline.md): rule for separating architecture, evidence, code presence, and runtime proof.
- [Runtime Verification Isolation Boundary](/wiki/concepts/runtime-verification-isolation-boundary.md): relevant if future uncertainty runtime probes touch Neo4j.

# Citations

[1] `../src/core/confidence_score.py`  
[2] `../src/core/confidence_scoring/`  
[3] `../src/core/confidence_scoring/data_models.py`  
[4] `../src/core/confidence_scoring/cerqual_assessment.py`  
[5] `../src/core/confidence_scoring/combination_methods.py`  
[6] `../src/core/confidence_scoring/temporal_range_methods.py`  
[7] `../src/core/tool_contract.py`  
[8] `../src/core/tool_adapter.py`  
[9] `../src/core/tool_adapter_layer2.py`  
[10] `../src/analytics/complete_pipeline.py`  
[11] `../src/core/confidence_scoring/__init__.py`  
[12] `../src/tools/phase1/t68_pagerank_optimized.py`  
[13] `../src/tools/phase2/t23c_ontology_aware_extractor_unified.py`  
[14] `../src/analytics/graph_builder.py`  
[15] `../src/analytics/graph_query_engine.py`  
[16] `../tests/`
