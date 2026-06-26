# T27 Relationship Extraction Bottleneck Investigation

Date: 2026-06-25
Status: repaired for analysis-agent bridge; spaCy model dependency installed; follow-up required for broader caller audit

## Question

Trace the T27 relationship-extraction bottleneck across current code and archived evidence. Determine whether the historical "entities extracted, zero relationships" failure is still plausibly present, what code paths invoke T27, what input contract T27 expects, and what should be verified next.

## Atoms

| ID | Question | Dependencies | Status |
| --- | --- | --- | --- |
| A1 | What does current T27 expect as input and emit as output? | none | answered |
| A2 | Which current code paths invoke T27, and do they pass the expected input shape? | A1 | answered; analysis-agent bridge repaired |
| A3 | What did archived bottleneck reports say failed: non-invocation, silent failure, format mismatch, or pattern coverage? | none | answered |
| A4 | What archived repair/test evidence exists for T23A -> T27 format conversion or relationship-pattern fixes? | A1, A3 | answered |
| A5 | Can we run a minimal current T27 test in the isolated `.venv` without external services? | A1 | answered |
| A6 | What is the current root cause hypothesis and the next verification step? | A1-A5 | answered |

## Assumptions Register

| # | Assumption | Confidence | How to verify | Round | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | The historical zero-relationship failure may be a T23A/T27 input-shape mismatch rather than only weak extraction patterns. | high | Compare T27 expected entity keys with T23A output and archived `test_pipeline_fix.py`. | 1 | Supported: current analysis-agent path forwarded raw T23A entities to T27 before the repair. |
| 2 | Current T27 can be exercised without Neo4j or live LLM services. | high | Import and instantiate T27 in `.venv`; run on fixture text/entities. | 1 | Supported. Pattern extraction produced relationships, and `en_core_web_sm` now loads for dependency parsing. |
| 3 | MCP exposure of `extract_relationships` is not the same as main pipeline invocation. | high | Trace both MCP `pipeline_tools.py` and core pipeline/orchestrator callers. | 1 | Supported: MCP function docs require pre-normalized T27 entities; analysis-agent was a separate caller path. |

## Evidence Log

### A1 - Current T27 Contract

Answer: current `T27RelationshipExtractorUnified` requires input data with `text`, `entities`, and at least two entity records. Each entity must include `text`, `entity_type`, `start`, and `end`; `confidence` is optional and defaults internally to `0.8` in confidence calculations. It emits `relationships`, `relationship_count`, `confidence`, `processing_method`, and `extraction_stats`. Evidence:

- `src/tools/phase1/t27_relationship_extractor_unified.py:319-350` validates `text`, `entities`, at least two entities, and the entity fields `text/entity_type/start/end`.
- `src/tools/phase1/t27_relationship_extractor_unified.py:282-304` returns `relationships`, `relationship_count`, confidence, method, stats, and metadata.
- `src/tools/phase1/t27_relationship_extractor_unified.py:447-468` constructs pattern-based relationships using the T27 entity shape.
- `src/tools/phase1/t27_relationship_extractor_unified.py:754-779` repeats the public contract with required `text/entity_type/start/end`.

Status: answered.

### A2 - Current Invocation Paths

Answer: the analysis-agent path was contract-incompatible before this investigation's repair. It extracted T23A entities, filtered by `chunk_ref`, and passed them directly into the MCP `extract_relationships` call. T23A emits `surface_form/entity_type/start_pos/end_pos/chunk_ref`, not T27's `text/entity_type/start/end`. Evidence:

- `src/tools/phase1/t23a_spacy_ner_unified.py:354-365` emits T23A entity records with `surface_form`, `entity_type`, `start_pos`, `end_pos`, and `chunk_ref`.
- `src/orchestration/agents/analysis_agent.py:477-493` gathers chunk entities after extraction and invokes relationship extraction.
- Before repair, `src/orchestration/agents/analysis_agent.py:563-571` forwarded `entities` directly to `self.mcp.call_tool("extract_relationships", ...)`.
- `src/tools/phase1/phase1_mcp_tools.py:191-224` exposes MCP `extract_relationships` and documents that entities should already be in T27 format.
- `src/mcp_tools/pipeline_tools.py:97-116` exposes another wrapper that forwards whatever entity list the caller provides.

Repair applied in this round:

- `src/orchestration/agents/analysis_agent.py` now normalizes entities via `_normalize_entities_for_t27(...)` before calling MCP relationship extraction.
- `tests/current_runtime/test_analysis_agent_t27_contract.py` covers T23A conversion, already-normalized T27 pass-through, and invalid-shape fail-loud behavior.

Status: answered and repaired for the analysis-agent bridge.

### A3 - Archived Bottleneck Reports

Answer: the archived bottleneck reports framed the zero-relationship symptom primarily as non-invocation or silent failure. They also contained a concrete code sketch that would still pass raw `chunk_entities` into T27, which is contract-incompatible with current T27. Evidence:

- `archive_full_record/lineage_variants/digimon_core_sparse/ARCHITECTURAL_BOTTLENECKS_ANALYSIS.md:8-18` says 25 documents produced 398 entities and zero relationships, with only chunking and NER tools executed.
- `archive_full_record/lineage_variants/digimon_core_sparse/ARCHITECTURAL_BOTTLENECKS_ANALYSIS.md:132-176` recommends adding relationship extraction after entity extraction but passes `chunk_entities` directly in the example.
- `archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/evidence_reports_2025_08/ARCHITECTURAL_BOTTLENECKS_ANALYSIS.md:8-18` repeats the same zero-relationship finding.
- `archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/evidence_reports_2025_08/ARCHITECTURAL_BOTTLENECKS_ANALYSIS.md:200-205` recommends auditing T27 invocation and validating a simple case.

Status: answered.

### A4 - Archived Repair/Test Evidence

Answer: archived debug files explicitly identified T23A/T27 entity-format conversion as the proposed fix, but one preserved script used `label` instead of the current T27-required `entity_type`. Later archived workflow code also converted `surface_form` to `text` and positions to `start/end`, but similarly used `label`; this is useful historical evidence, not a current-valid patch. Evidence:

- `archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/temp_debug_files/test_pipeline_fix.py:54-76` defines a converter from T23A `surface_form/entity_type/start_pos/end_pos` to T27-shaped `text/label/start/end`.
- `archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/temp_debug_files/test_pipeline_fix.py:119-122` declares the problem as incompatible entity formats and the solution as a conversion layer.
- `archive_full_record/lineage_variants/digimon_lineage_Digimons/tests/archived_root_tests/test_final_kgas_workflow.py:161-176` formats chunk entities for T27 using `text`, `label`, `start`, and `end`.
- Current T27 requires `entity_type`, not `label`, per `src/tools/phase1/t27_relationship_extractor_unified.py:345-348`.

Status: answered.

### A5 - Minimal Runtime Check

Answer: yes, a minimal `.venv` T27 check can run without live Neo4j or live LLM services for pattern/proximity extraction. The check reproduced the failure and verified the repair:

- Valid T27-shaped fixture entities over `Alice works at Acme Corp.` returned `success` and `relationship_count=2`.
- Raw T23A-shaped fixture entities returned `error` with `Entity 0 missing required field: text`.
- After applying `_normalize_entities_for_t27(...)`, converted T23A-shaped fixture entities returned `success` and `relationship_count=2`.

Runtime follow-up: `en_core_web_sm==3.8.0` was installed into the project `.venv` and added to `requirements.txt` as a direct spaCy model wheel dependency. The resource manager now loads the model successfully. The minimal fixture still produced pattern-based relationships; dependency-parser-specific relationship output should be tested with a richer syntax fixture if that method becomes a target criterion.

Verification commands:

```bash
.venv/bin/python -m pytest tests/current_runtime/test_analysis_agent_t27_contract.py tests/current_runtime/test_cross_modal_api_contract.py
.venv/bin/python -m pytest tests/current_runtime/test_spacy_model_dependency.py
.venv/bin/python - <<'PY'
# direct T27 fixture probe run during investigation
PY
python /home/brian/projects/.agents/skills/karpathy-wiki/scripts/lint.py /home/brian/projects/phd_thesis_work/thesis_record_wiki
```

Results:

```text
13 passed, 2 warnings
tests/current_runtime/test_spacy_model_dependency.py . [100%]
valid success None 2
converted_t23a success None 2
Wiki health: 100/100
```

Status: answered.

## Contraction After Round 1

The problem contracts from "relationship extraction might be broadly broken" to a specific current bridge bug plus an environment dependency gap. T27 itself can extract relationships from valid inputs. The current analysis-agent bridge needed to convert T23A output into T27 input before calling MCP. That bridge is now repaired and covered by focused tests. The remaining unresolved runtime question is not the T23A/T27 contract but whether the full extraction stack should install/manage `en_core_web_sm` so dependency parsing is available, and whether higher-level pipelines beyond `AnalysisAgent` need the same normalization boundary.

## Synthesis

Root cause found: current T27 expects `text/entity_type/start/end`, while T23A emits `surface_form/entity_type/start_pos/end_pos`. The analysis-agent path forwarded T23A entities directly into relationship extraction, reproducing the historical zero-relationship/silent-failure risk as a concrete contract mismatch.

Impact: this could cause analysis tasks to report successful entity extraction but no relationships, with relationship errors captured only as warnings rather than graph-rich output. That matches the archived concern that KGAS can become entity-only despite GraphRAG ambitions.

Repair completed: added `_normalize_entities_for_t27(...)` to `src/orchestration/agents/analysis_agent.py`, applied it before the MCP `extract_relationships` call, and added focused current-runtime tests in `tests/current_runtime/test_analysis_agent_t27_contract.py`.

Confidence: high for the analysis-agent contract mismatch and repair; medium for broader pipeline coverage because other direct callers may still need normalization audits.

Recommended next step: audit direct T27 callers outside `AnalysisAgent` for the same T23A/T27 entity-contract boundary, then add a richer dependency-parser fixture if parser-derived relationships are a target runtime capability.
