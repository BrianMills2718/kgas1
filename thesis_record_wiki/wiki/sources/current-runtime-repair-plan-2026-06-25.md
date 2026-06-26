---
type: Source
title: Current Runtime Repair Plan 2026-06-25
description: Non-invasive source-level plan for repairing the first current KGAS import/runtime blockers.
tags: [source, repair-plan, runtime, imports, kgas]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../src/api/cross_modal_api.py
  - ../src/analytics/__init__.py
  - ../src/analytics/cross_modal_orchestrator.py
  - ../src/mcp_server.py
  - ../src/mcp_tools/__init__.py
  - ../src/core/service_manager.py
  - ../requirements.txt
  - ../src/orchestration/agents/analysis_agent.py
  - ../tests/current_runtime/test_analysis_agent_t27_contract.py
  - ../tests/current_runtime/test_spacy_model_dependency.py
confidence: high
---

# Summary

This plan identifies the smallest responsible repair path for the current import failures found in [Current Runtime Import Check 2026-06-25](/wiki/sources/current-runtime-import-check-2026-06-25.md). It is intentionally non-invasive: no source code or shared virtual environment dependencies were changed in this slice.

# Findings

## Cross-Modal API

Observed failure:

```text
ImportError: cannot import name 'AnalysisRequest' from 'src.analytics'
```

Source-level cause:

- `src/api/cross_modal_api.py` imports `AnalysisRequest` from `src.analytics`. [1]
- `src/analytics/__init__.py` exports `CrossModalOrchestrator` and `WorkflowOptimizationLevel`, but not `AnalysisRequest`. [2]
- `AnalysisRequest` exists in `src/analytics/cross_modal_orchestrator.py`. [3]
- Direct import from `src.analytics.cross_modal_orchestrator` succeeds in the active environment. [3]

Important second-order issue:

`src/api/cross_modal_api.py` constructs `AnalysisRequest` with `data=`, `source_format=`, `target_formats=`, and `task=`, then calls `orchestrator.orchestrate_analysis(request)`. [1]

The current orchestrator signature is:

```python
async def orchestrate_analysis(
    self,
    research_question: str,
    data: Any,
    source_format: DataFormat,
    preferred_modes: Optional[List[AnalysisMode]] = None,
    validation_level: Optional[ValidationLevel] = None,
    optimization_level: Optional[WorkflowOptimizationLevel] = None,
    constraints: Optional[Dict[str, Any]] = None,
    preferences: Optional[Dict[str, Any]] = None
) -> AnalysisResult:
```

The current `AnalysisRequest` dataclass expects `request_id`, `research_question`, `data`, and `source_format`, not `target_formats` or `task`. [3]

Interpretation: adding `AnalysisRequest` to `src.analytics.__all__` may fix the first import blocker but is unlikely to make the endpoint run correctly. The API appears to preserve an older request-object calling convention while the orchestrator now exposes argument-based orchestration.

## MCP Server

Observed failure:

```text
ModuleNotFoundError: No module named 'neo4j'
```

Traceback path:

```text
src.mcp_server
  -> src.mcp_tools
  -> src.mcp_tools.identity_tools
  -> src.mcp_tools.server_config
  -> src.core.service_manager
  -> from neo4j import GraphDatabase
```

Source-level cause:

- `src/core/service_manager.py` imports `GraphDatabase` at module load. [6]
- `requirements.txt` declares `neo4j>=5.0.0`, but the active shared virtual environment does not have `neo4j` installed. [7]

Interpretation: the first MCP repair is environment dependency installation or per-project environment reconstruction, not source wiring. A later hardening pass could defer the `neo4j` import until `get_neo4j_driver()` if MCP importability without Neo4j is desired.

# Recommended Repair Order

1. **Do not install into the shared `/home/brian/projects/.venv` as the first move.** Create or activate a project-local `.venv` if the goal is to test KGAS runtime without perturbing other projects.
2. **For cross-modal API, update the API to the current orchestrator contract instead of only exporting `AnalysisRequest`.** The endpoint should call `orchestrator.orchestrate_analysis(research_question=task, data=mock_graph_data, source_format=DataFormat.GRAPH, validation_level=..., optimization_level=...)`.
3. **Add or update a minimal import/runtime test** that imports `src.api.cross_modal_api` and constructs the relevant endpoint path without needing a live Neo4j instance.
4. **For MCP, install declared dependencies in an isolated environment first.** Only after that, rerun `import src.mcp_server` to distinguish missing package from live Neo4j configuration failure.
5. **If MCP must import without Neo4j installed, plan a separate source hardening patch** that moves `from neo4j import GraphDatabase` inside `ServiceManager.get_neo4j_driver()` and fails loudly when Neo4j-backed services are invoked.

# Acceptance Criteria For A Future Code Fix

- `python -c "import src.core.tool_contract"` passes. Completed in the follow-up repair pass.
- `python -c "import src.api.cross_modal_api"` passes. Completed in the follow-up repair pass.
- A focused API test proves the cross-modal endpoint calls the current `orchestrate_analysis(...)` signature.
- In an isolated KGAS environment with requirements installed, `python -c "import src.mcp_server"` reaches the next real configuration or service-readiness state.
- Any remaining failures are recorded with full traceback and not collapsed into a generic "broken" label.

# Follow-Up Status

The cross-modal API import blocker has been repaired and covered by `tests/current_runtime/test_cross_modal_api_contract.py`. The isolated environment pass added missing declared-runtime dependencies (`python-multipart`, `fastmcp`, `psutil`, and `sympy`) to `requirements.txt`; after that, `src.core.tool_contract`, `src.api.cross_modal_api`, and `src.mcp_server` all import in the project-local `.venv`, and `src.mcp_server.mcp` is created. [1][4][6]

The first relationship-extraction follow-up is also complete for the analysis-agent bridge. Current T27 requires `text/entity_type/start/end` entity records, while T23A returns `surface_form/entity_type/start_pos/end_pos`; `src/orchestration/agents/analysis_agent.py` now normalizes entities before calling MCP relationship extraction. The focused contract tests pass, and a direct T27 fixture probe returns two relationships for both native T27 entities and converted T23A entities. [8][9]

The spaCy model follow-up is complete at the dependency layer: `en-core-web-sm==3.8.0` is installed in the isolated environment, declared in `requirements.txt`, and covered by a runtime test that verifies the parser component is available. Remaining follow-up: audit direct T27 callers outside `AnalysisAgent`; if dependency-parser-derived relationships matter, add a richer fixture that requires the parser path instead of pattern extraction. [7][10]

# Links

- [Current Runtime Import Check 2026-06-25](/wiki/sources/current-runtime-import-check-2026-06-25.md)
- [Current Status Verification Discipline](/wiki/concepts/current-status-verification-discipline.md)
- [Layered Tool Interface Architecture](/wiki/concepts/layered-tool-interface-architecture.md)

# Citations

[1] `../src/api/cross_modal_api.py`  
[2] `../src/analytics/__init__.py`  
[3] `../src/analytics/cross_modal_orchestrator.py`  
[4] `../src/mcp_server.py`  
[5] `../src/mcp_tools/__init__.py`  
[6] `../src/core/service_manager.py`  
[7] `../requirements.txt`
[8] `../src/orchestration/agents/analysis_agent.py`
[9] `../tests/current_runtime/test_analysis_agent_t27_contract.py`
[10] `../tests/current_runtime/test_spacy_model_dependency.py`
