---
type: Source
title: Current Runtime Import Check 2026-06-25
description: Import-only runtime check for current KGAS contract, cross-modal API, and MCP modules.
tags: [source, runtime, import-check, verification, kgas]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../src/core/tool_contract.py
  - ../src/api/cross_modal_api.py
  - ../src/mcp_server.py
  - ../src/analytics/__init__.py
  - ../src/analytics/cross_modal_orchestrator.py
  - ../requirements.txt
confidence: high
---

# Summary

This runtime slice imports three current modules without changing source code. It upgrades the previous file-presence check into a narrow execution check.

# Command

```bash
python - <<'PY'
import importlib
modules = [
    "src.core.tool_contract",
    "src.api.cross_modal_api",
    "src.mcp_server",
]
for module in modules:
    try:
        importlib.import_module(module)
        print(f"OK {module}")
    except Exception as exc:
        print(f"FAIL {module}: {type(exc).__name__}: {exc}")
PY
```

# Results

```text
Failed to load configuration: [Errno 2] No such file or directory: '/home/brian/projects/phd_thesis_work/config/default.yaml'
OK src.core.tool_contract
2026-06-25 10:15:35 [INFO] super_digimon.core.logging: Logging system initialized - Level: INFO, Console: True, File: logs/super_digimon.log
... RequestsDependencyWarning ...
FAIL src.api.cross_modal_api: ImportError: cannot import name 'AnalysisRequest' from 'src.analytics' (/home/brian/projects/phd_thesis_work/src/analytics/__init__.py)
FAIL src.mcp_server: ModuleNotFoundError: No module named 'neo4j'
```

# Interpretation

`src.core.tool_contract` imports successfully. That is stronger evidence than file presence for the current KGASTool contract surface. [1]

`src.api.cross_modal_api` does not currently import in this environment. Its import expects `AnalysisRequest` from `src.analytics`, but `src/analytics/__init__.py` does not export that symbol. A symbol named `AnalysisRequest` exists in `src/analytics/cross_modal_orchestrator.py`, so this looks like an export/import wiring issue rather than absence of the class in the repository. [2][4][5]

`src.mcp_server` does not currently import in this environment because `neo4j` is missing. `requirements.txt` lists `neo4j>=5.0.0`, so the failure is an environment/dependency-readiness fact for the active virtual environment, not proof that MCP source code is absent. [3][6]

The import attempt also emitted a missing `config/default.yaml` warning and initialized logging to `logs/super_digimon.log`. Those side effects matter for future runtime verification: even import-only checks are not completely side-effect free.

# Follow-Up Environment Inspection

A follow-up inspection used the active interpreter:

```text
/home/brian/projects/.venv/bin/python
Python 3.12.3
```

`python -m pip show neo4j fastapi pydantic requests urllib3 chardet charset-normalizer` confirmed:

- `neo4j` is not installed in the active shared virtual environment
- `fastapi`, `pydantic`, and `requests` are installed
- the earlier `RequestsDependencyWarning` is reproducible from the active dependency set

`python -m pip check` reported:

```text
graspologic 3.4.4 has requirement beartype<0.19.0,>=0.18.5, but you have beartype 0.22.9.
```

A direct symbol check succeeded:

```bash
python - <<'PY'
from src.analytics.cross_modal_orchestrator import AnalysisRequest
print("OK direct AnalysisRequest import")
print(AnalysisRequest)
PY
```

Result:

```text
OK direct AnalysisRequest import
<class 'src.analytics.cross_modal_orchestrator.AnalysisRequest'>
```

Interpretation: the cross-modal API import failure is specifically caused by the `src.analytics` package export path, not by absence of `AnalysisRequest` from the repository. The MCP import failure remains a dependency-installation gap in the active environment.

# Repair Follow-Up

After the repair pass, `src.api.cross_modal_api` imports successfully in the active environment. The patch changed the API boundary to import concrete cross-modal modules directly where they are needed, lazy-load the service registry so missing registry runtime dependencies do not prevent module import, normalize API enum strings against lowercase enum values, call the current `orchestrate_analysis(...)` signature, and map the current `AnalysisResult` fields into `AnalysisResponse`. [2][5]

The focused current-runtime test file `tests/current_runtime/test_cross_modal_api_contract.py` covers module import, enum parsing, document-placeholder graph construction, preferred-mode mapping, selected-mode extraction, and bad-value HTTP 400 behavior. The targeted test run passed:

```text
tests/current_runtime/test_cross_modal_api_contract.py ........ [100%]
8 passed, 2 warnings
```

The same import check now reports:

```text
OK src.core.tool_contract
OK src.api.cross_modal_api
FAIL src.mcp_server: ModuleNotFoundError: No module named 'neo4j'
```

An attempted eager import of `src.analytics.cross_modal_service_registry` exposed a separate dependency gap: `ModuleNotFoundError: No module named 'torchvision'`. The lazy registry import keeps the API module importable, but endpoints that initialize or use registry-backed services still require dependency/environment work before they can be called end to end. [2]

The mode-recommendation endpoint now returns an explicit 501 because its old `DataContext` construction no longer matches the current `DataContext` dataclass. That is a truthful status boundary rather than a silent fallback.

# Links

- [Current Code Verification 2026-06-25](/wiki/sources/current-code-verification-2026-06-25.md)
- [Current Status Verification Discipline](/wiki/concepts/current-status-verification-discipline.md)
- [Layered Tool Interface Architecture](/wiki/concepts/layered-tool-interface-architecture.md)
- [Analysis Expansion Architecture](/wiki/concepts/analysis-expansion-architecture.md)

# Citations

[1] `../src/core/tool_contract.py`  
[2] `../src/api/cross_modal_api.py`  
[3] `../src/mcp_server.py`  
[4] `../src/analytics/__init__.py`  
[5] `../src/analytics/cross_modal_orchestrator.py`  
[6] `../requirements.txt`
[7] `../tests/current_runtime/test_cross_modal_api_contract.py`
