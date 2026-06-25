---
type: Source
title: Current Code Verification 2026-06-25
description: Focused verification of current cleaned checkout claims around storage, tool contracts, MCP/API surfaces, UI, entry points, and tests.
tags: [source, current-code, verification, status, kgas]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../README.md
  - ../CLAUDE.md
  - ../src/core/qdrant_store.py
  - ../src/core/vector_store.py
  - ../src/core/tool_contract.py
  - ../src/api/cross_modal_api.py
  - ../src/mcp_server.py
  - ../Makefile
confidence: high
---

# Summary

This verification slice checks the current cleaned checkout, not the archived lineage bundle. It focuses on high-risk status claims already indexed in the wiki: storage, tool interface layers, API/MCP surfaces, UI, entry points, and test layout.

# Observed Current State

## Storage

Current README describes KGAS as an experimental GraphRAG system using Neo4j and lists Neo4j integration as working. [1]

Current `CLAUDE.md` says Neo4j + SQLite storage works in the vertical-slice context and lists Neo4j/SQLite configuration. [2]

Current code contains:

- `src/core/vector_store.py`: an abstract `VectorStore` interface. [4]
- `src/core/qdrant_store.py`: a simplified Qdrant compatibility/mock implementation and `QdrantVectorStore` alias. [3]
- `src/core/database_optimizer.py` includes a Neo4j vector-index command, found during verification grep.

Interpretation: the cleaned checkout is not a pure "Qdrant removed" state. It has a Neo4j/SQLite direction plus a Qdrant compatibility/mock artifact. This should be read alongside [Storage Architecture Evolution](/wiki/concepts/storage-architecture-evolution.md).

## Tool Contracts

`src/core/tool_contract.py` implements `ToolRequest`, `ToolResult`, `KGASTool`, `BaseKGASTool`, and `TheoryAwareKGASTool`, matching the Layer 2 contract direction in the ADRs. [5]

Search also found many `ToolRequest`/`ToolResult` usages across `src/tools` and `src/core`. That supports "contract surface exists," not "every tool is fully migrated."

## API And MCP

The current checkout has `src/api/cross_modal_api.py`, a FastAPI local cross-modal API, with explicit text saying it complements rather than replaces MCP. [6]

The current checkout has `src/mcp_server.py`. [7]

This supports the architectural existence of local API and MCP surfaces, but runtime functionality was not executed in this verification slice.

## UI And Entry Points

Current README says "Basic UI (Streamlit)" is a working feature. [1]

Current `CLAUDE.md` lists `main.py`, `streamlit_app.py`, `kgas_mcp_server.py`, and `kgas_simple_mcp_server.py` as entry points, and says `/src/ui` is part of the main system. [2]

Focused filesystem checks found:

- no root `main.py`
- no root `streamlit_app.py`
- no root `kgas_mcp_server.py`
- no root `kgas_simple_mcp_server.py`
- no `src/ui` directory
- `src/mcp_server.py` does exist

Interpretation: current UI/entry-point documentation is stale or refers to an earlier layout. Use [Recovered UI Demo Surface](/wiki/concepts/recovered-ui-demo-surface.md) for preserved UI lineage, but do not claim the cleaned checkout currently has `src/ui` without later verification.

## Tests

Focused filesystem check found no root `tests/` directory in the cleaned checkout, even though README and `CLAUDE.md` mention tests and integration-test evidence. [1][2]

Interpretation: test claims in current docs are historical or refer to archived/removed layout unless another test location is identified.

# Commands Run

Representative commands:

```bash
git status --short --branch
find . -maxdepth 3 -type f ...
find src -maxdepth 3 -type f ...
rg -n "Neo4j|SQLite|Postgres|Qdrant|VectorStore|KGASTool|ToolRequest|ToolResult|FastAPI|Streamlit|MCP|response_format|json_object|json_schema|src/ui|graphrag_ui|enhanced_dashboard" ...
ls -la main.py streamlit_app.py kgas_mcp_server.py kgas_simple_mcp_server.py
find tests -maxdepth 2 -type f
sha256sum README.md CLAUDE.md src/core/qdrant_store.py src/core/vector_store.py src/core/tool_contract.py src/api/cross_modal_api.py src/mcp_server.py Makefile
```

# Links

- [Current Status Verification Discipline](/wiki/concepts/current-status-verification-discipline.md)
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md)
- [Storage Architecture Evolution](/wiki/concepts/storage-architecture-evolution.md)
- [Layered Tool Interface Architecture](/wiki/concepts/layered-tool-interface-architecture.md)
- [Recovered UI Demo Surface](/wiki/concepts/recovered-ui-demo-surface.md)

# Citations

[1] `../README.md`  
[2] `../CLAUDE.md`  
[3] `../src/core/qdrant_store.py`  
[4] `../src/core/vector_store.py`  
[5] `../src/core/tool_contract.py`  
[6] `../src/api/cross_modal_api.py`  
[7] `../src/mcp_server.py`  
[8] `../Makefile`
