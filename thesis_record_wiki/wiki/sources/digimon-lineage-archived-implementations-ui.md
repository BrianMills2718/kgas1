---
type: SourceSummary
title: Digimon Lineage Archived Implementations UI
description: Small archived implementation-remnants bundle containing Streamlit GraphRAG UI variants, a simple PDF upload/debug UI, and a minimal upload smoke test.
tags: [source, digimon-lineage, archive, ui, streamlit, graphrag, pdf-upload, vertical-slice]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_implementations/
confidence: high
---

# Summary

`archive/archived_implementations/` is a four-file, 53,730-byte archive of KGAS UI implementation remnants. Its aggregate content-manifest hash is `280012f729474b780e992c63cabedef32a7aff3a319e34f67efdde46ca0dab09`. [1]

Despite the directory name, the preserved files are not general backend implementations. They are Streamlit UI variants and upload/debug surfaces for GraphRAG PDF workflows. [1]

# Inventory

| File | Bytes | Role |
| --- | ---: | --- |
| `web_ui.py` | 33,545 | Streamlit web UI for the Super-Digimon GraphRAG vertical-slice workflow. [2] |
| `graphrag_ui_v2.py` | 12,839 | Streamlit GraphRAG UI v2 using a standardized phase interface / UI phase adapter. [3] |
| `simple_ui.py` | 6,791 | Simple Streamlit PDF upload and workflow debug UI. [4] |
| `streamlit_test.py` | 555 | Minimal Streamlit upload smoke test. [5] |

# UI Evolution

`web_ui.py` is the broadest legacy UI. It uploads PDFs, validates file size/header, initializes a `VerticalSliceWorkflow`, runs a query, stores query history, and renders workflow metrics such as chunks, entities, relationships, graph entities, query answers, and graph visualization. [2]

`graphrag_ui_v2.py` is a later variant that routes through a UI phase adapter. It selects available phases, reads phase requirements, supports domain descriptions for later phases, validates queries, and presents a standardized phase-processing interface. [3]

`simple_ui.py` is a focused debug UI for testing PDF upload and basic workflow execution. It explicitly checks local Neo4j connectivity, workflow import availability, PDF validity, temporary-file saving, and result display. [4]

`streamlit_test.py` is a minimal file-upload component smoke test. [5]

# Caveats

These UIs depend on historical imports, local Neo4j assumptions, Streamlit runtime state, and vertical-slice workflow APIs. They are useful for understanding UI intent and debugging history, not as current runnable product surfaces without verification.

The targeted credential scan found no literal OpenAI or Google API keys in this archive. [1]

# Relationship To Wiki

- [Digimon Lineage UI Recovered Components](digimon-lineage-ui-recovered-components.md): broader preserved UI/demo/dashboard corpus.
- [Recovered UI Demo Surface](../concepts/recovered-ui-demo-surface.md): synthesis concept for static, Streamlit, FastAPI, and React UI/demo surfaces.
- [Digimon Lineage Demos Examples 2025 08](digimon-lineage-demos-examples-2025-08.md): neighboring small demo/example archive.
- [Digimon Lineage Scripts Archive 2025 08](digimon-lineage-scripts-archive-2025-08.md): overlapping scripts/demo archive.

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_implementations/`

[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_implementations/web_ui.py`

[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_implementations/graphrag_ui_v2.py`

[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_implementations/simple_ui.py`

[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/archived_implementations/streamlit_test.py`
