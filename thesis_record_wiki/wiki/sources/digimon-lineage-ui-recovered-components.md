---
type: Source
title: Digimon Lineage UI Recovered Components
description: Source summary for recovered and archived KGAS UI components, dashboard evidence, and UI guide material.
tags: [source, ui, dashboard, recovered, streamlit, react, fastapi]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/ui_components_recovered/documentation/README_KGAS_UI.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ui/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ui_archive_2025_08/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ui/README_KGAS_UI.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ui/UI_READY_TO_USE.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/evidence/current/Evidence_Phase_D4_Visualization_Dashboard.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/getting-started/ui-readme.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/ui_components_recovered/react_components/AnalysisDashboard.jsx
confidence: medium
---

# Summary

The large lineage bundle preserves several UI surfaces:

- `ui_components_recovered/` with recovered documentation and React components
- `archive/ui/` and `archive/ui_archive_2025_08/` with standalone/static/FastAPI UI material
- `src/ui/` and dashboard test/evidence references
- getting-started UI docs for an ontology generator Streamlit interface [1][2][3][4][5][6]

The recovered UI README and archived UI README have the same SHA-256 hash, so they are duplicate copies of the same backend-integration claim rather than independent evidence. [1][2]

# Full Archive UI Corpus

The `archive/ui/` subtree contains 95 files totaling 38,046,097 bytes, with aggregate hash `aggregate-sha256:1a43f78b882f4870307f2728215d896a31e95b2892fa3a217e0a920337e2082b`. The sibling `archive/ui_archive_2025_08/` subtree contains 91 files totaling 37,992,367 bytes, with aggregate hash `aggregate-sha256:a7bd22a90514ca2c153b0198d05e6d19614e34e6785cab22ebfd826c9062af1f`. [2][7]

The two trees are nearly duplicate. `ui_archive_2025_08/` has no files absent from `archive/ui/`; the only files unique to `archive/ui/` are four archived implementation scripts under `archived_implementations/`, with aggregate hash `aggregate-sha256:3685257065ff566d4f38048973e1f31207688846c8fc0cbefdc05327a57c4f1e`. [2][7]

The full `archive/ui/` corpus includes:

- nine top-level UI/readiness/automation markdown docs
- static HTML UIs: `functional_ui.html`, `functional_ui_backend.html`, `research_ui.html`, `simple_working_ui.html`, and tiny test pages
- FastAPI/server scripts: `kgas_web_server.py`, `real_kgas_server.py`, `simple_mock_server.py`, `quick_test_server.py`, and start scripts
- automation/testing scripts: `automated_ui_tester.py`, `continuous_ui_monitor.py`, Puppeteer test scripts, validation scripts, and debug scripts
- React/Vite app under `research-app/` with 32 files, including React Query service code, components, package files, and an MCP integration test
- four generated export files, three uploaded PDFs, and large preserved server/application logs [2]

# UI Modes

The archived/recovered UI docs describe three modes:

1. **FastAPI backend UI**: `kgas_web_server.py`, upload, analysis start/status, query, export, download endpoints. [1]
2. **Static/mock UI**: standalone HTML served locally, with tested tab navigation and simulated behavior. [3]
3. **Streamlit/dashboard UI**: ontology generator and enhanced dashboard materials. [4][5]

The React recovered component `AnalysisDashboard.jsx` shows a later dashboard direction with React Query, tool execution status, progress polling, performance charts, and API service calls. [6]

# Evidence

The Phase D.4 evidence file claims an interactive visualization dashboard with:

- enhanced dashboard framework
- interactive graph explorer
- batch processing monitor
- research analytics dashboard
- 22/22 tests passing
- Streamlit integration and graph/monitoring/analytics features [4]

The full archive UI corpus adds more direct local evidence. `automated_test_report.json` records 29/29 tests passed, one error-level issue for missing `vite`, and a recommendation to run `npm install` in `ui/research-app`. `UI_READY_TO_USE.md` claims 29 automated tests passed and 8/9 validation checks. `FINAL_VERIFICATION.md` says `functional_ui.html` was accessible at `localhost:8899` and that JavaScript functions for tabs, file upload, analysis, query, and export were present. [2]

The backend implementation docs claim a FastAPI server with upload, analysis, status, query, export, and download endpoints. They also explicitly define two operating modes: full KGAS mode when backend components are available, and mock mode when they are not. [2]

Preserved logs complicate the readiness claims. The UI archive includes `logs/super_digimon.log` with 146,800 lines, empty `super_digimon.rotating.log`, and server logs totaling tens of thousands of lines. Targeted inspection shows logs with "REAL KGAS imports successful" and generated export traces, but also Neo4j connection/authentication failures and PageRank/query components unable to connect to Neo4j. [2]

# Status Caveat

This slice should be read as UI lineage, not current UI status. Some docs say "fully functional" or "ready to use," while also describing mock mode and backend-availability detection. Current functionality would require running the preserved code against the preserved environment and dependencies.

The archive also preserves two duplicate uploaded Petty/Cacioppo PDFs and a tiny test PDF. These are raw historical inputs; the wiki indexes their presence but does not treat them as UI implementation evidence. [2]

# Links

- [Recovered UI Demo Surface](/wiki/concepts/recovered-ui-demo-surface.md)
- [Analysis Expansion Architecture](/wiki/concepts/analysis-expansion-architecture.md)
- [Layered Tool Interface Architecture](/wiki/concepts/layered-tool-interface-architecture.md)
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/ui_components_recovered/documentation/README_KGAS_UI.md`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ui/README_KGAS_UI.md`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ui/UI_READY_TO_USE.md`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/evidence/current/Evidence_Phase_D4_Visualization_Dashboard.md`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/docs/getting-started/ui-readme.md`  
[6] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/ui_components_recovered/react_components/AnalysisDashboard.jsx`  
[7] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ui_archive_2025_08/`
