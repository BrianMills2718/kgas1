---
type: Source
title: Digimon Lineage UI Recovered Components
description: Source summary for recovered and archived KGAS UI components, dashboard evidence, and UI guide material.
tags: [source, ui, dashboard, recovered, streamlit, react, fastapi]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/ui_components_recovered/documentation/README_KGAS_UI.md
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

# Status Caveat

This slice should be read as UI lineage, not current UI status. Some docs say "fully functional" or "ready to use," while also describing mock mode and backend-availability detection. Current functionality would require running the preserved code against the preserved environment and dependencies.

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
