---
type: Concept
title: Recovered UI Demo Surface
description: KGAS preserved UI material shows multiple demo and inspection surfaces, including static/mock UI, FastAPI backend UI, Streamlit dashboard, and recovered React components.
tags: [concept, ui, demo, dashboard, recovered]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/ui_components_recovered/documentation/README_KGAS_UI.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ui/UI_READY_TO_USE.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/evidence/current/Evidence_Phase_D4_Visualization_Dashboard.md
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/ui_components_recovered/react_components/AnalysisDashboard.jsx
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/ui/schema_analysis_ui.py
confidence: medium
---

# Summary

The recovered UI demo surface is not one product. It is a set of preserved interfaces for inspecting or demonstrating KGAS:

- static HTML UI with mock behavior
- FastAPI-backed research UI with upload/query/export endpoints
- Streamlit visualization dashboard
- React analysis dashboard component [1][2][3][4]

# Why It Matters

This material may be the closest preserved record of how KGAS was meant to feel to a researcher: upload documents, run a pipeline, inspect graph/analysis status, query results, and export reports.

It also connects the architecture to a human-facing thesis demo. Without this layer, the record can look like only backend architecture and evidence files.

The lit-review UI code adds a narrower research-review surface: a Streamlit app for selecting extracted theory schemas and text files, running schema-guided analysis, browsing saved JSON results, rendering charts/network views, and chatting over analysis outputs. [5]

The full archived UI corpus shows that `archive/ui/` and `archive/ui_archive_2025_08/` are near-duplicates: 95 files versus 91 files, with the four extra files in `archive/ui/archived_implementations/`. That makes the UI archive mostly one preserved surface with a duplicate snapshot, not two independent implementations. See [Digimon Lineage UI Recovered Components](/wiki/sources/digimon-lineage-ui-recovered-components.md).

# Caveat

The UI docs mix real backend integration claims with mock-mode fallback and archived status. Treat them as recoverable design/demo artifacts until current runtime verification is performed.

The full archive also preserves logs showing both successful local server/export traces and Neo4j connection/authentication failures. UI readiness claims should therefore be read as "a demo/UI surface existed and was locally tested," not as proof that all backend graph workflows were live.

The lit-review Streamlit app has its own recovery caveats: it uses hardcoded historical paths, direct OpenAI calls, and a launcher that expects a sibling `requirements_ui.txt`.

# Links

- [Digimon Lineage UI Recovered Components](/wiki/sources/digimon-lineage-ui-recovered-components.md)
- [Lit Review UI Code](/wiki/sources/lit-review-ui-code.md)
- [Analysis Expansion Architecture](/wiki/concepts/analysis-expansion-architecture.md)
- [Evidence Claim Discipline](/wiki/concepts/evidence-claim-discipline.md)
- [Academic Proof Of Concept Scope](/wiki/concepts/academic-proof-of-concept-scope.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/ui_components_recovered/documentation/README_KGAS_UI.md`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/ui/UI_READY_TO_USE.md`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/archive/evidence/current/Evidence_Phase_D4_Visualization_Dashboard.md`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/ui_components_recovered/react_components/AnalysisDashboard.jsx`  
[5] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/ui/schema_analysis_ui.py`
