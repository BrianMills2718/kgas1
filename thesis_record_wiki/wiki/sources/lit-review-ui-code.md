---
type: Source
title: Lit Review UI Code
description: Inventory and interpretation of the preserved Streamlit UI and YAML analysis scripts for schema analysis.
tags: [source, lit-review, code, ui, streamlit, schema-analysis, openai]
created: 2026-06-25
updated: 2026-06-25
sources:
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/ui/
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/ui/schema_analysis_ui.py
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/ui/yaml_analysis_script.py
  - ../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/ui/run_ui.sh
confidence: high
---

# Summary

The preserved `src/ui/` folder contains 3 files with aggregate hash `d0f916695a8e497378e0e771992ab71cb4d76b86c866004e2e2d3d613fb6794a`. [1]

This is a Streamlit-era review surface for applying extracted theory schemas to texts, browsing saved analysis results, visualizing structured outputs, and chatting over prior analysis JSON.

# File Inventory

| File | Hash | Role |
|---|---|---|
| `schema_analysis_ui.py` | `sha256:ae531595903b1b331ab560171b30a517f834de04d09edb80dc2ca22c03eda70a` | Streamlit app for selecting schemas/texts, running OpenAI analysis, browsing results, charting structured output, network visualization, and chat over results. |
| `yaml_analysis_script.py` | `sha256:c7d24768feb250f8d11a6a0658574b1a10abfb3f327fdc5e6d502432637d8462` | CLI-style statistics script over literature YAML files. |
| `run_ui.sh` | `sha256:c182f85fd5dd75188ac95e25c35fdb64bc94a927af05115462ce7bbd507ab5fb` | Shell launcher that creates `venv_ui`, installs `requirements_ui.txt`, and starts Streamlit on localhost port 8501. |

# Streamlit App Shape

`schema_analysis_ui.py` defines `SchemaAnalysisUI`, then exposes three modes:

- New Analysis
- Browse Results
- Chat with Results

The app scans a historical literature directory for schema files ending in `_raw.yml`, scans a text directory for `.txt` files, lets the user select a theoretical framework and text, and runs an OpenAI chat completion with `response_format={"type": "json_object"}`. [2]

Saved results include metadata such as schema name, citation, timestamp, text length, and model. The UI then renders:

- framework-specific Plotly charts
- structured data tables
- a NetworkX/Plotly relationship graph
- raw JSON search and download
- a chat interface over a selected result [2]

# YAML Statistics Script

`yaml_analysis_script.py` walks a historical literature directory, loads YAML files, and prints aggregate statistics:

- model-type distribution
- JSON-schema presence
- total and average definition counts
- most common vocabulary categories
- file-size statistics
- per-file detail sorted by definition count [3]

This script is useful provenance because it shows schema-corpus inspection before the later wiki inventory pages.

# Caveats

- The app has hardcoded `/home/brian/lit_review/...` paths.
- It initializes a direct OpenAI client and uses `json_object`, not the later shared `llm_client` and JSON Schema contract discipline.
- The launcher references `requirements_ui.txt`, which is not in `src/ui/`; runtime recovery would need dependency/context verification.
- The app catches several errors and reports them through the UI, so this should be treated as historical review code rather than a fail-loud current tool.

# Interpretation

This UI slice strengthens the evidence that the lit-review work had a human review loop. The pipeline was not only schema generation and batch application; it also included interactive selection, visual inspection, result browsing, and result interrogation.

It connects the thesis record to [Recovered UI Demo Surface](/wiki/concepts/recovered-ui-demo-surface.md), but with a narrower scope: this is specifically a lit-review/schema-analysis review surface rather than the broader KGAS project UI.

# Links

- [Lit Review Src Code Inventory](/wiki/sources/lit-review-src-code-inventory.md)
- [Lit Review Visualization Code](/wiki/sources/lit-review-visualization-code.md)
- [Lit Review Schema Application Code](/wiki/sources/lit-review-schema-application-code.md)
- [Recovered UI Demo Surface](/wiki/concepts/recovered-ui-demo-surface.md)
- [Analyst Assembler Pattern](/wiki/concepts/analyst-assembler-pattern.md)

# Citations

[1] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/ui/`  
[2] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/ui/schema_analysis_ui.py`  
[3] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/ui/yaml_analysis_script.py`  
[4] `../archive_full_record/lineage_variants/digimon_lineage_Digimons/experiments/lit_review/src/ui/run_ui.sh`
